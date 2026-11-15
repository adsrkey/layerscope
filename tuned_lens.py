#!/usr/bin/env python3
"""
tuned_lens.py — обучаемая линза (tuned lens, Belrose et al., 2023) для logit lens.

Обычный logit lens подаёт вектор промежуточного слоя прямо в финальную норму и lm_head.
Но ранние слои «говорят» в другом базисе, поэтому их промежуточные ответы систематически
искажены. Tuned lens для каждого слоя учит небольшое аффинное преобразование, которое
переводит вектор слоя в базис последнего слоя. Сама модель не меняется.

Обучение (один раз на модель, минуты):
    python tuned_lens.py /путь/к/модели --texts corpus.txt --out lens.safetensors
    python tuned_lens.py /путь/к/модели                  # без корпуса: на встроенных промптах

Использование:
    python trace_qwen.py /путь/к/модели "промпт" --tuned-lens lens.safetensors

Чем больше текста, тем лучше: встроенных промптов хватает только на грубую линзу,
для аккуратной нужен файл хотя бы на несколько сотен тысяч токенов.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

from qcore import (eos_ids, format_prompt, kl_div, lens_logits, load_model, log_softmax,
                   model_fingerprint, pick_token)

F32 = mx.float32


def translate(params: dict, layer: int, h: mx.array) -> mx.array:
    """h -> h + A·h + b для слоя layer; A полная («a») или малоранговая («u»·«v»)."""
    h = h.astype(F32)
    if "a" in params:
        return h + h @ params["a"][layer] + params["b"][layer]
    return h + (h @ params["u"][layer]) @ params["v"][layer] + params["b"][layer]


class TunedLens:
    def __init__(self, params: dict, meta: dict = None):
        self.params, self.meta = params, meta or {}

    @classmethod
    def load(cls, path):
        params, meta = mx.load(str(path), return_metadata=True)
        return cls(params, meta)

    def save(self, path):
        mx.save_safetensors(str(path), self.params, {k: str(v) for k, v in self.meta.items()})

    @property
    def n_layers(self) -> int:
        return self.params["b"].shape[0]

    def __call__(self, layer: int, h: mx.array) -> mx.array:
        return translate(self.params, layer, h)


def init_params(n_layers: int, hidden: int, rank: int) -> dict:
    """Начальное состояние — тождественное преобразование (линза совпадает с logit lens)."""
    b = mx.zeros((n_layers, hidden))
    if rank <= 0:
        return {"a": mx.zeros((n_layers, hidden, hidden)), "b": b}
    return {"u": mx.random.normal((n_layers, hidden, rank)) * hidden ** -0.5,
            "v": mx.zeros((n_layers, rank, hidden)), "b": b}


def chunks_from_text(tok, text: str, seq_len: int) -> list:
    ids = tok.encode(text)
    return [ids[i:i + seq_len] for i in range(0, len(ids) - 1, seq_len) if len(ids[i:i + seq_len]) > 4]


def chunks_from_prompts(model, tok, prompts, n_tokens: int, log=print) -> list:
    """Запасной корпус: промпты в шаблоне чата плюс собственные ответы модели."""
    lm, eos, out = model.language_model, eos_ids(model), []
    for p in prompts:
        ids = tok.encode(format_prompt(tok, p))
        cache, inputs, gen = lm.make_cache(), mx.array([ids]), []
        for _ in range(n_tokens):
            nxt = pick_token(lm(inputs, cache=cache).logits[0, -1])
            if nxt in eos:
                break
            gen.append(nxt)
            inputs = mx.array([[nxt]])
        out.append(ids + gen)
    log(f"корпус из {len(out)} встроенных промптов с ответами модели")
    return out


def layer_states(lm, ids: list, pos: list):
    """Скрытые состояния всех слоёв на позициях pos и целевые log p последнего слоя."""
    n_layers = len(lm.model.layers)
    out = lm(mx.array([ids]), capture_layer_ids=list(range(n_layers)))
    idx = mx.array(pos)
    H = mx.stack([h[0][idx].astype(F32) for h in out.hidden_states[:n_layers]])   # (L, N, D)
    target = log_softmax(out.logits[0][idx])                                      # (N, V)
    H, target = mx.stop_gradient(H), mx.stop_gradient(target)
    mx.eval(H, target)
    return H, target


def layer_kls(lm, params, H, target) -> mx.array:
    """KL(последний слой || линза слоя) для каждого слоя: (L,)."""
    return mx.stack([kl_div(target, log_softmax(lens_logits(lm, translate(params, l, H[l])))).mean()
                     for l in range(H.shape[0])])


def train(model, chunks: list, rank: int = 64, steps: int = 300, lr: float = 1e-3,
          positions: int = 16, seed: int = 0, log=print) -> TunedLens:
    import mlx.optimizers as optim
    lm = model.language_model
    n_layers, hidden = len(lm.model.layers), lm.args.hidden_size
    mx.random.seed(seed)
    rng = random.Random(seed)
    chunks = list(chunks)
    rng.shuffle(chunks)
    n_val = max(1, len(chunks) // 10) if len(chunks) > 1 else 0
    val, trn = chunks[:n_val], chunks[n_val:] or chunks
    params = init_params(n_layers, hidden, rank)
    identity = {k: mx.zeros_like(v) for k, v in params.items()}       # обычный logit lens
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(p, H, target):
        return layer_kls(lm, p, H, target).mean()

    step_fn = mx.value_and_grad(loss_fn)

    def sample(ids):
        return sorted(rng.sample(range(len(ids)), min(positions, len(ids))))

    def evaluate(p):
        tot = mx.zeros((n_layers,))
        for ids in val:
            H, target = layer_states(lm, ids, list(range(len(ids)))[-positions * 2:])
            tot = tot + layer_kls(lm, p, H, target)
            mx.eval(tot)
        return (tot / max(len(val), 1)).tolist()

    t0 = time.perf_counter()
    for step in range(steps):
        ids = trn[step % len(trn)]
        H, target = layer_states(lm, ids, sample(ids))
        loss, grads = step_fn(params, H, target)
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state, loss)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"  шаг {step + 1}/{steps}  средний KL по слоям {loss.item():.4f}  "
                f"({time.perf_counter() - t0:.0f} с)")

    meta = {"layers": n_layers, "hidden": hidden, "rank": rank, "steps": steps,
            "train_chunks": len(trn), "val_chunks": len(val)}
    if val:
        raw, tuned = evaluate(identity), evaluate(params)
        meta["kl_raw"] = json.dumps([round(x, 4) for x in raw])
        meta["kl_tuned"] = json.dumps([round(x, 4) for x in tuned])
        log("\nKL до последнего слоя на отложенных текстах (меньше — линза точнее):")
        log(f"{'слой':>6} {'logit lens':>12} {'tuned lens':>12}")
        for l in range(n_layers):
            log(f"{l:>6} {raw[l]:>12.4f} {tuned[l]:>12.4f}")
        log(f"{'средн.':>6} {sum(raw) / n_layers:>12.4f} {sum(tuned) / n_layers:>12.4f}")
    return TunedLens(params, meta)


def main():
    ap = argparse.ArgumentParser(description="Обучение tuned lens для Qwen (MLX)")
    ap.add_argument("model_path")
    ap.add_argument("--texts", help="текстовый файл-корпус (чем больше, тем лучше)")
    ap.add_argument("--out", default="tuned_lens.safetensors")
    ap.add_argument("--rank", type=int, default=64,
                    help="ранг преобразования; 0 — полная матрица (hidden² параметров на слой)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=128, help="длина куска текста в токенах")
    ap.add_argument("--positions", type=int, default=16, help="сколько позиций брать из куска за шаг")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model, tok = load_model(a.model_path)
    if a.texts:
        chunks = chunks_from_text(tok, Path(a.texts).read_text(encoding="utf-8"), a.seq_len)
        print(f"корпус: {len(chunks)} кусков по {a.seq_len} токенов")
    else:
        import layer_map
        chunks = chunks_from_prompts(model, tok, layer_map.load_prompts(), 48)
        print("ВНИМАНИЕ: корпус маленький, линза получится грубой. Для аккуратной укажите --texts.")
    if not chunks:
        raise SystemExit("Пустой корпус")
    lens = train(model, chunks, a.rank, a.steps, a.lr, a.positions, a.seed)
    lens.meta["model"] = model_fingerprint(a.model_path).get("fingerprint", "")
    lens.save(a.out)
    print(f"\nЛинза сохранена: {a.out}\nИспользование: python trace_qwen.py {a.model_path} "
          f"\"промпт\" --tuned-lens {a.out}")


if __name__ == "__main__":
    sys.exit(main())
