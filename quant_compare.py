#!/usr/bin/env python3
"""
quant_compare.py — где квантизация искажает модель: сравнение двух вариантов одной модели.

    python quant_compare.py /путь/к/исходной /путь/к/4-битной
    python quant_compare.py /путь/к/8-битной /путь/к/4-битной --prompts prompts/extended.txt
    python quant_compare.py REF TEST --run-dir trace_logs/<папка>      # добавить в отчёт

Первая модель — эталон (исходная или менее сжатая), вторая — проверяемая.
Модели загружаются по очереди, так что память нужна только на одну.

Что измеряется на одних и тех же текстах (промпт + ответ эталона):
    по слоям   косинус между векторами двух моделей и относительная ошибка —
               видно, в каком слое ошибка возникает и где она резко растёт;
    на выходе  KL между распределениями следующего токена (с 95% интервалом по промптам),
               доля совпавших лучших токенов, вероятность токена эталона.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_map import greedy_answer, load_prompts  # noqa: E402
from qcore import (bootstrap_ci, eos_ids, format_prompt, kl_div, load_model, log_softmax,  # noqa: E402
                   model_fingerprint)

F32 = mx.float32


def collect(model, tok, prompts, n_tokens=8, keep_pos=16, texts=None, thinking=False,
            raw=False) -> list:
    """Скрытые состояния и распределения модели на заданных текстах.

    texts=None — модель-эталон сама дописывает ответы; иначе берутся готовые тексты
    (результат collect для эталона), чтобы обе модели видели одни и те же токены."""
    lm, eos = model.language_model, eos_ids(model)
    n_layers = len(lm.model.layers)
    out = []
    for k, prompt in enumerate(prompts):
        if texts is None:
            ids = tok.encode(format_prompt(tok, prompt, thinking, raw))
            ans = greedy_answer(lm, ids, n_tokens, eos)
            if not ans:
                continue
            full, start = ids + ans, len(ids)
        else:
            full, start = texts[k]["full"], texts[k]["start"]
        x = mx.array([full])
        res = lm(x, capture_layer_ids=list(range(n_layers)))
        hs = [lm.model.embed_tokens(x)] + list(res.hidden_states[:n_layers])
        H = mx.stack([h[0, -keep_pos:].astype(F32) for h in hs]).astype(mx.float16)   # (L+1, P, D)
        lp = log_softmax(res.logits[0, start - 1:-1])
        mx.eval(H, lp)
        out.append({"prompt": prompt, "full": full, "start": start, "H": H, "logp": lp,
                    "answer": tok.decode(full[start:])})
    return out


def compare(ref: list, test: list) -> dict:
    n_states = ref[0]["H"].shape[0]
    cos_sum, err_sum = [0.0] * n_states, [0.0] * n_states
    kls, agree, p_ref = [], [], []
    for a, b in zip(ref, test):
        ha, hb = a["H"].astype(F32), b["H"].astype(F32)
        na, nb = mx.linalg.norm(ha, axis=-1), mx.linalg.norm(hb, axis=-1)
        cos = ((ha * hb).sum(axis=-1) / (na * nb + 1e-9)).mean(axis=-1)
        err = (mx.linalg.norm(ha - hb, axis=-1) / (na + 1e-9)).mean(axis=-1)
        la, lb = a["logp"], b["logp"]
        top = mx.argmax(la, axis=-1)
        kls.append(kl_div(la, lb).mean().item())
        agree.append((top == mx.argmax(lb, axis=-1)).astype(F32).mean().item())
        p_ref.append(mx.exp(mx.take_along_axis(lb, top[:, None], axis=-1)).mean().item())
        for i, (c, e) in enumerate(zip(cos.tolist(), err.tolist())):
            cos_sum[i] += c
            err_sum[i] += e
    n = len(ref)
    cos_l, err_l = [round(v / n, 5) for v in cos_sum], [round(v / n, 5) for v in err_sum]
    growth = [round(err_l[i] - err_l[i - 1], 5) for i in range(1, n_states)]
    lo, hi = bootstrap_ci(kls)
    return {"version": 1, "prompts": [{"prompt": a["prompt"], "answer": a["answer"],
                                       "kl": round(k, 5)} for a, k in zip(ref, kls)],
            # индекс 0 — эмбеддинги, далее выход каждого слоя
            "cos": cos_l, "rel_err": err_l, "err_growth": growth,
            "kl": round(sum(kls) / n, 5), "kl_lo": round(lo, 5), "kl_hi": round(hi, 5),
            "top1_agree": round(sum(agree) / n, 4), "p_ref_token": round(sum(p_ref) / n, 4)}


def render(res: dict) -> str:
    out = ["", "=" * 78, "СРАВНЕНИЕ МОДЕЛЕЙ (эталон -> проверяемая)", "=" * 78,
           f"эталон:      {res.get('ref', {}).get('path', '?')}",
           f"проверяемая: {res.get('test', {}).get('path', '?')}",
           f"промптов: {len(res['prompts'])}", "",
           f"KL распределений следующего токена: {res['kl']:.5f}  "
           f"[{res['kl_lo']:.5f}, {res['kl_hi']:.5f}]  (0 — модели неразличимы)",
           f"лучший токен совпал в {res['top1_agree']:.1%} позиций; "
           f"средняя вероятность токена эталона: {res['p_ref_token']:.3f}", "",
           f"{'после':>10} {'косинус':>9} {'отн. ошибка':>12} {'прирост':>9}"]
    vmax = max(res["rel_err"]) or 1.0
    for i, (c, e) in enumerate(zip(res["cos"], res["rel_err"])):
        name = "эмбеддинг" if i == 0 else f"слой {i - 1}"
        g = f"{res['err_growth'][i - 1]:>+9.4f}" if i else f"{'':>9}"
        out.append(f"{name:>10} {c:>9.5f} {e:>12.5f} {g}  {'█' * round(24 * e / vmax)}")
    jumps = sorted(range(len(res["err_growth"])), key=lambda i: -res["err_growth"][i])[:5]
    out += ["", "Слои, после которых ошибка растёт сильнее всего (первые кандидаты на более "
                "аккуратную квантизацию):"]
    out += [f"  слой {i:>3}: +{res['err_growth'][i]:.4f}" for i in jumps if res["err_growth"][i] > 0]
    worst = sorted(res["prompts"], key=lambda p: -p["kl"])[:5]
    out += ["", "Промпты, на которых модели расходятся сильнее всего:"]
    out += [f"  KL {p['kl']:.4f}  {p['prompt']!r}" for p in worst]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Сравнение двух вариантов модели по слоям")
    ap.add_argument("ref_path", help="эталон: исходная или менее сжатая модель")
    ap.add_argument("test_path", help="проверяемая (квантизованная) модель")
    ap.add_argument("--prompts", help="файл с промптами, один в строке")
    ap.add_argument("--tokens", type=int, default=8, help="сколько токенов ответа сравнивать")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--run-dir", help="папка прогона trace_qwen.py: результат добавится в неё и в отчёт")
    ap.add_argument("--out", default="quant_compare")
    a = ap.parse_args()

    prompts = load_prompts(a.prompts)
    print(f"эталон: {a.ref_path}")
    model, tok = load_model(a.ref_path)
    shape = (len(model.language_model.model.layers), model.language_model.args.hidden_size)
    ref = collect(model, tok, prompts, a.tokens, thinking=a.thinking, raw=a.raw)
    del model
    gc.collect()
    mx.clear_cache()

    print(f"проверяемая: {a.test_path}")
    model, tok = load_model(a.test_path)
    if (len(model.language_model.model.layers), model.language_model.args.hidden_size) != shape:
        raise SystemExit("Модели разной архитектуры (число слоёв или размер вектора) — "
                         "сравнивать по слоям нельзя.")
    test = collect(model, tok, [r["prompt"] for r in ref], a.tokens, texts=ref)
    res = compare(ref, test)
    res["ref"], res["test"] = model_fingerprint(a.ref_path), model_fingerprint(a.test_path)
    res["layer_types"] = "".join("L" if l.is_linear else "A"
                                 for l in model.language_model.model.layers)
    text = render(res)
    print(text)

    out = Path(a.run_dir) if a.run_dir else Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "quant_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"\nСохранено: {out / 'quant_compare.json'}")
    if a.run_dir:
        with open(out / "trace.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        try:
            import make_report
            print("Отчёт пересобран:", make_report.build_report(out))
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())
