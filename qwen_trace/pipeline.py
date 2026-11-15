"""Один подробный прогон: промпт -> шаблон -> токены -> слои -> ответ, с журналом каждого этапа."""
import time
from datetime import datetime

import mlx.core as mx

from .analysis import (analyze_hidden, embedding_report, lens_grid, logit_attribution,
                       prompt_attribution, top_k, weight_inventory)
from .core import (eos_ids, format_prompt, kl_div, layer_name, log_softmax, pick_token,
                   run_meta)
from .journal import Journal
from .tracer import Tracer


# ───────────────────────── этапы ─────────────────────────
def parse_layers(spec: str, n_layers: int) -> list:
    """'35' -> [35];  '3,35,40-43' -> [3, 35, 40, 41, 42, 43]. Нумерация с нуля."""
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if lo > hi or lo < 0 or hi >= n_layers:
            raise SystemExit(f"--skip-layers: '{part}' вне диапазона 0..{n_layers - 1}")
        out.update(range(lo, hi + 1))
    return sorted(out)


def generate(model, tok, ids, tracer: Tracer, journal: Journal, max_tokens: int,
             detail_steps: int, label: str = "", temperature: float = 0.0,
             top_p: float = 1.0, seed: int = 0, lens=None, token_attr: bool = False) -> dict:
    """Один прогон. Возвращает text, ids, logp0 (распределение первого токена) и
    first — выбранный первый токен и его главный соперник."""
    lm = model.language_model
    n_layers = len(lm.model.layers)
    tracer.tokens = [tok.decode([i]) for i in ids]
    cache = lm.make_cache()
    inputs = mx.array([ids])
    generated, logp0, first = [], None, None
    eos = eos_ids(model)
    mx.random.seed(seed)                         # оба прогона сравнения сэмплируют одинаково

    for step in range(max_tokens):
        detailed = step < detail_steps
        tracer.step = step
        tracer.parts = {}
        tracer.enabled = detailed
        what = "весь промпт (prefill)" if step == 0 else f"один токен {tracer.tokens[-1]!r}"
        if detailed:
            journal.section(f"ЭТАП 3. Шаг генерации {step}: вход — {what}. "
                            f"Каждый вызов модуля (подробности в trace.log)")
            journal.text("времена модулей диагностические: журнал завершает вычисления после "
                         "каждого модуля; реальная скорость — в событии timing", echo=False)
        t0 = time.perf_counter()
        out = lm(inputs, cache=cache,
                 capture_layer_ids=list(range(n_layers)) if detailed else None)
        logits = out.logits[0, -1]
        mx.eval(logits)
        tracer.enabled = False
        ms = round((time.perf_counter() - t0) * 1000, 1)

        best, extra = top_k(logits, tok, 5)
        nxt = pick_token(logits, temperature, top_p)
        if nxt == best[0]["id"]:
            chosen = best[0]
        else:                                    # сэмплирование выбрало не самый вероятный
            lp = log_softmax(logits)
            chosen = {"id": nxt, "token": tok.decode([nxt]),
                      "prob": round(mx.exp(lp[nxt]).item(), 4),
                      "logit": round(logits[nxt].item(), 3)}
        other = next(b for b in best if b["id"] != nxt)
        if step == 0:
            logp0 = log_softmax(logits)
            mx.eval(logp0)
            first = {"target": chosen, "other": other}

        hidden = getattr(out, "hidden_states", None)
        if detailed and hidden:
            emb_out = lm.model.embed_tokens(inputs)
            analyze_hidden(lm, hidden[:n_layers], emb_out, tok, nxt, logits, journal, step, lens)
            if step == 0 and hidden[0].shape[1] == len(ids):
                lens_grid(lm, hidden[:n_layers], ids, tok, journal, step, lens)
            cands = [chosen] + [b for b in best if b["id"] != chosen["id"]][:4]
            logit_attribution(lm, tracer, emb_out[0, -1], hidden[n_layers - 1][0, -1],
                              cands, logits, journal, step)

        # 4. Логиты -> вероятности -> выбор токена
        if step == 0 or detailed:
            journal.section(f"ЭТАП 4. {label}Шаг {step}: логиты -> вероятности -> токен")
        journal.text(f"шаг {step:3d} ({ms} мс)  выбран {chosen['token']!r}   кандидаты: " +
                     "  ".join(f"{b['token']!r} {b['prob']}" for b in best) +
                     f"   [энтропия {extra['entropy']}, токенов в nucleus(0.95): {extra['nucleus95']}]")
        if detailed:                              # пошаговый расчёт энтропии — в журнал
            d = extra["detail"]
            journal.text(f"   энтропия H = −Σ p·ln p = {d['top_term']} (топ-{len(d['top'])}) + "
                         f"{d['tail']['term']} (ещё {d['tail']['n']} токенов, масса "
                         f"{d['tail']['mass']}) = {extra['entropy']} нат = {d['bits']} бит; "
                         f"exp(H) = {d['perplexity']} (как выбор из стольких равных вариантов); "
                         f"максимум ln V = {d['max_entropy']}; ln Σ exp(z) = {d['lse']}",
                         echo=False)
        journal.event("sample", run=label.strip(" :") or "основной", step=step, ms=ms,
                      traced=detailed, chosen=chosen, candidates=best, **extra)
        if detailed and token_attr:
            prompt_attribution(lm, ids + generated, tok, chosen, other, journal, step, len(ids))

        if nxt in eos:
            journal.text("получен токен конца ответа — стоп")
            break
        generated.append(nxt)
        tracer.tokens.append(chosen["token"])
        inputs = mx.array([[nxt]])

    return {"text": tok.decode(generated), "ids": generated, "logp0": logp0, "first": first}


def compare(base: dict, ablated: dict, tok, journal: Journal, skipped: list):
    """Насколько отключение слоёв изменило предсказание первого токена и ответ."""
    lp_b, lp_a = base["logp0"], ablated["logp0"]
    ids_b, ids_a = base["ids"], ablated["ids"]
    journal.section(f"СРАВНЕНИЕ: обычный прогон и прогон без слоёв {skipped}")
    kl = kl_div(lp_b, lp_a).item()
    top_b = mx.argsort(-lp_b)[:5].tolist()
    top_a = mx.argsort(-lp_a)[:5].tolist()
    pb, pa = mx.exp(lp_b).tolist(), mx.exp(lp_a).tolist()
    rows = []
    journal.text("Первый токен ответа (вероятность: обычный прогон -> без слоёв):")
    for i in dict.fromkeys(top_b + top_a):
        rows.append({"id": i, "token": tok.decode([i]),
                     "base": round(pb[i], 4), "ablated": round(pa[i], 4)})
        journal.text(f"  {rows[-1]['token']!r:>14}  {pb[i]:.4f} -> {pa[i]:.4f}  "
                     f"({pa[i] - pb[i]:+.4f})")
    same = 0
    for x, y in zip(ids_b, ids_a):
        if x != y:
            break
        same += 1
    journal.text(f"\nKL-расхождение распределений первого токена: {kl:.4f} "
                 f"(0 — слои ни на что не повлияли)")
    journal.text(f"лучший токен {'НЕ изменился' if top_b[0] == top_a[0] else 'ИЗМЕНИЛСЯ'}; "
                 f"ответы совпадают в первых {same} токенах из {len(ids_b)}")
    journal.text(f"\nобычный ответ:   {base['text']!r}")
    journal.text(f"без слоёв:       {ablated['text']!r}")
    journal.event("ablation", skipped=skipped, kl=round(kl, 5), first_token=rows,
                  top_changed=top_b[0] != top_a[0], common_prefix=same,
                  answer_base=base["text"], answer_ablated=ablated["text"])


def clean_timing(lm, ids, journal: Journal, n: int = 2):
    """Реальное время обработки промпта — без журнала и без подробностей."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(lm(mx.array([ids])).logits)
        times.append((time.perf_counter() - t0) * 1000)
    ms = round(min(times), 1)
    journal.event("timing", prefill_ms=ms, tokens=len(ids),
                  tokens_per_s=round(len(ids) / (ms / 1000), 1) if ms else None)
    journal.text(f"реальное время prefill без журнала: {ms} мс на {len(ids)} токенов")


def run(model, tok, prompt: str, journal: Journal, max_tokens: int = 20,
        detail_steps: int = None, thinking: bool = False, raw: bool = False,
        skip_layers=(), wstats: bool = False, temperature: float = 0.0, top_p: float = 1.0,
        seed: int = 0, token_attr: bool = True, lens=None,
        model_path=None):
    lm = model.language_model
    n_layers = len(lm.model.layers)
    skip_layers = sorted(set(skip_layers))
    if detail_steps is None:             # по умолчанию — все токены ответа (и токен конца)
        detail_steps = max_tokens
    journal.event("meta", **run_meta(
        model_path, prompt=prompt, max_tokens=max_tokens, detail_steps=detail_steps,
        thinking=thinking, raw=raw, skip_layers=skip_layers, temperature=temperature,
        top_p=top_p, seed=seed,
        tuned_lens=dict(lens.meta) if lens is not None else None))

    # 0. Модель
    journal.section("ЭТАП 0. Модель")
    kinds = ["L" if l.is_linear else "A" for l in lm.model.layers]
    journal.text(f"слоёв: {n_layers}   скрытый размер: {lm.args.hidden_size}   "
                 f"словарь: {lm.args.vocab_size}")
    journal.text("типы слоёв (L — линейное внимание, A — полное): " + "".join(kinds))
    if skip_layers:
        journal.text("будут отключены слои: " + ", ".join(
            f"{i} ({'линейный' if kinds[i] == 'L' else 'полное внимание'})"
            for i in skip_layers))
    cfg = {k: getattr(lm.args, k) for k in
           ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "vocab_size", "num_hidden_layers", "full_attention_interval",
            "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
            "linear_value_head_dim", "tie_word_embeddings", "rms_norm_eps")
           if hasattr(lm.args, k)}
    cfg = {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in cfg.items()}
    emb = lm.model.embed_tokens
    journal.event("model", layers=n_layers, hidden=lm.args.hidden_size,
                  vocab=lm.args.vocab_size, layer_types="".join(kinds),
                  skip_layers=skip_layers, config=cfg, bits=getattr(emb, "bits", None),
                  group_size=getattr(emb, "group_size", None),
                  created=datetime.now().isoformat(timespec="seconds"))

    # 1. Шаблон чата
    journal.section("ЭТАП 1. Промпт -> шаблон чата")
    text = format_prompt(tok, prompt, thinking, raw)
    journal.text("исходный промпт: " + repr(prompt))
    journal.text("строка, которую реально получает модель:\n" + text)
    journal.event("prompt", prompt=prompt, formatted=text, raw=raw, thinking=thinking)

    # 2. Токенизация
    journal.section("ЭТАП 2. Токенизация")
    ids = tok.encode(text)
    journal.text(f"токенов: {len(ids)}")
    for pos, i in enumerate(ids):
        journal.text(f"  поз {pos:3d}  id {i:7d}  {tok.decode([i])!r}", echo=pos < 40)
    journal.event("tokens", ids=ids, tokens=[tok.decode([i]) for i in ids])
    embedding_report(lm, ids, tok, journal)
    clean_timing(lm, ids, journal)

    sampling = dict(temperature=temperature, top_p=top_p, seed=seed)
    if temperature > 0:
        journal.text(f"сэмплирование: temperature {temperature}, top_p {top_p}, seed {seed}")

    # 3–4. Прогон по слоям + генерация
    with Tracer(lm, journal, wstats=wstats) as tracer:
        base = None
        if skip_layers:
            journal.section("ОБЫЧНЫЙ ПРОГОН (все слои включены, без подробностей) — для сравнения")
            base = generate(model, tok, ids, tracer, journal, max_tokens, 0,
                            "Обычный прогон: ", **sampling)
            tracer.skip = {layer_name(i) for i in skip_layers}
            journal.section(f"ПРОГОН БЕЗ СЛОЁВ {skip_layers}")
        label = "Без слоёв: " if skip_layers else ""
        result = generate(model, tok, ids, tracer, journal, max_tokens, detail_steps, label,
                          lens=lens, token_attr=token_attr, **sampling)
        tracer.skip = set()

    # 5. Ответ
    journal.section("ЭТАП 5. Декодирование ответа")
    journal.text(result["text"])
    journal.event("answer", ids=result["ids"], text=result["text"], skip_layers=skip_layers)
    if base is not None:
        compare(base, result, tok, journal, skip_layers)

    # 6. Веса
    weight_inventory(lm, journal)
    return result["text"]
