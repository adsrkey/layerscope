"""Анализ одного шага генерации: сводка весов, эмбеддинги, распределение и энтропия, logit lens,
разложение логита по слоям, вклад токенов контекста."""
import math
import time

import mlx.core as mx

from .attribution import direct_logit_attribution, occlusion_attribution
from .core import attn_name, kl_div, layer_of, lens_logits, log_softmax, mlp_name
from .journal import Journal, weight_info
from .tracer import Tracer


# ───────────────────────── вспомогательное ─────────────────────────
def category(name: str) -> str:
    if "embed_tokens" in name:
        return "эмбеддинги"
    if "lm_head" in name:
        return "lm_head"
    if ".self_attn" in name:
        return "полное внимание"
    if ".linear_attn" in name:
        return "линейное внимание"
    if ".mlp" in name:
        return "MLP"
    if "norm" in name:
        return "нормализация"
    return "прочее"


def weight_inventory(lm, journal: Journal, prefix="language_model"):
    """Все тензоры модели: по модулям, категориям и слоям. Имена совпадают с trace.log."""
    mods, total_p, total_b = [], 0, 0
    for n, m in lm.named_modules():
        info = weight_info(m)
        if not info:
            continue
        name = f"{prefix}.{n}" if n else prefix
        mods.append({"m": name, "c": type(m).__name__, "cat": category(name),
                     "layer": layer_of(name), "shape": info.get("logical_shape"),
                     "bits": info.get("bits"), "params": info["params"], "bytes": info["bytes"],
                     "tensors": {k: v["shape"] for k, v in info["tensors"].items()}})
        total_p += info["params"]
        total_b += info["bytes"]
    journal.event("weights", modules=mods, total_params=total_p, total_bytes=total_b)

    journal.section("ЭТАП 6. Сводка весов (имена модулей совпадают с trace.log)")
    cats = {}
    for m in mods:
        c = cats.setdefault(m["cat"], {"n": 0, "p": 0, "b": 0})
        c["n"] += 1; c["p"] += m["params"]; c["b"] += m["bytes"]
    journal.text(f"{'категория':<22}{'модулей':>9}{'параметров':>16}{'МБ':>10}{'доля':>8}")
    for k, c in sorted(cats.items(), key=lambda kv: -kv[1]["p"]):
        journal.text(f"{k:<22}{c['n']:>9}{c['p']:>16,}{c['b'] / 1e6:>10.1f}"
                     f"{c['p'] / total_p:>8.1%}".replace(",", " "))
    journal.text(f"{'ИТОГО':<22}{len(mods):>9}{total_p:>16,}{total_b / 1e6:>10.1f}".replace(",", " "))
    return mods


def embedding_row_norms(emb, chunk: int = 8192) -> mx.array:
    """Длина вектора каждого токена словаря (по частям, чтобы не раздувать память)."""
    w = emb["weight"]
    out = []
    for s in range(0, w.shape[0], chunk):
        e = min(w.shape[0], s + chunk)
        if "scales" in emb:
            rows = mx.dequantize(w[s:e], emb["scales"][s:e], emb["biases"][s:e],
                                 group_size=emb.group_size, bits=emb.bits)
        else:
            rows = w[s:e]
        r = mx.linalg.norm(rows.astype(mx.float32), axis=-1)
        mx.eval(r)
        out.append(r)
    return mx.concatenate(out)


def embedding_report(lm, ids, tok, journal: Journal, k: int = 5, max_vec: int = 48):
    """Эмбеддинги токенов промпта: длина, первые значения, ближайшие токены словаря
    (по косинусу) и косинусы между токенами промпта."""
    emb = lm.model.embed_tokens
    shown = ids[:200]
    uniq = list(dict.fromkeys(shown))[:64]
    t0 = time.perf_counter()
    norms_all = embedding_row_norms(emb)
    vecs = emb(mx.array(uniq))                                    # (U, D)
    vf = vecs.astype(mx.float32)
    vn = mx.linalg.norm(vf, axis=-1)
    dots = emb.as_linear(vecs).astype(mx.float32)                # (U, V)
    cos = dots / (norms_all[None, :] * vn[:, None] + 1e-9)
    top = mx.argsort(-cos, axis=-1)[:, :k + 1]
    topc = mx.take_along_axis(cos, top, axis=-1)
    allv = emb(mx.array(shown)).astype(mx.float32)
    an = allv / (mx.linalg.norm(allv, axis=-1, keepdims=True) + 1e-9)
    cmat = an[:48] @ an[:48].T
    mx.eval(vn, top, topc, allv, cmat, norms_all)
    per = {}
    for j, tid in enumerate(uniq):
        nb = [{"id": i, "token": tok.decode([i]), "cos": round(c, 3)}
              for i, c in zip(top[j].tolist(), topc[j].tolist()) if i != tid][:k]
        per[tid] = {"norm": round(vn[j].item(), 3), "head": [round(x, 4) for x in vf[j, :max_vec].tolist()],
                    "neighbors": nb}
    toks = [{"pos": p, "id": tid, "token": tok.decode([tid]), **per[tid]}
            for p, tid in enumerate(shown) if tid in per]
    nm = norms_all
    journal.event("embeddings", dim=int(vf.shape[-1]), vocab=int(norms_all.shape[0]),
                  tokens=toks, cos_matrix=[[round(x, 3) for x in r] for r in cmat.tolist()],
                  row_norm_mean=round(nm.mean().item(), 3), row_norm_std=round(nm.std().item(), 3),
                  seconds=round(time.perf_counter() - t0, 2))
    journal.section("ЭТАП 2б. Эмбеддинги токенов промпта")
    journal.text(f"размер вектора: {vf.shape[-1]}, средняя длина вектора токена словаря: "
                 f"{nm.mean().item():.3f}")
    for t in toks[:40]:
        nb = "  ".join(f"{n['token']!r} {n['cos']}" for n in t["neighbors"][:4])
        journal.text(f"  поз {t['pos']:3d} {t['token']!r:>16}  длина {t['norm']:>7}  "
                     f"первые значения {t['head'][:4]}  ближайшие: {nb}", echo=False)


def sig(x: float, n: int = 4) -> float:
    """Число с n значащими цифрами (для малых вероятностей, которые иначе округлятся в 0)."""
    return float(f"{x:.{n}g}")


def top_k(logits_row: mx.array, tok, k: int = 5, detail_k: int = 20):
    """Лучшие кандидаты (токен, логит, вероятность) и характеристики распределения.

    Энтропия считается точно: p = exp(z − logsumexp(z)), вклад каждого токена −p·ln p.
    В поле detail — пошаговый расчёт для отчёта: вклады топ-20 токенов и хвоста."""
    lf = logits_row.astype(mx.float32)
    lse = mx.logsumexp(lf)
    logp = lf - lse
    probs = mx.exp(logp)
    term = -probs * logp                               # вклад токена в энтропию
    order = mx.argsort(-probs)
    top = order[:max(k, detail_k)]
    sp = probs[order]
    ent = term.sum()
    nuc = (mx.cumsum(sp) < 0.95).sum() + 1
    mass_top = sp[:detail_k].sum()
    pv, lv, tv = probs[top], lf[top], term[top]
    mx.eval(top, pv, lv, tv, ent, nuc, mass_top, lse)
    ids, pl, ll, tl = top.tolist(), pv.tolist(), lv.tolist(), tv.tolist()
    best = [{"id": i, "token": tok.decode([i]), "prob": round(p, 4), "logit": round(l, 3)}
            for i, p, l in zip(ids[:k], pl[:k], ll[:k])]
    H, V = ent.item(), int(lf.shape[-1])
    top_term = sum(tl[:detail_k])
    detail = {"lse": round(lse.item(), 4), "vocab": V,
              "top": [{"id": i, "token": tok.decode([i]), "logit": round(l, 3),
                       "prob": sig(p), "term": sig(t)}
                      for i, p, l, t in zip(ids[:detail_k], pl, ll, tl)],
              "top_term": sig(top_term),
              "tail": {"n": V - detail_k, "mass": sig(max(0.0, 1.0 - mass_top.item())),
                       "term": sig(max(0.0, H - top_term))},
              "bits": round(H / math.log(2), 4), "perplexity": round(math.exp(H), 3),
              "max_entropy": round(math.log(V), 4)}
    return best, {"entropy": round(H, 4), "nucleus95": int(nuc.item()),
                  "top20_mass": round(mass_top.item(), 4), "detail": detail}


def analyze_hidden(lm, hidden, emb_out, tok, final_id, final_logits, journal: Journal, step,
                   lens=None):
    """Для каждого слоя: logit lens (что модель ответила бы здесь), ранг итогового токена,
    расстояние до настоящего итогового распределения, нормы вектора каждой позиции и
    насколько слой изменил вектор последнего токена. С tuned lens — то же через линзу."""
    prev = emb_out[0].astype(mx.float32)
    resid, rel_l, cos_l = [], [], []
    emb_norms = mx.linalg.norm(prev, axis=-1)
    final_lp = log_softmax(final_logits)
    journal.text(f"\n--- Logit lens, шаг {step}: лучший токен после каждого слоя ---")
    journal.text("(в ранних слоях logit lens систематически неточен: их векторы ещё не в базисе "
                 "последнего слоя; столбец KL показывает, насколько)", echo=False)

    def read(h_last):
        lp = log_softmax(lens_logits(lm, h_last[None, None, :])[0, 0])
        pf = mx.exp(lp[final_id])
        rank = (lp > lp[final_id]).sum() + 1
        ent = -(mx.exp(lp) * lp).sum()
        kl = kl_div(final_lp, lp)
        top = mx.argsort(-lp)[:5]
        pt = mx.exp(lp[top])
        mx.eval(pf, rank, ent, kl, top, pt)
        best = [{"id": t, "token": tok.decode([t]), "prob": round(p, 4)}
                for t, p in zip(top.tolist(), pt.tolist())]
        return best, pf.item(), int(rank.item()), ent.item(), kl.item()

    for i, h in enumerate(hidden):
        hf = h[0].astype(mx.float32)
        norms = mx.linalg.norm(hf, axis=-1)
        last, plast = hf[-1], prev[-1]
        dn = mx.linalg.norm(plast)
        rel = mx.linalg.norm(last - plast) / (dn + 1e-9)
        cos = (last * plast).sum() / (norms[-1] * dn + 1e-9)
        mx.eval(norms, rel, cos)
        best, pf, rank, ent, kl = read(h[0, -1])
        kind = "линейный" if lm.model.layers[i].is_linear else "ПОЛНОЕ ВНИМАНИЕ"
        rec = {"step": step, "layer": i, "layer_type": kind, "top": best,
               "p_final": round(pf, 5), "rank_final": rank, "entropy": round(ent, 3),
               "kl_final": round(kl, 4)}
        line = (f"слой {i:2d} ({kind:15s})  " +
                "  ".join(f"{b['token']!r} {b['prob']}" for b in best[:3]) +
                f"   | итоговый токен: p={pf:.4f}, ранг {rank}, KL до итога {kl:.3f}")
        if lens is not None and i < lens.n_layers:
            tb, tpf, trank, _, tkl = read(lens(i, h[0, -1]))
            rec.update(tuned_top=tb, tuned_p_final=round(tpf, 5), tuned_rank=trank,
                       tuned_kl=round(tkl, 4))
            line += f"   | tuned lens: {tb[0]['token']!r} p={tpf:.4f}, ранг {trank}, KL {tkl:.3f}"
        journal.event("logit_lens", **rec)
        journal.text(line)
        resid.append([round(v, 2) for v in norms.tolist()[-256:]])
        rel_l.append(round(rel.item(), 4))
        cos_l.append(round(cos.item(), 4))
        prev = hf
    journal.event("residual", step=step, norms=resid, rel=rel_l, cos=cos_l,
                  emb_norms=[round(v, 2) for v in emb_norms.tolist()[-256:]],
                  final_id=final_id, final_token=tok.decode([final_id]))


def logit_attribution(lm, tracer: Tracer, emb_last, h_final, cands: list, final_logits,
                      journal: Journal, step):
    """Разложение логитов кандидатов на вклады эмбеддинга и каждого слоя.

    Раскладывается логит последней позиции шага — тот, по которому выбирается токен.
    cands[0] — выбранный токен, cands[1] — главный соперник; остальные нужны, чтобы в отчёте
    можно было сравнить любую пару."""
    skipped = {layer_of(n) for n in tracer.skip}
    parts, meta = [("emb", emb_last)], [{"layer": None, "kind": "emb"}]
    for i in range(len(lm.model.layers)):
        if i in skipped:
            continue
        for kind, name in (("attn", attn_name(lm, i)), ("mlp", mlp_name(i))):
            if name in tracer.parts:
                parts.append((name, tracer.parts[name]))
                meta.append({"layer": i, "kind": kind})
    if len(parts) < 2:
        return
    ids = [c["id"] for c in cands]
    rows, totals, gap = direct_logit_attribution(lm, parts, h_final, ids)
    actual = final_logits[mx.array(ids)].astype(mx.float32).tolist()
    comps = [{**m, "logit": round(r[0], 4), "diff": round(r[0] - r[1], 4),
              "logits": [round(v, 3) for v in r]} for m, r in zip(meta, rows)]
    target, other = cands[0], cands[1]
    journal.event("dla", step=step, target=target, other=other, cands=cands, components=comps,
                  logit_sum=round(totals[0], 4), logit_actual=round(actual[0], 4),
                  logits_sum=[round(v, 4) for v in totals],
                  logits_actual=[round(v, 4) for v in actual], residual_gap=round(gap, 5))
    journal.text(f"\n--- Разложение логита, шаг {step}: что подняло {target['token']!r} "
                 f"над {other['token']!r} ---")
    journal.text(f"сумма вкладов {totals[0]:.3f}, настоящий логит {actual[0]:.3f} "
                 f"(расхождение — точность половинных чисел; невязка потока {gap:.2%}); "
                 f"те же вклады посчитаны ещё для {len(cands) - 2} кандидатов — в отчёте можно "
                 f"выбрать любую пару")
    name = {"emb": "эмбеддинг", "attn": "внимание", "mlp": "MLP"}
    for c in sorted(comps, key=lambda c: -abs(c["diff"]))[:10]:
        where = name[c["kind"]] if c["layer"] is None else f"слой {c['layer']:2d} {name[c['kind']]}"
        journal.text(f"  {where:<18} вклад в разность {c['diff']:+8.3f}   в логит {c['logit']:+8.3f}")


def lens_grid(lm, hidden, ids, tok, journal: Journal, step, lens=None, max_pos: int = 64):
    """Logit lens на КАЖДОЙ позиции промпта (а не только на последней): что модель
    ожидала бы дальше после каждого токена на каждом слое. Для токенов промпта
    следующий токен известен, поэтому записывается и ранг настоящего продолжения."""
    L = len(ids)
    pos0 = L - min(L, max_pos)
    tgt = mx.array([ids[i + 1] if i + 1 < L else 0 for i in range(pos0, L)])
    tops, probs, ranks = [], [], []
    for i, h in enumerate(hidden):
        x = h[0, pos0:, :]
        if lens is not None and i < lens.n_layers:
            x = lens(i, x)
        lp = log_softmax(lens_logits(lm, x[None])[0])                     # (P, V)
        best = mx.max(lp, axis=-1)
        rank = (lp > mx.take_along_axis(lp, tgt[:, None], axis=-1)).sum(axis=-1) + 1
        top = mx.argmax(lp, axis=-1)
        pr = mx.exp(best)
        mx.eval(top, pr, rank)
        tops.append(top.tolist())
        probs.append([round(v, 3) for v in pr.tolist()])
        ranks.append(rank.tolist())
    vocab = {str(t): tok.decode([t]) for row in tops for t in row}
    journal.event("lens_grid", step=step, tuned=lens is not None, positions=list(range(pos0, L)),
                  top=tops, p=probs, rank=ranks, vocab=vocab)
    journal.text("\n--- Logit lens по всем позициям промпта: ожидание следующего токена "
                 "на последнем слое ---", echo=False)
    for j, pos in enumerate(range(pos0, L)):
        nxt = tok.decode([ids[pos + 1]]) if pos + 1 < L else "(ответ)"
        journal.text(f"  поз {pos:3d} {tok.decode([ids[pos]])!r:>16} -> {vocab[str(tops[-1][j])]!r:<14} "
                     f"p={probs[-1][j]:<6} | настоящий следующий {nxt!r}, ранг {ranks[-1][j]}",
                     echo=False)


def prompt_attribution(lm, ctx_ids, tok, target: dict, other: dict, journal: Journal,
                       step: int = 0, n_prompt: int = None):
    """Какие токены контекста сильнее всего повлияли на выбор токена на этом шаге.

    Контекст шага k — промпт плюс k уже выбранных токенов ответа, поэтому на поздних
    шагах часть влияния приходится на собственные предыдущие токены модели. Метод — прямая
    подмена токена средним эмбеддингом (occlusion): градиентные оценки на этой модели не
    совпали с прямыми измерениями и не используются."""
    n_prompt = len(ctx_ids) if n_prompt is None else n_prompt
    t0 = time.perf_counter()
    try:
        attr, value = occlusion_attribution(lm, ctx_ids, target["id"], other["id"])
    except Exception as e:                           # не должно ломать прогон
        journal.text(f"\n[шаг {step}: вклад токенов не посчитан: {type(e).__name__}: {e}]")
        return
    method = "подмена токена средним эмбеддингом (occlusion)"
    toks = [tok.decode([i]) for i in ctx_ids]
    journal.event("token_attr", step=step, method=method, target=target, other=other,
                  scores=[round(v, 4) for v in attr], tokens=toks, n_prompt=n_prompt,
                  metric=round(value, 4), seconds=round(time.perf_counter() - t0, 2))
    journal.section(f"ЭТАП 4б. Шаг {step}: вклад токенов контекста в выбор токена")
    journal.text(f"метод: {method}; метрика — логит {target['token']!r} минус логит "
                 f"{other['token']!r} = {value:.3f}")
    if len(ctx_ids) > n_prompt:
        total = sum(abs(v) for v in attr) or 1.0
        own = sum(abs(v) for v in attr[n_prompt:]) / total
        journal.text(f"контекст: {n_prompt} токенов промпта + {len(ctx_ids) - n_prompt} уже "
                     f"сгенерированных; на сгенерированные приходится {own:.0%} вклада")
    journal.text("плюс — токен подталкивал к выбранному токену, минус — к альтернативе "
                 "(насколько изменилась метрика, если токен заменить средним эмбеддингом):")
    for p in sorted(range(len(attr)), key=lambda p: -abs(attr[p]))[:12]:
        journal.text(f"  поз {p:3d} {toks[p]!r:>16}  {attr[p]:+.4f}" +
                     ("   (ответ)" if p >= n_prompt else ""))
