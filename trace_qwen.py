#!/usr/bin/env python3
"""
trace_qwen.py — полный журнал одного прогона модели: от промпта до ответа.

Запуск:
    python trace_qwen.py /путь/к/папке/модели "Столица Франции — это"
    python trace_qwen.py /путь/к/папке/модели "..." --max-tokens 30 --detail-steps 2
    python trace_qwen.py /путь/к/папке/модели "..." --skip-layers 3,35,40-43
    python trace_qwen.py /путь/к/папке/модели "..." --layer-map      # + карта важности слоёв
    python trace_qwen.py /путь/к/папке/модели "..." --temperature 0.7 --top-p 0.9 --seed 1
    python trace_qwen.py /путь/к/папке/модели "..." --tuned-lens tuned_lens.safetensors
    python trace_qwen.py /путь/к/папке/модели --batch prompts/extended.txt --layer-map
        # пакетный режим: разбор КАЖДОГО промпта в одном report.html с переключателем

Результат — папка trace_logs/<дата_время>/ (в пакетном режиме — по подпапке на промпт,
плюс один report.html со сводкой и переключателем промптов и layer_map.json, общий для всех):
    trace.log       читаемый журнал всех этапов (с картой важности, если включена)
    events.jsonl    те же события построчно в JSON (первое событие — метаданные прогона)
    layer_map.json  карта важности слоёв (только с --layer-map)
    report.html     интерактивный отчёт: открывается двойным щелчком, работает без сети

Что считается для подробных шагов (--detail-steps):
    каждый вызов модуля, logit lens (и tuned lens, если дана), внимание по головам,
    разложение итогового логита по слоям (DLA), вклад токенов контекста (промпт и уже
    выбранные токены ответа) в выбор каждого токена.

Рассчитан на Qwen3.5/3.8 в формате MLX (mlx-vlm, model_type "qwen3_5").
"""
import argparse
import contextlib
import io
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from attribution import direct_logit_attribution, token_attribution
from qcore import (Hooks, attn_name, eos_ids, format_prompt, kl_div, layer_name, layer_of,
                   lens_logits, load_model, log_softmax, mlp_name, pick_token, run_meta)


# ───────────────────────── журнал ─────────────────────────
class Journal:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.txt = open(out_dir / "trace.log", "w", encoding="utf-8")
        self.jsonl = open(out_dir / "events.jsonl", "w", encoding="utf-8")
        self.n = 0

    def section(self, title: str):
        line = f"\n{'=' * 78}\n{title}\n{'=' * 78}"
        self.txt.write(line + "\n")
        print(line)

    def text(self, line: str = "", echo: bool = True):
        self.txt.write(line + "\n")
        if echo:
            print(line)

    def event(self, kind: str, **data):
        self.n += 1
        rec = {"n": self.n, "kind": kind, **data}
        self.jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        self.txt.close()
        self.jsonl.close()


def stats(x: mx.array) -> dict:
    """Сводка по тензору: форма и основные статистики."""
    f = x.astype(mx.float32)
    last = f.reshape(-1, f.shape[-1])[-1]          # вектор последней позиции
    vals = [f.mean(), f.std(), f.min(), f.max(), mx.linalg.norm(last)]
    mx.eval(*vals)
    mean, std, mn, mxv, norm = (round(v.item(), 4) for v in vals)
    return {"shape": list(x.shape), "dtype": str(x.dtype).replace("mlx.core.", ""),
            "mean": mean, "std": std, "min": mn, "max": mxv, "last_norm": norm}


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def weight_info(module, with_stats: bool = False) -> dict:
    """Тензоры, принадлежащие самому модулю (без вложенных): формы, типы, размер,
    логическая форма матрицы (для 4-битных — распакованная) и число параметров."""
    tensors, nbytes = {}, 0
    for key, val in module.items():
        if isinstance(val, mx.array):
            tensors[key] = {"shape": list(val.shape),
                            "dtype": str(val.dtype).replace("mlx.core.", "")}
            nbytes += val.nbytes
    if not tensors:
        return {}
    info = {"tensors": tensors, "bytes": nbytes}
    bits = getattr(module, "bits", None)
    w = module["weight"] if "weight" in module else None
    if bits and isinstance(w, mx.array):
        out_f, in_f = w.shape[0], w.shape[-1] * 32 // bits
        info.update(bits=bits, group_size=getattr(module, "group_size", None),
                    logical_shape=[out_f, in_f], params=out_f * in_f)
    else:
        info["params"] = sum(_prod(t["shape"]) for t in tensors.values())
        if isinstance(w, mx.array):
            info["logical_shape"] = list(w.shape)
    if with_stats and isinstance(w, mx.array):
        try:                                         # для огромных матриц — по первым строкам
            rows = min(w.shape[0], 16384)
            if bits:
                full = mx.dequantize(w[:rows], module["scales"][:rows], module["biases"][:rows],
                                     group_size=module.group_size, bits=bits)
            else:
                full = w[:rows]
            f = full.astype(mx.float32)
            vals = [f.mean(), f.std(), mx.abs(f).max()]
            mx.eval(*vals)
            info["wstats"] = {"mean": round(vals[0].item(), 5), "std": round(vals[1].item(), 5),
                              "absmax": round(vals[2].item(), 4), "rows": rows}
        except Exception as e:                      # статистика не должна ломать прогон
            info["wstats"] = {"error": str(e)}
    return info


def weights_text(info: dict) -> str:
    if not info:
        return "нет весов"
    parts = [f"{k} {tuple(v['shape'])} {v['dtype']}" for k, v in info["tensors"].items()]
    s = ", ".join(parts)
    if "bits" in info:
        s += f", квантизация: {info['bits']} бит, группа {info['group_size']}"
        s += f"; матрица {info['logical_shape'][0]}×{info['logical_shape'][1]}"
    s += f"; {info['params']:,} парам., {info['bytes'] / 1e6:.2f} МБ".replace(",", " ")
    ws = info.get("wstats")
    if ws and "std" in ws:
        s += f"; веса: std={ws['std']} max|w|={ws['absmax']}"
    return s


# ───────────────── перехват вызовов модулей ─────────────────
class Tracer:
    """Журнал вычислений поверх qcore.Hooks: какой модуль, какие веса, что на входе/выходе,
    плюс внимание последней позиции. Работает только внутри `with`: на выходе все обёртки
    снимаются, и модель снова ведёт себя как обычно.

        with Tracer(lm, journal) as tracer:
            tracer.enabled = True
            lm(inputs)

    Времена модулей — диагностические: чтобы снять статистику выхода, после каждого модуля
    вычисления принудительно завершаются (mx.eval), а это ломает ленивый граф MLX.
    Реальная скорость измеряется отдельным прогоном без журнала (событие timing)."""

    def __init__(self, root, journal, wstats: bool = False):
        self.hooks = Hooks(root)
        self.hooks.observer = self
        self.j = journal
        self.enabled = False
        self.step = "?"
        self.stack = []
        self.tokens = []                      # строки всех токенов контекста
        self.wstats = wstats
        self.winfo = {}
        self.parts = {}                       # записи внимания и MLP в остаточный поток (посл. позиция)
        self._sdpa = None

    # отключённые слои живут в Hooks; здесь — для удобства
    @property
    def skip(self):
        return self.hooks.skip

    @skip.setter
    def skip(self, names):
        self.hooks.skip = set(names)

    def __enter__(self):
        self.hooks.install()
        self._patch_attention()
        return self

    def __exit__(self, *exc):
        self._unpatch_attention()
        self.hooks.uninstall()
        return False

    def skipped(self, name):
        self.j.event("layer_skipped", step=self.step, module=name)
        self.j.text(f"{'  ' * len(self.stack)}{name} ОТКЛЮЧЁН: выход заменён входом", echo=False)

    def call(self, name, module, orig, args, kwargs):
        depth = len(self.stack)
        self.stack.append(name)
        t0 = time.perf_counter()
        try:
            out = orig(module, *args, **kwargs)
            if isinstance(out, mx.array):
                mx.eval(out)
        finally:
            self.stack.pop()
        ms = round((time.perf_counter() - t0) * 1000, 2)

        first = name not in self.winfo
        if first:
            self.winfo[name] = weight_info(module, self.wstats)
        winfo = self.winfo[name]
        rec = {"step": self.step, "depth": depth, "module": name,
               "class": type(module).__name__, "ms": ms}
        if first:                                   # веса модуля не меняются: пишем один раз
            rec["weights"] = winfo
        x = args[0] if args else None
        if isinstance(x, mx.array):
            rec["input_shape"] = list(x.shape)
        if isinstance(out, mx.array):
            rec["output"] = stats(out)
            if name.endswith((".self_attn", ".linear_attn", ".mlp")) and out.ndim == 3:
                self.parts[name] = out[0, -1]
        self.j.event("module_call", **rec)

        o = rec.get("output", {})
        self.j.text(
            f"{'  ' * depth}{name} [{rec['class']}] {ms} мс\n" +
            (f"{'  ' * depth}   веса: {weights_text(winfo)}\n" if first else "") +
            f"{'  ' * depth}   вход {rec.get('input_shape')} -> выход {o.get('shape')}"
            f"  mean={o.get('mean')} std={o.get('std')} |h_посл|={o.get('last_norm')}",
            echo=False)
        return out

    # ── внимание: на какие токены смотрит последняя позиция (слои с полным вниманием) ──
    def _patch_attention(self):
        from mlx_vlm.models.qwen3_5 import language as lang
        orig, tracer = lang.scaled_dot_product_attention, self

        def wrapped(queries, keys, values, cache, scale, mask, *a, **kw):
            if tracer.enabled:
                try:
                    tracer._log_attention(queries, keys, scale)
                except Exception as e:                             # журнал не должен ломать прогон
                    tracer.j.text(f"   [внимание не записано: {e}]", echo=False)
            return orig(queries, keys, values, cache, scale, mask, *a, **kw)

        self._sdpa = (lang, orig)
        lang.scaled_dot_product_attention = wrapped

    def _unpatch_attention(self):
        if self._sdpa:
            lang, orig = self._sdpa
            lang.scaled_dot_product_attention = orig
            self._sdpa = None

    def _log_attention(self, queries, keys, scale):
        """Внимание последних позиций запроса (до 48) ко всем токенам контекста.
        Для подробного разбора по головам берётся только самая последняя позиция."""
        nq = min(queries.shape[2], 48)
        q = queries[0, :, -nq:, :].astype(mx.float32)      # (H, nq, D)
        k = keys[0].astype(mx.float32)                     # (Hkv, L, D)
        if k.shape[0] != q.shape[0]:
            k = mx.repeat(k, q.shape[0] // k.shape[0], axis=0)
        L = k.shape[1]
        scores = (q @ k.transpose(0, 2, 1)) * scale        # (H, nq, L)
        qpos = (L - nq) + mx.arange(nq)                    # абсолютные позиции запросов
        causal = mx.arange(L)[None, :] <= qpos[:, None]    # токен видит себя и всё до себя
        w_all = mx.softmax(mx.where(causal[None], scores, -mx.inf), axis=-1)   # (H, nq, L)
        w = w_all[:, -1, :]                                # (H, L): последний токен, по головам
        avg_all = w_all.mean(axis=0)                       # (nq, L): среднее по головам
        avg = avg_all[-1]
        head_pos, head_w = mx.argmax(w, axis=-1), mx.max(w, axis=-1)
        ent = -(avg * mx.log(avg + 1e-12)).sum()
        head_ent = -(w * mx.log(w + 1e-12)).sum(axis=-1)
        mx.eval(avg, head_pos, head_w, ent, head_ent, w, avg_all)
        top = mx.argsort(-avg)[:8].tolist()
        avg_l = avg.tolist()
        layer = self.stack[-1] if self.stack else "?"
        tk = lambda p: self.tokens[p] if p < len(self.tokens) else "?"
        items = [{"pos": p, "token": tk(p), "weight": round(avg_l[p], 4)} for p in top]
        rec = {"heads": q.shape[0], "context": L, "top": items,
               "entropy": round(ent.item(), 4),
               "head_top": [[p, round(v, 4)] for p, v in
                            zip(head_pos.tolist(), head_w.tolist())],
               "head_entropy": [round(v, 3) for v in head_ent.tolist()]}
        if L <= 512:
            rec["dist"] = [round(v, 4) for v in avg_l]
        if L <= 160:                                       # каждая голова отдельно: (H, L)
            rec["heads_dist"] = [[round(v, 3) for v in row] for row in w.tolist()]
            rec["matrix"] = [[round(v, 3) for v in row] for row in avg_all.tolist()]
            rec["q_start"] = L - nq                        # позиция первой строки матрицы
        self.j.event("attention", step=self.step, module=layer, **rec)
        pretty = "  ".join(f"{it['token']!r}:{it['weight']}" for it in items)
        self.j.text(f"{'  ' * len(self.stack)}ВНИМАНИЕ последнего токена "
                    f"(среднее по {q.shape[0]} головам): {pretty}", echo=False)


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
                       ig_steps: int = 0, step: int = 0, n_prompt: int = None):
    """Какие токены контекста сильнее всего повлияли на выбор токена на этом шаге.

    Контекст шага k — промпт плюс k уже выбранных токенов ответа, поэтому на поздних
    шагах часть влияния приходится на собственные предыдущие токены модели."""
    n_prompt = len(ctx_ids) if n_prompt is None else n_prompt
    # Считаем без кэша генерации, а LanguageModel хранит позиции последнего вызова:
    # возвращаем их на место, чтобы анализ не влиял на сам прогон.
    saved = (getattr(lm, "_position_ids", None), getattr(lm, "_rope_deltas", None))
    t0 = time.perf_counter()
    try:
        attr, value, start = token_attribution(lm, ctx_ids, target["id"], other["id"], ig_steps)
    except Exception as e:                           # не должно ломать прогон
        journal.text(f"\n[шаг {step}: вклад токенов не посчитан: {type(e).__name__}: {e}]")
        return
    finally:
        lm._position_ids, lm._rope_deltas = saved
    method = f"integrated gradients, {ig_steps} шагов" if ig_steps else "gradient × input"
    toks = [tok.decode([i]) for i in ctx_ids]
    # для integrated gradients сумма вкладов обязана равняться изменению метрики вдоль пути
    gap = None if start is None else round(abs(sum(attr) - (value - start)), 4)
    journal.event("token_attr", step=step, method=method, target=target, other=other,
                  scores=[round(v, 4) for v in attr], tokens=toks, n_prompt=n_prompt,
                  metric=round(value, 4), completeness_gap=gap,
                  seconds=round(time.perf_counter() - t0, 2))
    journal.section(f"ЭТАП 4б. Шаг {step}: вклад токенов контекста в выбор токена")
    journal.text(f"метод: {method}; метрика — логит {target['token']!r} минус логит "
                 f"{other['token']!r} = {value:.3f}")
    if len(ctx_ids) > n_prompt:
        total = sum(abs(v) for v in attr) or 1.0
        own = sum(abs(v) for v in attr[n_prompt:]) / total
        journal.text(f"контекст: {n_prompt} токенов промпта + {len(ctx_ids) - n_prompt} уже "
                     f"сгенерированных; на сгенерированные приходится {own:.0%} вклада")
    if gap is not None:
        journal.text(f"проверка: сумма вкладов расходится с изменением метрики на {gap} "
                     f"(из {abs(value - start):.3f})" +
                     ("  — много, увеличьте --ig-steps" if gap > 0.1 * max(abs(value - start), 1e-6) else ""))
    journal.text("плюс — токен подталкивал к выбранному токену, минус — к альтернативе:")
    for p in sorted(range(len(attr)), key=lambda p: -abs(attr[p]))[:12]:
        journal.text(f"  поз {p:3d} {toks[p]!r:>16}  {attr[p]:+.4f}" +
                     ("   (ответ)" if p >= n_prompt else ""))


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
             top_p: float = 1.0, seed: int = 0, lens=None, token_attr: bool = False,
             ig_steps: int = 0) -> dict:
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
            prompt_attribution(lm, ids + generated, tok, chosen, other, journal, ig_steps,
                               step, len(ids))

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
        detail_steps: int = 1, thinking: bool = False, raw: bool = False,
        skip_layers=(), wstats: bool = False, temperature: float = 0.0, top_p: float = 1.0,
        seed: int = 0, token_attr: bool = True, ig_steps: int = 0, lens=None,
        model_path=None):
    lm = model.language_model
    n_layers = len(lm.model.layers)
    skip_layers = sorted(set(skip_layers))
    journal.event("meta", **run_meta(
        model_path, prompt=prompt, max_tokens=max_tokens, detail_steps=detail_steps,
        thinking=thinking, raw=raw, skip_layers=skip_layers, temperature=temperature,
        top_p=top_p, seed=seed, ig_steps=ig_steps,
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
                          lens=lens, token_attr=token_attr, ig_steps=ig_steps, **sampling)
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


def prompt_slug(prompt: str, n: int = 36) -> str:
    """Короткое имя папки из текста промпта."""
    s = re.sub(r"[\W_]+", "-", prompt, flags=re.UNICODE).strip("-").lower()
    return s[:n].strip("-") or "prompt"


def run_batch(model, tok, prompts: list, out_dir: Path, run_kwargs: dict, model_path=None,
              map_cfg: dict = None, build_reports: bool = True, log=print) -> Path:
    """Подробный прогон на каждый промпт — журнал каждого в своей подпапке, а отчёт один:
    report.html в папке пакета, где вверху можно переключаться между промптами и «Сводкой».

    Порядок: сначала считаются все прогоны, затем (если нужна) карта важности слоёв по тем
    же промптам, потом пишется batch.json и собирается отчёт."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, t0 = [], time.perf_counter()
    for i, prompt in enumerate(prompts, 1):
        entry = {"prompt": prompt, "dir": f"{i:02d}_{prompt_slug(prompt)}", "status": "ok"}
        journal = Journal(out_dir / entry["dir"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):    # журнал каждого прогона — в его папку
                entry["answer"] = run(model, tok, prompt, journal, model_path=model_path,
                                      **run_kwargs)
        except Exception as e:                                 # один сбой не должен губить пакет
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            journal.close()
        el = time.perf_counter() - t0
        log(f"[{i}/{len(prompts)}] {entry['status']:5s} {prompt[:60]!r}"
            + (f" -> {entry['answer'][:40]!r}" if entry["status"] == "ok" else f"  {entry['error']}")
            + f"   (прошло {el:.0f} с, осталось ~{el / i * (len(prompts) - i):.0f} с)")
        entries.append(entry)
    ok = [e for e in entries if e["status"] == "ok"]

    if map_cfg and ok:
        import layer_map as LM
        log("\nКарта важности слоёв по тем же промптам:")
        res = LM.build_map(model, tok, [e["prompt"] for e in ok], map_cfg["tokens"],
                           map_cfg["window"], run_kwargs.get("thinking", False),
                           run_kwargs.get("raw", False), log=log, fast=map_cfg["fast"])
        LM.save_json(res, out_dir / "layer_map.json")
        (out_dir / "layer_map.txt").write_text(LM.render(res), encoding="utf-8")
        for e in ok:                       # копия нужна, чтобы подпапку можно было собрать отдельно
            shutil.copy(out_dir / "layer_map.json", out_dir / e["dir"] / "layer_map.json")

    (out_dir / "batch.json").write_text(json.dumps(
        {"created": datetime.now().isoformat(timespec="seconds"), "prompts": entries},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if build_reports:
        import make_report
        log(f"\nОтчёт: {make_report.build_batch_report(out_dir)}")
    bad = len(entries) - len(ok)
    log(f"готово: {len(ok)} из {len(entries)} промптов" + (f", с ошибками: {bad}" if bad else ""))
    return out_dir


def run_layer_map(model, tok, journal: Journal, a):
    """Карта важности слоёв в той же папке: таблица в trace.log, данные в layer_map.json."""
    import layer_map as LM
    prompts = LM.load_prompts(a.map_prompts, first=a.prompt)
    journal.section("ЭТАП 7. Карта важности слоёв (каждый слой отключается по очереди)")
    result = LM.build_map(model, tok, prompts, a.map_tokens, a.map_window,
                          a.thinking, a.raw, log=journal.text, fast=a.map_fast)
    journal.text(LM.render(result))
    path = journal.dir / "layer_map.json"
    LM.save_json(result, path)
    journal.event("layer_map_saved", path=str(path.name), windows=len(result["layers"]),
                  prompts=len(result["prompts"]))
    journal.text(f"\nкарта сохранена: {path}")


def main():
    ap = argparse.ArgumentParser(description="Журнал прогона Qwen (MLX): промпт -> ответ")
    ap.add_argument("model_path")
    ap.add_argument("prompt", nargs="?", help="промпт (в пакетном режиме — необязателен)")
    ap.add_argument("--batch", nargs="?", const="", metavar="ФАЙЛ",
                    help="пакетный режим: подробный разбор каждого промпта из файла (по одному "
                         "в строке; без файла — 24 встроенных) в одном report.html с "
                         "переключателем промптов. Промпт из командной строки добавляется первым")
    ap.add_argument("--max-prompts", type=int, help="в пакетном режиме взять только первые N промптов")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--detail-steps", type=int, default=3,
                    help="для скольких первых шагов писать каждый вызов модуля")
    ap.add_argument("--skip-layers", default="",
                    help="отключить слои, нумерация с нуля: 35  или  3,35,40-43")
    ap.add_argument("--thinking", action="store_true", help="включить режим рассуждений")
    ap.add_argument("--raw", action="store_true", help="не применять шаблон чата")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 — всегда самый вероятный токен (по умолчанию); больше — случайный выбор")
    ap.add_argument("--top-p", type=float, default=1.0, help="отсечение хвоста при сэмплировании")
    ap.add_argument("--seed", type=int, default=0, help="зерно случайности (для повторяемости)")
    ap.add_argument("--weight-stats", action="store_true",
                    help="считать статистику самих весов (среднее, std, max|w|) для каждого модуля")
    ap.add_argument("--no-token-attr", action="store_true",
                    help="не считать вклад токенов контекста (экономит по обратному проходу "
                         "на каждый подробный шаг)")
    ap.add_argument("--ig-steps", type=int, default=0,
                    help="вклад токенов через integrated gradients с таким числом шагов на каждый подробный шаг "
                         "(0 — быстрый gradient × input)")
    ap.add_argument("--tuned-lens", help="файл линзы из tuned_lens.py: точнее обычного logit lens")
    ap.add_argument("--layer-map", action="store_true",
                    help="после прогона построить карту важности слоёв (несколько минут)")
    ap.add_argument("--map-prompts", help="файл с промптами для карты (по одному в строке); "
                                              "в пакетном режиме карта строится по промптам пакета")
    ap.add_argument("--map-tokens", type=int, default=8, help="сколько токенов ответа сравнивать")
    ap.add_argument("--map-window", type=int, default=1, help="сколько слоёв отключать за раз")
    ap.add_argument("--map-fast", action="store_true",
                    help="карта без полного перебора: только градиентная оценка и угловые расстояния")
    ap.add_argument("--no-report", action="store_true", help="не строить report.html")
    ap.add_argument("--out", default="trace_logs")
    a = ap.parse_args()

    if not a.prompt and a.batch is None:
        ap.error("укажите промпт или --batch")
    model, tok = load_model(a.model_path)
    skip = parse_layers(a.skip_layers, len(model.language_model.model.layers))
    lens = None
    if a.tuned_lens:
        from tuned_lens import TunedLens
        lens = TunedLens.load(a.tuned_lens)

    root = Path(a.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_kwargs = dict(max_tokens=a.max_tokens, detail_steps=a.detail_steps, thinking=a.thinking,
                      raw=a.raw, skip_layers=skip, wstats=a.weight_stats,
                      temperature=a.temperature, top_p=a.top_p, seed=a.seed,
                      token_attr=not a.no_token_attr, ig_steps=a.ig_steps, lens=lens)

    if a.batch is not None:
        import layer_map as LM
        prompts = LM.load_prompts(a.batch or None, first=a.prompt)
        if a.max_prompts:
            prompts = prompts[:a.max_prompts]
        map_cfg = ({"tokens": a.map_tokens, "window": a.map_window, "fast": a.map_fast}
                   if a.layer_map or a.map_fast else None)
        print(f"пакетный режим: {len(prompts)} промптов -> {root}")
        run_batch(model, tok, prompts, root, run_kwargs, a.model_path, map_cfg,
                  build_reports=not a.no_report)
        return

    journal = Journal(root)
    try:
        run(model, tok, a.prompt, journal, model_path=a.model_path, **run_kwargs)
        if a.layer_map or a.map_fast:
            run_layer_map(model, tok, journal, a)
        journal.text(f"\nЖурнал: {journal.dir}/trace.log  и  {journal.dir}/events.jsonl "
                     f"({journal.n} событий)")
    finally:
        journal.close()

    if not a.no_report:
        try:
            import make_report
            out = make_report.build_report(journal.dir)
            print(f"Отчёт: {out}  (откройте в браузере)")
        except ImportError:
            print("make_report.py не найден рядом — отчёт не построен")


if __name__ == "__main__":
    sys.exit(main())
