"""Данные отчёта: чтение журналов прогона и короткая сводка одного прогона."""
import json
from pathlib import Path


def build_data(run_dir) -> dict:
    run_dir = Path(run_dir)
    ev = [json.loads(l) for l in open(run_dir / "events.jsonl", encoding="utf-8") if l.strip()]

    def first(kind):
        return next((e for e in ev if e["kind"] == kind), None)

    D = {"run": run_dir.name, "model": first("model"), "prompt": first("prompt"),
         "meta": first("meta"), "timing": first("timing"),
         "tokens": first("tokens"), "embeddings": first("embeddings"),
         "weights": first("weights"), "ablation": first("ablation")}
    answers = [e for e in ev if e["kind"] == "answer"]
    D["answer"] = answers[-1] if answers else None

    D["calls"] = [{"m": e["module"], "c": e["class"], "d": e["depth"], "ms": e["ms"], "n": e["n"],
                   "o": {k: e["output"][k] for k in ("shape", "mean", "std", "last_norm")
                         if k in e.get("output", {})},
                   "w": (e.get("weights") or {}).get("wstats")}
                  for e in ev if e["kind"] == "module_call" and e["step"] == 0]

    lens, attn, resid = {}, {}, {}
    for e in ev:
        if e["kind"] == "logit_lens":
            lens.setdefault(str(e["step"]), []).append(
                {k: e[k] for k in ("layer", "layer_type", "top", "p_final", "rank_final", "entropy",
                                   "kl_final", "tuned_top", "tuned_p_final", "tuned_rank", "tuned_kl")
                 if k in e})
        elif e["kind"] == "attention":
            attn.setdefault(str(e["step"]), []).append(
                {k: e[k] for k in ("module", "heads", "context", "top", "entropy", "head_top", "dist",
                                   "heads_dist", "head_entropy", "matrix", "q_start")
                 if k in e})
        elif e["kind"] == "residual":
            resid[str(e["step"])] = {k: e[k] for k in ("norms", "rel", "cos", "emb_norms",
                                                       "final_id", "final_token")}
    for v in lens.values():
        v.sort(key=lambda r: r["layer"])
    D["lens"], D["attention"], D["resid"] = lens, attn, resid
    # вклад токенов контекста по шагам (в старых журналах он был один, без номера шага)
    D["token_attrs"] = {str(e.get("step", 0)): {k: v for k, v in e.items() if k not in ("n", "kind")}
                        for e in ev if e["kind"] == "token_attr"}
    D["lens_grid"] = first("lens_grid")
    D["dla"] = {str(e["step"]): {k: e[k] for k in ("target", "other", "cands", "components",
                                                   "logit_sum", "logit_actual", "logits_sum",
                                                   "logits_actual", "residual_gap") if k in e}
                for e in ev if e["kind"] == "dla"}

    samples = [e for e in ev if e["kind"] == "sample"]
    D["samples"] = [{k: s[k] for k in ("step", "ms", "chosen", "candidates", "entropy",
                                       "nucleus95", "top20_mass", "traced", "detail") if k in s}
                    for s in samples if s.get("run") != "Обычный прогон"]
    D["skipped"] = sorted({int(e["module"].rsplit(".", 1)[-1]) for e in ev
                           if e["kind"] == "layer_skipped"})

    lm = run_dir / "layer_map.json"
    D["layer_map"] = json.loads(lm.read_text(encoding="utf-8")) if lm.exists() else None
    for key, name in (("patching", "patching.json"), ("quant", "quant_compare.json")):
        path = run_dir / name
        D[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return D


def summarize_run(src) -> dict:
    """Короткая сводка одного прогона (src — папка прогона или уже собранные данные)."""
    D = src if isinstance(src, dict) else build_data(src)
    NL = D["model"]["layers"]
    out = {"layers": NL, "layer_types": D["model"]["layer_types"],
           "answer": D["answer"]["text"] if D["answer"] else "",
           "n_prompt_tokens": len(D["tokens"]["ids"]) if D["tokens"] else None,
           "n_tokens": len(D["samples"]),
           "prefill_ms": D["timing"]["prefill_ms"] if D["timing"] else None}
    if D["samples"]:
        s0 = D["samples"][0]
        out["first"] = {"token": s0["chosen"]["token"], "prob": s0["chosen"]["prob"],
                        "entropy": s0["entropy"]}
    # слой, с которого итоговый токен остаётся самым вероятным до конца
    settle = None
    for r in reversed(D["lens"].get("0", [])):
        if r["rank_final"] != 1:
            break
        settle = r["layer"]
    out["settle"] = settle
    ta = D["token_attrs"].get("0")
    if ta and ta["scores"]:
        i = max(range(len(ta["scores"])), key=lambda k: abs(ta["scores"][k]))
        names = ta.get("tokens") or (D["tokens"]["tokens"] if D["tokens"] else [])
        out["top_token"] = {"token": names[i] if i < len(names) else "?", "pos": i,
                            "score": ta["scores"][i]}
    dla = D["dla"].get("0")
    if dla:
        comps = dla["components"]
        top = max(comps, key=lambda c: abs(c["diff"]))
        out["top_dla"] = {"kind": top["kind"], "layer": top["layer"], "diff": top["diff"]}
        attn, mlp = [0.0] * NL, [0.0] * NL
        for c in comps:
            if c["kind"] == "attn":
                attn[c["layer"]] = abs(c["diff"])
            elif c["kind"] == "mlp":
                mlp[c["layer"]] = abs(c["diff"])
        out["dla_abs"] = {"attn": attn, "mlp": mlp}
    return out
