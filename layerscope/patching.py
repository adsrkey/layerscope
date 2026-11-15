"""
patch — activation patching: где в модели хранится то, что отличает два промпта.

Отключение слоя отвечает на вопрос «нужен ли слой вообще». Патчинг отвечает на более
точный: «какой слой и на какой позиции переносит именно эту информацию».

Берутся два промпта, отличающиеся одной деталью:
    чистый      "Столица Франции —"   -> " Париж"
    испорченный "Столица Италии —"    -> " Рим"
Модель запускается на испорченном промпте, но выход одного слоя на одной позиции
подменяется выходом из чистого прогона. Если после этого модель снова отвечает «Париж»,
значит, именно там проходит информация о стране.

    python -m layerscope patch /путь/к/модели --clean "Столица Франции —" --corrupt "Столица Италии —"
    python -m layerscope patch /путь/к/модели --clean "..." --corrupt "..." --component all
    python -m layerscope patch /путь/к/модели --clean "..." --corrupt "..." --run-dir trace_logs/<папка>

Оценка клетки: 0 — подмена ничего не изменила, 1 — полностью вернула чистый ответ.
Промпты лучше подбирать одинаковой длины в токенах: позиции сопоставляются с конца.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from .core import Hooks, attn_name, format_prompt, layer_name, load_model, mlp_name

KINDS = {"resid": "выход слоя (остаточный поток)", "attn": "выход внимания", "mlp": "выход MLP"}


def component_names(lm, kind: str) -> list:
    n = len(lm.model.layers)
    if kind == "resid":
        return [layer_name(i) for i in range(n)]
    if kind == "attn":
        return [attn_name(lm, i) for i in range(n)]
    return [mlp_name(i) for i in range(n)]


def run_patching(model, tok, clean: str, corrupt: str, kinds=("resid",), answer: str = None,
                 wrong: str = None, thinking=False, raw=False, max_positions: int = 64,
                 log=print) -> dict:
    lm = model.language_model
    ids_c = tok.encode(format_prompt(tok, clean, thinking, raw))
    ids_x = tok.encode(format_prompt(tok, corrupt, thinking, raw))
    if len(ids_c) != len(ids_x):
        log(f"ВНИМАНИЕ: промпты разной длины ({len(ids_c)} и {len(ids_x)} токенов) — позиции "
            f"сопоставлены с конца; для чистого результата подберите промпты равной длины.")
    n_pos = min(len(ids_c), len(ids_x), max_positions)
    x_c, x_x = mx.array([ids_c]), mx.array([ids_x])

    def first_id(text):
        ids = tok.encode(text)
        if not ids:
            raise SystemExit(f"Не удалось токенизировать {text!r}")
        return ids[0]

    with Hooks(lm) as hooks:
        hooks.keep_cache = False
        names = {k: component_names(lm, k) for k in kinds}
        hooks.capture = {n for ns in names.values() for n in ns}
        lg_c = lm(x_c).logits[0, -1].astype(mx.float32)
        acts = dict(hooks.captured)
        mx.eval(lg_c, acts)
        hooks.capture, hooks.captured = set(), {}
        lg_x = lm(x_x).logits[0, -1].astype(mx.float32)
        mx.eval(lg_x)

        a_id = first_id(answer) if answer else int(mx.argmax(lg_c).item())
        w_id = first_id(wrong) if wrong else int(mx.argmax(lg_x).item())
        if a_id == w_id:
            raise SystemExit("На оба промпта модель отвечает одним и тем же токеном "
                             f"({tok.decode([a_id])!r}) — сравнивать нечего. Подберите другую "
                             "пару или задайте --answer и --wrong.")
        ld = lambda lg: (lg[a_id] - lg[w_id]).item()
        ld_c, ld_x = ld(lg_c), ld(lg_x)
        span = ld_c - ld_x
        log(f"ответ на чистый промпт: {tok.decode([a_id])!r}, на испорченный: {tok.decode([w_id])!r}")
        log(f"разность логитов: чистый {ld_c:+.3f}, испорченный {ld_x:+.3f}")
        if abs(span) < 1e-6:
            raise SystemExit("Разность логитов на двух промптах одинакова — патчинг ничего не покажет.")

        res, t0 = {}, time.perf_counter()
        total = sum(len(v) for v in names.values()) * n_pos
        done = 0
        for kind, ns in names.items():
            grid = []
            for name in ns:
                row = []
                for back in range(n_pos, 0, -1):                  # позиции с конца
                    pc, px = len(ids_c) - back, len(ids_x) - back
                    src = acts[name]

                    def patch(out, src=src, pc=pc, px=px):
                        return mx.concatenate([out[:, :px], src[:, pc:pc + 1].astype(out.dtype),
                                               out[:, px + 1:]], axis=1)

                    hooks.edit = {name: patch}
                    lg = lm(x_x).logits[0, -1].astype(mx.float32)
                    row.append(round((ld(lg) - ld_x) / span, 4))
                    done += 1
                grid.append(row)
                if done % max(1, total // 10) < n_pos:
                    el = time.perf_counter() - t0
                    log(f"  {done}/{total}  прошло {el:.0f} с, осталось ~{el / done * (total - done):.0f} с")
            res[kind] = grid
        hooks.edit = {}

    pos_x = list(range(len(ids_x) - n_pos, len(ids_x)))
    return {"version": 1,
            "clean": {"prompt": clean, "answer": tok.decode([a_id]), "answer_id": a_id},
            "corrupt": {"prompt": corrupt, "answer": tok.decode([w_id]), "answer_id": w_id},
            "ld_clean": round(ld_c, 4), "ld_corrupt": round(ld_x, 4),
            "positions": pos_x,
            "tokens_corrupt": [tok.decode([ids_x[p]]) for p in pos_x],
            "tokens_clean": [tok.decode([ids_c[len(ids_c) - n_pos + k]]) for k in range(n_pos)],
            "layer_types": "".join("L" if l.is_linear else "A" for l in lm.model.layers),
            "components": res}


def render(res: dict) -> str:
    shades = " ░▒▓█"
    out = ["", "=" * 78, "ACTIVATION PATCHING", "=" * 78,
           f"чистый:      {res['clean']['prompt']!r} -> {res['clean']['answer']!r}",
           f"испорченный: {res['corrupt']['prompt']!r} -> {res['corrupt']['answer']!r}",
           "клетка — насколько подмена вернула чистый ответ (0 — никак, 1 — полностью)", ""]
    toks = res["tokens_corrupt"]
    for kind, grid in res["components"].items():
        out += [f"--- {KINDS[kind]} ---",
                f"{'слой':>5} {'тип':<4} {'макс.':>7}  где (токен испорченного промпта)   карта по позициям"]
        for i, row in enumerate(grid):
            k = max(range(len(row)), key=lambda j: abs(row[j]))
            cells = "".join(shades[min(4, max(0, round(abs(v) * 4)))] for v in row)
            out.append(f"{i:>5} {res['layer_types'][i]:<4} {row[k]:>+7.2f}  "
                       f"поз {res['positions'][k]:>3} {toks[k]!r:<22} |{cells}|")
        flat = sorted(((abs(v), i, j) for i, row in enumerate(grid) for j, v in enumerate(row)),
                      reverse=True)[:5]
        out.append("сильнее всего: " + ";  ".join(
            f"слой {i}, поз {res['positions'][j]} {toks[j]!r} ({grid[i][j]:+.2f})" for _, i, j in flat))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="layerscope patch", description="Activation patching для Qwen (MLX)")
    ap.add_argument("model_path")
    ap.add_argument("--clean", required=True, help="промпт с правильным ответом")
    ap.add_argument("--corrupt", required=True, help="тот же промпт с изменённой деталью")
    ap.add_argument("--answer", help="токен правильного ответа (по умолчанию — что ответит модель)")
    ap.add_argument("--wrong", help="токен ответа на испорченный промпт (по умолчанию — что ответит модель)")
    ap.add_argument("--component", default="resid", choices=["resid", "attn", "mlp", "all"],
                    help="что подменять: выход слоя, только внимание, только MLP или всё")
    ap.add_argument("--max-positions", type=int, default=64, help="сколько последних позиций перебирать")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--raw", action="store_true", help="не применять шаблон чата")
    ap.add_argument("--run-dir", help="папка прогона trace: результат добавится в неё и в отчёт")
    ap.add_argument("--out", default="patching")
    a = ap.parse_args(argv)

    model, tok = load_model(a.model_path)
    kinds = ("resid", "attn", "mlp") if a.component == "all" else (a.component,)
    res = run_patching(model, tok, a.clean, a.corrupt, kinds, a.answer, a.wrong,
                       a.thinking, a.raw, a.max_positions)
    text = render(res)
    print(text)
    out = Path(a.run_dir) if a.run_dir else Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "patching.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"Сохранено: {out / 'patching.json'}")
    if a.run_dir:
        with open(out / "trace.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        try:
            from . import report
            print("Отчёт пересобран:", report.build_report(out))
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())
