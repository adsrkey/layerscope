#!/usr/bin/env python3
"""
layer_map.py — карта важности слоёв: что будет, если отключать слои по одному.

Обычно запускается внутри trace_qwen.py:
    python trace_qwen.py /путь/к/модели "промпт" --layer-map
Тогда layer_map.json ложится в ту же папку, что trace.log, а в trace.log попадает таблица.

Отдельно (карта для уже сделанного прогона — файл добавится в его папку):
    python layer_map.py /путь/к/модели --run-dir trace_logs/20261115_184500
    python layer_map.py /путь/к/модели --prompts prompts/extended.txt --tokens 12 --window 4
    python layer_map.py /путь/к/модели --fast          # без перебора, только угловое расстояние

Два измерения:

1. Отключение (перебор). Модель отвечает как обычно (N токенов), затем тот же текст
   прогоняется с одним отключённым слоем, и на каждой позиции ответа сравнивается:
       KL        насколько изменилось распределение вероятностей (0 — слой не важен)
       сломано   доля токенов ответа, где лучший кандидат стал другим
       p(исх.)   средняя вероятность исходного токена после отключения
       Δlog p    изменение логарифма вероятности исходного токена
   Для KL даётся 95% доверительный интервал (бутстреп по промптам): если интервалы двух
   слоёв пересекаются, по этим данным нельзя сказать, какой из них важнее.

2. Угловое расстояние (Gromov et al., 2024). Насколько блок из n слоёв поворачивает
   вектор остаточного потока (усреднено по всем позициям, кроме первой). Отвечает на другой
   вопрос — какой блок удалять ЦЕЛИКОМ: важности отдельных слоёв не складываются, потому
   что слои компенсируют друг друга.

Градиентную оценку важности слоёв (attribution patching) я убрал: на Qwen3.8-27B она
коррелировала с перебором на 0,12, то есть не давала надёжной подсказки. Функция atp
осталась в attribution.py как экспериментальная.

С --thinking сравниваются токены ответа ПОСЛЕ блока рассуждений, а не его начало.

Формат результата — один файл layer_map.json (он же читается отчётом report.html).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution import angular_distances, residual_stack  # noqa: E402
from qcore import (Hooks, bootstrap_ci, eos_ids, format_prompt, kl_div, layer_name,  # noqa: E402
                   load_model, log_softmax)

DEFAULT_PROMPTS = [
    # факты
    "Назови столицу Франции одним словом.",
    "What is the capital of Japan? Answer in one word.",
    "Кто написал роман «Война и мир»?",
    "Which planet is closest to the Sun? One word.",
    "В каком году человек впервые полетел в космос? Ответь только числом.",
    "What is the chemical symbol for gold?",
    # арифметика и последовательности
    "Сколько будет 17 умножить на 6? Ответь только числом.",
    "What is 144 divided by 12? Answer with a number only.",
    "Продолжи последовательность: 2, 4, 8, 16,",
    "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    # код
    "Напиши на Python функцию, которая возвращает сумму списка чисел.",
    "Write a one-line Python expression that reverses a string s.",
    "What does the SQL keyword DISTINCT do? One sentence.",
    # перевод
    "Переведи на английский: «Сегодня хорошая погода».",
    "Translate to Russian: \"The cat sleeps on the sofa.\"",
    "Переведи на немецкий слово «книга».",
    # логика и здравый смысл
    "Все птицы умеют летать. Пингвин — птица. Умеет ли пингвин летать? Ответь кратко.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Yes or no.",
    "Что тяжелее: килограмм ваты или килограмм железа? Ответь кратко.",
    "Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? One word.",
    # язык
    "Исправь ошибку: «Я пошол в магазин».",
    "What is the plural of \"mouse\"? One word.",
    "Назови антоним слова «горячий».",
    "Is the sentence \"I love this movie\" positive or negative? One word.",
]


def load_prompts(path=None, first=None) -> list:
    """Промпты для карты: свои из файла или встроенные; промпт текущего прогона — первым."""
    prompts = list(DEFAULT_PROMPTS)
    if path:
        prompts = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.startswith("#")]
    if first and first not in prompts:
        prompts = [first] + prompts
    return prompts


def greedy_answer(lm, ids: list, n_tokens: int, eos: set) -> list:
    """Обычный ответ модели: n_tokens самых вероятных токенов подряд."""
    cache = lm.make_cache()
    inputs = mx.array([ids])
    out = []
    for _ in range(n_tokens):
        logits = lm(inputs, cache=cache).logits[0, -1]
        nxt = mx.argmax(logits).item()
        if nxt in eos:
            break
        out.append(nxt)
        inputs = mx.array([[nxt]])
    return out


def after_thinking(tok, gen: list):
    """Индекс первого токена ответа после блока рассуждений (None — блок не закрыт)."""
    for k in range(len(gen) - 1, -1, -1):
        if tok.decode([gen[k]]).strip() == "</think>":
            return k + 1
    return None


def answer_logp(lm, full_ids: list, start: int) -> mx.array:
    """Логарифмы вероятностей на позициях ответа: (длина ответа, словарь)."""
    lp = log_softmax(lm(mx.array([full_ids])).logits[0, start - 1:-1])
    mx.eval(lp)
    return lp


def bar(value: float, vmax: float, width: int = 30) -> str:
    n = 0 if vmax <= 0 else round(width * min(max(value, 0) / vmax, 1.0))
    return "█" * n + "·" * (width - n)


def make_cases(model, tok, prompts, n_tokens, thinking, raw, think_max, log) -> list:
    """Обычные ответы модели и эталонные распределения на позициях ответа."""
    lm, eos, cases = model.language_model, eos_ids(model), []
    log("Обычные ответы модели:")
    for prompt in prompts:
        ids = tok.encode(format_prompt(tok, prompt, thinking, raw))
        if thinking and not raw:
            gen = greedy_answer(lm, ids, think_max + n_tokens, eos)
            k = after_thinking(tok, gen)
            if k is None:
                log(f"  [пропущен: рассуждение не закончилось за {think_max} токенов] {prompt!r}")
                continue
            ids, gen = ids + gen[:k], gen[k:k + n_tokens]
        else:
            gen = greedy_answer(lm, ids, n_tokens, eos)
        if not gen:
            log(f"  [пропущен, пустой ответ] {prompt!r}")
            continue
        full = ids + gen
        cases.append({"prompt": prompt, "start": len(ids), "full": full,
                      "base": answer_logp(lm, full, len(ids)),
                      "answer_ids": gen, "answer": tok.decode(gen)})
        log(f"  {prompt!r}\n      -> {cases[-1]['answer']!r}")
    if not cases:
        raise SystemExit("Нет ни одного промпта с непустым ответом")
    return cases


def ablate(lm, hooks, cases, layers) -> dict:
    """Метрики по каждому промпту при отключении набора слоёв."""
    hooks.skip = {layer_name(i) for i in layers}
    kls, changed, kept, dlp = [], [], [], []
    try:
        for case in cases:
            lp_b = case["base"]
            lp_a = answer_logp(lm, case["full"], case["start"])
            orig = mx.array(case["answer_ids"])[:, None]
            o_a = mx.take_along_axis(lp_a, orig, axis=-1)[:, 0]
            o_b = mx.take_along_axis(lp_b, orig, axis=-1)[:, 0]
            kls.append(kl_div(lp_b, lp_a).mean().item())
            changed.append((mx.argmax(lp_b, axis=-1) != mx.argmax(lp_a, axis=-1))
                           .astype(mx.float32).mean().item())
            kept.append(mx.exp(o_a).mean().item())
            dlp.append((o_a - o_b).mean().item())
    finally:
        hooks.skip = set()
    return {"kl": kls, "broken": changed, "p_orig": kept, "dlogp": dlp}


def build_map(model, tok, prompts, n_tokens=8, window=1, thinking=False, raw=False,
              log=print, fast=False, think_max=512, n_boot=1000) -> dict:
    lm = model.language_model
    n_layers = len(lm.model.layers)
    mean = lambda xs: sum(xs) / len(xs)
    cases = make_cases(model, tok, prompts, n_tokens, thinking, raw, think_max, log)
    starts = list(range(0, n_layers - window + 1, 1 if window == 1 else window))
    kinds = ["L" if l.is_linear else "A" for l in lm.model.layers]
    rows = []
    for start in starts:
        layers = list(range(start, start + window))
        rows.append({"layers": f"{start}" if window == 1 else f"{start}-{layers[-1]}",
                     "first_layer": start, "last_layer": layers[-1],
                     "type": "".join(kinds[i] for i in layers)})

    with Hooks(lm) as hooks:
        hooks.keep_cache = False                 # кэша генерации тут нет: слой можно не считать

        # 1. Угловые расстояния между слоями: один прямой проход на промпт, все позиции
        log(f"\nУгловые расстояния: {len(cases)} промптов")
        sizes = sorted({n for n in (1, 2, 4, 8, 16, window) if n <= max(1, n_layers // 2)})
        angular = angular_distances((residual_stack(lm, c["full"]) for c in cases), sizes)

        # 2. Полный перебор: отключаем слои по одному (или блоками)
        if not fast:
            log(f"\nОтключение слоёв: {len(starts)} вариантов x {len(cases)} промптов")
            t0 = time.perf_counter()
            for done, r in enumerate(rows, 1):
                m = ablate(lm, hooks, cases, range(r["first_layer"], r["last_layer"] + 1))
                lo, hi = bootstrap_ci(m["kl"], n_boot)
                r.update(kl=round(mean(m["kl"]), 4), kl_lo=round(lo, 4), kl_hi=round(hi, 4),
                         broken=round(mean(m["broken"]), 3), p_orig=round(mean(m["p_orig"]), 3),
                         dlogp=round(mean(m["dlogp"]), 4),
                         kl_per_prompt=[round(x, 4) for x in m["kl"]],
                         broken_per_prompt=[round(x, 3) for x in m["broken"]])
                if done % 8 == 0 or done == len(rows):
                    el = time.perf_counter() - t0
                    log(f"  {done}/{len(rows)}  прошло {el:.0f} с, "
                        f"осталось ~{el / done * (len(rows) - done):.0f} с")

    return {"version": 2, "window": window, "tokens": n_tokens, "thinking": thinking,
            "fast": fast, "n_boot": n_boot,
            "prompts": [{"prompt": c["prompt"], "answer": c["answer"]} for c in cases],
            "layers": rows, "angular": angular}


def render(res: dict) -> str:
    rows, window, fast = res["layers"], res["window"], res["fast"]
    n = min(8, len(rows))
    out = ["", "=" * 78, "КАРТА ВАЖНОСТИ СЛОЁВ", "=" * 78,
           f"промптов: {len(res['prompts'])}   отключается за раз: {window} сл.   "
           f"тип: L — линейное внимание, A — полное"]
    if len(res["prompts"]) < 20:
        out.append("ВНИМАНИЕ: промптов мало — интервалы широкие, выводы предварительные.")
    out.append("")
    if not fast:
        vmax = max(r["kl"] for r in rows) or 1.0
        out.append(f"{'слой':>7} {'тип':<5} {'KL':>8} {'95% интервал':>19} {'сломано':>8} "
                   f"{'p(исх.)':>8} {'Δlog p':>8}  важность")
        for r in rows:
            out.append(f"{r['layers']:>7} {r['type']:<5} {r['kl']:>8.4f} "
                       f"[{r['kl_lo']:>7.4f}, {r['kl_hi']:>7.4f}] {r['broken']:>7.0%} "
                       f"{r['p_orig']:>8.3f} {r['dlogp']:>8.3f}  {bar(r['kl'], vmax)}")
        line = lambda r: (f"  слой {r['layers']:>5} ({r['type']})  KL {r['kl']:.4f} "
                          f"[{r['kl_lo']:.4f}, {r['kl_hi']:.4f}]  сломано {r['broken']:.0%}")
        ranked = sorted(rows, key=lambda r: -r["kl"])
        out += ["", "Самые важные (без них ответ ломается сильнее всего):"]
        out += [line(r) for r in ranked[:n]]
        # наименее важные — по ВЕРХНЕЙ границе интервала: слой с малым средним, но широким
        # интервалом может оказаться важным на других промптах
        safe = sorted(rows, key=lambda r: r["kl_hi"])
        out += ["", "Наименее важные (по верхней границе интервала; кандидаты на удаление):"]
        out += [line(r) for r in safe[:n]]
        if n > 1 and safe[0]["kl_hi"] >= safe[n - 1]["kl_lo"]:
            out.append("  интервалы этих слоёв пересекаются: порядок внутри списка не значим")
        if window == 1:
            for kind, name in (("L", "линейное внимание"), ("A", "полное внимание")):
                sel = [r["kl"] for r in rows if r["type"] == kind]
                if sel:
                    out.append(f"\nсредний KL, {name}: {sum(sel) / len(sel):.4f}  (слоёв: {len(sel)})")
    if fast:
        out.append("Быстрый режим: перебор слоёв не запускался, есть только угловое расстояние.")
    ang = res.get("angular")
    if ang:
        out += ["", "-" * 78, "Угловое расстояние: какой блок слоёв меньше всего меняет вектор",
                "(0 — блок ничего не делает, 1 — разворачивает вектор в обратную сторону; "
                "усреднено по всем позициям, кроме первой)"]
        for k, v in ang.items():
            s = v["best_start"]
            out.append(f"  блок из {int(k):>2} сл.: слои {s}–{s + int(k) - 1}, расстояние {v['best']:.4f}")
        out.append("Удаление блока проверяйте отдельно (--skip-layers) и по возможности "
                   "дообучайте модель после удаления: метрика показывает кандидата, а не гарантию.")
    return "\n".join(out)


def save_json(res: dict, path):
    Path(path).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Карта важности слоёв Qwen (MLX)")
    ap.add_argument("model_path")
    ap.add_argument("--run-dir", help="папка прогона trace_qwen.py: карта добавится в неё "
                                      "(и report.html будет пересобран)")
    ap.add_argument("--prompts", help="файл с промптами, один в строке (по умолчанию "
                                      f"{len(DEFAULT_PROMPTS)} встроенных; есть prompts/extended.txt)")
    ap.add_argument("--tokens", type=int, default=8, help="сколько токенов ответа сравнивать")
    ap.add_argument("--window", type=int, default=1, help="сколько соседних слоёв отключать за раз")
    ap.add_argument("--fast", action="store_true",
                    help="без полного перебора слоёв: только угловое расстояние")
    ap.add_argument("--thinking", action="store_true",
                    help="режим рассуждений: сравниваются токены после блока рассуждений")
    ap.add_argument("--think-max", type=int, default=512, help="предел длины рассуждения в токенах")
    ap.add_argument("--raw", action="store_true", help="не применять шаблон чата")
    ap.add_argument("--out", default="layer_map")
    a = ap.parse_args()

    run_dir = Path(a.run_dir) if a.run_dir else None
    first = None
    if run_dir:
        for line in open(run_dir / "events.jsonl", encoding="utf-8"):
            e = json.loads(line)
            if e["kind"] == "prompt":
                first = e["prompt"]
                break
    prompts = load_prompts(a.prompts, first)

    model, tok = load_model(a.model_path)
    res = build_map(model, tok, prompts, a.tokens, a.window, a.thinking, a.raw,
                    fast=a.fast, think_max=a.think_max)
    text = render(res)
    print(text)

    out = run_dir or Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / "layer_map.json")
    if run_dir:
        with open(out / "trace.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")
    print(f"\nСохранено: {out / 'layer_map.json'}")
    if run_dir:
        try:
            import make_report
            print("Отчёт пересобран:", make_report.build_report(run_dir))
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())
