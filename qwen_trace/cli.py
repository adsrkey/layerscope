"""
trace — полный журнал одного прогона модели: от промпта до ответа.

Запуск:
    python -m qwen_trace trace /путь/к/папке/модели "Столица Франции — это"
    python -m qwen_trace trace /путь/к/папке/модели "..." --max-tokens 30 --detail-steps 2
    python -m qwen_trace trace /путь/к/папке/модели "..." --skip-layers 3,35,40-43
    python -m qwen_trace trace /путь/к/папке/модели "..." --layer-map      # + карта важности слоёв
    python -m qwen_trace trace /путь/к/папке/модели "..." --temperature 0.7 --top-p 0.9 --seed 1
    python -m qwen_trace trace /путь/к/папке/модели "..." --tuned-lens tuned_lens.safetensors
    python -m qwen_trace trace /путь/к/папке/модели --batch prompts/extended.txt --layer-map
        # пакетный режим: разбор КАЖДОГО промпта в одном report.html с переключателем

Результат — папка trace_logs/<дата_время>/ (в пакетном режиме — по подпапке на промпт,
плюс один report.html со сводкой и переключателем промптов и layer_map.json, общий для всех):
    trace.log       читаемый журнал всех этапов (с картой важности, если включена)
    events.jsonl    те же события построчно в JSON (первое событие — метаданные прогона)
    layer_map.json  карта важности слоёв (только с --layer-map)
    report.html     интерактивный отчёт: открывается двойным щелчком, работает без сети

Что считается для подробных шагов (--detail-steps; по умолчанию — все токены ответа,
но не больше --max-tokens):
    каждый вызов модуля, logit lens (и tuned lens, если дана), внимание по головам,
    разложение итогового логита по слоям (DLA), вклад токенов контекста (промпт и уже
    выбранные токены ответа) в выбор каждого токена.

Рассчитан на Qwen3.5/3.8 в формате MLX (mlx-vlm, model_type "qwen3_5").
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from .batch import run_batch
from .core import load_model
from .journal import Journal
from .pipeline import parse_layers, run


def run_layer_map(model, tok, journal: Journal, a):
    """Карта важности слоёв в той же папке: таблица в trace.log, данные в layer_map.json."""
    from . import layer_map as LM
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qwen-trace trace",
                                 description="Журнал прогона Qwen (MLX): промпт -> ответ")
    ap.add_argument("model_path")
    ap.add_argument("prompt", nargs="?", help="промпт (в пакетном режиме — необязателен)")
    ap.add_argument("--batch", nargs="?", const="", metavar="ФАЙЛ",
                    help="пакетный режим: подробный разбор каждого промпта из файла (по одному "
                         "в строке; без файла — 24 встроенных) в одном report.html с "
                         "переключателем промптов. Промпт из командной строки добавляется первым")
    ap.add_argument("--max-prompts", type=int, help="в пакетном режиме взять только первые N промптов")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--detail-steps", type=int, default=None,
                    help="для скольких первых шагов генерации делать подробный разбор (каждый "
                         "вызов модуля, logit lens, вклад слоёв, внимание, вклад токенов). По "
                         "умолчанию — для всех токенов ответа и токена конца, то есть до "
                         "--max-tokens; меньшее число экономит время и размер отчёта")
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
                    help="не считать вклад токенов контекста (экономит несколько прямых "
                         "проходов на каждый подробный шаг)")
    ap.add_argument("--tuned-lens", help="файл линзы из команды tuned-lens: точнее обычного logit lens")
    ap.add_argument("--layer-map", action="store_true",
                    help="после прогона построить карту важности слоёв (несколько минут)")
    ap.add_argument("--map-prompts", help="файл с промптами для карты (по одному в строке); "
                                              "в пакетном режиме карта строится по промптам пакета")
    ap.add_argument("--map-tokens", type=int, default=8, help="сколько токенов ответа сравнивать")
    ap.add_argument("--map-window", type=int, default=1, help="сколько слоёв отключать за раз")
    ap.add_argument("--map-fast", action="store_true",
                    help="карта без полного перебора слоёв: только угловое расстояние")
    ap.add_argument("--no-report", action="store_true", help="не строить report.html")
    ap.add_argument("--out", default="trace_logs")
    a = ap.parse_args(argv)

    if not a.prompt and a.batch is None:
        ap.error("укажите промпт или --batch")
    model, tok = load_model(a.model_path)
    skip = parse_layers(a.skip_layers, len(model.language_model.model.layers))
    lens = None
    if a.tuned_lens:
        from .tuned_lens import TunedLens
        lens = TunedLens.load(a.tuned_lens)

    root = Path(a.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_kwargs = dict(max_tokens=a.max_tokens, detail_steps=a.detail_steps, thinking=a.thinking,
                      raw=a.raw, skip_layers=skip, wstats=a.weight_stats,
                      temperature=a.temperature, top_p=a.top_p, seed=a.seed,
                      token_attr=not a.no_token_attr, lens=lens)

    if a.batch is not None:
        from . import layer_map as LM
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
            from . import report
            out = report.build_report(journal.dir)
            print(f"Отчёт: {out}  (откройте в браузере)")
        except ImportError:
            print("пакет report не найден — отчёт не построен")


if __name__ == "__main__":
    sys.exit(main())
