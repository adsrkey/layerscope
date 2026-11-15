"""
report — собирает report.html из папки прогона (команда trace).

    python -m layerscope report trace_logs/20261115_184500

Читает events.jsonl и (если есть) layer_map.json, patching.json, quant_compare.json,
пишет report.html рядом.

Для папки пакета (trace --batch) собирает ОДИН report.html на все промпты:
вверху переключатель между промптами и «Сводкой».
Файл самодостаточный: данные и код внутри, сеть не нужна, открывается двойным щелчком.
Модель для этого не нужна: отчёт можно пересобрать в любой момент.

Устройство: data.py читает журналы, build.py вставляет данные JSON-блоками в template.html
(вся страница: HTML, CSS и JS).
"""
from .build import build_batch_report, build_report
from .data import build_data, summarize_run

__all__ = ["build_batch_report", "build_report", "build_data", "summarize_run"]
