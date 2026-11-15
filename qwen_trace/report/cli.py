"""Команда report: пересобрать report.html из папки прогона или пакета, без модели."""
import argparse
from pathlib import Path

from .build import build_batch_report, build_report


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qwen-trace report",
        description="Пересобрать report.html из папки прогона (trace_logs/<дата_время>) "
                    "или из папки пакета (с batch.json)")
    ap.add_argument("run_dir", help="папка прогона или пакета")
    a = ap.parse_args(argv)
    target = Path(a.run_dir)
    if (target / "batch.json").exists():          # пакет: один отчёт на все промпты
        print(build_batch_report(target))
    else:
        print(build_report(target))
