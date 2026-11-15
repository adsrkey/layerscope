"""Отчёт: чтение журналов и выполнение скрипта страницы в Node.

Запуск: pytest tests   или   python tests/run_all.py
"""
import json
import shutil
import subprocess

from helpers import ROOT, make_batch, traced_run

from layerscope import report


def test_report_old_single_token_attr_format():
    """Журналы старого формата (один token_attr без шага и токенов) по-прежнему читаются."""
    d, ev = traced_run(max_tokens=2, detail_steps=1)
    lines = [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]
    with open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for e in lines:
            if e["kind"] == "token_attr":
                e = {k: v for k, v in e.items() if k not in ("step", "tokens", "n_prompt")}
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    assert report.build_data(d)["token_attrs"]["0"]["scores"]


def test_report_scripts_run_in_node():
    """Скрипт страницы выполняется: сводка и каждый промпт строятся без ошибок и NaN.
    Нужен node; без него тест ничего не проверяет."""
    node = shutil.which("node")
    if not node:
        print("node не найден — проверка скрипта страницы пропущена")
        return
    d, _ = traced_run(max_tokens=3, detail_steps=2)
    single = report.build_report(d)
    for path in (single, make_batch() / "report.html"):
        r = subprocess.run([node, str(ROOT / "tests" / "check_report.js"), str(path)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
