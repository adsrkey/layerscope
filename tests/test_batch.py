"""Пакетный режим: один отчёт на все промпты.

Запуск: pytest tests   или   python tests/run_all.py
"""
import json

from helpers import make_batch, of, page_blocks

from layerscope import report
from layerscope.batch import prompt_slug


# ───────────── пакетный режим ─────────────
def test_prompt_slug():
    assert prompt_slug("Назови столицу Франции одним словом.") == "назови-столицу-франции-одним-словом"
    assert "/" not in prompt_slug("a/b: c?") and prompt_slug("???") == "prompt"
    assert len(prompt_slug("x" * 200)) <= 36


def test_batch_mode_one_report_with_prompt_switcher():
    """Один report.html на весь пакет; сбой одного промпта не губит остальных."""
    out = make_batch()
    b = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    assert [e["status"] for e in b["prompts"]] == ["ok", "error", "ok"]
    assert "boom" in b["prompts"][1]["error"]
    good = [e for e in b["prompts"] if e["status"] == "ok"]
    for e in good:                                   # журнал каждого промпта — в своей подпапке
        d = out / e["dir"]
        assert all((d / n).exists() for n in ("events.jsonl", "trace.log", "layer_map.json"))
        assert of([json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")], "dla")
        assert not (d / "report.html").exists()      # отдельных отчётов на промпт нет
    html = (out / "report.html").read_text(encoding="utf-8")
    assert not (out / "index.html").exists()
    blocks = page_blocks(html)
    assert set(blocks) == {"batch", "d-0", "d-2"}    # у упавшего промпта данных нет
    batch = blocks["batch"]
    assert [e["status"] for e in batch["entries"]] == ["ok", "error", "ok"]
    assert batch["layer_map"] and blocks["d-0"]["layer_map"] is None   # карта — одна на всех
    assert blocks["d-0"]["prompt"]["prompt"] == "What is 2+2?"
    assert blocks["d-2"]["prompt"]["prompt"] == "Name a color."
    ok = [e for e in batch["entries"] if e["status"] == "ok"]
    assert all("first" in e and len(e["dla_abs"]["attn"]) == 8 for e in ok)
    assert all("map_top" not in e for e in ok)               # эвристики «особенно важный слой» нет
    assert report.build_batch_report(out).exists()      # пересборка из папки, без модели
