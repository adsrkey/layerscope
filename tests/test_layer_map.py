"""Карта важности слоёв.

Запуск: pytest tests   или   python tests/run_all.py
"""
from helpers import model, traced_run

from qwen_trace import layer_map as LM
from qwen_trace import report


def test_layer_map_and_report():
    m, tok = model()
    res = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:5], 4, 1, log=lambda *a: None, n_boot=200)
    assert len(res["layers"]) == 8 and res["version"] == 2
    for r in res["layers"]:
        assert r["kl_lo"] <= r["kl"] <= r["kl_hi"] and len(r["kl_per_prompt"]) == 5
        assert "atp" not in r                              # градиентной оценки слоёв в карте нет
    assert "atp" not in res and set(res["angular"]) >= {"1", "2", "4"}
    assert "95% интервал" in LM.render(res)
    fast = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:3], 4, 2, log=lambda *a: None, fast=True)
    assert len(fast["layers"]) == 4 and "kl" not in fast["layers"][0] and fast["angular"]
    assert "Attribution patching" not in LM.render(fast) and "Угловое расстояние" in LM.render(fast)
    d, _ = traced_run(max_tokens=3, detail_steps=1)
    LM.save_json(res, d / "layer_map.json")
    html = report.build_report(d).read_text(encoding="utf-8")
    assert "layer_map" in html and len(html) > 50_000


def test_thinking_answer_offset():
    class Tok:
        def decode(self, ids):
            return {7: "</think>", 8: "\n"}.get(ids[0], "x")
    assert LM.after_thinking(Tok(), [1, 2, 7, 3, 4]) == 3
    assert LM.after_thinking(Tok(), [1, 2, 3]) is None
