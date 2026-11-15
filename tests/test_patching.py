"""Activation patching.

Запуск: pytest tests   или   python tests/run_all.py
"""
from helpers import model

from layerscope import patching


# ───────────── патчинг ─────────────
def test_patching_bounds():
    m, tok = model()
    res = patching.run_patching(m, tok, "The cat sat on", "The dog sat on",
                                ("resid", "attn", "mlp"), answer="A", wrong="B",
                                log=lambda *a: None)
    grid = res["components"]["resid"]
    assert len(grid) == 8 and len(grid[0]) == len(res["positions"])
    # подмена выхода последнего слоя на последней позиции полностью возвращает чистый ответ
    assert abs(grid[-1][-1] - 1.0) < 1e-3
    # позиции до первого отличия промптов не несут различий: подмена ничего не меняет
    assert all(abs(row[0]) < 1e-3 for row in grid)
    assert "ACTIVATION PATCHING" in patching.render(res)
