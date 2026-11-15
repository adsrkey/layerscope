"""Ядро: перехват модулей (Hooks) и числовые утилиты.

Запуск: pytest tests   или   python tests/run_all.py
"""
import mlx.core as mx
import mlx.nn as nn

from qwen_trace import core

from helpers import PROMPT, model


# ───────────── ядро ─────────────
def test_hooks_restore_classes():
    m, _ = model()
    before = {c: c.__dict__.get("__call__") for c in (nn.Linear, nn.RMSNorm, nn.Embedding)}
    with core.Hooks(m.language_model):
        assert nn.Linear.__dict__["__call__"] is not before[nn.Linear]
    for c, own in before.items():
        assert c.__dict__.get("__call__") is own


def test_hooks_do_not_change_output():
    m, tok = model()
    lm, x = m.language_model, mx.array([tok.encode(PROMPT)])
    base = lm(x).logits
    with core.Hooks(lm) as h:
        h.capture = {core.layer_name(3)}
        same = lm(x).logits
        assert core.layer_name(3) in h.captured
        h.skip = {core.layer_name(3)}
        diff = lm(x).logits
    assert mx.abs(base - same).max().item() == 0
    assert mx.abs(base - diff).max().item() > 1e-3
    assert mx.abs(base - lm(x).logits).max().item() == 0       # после выхода всё как было


def test_skip_equals_removing_layer_writes():
    """Отключение слоя = вычесть из потока записи его внимания и MLP."""
    m, tok = model()
    lm, x = m.language_model, mx.array([tok.encode(PROMPT)])
    with core.Hooks(lm) as h:
        h.keep_cache = False
        h.skip = {core.layer_name(7)}
        skipped = lm(x).logits
        h.skip = set()
        h.capture = {core.layer_name(6)}
        lm(x)
        expect = core.lens_logits(lm, h.captured[core.layer_name(6)])
    assert mx.abs(skipped - expect).max().item() < 1e-4


def test_log_softmax_and_kl():
    lg = mx.array([[1.0, 2.0, 3.0], [0.0, 0.0, 50.0]])
    lp = core.log_softmax(lg)
    assert abs(mx.exp(lp).sum(axis=-1)[0].item() - 1) < 1e-6
    assert core.kl_div(lp, lp).abs().max().item() == 0
    assert core.kl_div(lp[0], lp[1]).item() > 0


def test_pick_token():
    lg = mx.array([0.0, 5.0, 1.0])
    assert core.pick_token(lg) == 1
    mx.random.seed(1)
    a = [core.pick_token(lg, 1.5, 0.9) for _ in range(20)]
    mx.random.seed(1)
    assert a == [core.pick_token(lg, 1.5, 0.9) for _ in range(20)]
    assert all(core.pick_token(lg, 1.0, 0.01) == 1 for _ in range(10))    # узкий top_p


def test_bootstrap_ci():
    lo, hi = core.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo < 3.0 < hi and lo >= 1.0 and hi <= 5.0
    assert core.bootstrap_ci([2.0]) == (2.0, 2.0)
