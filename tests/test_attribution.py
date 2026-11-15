"""Угловое расстояние и вклад токенов подменой.

Запуск: pytest tests   или   python tests/run_all.py
"""
import mlx.core as mx
from helpers import model

from qwen_trace import attribution as A


def test_angular_distance():
    a = mx.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    r = A.angular_distances([a], [1, 2])                     # одна позиция: форма (слоёв+1, D)
    assert r["1"]["dist"] == [0.0, 0.5, 0.5] and r["1"]["best_start"] == 0
    assert r["2"]["dist"] == [0.5, 1.0]
    # две позиции усредняются: у второй блок ничего не меняет, у первой — как выше
    b = mx.stack([a, mx.broadcast_to(a[:1], a.shape)], axis=1)       # (слоёв+1, 2, D)
    r2 = A.angular_distances(iter([b]), [1])                         # итератор тоже годится
    assert r2["1"]["dist"] == [0.0, 0.25, 0.25]


def test_angular_distance_uses_all_positions():
    m, tok = model()
    st = A.residual_stack(m.language_model, tok.encode("abcdefgh"))
    assert st.shape == (9, 7, 64)                            # слоёв+1, позиций без первой, D


def test_occlusion_matches_direct_replacement():
    """Подмена токена пачками (chunk) даёт то же, что подмена по одному."""
    m, tok = model()
    lm = m.language_model
    ids = tok.encode("abcdefg hij")
    x = mx.array([ids])
    emb = lm.model.embed_tokens(x)
    base = emb.astype(mx.float32).mean(axis=1, keepdims=True).astype(emb.dtype)

    def metric(e):
        lg = lm(x, inputs_embeds=e).logits[0, -1].astype(mx.float32)
        return (lg[5] - lg[9]).item()

    full = metric(emb)
    expect = [full - metric(mx.concatenate([emb[:, :i], base, emb[:, i + 1:]], axis=1))
              for i in range(len(ids))]
    # допуск 1e-2: умножение матриц на (1, D) и на (B, D) даёт разницу порядка 4e-3 даже в float32
    for chunk in (1, 3, 8):                                         # границы пачек в разных местах
        got, value = A.occlusion_attribution(lm, ids, 5, 9, chunk=chunk)
        assert abs(value - full) < 1e-2
        assert max(abs(a - b) for a, b in zip(got, expect)) < 1e-2, chunk
    # без соперника метрика — просто логит токена
    only, _ = A.occlusion_attribution(lm, ids, 5)
    assert len(only) == len(ids)


def test_occlusion_does_not_disturb_generation_state():
    m, tok = model()
    lm = m.language_model
    ids = tok.encode("abcdef")
    lm(mx.array([ids]))
    before = (lm._position_ids, lm._rope_deltas)
    A.occlusion_attribution(lm, ids, 5, 9)
    assert lm._position_ids is before[0] and lm._rope_deltas is before[1]
