"""Экспериментальные градиентные методы (в отчёты не входят).

Запуск: pytest tests   или   python tests/run_all.py
"""
import mlx.core as mx
from helpers import PROMPT, calm_model

from qwen_trace import core
from qwen_trace import experimental as X


def test_gradient_attribution_ig_completeness_experimental():
    """Integrated gradients: сумма вкладов равна изменению метрики вдоль пути."""
    m, tok = calm_model()
    lm, ids = m.language_model, tok.encode("abcdef")
    # случайные слои линейного внимания дают очень извилистую функцию входа (у обученной
    # модели такого нет), поэтому полноту проверяем на остальной части модели
    gdn = [core.attn_name(lm, i) for i in range(8) if lm.model.layers[i].is_linear]
    with core.Hooks(lm) as h:
        h.edit = {n: (lambda out: out * 0) for n in gdn}
        attr, value, start = X.token_attribution(lm, ids, 5, 9, ig_steps=256)
    assert abs(sum(attr) - (value - start)) < 0.05 * abs(value - start) + 0.02
    g_attr, g_value, g_start = X.token_attribution(lm, ids, 5, 9)
    assert len(g_attr) == len(ids) and g_start is None


def test_gradient_matches_finite_difference():
    m, tok = calm_model()
    lm, ids = m.language_model, tok.encode("abcdef")
    x = mx.array([ids])
    emb = lm.model.embed_tokens(x)
    d = emb - emb.mean(axis=1, keepdims=True)

    def f(e):
        lg = lm(x, inputs_embeds=e).logits[0, -1]
        return lg[5] - lg[9]

    gdn = [core.attn_name(lm, i) for i in range(8) if lm.model.layers[i].is_linear]
    with core.Hooks(lm) as h, X.grad_mode(lm):
        h.edit = {n: (lambda out: out * 0) for n in gdn}       # см. тест выше
        lin = (mx.grad(f)(emb) * d).sum().item()
        fd = (f(emb + 2e-3 * d).item() - f(emb - 2e-3 * d).item()) / 4e-3
    assert abs(lin - fd) < 0.1 * abs(fd) + 0.05


# ───────────── карта слоёв ─────────────
def test_atp_matches_small_intervention():
    """Градиентная оценка совпадает с настоящим малым вмешательством."""
    m, tok = calm_model()
    lm = m.language_model
    ids = tok.encode(PROMPT)
    full, start, ans = ids + [5, 9, 11], len(ids), [5, 9, 11]
    eps, layer = 2e-3, 4
    with core.Hooks(lm) as h:
        est = X.atp(lm, h, full, start, ans)

        def metric():
            lp = core.log_softmax(lm(mx.array([full])).logits[0, start - 1:-1])
            return mx.take_along_axis(lp, mx.array(ans)[:, None], axis=-1).mean().item()

        base = metric()
        assert abs(base - est["metric"]) < 2e-2      # обычный и градиентный режим считают чуть по-разному
        for kind, name in (("mlp", core.mlp_name(layer)), ("attn", core.attn_name(lm, layer))):
            h.edit = {name: lambda out: out * (1 - eps)}      # убрать eps-долю записи
            less = metric()
            h.edit = {name: lambda out: out * (1 + eps)}
            real = (less - metric()) / 2                      # центральная разность
            h.edit = {}
            assert abs(real - eps * est[kind][layer]) < 0.2 * abs(real) + 1e-5, (kind, real)
