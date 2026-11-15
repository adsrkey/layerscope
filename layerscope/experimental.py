"""Экспериментальные методы — в отчёты и в пайплайн НЕ входят.

На Qwen3.8-27B они не совпали с прямыми измерениями: ранговая корреляция gradient × input
с occlusion около 0,1, знак совпадает не чаще случайного, а оценка важности слоёв по
градиенту (attribution patching) коррелирует с перебором на 0,12. Оставлены для опытов и
сравнения; опираться на их числа нельзя. Вместо них используются
attribution.occlusion_attribution (вклад токенов) и отключение слоёв (layer_map).
"""
from contextlib import contextmanager

import mlx.core as mx

from .core import attn_name, log_softmax, mlp_name

F32 = mx.float32


@contextmanager
def grad_mode(lm):
    """Режим, в котором через модель можно считать градиенты.

    В обычном режиме слои линейного внимания используют готовое Metal-ядро, у которого
    нет обратного прохода. В режиме train они считаются обычными операциями MLX —
    результат тот же, но медленнее. Dropout в модели нет, так что ответ не меняется."""
    was = getattr(lm, "training", False)
    lm.train()
    try:
        yield
    finally:
        lm.train(was)


# ───────────────── вклад токенов промпта (градиенты, экспериментально) ─────────────────
def token_attribution(lm, ids: list, target: int, other: int = None, ig_steps: int = 0):
    """Насколько каждый токен промпта повлиял на выбор target (а не other).

    ig_steps = 0  — gradient × input: один обратный проход, быстро, но это линейное
                    приближение в одной точке;
    ig_steps > 0  — integrated gradients: путь от «безликого» промпта (на каждой позиции
                    средний эмбеддинг промпта) к настоящему. Точнее — сумма вкладов равна
                    изменению метрики на этом пути, — но во столько же раз дольше.

    Нулевой эмбеддинг как начало пути не годится: модель нормирует вектор на входе
    каждого слоя, поэтому в нуле у неё разрыв и интеграл расходится.

    Возвращает (вклады по позициям, метрика, метрика в начале пути или None).
    Для integrated gradients сумма вкладов должна равняться разности двух метрик;
    если расходится заметно — шагов мало (функция вдоль пути слишком извилиста)."""
    x = mx.array([ids])
    emb = lm.model.embed_tokens(x).astype(F32)

    def metric(e):
        lg = lm(x, inputs_embeds=e).logits[0, -1].astype(F32)
        return lg[target] - lg[other] if other is not None else lg[target]

    with grad_mode(lm):
        fn = mx.value_and_grad(metric)
        if ig_steps > 0:
            base = mx.broadcast_to(emb.mean(axis=1, keepdims=True), emb.shape)
            total = mx.zeros_like(emb)
            for k in range(ig_steps):
                _, g = fn(base + (emb - base) * ((k + 0.5) / ig_steps))
                total = total + g
                mx.eval(total)
            attr = ((total / ig_steps) * (emb - base)).sum(axis=-1)[0]
            value, start = metric(emb), metric(base)
            mx.eval(attr, value, start)
            return attr.tolist(), value.item(), start.item()
        value, grad = fn(emb)
        attr = (grad * emb).sum(axis=-1)[0]
        mx.eval(attr, value)
    return attr.tolist(), value.item(), None


# ───────────────── attribution patching ─────────────────
def atp(lm, hooks, full_ids: list, start: int, answer_ids: list) -> dict:
    """Оценка: насколько изменится средний log p токенов ответа, если убрать компоненту.

    Вместо отдельного прогона на каждый слой — линейное приближение:
        Δметрики ≈ −(запись компоненты в остаточный поток) · ∂метрика/∂(эта запись)
    Градиенты по всем компонентам получаются за один обратный проход.

    В качестве метрики взят log p исходных токенов, а не KL: у KL в исходной точке
    минимум, градиент там нулевой, и линейное приближение ничего бы не показало.

    Возвращает {"layer": [...], "attn": [...], "mlp": [...], "metric": базовое значение}."""
    n_layers = len(lm.model.layers)
    a_names = [attn_name(lm, i) for i in range(n_layers)]
    m_names = [mlp_name(i) for i in range(n_layers)]
    names = a_names + m_names
    x = mx.array([full_ids])
    answer = mx.array(answer_ids)

    saved = hooks.capture, hooks.captured, dict(hooks.edit)
    hooks.capture, hooks.captured = set(names), {}
    lm(x)
    acts = {n: hooks.captured[n].astype(F32) for n in names}
    mx.eval(acts)
    hooks.capture, hooks.captured = saved[0], saved[1]

    def metric(deltas):
        for n in names:
            # сложение во float32: дальше прогон идёт в повышенной точности
            hooks.edit[n] = lambda out, d=deltas[n]: out.astype(F32) + d
        lp = log_softmax(lm(x).logits[0, start - 1:-1])
        return mx.take_along_axis(lp, answer[:, None], axis=-1).mean()

    try:
        with grad_mode(lm):
            value, grads = mx.value_and_grad(metric)({n: mx.zeros_like(acts[n]) for n in names})
            mx.eval(value, grads)
    finally:
        hooks.edit = saved[2]

    def est(write, grad):
        return -(write * grad).sum()

    out = {"attn": [est(acts[a], grads[a]) for a in a_names],
           "mlp": [est(acts[m], grads[m]) for m in m_names],
           # отключение слоя целиком убирает обе записи; градиент по выходу слоя
           # совпадает с градиентом по выходу его MLP
           "layer": [est(acts[a] + acts[m], grads[m]) for a, m in zip(a_names, m_names)]}
    mx.eval(out)
    res = {k: [v.item() for v in vs] for k, vs in out.items()}
    res["metric"] = value.item()
    return res
