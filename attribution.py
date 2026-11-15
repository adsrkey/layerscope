#!/usr/bin/env python3
"""
attribution.py — причинные и градиентные методы поверх qcore.Hooks.

    direct_logit_attribution   вклад каждого слоя (внимание и MLP отдельно) в логиты любых токенов
    occlusion_attribution      вклад каждого токена контекста: токен по очереди заменяется
                               «нейтральным», и измеряется, насколько изменился выбор
                               (прямое измерение, без градиентов; работает для всех слоёв)
    angular_distances          угловое расстояние между входом и выходом блока слоёв
                               (метрика выбора блока для удаления, Gromov et al., 2024)

Экспериментальные, в отчёты не используются: token_attribution (gradient × input,
integrated gradients) и atp (attribution patching). На Qwen3.8-27B они не совпали с прямыми
измерениями: ранговая корреляция gradient × input с occlusion около 0,1, знак совпадает
не чаще случайного, оценка слоёв по градиенту коррелирует с перебором на 0,12.
"""
import math

import mlx.core as mx

from qcore import attn_name, grad_mode, layer_name, log_softmax, mlp_name, unembed

F32 = mx.float32


# ───────────────── разложение логита по компонентам ─────────────────
def direct_logit_attribution(lm, parts: list, h_final: mx.array, cand_ids: list):
    """Точное разложение логитов нескольких токенов на вклады компонент.

    Выход модели — сумма записей в остаточный поток: эмбеддинг + выход внимания и MLP
    каждого слоя. Финальная норма (RMSNorm) делит эту сумму на её же длину, поэтому при
    «замороженной» длине логит линеен по слагаемым:  logit = Σ W_U · (вклад / rms · вес).
    Раскладывается логит ТОЛЬКО последней позиции: именно по нему выбирается следующий
    токен. Токены берутся любые (обычно — пять лучших кандидатов), так что вклады
    разности логитов любой пары получаются вычитанием столбцов.

    parts     [(метка, вектор последней позиции)], в порядке вычисления
    h_final   вектор последней позиции после последнего слоя (до финальной нормы)
    cand_ids  id токенов, чьи логиты раскладываются
    Возвращает (матрица вкладов (компонент × токенов), суммы по столбцам, невязка потока)."""
    C = mx.stack([v.astype(F32).reshape(-1) for _, v in parts])            # (n, D)
    hf = h_final.astype(F32).reshape(-1)
    norm = lm.model.norm
    rms = mx.sqrt((hf * hf).mean() + norm.eps)
    scaled = C / rms * norm.weight.astype(F32)
    logits = unembed(lm)(scaled).astype(F32)                               # (n, V)
    sel = logits[:, mx.array(cand_ids)]                                    # (n, K)
    totals = sel.sum(axis=0)
    gap = mx.linalg.norm(C.sum(axis=0) - hf) / (mx.linalg.norm(hf) + 1e-9)
    mx.eval(sel, totals, gap)
    return sel.tolist(), totals.tolist(), gap.item()


# ───────────────── вклад токенов контекста: прямая подмена ─────────────────
def occlusion_attribution(lm, ids: list, target: int, other: int = None, chunk: int = 8):
    """Насколько каждый токен контекста подталкивал к target (а не к other).

    Токен по очереди заменяется на «нейтральный» — средний эмбеддинг контекста, — и
    измеряется, как изменилась метрика (разность логитов target и other на последней
    позиции):  вклад = метрика(полный контекст) − метрика(токен заменён).
    Положительный вклад — токен подталкивал к target, отрицательный — к other.

    Это прямое измерение, а не оценка по градиенту: ему можно верить в той мере, в какой
    «средний эмбеддинг» — разумная замена токена. Варианты считаются пачками по chunk
    штук (одна пачка — один прямой проход). Возвращает (вклады по позициям, метрика)."""
    L = len(ids)
    x = mx.array([ids])
    emb = lm.model.embed_tokens(x)                                          # (1, L, D)
    base = emb.astype(F32).mean(axis=1, keepdims=True).astype(emb.dtype)    # (1, 1, D)
    proj = unembed(lm)

    def metric(e, xs):
        out = lm(xs, inputs_embeds=e, skip_logits=True, return_hidden=True)
        lg = proj(out.hidden_states[-1][:, -1, :]).astype(F32)             # (B, V)
        return (lg[:, target] - lg[:, other]) if other is not None else lg[:, target]

    # LanguageModel хранит позиции последнего вызова; пачка с B > 1 их переписывает
    saved = (getattr(lm, "_position_ids", None), getattr(lm, "_rope_deltas", None))
    try:
        scores, full = [], None
        for s0 in range(0, L, chunk):
            idx = mx.arange(s0, min(s0 + chunk, L))
            B = idx.shape[0]
            hit = (mx.arange(L)[None, :] == idx[:, None])[..., None]        # (B, L, 1)
            E = mx.where(hit, base, mx.broadcast_to(emb, (B, L, emb.shape[-1])))
            # строка 0 — исходный контекст без замены: она считается в той же пачке, что и
            # варианты, чтобы общая для всех строк погрешность умножения матриц сокращалась
            E = mx.concatenate([emb, E], axis=0)
            m = metric(E, mx.broadcast_to(x, (B + 1, L)))
            scores.append(m[0] - m[1:])
            full = m[0] if full is None else full
        out = mx.concatenate(scores)
        mx.eval(out, full)
    finally:
        lm._position_ids, lm._rope_deltas = saved
    return out.tolist(), full.item()


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


# ───────────────── угловое расстояние между слоями ─────────────────
def residual_stack(lm, ids: list, max_pos: int = 128) -> mx.array:
    """Векторы остаточного потока на входе каждого слоя и после последнего:
    (слоёв + 1, позиций, D). Берутся последние max_pos позиций, кроме самой первой
    (первый токен — «якорь» внимания с нетипичной геометрией)."""
    n_layers = len(lm.model.layers)
    x = mx.array([ids])
    out = lm(x, capture_layer_ids=list(range(n_layers)))
    hs = [lm.model.embed_tokens(x)] + list(out.hidden_states[:n_layers])
    stack = mx.stack([h[0, 1:][-max_pos:].astype(F32) for h in hs])
    mx.eval(stack)
    return stack


def angular_distances(stacks, block_sizes: list) -> dict:
    """d(l, l+n) = arccos(cos(h_l, h_{l+n})) / π, усреднённое по позициям и промптам.

    Малое расстояние значит, что блок из n слоёв, начинающийся со слоя l, почти не
    поворачивает вектор, — такой блок первым идёт в кандидаты на удаление.
    stacks — итерируемое из массивов (слоёв + 1, позиций, D) (или (слоёв + 1, D) для одной
    позиции); обрабатываются по одному, так что их не нужно держать в памяти все сразу."""
    acc, count = {}, 0
    for s in stacks:
        s = s.astype(F32)
        if s.ndim == 2:
            s = s[:, None, :]
        n_layers = s.shape[0] - 1
        for n in block_sizes:
            if n < 1 or n > n_layers:
                continue
            a, b = s[:n_layers - n + 1], s[n:]
            cos = (a * b).sum(axis=-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-9)
            d = (mx.arccos(mx.clip(cos, -1.0, 1.0)) / math.pi).mean(axis=-1)    # по позициям
            acc[n] = d if n not in acc else acc[n] + d
        mx.eval(list(acc.values()))
        count += 1
    res = {}
    for n, d in acc.items():
        vals = (d / count).tolist()
        best = min(range(len(vals)), key=vals.__getitem__)
        res[str(n)] = {"dist": [round(v, 4) for v in vals], "best_start": best,
                       "best": round(vals[best], 4)}
    return res


__all__ = ["direct_logit_attribution", "occlusion_attribution", "token_attribution", "atp",
           "residual_stack", "angular_distances", "layer_name"]
