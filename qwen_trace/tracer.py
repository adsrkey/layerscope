"""Tracer: журнал вычислений модели поверх core.Hooks и пересчёт весов внимания."""
import time

import mlx.core as mx

from .core import Hooks
from .journal import stats, weight_info, weights_text


def attention_weights(queries, keys, scale, max_queries: int = 48):
    """Веса softmax-внимания (H, nq, L) последних nq позиций запроса ко всему контексту.

    Пересчитываются из запросов и ключей, которые получает функция внимания модели, с
    причинной маской (токен видит себя и всё до себя). Должны совпадать с настоящими
    весами: это проверено тестом (веса, умноженные на значения, дают выход настоящего
    внимания) и на Qwen3.8-27B."""
    nq = min(queries.shape[2], max_queries)
    q = queries[0, :, -nq:, :].astype(mx.float32)          # (H, nq, D)
    k = keys[0].astype(mx.float32)                         # (Hkv, L, D)
    if k.shape[0] != q.shape[0]:
        k = mx.repeat(k, q.shape[0] // k.shape[0], axis=0)
    L = k.shape[1]
    scores = (q @ k.transpose(0, 2, 1)) * scale            # (H, nq, L)
    qpos = (L - nq) + mx.arange(nq)                        # абсолютные позиции запросов
    causal = mx.arange(L)[None, :] <= qpos[:, None]
    return mx.softmax(mx.where(causal[None], scores, -mx.inf), axis=-1)


# ───────────────── перехват вызовов модулей ─────────────────
class Tracer:
    """Журнал вычислений поверх core.Hooks: какой модуль, какие веса, что на входе/выходе,
    плюс внимание последней позиции. Работает только внутри `with`: на выходе все обёртки
    снимаются, и модель снова ведёт себя как обычно.

        with Tracer(lm, journal) as tracer:
            tracer.enabled = True
            lm(inputs)

    Времена модулей — диагностические: чтобы снять статистику выхода, после каждого модуля
    вычисления принудительно завершаются (mx.eval), а это ломает ленивый граф MLX.
    Реальная скорость измеряется отдельным прогоном без журнала (событие timing)."""

    def __init__(self, root, journal, wstats: bool = False):
        self.hooks = Hooks(root)
        self.hooks.observer = self
        self.j = journal
        self.enabled = False
        self.step = "?"
        self.stack = []
        self.tokens = []                      # строки всех токенов контекста
        self.wstats = wstats
        self.winfo = {}
        self.parts = {}                       # записи внимания и MLP в остаточный поток (посл. позиция)
        self._sdpa = None

    # отключённые слои живут в Hooks; здесь — для удобства
    @property
    def skip(self):
        return self.hooks.skip

    @skip.setter
    def skip(self, names):
        self.hooks.skip = set(names)

    def __enter__(self):
        self.hooks.install()
        self._patch_attention()
        return self

    def __exit__(self, *exc):
        self._unpatch_attention()
        self.hooks.uninstall()
        return False

    def skipped(self, name):
        self.j.event("layer_skipped", step=self.step, module=name)
        self.j.text(f"{'  ' * len(self.stack)}{name} ОТКЛЮЧЁН: выход заменён входом", echo=False)

    def call(self, name, module, orig, args, kwargs):
        depth = len(self.stack)
        self.stack.append(name)
        t0 = time.perf_counter()
        try:
            out = orig(module, *args, **kwargs)
            if isinstance(out, mx.array):
                mx.eval(out)
        finally:
            self.stack.pop()
        ms = round((time.perf_counter() - t0) * 1000, 2)

        first = name not in self.winfo
        if first:
            self.winfo[name] = weight_info(module, self.wstats)
        winfo = self.winfo[name]
        rec = {"step": self.step, "depth": depth, "module": name,
               "class": type(module).__name__, "ms": ms}
        if first:                                   # веса модуля не меняются: пишем один раз
            rec["weights"] = winfo
        x = args[0] if args else None
        if isinstance(x, mx.array):
            rec["input_shape"] = list(x.shape)
        if isinstance(out, mx.array):
            rec["output"] = stats(out)
            if name.endswith((".self_attn", ".linear_attn", ".mlp")) and out.ndim == 3:
                self.parts[name] = out[0, -1]
        self.j.event("module_call", **rec)

        o = rec.get("output", {})
        self.j.text(
            f"{'  ' * depth}{name} [{rec['class']}] {ms} мс\n" +
            (f"{'  ' * depth}   веса: {weights_text(winfo)}\n" if first else "") +
            f"{'  ' * depth}   вход {rec.get('input_shape')} -> выход {o.get('shape')}"
            f"  mean={o.get('mean')} std={o.get('std')} |h_посл|={o.get('last_norm')}",
            echo=False)
        return out

    # ── внимание: на какие токены смотрит последняя позиция (слои с полным вниманием) ──
    def _patch_attention(self):
        from mlx_vlm.models.qwen3_5 import language as lang
        orig, tracer = lang.scaled_dot_product_attention, self

        def wrapped(queries, keys, values, cache, scale, mask, *a, **kw):
            if tracer.enabled:
                try:
                    tracer._log_attention(queries, keys, scale)
                except Exception as e:                             # журнал не должен ломать прогон
                    tracer.j.text(f"   [внимание не записано: {e}]", echo=False)
            return orig(queries, keys, values, cache, scale, mask, *a, **kw)

        self._sdpa = (lang, orig)
        lang.scaled_dot_product_attention = wrapped

    def _unpatch_attention(self):
        if self._sdpa:
            lang, orig = self._sdpa
            lang.scaled_dot_product_attention = orig
            self._sdpa = None

    def _log_attention(self, queries, keys, scale):
        """Внимание последних позиций запроса (до 48) ко всем токенам контекста.
        Для подробного разбора по головам берётся только самая последняя позиция."""
        w_all = attention_weights(queries, keys, scale)    # (H, nq, L)
        nq, L = w_all.shape[1], w_all.shape[2]
        w = w_all[:, -1, :]                                # (H, L): последний токен, по головам
        avg_all = w_all.mean(axis=0)                       # (nq, L): среднее по головам
        avg = avg_all[-1]
        head_pos, head_w = mx.argmax(w, axis=-1), mx.max(w, axis=-1)
        ent = -(avg * mx.log(avg + 1e-12)).sum()
        head_ent = -(w * mx.log(w + 1e-12)).sum(axis=-1)
        mx.eval(avg, head_pos, head_w, ent, head_ent, w, avg_all)
        top = mx.argsort(-avg)[:8].tolist()
        avg_l = avg.tolist()
        layer = self.stack[-1] if self.stack else "?"
        tk = lambda p: self.tokens[p] if p < len(self.tokens) else "?"
        items = [{"pos": p, "token": tk(p), "weight": round(avg_l[p], 4)} for p in top]
        rec = {"heads": w_all.shape[0], "context": L, "top": items,
               "entropy": round(ent.item(), 4),
               "head_top": [[p, round(v, 4)] for p, v in
                            zip(head_pos.tolist(), head_w.tolist())],
               "head_entropy": [round(v, 3) for v in head_ent.tolist()]}
        if L <= 512:
            rec["dist"] = [round(v, 4) for v in avg_l]
        if L <= 160:                                       # каждая голова отдельно: (H, L)
            rec["heads_dist"] = [[round(v, 3) for v in row] for row in w.tolist()]
            rec["matrix"] = [[round(v, 3) for v in row] for row in avg_all.tolist()]
            rec["q_start"] = L - nq                        # позиция первой строки матрицы
        self.j.event("attention", step=self.step, module=layer, **rec)
        pretty = "  ".join(f"{it['token']!r}:{it['weight']}" for it in items)
        self.j.text(f"{'  ' * len(self.stack)}ВНИМАНИЕ последнего токена "
                    f"(среднее по {w_all.shape[0]} головам): {pretty}", echo=False)
