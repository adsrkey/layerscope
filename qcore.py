#!/usr/bin/env python3
"""
qcore.py — общее ядро инструментов: перехват вызовов модулей, вмешательства в прогон
и мелкие численные утилиты. Сам ничего не запускает; его используют trace_qwen.py,
layer_map.py, patching.py, tuned_lens.py и quant_compare.py.

Главное здесь — Hooks. Это контекстный менеджер: на входе оборачивает __call__ у классов
модулей языковой модели, на выходе возвращает всё как было.

    with Hooks(lm) as h:
        h.skip = {layer_name(35)}                  # слой 35: выход заменён входом
        h.capture = {layer_name(10)}               # запомнить выход слоя 10 -> h.captured
        h.edit[layer_name(12)] = lambda out: out   # подменить выход слоя 12
        lm(inputs)
"""
import hashlib
import json
import platform
import random
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import mlx.core as mx

PREFIX = "language_model"


def layer_name(i: int) -> str:
    return f"{PREFIX}.model.layers.{i}"


def attn_name(lm, i: int) -> str:
    return f"{layer_name(i)}.{'linear_attn' if lm.model.layers[i].is_linear else 'self_attn'}"


def mlp_name(i: int) -> str:
    return f"{layer_name(i)}.mlp"


def layer_of(name: str):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None


# ───────────────── перехват вызовов модулей ─────────────────
class Hooks:
    """Перехват вызовов всех модулей под root. Работает только внутри `with`.

    skip      имена модулей, чей выход заменяется их первым аргументом (для слоя — входом)
    edit      имя -> функция(выход) -> новый выход (патчинг, добавление вектора)
    capture   имена модулей, чьи выходы нужно запомнить в captured
    observer  объект с полем enabled и методами call(...) и skipped(name) — для журнала
    keep_cache  считать ли отключённый слой (нужно, когда у прогона есть кэш генерации)

    Вложенные Hooks допустимы, но закрываться должны в обратном порядке (как обычные `with`).
    """

    def __init__(self, root, prefix: str = PREFIX):
        self.root = root
        self.names = {id(m): (f"{prefix}.{n}" if n else prefix) for n, m in root.named_modules()}
        self.skip = set()
        self.edit = {}
        self.capture = set()
        self.captured = {}
        self.observer = None
        self.keep_cache = True
        self._muted = 0
        self._installed = []

    def install(self):
        if self._installed:
            return self
        for cls in {type(m) for _, m in self.root.named_modules()}:
            own = cls.__dict__.get("__call__")          # None — метод унаследован
            cls.__call__ = self._wrap(cls.__call__)
            self._installed.append((cls, own))
        return self

    def uninstall(self):
        for cls, own in reversed(self._installed):
            if own is None:
                del cls.__call__
            else:
                cls.__call__ = own
        self._installed = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    def reset(self):
        """Снять все вмешательства (обёртки остаются до выхода из `with`)."""
        self.skip, self.edit, self.capture, self.captured = set(), {}, set(), {}

    def _wrap(self, orig):
        h = self

        def wrapped(module, *args, **kwargs):
            name = h.names.get(id(module))
            if name is None or h._muted:
                return orig(module, *args, **kwargs)
            obs = h.observer
            if name in h.skip:
                if h.keep_cache:
                    # Слой всё равно вычисляется (его кэш должен остаться согласованным),
                    # но результат отбрасывается, а вложенные вызовы не журналируются.
                    h._muted += 1
                    try:
                        orig(module, *args, **kwargs)
                    finally:
                        h._muted -= 1
                if obs is not None and obs.enabled:
                    obs.skipped(name)
                out = args[0]
            elif obs is not None and obs.enabled:
                out = obs.call(name, module, orig, args, kwargs)
            else:
                out = orig(module, *args, **kwargs)
            fn = h.edit.get(name)
            if fn is not None:
                out = fn(out)
            if name in h.capture:
                h.captured[name] = out
            return out

        return wrapped


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


# ───────────────── численные утилиты ─────────────────
def log_softmax(logits: mx.array) -> mx.array:
    """Логарифмы вероятностей напрямую из логитов (без log(p + eps), который шумит в хвостах)."""
    lf = logits.astype(mx.float32)
    return lf - mx.logsumexp(lf, axis=-1, keepdims=True)


def kl_div(logp_base: mx.array, logp_other: mx.array) -> mx.array:
    """KL(base || other) по последней оси; оба аргумента — логарифмы вероятностей."""
    return (mx.exp(logp_base) * (logp_base - logp_other)).sum(axis=-1)


def unembed(lm):
    """Отображение скрытого вектора в логиты (без финальной нормы)."""
    return lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head


def lens_logits(lm, h: mx.array) -> mx.array:
    """Логиты, которые получились бы, если бы h был выходом последнего слоя."""
    return unembed(lm)(lm.model.norm(h))


def eos_ids(model) -> set:
    eos = model.config.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}


def format_prompt(tok, prompt: str, thinking: bool = False, raw: bool = False) -> str:
    if raw:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def pick_token(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Выбор следующего токена: temperature 0 — самый вероятный, иначе случайно
    (с отсечением хвоста по top_p). Случайность задаётся mx.random.seed."""
    lf = logits.astype(mx.float32)
    if temperature <= 0:
        return int(mx.argmax(lf).item())
    lf = lf / temperature
    if top_p < 1.0:
        order = mx.argsort(-lf)
        probs = mx.softmax(lf[order], axis=-1)
        keep = (mx.cumsum(probs) - probs) < top_p        # первый токен остаётся всегда
        choice = mx.random.categorical(mx.where(keep, lf[order], -mx.inf))
        return int(order[choice].item())
    return int(mx.random.categorical(lf).item())


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Доверительный интервал среднего бутстрепом по наблюдениям (обычно — по промптам)."""
    values = list(values)
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx_, my_ = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys))
    sxx = sum((x - mx_) ** 2 for x in xs)
    syy = sum((y - my_) ** 2 for y in ys)
    return sxy / ((sxx * syy) ** 0.5 + 1e-30)


# ───────────────── модель и метаданные прогона ─────────────────
def load_model(path: str):
    from mlx_vlm import load
    model, processor = load(path)
    return model, getattr(processor, "tokenizer", processor)


def model_fingerprint(path) -> dict:
    """Отпечаток модели без чтения гигабайтов весов: config.json + имена и размеры файлов
    весов (в кэше Hugging Face имя файла-оригинала уже является его sha256)."""
    p = Path(path)
    if not p.is_dir():
        return {"path": str(path)}
    h = hashlib.sha256()
    cfg = p / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    files = []
    for f in sorted(p.glob("*.safetensors")):
        real = f.resolve()
        blob = real.name if re.fullmatch(r"[0-9a-f]{64}", real.name) else None
        files.append({"name": f.name, "bytes": real.stat().st_size, "sha256": blob})
        h.update(f"{f.name}:{real.stat().st_size}:{blob}".encode())
    quant = None
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            quant = c.get("quantization") or c.get("quantization_config")
        except ValueError:
            pass
    return {"path": str(p), "fingerprint": h.hexdigest()[:16], "files": files,
            "quantization": quant}


def _version(mod: str):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return None


def run_meta(model_path=None, **params) -> dict:
    """Всё, что нужно, чтобы повторить прогон: версии, модель, параметры запуска."""
    return {"created": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "versions": {m: _version(m) for m in ("mlx", "mlx-vlm", "mlx-lm", "transformers")},
            "model": model_fingerprint(model_path) if model_path else None,
            "params": params}
