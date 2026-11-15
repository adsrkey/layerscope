"""Общие помощники тестов: крошечная модель, прогон во временную папку, пакет из трёх промптов."""
import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

from tiny import tiny_model

from qwen_trace import batch, pipeline
from qwen_trace.journal import Journal

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "What is 2+2?"
_cache = {}


def model():
    if "m" not in _cache:
        _cache["m"] = tiny_model()
    return _cache["m"]


def calm_model():
    """Веса обычного масштаба: функция модели гладкая, годится для проверки градиентов.
    (У model() веса увеличены, чтобы слои заметно влияли на ответ, — она хаотична.)"""
    if "calm" not in _cache:
        _cache["calm"] = tiny_model(gain=1.0)
    return _cache["calm"]


def traced_run(**kw):
    """Прогон pipeline.run во временную папку; возвращает (папка, события)."""
    m, tok = model()
    d = Path(tempfile.mkdtemp())
    j = Journal(d)
    with contextlib.redirect_stdout(io.StringIO()):
        pipeline.run(m, tok, PROMPT, j, **kw)
    j.close()
    return d, [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]


def of(events, kind):
    return [e for e in events if e["kind"] == kind]


def make_batch():
    """Пакет из трёх промптов (средний падает) с картой слоёв; собирается один раз."""
    if "batch" in _cache:
        return _cache["batch"]
    m, tok = model()
    out = Path(tempfile.mkdtemp()) / "batch"
    prompts = ["What is 2+2?", "boom", "Name a color."]
    kw = dict(max_tokens=4, detail_steps=2, thinking=False, raw=False, skip_layers=[],
              wstats=False, temperature=0.0, top_p=1.0, seed=0, token_attr=True,
              lens=None)
    orig = batch.run

    def flaky(model, tok, prompt, *a, **k):
        if prompt == "boom":
            raise RuntimeError("boom")
        return orig(model, tok, prompt, *a, **k)

    batch.run = flaky
    try:
        batch.run_batch(m, tok, prompts, out, kw, map_cfg={"tokens": 3, "window": 1, "fast": False},
                    log=lambda *a: None)
    finally:
        batch.run = orig
    _cache["batch"] = out
    return out


def page_blocks(html: str) -> dict:
    return {k: json.loads(v.replace("<\\/", "</")) for k, v in
            re.findall(r'<script id="([\w-]+)" type="application/json">(.*?)</script>', html, re.S)}
