"""Сравнение квантизаций и трассировка квантизованной модели.

Запуск: pytest tests   или   python tests/run_all.py
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

from tiny import tiny_model
from helpers import PROMPT, of

from layerscope import layer_map as LM
from layerscope import pipeline
from layerscope import quant_compare
from layerscope.journal import Journal


# ───────────── квантизация ─────────────
def test_quant_compare():
    ref_m, tok = tiny_model()
    q_m, _ = tiny_model(bits=4)
    prompts = LM.DEFAULT_PROMPTS[:4]
    ref = quant_compare.collect(ref_m, tok, prompts, 4)
    same = quant_compare.compare(ref, quant_compare.collect(ref_m, tok, prompts, 4, texts=ref))
    assert same["kl"] < 1e-6 and same["top1_agree"] == 1.0 and max(same["rel_err"]) < 1e-3
    res = quant_compare.compare(ref, quant_compare.collect(q_m, tok, prompts, 4, texts=ref))
    assert res["kl"] > 1e-4 and res["rel_err"][-1] > res["rel_err"][0] >= 0
    assert len(res["cos"]) == 9 and "косинус" in quant_compare.render(res)


def test_quantized_model_traces():
    """Квантизованная модель (как реальные MLX-сборки): градиенты и разложение работают."""
    m, tok = tiny_model(bits=4)
    d = Path(tempfile.mkdtemp())
    j = Journal(d)
    with contextlib.redirect_stdout(io.StringIO()):
        pipeline.run(m, tok, PROMPT, j, max_tokens=3, detail_steps=1, wstats=True)
    j.close()
    ev = [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]
    assert of(ev, "token_attr") and of(ev, "dla")
    assert abs(of(ev, "dla")[0]["logit_sum"] - of(ev, "dla")[0]["logit_actual"]) < 5e-2
    res = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:2], 3, 1, log=lambda *a: None, fast=True)
    assert res["angular"]["1"]["best"] >= 0
