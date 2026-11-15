"""Tuned lens.

Запуск: pytest tests   или   python tests/run_all.py
"""
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from helpers import model, of, traced_run

from qwen_trace import tuned_lens


# ───────────── tuned lens ─────────────
def test_tuned_lens_trains_and_loads():
    m, tok = model()
    lm = m.language_model
    chunks = tuned_lens.chunks_from_text(tok, "the quick brown fox jumps over the lazy dog " * 40, 32)
    lens = tuned_lens.train(m, chunks, rank=8, steps=60, lr=3e-3, positions=8, log=lambda *a: None)
    raw, tuned = json.loads(lens.meta["kl_raw"]), json.loads(lens.meta["kl_tuned"])
    assert sum(tuned) < sum(raw)                              # линза точнее обычного logit lens
    path = Path(tempfile.mkdtemp()) / "lens.safetensors"
    lens.save(path)
    loaded = tuned_lens.TunedLens.load(path)
    h = mx.random.normal((lm.args.hidden_size,))
    assert mx.abs(loaded(2, h) - lens(2, h)).max().item() < 1e-6
    _, ev = traced_run(max_tokens=2, detail_steps=1, lens=loaded)
    assert all("tuned_kl" in e for e in of(ev, "logit_lens"))


def test_identity_lens_equals_logit_lens():
    m, _ = model()
    lens = tuned_lens.TunedLens(tuned_lens.init_params(8, 64, 4))
    h = mx.random.normal((64,))
    assert mx.abs(lens(0, h) - h).max().item() == 0
