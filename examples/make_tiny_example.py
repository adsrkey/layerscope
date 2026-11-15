"""Пересобирает examples/tiny_model.html: полный набор разделов отчёта на крошечной случайной
модели (веса не нужны). Запуск из корня проекта:

    python examples/make_tiny_example.py

Показывает всё, что есть в отчёте, включая tuned lens, патчинг и сравнение квантизаций.
Числа на случайной модели ничего не говорят о Qwen — пример нужен, чтобы увидеть разделы.
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from tiny import tiny_model  # noqa: E402

from layerscope import layer_map as LM, patching, quant_compare, report, tuned_lens  # noqa: E402
from layerscope.journal import Journal  # noqa: E402
from layerscope.pipeline import run  # noqa: E402


def main():
    out = Path(tempfile.mkdtemp()) / "tiny_run"
    model, tok = tiny_model()
    corpus = tuned_lens.chunks_from_text(tok, "the quick brown fox jumps over the lazy dog. " * 60, 32)
    lens = tuned_lens.train(model, corpus, rank=8, steps=150, lr=3e-3, positions=8, log=lambda *a: None)

    journal = Journal(out)
    with contextlib.redirect_stdout(io.StringIO()):
        run(model, tok, "What is the capital of France?", journal, max_tokens=12, detail_steps=2,
            wstats=True, lens=lens)
        res = LM.build_map(model, tok, LM.load_prompts(), 6, log=lambda *a: None)
    journal.text(LM.render(res))
    LM.save_json(res, out / "layer_map.json")
    journal.close()

    pr = patching.run_patching(model, tok, "The capital of France is", "The capital of Russia is",
                               ("resid", "attn", "mlp"), answer="P", wrong="M", log=lambda *a: None)
    (out / "patching.json").write_text(json.dumps(pr, ensure_ascii=False), encoding="utf-8")

    quantized, _ = tiny_model(bits=4)
    ref = quant_compare.collect(model, tok, LM.load_prompts(), 6)
    qc = quant_compare.compare(ref, quant_compare.collect(quantized, tok, [r["prompt"] for r in ref],
                                                          6, texts=ref))
    qc["ref"], qc["test"] = {"path": "tiny (float32)"}, {"path": "tiny (4 бита)"}
    qc["layer_types"] = "".join("L" if l.is_linear else "A" for l in model.language_model.model.layers)
    (out / "quant_compare.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    target = ROOT / "examples" / "tiny_model.html"
    shutil.copy(report.build_report(out), target)
    print(target)


if __name__ == "__main__":
    main()
