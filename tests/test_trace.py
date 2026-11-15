"""Один прогон: события, разложение логита, внимание, logit lens, энтропия, сэмплирование.

Запуск: pytest tests   или   python tests/run_all.py
"""
import mlx.core as mx
from helpers import PROMPT, model, of, traced_run

from qwen_trace import report
from qwen_trace.analysis import top_k
from qwen_trace.tracer import attention_weights


# ───────────── события и этапы прогона ─────────────
def test_run_events_and_meta():
    d, ev = traced_run(max_tokens=5, detail_steps=2)
    kinds = {e["kind"] for e in ev}
    assert {"meta", "model", "prompt", "tokens", "embeddings", "timing", "module_call",
            "logit_lens", "residual", "attention", "dla", "token_attr", "sample",
            "answer", "weights"} <= kinds
    meta = of(ev, "meta")[0]
    assert meta["versions"]["mlx"] and meta["params"]["seed"] == 0
    calls = of(ev, "module_call")
    assert sum("weights" in c for c in calls) == len({c["module"] for c in calls})
    assert all("kl_final" in e for e in of(ev, "logit_lens"))
    assert (d / "trace.log").stat().st_size > 0


def test_token_attribution_every_detailed_step():
    """Вклад токенов считается на каждом подробном шаге; контекст растёт вместе с ответом."""
    _, tok = model()
    _, ev = traced_run(max_tokens=5, detail_steps=3)
    ta = of(ev, "token_attr")
    n_prompt = len(of(ev, "tokens")[0]["ids"])
    steps = [e["step"] for e in ta]
    assert steps and steps == list(range(len(steps)))
    answer = [tok.decode([i]) for i in of(ev, "answer")[0]["ids"]]
    for e in ta:
        assert e["n_prompt"] == n_prompt
        assert len(e["scores"]) == len(e["tokens"]) == n_prompt + e["step"]
        assert e["tokens"][n_prompt:] == answer[:e["step"]]       # контекст = промпт + выбранное


def test_token_attribution_does_not_change_answer():
    """Анализ работает без кэша, но не должен влиять на сам прогон."""
    with_attr = of(traced_run(max_tokens=6, detail_steps=3)[1], "answer")[0]["ids"]
    without = of(traced_run(max_tokens=6, detail_steps=3, token_attr=False)[1], "answer")[0]["ids"]
    assert with_attr == without


def test_dla_is_exact():
    """Сумма вкладов компонент равна настоящему логиту выбранного токена."""
    _, ev = traced_run(max_tokens=3, detail_steps=2)
    for e in of(ev, "dla"):
        assert len(e["components"]) == 1 + 2 * 8
        assert abs(e["logit_sum"] - e["logit_actual"]) < 1e-2
        assert e["residual_gap"] < 1e-3


def test_every_answer_token_is_detailed_by_default():
    """Без --detail-steps разбирается каждый токен ответа (и токен конца), а не первые N."""
    _, ev = traced_run(max_tokens=7)
    samples = of(ev, "sample")
    n = len(samples)
    assert n >= 2
    for kind in ("dla", "token_attr", "residual", "logit_lens"):
        steps = {e["step"] for e in of(ev, kind)}
        assert steps == set(range(n)), kind
    assert all(e["traced"] for e in samples)
    assert of(ev, "meta")[0]["params"]["detail_steps"] == 7        # записано итоговое значение
    # явное ограничение по-прежнему работает
    _, ev2 = traced_run(max_tokens=7, detail_steps=2)
    assert {e["step"] for e in of(ev2, "dla")} == {0, 1}


def test_dla_any_pair_of_candidates():
    """Вклады посчитаны для пяти кандидатов: сумма по компонентам равна настоящему логиту
    каждого из них, а разность любой пары получается вычитанием."""
    _, ev = traced_run(max_tokens=3, detail_steps=1)
    e = of(ev, "dla")[0]
    assert len(e["cands"]) == 5 and e["cands"][0]["id"] == e["target"]["id"]
    assert e["cands"][1]["id"] == e["other"]["id"]
    for j in range(5):
        assert abs(e["logits_sum"][j] - e["logits_actual"][j]) < 1e-2
        assert abs(sum(c["logits"][j] for c in e["components"]) - e["logits_sum"][j]) < 1e-2
    assert all(abs(c["diff"] - (c["logits"][0] - c["logits"][1])) < 2e-3 for c in e["components"])


def test_entropy_detail_adds_up():
    """Пошаговый расчёт энтропии сходится: топ-20 плюс хвост равны полной энтропии."""
    m, tok = model()
    lm = m.language_model
    x = mx.array([tok.encode(PROMPT)])
    logits = lm(x).logits[0, -1]
    best, extra = top_k(logits, tok)
    d = extra["detail"]
    p = mx.softmax(logits.astype(mx.float32))
    H = -(p * mx.log(p)).sum().item()
    assert abs(extra["entropy"] - H) < 1e-3
    assert abs(d["top_term"] + d["tail"]["term"] - extra["entropy"]) < 2e-3
    assert len(d["top"]) == 20 and d["tail"]["n"] == d["vocab"] - 20
    assert abs(sum(t["prob"] for t in d["top"]) + d["tail"]["mass"] - 1) < 2e-3
    assert abs(d["perplexity"] - mx.exp(mx.array(H)).item()) < 0.05
    assert abs(d["bits"] * 0.6931471805599453 - H) < 1e-3 and d["max_entropy"] > H
    for t in d["top"][:3]:                                   # ln p = z − ln Σ exp(z)
        assert abs(t["logit"] - d["lse"] - mx.log(mx.array(t["prob"])).item()) < 2e-3
    assert abs(d["lse"] - mx.logsumexp(logits.astype(mx.float32)).item()) < 1e-3
    _, ev = traced_run(max_tokens=3, detail_steps=1)
    assert all("detail" in e for e in of(ev, "sample"))


def test_attention_matrix_every_query_position():
    """Внимание записано для каждой позиции запроса: строки нормированы, будущее не видно,
    последняя строка совпадает с вниманием последнего токена."""
    _, ev = traced_run(max_tokens=2, detail_steps=1)
    for a in of(ev, "attention"):
        mat, q0 = a["matrix"], a["q_start"]
        assert len(mat) == len(mat[0]) - q0
        for i, row in enumerate(mat):
            assert abs(sum(row) - 1) < 0.02
            assert all(v == 0 for v in row[q0 + i + 1:])
        assert max(abs(x - y) for x, y in zip(mat[-1], a["dist"])) < 2e-3


def test_lens_grid_all_prompt_positions():
    """Logit lens на каждой позиции промпта; на последнем слое — настоящие предсказания."""
    m, tok = model()
    lm = m.language_model
    d, ev = traced_run(max_tokens=2, detail_steps=1)
    g = of(ev, "lens_grid")[0]
    ids = of(ev, "tokens")[0]["ids"]
    assert g["positions"] == list(range(len(ids)))
    assert len(g["top"]) == 8 and all(len(r) == len(ids) for r in g["top"])
    real = mx.argmax(lm(mx.array([ids])).logits[0], axis=-1).tolist()
    assert g["top"][-1] == real                                  # последний слой = настоящий выход
    assert all(g["rank"][-1][i] >= 1 for i in range(len(ids) - 1))
    # ранг настоящего продолжения совпадает с прямым подсчётом
    lg = lm(mx.array([ids])).logits[0, 3].astype(mx.float32)
    assert g["rank"][-1][3] == int((lg > lg[ids[4]]).sum().item()) + 1
    assert set(map(str, {t for r in g["top"] for t in r})) <= set(g["vocab"])
    assert report.build_data(d)["lens_grid"]["positions"] == g["positions"]


def test_dla_with_skipped_layer():
    _, ev = traced_run(max_tokens=3, detail_steps=1, skip_layers=[2])
    e = of(ev, "dla")[0]
    assert all(c["layer"] != 2 for c in e["components"])
    assert abs(e["logit_sum"] - e["logit_actual"]) < 1e-2
    assert of(ev, "ablation") and of(ev, "layer_skipped")


def test_attention_heads():
    _, ev = traced_run(max_tokens=2, detail_steps=1)
    att = of(ev, "attention")
    assert len(att) == 2                                      # слои 3 и 7 — полное внимание
    for a in att:
        assert len(a["heads_dist"]) == a["heads"] == 4
        assert all(abs(sum(row) - 1) < 0.02 for row in a["heads_dist"])


def test_sampling_is_reproducible():
    run = lambda seed: of(traced_run(max_tokens=8, detail_steps=0, temperature=1.5,
                                     token_attr=False, seed=seed)[1], "answer")[0]["ids"]
    assert run(3) == run(3)
    assert len({tuple(run(s)) for s in range(4)}) > 1


def test_tracer_cleans_up():
    m, _ = model()
    from mlx_vlm.models.qwen3_5 import language as lang
    before = lang.scaled_dot_product_attention
    traced_run(max_tokens=2, detail_steps=1)
    assert lang.scaled_dot_product_attention is before
    assert "wrapped" not in type(m.language_model).__call__.__qualname__


def test_attention_weights_match_real_attention():
    """Восстановленные веса внимания, умноженные на значения, дают выход настоящего внимания
    (то же самое проверено на Qwen3.8-27B: ошибка около 1% из-за bf16)."""
    from mlx_vlm.models.qwen3_5 import language as lang
    m, tok = model()
    orig, errs = lang.scaled_dot_product_attention, []

    def wrapped(queries, keys, values, cache, scale, mask, *a, **kw):
        out = orig(queries, keys, values, cache, scale, mask, *a, **kw)
        W = attention_weights(queries, keys, scale)
        H, nq, _ = W.shape
        v = values[0].astype(mx.float32)
        v = mx.repeat(v, H // v.shape[0], axis=0) if v.shape[0] != H else v
        real = out[0, :, -nq:, :].astype(mx.float32)
        errs.append((mx.abs(W @ v - real).max() / (mx.abs(real).max() + 1e-9)).item())
        return out

    lang.scaled_dot_product_attention = wrapped
    try:
        lm = m.language_model
        ids = tok.encode("What is the capital of France?")
        cache = lm.make_cache()
        lm(mx.array([ids]), cache=cache)                            # весь промпт сразу
        lm(mx.array([[7]]), cache=cache)                            # один токен с кэшем
    finally:
        lang.scaled_dot_product_attention = orig
    assert len(errs) == 4 and max(errs) < 5e-3, errs
