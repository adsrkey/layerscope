"""Тесты на крошечной случайной модели: веса скачивать не нужно, идут секунды.

    pytest tests            или            python tests/test_all.py
"""
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import attribution as A  # noqa: E402
import layer_map as LM  # noqa: E402
import make_report  # noqa: E402
import patching  # noqa: E402
import qcore  # noqa: E402
import quant_compare  # noqa: E402
import trace_qwen as T  # noqa: E402
import tuned_lens  # noqa: E402
from tiny import tiny_model  # noqa: E402

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
    """Прогон trace_qwen во временную папку; возвращает (папка, события)."""
    m, tok = model()
    d = Path(tempfile.mkdtemp())
    j = T.Journal(d)
    with contextlib.redirect_stdout(io.StringIO()):
        T.run(m, tok, PROMPT, j, **kw)
    j.close()
    return d, [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]


def of(events, kind):
    return [e for e in events if e["kind"] == kind]


# ───────────── ядро ─────────────
def test_hooks_restore_classes():
    m, _ = model()
    before = {c: c.__dict__.get("__call__") for c in (nn.Linear, nn.RMSNorm, nn.Embedding)}
    with qcore.Hooks(m.language_model):
        assert nn.Linear.__dict__["__call__"] is not before[nn.Linear]
    for c, own in before.items():
        assert c.__dict__.get("__call__") is own


def test_hooks_do_not_change_output():
    m, tok = model()
    lm, x = m.language_model, mx.array([tok.encode(PROMPT)])
    base = lm(x).logits
    with qcore.Hooks(lm) as h:
        h.capture = {qcore.layer_name(3)}
        same = lm(x).logits
        assert qcore.layer_name(3) in h.captured
        h.skip = {qcore.layer_name(3)}
        diff = lm(x).logits
    assert mx.abs(base - same).max().item() == 0
    assert mx.abs(base - diff).max().item() > 1e-3
    assert mx.abs(base - lm(x).logits).max().item() == 0       # после выхода всё как было


def test_skip_equals_removing_layer_writes():
    """Отключение слоя = вычесть из потока записи его внимания и MLP."""
    m, tok = model()
    lm, x = m.language_model, mx.array([tok.encode(PROMPT)])
    with qcore.Hooks(lm) as h:
        h.keep_cache = False
        h.skip = {qcore.layer_name(7)}
        skipped = lm(x).logits
        h.skip = set()
        h.capture = {qcore.layer_name(6)}
        lm(x)
        expect = qcore.lens_logits(lm, h.captured[qcore.layer_name(6)])
    assert mx.abs(skipped - expect).max().item() < 1e-4


def test_log_softmax_and_kl():
    lg = mx.array([[1.0, 2.0, 3.0], [0.0, 0.0, 50.0]])
    lp = qcore.log_softmax(lg)
    assert abs(mx.exp(lp).sum(axis=-1)[0].item() - 1) < 1e-6
    assert qcore.kl_div(lp, lp).abs().max().item() == 0
    assert qcore.kl_div(lp[0], lp[1]).item() > 0


def test_pick_token():
    lg = mx.array([0.0, 5.0, 1.0])
    assert qcore.pick_token(lg) == 1
    mx.random.seed(1)
    a = [qcore.pick_token(lg, 1.5, 0.9) for _ in range(20)]
    mx.random.seed(1)
    assert a == [qcore.pick_token(lg, 1.5, 0.9) for _ in range(20)]
    assert all(qcore.pick_token(lg, 1.0, 0.01) == 1 for _ in range(10))    # узкий top_p


def test_bootstrap_ci():
    lo, hi = qcore.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo < 3.0 < hi and lo >= 1.0 and hi <= 5.0
    assert qcore.bootstrap_ci([2.0]) == (2.0, 2.0)


# ───────────── trace_qwen ─────────────
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


def test_report_old_single_token_attr_format():
    """Журналы старого формата (один token_attr без шага и токенов) по-прежнему читаются."""
    d, ev = traced_run(max_tokens=2, detail_steps=1)
    lines = [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]
    with open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for e in lines:
            if e["kind"] == "token_attr":
                e = {k: v for k, v in e.items() if k not in ("step", "tokens", "n_prompt")}
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    assert make_report.build_data(d)["token_attrs"]["0"]["scores"]


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
    best, extra = T.top_k(logits, tok)
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
    assert make_report.build_data(d)["lens_grid"]["positions"] == g["positions"]


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


def test_token_attribution_ig_completeness():
    """Integrated gradients: сумма вкладов равна изменению метрики вдоль пути."""
    m, tok = calm_model()
    lm, ids = m.language_model, tok.encode("abcdef")
    # случайные слои линейного внимания дают очень извилистую функцию входа (у обученной
    # модели такого нет), поэтому полноту проверяем на остальной части модели
    gdn = [qcore.attn_name(lm, i) for i in range(8) if lm.model.layers[i].is_linear]
    with qcore.Hooks(lm) as h:
        h.edit = {n: (lambda out: out * 0) for n in gdn}
        attr, value, start = A.token_attribution(lm, ids, 5, 9, ig_steps=256)
    assert abs(sum(attr) - (value - start)) < 0.05 * abs(value - start) + 0.02
    g_attr, g_value, g_start = A.token_attribution(lm, ids, 5, 9)
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

    gdn = [qcore.attn_name(lm, i) for i in range(8) if lm.model.layers[i].is_linear]
    with qcore.Hooks(lm) as h, qcore.grad_mode(lm):
        h.edit = {n: (lambda out: out * 0) for n in gdn}       # см. тест выше
        lin = (mx.grad(f)(emb) * d).sum().item()
        fd = (f(emb + 2e-3 * d).item() - f(emb - 2e-3 * d).item()) / 4e-3
    assert abs(lin - fd) < 0.1 * abs(fd) + 0.05


def test_tracer_cleans_up():
    m, _ = model()
    from mlx_vlm.models.qwen3_5 import language as lang
    before = lang.scaled_dot_product_attention
    traced_run(max_tokens=2, detail_steps=1)
    assert lang.scaled_dot_product_attention is before
    assert "wrapped" not in type(m.language_model).__call__.__qualname__


# ───────────── пакетный режим ─────────────
def test_prompt_slug():
    assert T.prompt_slug("Назови столицу Франции одним словом.") == "назови-столицу-франции-одним-словом"
    assert "/" not in T.prompt_slug("a/b: c?") and T.prompt_slug("???") == "prompt"
    assert len(T.prompt_slug("x" * 200)) <= 36


def make_batch():
    """Пакет из трёх промптов (средний падает) с картой слоёв; собирается один раз."""
    if "batch" in _cache:
        return _cache["batch"]
    m, tok = model()
    out = Path(tempfile.mkdtemp()) / "batch"
    prompts = ["What is 2+2?", "boom", "Name a color."]
    kw = dict(max_tokens=4, detail_steps=2, thinking=False, raw=False, skip_layers=[],
              wstats=False, temperature=0.0, top_p=1.0, seed=0, token_attr=True, ig_steps=0,
              lens=None)
    orig = T.run

    def flaky(model, tok, prompt, *a, **k):
        if prompt == "boom":
            raise RuntimeError("boom")
        return orig(model, tok, prompt, *a, **k)

    T.run = flaky
    try:
        T.run_batch(m, tok, prompts, out, kw, map_cfg={"tokens": 3, "window": 1, "fast": False},
                    log=lambda *a: None)
    finally:
        T.run = orig
    _cache["batch"] = out
    return out


def page_blocks(html: str) -> dict:
    return {k: json.loads(v.replace("<\\/", "</")) for k, v in
            re.findall(r'<script id="([\w-]+)" type="application/json">(.*?)</script>', html, re.S)}


def test_batch_mode_one_report_with_prompt_switcher():
    """Один report.html на весь пакет; сбой одного промпта не губит остальных."""
    out = make_batch()
    b = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    assert [e["status"] for e in b["prompts"]] == ["ok", "error", "ok"]
    assert "boom" in b["prompts"][1]["error"]
    good = [e for e in b["prompts"] if e["status"] == "ok"]
    for e in good:                                   # журнал каждого промпта — в своей подпапке
        d = out / e["dir"]
        assert all((d / n).exists() for n in ("events.jsonl", "trace.log", "layer_map.json"))
        assert of([json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")], "dla")
        assert not (d / "report.html").exists()      # отдельных отчётов на промпт нет
    html = (out / "report.html").read_text(encoding="utf-8")
    assert not (out / "index.html").exists()
    blocks = page_blocks(html)
    assert set(blocks) == {"batch", "d-0", "d-2"}    # у упавшего промпта данных нет
    batch = blocks["batch"]
    assert [e["status"] for e in batch["entries"]] == ["ok", "error", "ok"]
    assert batch["layer_map"] and blocks["d-0"]["layer_map"] is None   # карта — одна на всех
    assert blocks["d-0"]["prompt"]["prompt"] == "What is 2+2?"
    assert blocks["d-2"]["prompt"]["prompt"] == "Name a color."
    ok = [e for e in batch["entries"] if e["status"] == "ok"]
    assert all("first" in e and len(e["dla_abs"]["attn"]) == 8 for e in ok)
    # «особенно важный слой» показывается только при заметном превышении среднего
    assert all("map_top" not in e or e["map_top"]["ratio"] >= 1.5 for e in ok)
    assert make_report.build_batch_report(out).exists()      # пересборка из папки, без модели


def test_report_scripts_run_in_node():
    """Скрипт страницы выполняется: сводка и каждый промпт строятся без ошибок и NaN.
    Нужен node; без него тест ничего не проверяет."""
    node = shutil.which("node")
    if not node:
        print("node не найден — проверка скрипта страницы пропущена")
        return
    d, _ = traced_run(max_tokens=3, detail_steps=2)
    single = make_report.build_report(d)
    for path in (single, make_batch() / "report.html"):
        r = subprocess.run([node, str(ROOT / "tests" / "check_report.js"), str(path)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# ───────────── карта слоёв ─────────────
def test_atp_matches_small_intervention():
    """Градиентная оценка совпадает с настоящим малым вмешательством."""
    m, tok = calm_model()
    lm = m.language_model
    ids = tok.encode(PROMPT)
    full, start, ans = ids + [5, 9, 11], len(ids), [5, 9, 11]
    eps, layer = 2e-3, 4
    with qcore.Hooks(lm) as h:
        est = A.atp(lm, h, full, start, ans)

        def metric():
            lp = qcore.log_softmax(lm(mx.array([full])).logits[0, start - 1:-1])
            return mx.take_along_axis(lp, mx.array(ans)[:, None], axis=-1).mean().item()

        base = metric()
        assert abs(base - est["metric"]) < 2e-2      # обычный и градиентный режим считают чуть по-разному
        for kind, name in (("mlp", qcore.mlp_name(layer)), ("attn", qcore.attn_name(lm, layer))):
            h.edit = {name: lambda out: out * (1 - eps)}      # убрать eps-долю записи
            less = metric()
            h.edit = {name: lambda out: out * (1 + eps)}
            real = (less - metric()) / 2                      # центральная разность
            h.edit = {}
            assert abs(real - eps * est[kind][layer]) < 0.2 * abs(real) + 1e-5, (kind, real)


def test_layer_map_and_report():
    m, tok = model()
    res = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:5], 4, 1, log=lambda *a: None, n_boot=200)
    assert len(res["layers"]) == 8 and res["version"] == 2
    for r in res["layers"]:
        assert r["kl_lo"] <= r["kl"] <= r["kl_hi"] and "atp" in r and len(r["kl_per_prompt"]) == 5
    assert -1 <= res["atp"]["corr"] <= 1
    assert set(res["angular"]) >= {"1", "2", "4"}
    assert "95% интервал" in LM.render(res)
    fast = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:3], 4, 2, log=lambda *a: None, fast=True)
    assert len(fast["layers"]) == 4 and "kl" not in fast["layers"][0] and "atp" in fast["layers"][0]
    assert "Attribution patching" in LM.render(fast)
    d, _ = traced_run(max_tokens=3, detail_steps=1)
    LM.save_json(res, d / "layer_map.json")
    html = make_report.build_report(d).read_text(encoding="utf-8")
    assert "layer_map" in html and len(html) > 50_000


def test_angular_distance():
    a = mx.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    r = A.angular_distances([a], [1, 2])
    assert r["1"]["dist"] == [0.0, 0.5, 0.5] and r["1"]["best_start"] == 0
    assert r["2"]["dist"] == [0.5, 1.0]


def test_thinking_answer_offset():
    class Tok:
        def decode(self, ids):
            return {7: "</think>", 8: "\n"}.get(ids[0], "x")
    assert LM.after_thinking(Tok(), [1, 2, 7, 3, 4]) == 3
    assert LM.after_thinking(Tok(), [1, 2, 3]) is None


# ───────────── патчинг ─────────────
def test_patching_bounds():
    m, tok = model()
    res = patching.run_patching(m, tok, "The cat sat on", "The dog sat on",
                                ("resid", "attn", "mlp"), answer="A", wrong="B",
                                log=lambda *a: None)
    grid = res["components"]["resid"]
    assert len(grid) == 8 and len(grid[0]) == len(res["positions"])
    # подмена выхода последнего слоя на последней позиции полностью возвращает чистый ответ
    assert abs(grid[-1][-1] - 1.0) < 1e-3
    # позиции до первого отличия промптов не несут различий: подмена ничего не меняет
    assert all(abs(row[0]) < 1e-3 for row in grid)
    assert "ACTIVATION PATCHING" in patching.render(res)


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
    j = T.Journal(d)
    with contextlib.redirect_stdout(io.StringIO()):
        T.run(m, tok, PROMPT, j, max_tokens=3, detail_steps=1, wstats=True)
    j.close()
    ev = [json.loads(l) for l in open(d / "events.jsonl", encoding="utf-8")]
    assert of(ev, "token_attr") and of(ev, "dla")
    assert abs(of(ev, "dla")[0]["logit_sum"] - of(ev, "dla")[0]["logit_actual"]) < 5e-2
    res = LM.build_map(m, tok, LM.DEFAULT_PROMPTS[:2], 3, 1, log=lambda *a: None, fast=True)
    assert res["atp"] and res["atp"]["prompts"] == 2


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{'все тесты пройдены' if not failed else f'упало: {failed}'}")
    sys.exit(1 if failed else 0)
