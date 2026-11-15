#!/usr/bin/env python3
"""
make_report.py — собирает report.html из папки прогона trace_qwen.py.

    python make_report.py trace_logs/20261115_184500

Читает events.jsonl и (если есть) layer_map.json, patching.json, quant_compare.json,
пишет report.html рядом.
Файл самодостаточный: данные и код внутри, сеть не нужна, открывается двойным щелчком.
Модель для этого не нужна: отчёт можно пересобрать в любой момент.
"""
import json
import sys
from pathlib import Path


def build_data(run_dir) -> dict:
    run_dir = Path(run_dir)
    ev = [json.loads(l) for l in open(run_dir / "events.jsonl", encoding="utf-8") if l.strip()]

    def first(kind):
        return next((e for e in ev if e["kind"] == kind), None)

    D = {"run": run_dir.name, "model": first("model"), "prompt": first("prompt"),
         "meta": first("meta"), "timing": first("timing"),
         "tokens": first("tokens"), "embeddings": first("embeddings"),
         "weights": first("weights"), "ablation": first("ablation")}
    answers = [e for e in ev if e["kind"] == "answer"]
    D["answer"] = answers[-1] if answers else None

    D["calls"] = [{"m": e["module"], "c": e["class"], "d": e["depth"], "ms": e["ms"], "n": e["n"],
                   "o": {k: e["output"][k] for k in ("shape", "mean", "std", "last_norm")
                         if k in e.get("output", {})},
                   "w": (e.get("weights") or {}).get("wstats")}
                  for e in ev if e["kind"] == "module_call" and e["step"] == 0]

    lens, attn, resid = {}, {}, {}
    for e in ev:
        if e["kind"] == "logit_lens":
            lens.setdefault(str(e["step"]), []).append(
                {k: e[k] for k in ("layer", "layer_type", "top", "p_final", "rank_final", "entropy",
                                   "kl_final", "tuned_top", "tuned_p_final", "tuned_rank", "tuned_kl")
                 if k in e})
        elif e["kind"] == "attention":
            attn.setdefault(str(e["step"]), []).append(
                {k: e[k] for k in ("module", "heads", "context", "top", "entropy", "head_top", "dist",
                                   "heads_dist", "head_entropy")
                 if k in e})
        elif e["kind"] == "residual":
            resid[str(e["step"])] = {k: e[k] for k in ("norms", "rel", "cos", "emb_norms",
                                                       "final_id", "final_token")}
    for v in lens.values():
        v.sort(key=lambda r: r["layer"])
    D["lens"], D["attention"], D["resid"] = lens, attn, resid
    # вклад токенов контекста по шагам (в старых журналах он был один, без номера шага)
    D["token_attrs"] = {str(e.get("step", 0)): {k: v for k, v in e.items() if k not in ("n", "kind")}
                        for e in ev if e["kind"] == "token_attr"}
    D["dla"] = {str(e["step"]): {k: e[k] for k in ("target", "other", "components", "logit_sum",
                                                   "logit_actual", "residual_gap")}
                for e in ev if e["kind"] == "dla"}

    samples = [e for e in ev if e["kind"] == "sample"]
    D["samples"] = [{k: s[k] for k in ("step", "ms", "chosen", "candidates", "entropy",
                                       "nucleus95", "top20_mass", "traced") if k in s}
                    for s in samples if s.get("run") != "Обычный прогон"]
    D["skipped"] = sorted({int(e["module"].rsplit(".", 1)[-1]) for e in ev
                           if e["kind"] == "layer_skipped"})

    lm = run_dir / "layer_map.json"
    D["layer_map"] = json.loads(lm.read_text(encoding="utf-8")) if lm.exists() else None
    for key, name in (("patching", "patching.json"), ("quant", "quant_compare.json")):
        path = run_dir / name
        D[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return D


def build_report(run_dir, out=None) -> Path:
    run_dir = Path(run_dir)
    data = build_data(run_dir)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = TEMPLATE.replace("__TITLE__", "Отчёт: " + run_dir.name).replace("__DATA__", blob)
    out = Path(out) if out else run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


TEMPLATE = r'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  color-scheme: light;
  --bg:#f3f3f0; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --ink3:#77766f; --line:#dedcd6;
  --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --yellow:#eda100; --magenta:#e87ba4;
  --green:#008300; --violet:#6250d6; --red:#e34948; --mid:#f0efec;
  --seq-lo:#cde2fb; --seq-hi:#0d366b; --chip:#eceae4; --warn:#fff6dc;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --bg:#121211; --card:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --ink3:#97968c; --line:#34332f;
    --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --yellow:#c98500; --magenta:#d55181;
    --green:#0ca30c; --violet:#9085e9; --red:#e66767; --mid:#383835;
    --seq-lo:#14263d; --seq-hi:#9ec5f4; --chip:#2a2a27; --warn:#2c2616;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --bg:#121211; --card:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --ink3:#97968c; --line:#34332f;
  --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --yellow:#c98500; --magenta:#d55181;
  --green:#0ca30c; --violet:#9085e9; --red:#e66767; --mid:#383835;
  --seq-lo:#14263d; --seq-hi:#9ec5f4; --chip:#2a2a27; --warn:#2c2616;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:56px}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.mono,code,pre{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
nav{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);display:flex;gap:4px;padding:8px 16px;overflow-x:auto;align-items:center}
nav a{color:var(--ink2);text-decoration:none;padding:4px 10px;border-radius:6px;white-space:nowrap;font-size:13px}
nav a:hover{background:var(--chip);color:var(--ink)}
nav button{margin-left:auto}
main{max-width:1180px;margin:0 auto;padding:20px 16px 80px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 18px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:0 0 4px} h3{font-size:14px;margin:18px 0 6px}
.lead{color:var(--ink2);margin:2px 0 14px;max-width:900px}
.muted{color:var(--ink3)} .small{font-size:12px}
button,select{font:inherit;color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer}
button:hover{border-color:var(--ink3)} button.on{background:var(--ink);color:var(--card);border-color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;flex-direction:column;align-items:center;background:var(--chip);border:1px solid transparent;border-radius:6px;
  padding:2px 7px;font-size:13px;cursor:pointer;line-height:1.25}
.chip small{font-size:10px;color:var(--ink3)} .chip.sp{background:transparent;border:1px dashed var(--ink3);color:var(--ink2)}
.chip.on{border-color:var(--ink);outline:1px solid var(--ink)}
.chip.ans{border:1px solid var(--violet)}
.facts{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 0}
.fact{background:var(--chip);border-radius:8px;padding:6px 12px}
.fact b{display:block;font-size:16px;line-height:1.2} .fact span{font-size:11px;color:var(--ink3)}
.answer{font-size:16px;line-height:2;margin:6px 0 0;word-break:break-word}
.answer span{padding:2px 1px;border-radius:3px;white-space:pre-wrap}
pre{background:var(--chip);border-radius:8px;padding:10px 12px;overflow:auto;margin:6px 0;font-size:12.5px;white-space:pre-wrap}
pre .sp{color:var(--blue);font-weight:600}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink2);font-size:11px;font-family:system-ui,sans-serif}
svg .strong{fill:var(--ink);font-weight:600} svg .mute{fill:var(--ink3)}
.t-L{fill:var(--blue)} .t-A{fill:var(--orange)} .t-M{fill:var(--ink3)}
.axis{stroke:var(--line);stroke-width:1} .hair{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
.line{fill:none;stroke:var(--ink2);stroke-width:2}
.selframe{fill:none;stroke:var(--ink);stroke-width:2;pointer-events:none}
[data-tip]{cursor:pointer} .hit{fill:transparent}
.hit:hover,.hit:focus{fill:color-mix(in srgb,var(--ink) 7%,transparent);outline:none}
.box{fill:var(--chip);stroke:var(--line);cursor:pointer} .box:hover{stroke:var(--ink3)}
.arrow{fill:none;stroke:var(--ink3);stroke-width:1.5}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{text-align:left;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line);padding:4px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:4px 8px;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:color-mix(in srgb,var(--blue) 14%,transparent)}
.tiles{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.tile{background:var(--chip);border-radius:8px;padding:6px 12px;min-width:110px}
.tile b{display:block;font-size:17px;line-height:1.25;font-variant-numeric:tabular-nums} .tile span{font-size:11px;color:var(--ink3)}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px}
@media (max-width:860px){.cols{grid-template-columns:1fr}}
.note{background:var(--warn);border-radius:8px;padding:8px 12px;margin:10px 0;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
details{margin-top:10px} summary{cursor:pointer;color:var(--ink2);font-size:13px}
.scroll{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}
#tip{position:fixed;z-index:50;pointer-events:none;background:var(--ink);color:var(--card);padding:6px 9px;border-radius:6px;
  font-size:12px;line-height:1.4;white-space:pre;max-width:420px;display:none;box-shadow:0 4px 14px rgba(0,0,0,.25)}
.bar{display:inline-block;height:8px;border-radius:2px;background:var(--blue);vertical-align:middle}
.empty{color:var(--ink3);padding:14px;border:1px dashed var(--line);border-radius:8px}
</style>
</head>
<body>
<nav id="nav">
  <a href="#s-head">Обзор</a><a href="#s-pipeline">Конвейер</a><a href="#s-tokens">Токены</a>
  <a href="#s-emb">Эмбеддинги</a><a href="#s-layers">Слои</a><a href="#s-resid">Вектор по слоям</a>
  <a href="#s-lens">Logit lens</a><a href="#s-dla">Вклад слоёв</a><a href="#s-tokattr">Вклад токенов</a>
  <a href="#s-attn">Внимание</a><a href="#s-sample">Выбор токена</a>
  <a href="#s-map">Карта важности</a><a href="#s-patch" id="n-patch">Патчинг</a><a href="#s-quant" id="n-quant">Квантизация</a><a href="#s-weights">Веса</a>
  <button id="theme" title="Светлая / тёмная тема">◐</button>
</nav>
<main>
  <section id="s-head"></section>
  <section id="s-pipeline"></section>
  <section id="s-tokens"></section>
  <section id="s-emb"></section>
  <section id="s-layers"></section>
  <section id="s-resid"></section>
  <section id="s-lens"></section>
  <section id="s-dla"></section>
  <section id="s-tokattr"></section>
  <section id="s-attn"></section>
  <section id="s-sample"></section>
  <section id="s-map"></section>
  <section id="s-map2"></section>
  <section id="s-patch"></section>
  <section id="s-quant"></section>
  <section id="s-ablation"></section>
  <section id="s-weights"></section>
</main>
<div id="tip" role="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const D = JSON.parse(document.getElementById('data').textContent);
const M = D.model, NL = M.layers, TY = M.layer_types.split('');
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const vis = s => esc(String(s).replace(/\n/g,'↵').replace(/ /g,'␣'));
const f = (x, d=3) => x==null ? '—' : Number(x).toLocaleString('ru-RU', {maximumFractionDigits:d});
const big = n => n>=1e9 ? f(n/1e9,2)+' млрд' : n>=1e6 ? f(n/1e6,2)+' млн' : f(n,0);
const gb = b => b>=1e9 ? f(b/1e9,2)+' ГБ' : f(b/1e6,1)+' МБ';
const pct = x => x==null ? '—' : (x*100).toFixed(x<0.1 ? 2 : 1) + '%';
const tip = t => ` data-tip="${esc(t)}"`;
const clamp = (x,a=0,b=1) => Math.max(a, Math.min(b, x));
const seq = t => `color-mix(in oklab, var(--seq-hi) ${Math.round(clamp(t)*100)}%, var(--seq-lo))`;
const dvg = v => v>=0 ? `color-mix(in oklab, var(--blue) ${Math.round(clamp(v)*100)}%, var(--mid))`
                       : `color-mix(in oklab, var(--red) ${Math.round(clamp(-v)*100)}%, var(--mid))`;
const typeName = t => t==='L' ? 'линейное внимание (Gated DeltaNet)' : 'полное внимание';
const isSpecial = s => /^<\|.*\|>$/.test(s) || /^<\/?think>$/.test(s);
const layerOfName = n => { const m = /layers\.(\d+)/.exec(n); return m ? +m[1] : null; };
const sum = a => a.reduce((x,y)=>x+y,0);
const clip = (s,n) => s.length>n ? s.slice(0,n-1)+'…' : s;
const empty = msg => `<div class="empty">${msg}</div>`;
const hdr = (id, title, lead) => `<h2>${title}</h2><p class="lead">${lead}</p>`;

const lensSteps = Object.keys(D.lens).map(Number).sort((a,b)=>a-b);
const S = {tok: 0, layer: null, lens: lensSteps.length ? lensSteps[0] : null, sstep: 0, metric: 'kl', attn: null, hl: null};
const LMX = D.layer_map;                       // карта в любом виде (в т.ч. быстрая, без перебора)
const LM = LMX && !LMX.fast ? LMX : null;      // карта с полным перебором
const MP = (D.meta && D.meta.params) || {};
const nTok = D.tokens ? D.tokens.ids.length : 0;
const toks = D.tokens ? D.tokens.tokens : [];
const lensRows = () => S.lens==null ? [] : (D.lens[String(S.lens)] || []);
const residOf = () => S.lens==null ? null : D.resid[String(S.lens)];

// ───────────── Шапка ─────────────
function repro(){
  const m = D.meta; if (!m) return '';
  const mo = m.model || {}, v = m.versions || {};
  const rows = [['модель', mo.path], ['отпечаток модели', mo.fingerprint], ['квантизация', mo.quantization ? JSON.stringify(mo.quantization) : null],
    ['версии', Object.keys(v).filter(k => v[k]).map(k => k+' '+v[k]).join(', ')], ['Python', m.python], ['система', m.platform],
    ['параметры', JSON.stringify(m.params)], ['команда', (m.argv||[]).join(' ')]].filter(r => r[1]);
  return `<details><summary>как повторить этот прогон (версии, модель, параметры)</summary><table>${rows.map(r => `<tr><td class="muted">${r[0]}</td><td class="mono" style="word-break:break-all">${esc(r[1])}</td></tr>`).join('')}</table></details>`;
}
function renderHead(){
  const smp = D.samples, nGen = smp.length;
  let ans = '';
  smp.forEach(s => {
    const c = s.chosen;
    ans += `<span style="background:${seq(c.prob*0.55)}; color:${c.prob>0.6?'var(--card)':'var(--ink)'}"${tip(`шаг ${s.step}: ${c.token.replace(/\n/g,'↵')}\nвероятность ${pct(c.prob)}\nэнтропия ${f(s.entropy,2)}`)}>${esc(c.token)}</span>`;
  });
  const quant = M.bits ? `${M.bits} бит` : 'без квантизации';
  const tw = D.weights ? D.weights.total_params : null;
  const pre = D.samples.length && D.samples[0].traced !== false ? D.samples[0].ms : null;
  const skipped = D.skipped.length ? `<div class="note">Слои <b>${D.skipped.join(', ')}</b> отключены в этом прогоне (выход заменён входом). Все цифры ниже — для модели без них.</div>` : '';
  $('#s-head').innerHTML = `
    <h1>Как модель шла от промпта к ответу</h1>
    <div class="muted small">прогон ${esc(D.run)}${M.created ? ' · ' + esc(M.created.replace('T',' ')) : ''}</div>
    ${skipped}
    <h3>Промпт</h3><div>${esc(D.prompt.prompt)}</div>
    <h3>Ответ <span class="muted small">— цвет фона = вероятность токена, наведите курсор</span></h3>
    <div class="answer">${ans || '<span class="muted">пусто</span>'}</div>
    <div class="facts">
      <div class="fact"><b>${NL}</b><span>слоёв (${TY.filter(t=>t==='L').length} лин. + ${TY.filter(t=>t==='A').length} полн.)</span></div>
      <div class="fact"><b>${f(M.hidden,0)}</b><span>размер вектора токена</span></div>
      <div class="fact"><b>${f(M.vocab,0)}</b><span>токенов в словаре</span></div>
      <div class="fact"><b>${nTok}</b><span>токенов в промпте</span></div>
      <div class="fact"><b>${nGen}</b><span>токенов в ответе</span></div>
      <div class="fact"><b>${quant}</b><span>веса</span></div>
      ${tw ? `<div class="fact"><b>${big(tw)}</b><span>параметров (${gb(D.weights.total_bytes)})</span></div>` : ''}
      ${D.timing ? `<div class="fact"><b>${f(D.timing.prefill_ms,0)} мс</b><span>обработка промпта${D.timing.tokens_per_s ? ' ('+f(D.timing.tokens_per_s,0)+' ток/с)' : ''}</span></div>` : ''}
      ${pre ? `<div class="fact"><b>${f(pre,0)} мс</b><span>то же с журналом (времена модулей завышены)</span></div>` : ''}
      ${MP.temperature>0 ? `<div class="fact"><b>T=${MP.temperature}</b><span>сэмплирование, top_p ${MP.top_p}, seed ${MP.seed}</span></div>` : ''}
    </div>${repro()}`;
}

// ───────────── Конвейер ─────────────
function renderPipeline(){
  const nL = TY.filter(t=>t==='L').length, nA = NL - nL;
  const st = [
    ['Промпт', `${D.prompt.prompt.length} символов`, 's-head'],
    ['Шаблон чата', D.prompt.raw ? 'выключен (--raw)' : 'роли и <think>', 's-tokens'],
    ['Токенизация', `${nTok} токенов`, 's-tokens'],
    ['Эмбеддинги', `вектор из ${f(M.hidden,0)}`, 's-emb'],
    [`${NL} слоёв`, `${nL} лин. + ${nA} полн.`, 's-layers'],
    ['Норма + lm_head', `в ${f(M.vocab,0)} оценок`, 's-lens'],
    ['Softmax', 'оценки → вероятности', 's-sample'],
    ['Выбор токена', 'самый вероятный', 's-sample'],
    ['Ответ', `${D.samples.length} токенов`, 's-head'],
  ];
  const bw = 108, gap = 14, y = 8, bh = 62, W = 1100;
  let g = '';
  st.forEach((s, i) => {
    const x = 6 + i*(bw+gap);
    g += `<g><rect class="box" x="${x}" y="${y}" width="${bw}" height="${bh}" rx="8" data-go="${s[2]}"${tip('перейти к разделу')}></rect>
      <text class="strong" x="${x+bw/2}" y="${y+26}" text-anchor="middle" style="pointer-events:none">${esc(s[0])}</text>
      <text x="${x+bw/2}" y="${y+44}" text-anchor="middle" style="pointer-events:none">${esc(s[1])}</text></g>`;
    if (i < st.length-1) g += `<path class="arrow" d="M${x+bw+2},${y+bh/2} L${x+bw+gap-3},${y+bh/2}" marker-end="url(#ah)"/>`;
  });
  const cx = i => 6 + i*(bw+gap) + bw/2;
  g += `<path class="arrow" d="M${cx(7)},${y+bh+2} C${cx(7)},${y+bh+44} ${cx(4)},${y+bh+44} ${cx(4)},${y+bh+4}" marker-end="url(#ah)"/>
    <text x="${(cx(7)+cx(4))/2}" y="${y+bh+64}" text-anchor="middle" class="mute">выбранный токен возвращается на вход, и всё повторяется (старые токены берутся из кэша)</text>`;
  $('#s-pipeline').innerHTML = hdr('', 'Конвейер', 'Весь путь одной картинкой. Числа взяты из этого прогона; нажмите на блок, чтобы перейти к разделу.') +
    `<svg viewBox="0 0 ${W} 150" role="img" aria-label="Схема конвейера"><defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink3)"/></marker></defs>${g}</svg>`;
}

// ───────────── Токены ─────────────
function renderTokens(){
  const tpl = D.prompt.formatted.split(/(<\|[^|]*\|>|<\/?think>)/).map((p,i) => i%2 ? `<span class="sp">${esc(p)}</span>` : esc(p)).join('');
  const chips = toks.map((t,i) => `<button class="chip ${isSpecial(t)?'sp':''} ${S.tok===i?'on':''}" data-tok="${i}"${tip(`поз ${i} · id ${D.tokens.ids[i]}`)}>${vis(t)}<small>${D.tokens.ids[i]}</small></button>`).join('');
  const chars = D.prompt.formatted.length;
  $('#s-tokens').innerHTML = hdr('', 'Шаг 1–2. Шаблон чата и токены', 'Модель не видит ваш текст: сначала он оборачивается служебной разметкой (роли, пустой блок рассуждений), потом режется на токены, и каждому токену соответствует номер из словаря. Пробел перед словом входит в токен (показан как ␣).') +
    `<h3>Строка, которую получила модель</h3><pre>${tpl}</pre>
     <h3>${nTok} токенов <span class="muted small">— ${f(chars/nTok,1)} символа на токен; нажмите на токен, чтобы посмотреть его вектор ниже</span></h3>
     <div class="chips">${chips}</div>`;
}

// ───────────── Эмбеддинги ─────────────
function renderEmb(){
  const E = D.embeddings;
  if (!E) { $('#s-emb').innerHTML = hdr('', 'Шаг 3. Эмбеддинги', '') + empty('Нет данных об эмбеддингах.'); return; }
  const cur = E.tokens.find(t => t.pos === S.tok) || E.tokens.find(t => t.id === D.tokens.ids[S.tok]);
  let left = empty('Для этого токена вектор не сохранён (сохраняются первые 64 разных токена).');
  if (cur) {
    const v = cur.head, n = v.length, W = 560, H = 150, mid = 75, pad = 4, bw = (W-pad*2)/n;
    const mx = Math.max(...v.map(Math.abs)) || 1;
    let bars = `<line class="axis" x1="0" x2="${W}" y1="${mid}" y2="${mid}"/>`;
    v.forEach((x,i) => {
      const h = Math.abs(x)/mx*(mid-8), y = x>=0 ? mid-h : mid;
      bars += `<rect x="${pad+i*bw+1}" y="${y}" width="${bw-2}" height="${Math.max(h,0.5)}" rx="1.5" style="fill:var(${x>=0?'--blue':'--red'})"></rect>
        <rect class="hit" x="${pad+i*bw}" y="0" width="${bw}" height="${H}"${tip(`измерение ${i}\n${x}`)}></rect>`;
    });
    bars += `<text x="2" y="${H-2}">первые ${n} из ${f(E.dim,0)} чисел вектора · синий — положительные, красный — отрицательные</text>`;
    const nb = cur.neighbors.map(x => `<tr><td class="mono">${vis(x.token)}</td><td class="num">${x.cos.toFixed(3)}</td><td style="width:90px"><span class="bar" style="width:${clamp(x.cos)*80}px"></span></td></tr>`).join('');
    left = `<h3>Токен ${vis(cur.token)} <span class="muted small">(поз ${cur.pos}, id ${cur.id})</span></h3>
      <div class="tiles"><div class="tile"><b>${cur.norm.toFixed(3)}</b><span>длина вектора</span></div>
      <div class="tile"><b>${E.row_norm_mean.toFixed(3)}</b><span>средняя по словарю</span></div></div>
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Значения вектора эмбеддинга">${bars}</svg>
      <h3>Ближайшие токены словаря <span class="muted small">(косинус)</span></h3>
      <table><tr><th>токен</th><th class="num">косинус</th><th></th></tr>${nb}</table>`;
  }
  const k = Math.min(48, E.cos_matrix.length), cs = 11, pad = 92, top = 4, W = pad + k*cs + 4, H = top + k*cs + 4;
  let hm = '';
  for (let i=0;i<k;i++){
    hm += `<text x="${pad-4}" y="${top+i*cs+9}" text-anchor="end" class="mono" style="font-size:9px">${vis(clip(toks[i]||'',8))}</text>`;
    for (let j=0;j<k;j++){
      const v = E.cos_matrix[i][j];
      hm += `<rect x="${pad+j*cs}" y="${top+i*cs}" width="${cs-1}" height="${cs-1}" style="fill:${dvg(v)}"${tip(`${toks[i]} ↔ ${toks[j]}\nкосинус ${v}`)}></rect>`;
    }
  }
  $('#s-emb').innerHTML = hdr('', 'Шаг 3. Эмбеддинги: токен становится вектором', `Каждому токену соответствует строка из таблицы эмбеддингов: вектор из ${f(E.dim,0)} чисел. Он зависит только от токена, контекста модель пока не учитывала. Близкие по смыслу токены имеют близкие векторы (большой косинус).`) +
    `<div class="cols"><div>${left}</div><div><h3>Косинус между токенами промпта <span class="muted small">— до слоёв почти ничего общего, кроме одинаковых токенов</span></h3>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Матрица косинусов">${hm}</svg></div></div>`;
}

// ───────────── Слои ─────────────
function metricVal(r){
  if (S.metric === 'broken') return r.broken;
  if (S.metric === 'p') return 1 - r.p_orig;
  return r.kl;
}
const metricName = () => ({kl:'KL-расхождение', broken:'доля сломанных токенов', p:'1 − p(исх.)'}[S.metric]);
function mapRowFor(i){ return LM ? LM.layers.find(r => r.first_layer<=i && i<=r.last_layer) : null; }

function layersOverview(){
  const W = 1100, padL = 122, padR = 8, cw = (W-padL-padR)/NL, X = i => padL + i*cw;
  const R = residOf(), last = R ? R.norms[0].length-1 : 0;
  const normLast = R ? R.norms.map(r => r[last]) : null;
  let g = '', H = 372;
  const lab = (y, t) => `<text x="${padL-8}" y="${y}" text-anchor="end" class="strong">${t}</text>`;
  g += lab(35, 'тип слоя');
  TY.forEach((t,i) => {
    const sk = D.skipped.includes(i);
    g += `<rect class="t-${t}" x="${X(i)+1}" y="22" width="${cw-2}" height="18" rx="2" style="${sk?'opacity:.25':''}"></rect>`;
    if (t==='A') g += `<text x="${X(i)+cw/2}" y="16" text-anchor="middle" class="mute">${i}</text>`;
    if (sk) g += `<text x="${X(i)+cw/2}" y="36" text-anchor="middle" class="strong">✕</text>`;
  });
  // важность
  const y0 = 66, y1 = 146;
  g += lab(y0+12, LM ? 'важность' : 'важность') + `<text x="${padL-8}" y="${y0+26}" text-anchor="end" class="mute">${LM ? metricName() : ''}</text>`;
  if (LM) {
    const vm = Math.max(...LM.layers.map(metricVal)) || 1;
    g += `<line class="axis" x1="${padL}" x2="${W-padR}" y1="${y1}" y2="${y1}"/><text x="${padL-8}" y="${y1}" text-anchor="end" class="mute">0</text><text x="${padL-8}" y="${y0+4}" text-anchor="end" class="mute" style="display:none"></text>`;
    LM.layers.forEach(r => {
      const v = metricVal(r), h = v/vm*(y1-y0-6), w = (r.last_layer-r.first_layer+1)*cw-2;
      const cls = r.type.length===1 ? 't-'+r.type : 't-M';
      g += `<rect class="${cls}" x="${X(r.first_layer)+1}" y="${y1-h}" width="${w}" height="${Math.max(h,0.8)}" rx="1.5"></rect>`;
    });
    g += `<text x="${W-padR}" y="${y0-2}" text-anchor="end" class="mute">максимум ${f(vm,3)}</text>`;
  } else {
    g += `<text x="${padL}" y="${y0+40}" class="mute">нет данных — запустите trace_qwen.py с флагом --layer-map</text>`;
  }
  // норма
  const y2 = 190, y3 = 262;
  g += lab(y2+12, '‖h‖') + `<text x="${padL-8}" y="${y2+26}" text-anchor="end" class="mute">длина вектора</text><text x="${padL-8}" y="${y2+40}" text-anchor="end" class="mute">последнего токена</text>`;
  if (normLast) {
    const lo = Math.min(...normLast), hi = Math.max(...normLast), sc = v => y3 - (v-lo)/((hi-lo)||1)*(y3-y2-6);
    g += `<line class="axis" x1="${padL}" x2="${W-padR}" y1="${y3}" y2="${y3}"/>`;
    g += `<path class="line" d="${normLast.map((v,i)=>(i?'L':'M')+(X(i)+cw/2)+','+sc(v)).join(' ')}"/>`;
    normLast.forEach((v,i) => g += `<circle class="t-${TY[i]}" cx="${X(i)+cw/2}" cy="${sc(v)}" r="2.6" style="stroke:var(--card);stroke-width:1.5"></circle>`);
    g += `<text x="${W-padR}" y="${y2-2}" text-anchor="end" class="mute">от ${f(lo,1)} до ${f(hi,1)}</text>`;
  } else g += `<text x="${padL}" y="${y2+40}" class="mute">нужен --detail-steps 1 или больше</text>`;
  // изменение слоя
  const y4 = 286, y5 = 350;
  g += lab(y4+12, 'изменение') + `<text x="${padL-8}" y="${y4+26}" text-anchor="end" class="mute">‖Δh‖ / ‖h‖ слоя</text>`;
  if (R) {
    const vm = Math.max(...R.rel) || 1;
    g += `<line class="axis" x1="${padL}" x2="${W-padR}" y1="${y5}" y2="${y5}"/>`;
    R.rel.forEach((v,i) => { const h = v/vm*(y5-y4-6); g += `<rect class="t-${TY[i]}" x="${X(i)+1}" y="${y5-h}" width="${cw-2}" height="${Math.max(h,0.8)}" rx="1.5"></rect>`; });
    g += `<text x="${W-padR}" y="${y4-2}" text-anchor="end" class="mute">максимум ${pct(vm)}</text>`;
  }
  // попадание и выбор
  for (let i=0;i<NL;i++){
    const r = mapRowFor(i);
    const t = [`слой ${i} · ${typeName(TY[i])}`];
    if (D.skipped.includes(i)) t.push('ОТКЛЮЧЁН в этом прогоне');
    if (r) t.push(`важность: KL ${r.kl}, сломано ${pct(r.broken)}, p(исх.) ${r.p_orig}`);
    if (R) t.push(`‖h‖ после слоя ${f(normLast[i],2)}`, `изменение вектора ${pct(R.rel[i])}`, `cos(вход, выход) ${R.cos[i]}`);
    g += `<rect class="hit" x="${X(i)}" y="14" width="${cw}" height="${H-14-12}" tabindex="0" role="button" aria-label="слой ${i}" data-layer="${i}"${tip(t.join('\n'))}></rect>`;
    if (S.layer===i) g += `<rect class="selframe" x="${X(i)}" y="14" width="${cw}" height="${H-14-12}" rx="3"></rect>`;
  }
  const leg = `<div class="legend"><span><i style="background:var(--blue)"></i>линейное внимание (Gated DeltaNet)</span><span><i style="background:var(--orange)"></i>полное внимание</span><span class="muted">нажмите на слой, чтобы увидеть его веса и вычисления</span></div>`;
  return leg + `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Обзор слоёв">${g}</svg>`;
}

function moduleRows(i){
  const inv = D.weights ? D.weights.modules.filter(m => m.layer===i) : [];
  const calls = D.calls.filter(c => layerOfName(c.m)===i);
  const map = new Map();
  inv.forEach(m => map.set(m.m, {inv:m}));
  calls.forEach(c => { const r = map.get(c.m) || {}; r.call = c; map.set(c.m, r); });
  const rows = [...map.entries()].map(([name, r]) => ({name, ...r}));
  rows.forEach(r => {
    const ns = calls.filter(c => c.m===r.name || c.m.startsWith(r.name+'.')).map(c => c.n);
    r.first = ns.length ? Math.min(...ns) : 1e9; r.depth = r.name.split('.').length;
  });
  rows.sort((a,b) => a.first-b.first || a.depth-b.depth || (a.name<b.name?-1:1));
  return rows;
}

function layerDetail(){
  const i = S.layer;
  if (i == null) return `<div class="empty" style="margin-top:12px">Нажмите на любой слой на схеме: здесь появятся его тензоры весов, размеры, время и выход каждого модуля.</div>`;
  const t = TY[i], R = residOf(), r = mapRowFor(i);
  const rows = moduleRows(i);
  const inv = rows.filter(x => x.inv);
  const params = sum(inv.map(x => x.inv.params)), bytes = sum(inv.map(x => x.inv.bytes));
  const layerCall = D.calls.find(c => c.m === `language_model.model.layers.${i}`);
  let tiles = `<div class="tile"><b class="${''}">${t==='L'?'линейное':'полное'}</b><span>внимание</span></div>`;
  if (r) tiles += `<div class="tile"><b>${r.kl}</b><span>важность, KL${r.type.length>1?' (блок слоёв)':''}</span></div><div class="tile"><b>${pct(r.broken)}</b><span>токенов сломано без слоя</span></div><div class="tile"><b>${r.p_orig}</b><span>p(исх.) без слоя</span></div>`;
  if (R) tiles += `<div class="tile"><b>${f(R.norms[i][R.norms[i].length-1],2)}</b><span>‖h‖ после слоя</span></div><div class="tile"><b>${pct(R.rel[i])}</b><span>изменение вектора</span></div><div class="tile"><b>${R.cos[i]}</b><span>cos(вход, выход)</span></div>`;
  if (params) tiles += `<div class="tile"><b>${f(params/1e6,1)} млн</b><span>параметров слоя</span></div><div class="tile"><b>${f(bytes/1e6,1)} МБ</b><span>в памяти</span></div>`;
  if (layerCall) tiles += `<div class="tile"><b>${f(layerCall.ms,1)} мс</b><span>время слоя (с журналом)</span></div>`;
  const pre = `language_model.model.layers.${i}`;
  const trs = rows.map(x => {
    const rel = x.name === pre ? '(весь слой)' : x.name.slice(pre.length+1);
    const ind = x.name === pre ? 0 : rel.split('.').length - 1;
    const iv = x.inv, c = x.call;
    const shape = iv ? (iv.shape && iv.shape.length===2 ? `${iv.shape[0]}×${iv.shape[1]}${iv.bits ? ' · '+iv.bits+' бит' : ''}` : Object.entries(iv.tensors).map(([k,v]) => `${k} (${v.join(',')})`).join(', ')) : '';
    const ws = c && c.w && c.w.std!=null ? `σ=${c.w.std}, max|w|=${c.w.absmax}` : '';
    const out = c && c.o && c.o.shape ? `${c.o.shape.join('×')}  ‖h‖=${c.o.last_norm}  σ=${c.o.std}` : '';
    return `<tr><td class="mono" style="padding-left:${8+ind*14}px">${esc(rel)}</td><td>${esc(x.inv ? x.inv.c : c.c)}</td><td>${esc(shape)}</td><td class="num">${iv ? f(iv.params,0) : ''}</td><td class="num">${iv ? (iv.bytes/1e6).toFixed(2) : ''}</td><td class="num">${c ? f(c.ms,2) : ''}</td><td class="mono small">${esc(out)}</td><td class="small">${esc(ws)}</td></tr>`;
  }).join('');
  let extra = '';
  const lr = lensRows()[i];
  if (lr) extra += `<h3>Что модель «ответила бы» после этого слоя <span class="muted small">(logit lens, шаг ${S.lens})</span></h3><div class="chips">${lr.top.map(x => `<span class="chip" style="cursor:default">${vis(x.token)}<small>${pct(x.prob)}</small></span>`).join('')}</div><div class="muted small" style="margin-top:4px">итоговый токен: вероятность ${pct(lr.p_final)}, ранг ${lr.rank_final}</div>`;
  if (r && LM && LM.prompts.length) {
    const vm = Math.max(...r.kl_per_prompt, 1e-9);
    extra += `<h3>Как сильно слой нужен для каждого промпта <span class="muted small">(KL)</span></h3><table>` + LM.prompts.map((p,k) => `<tr><td title="${esc(p.prompt)}">${esc(clip(p.prompt,60))}</td><td class="num">${r.kl_per_prompt[k]}</td><td style="width:140px"><span class="bar" style="width:${r.kl_per_prompt[k]/vm*130}px"></span></td></tr>`).join('') + `</table>`;
  }
  const at = S.lens==null ? null : (D.attention[String(S.lens)]||[]).find(a => a.module.endsWith(`layers.${i}.self_attn`));
  if (at) extra += `<h3>Внимание этого слоя <span class="muted small">(последний токен → какие токены, шаг ${S.lens})</span></h3>${attnBars(at)}`;
  return `<h3 style="margin-top:22px">Слой ${i} · ${typeName(t)}${D.skipped.includes(i)?' · ОТКЛЮЧЁН':''}</h3><div class="tiles">${tiles}</div>
    <h3>Вычисления и веса (имена совпадают с trace.log)</h3>
    <div class="scroll"><table><tr><th>модуль</th><th>класс</th><th>матрица</th><th class="num">параметров</th><th class="num">МБ</th><th class="num">мс</th><th>выход</th><th>веса</th></tr>${trs}</table></div>${extra}`;
}

function attnBars(a){
  if (!a.dist) return empty('Контекст слишком длинный, распределение не сохранено.');
  const n = a.dist.length, W = 1100, H = 90, bw = W/n, mx = Math.max(...a.dist)||1;
  let g = `<line class="axis" x1="0" x2="${W}" y1="${H-14}" y2="${H-14}"/>`;
  a.dist.forEach((v,i) => {
    const h = v/mx*(H-24);
    g += `<rect class="t-A" x="${i*bw+1}" y="${H-14-h}" width="${Math.max(bw-2,1)}" height="${Math.max(h,0.5)}" rx="1"></rect><rect class="hit" x="${i*bw}" y="0" width="${bw}" height="${H}"${tip(`позиция ${i}: ${toks[i]!=null?toks[i]:''}\nвес ${pct(v)}`)}></rect>`;
  });
  g += `<text x="0" y="${H-2}">позиции 0…${n-1} · среднее по ${a.heads} головам · энтропия ${f(a.entropy,2)}</text>`;
  const ht = a.head_top.map((h,k) => `<tr><td>голова ${k}</td><td class="mono">${vis(toks[h[0]]||'')}</td><td class="num">${h[0]}</td><td class="num">${pct(h[1])}</td></tr>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg><details><summary>на что смотрит каждая голова (самый сильный токен)</summary><div class="scroll"><table><tr><th>голова</th><th>токен</th><th class="num">позиция</th><th class="num">вес</th></tr>${ht}</table></div></details>`;
}

function renderLayers(){
  $('#s-layers').innerHTML = hdr('', `Шаг 4. Слои: ${NL} преобразований вектора`, 'Вектор каждого токена проходит все слои по очереди. В каждом слое внимание берёт информацию из других токенов, а MLP перерабатывает результат; итог прибавляется к вектору. Тип слоя, важность (если построена карта), длина вектора и величина изменения видны на одной оси.') + layersOverview() + layerDetail();
}

// ───────────── Вектор по слоям ─────────────
function renderResid(){
  const R = residOf();
  if (!R) { $('#s-resid').innerHTML = hdr('','Вектор токена по слоям','') + empty('Нет данных: запустите с --detail-steps 1 или больше.'); return; }
  const nr = R.norms[0].length, rows = Math.min(nr, 64), off = nr-rows;
  const tk = toks.slice(Math.max(0, toks.length-nr)); // подписи для последних nr позиций
  const lab = p => (S.lens===0 ? (tk[p+off]||'') : (S.lens>0 ? (R.final_token||'') : ''));
  const lg = []; R.norms.forEach(r => r.slice(off).forEach(v => { if (v>0) lg.push(Math.log10(v)); }));
  const lo = Math.min(...lg), hi = Math.max(...lg);
  const padL = 92, padR = 8, W = 1100, cw = (W-padL-padR)/NL, ch = 13, H = rows*ch + 30;
  let g = '';
  for (let p=0;p<rows;p++){
    const label = S.lens===0 ? tk[p+off] : '';
    g += `<text x="${padL-6}" y="${p*ch+10}" text-anchor="end" class="mono" style="font-size:10px">${vis(clip(label||(`поз ${p+off}`),12))}</text>`;
    for (let i=0;i<NL;i++){
      const v = R.norms[i][p+off];
      g += `<rect x="${padL+i*cw}" y="${p*ch}" width="${cw-1}" height="${ch-1}" style="fill:${seq((Math.log10(Math.max(v,1e-6))-lo)/((hi-lo)||1))}"${tip(`слой ${i} · ${label||('поз '+(p+off))}\nдлина вектора ${v}`)}></rect>`;
    }
  }
  for (let i=3;i<NL;i+=4) g += `<text x="${padL+i*cw+cw/2}" y="${rows*ch+12}" text-anchor="middle" class="mute">${i}</text>`;
  g += `<text x="${padL}" y="${rows*ch+26}">слой →   · цвет = длина вектора токена после слоя (логарифмическая шкала от ${f(Math.pow(10,lo),1)} до ${f(Math.pow(10,hi),0)})</text>`;
  $('#s-resid').innerHTML = hdr('', 'Вектор токена по слоям', 'Длина вектора каждого токена после каждого слоя. Обычно первый токен («якорь внимания») получает очень длинный вектор, а остальные растут постепенно: так модель накапливает информацию в одном и том же векторе.') + stepPicker() +
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Тепловая карта длин векторов">${g}</svg>`;
}

function stepPicker(){
  if (lensSteps.length < 2) return '';
  return `<div class="legend">шаг генерации: ${lensSteps.map(s => `<button class="${S.lens===s?'on':''}" data-lens="${s}">${s===0?'0 (промпт)':s}</button>`).join(' ')}</div>`;
}

// ───────────── Logit lens ─────────────
function renderLens(){
  const L = lensRows();
  if (!L.length) { $('#s-lens').innerHTML = hdr('','Logit lens','') + empty('Нет данных: запустите с --detail-steps 1 или больше.'); return; }
  const R = residOf(), fin = R ? R.final_token : '';
  let settle = null;
  for (let i=NL-1;i>=0;i--){ if (L[i].rank_final===1) settle = i; else break; }
  const top10 = L.findIndex(r => r.rank_final<=10);
  const W = 1100, padL = 60, padR = 8, cw = (W-padL-padR)/NL, X = i => padL+i*cw;
  let g = '', y0 = 14, y1 = 104;
  g += `<text x="${padL-6}" y="${y0+8}" text-anchor="end" class="strong">p итога</text><line class="axis" x1="${padL}" x2="${W-padR}" y1="${y1}" y2="${y1}"/>`;
  g += `<text x="${padL-6}" y="${y1}" text-anchor="end" class="mute">0</text><text x="${padL-6}" y="${y0+20}" text-anchor="end" class="mute">1</text>`;
  L.forEach((r,i) => {
    const h = r.p_final*(y1-y0-4);
    g += `<rect class="t-${TY[i]}" x="${X(i)+1}" y="${y1-h}" width="${cw-2}" height="${Math.max(h,0.8)}" rx="1.5"></rect>
      <rect class="hit" x="${X(i)}" y="${y0}" width="${cw}" height="${y1-y0+60}" data-layer="${i}"${tip(`слой ${i}\nвероятность итогового токена ${pct(r.p_final)}\nранг ${r.rank_final}\nэнтропия ${r.entropy}`)}></rect>`;
  });
  const y2 = 120, y3 = 170, emax = Math.max(...L.map(r=>r.entropy)) || 1;
  g += `<text x="${padL-6}" y="${y2+8}" text-anchor="end" class="strong">энтропия</text><line class="axis" x1="${padL}" x2="${W-padR}" y1="${y3}" y2="${y3}"/><text x="${padL-6}" y="${y3}" text-anchor="end" class="mute">0</text><text x="${padL-6}" y="${y2+20}" text-anchor="end" class="mute">${f(emax,1)}</text>`;
  g += `<path class="line" d="${L.map((r,i)=>(i?'L':'M')+(X(i)+cw/2)+','+(y3-r.entropy/emax*(y3-y2-4))).join(' ')}"/>`;
  for (let i=3;i<NL;i+=4) g += `<text x="${X(i)+cw/2}" y="${y3+13}" text-anchor="middle" class="mute">${i}</text>`;
  if (settle!=null) g += `<line x1="${X(settle)}" x2="${X(settle)}" y1="${y0}" y2="${y3}" class="hair"/>`;
  const tuned = L.some(r => r.tuned_p_final!=null), hasKl = L.some(r => r.kl_final!=null);
  if (tuned) g += `<path d="${L.map((r,i)=>(i?'L':'M')+(X(i)+cw/2)+','+(y1-(r.tuned_p_final||0)*(y1-y0-4))).join(' ')}" style="fill:none;stroke:var(--violet);stroke-width:2"/>`;
  let tsettle = null;
  if (tuned) for (let i=NL-1;i>=0;i--){ if (L[i].tuned_rank===1) tsettle = i; else break; }
  const trs = L.map((r,i) => `<tr class="${i===settle?'hl':''}"><td class="num">${i}</td><td>${TY[i]==='L'?'лин.':'<b>полн.</b>'}</td><td>${r.top.slice(0,3).map(x=>`<span class="mono">${vis(x.token)}</span> <span class="muted">${pct(x.prob)}</span>`).join(' &nbsp; ')}</td><td class="num">${pct(r.p_final)}</td><td class="num">${f(r.rank_final,0)}</td>${hasKl?`<td class="num">${f(r.kl_final,2)}</td>`:''}${tuned?`<td><span class="mono">${vis(r.tuned_top[0].token)}</span> <span class="muted">${pct(r.tuned_top[0].prob)}</span></td><td class="num">${f(r.tuned_rank,0)}</td><td class="num">${f(r.tuned_kl,2)}</td>`:''}</tr>`).join('');
  const msg = settle==null ? 'итоговый токен до конца не становится самым вероятным' : (settle===NL-1 ? `итоговый токен становится самым вероятным только на последнем слое` : `итоговый токен становится самым вероятным со слоя <b>${settle}</b> и остаётся им до конца`);
  $('#s-lens').innerHTML = hdr('', 'Шаг 5. Logit lens: когда «созревает» ответ', 'Если остановить модель после любого слоя, пропустить вектор через финальную норму и lm_head, получится «промежуточный ответ». Так видно, как догадка превращается в итог: от пробелов и знаков препинания до конкретного слова.') + stepPicker() +
    `<div class="tiles"><div class="tile"><b class="mono">${vis(fin)}</b><span>итоговый токен</span></div><div class="tile"><b>${settle==null?'—':settle}</b><span>слой, с которого он лидирует</span></div><div class="tile"><b>${top10<0?'—':top10}</b><span>слой, где он попал в топ-10</span></div>${tuned?`<div class="tile"><b>${tsettle==null?'—':tsettle}</b><span>слой, с которого он лидирует по tuned lens</span></div>`:''}</div>
     <p class="lead">${msg}.</p>
     <div class="note">${tuned ? 'Фиолетовая линия — tuned lens: для каждого слоя обучено преобразование в базис последнего слоя, поэтому ей можно верить и в ранних слоях. Столбцы — обычный logit lens.' : 'В ранних слоях logit lens систематически занижает готовность ответа: их векторы ещё не в том базисе, который читает lm_head. Столбец «KL до итога» в таблице показывает, насколько промежуточное распределение далеко от настоящего. Точнее — tuned lens: обучите её через tuned_lens.py и передайте в --tuned-lens.'}</div>
     <svg viewBox="0 0 ${W} ${y3+20}" role="img" aria-label="Вероятность итогового токена по слоям">${g}</svg>
     <div class="legend"><span><i style="background:var(--blue)"></i>линейный слой</span><span><i style="background:var(--orange)"></i>полное внимание</span>${tuned?'<span><i style="background:var(--violet)"></i>tuned lens</span>':''}</div>
     <details><summary>таблица по слоям (топ-3 токена, вероятность и ранг итогового)</summary><div class="scroll"><table><tr><th class="num">слой</th><th>тип</th><th>топ-3 после слоя</th><th class="num">p итога</th><th class="num">ранг итога</th>${hasKl?'<th class="num">KL до итога</th>':''}${tuned?'<th>tuned lens: лучший</th><th class="num">ранг итога</th><th class="num">KL</th>':''}</tr>${trs}</table></div></details>`;
}

// ───────────── Внимание ─────────────
function renderAttn(){
  const A = S.lens==null ? [] : (D.attention[String(S.lens)] || []);
  if (!A.length) { $('#s-attn').innerHTML = hdr('','Внимание','') + empty('Нет данных о внимании: запустите с --detail-steps 1 или больше.'); return; }
  const rows = A.filter(a => a.dist).map(a => ({a, layer: layerOfName(a.module)})).sort((x,y)=>x.layer-y.layer);
  if (!rows.length) { $('#s-attn').innerHTML = hdr('','Внимание','') + empty('Контекст слишком длинный для карты внимания.'); return; }
  const n = rows[0].a.dist.length, padL = 70, padB = 70, W = 1100, cw = Math.min(40, (W-padL-10)/n), ch = 18, H = rows.length*ch + padB;
  const labelEvery = n>60 ? Math.ceil(n/60) : 1;
  let g = '';
  rows.forEach((r,k) => {
    g += `<text x="${padL-8}" y="${k*ch+13}" text-anchor="end" class="strong" data-layer="${r.layer}" style="cursor:pointer">слой ${r.layer}</text>`;
    r.a.dist.forEach((v,j) => {
      g += `<rect x="${padL+j*cw}" y="${k*ch}" width="${cw-1}" height="${ch-1}" style="fill:${seq(Math.sqrt(v))}"${tip(`слой ${r.layer} · смотрит на «${(toks[j]||'').replace(/\n/g,'↵')}» (поз ${j})\nвес ${pct(v)}`)}></rect>`;
    });
    if (S.layer===r.layer) g += `<rect class="selframe" x="${padL-2}" y="${k*ch-1}" width="${n*cw+2}" height="${ch+1}"></rect>`;
  });
  for (let j=0;j<n;j+=labelEvery) g += `<text transform="translate(${padL+j*cw+cw/2},${rows.length*ch+8}) rotate(60)" class="mono" style="font-size:10px">${vis(clip(toks[j]||'',10))}</text>`;
  $('#s-attn').innerHTML = hdr('', 'Внимание: на какие токены смотрит последний токен', `В ${TY.filter(t=>t==='A').length} слоях из ${NL} есть полное внимание: последний токен сравнивается со всеми предыдущими и берёт информацию из тех, что важнее. Строка — слой, столбец — токен контекста, цвет — вес (усреднён по головам; шкала корневая, чтобы были видны слабые связи). Нажмите на строку, чтобы открыть слой.`) + stepPicker() +
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Карта внимания">${g}</svg>
     ${headsBlock(rows)}
     <p class="muted small">В остальных ${TY.filter(t=>t==='L').length} слоях внимание линейное: карты «кто на кого смотрит» у них нет, информация копится во внутреннем состоянии. Что на них влияет, показывает раздел «Вклад токенов».</p>`;
}

// ───────────── Выбор токена ─────────────
function renderSample(){
  const smp = D.samples;
  if (!smp.length) { $('#s-sample').innerHTML = hdr('','Выбор токена','') + empty('Нет данных.'); return; }
  const k = Math.min(S.sstep, smp.length-1), s = smp[k];
  const W = 1100, padL = 46, padR = 8, n = smp.length, bw = Math.min(56,(W-padL-padR)/n), H = 150;
  let g = `<line class="axis" x1="${padL}" x2="${W-padR}" y1="90" y2="90"/><text x="${padL-6}" y="30" text-anchor="end" class="mute">1</text><text x="${padL-6}" y="90" text-anchor="end" class="mute">0</text><text x="${padL-6}" y="22" text-anchor="end" class="strong">p</text>`;
  const emax = Math.max(...smp.map(x=>x.entropy)) || 1;
  smp.forEach((x,i) => {
    const h = x.chosen.prob*58, X = padL+i*bw;
    g += `<rect x="${X+1}" y="${90-h}" width="${bw-2}" height="${Math.max(h,0.8)}" rx="1.5" style="fill:var(--blue);${i===k?'':'opacity:.8'}"></rect>
      <text x="${X+bw/2}" y="${90+12}" text-anchor="middle" class="mono" style="font-size:10px">${vis(clip(x.chosen.token,4))}</text>
      <rect class="hit" x="${X}" y="10" width="${bw}" height="${H-10}" data-step="${i}" tabindex="0" role="button"${tip(`шаг ${i}: ${x.chosen.token.replace(/\n/g,'↵')}\np ${pct(x.chosen.prob)}\nэнтропия ${x.entropy}\nтокенов в nucleus(0.95): ${x.nucleus95}`)}></rect>`;
    if (i===k) g += `<rect class="selframe" x="${X}" y="10" width="${bw}" height="${H-10-2}" rx="3"></rect>`;
  });
  g += `<path class="line" d="${smp.map((x,i)=>(i?'L':'M')+(padL+i*bw+bw/2)+','+(146-x.entropy/emax*28)).join(' ')}"/><text x="${padL-6}" y="130" text-anchor="end" class="strong">энтропия</text>`;
  const cand = s.candidates, cmx = Math.max(...cand.map(c=>c.prob)) || 1;
  const cr = cand.map((c,i) => `<tr class="${i===0?'hl':''}"><td class="mono">${vis(c.token)}</td><td class="num">${c.logit}</td><td class="num">${pct(c.prob)}</td><td style="width:220px"><span class="bar" style="width:${c.prob/cmx*200}px"></span></td></tr>`).join('');
  $('#s-sample').innerHTML = hdr('', 'Шаг 6–7. Как выбирается каждый токен ответа', `Последний слой даёт вектор последнего токена, lm_head превращает его в ${f(M.vocab,0)} оценок (логитов), softmax делает из них вероятности, и выбирается один токен. ${MP.temperature>0 ? `В этом прогоне токен выбирался случайно (temperature ${MP.temperature}, top_p ${MP.top_p}, seed ${MP.seed}), поэтому выбранный токен не всегда самый вероятный.` : 'Здесь выбирается самый вероятный; LM Studio и oMLX по умолчанию сэмплируют случайно из лучших, поэтому их ответы могут отличаться.'}`) +
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Вероятность выбранных токенов по шагам">${g}</svg>
     <div class="legend muted">столбец — вероятность выбранного токена, линия — энтропия распределения (чем выше, тем больше модель сомневалась). Нажмите на столбец.</div>
     <h3>Шаг ${k}: кандидаты <span class="muted small">(${f(s.ms,0)} мс)</span></h3>
     <div class="tiles"><div class="tile"><b>${s.entropy}</b><span>энтропия</span></div><div class="tile"><b>${f(s.nucleus95,0)}</b><span>токенов в nucleus (95% массы)</span></div><div class="tile"><b>${pct(s.top20_mass)}</b><span>масса топ-20</span></div></div>
     <table><tr><th>токен</th><th class="num">логит</th><th class="num">вероятность</th><th></th></tr>${cr}</table>
     <p class="muted small">Если nucleus большой (сотни токенов), при случайном сэмплировании модель могла бы выбрать много разных продолжений; если равен 1–2, выбор почти предопределён.</p>`;
}

// ───────────── Карта важности ─────────────
function renderMap(){
  if (!LM && LMX) { $('#s-map').innerHTML = hdr('', 'Карта важности слоёв', 'Показывает, какие слои модель не может потерять без последствий.') + empty('Построена только быстрая карта (--fast, без перебора слоёв) — она в следующем разделе. Для полной карты с доверительными интервалами запустите layer_map.py без --fast.'); return; }
  if (!LM) { $('#s-map').innerHTML = hdr('', 'Карта важности слоёв', 'Показывает, какие слои модель не может потерять без последствий.') + empty('Карта не построена. Запустите trace_qwen.py с флагом --layer-map (или layer_map.py --run-dir этой папки) и пересоберите отчёт.'); return; }
  const rows = LM.layers, ci = S.metric==='kl' && rows[0].kl_hi!=null;
  const vm = Math.max(...rows.map(r => ci ? r.kl_hi : metricVal(r))) || 1;
  const W = 1100, padL = 60, padR = 8, cw = (W-padL-padR)/rows.length, y1 = 150, y0 = 16;
  let g = `<line class="axis" x1="${padL}" x2="${W-padR}" y1="${y1}" y2="${y1}"/><text x="${padL-6}" y="${y1}" text-anchor="end" class="mute">0</text><text x="${padL-6}" y="${y0+8}" text-anchor="end" class="mute">${f(vm,3)}</text>`;
  rows.forEach((r,i) => {
    const v = metricVal(r), h = v/vm*(y1-y0-6), cls = r.type.length===1 ? 't-'+r.type : 't-M';
    g += `<rect class="${cls}" x="${padL+i*cw+1}" y="${y1-h}" width="${Math.max(cw-2,1)}" height="${Math.max(h,0.8)}" rx="1.5"></rect>
      <rect class="hit" x="${padL+i*cw}" y="${y0}" width="${cw}" height="${y1-y0+20}" tabindex="0" data-layer="${r.first_layer}"${tip(`слой ${r.layers} (${r.type})\nKL ${r.kl}${r.kl_hi!=null ? ' [' + r.kl_lo + '; ' + r.kl_hi + ']' : ''}\nсломано ${pct(r.broken)}\np(исх.) ${r.p_orig}${r.dlogp!=null ? '\nΔlog p ' + r.dlogp : ''}${r.atp!=null ? '\nоценка по градиенту ' + r.atp : ''}`)}></rect>`;
    if (ci) { const xc = padL+i*cw+cw/2, sc = (y1-y0-6)/vm; g += `<line x1="${xc}" x2="${xc}" y1="${y1-r.kl_hi*sc}" y2="${y1-r.kl_lo*sc}" style="stroke:var(--ink);stroke-width:1.5;pointer-events:none"/>`; }
    if (rows.length<=64 && (r.first_layer%4===3 || rows.length<=16)) g += `<text x="${padL+i*cw+cw/2}" y="${y1+14}" text-anchor="middle" class="mute">${r.layers}</text>`;
  });
  // тепловая карта промптов
  const pk = S.metric==='broken' ? 'broken_per_prompt' : 'kl_per_prompt';
  const hp = LM.prompts.length, ch = 18, ty = 6, padL2 = 250;
  const cw2 = (W-padL2-padR)/rows.length;
  const hv = Math.max(...rows.map(r => Math.max(...r[pk]))) || 1;
  let hm = '';
  LM.prompts.forEach((p,k) => {
    hm += `<text x="${padL2-6}" y="${ty+k*ch+13}" text-anchor="end"${tip(p.prompt+'\n→ '+p.answer)}>${esc(clip(p.prompt,40))}</text>`;
    rows.forEach((r,i) => {
      const v = r[pk][k];
      hm += `<rect x="${padL2+i*cw2}" y="${ty+k*ch}" width="${Math.max(cw2-1,1)}" height="${ch-1}" style="fill:${seq(Math.sqrt(v/hv))}" data-layer="${r.first_layer}"${tip(`${p.prompt}\nслой ${r.layers}: ${S.metric==='broken'?'сломано '+pct(v):'KL '+v}`)}></rect>`;
    });
  });
  const Hh = ty + hp*ch + 8;
  const rk = [...rows].sort((a,b)=>metricVal(b)-metricVal(a)), nn = Math.min(8, rk.length);
  const hasCi = rows[0].kl_hi!=null, iv = r => hasCi ? `<td class="num muted">${r.kl_lo} – ${r.kl_hi}</td>` : '';
  const lst = a => a.map(r => `<tr><td>слой ${r.layers} <span class="muted">(${r.type})</span></td><td class="num">${r.kl}</td>${iv(r)}<td class="num">${pct(r.broken)}</td></tr>`).join('');
  const tbl = rows.map(r => `<tr><td>${r.layers}</td><td>${r.type}</td><td class="num">${r.kl}</td>${iv(r)}<td class="num">${pct(r.broken)}</td><td class="num">${r.p_orig}</td></tr>`).join('');
  // наименее важные — по верхней границе интервала: малое среднее при широком интервале ничего не доказывает
  const safe = hasCi ? [...rows].sort((a,b)=>a.kl_hi-b.kl_hi).slice(0,nn) : rk.slice(-nn).reverse();
  const overlap = hasCi && safe.length>1 && safe[0].kl_hi >= safe[safe.length-1].kl_lo;
  const few = LM.prompts.length < 20 ? `<div class="note">Промптов всего ${LM.prompts.length}: интервалы широкие, выводы предварительные. Для надёжной карты используйте --map-prompts prompts/extended.txt.</div>` : '';
  const ih = hasCi ? '<th class="num">95% интервал</th>' : '';
  $('#s-map').innerHTML = hdr('', 'Карта важности слоёв', `Каждый слой по очереди отключается (его выход заменяется входом), и измеряется, как изменились ответы на ${LM.prompts.length} промптов (по ${LM.tokens} токенов). Высокий столбец — без слоя модель заметно хуже; почти нулевой — слой для этих задач почти не нужен.`) +
    `<div class="legend">метрика: ${[['kl','KL'],['broken','сломано'],['p','1 − p(исх.)']].map(([k,t]) => `<button class="${S.metric===k?'on':''}" data-metric="${k}">${t}</button>`).join(' ')}
      <span><i style="background:var(--blue)"></i>линейный</span><span><i style="background:var(--orange)"></i>полное внимание</span>${ci?'<span class="muted">чёрная черта — 95% доверительный интервал (бутстреп по промптам)</span>':''}<span class="muted">нажмите на столбец или клетку — откроется слой</span></div>
    <svg viewBox="0 0 ${W} ${y1+24}" role="img" aria-label="Важность слоёв">${g}</svg>
    <h3>По промптам <span class="muted small">— видно, какие слои нужны только для отдельных задач</span></h3>
    <svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Важность слоёв по промптам">${hm}</svg>
    ${few}<div class="cols"><div><h3>Самые важные</h3><table><tr><th></th><th class="num">KL</th>${ih}<th class="num">сломано</th></tr>${lst(rk.slice(0,nn))}</table></div>
    <div><h3>Наименее важные <span class="muted small">(${hasCi?'по верхней границе интервала; ':''}кандидаты на удаление)</span></h3><table><tr><th></th><th class="num">KL</th>${ih}<th class="num">сломано</th></tr>${lst(safe)}</table>${overlap?'<p class="muted small">Интервалы этих слоёв пересекаются: порядок внутри списка статистически не значим.</p>':''}</div></div>
    <div class="note">Низкий KL не гарантирует, что слой можно удалить: слои работают вместе, и несколько «ненужных» слоёв, отключённых сразу, могут сломать модель сильнее, чем каждый по отдельности. Какой блок удалять целиком, подсказывает угловое расстояние в следующем разделе. Карта верна только для этих промптов.</div>
    <details><summary>таблица</summary><div class="scroll"><table><tr><th>слой</th><th>тип</th><th class="num">KL</th>${ih}<th class="num">сломано</th><th class="num">p(исх.)</th></tr>${tbl}</table></div></details>`;
}

// ───────────── Общий рисовальщик: столбцы вверх и вниз от нуля ─────────────
function divBars(series, opt){
  const n = series[0].vals.length;
  let vmax = 0, vmin = 0;
  for (let i=0;i<n;i++){ vmax = Math.max(vmax, sum(series.map(s=>Math.max(s.vals[i],0)))); vmin = Math.min(vmin, sum(series.map(s=>Math.min(s.vals[i],0)))); }
  const span = (vmax-vmin) || 1, W = 1100, padL = 60, padR = 8, cw = (W-padL-padR)/n, y0 = 14, H = opt.h || 170, Y = v => y0+(vmax-v)/span*H;
  let g = `<line class="axis" x1="${padL}" x2="${W-padR}" y1="${Y(0)}" y2="${Y(0)}"/><text x="${padL-6}" y="${y0+4}" text-anchor="end" class="mute">${f(vmax,2)}</text><text x="${padL-6}" y="${y0+H+4}" text-anchor="end" class="mute">${f(vmin,2)}</text>`;
  for (let i=0;i<n;i++){
    let py = Y(0), ny = Y(0); const x = padL+i*cw;
    series.forEach(s => { const v = s.vals[i], h = Math.abs(v)/span*H; if (!(h>0)) return;
      if (v>=0) { py -= h; g += `<rect x="${x+1}" y="${py}" width="${Math.max(cw-2,1)}" height="${h}" rx="1" style="fill:var(${s.col})"></rect>`; }
      else { g += `<rect x="${x+1}" y="${ny}" width="${Math.max(cw-2,1)}" height="${h}" rx="1" style="fill:var(${s.col})"></rect>`; ny += h; } });
    g += `<rect class="hit" x="${x}" y="${y0}" width="${cw}" height="${H}" tabindex="0" data-layer="${opt.layerOf ? opt.layerOf(i) : i}"${tip(opt.tip(i))}></rect>`;
    if (n<=16 || i%4===3) g += `<text x="${x+cw/2}" y="${y0+H+16}" text-anchor="middle" class="mute">${opt.name ? opt.name(i) : i}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${y0+H+22}" role="img" aria-label="${esc(opt.label)}">${g}</svg>`;
}
const sgn = (x,d=3) => x==null ? '—' : (x>0?'+':'') + f(x,d);

// ───────────── Разложение логита по слоям ─────────────
function renderDla(){
  const sec = $('#s-dla'), A = S.lens==null ? null : D.dla[String(S.lens)];
  if (!A) { sec.innerHTML = hdr('', 'Разложение логита по слоям', '') + empty('Нет данных: запустите с --detail-steps 1 или больше.'); return; }
  const att = Array(NL).fill(0), mlp = Array(NL).fill(0); let emb = 0;
  A.components.forEach(c => { if (c.kind==='emb') emb = c.diff; else if (c.kind==='attn') att[c.layer] = c.diff; else mlp[c.layer] = c.diff; });
  const total = emb + sum(att) + sum(mlp);
  const bars = divBars([{col:'--violet', vals:att}, {col:'--aqua', vals:mlp}], {label:'Вклад слоёв в разность логитов',
    tip: i => `слой ${i} (${TY[i]==='L'?'линейный':'полное внимание'})\nвнимание ${sgn(att[i])}\nMLP ${sgn(mlp[i])}\nвсего ${sgn(att[i]+mlp[i])}`});
  // накопленная разность: как преимущество выбранного токена росло от слоя к слою
  const cum = []; let run = emb; for (let i=0;i<NL;i++){ run += att[i]+mlp[i]; cum.push(run); }
  const cmx = Math.max(...cum, emb, 0), cmn = Math.min(...cum, emb, 0), cs = (cmx-cmn)||1, W = 1100, padL = 60, cw = (W-68)/NL, Yc = v => 10+(cmx-v)/cs*70;
  const line = `<svg viewBox="0 0 ${W} 96" role="img" aria-label="Накопленная разность логитов"><line class="axis" x1="${padL}" x2="${W-8}" y1="${Yc(0)}" y2="${Yc(0)}"/><text x="${padL-6}" y="14" text-anchor="end" class="mute">${f(cmx,1)}</text><text x="${padL-6}" y="84" text-anchor="end" class="mute">${f(cmn,1)}</text><path class="line" d="${cum.map((v,i)=>(i?'L':'M')+(padL+i*cw+cw/2)+','+Yc(v)).join(' ')}"/>${cum.map((v,i)=>`<rect class="hit" x="${padL+i*cw}" y="6" width="${cw}" height="80" data-layer="${i}"${tip(`после слоя ${i}: разность ${sgn(v)}`)}></rect>`).join('')}</svg>`;
  const nm = {emb:'эмбеддинг', attn:'внимание', mlp:'MLP'};
  const top = [...A.components].sort((a,b)=>Math.abs(b.diff)-Math.abs(a.diff)).slice(0,10), tm = Math.abs(top[0].diff) || 1;
  const trs = top.map(c => `<tr><td>${c.layer==null ? nm[c.kind] : `слой ${c.layer} · ${nm[c.kind]}`}</td><td class="num">${sgn(c.diff)}</td><td style="width:240px"><span class="bar" style="width:${Math.abs(c.diff)/tm*220}px;background:var(${c.diff>=0?'--blue':'--red'})"></span></td><td class="num">${sgn(c.logit)}</td></tr>`).join('');
  sec.innerHTML = hdr('', 'Разложение логита: какие слои привели к ответу', `Выход модели — сумма записей в «остаточный поток»: эмбеддинг плюс вклад внимания и MLP каждого слоя. Поэтому преимущество выбранного токена над главным соперником раскладывается по слоям точно, без дополнительных прогонов. Столбец вверх — слой подталкивал к выбранному токену, вниз — к сопернику.`) + stepPicker() +
    `<div class="tiles"><div class="tile"><b class="mono">${vis(A.target.token)}</b><span>выбранный токен</span></div><div class="tile"><b class="mono">${vis(A.other.token)}</b><span>главный соперник</span></div><div class="tile"><b>${sgn(total,2)}</b><span>разность логитов (сумма вкладов)</span></div><div class="tile"><b>${sgn(emb,2)}</b><span>из них эмбеддинг</span></div></div>
     ${bars}
     <div class="legend"><span><i style="background:var(--violet)"></i>внимание (полное или линейное)</span><span><i style="background:var(--aqua)"></i>MLP</span></div>
     <h3>Накопленное преимущество <span class="muted small">— где ответ «решился»</span></h3>${line}
     <h3>Самые влиятельные компоненты</h3><table><tr><th>компонента</th><th class="num">вклад в разность</th><th></th><th class="num">вклад в логит токена</th></tr>${trs}</table>
     <p class="muted small">Проверка: сумма вкладов в логит ${f(A.logit_sum,3)}, настоящий логит ${f(A.logit_actual,3)}. Разложение показывает прямой вклад в выход; косвенное влияние слоя через следующие слои видно в карте важности и патчинге.</p>`;
}

// ───────────── Вклад токенов контекста ─────────────
function renderTokAttr(){
  const sec = $('#s-tokattr'), A = S.lens==null ? null : D.token_attrs[String(S.lens)];
  if (!A) { sec.innerHTML = hdr('', 'Вклад токенов контекста', '') + empty('Нет данных: прогон был с --no-token-attr, этот шаг не подробный (--detail-steps) или градиенты посчитать не удалось (причина — в trace.log).'); return; }
  const T = A.tokens || toks, np = A.n_prompt != null ? A.n_prompt : A.scores.length, st = A.step || 0;
  const mxv = Math.max(...A.scores.map(Math.abs)) || 1, tot = sum(A.scores.map(Math.abs)) || 1;
  const own = sum(A.scores.slice(np).map(Math.abs)) / tot;
  const chips = A.scores.map((v,i) => `<span class="chip${isSpecial(T[i]||'')?' sp':''}${i>=np?' ans':''}"${i<np?` data-tok="${i}"`:''} style="background:${dvg(v/mxv*0.75)}"${tip(`${i>=np?'токен ответа':'токен промпта'}, поз ${i}: ${(T[i]||'').replace(/\n/g,'↵')}\nвклад ${sgn(v,4)}`)}>${vis(T[i]||'')}<small>${sgn(v,2)}</small></span>`).join('');
  const top = A.scores.map((v,i)=>({v,i})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,8);
  const trs = top.map(r => `<tr><td class="num">${r.i}</td><td class="mono">${vis(T[r.i]||'')}</td><td>${r.i>=np?'ответ':'промпт'}</td><td class="num">${sgn(r.v,4)}</td><td style="width:240px"><span class="bar" style="width:${Math.abs(r.v)/mxv*220}px;background:var(${r.v>=0?'--blue':'--red'})"></span></td></tr>`).join('');
  const gap = A.completeness_gap!=null ? `<p class="muted small">Проверка метода: сумма вкладов расходится с изменением метрики на ${f(A.completeness_gap,3)}. Если это много, увеличьте --ig-steps.</p>` : '';
  const ctx = np < A.scores.length ? `На шаге ${st} контекст — это ${np} токенов промпта и ${A.scores.length-np} уже выбранных токенов ответа (фиолетовая рамка). ` : '';
  sec.innerHTML = hdr('', 'Вклад токенов контекста в выбор токена', `Карта внимания есть только у слоёв полного внимания, а их ${TY.filter(t=>t==='A').length} из ${NL}. Этот метод работает через градиент и учитывает все слои: показывает, какие токены подтолкнули модель выбрать <span class="mono">${vis(A.target.token)}</span>, а не <span class="mono">${vis(A.other.token)}</span>. ${ctx}`) + stepPicker() +
    (np < A.scores.length ? `<div class="tiles"><div class="tile"><b>${pct(own)}</b><span>вклада — на уже сгенерированные токены</span></div><div class="tile"><b>${pct(1-own)}</b><span>вклада — на промпт</span></div></div>` : '') +
    `<div class="chips">${chips}</div>
     <div class="legend"><span><i style="background:var(--blue)"></i>подталкивал к выбранному токену</span><span><i style="background:var(--red)"></i>подталкивал к сопернику</span><span class="muted">метод: ${esc(A.method)} · ${f(A.seconds,1)} с</span></div>
     <table><tr><th class="num">поз</th><th>токен</th><th>где</th><th class="num">вклад</th><th></th></tr>${trs}</table>${gap}
     <p class="muted small">Gradient × input — линейное приближение в одной точке: знак и порядок величин надёжны, точные числа — нет. Для точных значений запустите с --ig-steps 32.</p>`;
}

// ───────────── Внимание по головам ─────────────
function headsBlock(rows){
  const withH = rows.filter(r => r.a.heads_dist);
  if (!withH.length) return '';
  if (S.hl==null || !withH.some(r => r.layer===S.hl)) S.hl = withH[0].layer;
  const r = withH.find(x => x.layer===S.hl), Hd = r.a.heads_dist, n = Hd[0].length;
  const padL = 70, W = 1100, cw = Math.min(40,(W-padL-10)/n), ch = 14, H = Hd.length*ch + 66, every = n>60 ? Math.ceil(n/60) : 1;
  let g = '';
  Hd.forEach((row,h) => {
    g += `<text x="${padL-8}" y="${h*ch+11}" text-anchor="end" class="mute">гол. ${h}</text>`;
    row.forEach((v,j) => { g += `<rect x="${padL+j*cw}" y="${h*ch}" width="${cw-1}" height="${ch-1}" style="fill:${seq(Math.sqrt(v))}"${tip(`слой ${r.layer}, голова ${h}\nсмотрит на «${(toks[j]||'').replace(/\n/g,'↵')}» (поз ${j})\nвес ${pct(v)}${r.a.head_entropy ? '\nэнтропия головы '+r.a.head_entropy[h] : ''}`)}></rect>`; });
  });
  for (let j=0;j<n;j+=every) g += `<text transform="translate(${padL+j*cw+cw/2},${Hd.length*ch+8}) rotate(60)" class="mono" style="font-size:10px">${vis(clip(toks[j]||'',10))}</text>`;
  return `<h3>По головам <span class="muted small">— среднее скрывает специализацию: разные головы смотрят на разное</span></h3>
    <div class="legend">слой: ${withH.map(x => `<button class="${x.layer===S.hl?'on':''}" data-hl="${x.layer}">${x.layer}</button>`).join(' ')}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Внимание по головам">${g}</svg>`;
}

// ───────────── Быстрые оценки важности ─────────────
function renderMap2(){
  const sec = $('#s-map2');
  if (!LMX || (!LMX.atp && !LMX.angular)) { sec.style.display = 'none'; return; }
  let html = hdr('', 'Быстрые оценки важности: градиент и угловое расстояние', 'Две оценки, которым не нужен перебор слоёв. Если они согласуются с картой выше, выводу можно верить; если нет — это повод проверить слой отдельно.');
  const A = LMX.atp;
  if (A) {
    html += `<h3>Attribution patching <span class="muted small">— оценка изменения log p исходных токенов, если убрать компоненту (один обратный проход на промпт)</span></h3>
      <div class="tiles"><div class="tile"><b>${A.corr!=null ? sgn(A.corr,2) : '—'}</b><span>корреляция с полным перебором</span></div><div class="tile"><b>${A.prompts}</b><span>промптов</span></div></div>` +
      divBars([{col:'--violet', vals:A.attn}, {col:'--aqua', vals:A.mlp}], {label:'Оценка важности компонент',
        tip: i => `слой ${i} (${TY[i]})\nбез внимания: Δlog p ≈ ${sgn(A.attn[i])}\nбез MLP: Δlog p ≈ ${sgn(A.mlp[i])}\nбез слоя целиком: ≈ ${sgn(A.layer[i])}`}) +
      `<div class="legend"><span><i style="background:var(--violet)"></i>внимание</span><span><i style="background:var(--aqua)"></i>MLP</span><span class="muted">столбец вниз — без компоненты ответ хуже; вверх — оценка говорит, что без неё было бы лучше</span></div>
       <div class="note">Это линейное приближение. ${A.corr==null ? 'Полный перебор не запускался (режим --fast), поэтому проверить его нечем.' : (A.corr>0.8 ? 'Корреляция с перебором высокая — для этой модели оценке можно доверять.' : 'Корреляция с перебором невысокая — используйте оценку только для грубого отбора, а выводы делайте по карте с перебором.')}</div>`;
  }
  const G = LMX.angular;
  if (G) {
    const keys = Object.keys(G).sort((a,b)=>a-b), cols = ['--blue','--orange','--aqua','--violet','--magenta','--green'];
    const vm = Math.max(...keys.map(k => Math.max(...G[k].dist))) || 1, W = 1100, padL = 60, cw = (W-68)/NL, Y = v => 12+(1-v/vm)*130;
    let g = `<line class="axis" x1="${padL}" x2="${W-8}" y1="${Y(0)}" y2="${Y(0)}"/><text x="${padL-6}" y="16" text-anchor="end" class="mute">${f(vm,2)}</text><text x="${padL-6}" y="${Y(0)}" text-anchor="end" class="mute">0</text>`;
    keys.forEach((k,q) => {
      const d = G[k].dist, c = cols[q%cols.length];
      g += `<path d="${d.map((v,i)=>(i?'L':'M')+(padL+i*cw+cw/2)+','+Y(v)).join(' ')}" style="fill:none;stroke:var(${c});stroke-width:2"/>`;
      d.forEach((v,i) => { g += `<circle cx="${padL+i*cw+cw/2}" cy="${Y(v)}" r="${i===G[k].best_start?4:2}" style="fill:var(${c})" data-layer="${i}"${tip(`блок из ${k} сл., начало — слой ${i}\nрасстояние ${v}`)}></circle>`; });
    });
    for (let i=3;i<NL;i+=4) g += `<text x="${padL+i*cw+cw/2}" y="${Y(0)+14}" text-anchor="middle" class="mute">${i}</text>`;
    const trs = keys.map(k => `<tr><td>${k}</td><td>слои ${G[k].best_start}–${G[k].best_start+(+k)-1}</td><td class="num">${G[k].best}</td></tr>`).join('');
    html += `<h3>Угловое расстояние <span class="muted small">— насколько блок слоёв поворачивает вектор последнего токена</span></h3>
      <svg viewBox="0 0 ${W} 164" role="img" aria-label="Угловое расстояние по слоям">${g}</svg>
      <div class="legend">${keys.map((k,q)=>`<span><i style="background:var(${cols[q%cols.length]})"></i>блок из ${k} сл.</span>`).join('')}<span class="muted">по горизонтали — первый слой блока; крупная точка — блок, который меняет вектор меньше всего</span></div>
      <table><tr><th>размер блока</th><th>кандидат на удаление</th><th class="num">расстояние</th></tr>${trs}</table>
      <div class="note">Важности отдельных слоёв не складываются: слои компенсируют друг друга. Поэтому блок для удаления выбирают по расстоянию между его входом и выходом, а не по сумме важностей. После удаления модель обычно дообучают; без этого проверьте блок через --skip-layers.</div>`;
  }
  sec.innerHTML = html;
}

// ───────────── Activation patching ─────────────
function renderPatch(){
  const P = D.patching, sec = $('#s-patch');
  if (!P) { sec.style.display = 'none'; return; }
  const kn = {resid:'выход слоя (остаточный поток)', attn:'выход внимания', mlp:'выход MLP'};
  const n = P.positions.length, W = 1100, padL = 70, cw = Math.min(40,(W-padL-10)/n), every = n>60 ? Math.ceil(n/60) : 1;
  let body = '';
  Object.keys(P.components).forEach(k => {
    const G = P.components[k], ch = G.length>40 ? 9 : 14, H = G.length*ch + 66;
    let g = '';
    G.forEach((row,i) => {
      if (G.length<=16 || i%4===3) g += `<text x="${padL-8}" y="${i*ch+ch-3}" text-anchor="end" class="mute">слой ${i}</text>`;
      row.forEach((v,j) => { g += `<rect x="${padL+j*cw}" y="${i*ch}" width="${cw-1}" height="${ch-1}" style="fill:${dvg(v)}" data-layer="${i}"${tip(`слой ${i} (${P.layer_types[i]}), поз ${P.positions[j]}\nиспорченный: «${P.tokens_corrupt[j].replace(/\n/g,'↵')}»  чистый: «${P.tokens_clean[j].replace(/\n/g,'↵')}»\nвосстановление ответа ${sgn(v,2)}`)}></rect>`; });
    });
    for (let j=0;j<n;j+=every) g += `<text transform="translate(${padL+j*cw+cw/2},${G.length*ch+8}) rotate(60)" class="mono" style="font-size:10px">${vis(clip(P.tokens_corrupt[j],10))}</text>`;
    body += `<h3>${kn[k]}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Патчинг: ${kn[k]}">${g}</svg>`;
  });
  sec.innerHTML = hdr('', 'Activation patching: где проходит информация', 'Модель запущена на «испорченном» промпте, но выход одного слоя на одной позиции подменён выходом из прогона на «чистом» промпте. Синяя клетка — подмена вернула правильный ответ: именно здесь проходит то, чем промпты отличаются.') +
    `<table><tr><th></th><th>промпт</th><th>ответ</th></tr><tr><td>чистый</td><td>${esc(P.clean.prompt)}</td><td class="mono">${vis(P.clean.answer)}</td></tr><tr><td>испорченный</td><td>${esc(P.corrupt.prompt)}</td><td class="mono">${vis(P.corrupt.answer)}</td></tr></table>
     <div class="tiles"><div class="tile"><b>${sgn(P.ld_clean,2)}</b><span>разность логитов на чистом</span></div><div class="tile"><b>${sgn(P.ld_corrupt,2)}</b><span>на испорченном</span></div></div>
     ${body}
     <div class="legend"><span><i style="background:var(--blue)"></i>1 — ответ полностью восстановлен</span><span><i style="background:var(--mid)"></i>0 — подмена ничего не изменила</span><span><i style="background:var(--red)"></i>меньше 0 — стало хуже</span></div>`;
}

// ───────────── Сравнение с эталоном (квантизация) ─────────────
function renderQuant(){
  const Q = D.quant, sec = $('#s-quant');
  if (!Q) { sec.style.display = 'none'; return; }
  const ty = Q.layer_types || '';
  const bars = divBars([{col:'--orange', vals:Q.rel_err}], {label:'Относительная ошибка по слоям', h:130, layerOf: i => Math.max(i-1,0), name: i => i===0 ? 'эмб' : i-1,
    tip: i => `${i===0 ? 'эмбеддинги' : 'после слоя '+(i-1)+' ('+(ty[i-1]||'')+')'}\nотносительная ошибка ${Q.rel_err[i]}\nкосинус ${Q.cos[i]}${i ? '\nприрост ошибки '+sgn(Q.err_growth[i-1],4) : ''}`});
  const jumps = Q.err_growth.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,6).filter(r => r.v>0);
  const jt = jumps.map(r => `<tr><td>слой ${r.i} <span class="muted">(${ty[r.i]||''})</span></td><td class="num">${sgn(r.v,4)}</td></tr>`).join('');
  const worst = [...Q.prompts].sort((a,b)=>b.kl-a.kl).slice(0,6).map(p => `<tr><td>${esc(clip(p.prompt,80))}</td><td class="num">${p.kl}</td></tr>`).join('');
  sec.innerHTML = hdr('', 'Квантизация: где проверяемая модель расходится с эталоном', 'Обе модели прочитали одни и те же тексты. По слоям видно, где возникает ошибка сжатия и как она накапливается; на выходе — насколько изменились предсказания.') +
    `<div class="muted small">эталон: ${esc((Q.ref||{}).path||'?')} · проверяемая: ${esc((Q.test||{}).path||'?')}</div>
     <div class="tiles"><div class="tile"><b>${Q.kl}</b><span>KL на выходе [${Q.kl_lo}; ${Q.kl_hi}]</span></div><div class="tile"><b>${pct(Q.top1_agree)}</b><span>лучший токен совпал</span></div><div class="tile"><b>${f(Q.p_ref_token,3)}</b><span>вероятность токена эталона</span></div><div class="tile"><b>${f(Q.rel_err[Q.rel_err.length-1],3)}</b><span>ошибка вектора на выходе</span></div></div>
     <h3>Относительная ошибка вектора после каждого слоя</h3>${bars}
     <div class="cols"><div><h3>Где ошибка растёт сильнее всего</h3><table><tr><th>слой</th><th class="num">прирост ошибки</th></tr>${jt}</table></div>
     <div><h3>Промпты с наибольшим расхождением</h3><table><tr><th>промпт</th><th class="num">KL</th></tr>${worst}</table></div></div>
     <p class="muted small">Слои с резким приростом ошибки — первые кандидаты на более аккуратную квантизацию (больше бит для этих слоёв).</p>`;
}

// ───────────── Отключение слоёв ─────────────
function renderAblation(){
  const A = D.ablation, sec = $('#s-ablation');
  if (!A) { sec.style.display = 'none'; return; }
  const mx = Math.max(...A.first_token.map(r => Math.max(r.base, r.ablated))) || 1;
  const trs = A.first_token.map(r => `<tr><td class="mono">${vis(r.token)}</td><td class="num">${pct(r.base)}</td><td><span class="bar" style="width:${r.base/mx*120}px"></span></td><td class="num">${pct(r.ablated)}</td><td><span class="bar" style="width:${r.ablated/mx*120}px;background:var(--orange)"></span></td></tr>`).join('');
  sec.innerHTML = hdr('', `Сравнение: без слоёв ${A.skipped.join(', ')}`, 'Тот же промпт, обычный прогон и прогон с отключёнными слоями.') +
    `<div class="tiles"><div class="tile"><b>${A.kl}</b><span>KL первого токена</span></div><div class="tile"><b>${A.top_changed?'изменился':'тот же'}</b><span>лучший токен</span></div><div class="tile"><b>${A.common_prefix}</b><span>токенов ответа совпало</span></div></div>
     <table><tr><th>первый токен</th><th class="num" colspan="2">обычный</th><th class="num" colspan="2">без слоёв</th></tr>${trs}</table>
     <h3>Ответы</h3><p><b>обычный:</b> ${esc(A.answer_base)}</p><p><b>без слоёв:</b> ${esc(A.answer_ablated)}</p>`;
}

// ───────────── Веса ─────────────
function renderWeights(){
  const W0 = D.weights;
  if (!W0) { $('#s-weights').innerHTML = hdr('','Веса','') + empty('Нет данных.'); return; }
  const order = ['линейное внимание','полное внимание','MLP','эмбеддинги','lm_head','нормализация','прочее'];
  const col = ['--blue','--orange','--aqua','--yellow','--magenta','--green','--violet'];
  const cats = {};
  W0.modules.forEach(m => { const c = cats[m.cat] || (cats[m.cat] = {n:0,p:0,b:0}); c.n++; c.p+=m.params; c.b+=m.bytes; });
  const names = order.filter(k => cats[k]).concat(Object.keys(cats).filter(k => !order.includes(k)));
  const tp = W0.total_params, W = 1100;
  let x = 0, seg = '';
  names.forEach(k => {
    const w = cats[k].p/tp*(W-2*names.length);
    const ci = order.indexOf(k);
    seg += `<rect x="${x}" y="0" width="${Math.max(w,1)}" height="34" rx="3" style="fill:var(${col[ci<0?6:ci]})"${tip(`${k}\n${f(cats[k].p,0)} параметров (${pct(cats[k].p/tp)})\n${(cats[k].b/1e6).toFixed(1)} МБ`)}></rect>`;
    x += w + 2;
  });
  const leg = names.map(k => { const ci = order.indexOf(k); return `<span><i style="background:var(${col[ci<0?6:ci]})"></i>${k}</span>`; }).join('');
  const ct = names.map(k => `<tr><td>${k}</td><td class="num">${cats[k].n}</td><td class="num">${f(cats[k].p,0)}</td><td class="num">${(cats[k].b/1e6).toFixed(1)}</td><td class="num">${pct(cats[k].p/tp)}</td></tr>`).join('');
  const perLayer = Array(NL).fill(0); W0.modules.forEach(m => { if (m.layer!=null) perLayer[m.layer] += m.bytes; });
  const pm = Math.max(...perLayer) || 1, pw = (W-60-8)/NL;
  let pl = `<line class="axis" x1="60" x2="${W-8}" y1="80" y2="80"/><text x="54" y="80" text-anchor="end" class="mute">0</text><text x="54" y="14" text-anchor="end" class="mute">${(pm/1e6).toFixed(0)} МБ</text>`;
  perLayer.forEach((b,i) => { const h = b/pm*64; pl += `<rect class="t-${TY[i]}" x="${60+i*pw+1}" y="${80-h}" width="${pw-2}" height="${Math.max(h,0.5)}" rx="1.5"></rect><rect class="hit" x="${60+i*pw}" y="10" width="${pw}" height="72" data-layer="${i}"${tip(`слой ${i} (${TY[i]})\n${(b/1e6).toFixed(1)} МБ`)}></rect>`; });
  const bigT = [...W0.modules].sort((a,b)=>b.bytes-a.bytes).slice(0,10).map(m => `<tr><td class="mono">${esc(m.m.replace('language_model.',''))}</td><td>${m.shape && m.shape.length===2 ? m.shape[0]+'×'+m.shape[1] : ''}${m.bits ? ' · '+m.bits+' бит' : ''}</td><td class="num">${f(m.params,0)}</td><td class="num">${(m.bytes/1e6).toFixed(1)}</td></tr>`).join('');
  const bpw = W0.total_bytes*8/tp;
  $('#s-weights').innerHTML = hdr('', 'Веса: из чего состоит модель', 'Все тензоры, которые участвовали в вычислениях. Имена модулей те же, что в trace.log, поэтому любую строку журнала можно найти здесь.') +
    `<div class="tiles"><div class="tile"><b>${big(tp)}</b><span>параметров</span></div><div class="tile"><b>${gb(W0.total_bytes)}</b><span>в памяти</span></div><div class="tile"><b>${bpw.toFixed(2)}</b><span>бит на параметр (с масштабами)</span></div><div class="tile"><b>${W0.modules.length}</b><span>модулей с весами</span></div></div>
     <svg viewBox="0 0 ${W} 36" role="img" aria-label="Доли параметров по категориям">${seg}</svg><div class="legend">${leg}</div>
     <table><tr><th>категория</th><th class="num">модулей</th><th class="num">параметров</th><th class="num">МБ</th><th class="num">доля</th></tr>${ct}</table>
     <h3>Размер каждого слоя <span class="muted small">— слои полного и линейного внимания различаются по размеру</span></h3>
     <svg viewBox="0 0 ${W} 96" role="img" aria-label="Размер слоёв">${pl}</svg>
     <h3>Самые большие тензоры</h3><table><tr><th>модуль</th><th>матрица</th><th class="num">параметров</th><th class="num">МБ</th></tr>${bigT}</table>`;
}

// ───────────── Отрисовка и события ─────────────
function renderAll(){ renderHead(); renderPipeline(); renderTokens(); renderEmb(); renderLayers(); renderResid(); renderLens(); renderDla(); renderTokAttr(); renderAttn(); renderSample(); renderMap(); renderMap2(); renderPatch(); renderQuant(); renderAblation(); renderWeights();
  if (!D.patching) $('#n-patch').style.display = 'none';
  if (!D.quant) $('#n-quant').style.display = 'none'; }
renderAll();

const tipEl = $('#tip');
function showTip(el, x, y){
  const t = el.getAttribute('data-tip'); if (!t) return;
  tipEl.textContent = t; tipEl.style.display = 'block';
  const r = tipEl.getBoundingClientRect();
  tipEl.style.left = Math.min(x+14, innerWidth - r.width - 8) + 'px';
  tipEl.style.top = (y+16+r.height > innerHeight ? y-r.height-10 : y+16) + 'px';
}
document.addEventListener('mousemove', e => { const el = e.target.closest && e.target.closest('[data-tip]'); if (el) showTip(el, e.clientX, e.clientY); else tipEl.style.display = 'none'; });
document.addEventListener('focusin', e => { const el = e.target.closest && e.target.closest('[data-tip]'); if (el) { const b = el.getBoundingClientRect(); showTip(el, b.left+b.width/2, b.top); } });
document.addEventListener('focusout', () => { tipEl.style.display = 'none'; });
function act(el){
  if (el.dataset.layer != null) { S.layer = +el.dataset.layer; renderLayers(); renderAttn(); }
  else if (el.dataset.tok != null) { S.tok = +el.dataset.tok; renderTokens(); renderEmb(); }
  else if (el.dataset.hl != null) { S.hl = +el.dataset.hl; renderAttn(); }
  else if (el.dataset.lens != null) { S.lens = +el.dataset.lens; renderLayers(); renderResid(); renderLens(); renderDla(); renderTokAttr(); renderAttn(); }
  else if (el.dataset.step != null) { S.sstep = +el.dataset.step; renderSample(); }
  else if (el.dataset.metric != null) { S.metric = el.dataset.metric; renderLayers(); renderMap(); }
  else if (el.dataset.go) { const t = document.getElementById(el.dataset.go); if (t) t.scrollIntoView(); }
}
document.addEventListener('click', e => { const el = e.target.closest && e.target.closest('[data-hl],[data-layer],[data-tok],[data-lens],[data-step],[data-metric],[data-go]'); if (el) act(el); });
document.addEventListener('keydown', e => { if ((e.key==='Enter'||e.key===' ') && e.target.dataset && (e.target.dataset.layer!=null||e.target.dataset.step!=null)) { e.preventDefault(); act(e.target); } });
$('#theme').addEventListener('click', () => {
  const root = document.documentElement, cur = root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  root.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
});
})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("python make_report.py trace_logs/<папка_прогона>")
    print(build_report(sys.argv[1]))
