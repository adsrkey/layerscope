// Прогоняет скрипт страницы report.html в Node с заглушкой DOM и проверяет, что все разделы
// строятся без ошибок и без "NaN"/"undefined" в разметке. Для пакетного отчёта по очереди
// открывает «Сводку» и каждый промпт, как это делает переключатель.
//
//     node tests/check_report.js путь/к/report.html
//
// Печатает по строке на вид и завершается с кодом 1 при ошибке.
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const jsons = {};
for (const m of html.matchAll(/<script id="([\w-]+)" type="application\/json">([\s\S]*?)<\/script>/g))
  jsons[m[1]] = m[2].replace(/<\\\//g, '</');
const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

const els = {};
const mk = id => els[id] || (els[id] = {id, innerHTML: '', style: {}, textContent: '', dataset: {}, value: '',
  classList: {add() {}, remove() {}}, handlers: {},
  addEventListener(t, fn) { this.handlers[t] = fn; }, getAttribute() { return null; },
  setAttribute() {}, scrollIntoView() {}, getBoundingClientRect() { return {width: 0, height: 0, left: 0, top: 0}; }});
const KNOWN = new Set(['psel', 'pprev', 'pnext', 'pswitch', 'theme', 'tip', 'seclinks', 'n-patch', 'n-quant', 'nav']);
const document = {
  getElementById: id => id in jsons ? {textContent: jsons[id]} : (KNOWN.has(id) || id.startsWith('s-') ? mk(id) : null),
  querySelector: s => mk(s.replace('#', '')), querySelectorAll: () => [],
  addEventListener() {}, documentElement: mk('root'), body: mk('body')};
const matchMedia = () => ({matches: false});
const location = {hash: process.env.HASH || ''};      // HASH='shot=s-lens&step=0' — режим снимка
const innerWidth = 1200, innerHeight = 800;

let failed = false;
const check = label => {
  const bad = [];
  const out = Object.keys(els).filter(id => id.startsWith('s-')).map(id => {
    const s = els[id];
    if (/NaN|undefined/.test(s.innerHTML)) bad.push(id);
    return s.innerHTML.length;
  });
  const empty = Object.keys(els).filter(id => id.startsWith('s-') && !els[id].innerHTML.length && els[id].style.display !== 'none');
  console.log(label.padEnd(28), bad.length ? 'NaN/undefined в: ' + bad.join(',') : 'ok', empty.length ? ' пустые: ' + empty.join(',') : '', ' разделов: ' + out.length);
  if (bad.length) failed = true;
};
try {
  new Function('document', 'matchMedia', 'innerWidth', 'innerHeight', 'location', code)(document, matchMedia, innerWidth, innerHeight, location);
  check('загрузка');
  const psel = els['psel'];
  if (jsons.batch) {
    const values = [...psel.innerHTML.matchAll(/<option value="(-?\d+)"(?! disabled)/g)].map(m => m[1]);
    for (const v of values) { psel.value = v; psel.handlers.change(); check(v === '-1' ? 'вид: сводка' : 'вид: промпт ' + v); }
    if (values.length < 2) { console.log('нет ни одного разобранного промпта'); failed = true; }
    // кнопки «вперёд/назад»
    psel.value = values[1]; els['pnext'].handlers.click(); els['pprev'].handlers.click(); check('кнопки ‹ ›');
  }
} catch (e) { console.log('ОШИБКА:', e.stack.split('\n').slice(0, 4).join(' | ')); failed = true; }
// MARKERS="s-attn:data-layer=;s-dla:data-dla" — вывести, сколько раз шаблон встречается в разделе
// (для ручной проверки содержимого после последнего открытого вида)
for (const spec of (process.env.MARKERS || '').split(';').filter(Boolean)) {
  const [id, re] = spec.split(':');
  console.log('маркер', id, JSON.stringify(re), (((els[id] || {}).innerHTML || '').match(new RegExp(re, 'g')) || []).length);
}
process.exit(failed ? 1 : 0);
