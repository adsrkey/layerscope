"""Сборка report.html: один прогон или пакет промптов. Страница целиком лежит в template.html
(HTML, CSS и JS), данные вставляются в неё JSON-блоками."""
import json
from pathlib import Path

from .data import build_data, summarize_run

TEMPLATE_PATH = Path(__file__).with_name("template.html")


def _template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def build_report(run_dir, out=None) -> Path:
    run_dir = Path(run_dir)
    data = build_data(run_dir)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = _template().replace("__TITLE__", "Отчёт: " + run_dir.name).replace("__DATA__", blob)
    out = Path(out) if out else run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def _blob(obj) -> str:
    """JSON для вставки в <script>: без закрывающих тегов и спецсимволов строк."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_batch_report(batch_dir, out=None) -> Path:
    """Один report.html на весь пакет (trace --batch): вверху переключатель между
    «Сводкой» и подробным разбором каждого промпта. Данные промптов лежат в странице
    отдельными блоками и разбираются только при открытии, карта слоёв — одна на всех."""
    batch_dir = Path(batch_dir)
    b = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    lm_path = batch_dir / "layer_map.json"
    shared_map = json.loads(lm_path.read_text(encoding="utf-8")) if lm_path.exists() else None
    entries, blobs, meta = [], [], None
    for i, e in enumerate(b["prompts"]):
        row = {k: e[k] for k in ("prompt", "dir", "status") if k in e}
        blob = None
        if e["status"] == "ok":
            try:
                D = build_data(batch_dir / e["dir"])
                row.update(summarize_run(D))
                D["layer_map"] = None                     # общая карта хранится один раз
                blob = _blob(D)
                meta = meta or D["meta"]
            except Exception as ex:                       # плохой журнал не должен ломать отчёт
                row.update(status="error", error=f"{type(ex).__name__}: {ex}")
        else:
            row["error"] = e.get("error", "")
        entries.append(row)
        blobs.append(blob)
    first = next((r for r in entries if r["status"] == "ok"), None)
    head = {"run": batch_dir.name, "created": b.get("created"), "meta": meta,
            "layer_map": shared_map, "has_map": shared_map is not None,
            "layers": first["layers"] if first else 0,
            "layer_types": first["layer_types"] if first else "", "entries": entries}
    scripts = f'<script id="batch" type="application/json">{_blob(head)}</script>\n' + "\n".join(
        f'<script id="d-{i}" type="application/json">{x}</script>'
        for i, x in enumerate(blobs) if x is not None)
    page = (_template().replace("__TITLE__", "Отчёт по пакету: " + batch_dir.name)
            .replace('<script id="data" type="application/json">__DATA__</script>', scripts))
    out = Path(out) if out else batch_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
