"""Пакетный режим: подробный прогон на каждый промпт и один общий отчёт с переключателем."""
import contextlib
import io
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from .journal import Journal
from .pipeline import run


def prompt_slug(prompt: str, n: int = 36) -> str:
    """Короткое имя папки из текста промпта."""
    s = re.sub(r"[\W_]+", "-", prompt, flags=re.UNICODE).strip("-").lower()
    return s[:n].strip("-") or "prompt"


def run_batch(model, tok, prompts: list, out_dir: Path, run_kwargs: dict, model_path=None,
              map_cfg: dict = None, build_reports: bool = True, log=print) -> Path:
    """Подробный прогон на каждый промпт — журнал каждого в своей подпапке, а отчёт один:
    report.html в папке пакета, где вверху можно переключаться между промптами и «Сводкой».

    Порядок: сначала считаются все прогоны, затем (если нужна) карта важности слоёв по тем
    же промптам, потом пишется batch.json и собирается отчёт."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, t0 = [], time.perf_counter()
    for i, prompt in enumerate(prompts, 1):
        entry = {"prompt": prompt, "dir": f"{i:02d}_{prompt_slug(prompt)}", "status": "ok"}
        journal = Journal(out_dir / entry["dir"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):    # журнал каждого прогона — в его папку
                entry["answer"] = run(model, tok, prompt, journal, model_path=model_path,
                                      **run_kwargs)
        except Exception as e:                                 # один сбой не должен губить пакет
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            journal.close()
        el = time.perf_counter() - t0
        log(f"[{i}/{len(prompts)}] {entry['status']:5s} {prompt[:60]!r}"
            + (f" -> {entry['answer'][:40]!r}" if entry["status"] == "ok" else f"  {entry['error']}")
            + f"   (прошло {el:.0f} с, осталось ~{el / i * (len(prompts) - i):.0f} с)")
        entries.append(entry)
    ok = [e for e in entries if e["status"] == "ok"]

    if map_cfg and ok:
        from . import layer_map as LM
        log("\nКарта важности слоёв по тем же промптам:")
        res = LM.build_map(model, tok, [e["prompt"] for e in ok], map_cfg["tokens"],
                           map_cfg["window"], run_kwargs.get("thinking", False),
                           run_kwargs.get("raw", False), log=log, fast=map_cfg["fast"])
        LM.save_json(res, out_dir / "layer_map.json")
        (out_dir / "layer_map.txt").write_text(LM.render(res), encoding="utf-8")
        for e in ok:                       # копия нужна, чтобы подпапку можно было собрать отдельно
            shutil.copy(out_dir / "layer_map.json", out_dir / e["dir"] / "layer_map.json")

    (out_dir / "batch.json").write_text(json.dumps(
        {"created": datetime.now().isoformat(timespec="seconds"), "prompts": entries},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if build_reports:
        from . import report
        log(f"\nОтчёт: {report.build_batch_report(out_dir)}")
    bad = len(entries) - len(ok)
    log(f"готово: {len(ok)} из {len(entries)} промптов" + (f", с ошибками: {bad}" if bad else ""))
    return out_dir
