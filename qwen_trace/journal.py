"""Журнал прогона: события (events.jsonl), читаемый лог (trace.log) и сводки по тензорам и весам."""
import json
from pathlib import Path

import mlx.core as mx


# ───────────────────────── журнал ─────────────────────────
class Journal:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.txt = open(out_dir / "trace.log", "w", encoding="utf-8")
        self.jsonl = open(out_dir / "events.jsonl", "w", encoding="utf-8")
        self.n = 0

    def section(self, title: str):
        line = f"\n{'=' * 78}\n{title}\n{'=' * 78}"
        self.txt.write(line + "\n")
        print(line)

    def text(self, line: str = "", echo: bool = True):
        self.txt.write(line + "\n")
        if echo:
            print(line)

    def event(self, kind: str, **data):
        self.n += 1
        rec = {"n": self.n, "kind": kind, **data}
        self.jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        self.txt.close()
        self.jsonl.close()


def stats(x: mx.array) -> dict:
    """Сводка по тензору: форма и основные статистики."""
    f = x.astype(mx.float32)
    last = f.reshape(-1, f.shape[-1])[-1]          # вектор последней позиции
    vals = [f.mean(), f.std(), f.min(), f.max(), mx.linalg.norm(last)]
    mx.eval(*vals)
    mean, std, mn, mxv, norm = (round(v.item(), 4) for v in vals)
    return {"shape": list(x.shape), "dtype": str(x.dtype).replace("mlx.core.", ""),
            "mean": mean, "std": std, "min": mn, "max": mxv, "last_norm": norm}


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def weight_info(module, with_stats: bool = False) -> dict:
    """Тензоры, принадлежащие самому модулю (без вложенных): формы, типы, размер,
    логическая форма матрицы (для 4-битных — распакованная) и число параметров."""
    tensors, nbytes = {}, 0
    for key, val in module.items():
        if isinstance(val, mx.array):
            tensors[key] = {"shape": list(val.shape),
                            "dtype": str(val.dtype).replace("mlx.core.", "")}
            nbytes += val.nbytes
    if not tensors:
        return {}
    info = {"tensors": tensors, "bytes": nbytes}
    bits = getattr(module, "bits", None)
    w = module["weight"] if "weight" in module else None
    if bits and isinstance(w, mx.array):
        out_f, in_f = w.shape[0], w.shape[-1] * 32 // bits
        info.update(bits=bits, group_size=getattr(module, "group_size", None),
                    logical_shape=[out_f, in_f], params=out_f * in_f)
    else:
        info["params"] = sum(_prod(t["shape"]) for t in tensors.values())
        if isinstance(w, mx.array):
            info["logical_shape"] = list(w.shape)
    if with_stats and isinstance(w, mx.array):
        try:                                         # для огромных матриц — по первым строкам
            rows = min(w.shape[0], 16384)
            if bits:
                full = mx.dequantize(w[:rows], module["scales"][:rows], module["biases"][:rows],
                                     group_size=module.group_size, bits=bits)
            else:
                full = w[:rows]
            f = full.astype(mx.float32)
            vals = [f.mean(), f.std(), mx.abs(f).max()]
            mx.eval(*vals)
            info["wstats"] = {"mean": round(vals[0].item(), 5), "std": round(vals[1].item(), 5),
                              "absmax": round(vals[2].item(), 4), "rows": rows}
        except Exception as e:                      # статистика не должна ломать прогон
            info["wstats"] = {"error": str(e)}
    return info


def weights_text(info: dict) -> str:
    if not info:
        return "нет весов"
    parts = [f"{k} {tuple(v['shape'])} {v['dtype']}" for k, v in info["tensors"].items()]
    s = ", ".join(parts)
    if "bits" in info:
        s += f", квантизация: {info['bits']} бит, группа {info['group_size']}"
        s += f"; матрица {info['logical_shape'][0]}×{info['logical_shape'][1]}"
    s += f"; {info['params']:,} парам., {info['bytes'] / 1e6:.2f} МБ".replace(",", " ")
    ws = info.get("wstats")
    if ws and "std" in ws:
        s += f"; веса: std={ws['std']} max|w|={ws['absmax']}"
    return s
