"""Снимки разделов отчёта для README: docs/img/*.png.

Запуск из корня проекта (нужен Google Chrome; больше ничего ставить не надо):

    python docs/make_screenshots.py            # все снимки
    python docs/make_screenshots.py lens dla   # только перечисленные (по части имени)

Как это работает: у страницы отчёта есть режим снимка (#shot=<раздел>&step=...), в котором
виден один раздел без меню. Chrome в режиме headless делает снимок, а лишняя пустота снизу
обрезается и цвета сжимаются в палитру, чтобы PNG весили меньше.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SINGLE = ROOT / "examples" / "qwen38_27b_single.html"
BATCH = ROOT / "examples" / "qwen38_27b_batch.html"
TINY = ROOT / "examples" / "tiny_model.html"

# имя файла, страница, хэш режима снимка
SHOTS = [
    ("01-overview",  SINGLE, "shot=s-head"),
    ("02-pipeline",  SINGLE, "shot=s-pipeline"),
    ("03-tokens",    SINGLE, "shot=s-tokens"),
    ("04-embeddings", SINGLE, "shot=s-emb"),
    ("05-layers",    SINGLE, "shot=s-layers&layer=62"),
    ("06-residual",  SINGLE, "shot=s-resid"),
    ("07-logit-lens", SINGLE, "shot=s-lens&step=0"),
    ("08-dla",       SINGLE, "shot=s-dla&step=0"),
    ("09-token-contribution", SINGLE, "shot=s-tokattr&step=0"),
    ("10-attention", SINGLE, "shot=s-attn&step=0"),
    ("11-sampling-entropy", SINGLE, "shot=s-sample"),
    ("12-layer-map", SINGLE, "shot=s-map"),
    ("13-angular",   SINGLE, "shot=s-map2"),
    ("14-patching",  SINGLE, "shot=s-patch"),
    ("15-quant-compare", TINY, "shot=s-quant"),
    ("16-batch-summary", BATCH, "shot=s-summary"),
    ("17-batch-prompt", BATCH, "shot=s-sample&prompt=1"),
    ("18-weights",   SINGLE, "shot=s-weights"),
]
# слишком длинные разделы обрезаем сверху до этой высоты (в пикселях снимка)
CROP = {"05-layers": 1765}
WIDTH, HEIGHT, SCALE = 1240, 6000, 1.5
BG = (243, 243, 240)          # --bg светлой темы


def close_card(img: Image.Image) -> Image.Image:
    """Обрезанный раздел дорисовывает до закрытой карточки: нижняя граница и поля."""
    left, right = round(46 * SCALE), round((46 + 1148) * SCALE)
    card, line, tail = (252, 252, 251), (222, 220, 214), round(18 * SCALE)
    out = Image.new("RGB", (img.width, img.height + tail + round(12 * SCALE)), BG)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle([left, img.height, right, img.height + tail], fill=card)
    draw.line([(left, img.height), (left, img.height + tail)], fill=line, width=2)
    draw.line([(right, img.height), (right, img.height + tail)], fill=line, width=2)
    draw.line([(left, img.height + tail), (right, img.height + tail)], fill=line, width=2)
    return out


def shoot(page: Path, frag: str, dest: Path):
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={WIDTH},{HEIGHT}", f"--force-device-scale-factor={SCALE}",
                        "--virtual-time-budget=4000", f"--screenshot={raw}",
                        f"file://{page}#{frag}"], check=True, capture_output=True)
        img = Image.open(raw).convert("RGB")
    diff = ImageChops.difference(img, Image.new("RGB", img.size, BG)).getbbox()
    if not diff:
        raise SystemExit(f"пустой снимок: {dest.name}")
    pad = int(12 * SCALE)
    bottom = min(img.height, diff[3] + pad)
    cut = CROP.get(dest.stem)
    if cut and cut < bottom:
        img = close_card(img.crop((0, 0, img.width, cut)))
    else:
        img = img.crop((0, 0, img.width, bottom))
    img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(dest, optimize=True)
    return img.size


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, page, frag in SHOTS:
        if argv and not any(a in name for a in argv):
            continue
        w, h = shoot(page, frag, OUT / f"{name}.png")
        print(f"{name}.png  {w}x{h}  {(OUT / f'{name}.png').stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main(sys.argv[1:])
