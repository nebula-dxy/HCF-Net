from pathlib import Path
from typing import List

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_external_credible"
DATASETS = ["ACM", "DBLP", "Yelp"]
SUBLABELS = ["(a)", "(b)", "(c)"]

DPI = 600

TOP_PAD = 10
OUTER_PAD_X = 6
OUTER_PAD_Y = 10
PANEL_GAP = 6
LABEL_GAP = 4
LABEL_BLOCK = 54
LABEL_FONT_SIZE = 50


def try_font(size: int, bold: bool = False):
    if bold:
        names = ["timesbd.ttf", "Times New Roman Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
    else:
        names = ["times.ttf", "Times New Roman.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


LABEL_FONT = try_font(LABEL_FONT_SIZE, bold=False)


def crop_side_whitespace(image: Image.Image, pad_x: int = 14) -> Image.Image:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left = max(0, bbox[0] - pad_x)
    right = min(rgb.width, bbox[2] + pad_x)
    return rgb.crop((left, 0, right, rgb.height))


def centered_text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill="black")


def build_triptych(kind: str) -> None:
    images: List[Image.Image] = []
    for dataset in DATASETS:
        img = Image.open(OUT_DIR / f"{dataset}_{kind}_full.tiff")
        img = crop_side_whitespace(img)
        images.append(img)

    panel_w = max(img.width for img in images)
    panel_h = max(img.height for img in images)
    canvas_w = OUTER_PAD_X * 2 + panel_w * 3 + PANEL_GAP * 2
    canvas_h = TOP_PAD + panel_h + LABEL_GAP + LABEL_BLOCK + OUTER_PAD_Y

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        x0 = OUTER_PAD_X + idx * (panel_w + PANEL_GAP) + (panel_w - img.width) // 2
        y0 = TOP_PAD
        canvas.paste(img, (x0, y0))
        label = f"{SUBLABELS[idx]} {DATASETS[idx]}-{kind.upper()}"
        centered_text(draw, x0 + img.width / 2, y0 + panel_h + LABEL_GAP + LABEL_BLOCK / 2, label, LABEL_FONT)

    png_path = OUT_DIR / f"combined_{kind}_triptych_600dpi.png"
    tiff_path = OUT_DIR / f"combined_{kind}_triptych_600dpi.tiff"
    canvas.save(tiff_path, format="TIFF", dpi=(DPI, DPI))
    tmp_png = png_path.with_suffix(".tmp.png")
    canvas.save(tmp_png, format="PNG", dpi=(DPI, DPI))
    tmp_png.replace(png_path)


def main() -> None:
    build_triptych("sir")
    build_triptych("si")
    print("saved stitched SIR/SI triptych figures to", OUT_DIR)


if __name__ == "__main__":
    main()
