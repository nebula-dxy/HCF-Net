import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "results_external_credible" / "full_comparison.json"
OUT_DIR = ROOT / "results_external_credible"


PAPER_METHODS = {
    "ACM": ["HCF-Net", "PageRank", "DegreeDiscount", "AdaptiveDegree", "HeCo", "RGTN", "EASING"],
    "DBLP": ["HCF-Net", "PageRank", "DegreeDiscount", "AdaptiveDegree", "HeCo", "RGTN", "EASING"],
    "Yelp": ["HCF-Net", "HeCo", "PageRank", "DegreeDiscount", "AdaptiveDegree", "RGTN", "EASING"],
}


STYLE = {
    "HCF-Net": {"color": "#d62728", "width": 8, "marker": "diamond", "dash": None},
    "PageRank": {"color": "#1f77b4", "width": 5, "marker": "square", "dash": None},
    "DegreeDiscount": {"color": "#2ca02c", "width": 5, "marker": "triangle", "dash": None},
    "AdaptiveDegree": {"color": "#7b1fa2", "width": 5, "marker": "diamond", "dash": None},
    "HeCo": {"color": "#ff7f0e", "width": 5, "marker": "circle", "dash": None},
    "RGTN": {"color": "#9467bd", "width": 4, "marker": "down_triangle", "dash": (18, 12)},
    "EASING": {"color": "#8c564b", "width": 4, "marker": "plus", "dash": (18, 12)},
}


CANVAS = (1550, 1000)
PLOT_BOX = (150, 120, 1200, 880)
INSET_BOX = (705, 480, 1125, 800)
LEGEND_ORIGIN = (1235, 205)


def choose_methods(dataset: str, curves: Dict[str, List[float]]) -> List[str]:
    methods = [m for m in PAPER_METHODS.get(dataset, []) if m in curves]
    if "HCF-Net" in curves and "HCF-Net" not in methods:
        methods.insert(0, "HCF-Net")
    return methods


def marker_positions(length: int) -> List[int]:
    if length <= 8:
        return list(range(length))
    base = [0, 3, 7, 11, 15, length - 1]
    return sorted({min(length - 1, idx) for idx in base})


def try_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "DejaVuSans-Bold.ttf"])
    else:
        candidates.extend(["arial.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = try_font(40, bold=True)
FONT_AXIS = try_font(26, bold=False)
FONT_TICK = try_font(21, bold=False)
FONT_LEGEND = try_font(24, bold=False)
FONT_SMALL = try_font(18, bold=False)
FONT_ANNOT = try_font(24, bold=True)
TIGHT_PAD = 8


def draw_centered_text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def tight_crop(image: Image.Image, pad: int = TIGHT_PAD) -> Image.Image:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(rgb.width, bbox[2] + pad)
    bottom = min(rgb.height, bbox[3] + pad)
    return rgb.crop((left, top, right, bottom))


def save_curve_exports(image: Image.Image, save_path: Path) -> None:
    tight = tight_crop(image)
    tight.save(save_path, dpi=(300, 300), optimize=True)
    tight.save(save_path.with_suffix(".tiff"), format="TIFF", dpi=(600, 600))


def draw_rotated_text(image: Image.Image, xy: Tuple[int, int], text: str, font, fill: str, angle: float) -> None:
    bbox = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
    txt = Image.new("RGBA", (bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt)
    txt_draw.text((5, 5), text, font=font, fill=fill)
    rotated = txt.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=xy)


def scale_point(
    x_value: float,
    y_value: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    box: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    left, top, right, bottom = box
    px = left + (x_value - x_min) / max(x_max - x_min, 1e-9) * (right - left)
    py = bottom - (y_value - y_min) / max(y_max - y_min, 1e-9) * (bottom - top)
    return px, py


def draw_dashed_polyline(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], fill: str, width: int, dash: Tuple[int, int]) -> None:
    dash_on, dash_off = dash
    for start, end in zip(points[:-1], points[1:]):
        x1, y1 = start
        x2, y2 = end
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue
        dx = (x2 - x1) / seg_len
        dy = (y2 - y1) / seg_len
        cursor = 0.0
        while cursor < seg_len:
            end_pos = min(cursor + dash_on, seg_len)
            sx = x1 + dx * cursor
            sy = y1 + dy * cursor
            ex = x1 + dx * end_pos
            ey = y1 + dy * end_pos
            draw.line((sx, sy, ex, ey), fill=fill, width=width)
            cursor += dash_on + dash_off


def draw_marker(draw: ImageDraw.ImageDraw, point: Tuple[float, float], method: str) -> None:
    x, y = point
    color = STYLE[method]["color"]
    size = 12 if method == "HCF-Net" else 9
    outline = "white"
    marker = STYLE[method]["marker"]
    if marker == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=outline)
        draw.ellipse((x - size + 2, y - size + 2, x + size - 2, y + size - 2), fill=color)
    elif marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=outline)
        draw.rectangle((x - size + 2, y - size + 2, x + size - 2, y + size - 2), fill=color)
    elif marker == "triangle":
        pts = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        draw.polygon(pts, fill=outline)
        pts2 = [(x, y - size + 2), (x - size + 2, y + size - 2), (x + size - 2, y + size - 2)]
        draw.polygon(pts2, fill=color)
    elif marker == "down_triangle":
        pts = [(x - size, y - size), (x + size, y - size), (x, y + size)]
        draw.polygon(pts, fill=outline)
        pts2 = [(x - size + 2, y - size + 2), (x + size - 2, y - size + 2), (x, y + size - 2)]
        draw.polygon(pts2, fill=color)
    elif marker == "diamond":
        pts = [(x, y - size), (x - size, y), (x, y + size), (x + size, y)]
        draw.polygon(pts, fill=outline)
        pts2 = [(x, y - size + 2), (x - size + 2, y), (x, y + size - 2), (x + size - 2, y)]
        draw.polygon(pts2, fill=color)
    elif marker == "plus":
        draw.line((x - size, y, x + size, y), fill=outline, width=6)
        draw.line((x, y - size, x, y + size), fill=outline, width=6)
        draw.line((x - size, y, x + size, y), fill=color, width=3)
        draw.line((x, y - size, x, y + size), fill=color, width=3)


def draw_series(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    method: str,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    points = [scale_point(x, y, x_vals[0], x_vals[-1], y_min, y_max, box) for x, y in zip(x_vals, y_vals)]
    shadow_points = [(px, py + 1.2) for px, py in points]
    draw.line(shadow_points, fill=(255, 255, 255), width=STYLE[method]["width"] + 2)
    if STYLE[method]["dash"] is None:
        draw.line(points, fill=STYLE[method]["color"], width=STYLE[method]["width"])
    else:
        draw_dashed_polyline(draw, points, STYLE[method]["color"], STYLE[method]["width"], STYLE[method]["dash"])
    for idx in marker_positions(len(points)):
        draw_marker(draw, points[idx], method)
    return points


def draw_axes(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    x_max: int,
    y_max: float,
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle((0, 0, CANVAS[0], CANVAS[1]), fill="white")
    draw.rectangle((left, top, right, bottom), outline="#333333", width=3)

    x_ticks = list(range(1, x_max + 1, 2))
    y_ticks = np.linspace(0.0, y_max, 6)

    for tick in x_ticks:
        x, _ = scale_point(tick, 0.0, 1, x_max, 0.0, y_max, box)
        draw.line((x, top, x, bottom), fill="#e5e5e5", width=2)
        draw_centered_text(draw, x, bottom + 34, str(tick), FONT_TICK, "#333333")

    for tick in y_ticks:
        _, y = scale_point(1, float(tick), 1, x_max, 0.0, y_max, box)
        draw.line((left, y, right, y), fill="#e5e5e5", width=2)
        label = f"{tick:.2f}"
        bbox = draw.textbbox((0, 0), label, font=FONT_TICK)
        draw.text((left - 18 - (bbox[2] - bbox[0]), y - 10), label, font=FONT_TICK, fill="#333333")

    draw_centered_text(draw, (left + right) / 2, 50, title, FONT_TITLE, "#111111")
    draw_centered_text(draw, (left + right) / 2, bottom + 85, "Propagation step t", FONT_AXIS, "#222222")
    draw_rotated_text(image, (45, 365), "Infected proportion f(t)", FONT_AXIS, "#222222", 90)


def draw_legend(draw: ImageDraw.ImageDraw, methods: List[str]) -> None:
    x0, y0 = LEGEND_ORIGIN
    box_w = 420
    box_h = 56 * len(methods) + 60
    draw.rounded_rectangle((x0 - 25, y0 - 35, x0 + box_w, y0 + box_h), radius=18, fill="#fcfcfc", outline="#d9d9d9", width=2)
    draw.text((x0, y0 - 8), "Methods", font=FONT_AXIS, fill="#111111")
    for idx, method in enumerate(methods):
        y = y0 + 44 + idx * 54
        style = STYLE[method]
        draw.line((x0, y, x0 + 70, y), fill=style["color"], width=style["width"])
        if style["dash"] is not None:
            draw_dashed_polyline(draw, [(x0, y), (x0 + 70, y)], style["color"], style["width"], style["dash"])
        draw_marker(draw, (x0 + 35, y), method)
        draw.text((x0 + 92, y - 16), method, font=FONT_LEGEND, fill="#222222")


def draw_annotation(draw: ImageDraw.ImageDraw, endpoint: Tuple[float, float]) -> None:
    x, y = endpoint
    tx = x - 150
    ty = y - 78
    draw.line((tx + 110, ty + 34, x - 12, y - 8), fill=STYLE["HCF-Net"]["color"], width=3)
    draw.polygon([(x - 12, y - 8), (x - 24, y - 14), (x - 18, y - 1)], fill=STYLE["HCF-Net"]["color"])
    draw.rounded_rectangle((tx, ty, tx + 128, ty + 44), radius=12, fill="#fff3f3", outline="#f3b3b3", width=2)
    draw.text((tx + 14, ty + 8), "HCF-Net", font=FONT_ANNOT, fill=STYLE["HCF-Net"]["color"])


def draw_inset(draw: ImageDraw.ImageDraw, methods: List[str], curves: Dict[str, List[float]], kind: str) -> None:
    left, top, right, bottom = INSET_BOX
    draw.rounded_rectangle((left, top, right, bottom), radius=16, fill="white", outline="#bbbbbb", width=2)
    draw.text((left + 20, top + 12), f"Early stage zoom ({kind}, t=1-8)", font=FONT_SMALL, fill="#333333")

    plot_box = (left + 36, top + 44, right - 22, bottom - 26)
    draw.rectangle(plot_box, outline="#cccccc", width=1)
    x_vals = list(range(1, 9))
    y_pool = [float(curves[m][t]) for m in methods for t in range(8)]
    y_max = max(y_pool) * 1.12

    for tick in [1, 3, 5, 7]:
        x, _ = scale_point(tick, 0.0, 1, 8, 0.0, y_max, plot_box)
        draw.line((x, plot_box[1], x, plot_box[3]), fill="#ededed", width=1)
        draw_centered_text(draw, x, plot_box[3] + 20, str(tick), FONT_SMALL, "#555555")

    for tick in np.linspace(0.0, y_max, 4):
        _, y = scale_point(1, float(tick), 1, 8, 0.0, y_max, plot_box)
        draw.line((plot_box[0], y, plot_box[2], y), fill="#f0f0f0", width=1)

    for method in methods:
        y_vals = [float(v) for v in curves[method][:8]]
        draw_series(draw, plot_box, x_vals, y_vals, method, 0.0, y_max)


def write_reduced_curve_table(dataset: str, curves: Dict[str, List[float]], filename: str) -> None:
    methods = choose_methods(dataset, curves)
    rows = ["t," + ",".join(methods)]
    length = len(next(iter(curves.values())))
    for t in range(length):
        vals = ",".join(f"{float(curves[m][t]):.6f}" for m in methods)
        rows.append(f"{t + 1},{vals}")
    (OUT_DIR / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_one(dataset: str, curves: Dict[str, List[float]], kind: str, filename: str) -> None:
    methods = choose_methods(dataset, curves)
    if not methods:
        return

    image = Image.new("RGBA", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    x_vals = list(range(1, len(next(iter(curves.values()))) + 1))
    y_max = max(float(max(curves[m])) for m in methods)
    y_max *= 1.08

    draw_axes(image, draw, PLOT_BOX, x_vals[-1], y_max, f"{dataset} {kind} diffusion comparison")
    endpoints = {}
    for method in methods:
        y_vals = [float(v) for v in curves[method]]
        points = draw_series(draw, PLOT_BOX, x_vals, y_vals, method, 0.0, y_max)
        endpoints[method] = points[-1]

    draw_legend(draw, methods)
    draw_inset(draw, methods, curves, kind.replace("Semantic-", ""))
    draw_annotation(draw, endpoints["HCF-Net"])

    save_curve_exports(image, OUT_DIR / filename)


def main() -> None:
    report = json.loads(IN_PATH.read_text(encoding="utf-8"))
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        plot_one(dataset, report[dataset]["sir_curves"], "Semantic-SIR", f"{dataset}_sir_paper.png")
        plot_one(dataset, report[dataset]["si_curves"], "Semantic-SI", f"{dataset}_si_paper.png")
        write_reduced_curve_table(dataset, report[dataset]["sir_curves"], f"{dataset}_sir_paper_table.csv")
        write_reduced_curve_table(dataset, report[dataset]["si_curves"], f"{dataset}_si_paper_table.csv")
    print("saved paper-ready curve plots and reduced curve tables to", OUT_DIR)


if __name__ == "__main__":
    main()
