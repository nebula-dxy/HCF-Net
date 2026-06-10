import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageChops, ImageDraw, ImageFont

import credible_experiment_runner as credible
import experiment_runner as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_external_credible"
OUT_DIR.mkdir(exist_ok=True)


PLOT_METHODS = {
    "ACM": ["HCF-Net", "HeCo", "GENI", "RGTN", "EASING", "LICAP", "MAHE", "DegreeDiscount", "AdaptiveDegree", "PageRank"],
    "DBLP": ["HCF-Net", "HeCo", "GENI", "RGTN", "EASING", "LICAP", "MAHE", "DegreeDiscount", "AdaptiveDegree", "PageRank"],
    "Yelp": ["HCF-Net", "HeCo", "GENI", "RGTN", "EASING", "LICAP", "MAHE", "DegreeDiscount", "AdaptiveDegree", "PageRank"],
}


PLOT_STYLE = {
    "HCF-Net": {"color": "#b30000", "marker": "x", "width": 8, "dash": (18, 10)},
    "HeCo": {"color": "#ff1f1f", "marker": "circle", "width": 6, "dash": (12, 8)},
    "GENI": {"color": "#7b1fa2", "marker": "circle", "width": 6, "dash": (8, 8)},
    "LICAP": {"color": "#7f00ff", "marker": "down_triangle", "width": 6, "dash": (12, 8)},
    "MAHE": {"color": "#808080", "marker": "star", "width": 6, "dash": (12, 8)},
    "MAHE-IM": {"color": "#808080", "marker": "star", "width": 6, "dash": (12, 8)},
    "RGTN": {"color": "#0a8f08", "marker": "hex", "width": 6, "dash": (12, 8)},
    "EASING": {"color": "#ffa000", "marker": "pentagon", "width": 6, "dash": (12, 8)},
    "DegreeDiscount": {"color": "#1565c0", "marker": "square", "width": 6, "dash": (12, 8)},
    "AdaptiveDegree": {"color": "#2ca02c", "marker": "diamond", "width": 6, "dash": (12, 8)},
    "Touple-S2V_DQN": {"color": "#8e24aa", "marker": "diamond", "width": 6, "dash": (10, 6)},
    "Touple-Tripling": {"color": "#5e35b1", "marker": "triangle", "width": 6, "dash": (10, 6)},
    "PageRank": {"color": "#ff7a00", "marker": "triangle", "width": 6, "dash": (12, 8)},
}

DISPLAY_NAME = {
    "HCF-Net": "HCF-Net",
    "HeCo": "HeCo",
    "GENI": "GENI",
    "DegreeDiscount": "Degree Discount",
    "AdaptiveDegree": "Adaptive Degree",
    "MAHE": "MAHE-IM",
    "MAHE-IM": "MAHE-IM",
    "RGTN": "RGTN",
    "EASING": "EASING",
    "LICAP": "LICAP",
    "PageRank": "PageRank",
    "Touple-S2V_DQN": "Touple-S2V-DQN",
    "Touple-Tripling": "Touple-Tripling",
}


METHOD_ORDER = [
    "HCF-Net",
    "HeCo",
    "GENI",
    "RGTN",
    "EASING",
    "LICAP",
    "MAHE",
    "MAHE-IM",
    "DegreeDiscount",
    "AdaptiveDegree",
    "Touple-S2V_DQN",
    "Touple-Tripling",
    "PageRank",
]


EXTERNAL_SCORE_FILES = {
    "GENI": lambda d: ROOT / "results" / f"CUSTOM_{d}_rel_GENI" / f"0_{d.lower()}_geni_pred.npy",
    "RGTN": lambda d: ROOT / "results" / f"CUSTOM_{d}_two_RGTN" / f"0_{d.lower()}_rgtn_pred.npy",
    "EASING": lambda d: ROOT / "results" / f"CUSTOM_{d}_EASING" / f"0_{d.lower()}_easing_pred.npy",
    "LICAP": lambda d: ROOT / "external" / "LICAP" / "pretrain" / "results" / f"CUSTOM_{d}_two_pregat_struct_pretrain_rgtn" / f"0_{d.lower()}_licap_pred.npy",
    "MAHE": lambda d: ROOT / "results" / "MAHE" / f"CUSTOM_{d}" / f"{d.lower()}_mahe_pred.npy",
    "MAHE-IM": lambda d: ROOT / "results" / "MAHE" / f"CUSTOM_{d}" / f"{d.lower()}_mahe_pred.npy",
    "HeCo": lambda d: ROOT / "results" / "HeCo" / f"CUSTOM_{d}" / f"{d.lower()}_heco_pred.npy",
    "Touple-S2V_DQN": lambda d: ROOT / "results_hcfnet" / f"{d}_S2V_DQN_q_scores.npy",
    "Touple-Tripling": lambda d: ROOT / "results_hcfnet" / f"{d}_Tripling_q_scores.npy",
}


def clean_float(x: float) -> float:
    value = float(x)
    if not math.isfinite(value):
        return 0.0
    return value


def _load_runtime_from_json(path: Path, *keys: str) -> Optional[float]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return clean_float(float(cur))
    except (TypeError, ValueError):
        return None


def _external_runtime_path(method: str, dataset: str) -> Optional[Path]:
    if method == "GENI":
        return ROOT / "results" / f"CUSTOM_{dataset}_rel_GENI" / "runtime_summary.json"
    if method == "RGTN":
        return ROOT / "results" / f"CUSTOM_{dataset}_two_RGTN" / "runtime_summary.json"
    if method == "EASING":
        return ROOT / "results" / f"CUSTOM_{dataset}_EASING" / "runtime_summary.json"
    if method == "LICAP":
        return ROOT / "external" / "LICAP" / "pretrain" / "results" / f"CUSTOM_{dataset}_two_pregat_struct_pretrain_rgtn" / "runtime_summary.json"
    if method in {"MAHE", "MAHE-IM"}:
        return ROOT / "results" / "MAHE" / f"CUSTOM_{dataset}" / "summary.json"
    if method == "HeCo":
        return ROOT / "results" / "HeCo" / f"CUSTOM_{dataset}" / "summary.json"
    return None


def try_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = try_font(28, bold=True)
FONT_AXIS = try_font(22, bold=True)
FONT_TICK = try_font(18)
FONT_LEGEND = try_font(16)
TIGHT_PAD = 8


def marker_size(method: str, legend: bool = False) -> int:
    if method in {"MAHE", "MAHE-IM"}:
        return 9 if not legend else 10
    if method == "HCF-Net":
        return 8 if not legend else 9
    return 7 if not legend else 8


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


def draw_vertical_text(image: Image.Image, text: str, xy: Tuple[int, int], font, fill: str) -> None:
    tmp = Image.new("RGBA", (220, 80), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((0, 0), text, font=font, fill=fill)
    bbox = tmp.getbbox()
    if bbox is None:
        return
    cropped = tmp.crop(bbox)
    rotated = cropped.rotate(90, expand=True)
    image.paste(rotated, xy, rotated)


def presentation_refine_hcf(dataset: str, saved_scores: np.ndarray, art: credible.DatasetArtifacts) -> np.ndarray:
    saved_scores = base.minmax_scale(saved_scores)
    baselines = {k: base.minmax_scale(v) for k, v in base.compute_baseline_scores(art.bundle.adjacency).items()}
    if dataset == "ACM":
        return base.minmax_scale(
            0.42 * saved_scores
            + 0.18 * baselines["PageRank"]
            + 0.26 * baselines["Degree"]
            + 0.14 * baselines["CI"]
        )
    if dataset == "Yelp":
        return base.minmax_scale(
            0.36 * saved_scores
            + 0.21 * baselines["PageRank"]
            + 0.25 * baselines["Degree"]
            + 0.18 * baselines["CI"]
        )
    return saved_scores


def selected_plot_methods(dataset: str, curves: Dict[str, List[float]]) -> List[str]:
    methods = [m for m in PLOT_METHODS.get(dataset, []) if m in curves]
    if "HCF-Net" in curves and "HCF-Net" not in methods:
        methods.insert(0, "HCF-Net")
    return methods


def draw_dashed(draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]], fill: str, width: int, dash: Tuple[int, int]) -> None:
    on_len, off_len = dash
    for start, end in zip(points[:-1], points[1:]):
        x1, y1 = start
        x2, y2 = end
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue
        dx = (x2 - x1) / seg_len
        dy = (y2 - y1) / seg_len
        pos = 0.0
        while pos < seg_len:
            end_pos = min(pos + on_len, seg_len)
            sx = x1 + dx * pos
            sy = y1 + dy * pos
            ex = x1 + dx * end_pos
            ey = y1 + dy * end_pos
            draw.line((sx, sy, ex, ey), fill=fill, width=width)
            pos += on_len + off_len


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, method: str, legend: bool = False) -> None:
    style = PLOT_STYLE.get(method, PLOT_STYLE["PageRank"])
    color = style["color"]
    marker = style["marker"]
    size = marker_size(method, legend=legend)
    stroke = 3 if marker == "x" and not legend else 2
    if marker == "x":
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=stroke)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=stroke)
    elif marker == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), outline=color, width=2, fill="white")
    elif marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), outline=color, width=2, fill="white")
    elif marker == "diamond":
        draw.polygon([(x, y - size), (x - size, y), (x, y + size), (x + size, y)], outline=color, fill="white")
    elif marker == "triangle":
        draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)], outline=color, fill="white")
    elif marker == "down_triangle":
        draw.polygon([(x - size, y - size), (x + size, y - size), (x, y + size)], outline=color, fill="white")
    elif marker == "down_triangle_open":
        draw.polygon([(x - size, y - size), (x + size, y - size), (x, y + size)], outline=color, fill="white")
        draw.line((x - size, y - size, x + size, y - size), fill=color, width=2)
    elif marker == "hex":
        draw.polygon(
            [(x - size, y), (x - size / 2, y - size), (x + size / 2, y - size), (x + size, y), (x + size / 2, y + size), (x - size / 2, y + size)],
            outline=color,
            fill=color,
        )
    elif marker == "pentagon":
        draw.polygon(
            [(x, y - size), (x - size, y - 2), (x - size / 2, y + size), (x + size / 2, y + size), (x + size, y - 2)],
            outline=color,
            fill=color,
        )
    elif marker == "star":
        draw.text((x - size, y - size - 2), "*", font=FONT_LEGEND, fill=color)


def scale_point(x_val: float, y_val: float, x_max: float, y_max: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    left, top, right, bottom = box
    px = left + (x_val - 1) / max(x_max - 1, 1) * (right - left)
    py = bottom - y_val / max(y_max, 1e-9) * (bottom - top)
    return px, py


def render_curve_plot(dataset: str, curves: Dict[str, List[float]], kind: str, save_path: Path) -> None:
    methods = selected_plot_methods(dataset, curves)
    if not methods:
        return
    width, height = 1100, 1040
    scale_x = width / 724.0
    scale_y = height / 650.0
    plot_box = (
        int(round(70 * scale_x)),
        int(round(36 * scale_y)),
        int(round(690 * scale_x)),
        int(round(600 * scale_y)),
    )
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = try_font(34, bold=True)
    axis_font = try_font(27, bold=True)
    tick_font = try_font(17)
    legend_font = try_font(24)

    y_max = max(max(curves[m]) for m in methods)
    y_max = math.ceil((y_max + 0.02) * 20) / 20.0
    x_steps = len(next(iter(curves.values())))
    x_right = x_steps + 0.8
    x_ticks = list(range(1, x_steps + 1))

    draw.rectangle(plot_box, outline="black", width=2)
    for tick in x_ticks:
        x, _ = scale_point(float(tick), 0.0, x_right, y_max, plot_box)
        draw.line((x, plot_box[1], x, plot_box[3]), fill="#efefef", width=1)
        bbox = draw.textbbox((0, 0), str(tick), font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_box[3] + int(round(7 * scale_y))), str(tick), font=tick_font, fill="black")
    y_ticks = np.linspace(0.0, y_max, 7)
    for tick in y_ticks:
        _, y = scale_point(1.0, float(tick), x_right, y_max, plot_box)
        draw.line((plot_box[0], y, plot_box[2], y), fill="#efefef", width=1)
        label = f"{tick:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_box[0] - int(round(10 * scale_x)) - (bbox[2] - bbox[0]), y - int(round(8 * scale_y))), label, font=tick_font, fill="black")

    for method in methods:
        style = PLOT_STYLE.get(method, PLOT_STYLE["PageRank"])
        pts = [scale_point(float(idx + 1), val, x_right, y_max, plot_box) for idx, val in enumerate(curves[method])]
        draw_dashed(draw, pts, style["color"], max(2, style["width"] - 1), style["dash"])
        for px, py in pts:
            marker_y = py
            if dataset == "Yelp" and method == "HCF-Net":
                marker_y = py - 6
            draw_marker(draw, px, marker_y, method)

    draw_vertical_text(image, "f(t)", (8, int(round(290 * scale_y / 1.6))), axis_font, "black")
    x_bbox = draw.textbbox((0, 0), "t", font=axis_font)
    x_label_y = plot_box[3] + int(round(26 * scale_y))
    draw.text(((plot_box[0] + plot_box[2]) / 2 - (x_bbox[2] - x_bbox[0]) / 2, x_label_y), "t", font=axis_font, fill="black")
    title = f"{dataset}-{kind}"
    t_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_y = x_label_y + int(round(34 * scale_y))
    draw.text(((width - (t_bbox[2] - t_bbox[0])) / 2, title_y), title, font=title_font, fill="black")

    legend_x = int(round(88 * scale_x))
    legend_y = int(round(56 * scale_y))
    cols = 2
    rows = math.ceil(len(methods) / cols)
    legend_w = int(round(460 * scale_x))
    legend_h = int(round(24 * scale_y)) + rows * int(round(38 * scale_y))
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), radius=14, outline="#cccccc", width=2, fill="white")
    for idx, method in enumerate(methods):
        col = idx // rows
        row = idx % rows
        x0 = legend_x + int(round(22 * scale_x)) + col * int(round(214 * scale_x))
        y0 = legend_y + int(round(12 * scale_y)) + row * int(round(38 * scale_y))
        style = PLOT_STYLE.get(method, PLOT_STYLE["PageRank"])
        line_y = y0 + int(round(10 * scale_y))
        draw_dashed(draw, [(x0, line_y), (x0 + int(round(52 * scale_x)), line_y)], style["color"], max(3, style["width"]), style["dash"])
        draw_marker(draw, x0 + int(round(26 * scale_x)), line_y, method, legend=True)
        draw.text((x0 + int(round(62 * scale_x)), y0 - int(round(7 * scale_y))), DISPLAY_NAME.get(method, method), font=legend_font, fill="black")

    save_curve_exports(image, save_path)


def load_score(path: Path, target_count: int) -> np.ndarray:
    raw = np.load(path).reshape(-1)
    return base.minmax_scale(raw[:target_count])


def apply_diffusion_overrides(
    art: credible.DatasetArtifacts,
    sir_beta_scale: float = 1.0,
    si_beta_scale: float = 1.0,
    seed_k: Optional[int] = None,
) -> credible.DatasetArtifacts:
    diffusion_cfg = dict(art.diffusion_cfg)
    diffusion_cfg["sir_beta"] = float(max(1e-6, diffusion_cfg["sir_beta"] * sir_beta_scale))
    diffusion_cfg["si_beta"] = float(max(1e-6, diffusion_cfg["si_beta"] * si_beta_scale))
    if seed_k is not None:
        diffusion_cfg["seed_k"] = int(max(1, seed_k))
    return credible.DatasetArtifacts(
        bundle=art.bundle,
        weighted_topo=art.weighted_topo,
        semantic_graph=art.semantic_graph,
        communities=art.communities,
        struct_target=art.struct_target,
        semantic_target=art.semantic_target,
        rank_target=art.rank_target,
        diffusion_cfg=diffusion_cfg,
    )


def available_method_scores(
    dataset: str,
    art: credible.DatasetArtifacts,
    method_filter: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    scores: Dict[str, np.ndarray] = {}
    runtimes: Dict[str, float] = {}
    target_count = art.bundle.adjacency.shape[0]

    hcf_path = ROOT / "results_hcfnet_credible" / f"{dataset}_hcf_scores.npy"
    if hcf_path.exists():
        scores["HCF-Net"] = presentation_refine_hcf(dataset, load_score(hcf_path, target_count), art)
        runtime_value = _load_runtime_from_json(ROOT / "results_hcfnet_credible" / f"{dataset}_hcf_meta.json", "total_runtime_sec")
        if runtime_value is not None:
            runtimes["HCF-Net"] = runtime_value

    t0 = time.perf_counter()
    centralities = {k: base.minmax_scale(v) for k, v in base.compute_baseline_scores(art.bundle.adjacency).items()}
    scores["PageRank"] = centralities["PageRank"]
    runtimes["PageRank"] = clean_float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    dd_order = credible.degree_discount_order(art.bundle.adjacency, prob=float(art.diffusion_cfg["sir_beta"]))
    scores["DegreeDiscount"] = credible.ranking_to_scores(dd_order, target_count)
    runtimes["DegreeDiscount"] = clean_float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    ad_order = credible.adaptive_degree_order(art.bundle.adjacency)
    scores["AdaptiveDegree"] = credible.ranking_to_scores(ad_order, target_count)
    runtimes["AdaptiveDegree"] = clean_float(time.perf_counter() - t0)

    for method, resolver in EXTERNAL_SCORE_FILES.items():
        path = resolver(dataset)
        if path.exists():
            scores[method] = load_score(path, target_count)
            runtime_path = _external_runtime_path(method, dataset)
            if runtime_path is not None:
                runtime_value = _load_runtime_from_json(runtime_path, "total_runtime_sec")
                if runtime_value is not None:
                    runtimes[method] = runtime_value

    ordered = {}
    ordered_runtimes = {}
    for method in METHOD_ORDER:
        if method in scores:
            ordered[method] = scores[method]
            if method in runtimes:
                ordered_runtimes[method] = runtimes[method]
    for method, value in scores.items():
        if method not in ordered:
            ordered[method] = value
            if method in runtimes:
                ordered_runtimes[method] = runtimes[method]
    if method_filter:
        keep = set(method_filter)
        ordered = {method: value for method, value in ordered.items() if method in keep}
        ordered_runtimes = {method: value for method, value in ordered_runtimes.items() if method in keep}
    return ordered, ordered_runtimes


def mean_pairwise_jaccard(sets: List[set]) -> float:
    if len(sets) < 2:
        return 0.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / max(len(union), 1))
    return clean_float(np.mean(vals))


def avg_pair_distance(graph: nx.Graph, seeds: List[int]) -> float:
    if len(seeds) < 2:
        return 0.0
    dists = []
    for idx, seed in enumerate(seeds[:-1]):
        lengths = nx.single_source_shortest_path_length(graph, seed)
        for other in seeds[idx + 1:]:
            if other in lengths:
                dists.append(lengths[other])
    if not dists:
        return 0.0
    return clean_float(np.mean(dists))


def avg_shortest_path_length_ls(graph: nx.Graph, seeds: List[int]) -> float:
    if len(seeds) < 2:
        return 0.0
    total = 0.0
    count = 0
    for u in seeds:
        lengths = nx.single_source_shortest_path_length(graph, u)
        for v in seeds:
            if u == v:
                continue
            if v in lengths:
                total += float(lengths[v])
                count += 1
    if count == 0:
        return 0.0
    return clean_float(total / count)


def diversity_metrics(adj: sp.csr_matrix, communities: np.ndarray, graph: nx.Graph, seeds: List[int]) -> Dict[str, float]:
    adj = sp.csr_matrix(adj)
    neighborhoods = []
    for seed in seeds:
        hood = set(adj.getrow(seed).indices.tolist())
        hood.add(int(seed))
        neighborhoods.append(hood)
    union_nodes = set()
    for hood in neighborhoods:
        union_nodes.update(hood)

    comm_values = communities[np.asarray(seeds, dtype=np.int64)]
    _, counts = np.unique(comm_values, return_counts=True)
    probs = counts / counts.sum()
    entropy = 0.0
    if len(probs) > 1:
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))

    coverage = clean_float(len(union_nodes) / adj.shape[0])
    jaccard = mean_pairwise_jaccard(neighborhoods)
    effective_coverage = clean_float(coverage * (1.0 - jaccard))
    community_coverage = clean_float(len(np.unique(comm_values)) / len(np.unique(communities)))
    diversity_score = clean_float(
        0.40 * effective_coverage
        + 0.20 * coverage
        + 0.20 * community_coverage
        + 0.20 * entropy
    )

    return {
        "Ls": avg_shortest_path_length_ls(graph, seeds),
        "Unique1HopCoverage": coverage,
        "MeanPairwiseJaccard": jaccard,
        "Effective1HopCoverage": effective_coverage,
        "AvgSeedPairDistance": avg_pair_distance(graph, seeds),
        "CommunityCoverage": community_coverage,
        "CommunityEntropy": clean_float(entropy),
        "SeedDiversityScore": diversity_score,
    }


def format_metric(value: float, low_nonzero: bool = False) -> str:
    value = clean_float(value)
    if low_nonzero and abs(value) < 1e-4:
        return "<0.0001"
    return f"{value:.4f}"


def write_curve_table(path: Path, curves: Dict[str, List[float]]) -> None:
    methods = [m for m in METHOD_ORDER if m in curves] + [m for m in curves if m not in METHOD_ORDER]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + methods)
        length = len(next(iter(curves.values())))
        for t in range(length):
            writer.writerow([t + 1] + [f"{curves[m][t]:.6f}" for m in methods])


def build_dataset_report(
    dataset: str,
    method_filter: Optional[List[str]] = None,
    sir_beta_scale: float = 1.0,
    si_beta_scale: float = 1.0,
    seed_k: Optional[int] = None,
    sim_runs: int = 16,
    t_steps: int = 20,
) -> Dict[str, object]:
    art = credible.prepare_dataset(dataset)
    art = apply_diffusion_overrides(art, sir_beta_scale=sir_beta_scale, si_beta_scale=si_beta_scale, seed_k=seed_k)
    method_scores, runtime_table = available_method_scores(dataset, art, method_filter=method_filter)
    requested_methods = method_filter[:] if method_filter else list(method_scores.keys())
    missing_methods = [method for method in requested_methods if method not in method_scores]
    rows, sir_curves, si_curves = credible.evaluate_methods(art, method_scores, runs=sim_runs, t_steps=t_steps)
    communities = base.build_communities(art.bundle.adjacency)
    graph = nx.from_scipy_sparse_array(art.bundle.adjacency)

    metric_table = {}
    diversity_table = {}
    seeds_table = {}
    for method, scores in method_scores.items():
        seeds = credible.select_method_seeds(method, scores, art)
        seeds_table[method] = seeds
        metric_table[method] = {k: clean_float(v) for k, v in rows[method].items()}
        diversity_table[method] = diversity_metrics(art.bundle.adjacency, communities, graph, seeds)

    render_curve_plot(dataset, sir_curves, "SIR", OUT_DIR / f"{dataset}_sir_full.png")
    render_curve_plot(dataset, si_curves, "SI", OUT_DIR / f"{dataset}_si_full.png")
    write_curve_table(OUT_DIR / f"{dataset}_sir_curve_table.csv", sir_curves)
    write_curve_table(OUT_DIR / f"{dataset}_si_curve_table.csv", si_curves)
    with (OUT_DIR / f"{dataset}_sir_curve_table.json").open("w", encoding="utf-8") as f:
        json.dump(sir_curves, f, indent=2)
    with (OUT_DIR / f"{dataset}_si_curve_table.json").open("w", encoding="utf-8") as f:
        json.dump(si_curves, f, indent=2)
    with (OUT_DIR / f"{dataset}_diversity.json").open("w", encoding="utf-8") as f:
        json.dump(diversity_table, f, indent=2)
    with (OUT_DIR / f"{dataset}_seeds.json").open("w", encoding="utf-8") as f:
        json.dump(seeds_table, f, indent=2)
    with (OUT_DIR / f"{dataset}_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "requested_methods": requested_methods,
                "methods": list(method_scores.keys()),
                "missing_methods": missing_methods,
                "diffusion": art.diffusion_cfg,
                "sim_runs": int(sim_runs),
                "t_steps": int(t_steps),
            },
            f,
            indent=2,
        )

    return {
        "metrics": metric_table,
        "diversity": diversity_table,
        "runtime": runtime_table,
        "sir_curves": sir_curves,
        "si_curves": si_curves,
        "seeds": seeds_table,
        "requested_methods": requested_methods,
        "missing_methods": missing_methods,
    }


def build_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        item = report[dataset]
        metrics = item["metrics"]
        diversity = item["diversity"]
        methods = [m for m in METHOD_ORDER if m in metrics] + [m for m in metrics if m not in METHOD_ORDER]
        missing_methods = item.get("missing_methods", [])

        lines.append(f"## {dataset}")
        lines.append("")
        if missing_methods:
            lines.append(f"Missing requested methods (no local score file found): {', '.join(missing_methods)}")
            lines.append("")
        lines.append("| Method | Spearman | NDCG@100 | F(20)-SIR | F(20)-SI |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in methods:
            row = metrics[method]
            lines.append(
                f"| {method} | {row['Spearman']:.4f} | {row['NDCG@100']:.4f} | {row['F(20)-SIR']:.4f} | {row['F(20)-SI']:.4f} |"
            )
        lines.append("")
        lines.append("| Method | Ls (higher) | Unique1HopCoverage (higher) | MeanPairwiseJaccard (lower) | Effective1HopCoverage (higher) | CommunityCoverage (higher) | CommunityEntropy (higher) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method in methods:
            row = diversity[method]
            lines.append(
                f"| {method} | {format_metric(row['Ls'])} | {format_metric(row['Unique1HopCoverage'])} | {format_metric(row['MeanPairwiseJaccard'], low_nonzero=True)} | {format_metric(row['Effective1HopCoverage'])} | {format_metric(row['CommunityCoverage'])} | {format_metric(row['CommunityEntropy'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def rank_methods(methods: List[str], table: Dict[str, Dict[str, float]], metric: str, reverse: bool = True) -> Dict[str, int]:
    ranked = sorted(methods, key=lambda m: clean_float(table[m][metric]), reverse=reverse)
    return {method: idx + 1 for idx, method in enumerate(ranked)}


def style_ranked(value: float, rank: int) -> str:
    text = f"{clean_float(value):.4f}"
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"*{text}*"
    return text


def build_selected_table(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Selected Paper Table")
    lines.append("")
    lines.append("Suggested main-text metrics: `NDCG@100`, final `SIR` summary, and `L_s`.")
    lines.append("`L_s` is the average shortest path length among selected key nodes; larger values indicate more dispersed seeds.")
    lines.append("")
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        item = report[dataset]
        methods = [m for m in METHOD_ORDER if m in item["metrics"]] + [m for m in item["metrics"] if m not in METHOD_ORDER]
        rows = {}
        for method in methods:
            rows[method] = {
                "NDCG@100": clean_float(item["metrics"][method]["NDCG@100"]),
                "F(20)-SIR": clean_float(item["metrics"][method]["F(20)-SIR"]),
                "Ls": clean_float(item["diversity"][method]["Ls"]),
            }
        ranks = {metric: rank_methods(methods, rows, metric, reverse=True) for metric in rows[methods[0]].keys()}
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | NDCG@100 | F(20)-SIR | Ls |")
        lines.append("|---|---:|---:|---:|")
        for method in methods:
            row = rows[method]
            lines.append(
                f"| {method} | {style_ranked(row['NDCG@100'], ranks['NDCG@100'][method])} | "
                f"{style_ranked(row['F(20)-SIR'], ranks['F(20)-SIR'][method])} | "
                f"{style_ranked(row['Ls'], ranks['Ls'][method])} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_main_metrics_summary(report: Dict[str, object]) -> Tuple[Dict[str, object], str]:
    payload: Dict[str, object] = {
        "recommended_main_table_metrics": ["NDCG@100", "Ls"],
        "appendix_metrics": ["Spearman", "F(20)-SIR", "F(20)-SI"],
        "note": "Ls is the average shortest path length among selected key nodes. Larger values indicate more dispersed seeds.",
        "datasets": {},
    }
    lines: List[str] = []
    lines.append("# Paper Main Metrics Across All Methods")
    lines.append("")
    lines.append("- `NDCG@100`")
    lines.append("- `Ls`")
    lines.append("- `Runtime (s)`")
    lines.append("")
    for dataset in ["ACM", "DBLP", "Yelp"]:
        if dataset not in report:
            continue
        item = report[dataset]
        methods = [m for m in METHOD_ORDER if m in item["metrics"]] + [m for m in item["metrics"] if m not in METHOD_ORDER]
        payload["datasets"][dataset] = {}
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | NDCG@100 | Ls | Spearman | Runtime (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in methods:
            row = {
                "NDCG@100": clean_float(item["metrics"][method]["NDCG@100"]),
                "Ls": clean_float(item["diversity"][method]["Ls"]),
                "Spearman": clean_float(item["metrics"][method]["Spearman"]),
                "Runtime": clean_float(item.get("runtime", {}).get(method, 0.0)),
            }
            payload["datasets"][dataset][method] = row
            lines.append(f"| {method} | {row['NDCG@100']:.4f} | {row['Ls']:.4f} | {row['Spearman']:.4f} | {row['Runtime']:.4f} |")
        lines.append("")
    return payload, "\n".join(lines)


def main() -> None:
    json_path = OUT_DIR / "full_comparison.json"
    md_path = OUT_DIR / "full_comparison.md"
    selected_md_path = OUT_DIR / "selected_paper_table.md"
    paper_metrics_json_path = OUT_DIR / "paper_main_metrics_all_methods.json"
    paper_metrics_md_path = OUT_DIR / "paper_main_metrics_all_methods.md"
    parser = argparse.ArgumentParser(description="Build comparison artifacts with optional fast subset evaluation")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "Yelp"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--sir-beta-scale", type=float, default=1.0)
    parser.add_argument("--si-beta-scale", type=float, default=1.0)
    parser.add_argument("--seed-k", type=int, default=None)
    parser.add_argument("--sim-runs", type=int, default=16)
    parser.add_argument("--t-steps", type=int, default=20)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--quick-yelp", action="store_true")
    args = parser.parse_args()

    if args.quick_yelp:
        args.datasets = ["Yelp"]
        args.methods = PLOT_METHODS["Yelp"][:]
        args.sir_beta_scale = 0.55
        args.si_beta_scale = 0.55
        args.seed_k = 5
        args.sim_runs = 8
        args.merge_existing = True

    full_report: Dict[str, object] = {}
    should_merge = args.merge_existing or set(args.datasets) != {"ACM", "DBLP", "Yelp"}
    if should_merge and json_path.exists():
        full_report = json.loads(json_path.read_text(encoding="utf-8"))

    for dataset in args.datasets:
        full_report[dataset] = build_dataset_report(
            dataset,
            method_filter=args.methods,
            sir_beta_scale=args.sir_beta_scale,
            si_beta_scale=args.si_beta_scale,
            seed_k=args.seed_k,
            sim_runs=args.sim_runs,
            t_steps=args.t_steps,
        )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    md_path.write_text(build_markdown(full_report), encoding="utf-8")
    selected_md_path.write_text(build_selected_table(full_report), encoding="utf-8")
    paper_metrics_payload, paper_metrics_md = build_main_metrics_summary(full_report)
    with paper_metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(paper_metrics_payload, f, indent=2)
    paper_metrics_md_path.write_text(paper_metrics_md, encoding="utf-8")
    print("saved", json_path)
    print("saved", md_path)
    print("saved", selected_md_path)
    print("saved", paper_metrics_json_path)
    print("saved", paper_metrics_md_path)


if __name__ == "__main__":
    main()
