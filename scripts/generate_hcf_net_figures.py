from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.unicode_minus"] = False

CN_FONT = "SimSun"
EN_FONT = "Times New Roman"
CN_SIZE = 7.5
EN_SIZE = 7.5
LINE_W = 1.2


def zh(text):
    return text.encode("utf-8").decode("utf-8")


def add_box(ax, x, y, w, h, title_cn, subtitle_en=None, fc="#ffffff", ec="#333333", rounded=True):
    if rounded:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=LINE_W,
            edgecolor=ec,
            facecolor=fc,
        )
    else:
        box = patches.Rectangle((x, y), w, h, linewidth=LINE_W, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title_cn,
        ha="center",
        va="center",
        fontname=CN_FONT,
        fontsize=CN_SIZE,
        color="#1f1f1f",
    )
    if subtitle_en:
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle_en,
            ha="center",
            va="center",
            fontname=EN_FONT,
            fontsize=EN_SIZE,
            color="#333333",
        )
    return box


def add_arrow(ax, x1, y1, x2, y2, text=None, text_offset=(0, 0), color="#444444"):
    arrow = patches.FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=LINE_W,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3",
    )
    ax.add_patch(arrow)
    if text:
        mx = (x1 + x2) / 2 + text_offset[0]
        my = (y1 + y2) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontname=CN_FONT, fontsize=CN_SIZE, color=color)


def make_framework():
    fig = plt.figure(figsize=(12.5, 6.8))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "\u0048\u0043\u0046\u002d\u004e\u0065\u0074 \u65b9\u6cd5\u6846\u67b6\u56fe", fontname=CN_FONT, fontsize=9, weight="bold")
    ax.text(0.03, 0.915, "Framework of HCF-Net", fontname=EN_FONT, fontsize=8)

    add_box(ax, 0.04, 0.63, 0.17, 0.16, "\u8f93\u5165\u5c5e\u6027\u5f02\u6784\u4fe1\u606f\u7f51\u7edc", "Input AHIN", fc="#f7f7f7")
    add_box(ax, 0.04, 0.36, 0.17, 0.16, "\u8282\u70b9\u5c5e\u6027\u77e9\u9635", "Node Attributes", fc="#fff7e8")
    add_box(ax, 0.27, 0.63, 0.18, 0.16, "\u539f\u59cb\u62d3\u6251\u56fe", "Topology Graph", fc="#eaf2ff")
    add_box(ax, 0.27, 0.36, 0.18, 0.16, "\u8de8\u793e\u533a\u8bed\u4e49\u8fd1\u90bb\u56fe", "Semantic kNN Graph", fc="#eef8ea")

    add_box(ax, 0.51, 0.63, 0.16, 0.16, "\u62d3\u6251\u7f16\u7801\u5668", "Topology Encoder", fc="#dce9ff")
    add_box(ax, 0.51, 0.36, 0.16, 0.16, "\u8bed\u4e49\u7f16\u7801\u5668", "Semantic Encoder", fc="#dff2db")
    add_box(
        ax,
        0.72,
        0.48,
        0.17,
        0.18,
        "\u975e\u5bf9\u79f0\u5bf9\u6bd4\u5b66\u4e60\n\u505c\u6b62\u68af\u5ea6\u5bf9\u9f50",
        "Asymmetric Contrastive",
        fc="#fff0f0",
    )

    add_box(ax, 0.72, 0.77, 0.17, 0.10, "\u62d3\u6251\u5f97\u5206\u5206\u652f", "Topology Score", fc="#eaf2ff")
    add_box(ax, 0.72, 0.25, 0.17, 0.10, "\u8bed\u4e49\u5f97\u5206\u5206\u652f", "Semantic Score", fc="#eef8ea")
    add_box(ax, 0.91, 0.48, 0.07, 0.18, "\u665a\u671f\u878d\u5408", "Late\nFusion", fc="#fff7e8")
    add_box(ax, 0.91, 0.22, 0.07, 0.18, "\u90bb\u57df\u6298\u6263\n\u91cd\u6392\u5e8f", "Neighbor\nDiscount", fc="#f7f7f7")
    add_box(ax, 0.91, 0.77, 0.07, 0.10, "Top-K\n\u79cd\u5b50\u96c6", "Seeds", fc="#ffe8d8")

    add_arrow(ax, 0.21, 0.71, 0.27, 0.71)
    add_arrow(ax, 0.13, 0.63, 0.13, 0.52)
    add_arrow(ax, 0.21, 0.44, 0.27, 0.44)
    add_arrow(ax, 0.45, 0.71, 0.51, 0.71)
    add_arrow(ax, 0.45, 0.44, 0.51, 0.44)
    add_arrow(ax, 0.67, 0.71, 0.72, 0.57)
    add_arrow(ax, 0.67, 0.44, 0.72, 0.57)
    add_arrow(ax, 0.67, 0.71, 0.72, 0.82)
    add_arrow(ax, 0.67, 0.44, 0.72, 0.30)
    add_arrow(ax, 0.89, 0.82, 0.91, 0.82)
    add_arrow(ax, 0.89, 0.30, 0.91, 0.57)
    add_arrow(ax, 0.945, 0.48, 0.945, 0.40)
    add_arrow(ax, 0.945, 0.22, 0.945, 0.77)

    ax.text(0.58, 0.54, "h_t", fontname=EN_FONT, fontsize=EN_SIZE)
    ax.text(0.58, 0.27, "h_s", fontname=EN_FONT, fontsize=EN_SIZE)
    ax.text(0.77, 0.69, "stop-gradient", fontname=EN_FONT, fontsize=EN_SIZE, color="#a33a3a")
    ax.text(0.93, 0.69, r"$\hat{y}=\alpha y_t + (1-\alpha) y_s$", fontname=EN_FONT, fontsize=EN_SIZE, ha="center")
    ax.text(0.945, 0.15, "\u8f93\u51fa\u5173\u952e\u8282\u70b9\u6392\u5e8f", fontname=CN_FONT, fontsize=CN_SIZE, ha="center")

    for suffix in ("eps", "pdf"):
        fig.savefig(OUT_DIR / f"hcf_net_framework.{suffix}", format=suffix, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_flowchart():
    fig = plt.figure(figsize=(8.2, 10.6))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.07, 0.96, "\u0048\u0043\u0046\u002d\u004e\u0065\u0074 \u6d41\u7a0b\u56fe", fontname=CN_FONT, fontsize=9, weight="bold")
    ax.text(0.07, 0.93, "Flowchart of HCF-Net", fontname=EN_FONT, fontsize=8)

    steps = [
        ("\u8f93\u5165\u5c5e\u6027\u5f02\u6784\u4fe1\u606f\u7f51\u7edc\n\u4e0e\u8282\u70b9\u5c5e\u6027", "Input graph and attributes", "#f7f7f7"),
        ("\u6784\u5efa\u8de8\u793e\u533a\u8bed\u4e49\u8fd1\u90bb\u56fe", "Build semantic kNN graph", "#eef8ea"),
        ("\u53cc\u5206\u652f\u72ec\u7acb\u7f16\u7801\n\u62d3\u6251\u89c6\u56fe\u4e0e\u8bed\u4e49\u89c6\u56fe", "Dual-branch encoding", "#eaf2ff"),
        ("\u57fa\u4e8e\u505c\u6b62\u68af\u5ea6\u7684\n\u975e\u5bf9\u79f0\u5bf9\u6bd4\u5b66\u4e60", "Asymmetric contrastive", "#fff0f0"),
        ("\u8ba1\u7b97\u62d3\u6251\u5f97\u5206\n\u4e0e\u8bed\u4e49\u5f97\u5206", "Score prediction", "#fff7e8"),
        ("\u665a\u671f\u5f97\u5206\u878d\u5408", "Late score fusion", "#fff7e8"),
        ("\u90bb\u57df\u6298\u6263\u91cd\u6392\u5e8f", "Neighbor discount reranking", "#f7f7f7"),
        ("\u8f93\u51fa Top-K \u5173\u952e\u8282\u70b9", "Output seeds", "#ffe8d8"),
    ]

    y = 0.84
    boxes = []
    for title, sub, color in steps:
        box = add_box(ax, 0.18, y, 0.64, 0.08, title, sub, fc=color)
        boxes.append(box)
        y -= 0.105

    for i in range(len(boxes) - 1):
        x1 = 0.50
        y1 = boxes[i].get_y()
        x2 = 0.50
        y2 = boxes[i + 1].get_y() + boxes[i + 1].get_height()
        add_arrow(ax, x1, y1, x2, y2)

    ax.text(0.85, 0.68, "\u8bad\u7ec3\u9636\u6bb5", fontname=CN_FONT, fontsize=CN_SIZE, rotation=90, color="#555555")
    ax.text(0.85, 0.42, "\u63a8\u7406\u9636\u6bb5", fontname=CN_FONT, fontsize=CN_SIZE, rotation=90, color="#555555")
    ax.plot([0.14, 0.86], [0.515, 0.515], color="#aaaaaa", linewidth=0.9, linestyle="--")

    for suffix in ("eps", "pdf"):
        fig.savefig(OUT_DIR / f"hcf_net_flowchart.{suffix}", format=suffix, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_framework()
    make_flowchart()
