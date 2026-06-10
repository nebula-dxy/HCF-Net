from pathlib import Path
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_CN = "\u65b9\u6b63\u4e66\u5b8b\u7b80\u4f53"
FONT_EN = "Times New Roman"


def S(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def add(parent, tag, text=None, **attrs):
    elem = ET.SubElement(parent, S(tag), {k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None})
    if text is not None:
        elem.text = text
    return elem


def add_text(parent, x, y, text, cls="t", anchor="middle"):
    return add(parent, "text", text=text, x=x, y=y, **{"class": cls, "text-anchor": anchor, "dominant-baseline": "middle"})


def add_rect(parent, x, y, w, h, cls, rx=14, ry=14):
    return add(parent, "rect", x=x, y=y, width=w, height=h, rx=rx, ry=ry, **{"class": cls})


def add_circle(parent, cx, cy, r, cls):
    return add(parent, "circle", cx=cx, cy=cy, r=r, **{"class": cls})


def add_line(parent, x1, y1, x2, y2, cls, arrow=False):
    attrs = {"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if arrow:
        attrs["marker_end"] = "url(#arrow)"
    return add(parent, "line", **attrs)


def add_polyline(parent, points, cls, arrow=False):
    attrs = {"class": cls, "points": " ".join(f"{x},{y}" for x, y in points), "fill": "none"}
    if arrow:
        attrs["marker_end"] = "url(#arrow)"
    return add(parent, "polyline", **attrs)


def add_dashed_path(parent, points, cls, marker="url(#arrowRed)"):
    attrs = {"class": cls, "points": " ".join(f"{x},{y}" for x, y in points), "fill": "none", "marker_end": marker}
    return add(parent, "polyline", **attrs)


def add_polygon(parent, points, cls):
    return add(parent, "polygon", points=" ".join(f"{x},{y}" for x, y in points), **{"class": cls})


def score_vector(parent, x, y, colors, label):
    add_rect(parent, x, y, 124, 48, "vector", rx=10, ry=10)
    for idx, color in enumerate(colors):
        add(parent, "rect", x=x + 12 + idx * 18, y=y + 14, width=12, height=20, rx=2, ry=2, fill=color, stroke="#969892", stroke_width="0.7")
    add_text(parent, x + 62, y + 62, label, cls="s")


def small_matrix(parent, x, y):
    colors = ["#d6ebfa", "#9ccaf0", "#5aa7df", "#2e7ec2"]
    for row in range(4):
        for col in range(4):
            add(parent, "rect", x=x + col * 16, y=y + row * 16, width=12, height=12, rx=2, ry=2, fill=colors[(row + col) % len(colors)], stroke="#97a5b2", stroke_width="0.5")


def attribute_rows(parent, x, y):
    row_colors = [
        ("#e68b92", "#f5d5d8"),
        ("#86bdb5", "#d7eee9"),
        ("#e0bf57", "#f8efc7"),
    ]
    for row, (c_dark, c_light) in enumerate(row_colors):
        for col in range(10):
            fill = c_dark if col in {0, 2, 5, 8} else c_light
            add(parent, "rect", x=x + col * 18, y=y + row * 22, width=13, height=13, rx=1.8, ry=1.8, fill=fill, stroke=c_dark, stroke_width="0.7")


def similarity_heatmap(parent, x, y):
    palette = [
        ["#d6ebfa", "#c4e1f8", "#9ccaf0", "#77b2e2", "#4f96cf"],
        ["#c4e1f8", "#d6ebfa", "#b5d8f5", "#88bce8", "#5fa1d8"],
        ["#9ccaf0", "#b5d8f5", "#d6ebfa", "#9ccaf0", "#6ca9dd"],
        ["#77b2e2", "#88bce8", "#9ccaf0", "#d6ebfa", "#8abde8"],
        ["#4f96cf", "#5fa1d8", "#6ca9dd", "#8abde8", "#d6ebfa"],
    ]
    for row in range(5):
        for col in range(5):
            add(parent, "rect", x=x + col * 13, y=y + row * 13, width=10, height=10, rx=1.6, ry=1.6, fill=palette[row][col], stroke="#93a9bc", stroke_width="0.5")
    for idx in range(4):
        add_circle(parent, x - 12, y + 5 + idx * 16, 3.2, "node_g_dim" if idx % 2 == 0 else "node_o_dim")
        add_circle(parent, x + 5 + idx * 16, y - 12, 3.2, "node_r_dim" if idx % 2 == 0 else "node_g_dim")


def draw_type_node(parent, cx, cy, r, cls, label, dx=0, dy=-22):
    add_circle(parent, cx, cy, r, cls)
    add_text(parent, cx + dx, cy + dy, label, cls="z")


def draw_input_network(parent):
    add_rect(parent, 36, 112, 238, 640, "panel", rx=18, ry=18)
    add_text(parent, 155, 140, "\u5c5e\u6027\u5f02\u6784\u7f51\u7edc", cls="u")
    edges = [(82, 240, 136, 190), (136, 190, 214, 222), (82, 240, 98, 330), (98, 330, 168, 314), (168, 314, 228, 334), (136, 190, 168, 314), (214, 222, 168, 314)]
    for x1, y1, x2, y2 in edges:
        add_line(parent, x1, y1, x2, y2, "edge_mix")
    draw_type_node(parent, 82, 240, 15, "node_g", "\u4f5c\u8005A")
    draw_type_node(parent, 136, 190, 15, "node_o", "\u8bba\u6587P1")
    draw_type_node(parent, 214, 222, 15, "node_r", "\u4e3b\u9898T1")
    draw_type_node(parent, 168, 314, 15, "node_o", "\u8bba\u6587P2")
    draw_type_node(parent, 98, 330, 15, "node_g", "\u4f5c\u8005B")
    draw_type_node(parent, 228, 334, 15, "node_r", "\u4e3b\u9898T2")
    add_rect(parent, 86, 500, 138, 106, "chip", rx=14, ry=14)
    add_text(parent, 155, 526, "\u8282\u70b9\u5c5e\u6027", cls="s")
    attribute_rows(parent, 107, 548)
    add_text(parent, 155, 622, "\u8bed\u4e49\u7279\u5f81", cls="zs")
    add_polyline(parent, [(152, 352), (152, 420), (155, 500)], "flow_soft", arrow=True)


def draw_topology_view(parent):
    add_rect(parent, 320, 124, 224, 182, "topo_panel", rx=16, ry=16)
    add_text(parent, 432, 150, "\u7269\u7406\u62d3\u6251\u7ed3\u6784\u56fe", cls="u")
    edges = [(350, 220, 398, 184), (398, 184, 456, 214), (350, 220, 404, 274), (404, 274, 472, 268), (456, 214, 404, 274)]
    for x1, y1, x2, y2 in edges:
        add_line(parent, x1, y1, x2, y2, "edge_g")
    for cx, cy, cls in [(350, 220, "node_g"), (398, 184, "node_o"), (456, 214, "node_r"), (404, 274, "node_o"), (472, 268, "node_g")]:
        add_circle(parent, cx, cy, 10, cls)


def draw_similarity_block(parent):
    add_rect(parent, 320, 434, 108, 126, "sema_panel", rx=14, ry=14)
    add_text(parent, 374, 460, "\u5c5e\u6027\u76f8\u4f3c\u5ea6", cls="s")
    similarity_heatmap(parent, 352, 490)


def draw_semantic_view(parent):
    add_rect(parent, 434, 378, 176, 236, "sema_panel", rx=16, ry=16)
    add_text(parent, 522, 404, "\u8de8\u793e\u533a\u8bed\u4e49\u8fd1\u90bb\u56fe", cls="u")
    add(parent, "ellipse", cx=486, cy=496, rx=44, ry=34, fill="#f6eceb", stroke="#d7c4c1", stroke_width="1.0")
    add(parent, "ellipse", cx=560, cy=520, rx=42, ry=36, fill="#edf4fb", stroke="#c6d7e7", stroke_width="1.0")
    add(parent, "ellipse", cx=516, cy=572, rx=52, ry=28, fill="#f7f1df", stroke="#dbc88d", stroke_width="1.0")
    solid_edges = [(466, 488, 510, 482), (466, 488, 480, 530), (548, 498, 572, 530), (548, 498, 528, 560), (492, 586, 528, 560)]
    dash_edges = [(510, 482, 548, 498), (480, 530, 528, 560), (466, 488, 572, 530)]
    for x1, y1, x2, y2 in solid_edges:
        add_line(parent, x1, y1, x2, y2, "edge_b")
    for x1, y1, x2, y2 in dash_edges:
        add_line(parent, x1, y1, x2, y2, "edge_bd")
    for cx, cy, cls in [(466, 488, "node_g"), (510, 482, "node_r"), (480, 530, "node_r"), (548, 498, "node_o"), (572, 530, "node_r"), (528, 560, "node_g"), (492, 586, "node_o")]:
        add_circle(parent, cx, cy, 8.5, cls)


def draw_encoder(parent, x, y, title, branch_cls, preview_kind="topo", z_label="z"):
    add_rect(parent, x, y, 360, 220, "module", rx=18, ry=18)
    add_text(parent, x + 180, y + 28, title, cls="u")
    add_rect(parent, x + 18, y + 50, 92, 126, branch_cls, rx=12, ry=12)
    if preview_kind == "topo":
        add_text(parent, x + 64, y + 72, "A", cls="zs")
        for e in [(x + 38, y + 98, x + 64, y + 82), (x + 64, y + 82, x + 90, y + 98), (x + 38, y + 98, x + 68, y + 140), (x + 68, y + 140, x + 94, y + 146), (x + 90, y + 98, x + 68, y + 140)]:
            add_line(parent, e[0], e[1], e[2], e[3], "edge_g")
        for cx, cy, cls in [(x + 38, y + 98, "node_g"), (x + 64, y + 82, "node_o"), (x + 90, y + 98, "node_r"), (x + 68, y + 140, "node_o"), (x + 94, y + 146, "node_g")]:
            add_circle(parent, cx, cy, 8, cls)
    else:
        add_text(parent, x + 64, y + 72, "A\u2032", cls="zs")
        add(parent, "ellipse", cx=x + 54, cy=y + 112, rx=24, ry=18, fill="#f6eceb", stroke="#d7c4c1", stroke_width="0.8")
        add(parent, "ellipse", cx=x + 78, cy=y + 124, rx=24, ry=18, fill="#edf4fb", stroke="#c6d7e7", stroke_width="0.8")
        add_line(parent, x + 40, y + 104, x + 58, y + 98, "edge_b")
        add_line(parent, x + 40, y + 104, x + 46, y + 130, "edge_b")
        add_line(parent, x + 78, y + 106, x + 92, y + 124, "edge_b")
        add_line(parent, x + 58, y + 98, x + 78, y + 106, "edge_bd")
        for cx, cy, cls in [(x + 40, y + 104, "node_g"), (x + 58, y + 98, "node_r"), (x + 46, y + 130, "node_r"), (x + 78, y + 106, "node_o"), (x + 92, y + 124, "node_g")]:
            add_circle(parent, cx, cy, 7, cls)
    layer_x = [x + 132, x + 206, x + 280]
    layer_titles = ["\u56fe\u5377\u79ef\u5c421", "\u56fe\u5377\u79ef\u5c422", "\u56fe\u5377\u79ef\u5c423"]
    layer_descs = ["\u90bb\u5c45\u805a\u5408", "\u7279\u5f81\u4f20\u64ad", "\u8868\u5f81\u8f93\u51fa"]
    layer_ops = ["GCNConv + ReLU", "GCNConv + ReLU", "GCNConv"]
    for idx, lx in enumerate(layer_x):
        add_rect(parent, lx, y + 58, 58, 112, branch_cls, rx=10, ry=10)
        add_text(parent, lx + 29, y + 84, layer_titles[idx], cls="zs")
        add_text(parent, lx + 29, y + 116, layer_descs[idx], cls="z")
        add_text(parent, lx + 29, y + 144, layer_ops[idx], cls="en_small")
    add_rect(parent, x + 336, y + 82, 14, 64, "chip", rx=7, ry=7)
    add_text(parent, x + 343, y + 114, z_label, cls="en_small")
    add_line(parent, x + 110, y + 114, x + 132, y + 114, "flow", arrow=True)
    add_line(parent, x + 190, y + 114, x + 206, y + 114, "flow", arrow=True)
    add_line(parent, x + 264, y + 114, x + 280, y + 114, "flow", arrow=True)


def draw_scoring_head(parent, x, y, title, panel_cls, vector_colors, label):
    add_rect(parent, x, y, 236, 126, panel_cls, rx=16, ry=16)
    add_text(parent, x + 118, y + 24, title, cls="u")
    add_polygon(parent, [(x + 20, y + 60), (x + 48, y + 70), (x + 48, y + 98), (x + 20, y + 108)], "mlp")
    add_polygon(parent, [(x + 56, y + 66), (x + 82, y + 74), (x + 82, y + 94), (x + 56, y + 102)], "mlp")
    add_polygon(parent, [(x + 90, y + 70), (x + 112, y + 76), (x + 112, y + 92), (x + 90, y + 98)], "mlp")
    add_text(parent, x + 66, y + 48, "MLP", cls="en")
    add_line(parent, x + 114, y + 84, x + 138, y + 84, "flow", arrow=True)
    score_vector(parent, x + 138, y + 60, vector_colors, label)


def draw_fusion(parent):
    add_circle(parent, 1310, 388, 24, "fusion")
    add_text(parent, 1310, 388, "+", cls="t")
    score_vector(parent, 1350, 354, ["#76b48d", "#a791ce", "#6ea9d8", "#d5b14b", "#8ec7a8", "#7b88cf"], "\u7efc\u5408\u5f97\u5206\u5411\u91cf")


def draw_rerank(parent):
    add_rect(parent, 1492, 246, 248, 324, "warn_panel", rx=18, ry=18)
    add_text(parent, 1616, 272, "\u90bb\u57df\u6298\u6263\u91cd\u6392\u5e8f", cls="u")
    score_vector(parent, 1554, 296, ["#76b48d", "#a791ce", "#6ea9d8", "#d5b14b", "#8ec7a8", "#7b88cf"], "\u521d\u59cb\u5019\u9009\u5f97\u5206")
    add_polyline(parent, [(1616, 344), (1616, 360)], "flow", arrow=True)
    graph_edges = [(1544, 430, 1610, 388), (1610, 388, 1674, 424), (1544, 430, 1564, 498), (1564, 498, 1612, 530), (1612, 530, 1672, 500), (1672, 500, 1694, 438), (1610, 388, 1612, 530), (1674, 424, 1694, 438)]
    for x1, y1, x2, y2 in graph_edges:
        add_line(parent, x1, y1, x2, y2, "edge_g")
    for x1, y1, x2, y2 in [(1610, 388, 1544, 430), (1610, 388, 1674, 424), (1610, 388, 1612, 530)]:
        add_line(parent, x1, y1, x2, y2, "edge_red_dash")
    for cx, cy, cls in [(1544, 430, "node_g"), (1610, 388, "node_o"), (1674, 424, "node_r_dim"), (1564, 498, "node_g_dim"), (1612, 530, "node_o_dim"), (1672, 500, "node_r"), (1694, 438, "node_g_dim")]:
        add_circle(parent, cx, cy, 9.5, cls)
    add(parent, "circle", cx=1610, cy=388, r=18, fill="none", stroke="#f0ae46", stroke_width="2.2")
    for cx, cy in [(1544, 430), (1674, 424), (1612, 530)]:
        add(parent, "circle", cx=cx, cy=cy, r=13, fill="#ffffff", stroke="#d95f4f", stroke_width="1.3")
        add_text(parent, cx, cy + 1, "-", cls="minus")
    add_text(parent, 1616, 548, "\u9009\u4e2d\u9ad8\u5206\u8282\u70b9\u540e\u5bf9\u4e00\u8df3\u90bb\u5c45\u6298\u6263", cls="s")


def draw_output(parent):
    add_rect(parent, 1760, 266, 104, 286, "panel", rx=18, ry=18)
    add_text(parent, 1812, 292, "\u5168\u5c40\u5173\u952e\u8282\u70b9\u96c6\u5408", cls="u")
    for cx, cy, cls in [(1788, 356, "node_r"), (1836, 336, "node_g"), (1800, 422, "node_o"), (1844, 456, "node_r"), (1810, 506, "node_g")]:
        add(parent, "circle", cx=cx, cy=cy, r=17, fill="none", stroke="#efb257", stroke_width="2.0")
        add_circle(parent, cx, cy, 11, cls)
    add_text(parent, 1812, 532, "\u7a7a\u95f4\u5206\u6563\u4e14\u4e92\u4e0d\u91cd\u53e0", cls="s")


def build_svg():
    svg = ET.Element(S("svg"), width="1880", height="820", viewBox="0 0 1880 820", version="1.1")
    defs = add(svg, "defs")
    style = f"""
.section{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:22px;fill:#43453f;font-weight:400}}
.t{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:26px;fill:#43453f;font-weight:400}}
.u{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:18px;fill:#43453f;font-weight:400}}
.s{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:14px;fill:#656960;font-weight:400}}
.z{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:12px;fill:#5b5e58;font-weight:400}}
.zs{{font-family:'{FONT_CN}','FZShuSong-Z01S','STSong','SimSun','Noto Serif SC',serif;font-size:11px;fill:#5b5e58;font-weight:400}}
.en{{font-family:'{FONT_EN}',serif;font-size:16px;fill:#444842;font-weight:400}}
.en_small{{font-family:'{FONT_EN}',serif;font-size:12px;fill:#444842;font-weight:400}}
.minus{{font-family:'{FONT_EN}',serif;font-size:18px;fill:#d95f4f;font-weight:400}}
.panel{{fill:#fffefa;stroke:#787a73;stroke-width:1.2}}
.module{{fill:#fffefa;stroke:#787a73;stroke-width:1.2}}
.chip{{fill:#ffffff;stroke:#888b84;stroke-width:1.0}}
.vector{{fill:#ffffff;stroke:#8d9089;stroke-width:1.0}}
.topo_panel{{fill:#eef7f0;stroke:#79b08b;stroke-width:1.1}}
.sema_panel{{fill:#edf5fd;stroke:#6fa7d5;stroke-width:1.1}}
.warn_panel{{fill:#fff6f1;stroke:#e18663;stroke-width:1.1}}
.fusion{{fill:#ffffff;stroke:#8f928c;stroke-width:1.2}}
.mlp{{fill:#ffffff;stroke:#7c8078;stroke-width:1.0}}
.flow{{stroke:#41443f;stroke-width:1.6;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.flow_soft{{stroke:#62665f;stroke-width:1.4;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.flow_red{{stroke:#d95f4f;stroke-width:1.6;fill:none;stroke-dasharray:7 5;stroke-linecap:round;stroke-linejoin:round}}
.edge_mix{{stroke:#7a9c92;stroke-width:1.35;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.edge_g{{stroke:#79b08b;stroke-width:1.35;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.edge_b{{stroke:#6fa7d5;stroke-width:1.35;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.edge_bd{{stroke:#6fa7d5;stroke-width:1.35;fill:none;stroke-dasharray:6 5;stroke-linecap:round;stroke-linejoin:round}}
.edge_red_dash{{stroke:#d95f4f;stroke-width:1.25;fill:none;stroke-dasharray:6 5;stroke-linecap:round;stroke-linejoin:round}}
.chipdot{{fill:#ffffff;stroke:#8b8e88;stroke-width:0.8}}
.node_g{{fill:#7eb8af;stroke:#5d958c;stroke-width:1.0}}
.node_o{{fill:#d7b24f;stroke:#af8c2f;stroke-width:1.0}}
.node_r{{fill:#e18186;stroke:#b66469;stroke-width:1.0}}
.node_g_dim{{fill:#7eb8af;stroke:#5d958c;stroke-width:1.0;opacity:0.35}}
.node_o_dim{{fill:#d7b24f;stroke:#af8c2f;stroke-width:1.0;opacity:0.35}}
.node_r_dim{{fill:#e18186;stroke:#b66469;stroke-width:1.0;opacity:0.35}}
""".strip()
    add(defs, "style", text=style)
    m1 = add(defs, "marker", id="arrow", markerWidth="8", markerHeight="8", refX="7", refY="4", orient="auto")
    add(m1, "path", d="M 0 0 L 7 4 L 0 8 z", fill="#41443f")
    m2 = add(defs, "marker", id="arrowRed", markerWidth="8", markerHeight="8", refX="7", refY="4", orient="auto")
    add(m2, "path", d="M 0 0 L 7 4 L 0 8 z", fill="#d95f4f")
    add(svg, "rect", x="0", y="0", width="1880", height="820", fill="#ffffff")
    for cx, label in [(155, "\u8f93\u5165"), (465, "\u53cc\u89c6\u56fe\u6784\u5efa"), (805, "\u53cc\u901a\u9053\u7f16\u7801\u4e0e\u975e\u5bf9\u79f0\u5bf9\u6bd4\u5b66\u4e60"), (1205, "\u89e3\u8026\u6253\u5206\u4e0e\u665a\u671f\u878d\u5408"), (1616, "\u90bb\u57df\u6298\u6263\u91cd\u6392\u5e8f"), (1812, "\u8f93\u51fa")]:
        add_text(svg, cx, 54, label, cls="section")
    draw_input_network(svg)
    draw_topology_view(svg)
    draw_similarity_block(svg)
    draw_semantic_view(svg)
    draw_encoder(svg, 640, 118, "\u62d3\u6251\u795e\u7ecf\u7f51\u7edc\u5c42", "topo_panel", "topo", "z^T")
    draw_encoder(svg, 640, 432, "\u8bed\u4e49\u795e\u7ecf\u7f51\u7edc\u5c42", "sema_panel", "sema", "z^S")
    draw_scoring_head(svg, 1040, 162, "\u62d3\u6251\u8bc4\u5206\u5934", "topo_panel", ["#79b08b", "#8cc19a", "#6ea79d", "#9dd0ad", "#5f9286", "#7bb492"], "\u62d3\u6251\u5f97\u5206\u5411\u91cf")
    draw_scoring_head(svg, 1040, 482, "\u8bed\u4e49\u8bc4\u5206\u5934", "sema_panel", ["#6fa7d5", "#82b7df", "#5c97c8", "#97c8ea", "#4e87b3", "#76add9"], "\u8bed\u4e49\u5f97\u5206\u5411\u91cf")
    draw_fusion(svg)
    draw_rerank(svg)
    draw_output(svg)
    add_polyline(svg, [(228, 236), (320, 236)], "flow", arrow=True)
    add_polyline(svg, [(224, 552), (320, 552)], "flow", arrow=True)
    add_polyline(svg, [(428, 498), (434, 498)], "flow", arrow=True)
    add_polyline(svg, [(544, 216), (640, 216)], "flow", arrow=True)
    add_polyline(svg, [(610, 498), (640, 498)], "flow", arrow=True)
    add_polyline(svg, [(990, 232), (1040, 232)], "flow", arrow=True)
    add_polyline(svg, [(990, 546), (1040, 546)], "flow", arrow=True)
    add_dashed_path(svg, [(983, 546), (983, 388), (983, 232)], "flow_red")
    add(svg, "circle", cx="983", cy="388", r="18", fill="#e56f59", stroke="#bf5341", stroke_width="1.2")
    add_text(svg, 983, 389, "sg(\u00b7)", cls="en_small")
    add_text(svg, 1016, 362, "\u975e\u5bf9\u79f0\u5bf9\u6bd4\u5b66\u4e60", cls="s", anchor="start")
    add_polyline(svg, [(1288, 246), (1310, 246), (1310, 364)], "flow", arrow=True)
    add_polyline(svg, [(1288, 566), (1310, 566), (1310, 412)], "flow", arrow=True)
    add_polyline(svg, [(1334, 388), (1350, 388)], "flow", arrow=True)
    add_polyline(svg, [(1474, 388), (1492, 388)], "flow", arrow=True)
    add_polyline(svg, [(1740, 408), (1760, 408)], "flow", arrow=True)
    xml_bytes = ET.tostring(svg, encoding="utf-8", xml_declaration=True)
    for out_name in ["hcf_net_framework.svg", "hcf_net_framework_fzss.svg"]:
        (OUT_DIR / out_name).write_bytes(xml_bytes)


if __name__ == "__main__":
    build_svg()
