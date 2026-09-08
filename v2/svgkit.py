# -*- coding: utf-8 -*-
"""極小的 SVG 產生器，讓圖表原始碼保持可讀。"""
from __future__ import annotations

C = {  # 對應 CSS 變數
    "compute": "#1f6feb", "mem": "#c2740a", "ssm": "#0d8f84", "moe": "#7048d8",
    "spec": "#c81e78", "ok": "#15954b", "bad": "#d3392b", "neutral": "#7b8794",
    "computeW": "#eaf1fe", "memW": "#fdf3e3", "ssmW": "#e4f6f4", "moeW": "#f1ecfd",
    "specW": "#fdeaf4", "okW": "#e6f6ec", "badW": "#fdecea", "line": "#e3e6ea",
    "line2": "#eef0f3", "ink": "#151a21", "ink2": "#3c4652", "mut": "#6d7885",
    "mut2": "#98a1ac", "card2": "#fafbfc", "white": "#ffffff",
}


def _a(kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        k = {"s": "data-s", "dim": "data-dur", "cls": "class", "dur": "style"}.get(k, k)
        k = k.replace("_", "-")
        out.append(f'{k}="{v}"')
    return (" " + " ".join(out)) if out else ""


def el(tag, body=None, **kw):
    a = _a(kw)
    return f"<{tag}{a}>{body}</{tag}>" if body is not None else f"<{tag}{a}/>"


def g(body, s=None, cls=None, **kw):
    if s is not None:
        kw["data-s"] = s
        cls = ((cls + " ") if cls else "") + "stp"
    if cls:
        kw["class"] = cls
    return el("g", "".join(body) if isinstance(body, (list, tuple)) else body, **kw)


def rect(x, y, w, h, r=6, fill=C["white"], stroke=C["line"], sw=1, s=None, cls=None, **kw):
    if s is not None:
        kw["data-s"] = s
        cls = ((cls + " ") if cls else "") + "stp"
    if cls:
        kw["class"] = cls
    return el("rect", None, x=round(x, 2), y=round(y, 2), width=round(w, 2), height=round(h, 2),
              rx=r, fill=fill, stroke=stroke, **({"stroke-width": sw} if stroke else {}), **kw)


def txt(x, y, s, cls="lbl", anchor="start", fill=None, size=None, weight=None, ds=None, **kw):
    st = []
    if fill:
        st.append(f"fill:{fill}")
    if size:
        st.append(f"font-size:{size}px")
    if weight:
        st.append(f"font-weight:{weight}")
    if ds is not None:
        kw["data-s"] = ds
        cls = cls + " stp"
    return el("text", s, x=round(x, 2), y=round(y, 2), **{"class": cls,
              "text-anchor": {"start": "start", "middle": "middle", "end": "end"}[anchor]},
              style=";".join(st) or None, **kw)


def line(x1, y1, x2, y2, stroke=C["line"], sw=1.4, dash=None, s=None, cls=None, **kw):
    if s is not None:
        kw["data-s"] = s
        cls = ((cls + " ") if cls else "") + "stp"
    if cls:
        kw["class"] = cls
    return el("line", None, x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2),
              stroke=stroke, **{"stroke-width": sw, "stroke-linecap": "round"},
              **({"stroke-dasharray": dash} if dash else {}), **kw)


def path(d, stroke=C["line"], sw=1.4, fill="none", dash=None, s=None, cls=None, **kw):
    if s is not None:
        kw["data-s"] = s
        cls = ((cls + " ") if cls else "") + "stp"
    if cls:
        kw["class"] = cls
    return el("path", None, d=d, stroke=stroke, fill=fill,
              **{"stroke-width": sw, "stroke-linecap": "round", "stroke-linejoin": "round"},
              **({"stroke-dasharray": dash} if dash else {}), **kw)


def circle(cx, cy, r, fill=C["compute"], stroke=None, sw=1, s=None, cls=None, **kw):
    if s is not None:
        kw["data-s"] = s
        cls = ((cls + " ") if cls else "") + "stp"
    if cls:
        kw["class"] = cls
    return el("circle", None, cx=round(cx, 2), cy=round(cy, 2), r=round(r, 2), fill=fill,
              stroke=stroke, **({"stroke-width": sw} if stroke else {}), **kw)


def arrow(x1, y1, x2, y2, color=C["mut2"], sw=1.5, s=None, march=False, head=True, dash=None):
    cls = ("march" if march else None)
    parts = [line(x1, y1, x2, y2, stroke=color, sw=sw, s=s, cls=cls, dash=dash,
                  **({"marker-end": "url(#ah)"} if head else {}))]
    return "".join(parts)


def defs(extra=""):
    return el("defs",
              el("marker", el("path", None, d="M0,0 L7,3 L0,6 z", fill=C["mut2"]),
                 id="ah", viewBox="0 0 8 6", refX="7", refY="3",
                 markerWidth="7", markerHeight="6", orient="auto")
              + el("marker", el("path", None, d="M0,0 L7,3 L0,6 z", fill=C["compute"]),
                   id="ahb", viewBox="0 0 8 6", refX="7", refY="3",
                   markerWidth="7", markerHeight="6", orient="auto")
              + el("marker", el("path", None, d="M0,0 L7,3 L0,6 z", fill=C["ok"]),
                   id="ahg", viewBox="0 0 8 6", refX="7", refY="3",
                   markerWidth="7", markerHeight="6", orient="auto")
              + el("marker", el("path", None, d="M0,0 L7,3 L0,6 z", fill=C["spec"]),
                   id="ahp", viewBox="0 0 8 6", refX="7", refY="3",
                   markerWidth="7", markerHeight="6", orient="auto")
              + extra)


def svg(w, h, body, extra_defs=""):
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg">' + defs(extra_defs)
            + ("".join(body) if isinstance(body, (list, tuple)) else body) + "</svg>")


def node(x, y, w, h, title, sub=None, color=C["compute"], tint=C["computeW"],
         s=None, r=7, tsize=12, ssize=9.4, cls=None):
    """圓角方塊 + 標題（可含副標）。"""
    body = [rect(0, 0, w, h, r=r, fill=tint, stroke=color, sw=1.2)]
    if sub:
        body.append(txt(w / 2, h / 2 - 2.5, title, cls="lbl b", anchor="middle",
                        fill=color, size=tsize))
        body.append(txt(w / 2, h / 2 + 10.5, sub, cls="lbl xs", anchor="middle", size=ssize))
    else:
        body.append(txt(w / 2, h / 2 + 4, title, cls="lbl b", anchor="middle",
                        fill=color, size=tsize))
    return g(body, s=s, cls=cls, transform=f"translate({round(x,2)},{round(y,2)})")


def pill(x, y, w, h, label, color=C["compute"], tint=C["computeW"], s=None, size=10.5):
    return g([rect(0, 0, w, h, r=h / 2, fill=tint, stroke=color, sw=1),
              txt(w / 2, h / 2 + 3.4, label, cls="lbl b", anchor="middle", fill=color, size=size)],
             s=s, transform=f"translate({round(x,2)},{round(y,2)})")


def cell(x, y, w, h, label, fill=C["white"], stroke=C["line"], color=C["ink2"],
         s=None, size=10.5, r=4, weight=None):
    return g([rect(0, 0, w, h, r=r, fill=fill, stroke=stroke, sw=1),
              txt(w / 2, h / 2 + 3.6, label, cls="lbl", anchor="middle", fill=color,
                  size=size, weight=weight)],
             s=s, transform=f"translate({round(x,2)},{round(y,2)})")


def bracket(x, y1, y2, label, color=C["mut2"], side="left", s=None, size=9.6):
    d = 7 if side == "left" else -7
    body = [path(f"M{x+d},{y1} L{x},{y1} L{x},{y2} L{x+d},{y2}", stroke=color, sw=1.1),
            txt(x + (d * 1.5 if side == "left" else d * 1.5), (y1 + y2) / 2 + 3.2, label,
                cls="lbl xs", anchor="end" if side == "left" else "start", fill=color, size=size)]
    return g(body, s=s)


def axes(x0, y0, w, h, xlab=None, ylab=None, color=C["line"]):
    out = [line(x0, y0, x0 + w, y0, stroke=color, sw=1.2),
           line(x0, y0, x0, y0 - h, stroke=color, sw=1.2)]
    if xlab:
        out.append(txt(x0 + w, y0 + 20, xlab, cls="lbl s", anchor="end"))
    if ylab:
        out.append(txt(x0 - 6, y0 - h - 7, ylab, cls="lbl s", anchor="start"))
    return "".join(out)


def flow_dot(pathd, color=C["compute"], r=3.5, dur="1.6s", delay="0s", s=None):
    """沿路徑跑的小圓點（CSS motion path）。"""
    st = (f"offset-path:path('{pathd}');--dur:{dur};animation-delay:{delay};"
          f"offset-rotate:0deg")
    kw = {"class": "zip" + (" stp" if False else ""), "style": st}
    if s is not None:
        kw["data-s"] = s
    return el("circle", None, cx=0, cy=0, r=r, fill=color, **kw)
