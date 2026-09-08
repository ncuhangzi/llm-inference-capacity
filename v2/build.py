# -*- coding: utf-8 -*-
"""產生《LLM 推論與服務容量》第二版報告。

  python build.py
"""
from __future__ import annotations
import sys, base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model as M
from model import Q35, Q38, Q332, GPU, GiB, MiB, KiB, SPEC_TAU, SPEC_ALPHA
import common as K
from common import (chapter, slide, fig, gist, take, warn, quote, pane, stats,
                    table, bars, legend, code, num, gib, gb, mib, kib, T, term)
from svgkit import *
import figs_core as F1
import figs_arch as F2
import figs_spec as F3
import figs_load as F4
import assemble

ROOT = HERE.parent
IMG = {p.stem: assemble.data_uri(p) for p in (ROOT / "assets").glob("*.*")}
KNEE = GPU.knee_tokens("fp8")


def img(name, cap=None, style="max-height:100%;"):
    s = f'<img src="{IMG[name]}" style="width:100%;{style}border-radius:8px;border:1px solid var(--line2)">'
    return s + (f'<div class="hint" style="margin-top:6px">{cap}</div>' if cap else "")


def widget(name):
    return f'<div data-widget="{name}"></div>'


exec((HERE / "slides_part1.py").read_text(encoding="utf-8"))
exec((HERE / "slides_part2.py").read_text(encoding="utf-8"))
exec((HERE / "slides_part3.py").read_text(encoding="utf-8"))
exec((HERE / "slides_part4.py").read_text(encoding="utf-8"))

if __name__ == "__main__":
    n = assemble.build(ROOT / "LLM推論報告_v2.html", ROOT / "講者筆記_v2.md",
                       "LLM 推論與服務容量 · 第二版")
    print(f"slides={len(K.SLIDES)} chapters={len(K.CHAPTERS)} bytes={n:,}")
