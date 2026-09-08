# -*- coding: utf-8 -*-
"""產生技術部落格版：LLM 推論與服務容量。

  python blog.py   →  ../LLM推論技術部落格.html
"""
from __future__ import annotations
import sys, re, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model as M
from model import Q35, Q38, Q332, GPU, GiB, MiB, KiB, SPEC_TAU
import common as K
import assemble
import figs_core as F1
import figs_arch as F2
import figs_spec as F3
import figs_load as F4

ROOT = HERE.parent
KNEE = GPU.knee_tokens("fp8")

# ------------------------------------------------------------------ 工具 ---
TOC: list[tuple[str, str, bool]] = []      # (id, 標題, 是否為次層)
REFS: list[str] = []                       # 依首次引用順序排列的 SOURCES key
BUF: list[str] = []


def w(s: str):
    BUF.append(s)


def cite(*keys):
    """行內引用，回傳上標的 [n]。"""
    out = []
    for k in keys:
        if k not in REFS:
            REFS.append(k)
        n = REFS.index(k) + 1
        name = K.SOURCES[k][0]
        out.append(f'<a href="#ref{n}" title="{name}">[{n}]</a>')
    return '<sup class="c">' + "".join(out) + "</sup>"


def h2(no, title, anchor):
    TOC.append((anchor, title, False))
    w(f'<h2 id="{anchor}"><span class="no">{no}</span>{title}</h2>')


def h3(title, anchor):
    TOC.append((anchor, title, True))
    w(f'<h3 id="{anchor}">{title}</h3>')


def p(*parts):
    w("<p>" + "".join(parts) + "</p>")


def ul(items, ordered=False):
    tag = "ol" if ordered else "ul"
    w(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")


def note(body, cls="", title=None):
    h = f"<h4>{title}</h4>" if title else ""
    w(f'<div class="note {cls}">{h}{body}</div>')


def quote(text, src):
    w(f"<blockquote><p>{text}</p><cite>{src}</cite></blockquote>")


def table(head, rows, cls="", hi=()):
    th = "".join(f'<th class="{"n" if x.startswith("~") else ""}">{x.lstrip("~")}</th>'
                 for x in head)
    body = []
    for i, r in enumerate(rows):
        tds = "".join(f'<td class="{"n" if h.startswith("~") else ""}">{c}</td>'
                      for h, c in zip(head, r))
        body.append(f'<tr class="{"hi" if i in hi else ""}">{tds}</tr>')
    w(f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
      f'<tbody>{"".join(body)}</tbody></table></div>')


def formula(body, sub=None):
    w(f'<div class="formula">{body}{f"<small>{sub}</small>" if sub else ""}</div>')


def code(text, lang=""):
    w(f"<pre><code>{text}</code></pre>")


def figure(name, svg, steps, caps, caption=None, wide=True, accent=None):
    cap = "".join(f'<p data-s="{i}">{c}</p>' for i, c in enumerate(caps))
    st = f' style="--accent:var(--{accent})"' if accent else ""
    cls = "wide" if wide else ""
    w(f'<figure class="{cls}"><div class="fig" data-fig="{name}" data-steps="{steps}"{st}>'
      f'{svg}<div class="figbar"></div><div class="figcap">{cap}</div></div>'
      + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>")


def widget(name, caption=None, wide=True):
    cls = "wide" if wide else ""
    w(f'<figure class="{cls}"><div data-widget="{name}"></div>'
      + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>")


T = K.term


def gb(b, d=2):
    return f"{b / 10**9:,.{d}f}"


def gib(b, d=2):
    return f"{b / GiB:,.{d}f}"


def mib(b, d=1):
    return f"{b / MiB:,.{d}f}"


exec((HERE / "blog_body.py").read_text(encoding="utf-8"))


# ------------------------------------------------------------- 組裝輸出 ---
def build():
    tpl = (HERE / "blog_template.html").read_text(encoding="utf-8")
    widgets = K.model_js() + (HERE / "widgets.js").read_text(encoding="utf-8")

    body = "".join(BUF)
    body = assemble.autolink(body, K.GLOSSARY, max_terms=60)

    toc = "".join(f'<a href="#{a}" class="{"sub" if sub else ""}">{t}</a>'
                  for a, t, sub in TOC)

    refs = ['<ol class="refs">']
    for i, k in enumerate(REFS, 1):
        name, url = K.SOURCES[k]
        if url == "#":
            refs.append(f'<li id="ref{i}">{name}</li>')
        else:
            refs.append(f'<li id="ref{i}"><a href="{url}" target="_blank" '
                        f'rel="noopener">{name}</a><span class="u">{url}</span></li>')
    refs.append("</ol>")
    body = body.replace("@@REFS@@", "".join(refs))

    n_fig = body.count("data-fig=")
    n_widget = len(set(re.findall(r'data-widget="([a-z0-9]+)"', body)))
    n_char = len(re.sub("<[^>]+>", "", body))
    meta = (f'<span><b>查核日期</b> {K.CHECK}</span>'
            f'<span><b>字數</b> 約 {n_char // 1000} 千字</span>'
            f'<span><b>互動圖表</b> {n_fig} 張動畫 + {n_widget} 個試算器</span>'
            f'<span><b>參考來源</b> {len(REFS)} 筆</span>')

    out = (tpl.replace("@@TITLE@@", "LLM 推論與服務容量：從一次 forward 到「能服務幾個人」")
              .replace("@@DESC@@", "以 Qwen3.5-122B-A10B（NVFP4）與 Qwen3.8-27B（FP8）"
                                   "在 vLLM / B200 上的實例，拆解 LLM 推論的成本結構、"
                                   "混合架構的記憶體帳本，以及 speculative decoding 什麼時候會賠錢。")
              .replace("@@SHORT@@", "LLM 推論與服務容量")
              .replace("@@KICKER@@", "Inference · Memory · Capacity")
              .replace("@@H1@@", "從一次 forward 到「能服務幾個人」")
              .replace("@@LEDE@@", "把 LLM 推論的成本結構拆開來看：為什麼 decode 卡在頻寬、"
                                   "混合架構把 KV cache 換成什麼、NVFP4 到底量化了哪些張量，"
                                   "以及 speculative decoding 在什麼併發數之後會開始賠錢。"
                                   "每個數字都附推導與來源，圖都可以自己按。")
              .replace("@@META@@", meta)
              .replace("@@DECKHREF@@", "LLM推論報告_v2.html")
              .replace("@@TOC@@", toc)
              .replace("@@BODY@@", body)
              .replace("@@GLOSSARY@@", json.dumps(K.GLOSSARY, ensure_ascii=False))
              .replace("@@WIDGETS@@", widgets))
    assert "@@" not in out, re.findall(r"@@\w+@@", out)
    path = ROOT / "LLM推論技術部落格.html"
    path.write_text(out, encoding="utf-8")
    return path, len(out)


if __name__ == "__main__":
    pth, n = build()
    print(f"{pth.name}  sections={len(TOC)}  refs={len(REFS)}  bytes={n:,}")
