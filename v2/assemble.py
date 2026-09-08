# -*- coding: utf-8 -*-
"""把 SLIDES 組裝成單一 HTML，並產生講者筆記 Markdown。"""
from __future__ import annotations
import json, base64, re
import html as htmlmod
from pathlib import Path
import common as K

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def data_uri(p: Path):
    mime = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg",
            "svg": "image/svg+xml", "woff2": "font/woff2"}[p.suffix.lstrip(".").lower()]
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


# 名詞自動標註：把每頁「第一次出現」的專有名詞包成可 hover 的 .t
# 只在純文字區域動作，跳過標籤內部、<svg>、<pre>、<code>、<a> 與已標註的區塊。
ALIASES = [
    ("PagedAttention", "PagedAttention"), ("Continuous batching", "Continuous batching"),
    ("continuous batching", "Continuous batching"), ("Chunked prefill", "Chunked prefill"),
    ("chunked prefill", "Chunked prefill"), ("Prefix caching", "Prefix caching"),
    ("prefix caching", "Prefix caching"), ("KV cache", "KV cache"),
    ("Gated DeltaNet", "Gated DeltaNet (GDN)"), ("GDN", "Gated DeltaNet (GDN)"),
    ("Speculative decoding", "Speculative decoding"),
    ("speculative decoding", "Speculative decoding"),
    ("Rejection sampling", "Rejection sampling"), ("rejection sampling", "Rejection sampling"),
    ("acceptance length", "Acceptance length (τ)"), ("接受長度", "Acceptance length (τ)"),
    ("NVFP4", "NVFP4"), ("DFlash2", "DFlash2"), ("MTP", "MTP"), ("MoE", "MoE"),
    ("EAGLE-3", "EAGLE / EAGLE-3"), ("EAGLE", "EAGLE / EAGLE-3"),
    ("goodput", "Goodput"), ("Goodput", "Goodput"),
    ("Little's Law", "Little's Law"), ("roofline", "Roofline"), ("屋頂線", "Roofline"),
    ("算術強度", "Arithmetic intensity"), ("arithmetic intensity", "Arithmetic intensity"),
    ("TTFT", "TTFT"), ("TPOT", "TPOT"), ("ITL", "ITL"), ("E2EL", "E2EL"),
    ("SLO", "SLO"), ("GQA", "GQA"), ("HBM", "HBM"), ("CUDA graph", "CUDA graph"),
    ("Activated parameters", "Activated parameters"),
    ("activated parameters", "Activated parameters"),
    ("Tensor parallel", "Tensor parallelism (TP)"),
    ("tensor parallel", "Tensor parallelism (TP)"),
    ("Expert parallel", "Expert parallelism (TP)".replace("TP", "EP")),
    ("百分位", "Percentile (p95/p99)"), ("RoPE", "RoPE"),
    ("SSM state", "SSM cache / recurrent state"),
    ("recurrent state", "SSM cache / recurrent state"),
    ("Prefill", "Prefill"), ("Decode", "Decode"),
    ("Full attention", "Full attention"), ("full attention", "Full attention"),
    ("Gated Attention", "Gated Attention"), ("Block diffusion", "Block diffusion"),
    ("block diffusion", "Block diffusion"), ("FP8", "FP8 (E4M3)"), ("BF16", "BF16"),
    ("Autoregressive", "Autoregressive"), ("autoregressive", "Autoregressive"),
    ("vLLM", "vLLM"),
]
SKIP_BLOCKS = ("svg", "pre", "code", "a", "kbd")
MAX_TERMS_PER_SLIDE = 6


def autolink(body: str, glossary, max_terms: int = MAX_TERMS_PER_SLIDE) -> str:
    out, i, n = [], 0, len(body)
    used, count = set(), 0
    skip_depth, skip_tag = 0, None
    in_t = False
    while i < n:
        ch = body[i]
        if ch == "<":
            j = body.find(">", i)
            if j < 0:
                out.append(body[i:])
                break
            tag = body[i:j + 1]
            low = tag.lower()
            name = re.match(r"</?\s*([a-z0-9]+)", low)
            nm = name.group(1) if name else ""
            if nm in SKIP_BLOCKS:
                if low.startswith("</"):
                    if skip_tag == nm and skip_depth:
                        skip_depth -= 1
                        if not skip_depth:
                            skip_tag = None
                elif not low.rstrip("/> ").endswith("/"):
                    if skip_tag in (None, nm):
                        skip_tag, skip_depth = nm, skip_depth + 1
            if 'class="t"' in tag:
                in_t = True
            elif low.startswith("</b") and in_t:
                in_t = False
            out.append(tag)
            i = j + 1
            continue
        j = body.find("<", i)
        if j < 0:
            j = n
        text = body[i:j]
        if skip_depth or in_t or count >= max_terms:
            out.append(text)
        else:
            for disp, key in ALIASES:
                if count >= max_terms or key in used or key not in glossary:
                    continue
                pos = text.find(disp)
                if pos < 0:
                    continue
                # 避免切到更長的英數字詞（例如 MoE 出現在 MoEX 裡）
                before = text[pos - 1] if pos else " "
                after = text[pos + len(disp)] if pos + len(disp) < len(text) else " "
                if (before.isalnum() and before.isascii()) or (after.isalnum() and after.isascii()):
                    continue
                text = (text[:pos] + f'<b class="t" data-term="{key}">{disp}</b>'
                        + text[pos + len(disp):])
                used.add(key)
                count += 1
            out.append(text)
        i = j
    return "".join(out)


def render_slide(m, i, n):
    ac = f"--accent:var(--{m['accent']})"
    ch = m.get("ch", "")
    srcs = "".join(
        f'<a href="{K.SOURCES[s][1]}" target="_blank" rel="noopener">{K.SOURCES[s][0]}</a>'
        for s in m["sources"] if s in K.SOURCES)
    if m["kind"] == "cover":
        return (f'<section class="slide cover" style="{ac}">{m["body"]}</section>')
    if m["kind"] == "section":
        return (f'<section class="slide section" style="{ac}"><div class="wrap">{m["body"]}'
                f'</div></section>')
    lt = f' <span class="lt">{m["lt"]}</span>' if m.get("lt") else ""
    chip = f'<span class="chip">{m["chip"]}</span>' if m.get("chip") else ""
    return (
        f'<section class="slide" style="{ac}">'
        f'<div class="rail"><span class="num">{i+1:02d}</span>'
        f'<span class="ch">{ch}</span><span class="dot"></span></div>'
        f'<div class="head"><div class="txt"><h1>{m["title"]}{lt}</h1>'
        f'{f"<div class=sub>{m['sub']}</div>" if m['sub'] else ""}</div>{chip}</div>'
        f'<div class="body">{m["body"]}</div>'
        f'<div class="foot"><span class="srcs">{srcs}</span>'
        f'<span class="pg">{i+1} / {n}</span></div>'
        f'</section>')


def build(out_html: Path, out_notes: Path, title: str):
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    widgets = K.model_js() + (HERE / "widgets.js").read_text(encoding="utf-8")

    n = len(K.SLIDES)
    for c_i, c in enumerate(K.CHAPTERS):
        c["end"] = K.CHAPTERS[c_i + 1]["at"] if c_i + 1 < len(K.CHAPTERS) else n
    for m in K.SLIDES:
        if m["kind"] not in ("cover",):
            m["body"] = autolink(m["body"], K.GLOSSARY)
    html = "".join(render_slide(m, i, n) for i, m in enumerate(K.SLIDES))

    meta = [{"title": re.sub("<[^>]+>", "", m["title"]), "sub": re.sub("<[^>]+>", "", m["sub"]),
             "notes": m["notes"], "sources": m["sources"], "ch": m.get("ch", "")}
            for m in K.SLIDES]

    out = (tpl.replace("@@TITLE@@", title)
              .replace("@@SLIDES@@", html)
              .replace("@@META@@", json.dumps(meta, ensure_ascii=False))
              .replace("@@SOURCES@@", json.dumps(K.SOURCES, ensure_ascii=False))
              .replace("@@GLOSSARY@@", json.dumps(K.GLOSSARY, ensure_ascii=False))
              .replace("@@CHAPTERS@@", json.dumps(K.CHAPTERS, ensure_ascii=False))
              .replace("@@WIDGETS@@", widgets))
    assert "@@" not in out, "未替換的樣板變數：" + str(re.findall(r"@@\w+@@", out))
    out_html.write_text(out, encoding="utf-8")

    md = [f"# {title} — 講者筆記", "",
          f"共 {n} 頁。資料查核日期 {K.CHECK}。", ""]
    for c in K.CHAPTERS:
        md.append(f"## {c['name']}")
        md.append("")
        for i in range(c["at"], c["end"]):
            m = K.SLIDES[i]
            md.append(f"### {i+1}. {htmlmod.unescape(re.sub('<[^>]+>','',m['title']))}")
            if m["sub"]:
                md.append(f"*{htmlmod.unescape(re.sub('<[^>]+>','',m['sub']))}*")
            md.append("")
            _n = re.sub(r"</p>|<br\s*/?>", "\n", m["notes"] or "（無）")
            md.append(htmlmod.unescape(re.sub("<[^>]+>", "", _n)).strip())
            if m["sources"]:
                md.append("")
                md.append("來源：" + "、".join(
                    f"[{K.SOURCES[s][0]}]({K.SOURCES[s][1]})" for s in m["sources"]
                    if s in K.SOURCES))
            md.append("")
    out_notes.write_text("\n".join(md), encoding="utf-8")
    return len(out)
