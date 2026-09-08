# -*- coding: utf-8 -*-
"""CH1–CH2 的圖：推論迴圈、KV cache、roofline、vLLM 堆疊。"""
from svgkit import *
import model as M

W = 1000  # 圖的邏輯寬度


# ---------------------------------------------------------------- 1. token 化
def fig_tokenize():
    toks = [("量", 1), ("子", 1), ("電", 1), ("腦", 1), ("的", 1), ("錯", 1), ("誤", 1),
            ("修", 1), ("正", 1), ("是", 1), ("什", 1), ("麼", 1), ("？", 1)]
    b = [txt(0, 14, "使用者輸入", cls="lbl s")]
    b.append(g([rect(0, 24, 470, 40, fill=C["card2"]),
                txt(16, 50, "量子電腦的錯誤修正是什麼？", cls="lbl b", size=15)]))
    y = 100
    b.append(txt(0, y - 10, "① tokenizer → token IDs（每個 id 就是 vocabulary 裡的一格）", cls="lbl s", ds="1"))
    x = 0
    ids = [104512, 3927, 41027, 8104, 279, 66214, 25181, 6612, 1114, 30]
    for i, t in enumerate(["量子", "電腦", "的", "錯誤", "修正", "是", "什麼", "？"]):
        w = 13 + len(t) * 15
        b.append(cell(x, y, w, 30, t, fill=C["computeW"], stroke=C["compute"],
                      color=C["compute"], s=1, size=12.5, weight=650))
        b.append(txt(x + w / 2, y + 45, str(ids[i % len(ids)]), cls="num", anchor="middle", ds="1"))
        x += w + 6
    b.append(txt(x + 8, y + 20, f"共 8 個 token", cls="lbl s", ds="1"))
    y2 = 190
    b.append(txt(0, y2 - 10, "② chat template 把它包成模型看得懂的對話格式", cls="lbl s", ds="2"))
    b.append(g([rect(0, y2, 700, 52, fill="#fbfcfd"),
                txt(14, y2 + 21, "&lt;|im_start|&gt;user\\n量子電腦的錯誤修正是什麼？&lt;|im_end|&gt;",
                    cls="num", size=11.5, fill=C["ink2"]),
                txt(14, y2 + 39, "&lt;|im_start|&gt;assistant\\n&lt;think&gt;", cls="num", size=11.5,
                    fill=C["moe"])], s=2))
    b.append(txt(720, y2 + 30, "→ 實際送進模型的長度\n通常比原文多 10–20 個 token",
                 cls="lbl s", ds="2"))
    y3 = 272
    b.append(txt(0, y3 + 4, "③ 這串 id 的長度 T，決定了 prefill 的計算量與 KV cache 的大小",
                 cls="lbl b", ds="3", fill=C["mem"]))
    return svg(W, 300, b)


# ------------------------------------------------------- 2. autoregressive
def fig_autoregress():
    b = []
    by, bh = 26, 58
    b.append(node(250, by, 500, bh, "Decoder layers × L", "每一層：token mixer → FFN",
                  color=C["compute"], tint=C["computeW"]))
    b.append(node(30, by + 7, 150, 44, "Embedding", "id → 向量",
                  color=C["neutral"], tint="#f4f6f8"))
    b.append(node(820, by + 7, 150, 44, "LM head", "向量 → logits",
                  color=C["neutral"], tint="#f4f6f8"))
    b.append(arrow(182, by + 29, 246, by + 29))
    b.append(arrow(752, by + 29, 816, by + 29))
    b.append(node(820, 104, 150, 38, "Sampling", None, color=C["moe"], tint=C["moeW"], s=1))
    b.append(arrow(895, by + 51, 895, 103, s=1))
    b.append(path("M816,123 L112,123 L112,79", stroke=C["spec"], sw=1.6, dash="6 5", s=2,
                  **{"marker-end": "url(#ahp)"}))
    b.append(txt(464, 141, "新產生的 token 接回輸入尾端，一輪一輪這樣跑下去",
                 cls="lbl b", anchor="middle", fill=C["spec"], ds="2", size=12))
    y = 186
    b.append(txt(0, y - 8, "序列狀態", cls="lbl s"))
    x = 0
    for t in ["量子", "電腦", "的", "錯誤", "修正", "是", "什麼", "？"]:
        w = 13 + len(t) * 14
        b.append(cell(x, y, w, 28, t, fill=C["card2"], size=11.5))
        x += w + 4
    for t, st in (("錯誤", 3), ("修正", 4), ("是", 5)):
        w = 13 + len(t) * 14
        b.append(cell(x, y, w, 28, t, fill=C["okW"], stroke=C["ok"], color=C["ok"],
                      s=st, size=11.5, weight=650))
        x += w + 4
    b.append(txt(x + 10, y + 19, "每一輪只加一格", cls="lbl s", ds="3"))
    b.append(txt(x + 106, y + 19, "→ T 次 forward 才生 T 個 token", cls="lbl b", ds="3",
                 fill=C["mem"], size=11.5))
    b.append(g([rect(0, 230, 300, 34, fill=C["memW"], stroke="#f0dcb4"),
                txt(13, 251, "本輪送進模型的 token 數：1", cls="lbl b", size=11.5,
                    fill=C["mem"])], s=5))
    b.append(g([rect(316, 230, 420, 34, fill=C["computeW"], stroke="#d5e4fd"),
                txt(329, 251, "本輪要讀的權重：整個模型（不管只算 1 個 token）",
                    cls="lbl b", size=11.5, fill=C["compute"])], s=5))
    b.append(g([rect(752, 230, 248, 34, fill=C["badW"], stroke="#f5cdc8"),
                txt(765, 251, "→ 算力幾乎全部閒置", cls="lbl b", size=11.5,
                    fill=C["bad"])], s=5))
    return svg(W, 274, b)


# ------------------------------------------------------- 3. decoder layer
def fig_layer():
    b = []
    x0, y0, lw, lh, gp = 66, 20, 176, 33, 7
    rows = [("RMSNorm", "穩定數值", C["neutral"], "#f4f6f8", 0),
            ("QKV projection", "hidden → Q, K, V", C["compute"], C["computeW"], 1),
            ("QK-norm + RoPE", "位置寫進 Q, K", C["compute"], C["computeW"], 2),
            ("Attention", "查歷史 K/V", C["compute"], C["computeW"], 3),
            ("Output proj + gate", "回到 hidden 寬度", C["compute"], C["computeW"], 4),
            ("RMSNorm", None, C["neutral"], "#f4f6f8", 5),
            ("FFN / MoE", "每個 token 各自算", C["moe"], C["moeW"], 5)]
    y = y0
    for i, (t, s2, col, tint, st) in enumerate(rows):
        if i == 5:
            y += 8
        b.append(node(x0, y, lw, lh, t, s2, color=col, tint=tint,
                      s=(st or None), tsize=11, ssize=8.8))
        if i:
            b.append(arrow(x0 + lw / 2, y - (14 if i == 5 else 7), x0 + lw / 2, y - 1,
                           s=(st or None)))
        rows[i] = (t, s2, col, tint, st, y)
        y += lh + gp
    b.append(bracket(x0 - 10, y0, rows[4][5] + lh, "token mixer", color=C["compute"]))
    b.append(bracket(x0 - 10, rows[5][5], rows[6][5] + lh, "channel mixer",
                     color=C["moe"], s=5))
    b.append(path(f"M{x0+lw+8},{y0+16} L{x0+lw+26},{y0+16} "
                  f"L{x0+lw+26},{rows[4][5]+lh-6} L{x0+lw+8},{rows[4][5]+lh-6}",
                  stroke=C["ok"], sw=1.2, dash="4 4", s=4))
    b.append(txt(x0 + lw + 30, (y0 + rows[4][5]) / 2 + 20, "residual", cls="lbl xs",
                 fill=C["ok"], ds="4"))
    tx = 356
    b.append(txt(tx, 14, "資料形狀（Qwen3.5-122B-A10B，decode，batch B，每序列 1 個 token）",
                 cls="lbl s"))
    shapes = [("hidden", "[B, 3072]", C["neutral"], 0),
              ("Q", "[B, 32 heads, 256]　總寬 8192", C["compute"], 1),
              ("K, V", "[B, 2 heads, 256]　← 只有 2 組", C["mem"], 1),
              ("RoPE 旋轉範圍", "只有 head_dim 的 1/4 = 64 維", C["compute"], 2),
              ("讀取 KV cache", "[B, 2, T, 256] × 2　T = 目前長度", C["mem"], 3),
              ("attention 輸出", "[B, 32, 256] → o_proj → [B, 3072]", C["compute"], 4),
              ("MoE", "256 個 expert 只算 top-8 + 1 shared", C["moe"], 5)]
    yy = 24
    for k, v, col, st in shapes:
        b.append(g([rect(tx, yy, 566, 29, fill="#fff", stroke=C["line2"]),
                    rect(tx, yy, 3, 29, r=0, fill=col, stroke=None),
                    txt(tx + 13, yy + 19, k, cls="lbl b", size=11, fill=col),
                    txt(tx + 152, yy + 19, v, cls="num", size=10.5)],
                   s=(st or None)))
        yy += 33
    b.append(g([rect(tx, yy + 4, 566, 44, fill=C["memW"], stroke="#f0dcb4"),
                txt(tx + 13, yy + 22, "KV cache 的大小只跟 K/V head 數與 head_dim 有關，跟 Q 的 32 組無關。",
                    cls="lbl b", size=11, fill=C["mem"]),
                txt(tx + 13, yy + 38, "所以「Q 很寬」不花記憶體，只花算力。GQA 的好處就在這裡。",
                    cls="lbl", size=10.5, fill=C["mem"])], s=5))
    return svg(W, max(y, yy + 54), b)


# --------------------------------------------------------- 4. KV cache 機制
def fig_kv():
    b = []
    cw, ch, gap = 52, 30, 6
    T = 8
    x0, y0 = 120, 26
    b.append(txt(0, y0 + 20, "Q（本輪）", cls="lbl s"))
    for i in range(T - 1):
        b.append(cell(x0 + i * (cw + gap), y0, cw, ch, "—", fill="#fff",
                      stroke=C["line2"], color=C["mut2"]))
    b.append(cell(x0 + (T - 1) * (cw + gap), y0, cw, ch, "q₈", fill=C["computeW"],
                  stroke=C["compute"], color=C["compute"], weight=650))
    for row, name in ((1, "K cache"), (2, "V cache")):
        y = y0 + row * (ch + 14) + 6
        b.append(txt(0, y + 20, name, cls="lbl s", ds="1"))
        for i in range(T):
            new = i == T - 1
            b.append(cell(x0 + i * (cw + gap), y, cw, ch,
                          f"{'k' if row==1 else 'v'}{i+1}",
                          fill=C["memW"] if not new else C["okW"],
                          stroke=C["mem"] if not new else C["ok"],
                          color=C["mem"] if not new else C["ok"],
                          s=(1 if not new else 2), weight=650 if new else None))
    ybot = y0 + 3 * (ch + 14)
    for i in range(T):
        cx = x0 + i * (cw + gap) + cw / 2
        b.append(path(f"M{x0+(T-1)*(cw+gap)+cw/2},{y0+ch+2} L{cx},{y0+ch+18}",
                      stroke=C["mem"], sw=1, dash="3 3", s=3))
    px = x0 + 8 * (cw + gap) + 14
    b.append(g([rect(px, y0, 214, 3 * (ch + 14) - 14, fill=C["computeW"], stroke="#d5e4fd"),
                txt(px + 14, y0 + 24, "oₜ = softmax(qₜKᵀ/√d)·V", cls="num", size=11.5,
                    fill=C["compute"]),
                txt(px + 14, y0 + 48, "算 1 個 token，卻要讀：", cls="lbl s"),
                txt(px + 14, y0 + 68, "· T 個位置的 K/V", cls="lbl b", fill=C["mem"], size=11.5),
                txt(px + 14, y0 + 86, "· 整個模型的權重", cls="lbl b", fill=C["mem"], size=11.5),
                txt(px + 14, y0 + 108, "→ 於是卡在 memory-bound", cls="lbl b",
                    fill=C["compute"], size=11)], s=3))
    b.append(g([rect(0, ybot + 4, 640, 34, fill=C["badW"], stroke="#f5cdc8"),
                txt(14, ybot + 25, "沒有 cache：每一輪都要把前面 T−1 個位置重算，總成本 O(T²)",
                    cls="lbl b", size=11.5, fill=C["bad"])], s=4))
    return svg(W, ybot + 46, b)


# ------------------------------------------------------------------ 5. GQA
def fig_gqa(m_key="q35"):
    m = M.MODELS[m_key]
    b = []
    b.append(txt(0, 14, f"{m.label}　full-attention 層：{m.n_heads} 個 Q head 共用 "
                        f"{m.n_kv} 組 K/V head（head_dim {m.head_dim}）", cls="lbl b", size=12))
    nq, nkv = m.n_heads, m.n_kv
    x0, y0 = 20, 40
    qw = 20
    gapq = 3.6
    per = nq // nkv
    for j in range(nkv):
        gx = x0 + j * (per * (qw + gapq) + 34)
        for i in range(per):
            b.append(rect(gx + i * (qw + gapq), y0, qw, 26, r=3, fill=C["computeW"],
                          stroke=C["compute"], sw=.9))
        # kv
        kvx = gx + (per * (qw + gapq) - gapq) / 2 - 26
        b.append(rect(kvx, y0 + 58, 26, 24, r=3, fill=C["memW"], stroke=C["mem"], sw=1.1, s=1))
        b.append(rect(kvx + 30, y0 + 58, 26, 24, r=3, fill=C["memW"], stroke=C["mem"], sw=1.1, s=1))
        b.append(txt(kvx + 13, y0 + 74, "K", cls="lbl b", anchor="middle", size=10,
                     fill=C["mem"], ds="1"))
        b.append(txt(kvx + 43, y0 + 74, "V", cls="lbl b", anchor="middle", size=10,
                     fill=C["mem"], ds="1"))
        for i in range(per):
            b.append(path(f"M{gx+i*(qw+gapq)+qw/2},{y0+27} L{kvx+28},{y0+56}",
                          stroke=C["mut2"], sw=.7, s=1))
    b.append(txt(x0, y0 + 20, "", cls="lbl"))
    b.append(txt(0, y0 + 108, f"→ 每個 token 只需要存 2 × {nkv} × {m.head_dim} = "
                              f"{2*nkv*m.head_dim:,} 個數字（K 與 V）", cls="lbl b",
                 ds="2", fill=C["mem"], size=12))
    return svg(W, 160, b)


# ---------------------------------------------------- 6. prefill vs decode
def fig_prefill_decode():
    b = []
    # left: prefill
    b.append(txt(20, 16, "PREFILL　一次吃完整個 prompt", cls="lbl b", size=12, fill=C["compute"]))
    n = 10
    cw = 26
    for i in range(n):
        b.append(rect(20 + i * (cw + 4), 28, cw, 26, r=4, fill=C["computeW"],
                      stroke=C["compute"], sw=1))
    b.append(txt(20, 76, "矩陣運算：[T, 3072] × 權重 → 大 GEMM", cls="lbl s", ds="1"))
    b.append(txt(20, 94, "T 個 token 一起攤提同一份權重讀取", cls="lbl b", size=11.5,
                 fill=C["ok"], ds="1"))
    # arithmetic intensity bar
    b.append(g([txt(20, 124, "算術強度（每讀 1 byte 做幾次 FLOP）", cls="lbl s"),
                rect(20, 132, 400, 16, r=4, fill=C["line2"], stroke=None),
                rect(20, 132, 340, 16, r=4, fill=C["compute"], stroke=None),
                txt(430, 145, "高 → 吃算力", cls="lbl b", size=11, fill=C["compute"])], s=2))
    # divider
    b.append(line(500, 8, 500, 200, stroke=C["line2"], sw=1))
    b.append(txt(530, 16, "DECODE　每輪每條序列只 1 個 token", cls="lbl b", size=12, fill=C["mem"]))
    for i in range(4):
        b.append(rect(530 + i * 70, 28, cw, 26, r=4, fill=C["memW"], stroke=C["mem"], sw=1))
        b.append(txt(530 + i * 70 + 13, 66, f"req {i+1}", cls="lbl xs", anchor="middle"))
    b.append(txt(530, 94, "矩陣運算退化成 GEMV（瘦長矩陣）", cls="lbl s", ds="1"))
    b.append(txt(530, 112, "同一份權重只服務 B 個 token", cls="lbl b", size=11.5,
                 fill=C["bad"], ds="1"))
    b.append(g([txt(530, 142, "算術強度", cls="lbl s"),
                rect(530, 150, 400, 16, r=4, fill=C["line2"], stroke=None),
                rect(530, 150, 46, 16, r=4, fill=C["mem"], stroke=None),
                txt(586, 163, "低 → 吃頻寬（memory-bound）", cls="lbl b", size=11,
                    fill=C["mem"])], s=2))
    b.append(g([rect(20, 176, 910, 40, fill=C["okW"], stroke="#cbe9d5"),
                txt(34, 193, "所有 serving 最佳化都在回答同一件事：", cls="lbl b", size=11.5,
                    fill="#14532d"),
                txt(34, 209, "怎麼讓每次讀取權重的成本，被更多 token 分攤。"
                             "continuous batching、chunked prefill、speculative decoding 都在做這件事。",
                    cls="lbl", size=11.5, fill="#14532d")], s=3))
    return svg(W, 226, b)


# ---------------------------------------------------------- 7. roofline
def fig_roofline():
    import math
    b = []
    x0, y0, w, h = 72, 300, 640, 246
    knee = M.GPU.knee_tokens("fp8")

    def X(n):
        return x0 + (math.log10(max(n, 1)) / math.log10(4096)) * w

    def Y(t):
        lo, hi = 1.0 / 4096, 1.0
        return y0 - (math.log10(max(t, lo) / lo) / math.log10(hi / lo)) * h

    b.append(g([rect(x0 + 1, y0 - h, X(knee) - x0 - 1, h, r=0, fill="#f2f7ff", stroke=None),
                txt((x0 + X(knee)) / 2, y0 - h + 26, "memory-bound", cls="lbl b",
                    anchor="middle", fill=C["compute"], size=15),
                txt((x0 + X(knee)) / 2, y0 - h + 46,
                    "step 時間 = 讀權重的時間，與 N 無關", cls="lbl s", anchor="middle"),
                txt((x0 + X(knee)) / 2, y0 - h + 62,
                    "→ 多塞 token 幾乎免費", cls="lbl s", anchor="middle")], s=3))
    b.append(g([rect(X(knee), y0 - h, x0 + w - X(knee), h, r=0, fill="#fff8f0", stroke=None),
                txt((X(knee) + x0 + w) / 2, y0 - h + 26, "compute-bound", cls="lbl b",
                    anchor="middle", fill=C["mem"], size=15),
                txt((X(knee) + x0 + w) / 2, y0 - h + 46, "step 時間 ∝ N",
                    cls="lbl s", anchor="middle")], s=3))
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 14, "↑ 每個 token 的成本（對數刻度）", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 36, "一次 forward 送進去的 token 數 N（對數刻度）→",
                 cls="lbl s", anchor="end"))
    b.append(path(f"M{X(knee):.1f},{Y(1/knee):.1f} L{X(4096):.1f},{Y(1/4096):.1f}",
                  stroke=C["mut2"], sw=1.2, dash="5 5", s=4))
    b.append(txt(X(2600), Y(1 / 2600) + 15, "如果一直是 memory-bound（辦不到）",
                 cls="lbl xs", anchor="end", ds="4"))
    b.append(path(f"M{X(1):.1f},{Y(1):.1f} L{X(knee):.1f},{Y(1/knee):.1f}",
                  stroke=C["compute"], sw=3, s=1))
    b.append(path(f"M{X(knee):.1f},{Y(1/knee):.1f} L{X(4096):.1f},{Y(1/knee):.1f}",
                  stroke=C["mem"], sw=3, s=2))
    b.append(line(X(knee), y0, X(knee), Y(1 / knee), stroke=C["mem"], sw=1.3, dash="4 4", s=2))
    b.append(circle(X(knee), Y(1 / knee), 5.5, fill=C["mem"], stroke="#fff", sw=2, s=2))
    b.append(txt(X(knee) + 13, Y(1 / knee) - 26, f"轉折點 N* ≈ {knee:.0f} token / step",
                 cls="lbl b", fill=C["mem"], size=12.5, ds="2"))
    b.append(txt(X(knee) + 13, Y(1 / knee) - 10, "再往右，每個 token 的成本不再下降",
                 cls="lbl s", ds="2"))
    for n in (1, 4, 16, 64, 292, 1024, 4096):
        b.append(line(X(n), y0, X(n), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(n), y0 + 18, f"{n:,}", cls="num", anchor="middle"))
    marks = [(1, 1.0, "單一使用者：batch 1", C["neutral"], 5, -1),
             (32, 1 / 32, "32 併發，不用 spec", C["compute"], 5, 1),
             (256, 1 / knee, "32 併發 × DFlash2 k=7 → 256 token/step", C["spec"], 5, -1)]
    for n, t, lab, col, st, side in marks:
        b.append(circle(X(n), Y(t), 4.5, fill=col, stroke="#fff", sw=1.6, s=st))
        b.append(txt(X(n) + 9, Y(t) + (-8 if side > 0 else 20), lab,
                     cls="lbl xs", anchor="start", fill=col, ds=st, size=9.8))
    # 右欄：公式與精度不變性
    bx = 744
    b.append(g([rect(bx, 26, 256, 250, fill=C["card2"]),
                txt(bx + 15, 50, "N* = F · b / (2 · BW)", cls="num", size=13,
                    fill=C["compute"]),
                txt(bx + 15, 72, "F　峰值算力（FLOP/s）", cls="lbl xs"),
                txt(bx + 15, 87, "b　每個參數的位元組數", cls="lbl xs"),
                txt(bx + 15, 102, "BW　HBM 頻寬（B/s）", cls="lbl xs"),
                line(bx + 15, 114, bx + 241, 114, stroke=C["line"], sw=1),
                txt(bx + 15, 133, "Blackwell 每把位寬砍半，", cls="lbl", size=10.5),
                txt(bx + 15, 148, "張量核心吞吐就剛好加倍：", cls="lbl", size=10.5),
                txt(bx + 15, 163, "F 加倍、b 減半 → 相消。", cls="lbl b", size=10.5,
                    fill=C["mem"]),
                txt(bx + 15, 190, "精度", cls="lbl xs"),
                txt(bx + 96, 190, "b", cls="lbl xs", anchor="end"),
                txt(bx + 168, 190, "F (PF)", cls="lbl xs", anchor="end"),
                txt(bx + 241, 190, "N*", cls="lbl xs", anchor="end"),
                line(bx + 15, 196, bx + 241, 196, stroke=C["line2"], sw=1)], s=2))
    for i, (pn, bp, fl) in enumerate([("BF16", "2", "2.25"), ("FP8", "1", "4.5"),
                                      ("NVFP4", "0.5", "9.0")]):
        yy = 214 + i * 21
        b.append(g([txt(bx + 15, yy, pn, cls="num", size=10.5),
                    txt(bx + 96, yy, bp, cls="num", size=10.5, anchor="end"),
                    txt(bx + 168, yy, fl, cls="num", size=10.5, anchor="end"),
                    txt(bx + 241, yy, f"{knee:.0f}", cls="num", size=11, anchor="end",
                        fill=C["mem"])], s=2))
    return svg(W, y0 + 46, b)
