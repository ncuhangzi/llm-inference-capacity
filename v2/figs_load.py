# -*- coding: utf-8 -*-
"""CH8–CH10 的圖：延遲時間軸、負載測試、決策樹。"""
from svgkit import *
import model as M

W = 1000


# ------------------------------------------------------- 一個 request 的時間軸
def fig_latency():
    b = []
    y = 40
    x0 = 128
    total = 830
    segs = [("網路 + 佇列", 90, C["neutral"], "#f4f6f8", 1),
            ("prefill", 150, C["compute"], C["computeW"], 2),
            ("decode ×N", 590, C["mem"], C["memW"], 3)]
    x = x0
    for lab, wd, col, tint, st in segs:
        b.append(g([rect(x, y, wd, 40, r=5, fill=tint, stroke=col, sw=1.2),
                    txt(x + wd / 2, y + 25, lab, cls="lbl b", anchor="middle", size=11.5,
                        fill=col)], s=st))
        x += wd + 3
    b.append(txt(0, y + 25, "伺服器端", cls="lbl s"))
    # token 刻度
    ty = y + 48
    n = 14
    dx = 590 / n
    sx = x0 + 90 + 150 + 6
    for i in range(n):
        b.append(rect(sx + i * dx, ty, dx - 3, 12, r=2, fill=C["mem"], stroke=None,
                      opacity=.45, s=3))
    # 標註
    ay = ty + 34
    marks = [("TTFT", x0, sx + dx, C["compute"], "送出 → 第一個 token（含排隊與 prefill）", 4),
             ("ITL", sx + 4 * dx, sx + 5 * dx, C["mem"], "相鄰兩個 token 之間", 5),
             ("E2EL", x0, x0 + total + 6, C["ok"], "送出 → 收到完整回覆", 6)]
    for i, (lab, a, bb, col, note, st) in enumerate(marks):
        yy = ay + i * 30
        b.append(g([path(f"M{a},{yy} L{a},{yy+8} L{bb},{yy+8} L{bb},{yy}", stroke=col,
                         sw=1.4),
                    txt((a + bb) / 2, yy + 22, f"{lab}　{note}", cls="lbl b",
                        anchor="middle", size=11.2, fill=col)], s=st))
    yy = ay + 3 * 30 + 4
    b.append(g([rect(0, yy, 1000, 52, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, yy + 22, "TPOT = (E2EL − TTFT) ÷ (輸出 token 數 − 1)　"
                                 "算的是「每個 request 自己的平均」，不是所有 ITL 的平均。",
                    cls="lbl b", size=11.5, fill=C["mem"]),
                txt(14, yy + 40, "兩者的統計母體不同：ITL 的母體是「每一次間隔」，"
                                 "TPOT 的母體是「每一個 request」。混用會得到不同的 p95。",
                    cls="lbl", size=11, fill=C["mem"])], s=7))
    return svg(W, yy + 66, b)


# ---------------------------------------------------------- 百分位與 goodput
def fig_percentile():
    import math
    b = []
    x0, y0, w, h = 70, 210, 560, 170
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 12, "↑ request 數量", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 32, "延遲 →", cls="lbl s", anchor="end"))
    # 長尾分布
    pts = []
    for i in range(61):
        t = i / 60
        v = math.exp(-((t - 0.16) ** 2) / 0.012) + 0.30 * math.exp(-((t - 0.52) ** 2) / 0.09)
        pts.append((x0 + t * w, y0 - v / 1.15 * h))
    d = "M" + " L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    b.append(path(d + f" L{x0+w},{y0} L{x0},{y0} Z", stroke=C["compute"], sw=2,
                  fill="#eaf1fe", s=1))
    for frac, lab, col, st in ((0.20, "平均", C["neutral"], 2), (0.62, "p95", C["mem"], 3),
                               (0.80, "p99", C["bad"], 3)):
        x = x0 + frac * w
        b.append(line(x, y0, x, y0 - h * 0.92, stroke=col, sw=1.5, dash="5 4", s=st))
        b.append(txt(x, y0 - h * 0.92 - 8, lab, cls="lbl b", anchor="middle", size=11,
                     fill=col, ds=st))
    b.append(txt(x0 + 0.2 * w + 8, y0 - 20, "多數人很快", cls="lbl xs", ds="2"))
    b.append(txt(x0 + 0.72 * w, y0 - 62, "少數人很慢，而他們就是會來抱怨的人",
                 cls="lbl b", anchor="middle", size=10.6, fill=C["bad"], ds="3"))
    # goodput
    bx = 672
    b.append(g([rect(bx, 30, 328, 176, fill=C["card2"]),
                txt(bx + 14, 54, "Throughput vs Goodput", cls="lbl b", size=12.5),
                txt(bx + 14, 78, "throughput：所有輸出 token / 秒", cls="lbl", size=11),
                txt(bx + 14, 96, "goodput：只算「滿足 SLO」的部分", cls="lbl b", size=11,
                    fill=C["ok"]),
                line(bx + 14, 108, bx + 314, 108, stroke=C["line"], sw=1),
                txt(bx + 14, 128, "系統過載時 throughput 可能還很高，", cls="lbl", size=10.8),
                txt(bx + 14, 145, "但全部的人都超過 SLO，goodput = 0。", cls="lbl b",
                    size=10.8, fill=C["bad"]),
                txt(bx + 14, 168, "vLLM 的 bench serve 支援", cls="lbl", size=10.8),
                txt(bx + 14, 185, "--goodput ttft:500 tpot:40", cls="num", size=10.4,
                    fill=C["compute"])], s=4))
    b.append(g([rect(0, 244, 1000, 52, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, 266, "報告紀律：任何一個延遲數字都要寫清楚指標、百分位、量測窗與 workload。",
                    cls="lbl b", size=11.5, fill="#14532d"),
                txt(14, 284, "「TPOT 30 ms」沒有意義；「2k in / 512 out、32 併發、"
                             "穩定 5 分鐘窗內，p95 TPOT = 30 ms」才有。",
                    cls="lbl", size=11, fill="#14532d")], s=5))
    return svg(W, 312, b)


# ------------------------------------------------------------ 兩階段測試 ---
def fig_twostage():
    b = []
    boxes = [
        ("Stage 1　閉環飽和測試", C["compute"], C["computeW"], 0,
         ["控制變數：同時併發數（固定 N 個 client，回一個發一個）",
          "量測：achieved throughput、TPOT、ITL",
          "回答：這台機器<b>最多</b>能做多少",
          "特性：佇列不會爆炸，延遲隨併發單調上升"]),
        ("Stage 2　開環 SLO 容量測試", C["mem"], C["memW"], 1,
         ["控制變數：到達率（Poisson，與伺服器狀態無關）",
          "量測：p95/p99 TTFT、TPOT、goodput、佇列長度",
          "回答：在 SLO 之下<b>能承諾</b>多少",
          "特性：超過容量時延遲以 1/(1−ρ) 爆炸"]),
    ]
    y = 22
    for title, col, tint, st, items in boxes:
        b.append(g([rect(0, y, 1000, 116, r=8, fill=tint, stroke=col, sw=1.2),
                    txt(18, y + 26, title, cls="lbl b", size=13.5, fill=col)], s=st or None))
        for i, it in enumerate(items):
            b.append(g([circle(26, y + 48 + i * 18, 2.4, fill=col),
                        txt(38, y + 52 + i * 18, it.replace("<b>", "").replace("</b>", ""),
                            cls="lbl", size=11.2)], s=st or None))
        y += 130
    b.append(g([rect(0, y, 1000, 62, fill=C["badW"], stroke="#f5cdc8"),
                txt(16, y + 24, "為什麼不能只做 Stage 1：閉環測試的延遲不會爆炸，"
                                "因為 client 要等回覆才發下一個，", cls="lbl b", size=11.5,
                    fill=C["bad"]),
                txt(16, y + 44, "它自帶背壓。真實使用者不會等你，所以只有開環測試才能看到佇列失控的那條線。",
                    cls="lbl", size=11.2, fill=C["bad"])], s=2))
    return svg(W, y + 78, b)


# --------------------------------------------------------- SLO 邊界搜尋 ----
def fig_slo():
    b = []
    x0, y0, w, h = 70, 250, 680, 200
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 12, "↑ p95 TTFT", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 32, "到達率（req/s）→", cls="lbl s", anchor="end"))
    cap = 14.0
    mx = 2400.0

    def X(r):
        return x0 + r / 26 * w

    def Y(v):
        return y0 - min(v, mx) / mx * h

    def lat(r):
        rho = min(0.985, r / cap * 0.94)
        return 45 / (1 - rho)
    pts = [f"{X(r/2):.1f},{Y(lat(r/2)):.1f}" for r in range(2, 53)]
    b.append(path("M" + " L".join(pts), stroke=C["compute"], sw=2.6, s=1))
    slo = 800
    b.append(line(x0, Y(slo), x0 + w, Y(slo), stroke=C["bad"], sw=1.5, dash="6 5", s=2))
    b.append(txt(x0 + 8, Y(slo) - 8, "SLO：p95 TTFT ≤ 800 ms", cls="lbl b", size=11,
                 fill=C["bad"], ds="2"))
    # 找交點
    rc = 0
    for i in range(1, 2600):
        if lat(i / 100) > slo:
            rc = (i - 1) / 100
            break
    b.append(line(X(rc), y0, X(rc), Y(slo), stroke=C["ok"], sw=1.5, dash="5 4", s=3))
    b.append(circle(X(rc), Y(slo), 6, fill=C["ok"], stroke="#fff", sw=2, s=3))
    b.append(txt(X(rc) - 10, Y(slo) - 30, f"SLO 容量 ≈ {rc:.1f} req/s", cls="lbl b",
                 anchor="end", size=12, fill=C["ok"], ds="3"))
    # 掃描步驟
    coarse = [2, 6, 10, 14, 18, 22]
    for r in coarse:
        b.append(circle(X(r), Y(lat(r)), 4, fill=C["neutral"], s=1))
    fine = [11, 12, 13, 14]
    for r in fine:
        b.append(circle(X(r), Y(lat(r)), 4.5, fill=C["mem"], stroke="#fff", sw=1.4, s=2))
    for r in (0, 5, 10, 15, 20, 25):
        b.append(line(X(r), y0, X(r), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(r), y0 + 18, str(r), cls="num", anchor="middle"))
    for v in (0, 500, 1000, 1500, 2000):
        b.append(txt(x0 - 8, Y(v) + 4, str(v), cls="num", anchor="end"))
    b.append(g([rect(772, 40, 228, 200, fill=C["card2"]),
                txt(786, 62, "三段式搜尋", cls="lbl b", size=12),
                txt(786, 86, "① 粗掃", cls="lbl b", size=11, fill=C["neutral"]),
                txt(786, 102, "　大步長找出崩潰的區間", cls="lbl", size=10.6),
                txt(786, 124, "② 細掃", cls="lbl b", size=11, fill=C["mem"]),
                txt(786, 140, "　在區間內二分逼近", cls="lbl", size=10.6),
                txt(786, 162, "③ 重複確認", cls="lbl b", size=11, fill=C["ok"]),
                txt(786, 178, "　邊界點跑 3 次以上", cls="lbl", size=10.6),
                txt(786, 194, "　確認可重現、無趨勢", cls="lbl", size=10.6),
                line(786, 206, 986, 206, stroke=C["line"], sw=1),
                txt(786, 226, "每個點都要跑到穩定態", cls="lbl b", size=10.6,
                    fill=C["bad"])], s=4))
    b.append(g([rect(0, 288, 1000, 34, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, 310, "教學示例曲線（以 M/M/1 近似生成），不是實測結果。"
                             "真實曲線的形狀相同，但轉折位置完全取決於你的模型、硬體與 workload。",
                    cls="lbl b", size=11, fill=C["mem"])], s=5))
    return svg(W, 338, b)


# ------------------------------------------------------------- 決策樹 ------
def fig_decision():
    b = []

    def box(x, y, w, h, title, sub, col, tint, s=None):
        return g([rect(x, y, w, h, r=8, fill=tint, stroke=col, sw=1.3),
                  txt(x + w / 2, y + 22, title, cls="lbl b", anchor="middle", size=12,
                      fill=col),
                  txt(x + w / 2, y + 39, sub, cls="lbl xs", anchor="middle")], s=s)

    b.append(box(370, 10, 260, 50, "症狀是什麼？", "先看指標，別先動旗標",
                 C["neutral"], "#f4f6f8"))
    branches = [
        (0, "TTFT 太長", "第一個 token 等太久", C["compute"], C["computeW"],
         ["提高 max_num_batched_tokens", "開 prefix caching 並量命中率",
          "確認 chunked prefill 沒把它切太碎", "檢查佇列是否已飽和（那是容量問題）"], 1),
        (1, "TPOT / ITL 太長", "打字速度太慢", C["mem"], C["memW"],
         ["降低併發（max_num_seqs）", "開 speculative decoding（若 B 夠小）",
          "確認沒有落在 compute-bound 區", "量化 KV cache 減少讀取量"], 2),
        (2, "吞吐上不去", "整機 tok/s 太低", C["moe"], C["moeW"],
         ["提高併發直到撞到 N*", "關掉 speculative decoding（高併發時）",
          "確認 CUDA graph 有啟用", "檢查 preempt 次數"], 3),
        (3, "常常 preempt", "cache 不夠用", C["bad"], C["badW"],
         ["降 max_model_len 或 max_num_seqs", "KV 改 FP8", "考慮 SSM state 改 BF16",
          "換更小的模型或加卡"], 4),
    ]
    bw, gap = 236, 18
    for i, (idx, title, sub, col, tint, items, st) in enumerate(branches):
        x = idx * (bw + gap)
        b.append(path(f"M500,62 L500,80 L{x+bw/2},80 L{x+bw/2},96", stroke=col, sw=1.3,
                      s=st, **{"marker-end": "url(#ah)"}))
        b.append(box(x, 98, bw, 50, title, sub, col, tint, s=st))
        for j, it in enumerate(items):
            b.append(g([rect(x, 158 + j * 30, bw, 26, r=5, fill="#fff", stroke=C["line2"]),
                        rect(x, 158 + j * 30, 3, 26, r=0, fill=col, stroke=None),
                        txt(x + 12, 158 + j * 30 + 17, it, cls="lbl", size=10.2)], s=st))
    y = 158 + 4 * 30 + 8
    b.append(g([rect(0, y, 1000, 52, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, y + 22, "共通原則：一次只動一個旗標，每次用同一組 workload 與 SLO 重測，並記錄下來。",
                    cls="lbl b", size=11.5, fill="#14532d"),
                txt(14, y + 40, "沒有放諸四海的最佳設定，只有「這個 workload 配這組 SLO」的最佳設定。",
                    cls="lbl", size=11, fill="#14532d")], s=5))
    return svg(W, y + 66, b)


# ---------------------------------------------------------- TP / EP / DP ---
def fig_parallel():
    b = []
    modes = [
        ("Tensor Parallel (TP)", "把每一層的矩陣切開", C["compute"], C["computeW"],
         ["每層都要 all-reduce", "降低單卡權重與 KV（KV head 夠切時）",
          "122B 只有 2 組 KV head → TP 最多切到 2", "延遲敏感場景常用"], 1),
        ("Expert Parallel (EP)", "把 MoE 的專家分散", C["moe"], C["moeW"],
         ["每層 all-to-all 把 token 送到專家", "只有 MoE 模型適用",
          "大幅降低單卡權重", "負載不均會放大尾端延遲"], 2),
        ("Data Parallel (DP)", "整份模型複製多份", C["ok"], C["okW"],
         ["沒有跨卡通訊", "吞吐線性成長", "每張卡都要放得下整個模型",
          "NVFP4 讓 122B 可以這樣做"], 3),
    ]
    x = 0
    for title, sub, col, tint, items, st in modes:
        b.append(g([rect(x, 10, 322, 60, r=8, fill=tint, stroke=col, sw=1.3),
                    txt(x + 161, 34, title, cls="lbl b", anchor="middle", size=13,
                        fill=col),
                    txt(x + 161, 54, sub, cls="lbl s", anchor="middle")], s=st))
        for j, it in enumerate(items):
            b.append(g([rect(x, 82 + j * 32, 322, 28, r=5, fill="#fff", stroke=C["line2"]),
                        rect(x, 82 + j * 32, 3, 28, r=0, fill=col, stroke=None),
                        txt(x + 12, 82 + j * 32 + 18, it, cls="lbl", size=10.6)], s=st))
        x += 339
    y = 82 + 4 * 32 + 8
    b.append(g([rect(0, y, 1000, 70, fill=C["computeW"], stroke="#d5e4fd"),
                txt(14, y + 24, "本次評估的關鍵結論：NVFP4 把 122B 壓到 83.5 GB，"
                                "可以用 TP=1 單卡跑。", cls="lbl b", size=12,
                    fill=C["compute"]),
                txt(14, y + 44, "8 卡的節點因此可以跑 8 份獨立的 DP 副本，取代 1 份 TP=8。"
                                "沒有 all-reduce，也沒有通訊抖動，", cls="lbl", size=11,
                    fill=C["compute"]),
                txt(14, y + 62, "而且每份副本的 cache 空間完全獨立。缺點是無法服務超長 context"
                                "（單卡 cache 有限）。", cls="lbl", size=11,
                    fill=C["compute"])], s=4))
    return svg(W, y + 86, b)
