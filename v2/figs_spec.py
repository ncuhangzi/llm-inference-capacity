# -*- coding: utf-8 -*-
"""CH5–CH7 的圖：量化位元佈局、量化地圖、speculative decoding。"""
from svgkit import *
import model as M

W = 1000


# --------------------------------------------------------- NVFP4 位元佈局 --
def fig_nvfp4():
    b = []
    b.append(txt(0, 14, "一個 NVFP4 block：16 個 4-bit 值 + 1 個 FP8 block scale",
                 cls="lbl b", size=12.5, fill=C["moe"]))
    x0, y0, cw = 0, 24, 46
    vals = ["-1.5", "0.5", "3.0", "-6.0", "1.0", "0.0", "2.0", "-0.5",
            "4.0", "1.5", "-3.0", "0.5", "6.0", "-2.0", "1.0", "-4.0"]
    for i, v in enumerate(vals):
        b.append(cell(x0 + i * (cw + 3), y0, cw, 30, v, fill=C["moeW"], stroke=C["moe"],
                      color=C["moe"], size=10, r=3))
    b.append(bracket(x0, y0 + 34, y0 + 44, "", color=C["moe"]))
    b.append(txt(x0 + 8 * (cw + 3), y0 + 50, "16 個 E2M1（1 符號 + 2 指數 + 1 尾數 = 4 bit）",
                 cls="lbl s", anchor="middle"))
    b.append(txt(x0 + 8 * (cw + 3), y0 + 66,
                 "可表示的非零大小只有 {0.5, 1, 1.5, 2, 3, 4, 6}", cls="lbl xs",
                 anchor="middle"))
    y1 = y0 + 80
    b.append(g([rect(x0, y1, 210, 40, r=6, fill=C["memW"], stroke=C["mem"], sw=1.2),
                txt(x0 + 105, y1 + 18, "block scale", cls="lbl b", anchor="middle",
                    size=11.5, fill=C["mem"]),
                txt(x0 + 105, y1 + 32, "FP8 E4M3（8 bit）／每 16 個值一份",
                    cls="lbl xs", anchor="middle")], s=1))
    b.append(g([rect(x0 + 224, y1, 210, 40, r=6, fill=C["computeW"], stroke=C["compute"],
                     sw=1.2),
                txt(x0 + 329, y1 + 18, "tensor scale", cls="lbl b", anchor="middle",
                    size=11.5, fill=C["compute"]),
                txt(x0 + 329, y1 + 32, "FP32／每個張量一份（weight_scale_2）",
                    cls="lbl xs", anchor="middle")], s=2))
    b.append(txt(x0 + 448, y1 + 25, "→ 平均 (16×4 + 8) / 16 = 4.5 bit / 值",
                 cls="lbl b", size=12.5, fill=C["moe"], ds="3"))
    # 對照 FP8
    y2 = y1 + 58
    b.append(txt(0, y2, "對照：Qwen3.8-27B-FP8 用的 blockwise-128", cls="lbl b", size=12.5,
                 fill=C["ssm"], ds="4"))
    b.append(g([rect(0, y2 + 10, 640, 34, r=5, fill=C["ssmW"], stroke=C["ssm"], sw=1),
                txt(16, y2 + 32, "128 個 FP8 E4M3 值（每個 8 bit）", cls="lbl", size=11.5,
                    fill=C["ssm"]),
                rect(650, y2 + 10, 160, 34, r=5, fill=C["memW"], stroke=C["mem"], sw=1),
                txt(730, y2 + 32, "1 個 FP32 scale", cls="lbl", anchor="middle", size=11,
                    fill=C["mem"]),
                txt(824, y2 + 32, "→ 8.03 bit / 值", cls="lbl b", size=11.5,
                    fill=C["ssm"])], s=4))
    y3 = y2 + 58
    rows = [("為什麼 block 要小", "block 越小，一組值裡的動態範圍越窄，"
                                "4 bit 的 8 個大小級距就越夠用。NVFP4 用 16，MXFP4 用 32。", 5),
            ("為什麼 scale 要用 FP8 而不是 2 的冪次",
             "E4M3 是真正的浮點數，能表示 1.75× 這種比例；MXFP4 的 E8M0 只能表示 2ⁿ，"
             "誤差較大。代價是 scale 的儲存開銷加倍。", 5),
            ("兩層 scale 的分工", "block scale 處理局部差異，tensor scale 處理整體量級。"
                                "兩者相乘才還原成原始權重。", 5)]
    for i, (k, v, st) in enumerate(rows):
        b.append(g([rect(0, y3 + i * 34, 1000, 30, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y3 + i * 34, 3, 30, r=0, fill=C["moe"], stroke=None),
                    txt(13, y3 + i * 34 + 19, k, cls="lbl b", size=11, fill=C["moe"]),
                    txt(250, y3 + i * 34 + 19, v, cls="lbl", size=11)], s=st))
    return svg(W, y3 + 3 * 34 + 4, b)


# ------------------------------------------------- checkpoint 的量化地圖 ---
def fig_quantmap():
    b = []
    m35, m38 = M.Q35, M.Q38
    b.append(txt(0, 14, "從 safetensors 的 weight map 直接看出來：誰有 weight_scale，誰就被量化了",
                 cls="lbl s"))

    def barrow(y, title, segs, total, note):
        out = [txt(0, y + 15, title, cls="lbl b", size=12)]
        x = 210
        wtot = 640
        for lab, v, col, tint in segs:
            wd = v / total * wtot
            out.append(rect(x, y, wd, 30, r=3, fill=tint, stroke=col, sw=1))
            if wd > 42:
                out.append(txt(x + wd / 2, y + 19, lab, cls="lbl", anchor="middle",
                               size=9.6, fill=col))
            x += wd
        out.append(txt(866, y + 19, note, cls="lbl b", size=11, fill=C["mem"]))
        return out

    y = 30
    q = m35.real_dtypes
    b.extend(barrow(y, "122B　NVFP4 checkpoint",
                    [("NVFP4 routed experts", q["U8"], C["moe"], C["moeW"]),
                     ("FP8 scale", q["F8_E4M3"], C["mem"], C["memW"]),
                     ("BF16 其餘全部", q["BF16"] * 2, C["compute"], C["computeW"])],
                    m35.real_weight_bytes, f"{m35.real_weight_bytes/1e9:,.1f} GB"))
    y += 44
    detail = [("被量化成 NVFP4", f"48 層 × 256 個 routed expert 的 gate/up/down "
                                f"= {m35.quant_params/1e9:,.1f} B 參數", C["moe"]),
              ("維持 BF16 不量化", "全部 36 層 GDN、12 層 full attention、48 個 shared expert、"
                                "router、embedding、lm_head、視覺塔、MTP head", C["compute"])]
    for i, (k, v, col) in enumerate(detail):
        b.append(g([rect(0, y + i * 28, 1000, 24, r=4, fill="#fff", stroke=C["line2"]),
                    rect(0, y + i * 28, 3, 24, r=0, fill=col, stroke=None),
                    txt(13, y + i * 28 + 16, k, cls="lbl b", size=10.6, fill=col),
                    txt(160, y + i * 28 + 16, v, cls="lbl", size=10.6)], s=1))
    y += 66
    q = m38.real_dtypes
    b.extend(barrow(y, "27B　FP8 checkpoint",
                    [("FP8 blockwise-128", q["F8_E4M3"], C["ssm"], C["ssmW"]),
                     ("BF16 其餘", q["BF16"] * 2, C["compute"], C["computeW"])],
                    m38.real_weight_bytes, f"{m38.real_weight_bytes/1e9:,.1f} GB"))
    y += 44
    detail = [("被量化成 FP8", "64 層 dense FFN、16 層 attention 的 q/k/v/o、"
                             "48 層 GDN 的 in_proj_qkv / in_proj_z / out_proj、MTP head", C["ssm"]),
              ("維持 BF16 不量化", "GDN 的 in_proj_a / in_proj_b / conv1d / A_log / dt_bias、"
                                "所有 norm、embedding、lm_head、視覺塔", C["compute"])]
    for i, (k, v, col) in enumerate(detail):
        b.append(g([rect(0, y + i * 28, 1000, 24, r=4, fill="#fff", stroke=C["line2"]),
                    rect(0, y + i * 28, 3, 24, r=0, fill=col, stroke=None),
                    txt(13, y + i * 28 + 16, k, cls="lbl b", size=10.6, fill=col),
                    txt(160, y + i * 28 + 16, v, cls="lbl", size=10.6)], s=2))
    y += 68
    b.append(g([rect(0, y, 1000, 54, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, y + 22, "驗算：48 × 256 × 3 × 3072 × 1024 = 115,964,116,992 個 4-bit 權重"
                                " → 57,982,058,496 bytes，與 checkpoint 的 U8 統計完全一致；",
                    cls="lbl b", size=11, fill=C["mem"]),
                txt(14, y + 40, "÷ 16 = 7,247,757,312 個 FP8 block scale，也與 F8_E4M3 統計完全一致。"
                                "27B 的 FP8 參數量同樣可以逐張量湊到分毫不差。",
                    cls="lbl", size=11, fill=C["mem"])], s=3))
    return svg(W, y + 68, b)


# ---------------------------------------------- 為什麼可以「猜」 ------------
def fig_spec_why():
    b = []
    m = M.Q38
    b.append(txt(0, 14, "decode 時，GPU 的兩種資源使用率差距很大", cls="lbl s"))
    rows = [("記憶體頻寬", 1.0, C["mem"], C["memW"],
             f"每個 step 要讀 {m.dense_read_bytes/1e9:,.1f} GB 權重"),
            ("張量核心算力", 32 * m.token_compute_s / (m.dense_read_bytes / M.GPU.bw),
             C["compute"], C["computeW"], "32 併發時只用到這麼多")]
    y = 30
    for lab, frac, col, tint, note in rows:
        b.append(txt(0, y + 22, lab, cls="lbl b", size=12))
        b.append(rect(150, y, 640, 32, r=5, fill=C["line2"], stroke=None))
        b.append(rect(150, y, 640 * min(1, frac), 32, r=5, fill=col, stroke=None))
        b.append(txt(806, y + 21, f"{frac*100:,.0f}%", cls="num", size=13, fill=col))
        b.append(txt(864, y + 21, note, cls="lbl xs"))
        y += 44
    b.append(g([rect(150, y + 4, 640 * (1 - 32 * m.token_compute_s /
                                        (m.dense_read_bytes / M.GPU.bw)), 26, r=5,
                     fill=C["specW"], stroke=C["spec"], sw=1.2),
                txt(150 + 640 * (1 - 32 * m.token_compute_s /
                                 (m.dense_read_bytes / M.GPU.bw)) / 2, y + 22,
                    "這一塊算力是免費的", cls="lbl b", anchor="middle", size=11.5,
                    fill=C["spec"])], s=1))
    y += 44
    b.append(g([rect(0, y, 1000, 56, fill=C["specW"], stroke="#f6cfe3"),
                txt(14, y + 22, "speculative decoding 就是打這塊閒置算力的主意：拿它去猜後面幾個 token，",
                    cls="lbl b", size=12, fill=C["spec"]),
                txt(14, y + 42, "再讓目標模型用同一次權重讀取一口氣驗證。猜對就一次前進好幾格。",
                    cls="lbl", size=12, fill=C["spec"])], s=2))
    y += 70
    b.append(g([rect(0, y, 490, 74, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, y + 22, "輸出品質不變", cls="lbl b", size=12, fill="#14532d"),
                txt(14, y + 42, "驗證用 rejection sampling：逐位置接受或拒絕，", cls="lbl",
                    size=11, fill="#14532d"),
                txt(14, y + 60, "數學上與不猜時的輸出分布相同。", cls="lbl", size=11,
                    fill="#14532d")], s=3))
    b.append(g([rect(510, y, 490, 74, fill=C["badW"], stroke="#f5cdc8"),
                txt(524, y + 22, "但這塊算力不是無限的", cls="lbl b", size=12, fill=C["bad"]),
                txt(524, y + 42, "併發一高，閒置算力就被用光；此時再猜就是純浪費。",
                    cls="lbl", size=11, fill=C["bad"]),
                txt(524, y + 60, "本章要量的就是那條界線。", cls="lbl", size=11,
                    fill=C["bad"])], s=3))
    return svg(W, y + 88, b)


# ------------------------------------------- draft → verify → accept -------
def fig_spec_cycle():
    b = []
    cw, chh = 60, 34
    x0 = 116
    b.append(txt(0, 22, "已確定的輸出", cls="lbl s"))
    ctx = ["模型", "推論", "的", "瓶頸"]
    for i, t in enumerate(ctx):
        b.append(cell(x0 + i * (cw + 4), 4, cw, chh, t, fill=C["card2"], size=11))
    dx = x0 + len(ctx) * (cw + 4)
    # 草稿
    drafts = [("在於", True), ("記憶", True), ("體", True), ("速度", False),
              ("而", False), ("不是", False), ("算力", False)]
    b.append(txt(0, 68, "起草 k=7", cls="lbl b", size=11.5, fill=C["spec"], ds="1"))
    for i, (t, ok) in enumerate(drafts):
        b.append(cell(dx + i * (cw + 4), 50, cw, chh, t, fill=C["specW"],
                      stroke=C["spec"], color=C["spec"], size=11, s=1))
    b.append(txt(dx + len(drafts) * (cw + 4) + 12, 72,
                 "一次產生 7 個草稿", cls="lbl xs", fill=C["spec"], ds="1"))
    # 驗證
    b.append(txt(0, 114, "驗證", cls="lbl b", size=11.5, fill=C["compute"], ds="2"))
    b.append(g([rect(dx - 8, 96, len(drafts) * (cw + 4) + 8, chh + 4, r=6,
                     fill=C["computeW"], stroke=C["compute"], sw=1.2),
                txt(dx + len(drafts) * (cw + 4) / 2 - 4, 118,
                    "目標模型一次 forward 同時驗證全部 8 個位置（1 個已知 + 7 個草稿）",
                    cls="lbl b", anchor="middle", size=11, fill=C["compute"])], s=2))
    # 結果
    b.append(txt(0, 168, "結果", cls="lbl b", size=11.5, fill=C["ok"], ds="3"))
    for i, (t, ok) in enumerate(drafts):
        b.append(cell(dx + i * (cw + 4), 150, cw, chh, t,
                      fill=C["okW"] if ok else C["badW"],
                      stroke=C["ok"] if ok else C["bad"],
                      color=C["ok"] if ok else C["bad"], size=11, s=3, weight=650))
    b.append(cell(dx + 3 * (cw + 4), 150, cw, chh, "而", fill=C["okW"], stroke=C["ok"],
                  color=C["ok"], size=11, s=4, weight=650))
    b.append(txt(dx + 3 * (cw + 4) + cw / 2, 202, "↑ 保底 token", cls="lbl xs",
                 anchor="middle", fill=C["ok"], ds="4"))
    b.append(txt(dx + len(drafts) * (cw + 4) + 12, 172, "第 4 個起被拒絕",
                 cls="lbl b", size=10.6, fill=C["bad"], ds="3"))
    b.append(txt(dx + len(drafts) * (cw + 4) + 12, 188, "→ 後面整串丟掉",
                 cls="lbl", size=10.6, fill=C["bad"], ds="3"))
    y = 214
    b.append(g([rect(0, y, 1000, 34, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, y + 22, "這一輪前進了 4 格（3 個被接受的草稿 + 1 個保底 token）"
                                "，只花了 1 次起草 + 1 次目標模型 forward。",
                    cls="lbl b", size=11.5, fill="#14532d")], s=4))
    y += 46
    rows = [("被接受的 token", "確定與不猜時完全一樣（rejection sampling 保證）", C["ok"], 5),
            ("被拒絕之後", "整串後面都丟掉，因為它們是以錯誤的前綴為條件產生的", C["bad"], 5),
            ("保底 token", "拒絕發生的位置，直接採用目標模型自己的輸出。"
                          "所以最差情況也會前進 1 格，就 token 數而言不會比不猜慢",
             C["compute"], 5)]
    for i, (k, v, col, st) in enumerate(rows):
        b.append(g([rect(0, y + i * 30, 1000, 26, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y + i * 30, 3, 26, r=0, fill=col, stroke=None),
                    txt(13, y + i * 30 + 17, k, cls="lbl b", size=11, fill=col),
                    txt(160, y + i * 30 + 17, v, cls="lbl", size=11)], s=st))
    return svg(W, y + 3 * 30 + 4, b)


# ------------------------------------------------ MTP vs DFlash2 時間軸 ----
def fig_spec_timeline():
    b = []
    unit = 62
    x0 = 176
    b.append(txt(0, 16, "同樣要產生 8 個位置的草稿，兩種起草器的時間軸完全不同",
                 cls="lbl s"))
    # MTP：序列 k 次
    y = 34
    b.append(txt(0, y + 24, "122B · MTP", cls="lbl b", size=12, fill=C["spec"]))
    b.append(txt(0, y + 40, "k = 3", cls="lbl s"))
    for i in range(3):
        b.append(g([rect(x0 + i * (unit + 4), y, unit, 34, r=4, fill=C["specW"],
                         stroke=C["spec"], sw=1.1),
                    txt(x0 + i * (unit + 4) + unit / 2, y + 22, f"draft {i+1}",
                        cls="lbl", anchor="middle", size=10, fill=C["spec"])], s=1))
    vx = x0 + 3 * (unit + 4)
    b.append(g([rect(vx, y, unit * 2.6, 34, r=4, fill=C["computeW"], stroke=C["compute"],
                     sw=1.2),
                txt(vx + unit * 1.3, y + 22, "verify 4 個位置", cls="lbl b",
                    anchor="middle", size=10.5, fill=C["compute"])], s=2))
    b.append(txt(vx + unit * 2.6 + 14, y + 22,
                 "每次 draft 都要重讀一次 lm_head（1.53 GB）→ 起草成本 × 3",
                 cls="lbl", size=10.6, fill=C["spec"], ds="2"))
    # DFlash2：一次
    y = 108
    b.append(txt(0, y + 24, "27B · DFlash2", cls="lbl b", size=12, fill=C["ssm"]))
    b.append(txt(0, y + 40, "k = 7（block size 8）", cls="lbl s"))
    b.append(g([rect(x0, y, unit * 1.4, 34, r=4, fill=C["ssmW"], stroke=C["ssm"], sw=1.1),
                txt(x0 + unit * 0.7, y + 22, "draft ×1", cls="lbl b", anchor="middle",
                    size=10.5, fill=C["ssm"])], s=3))
    vx2 = x0 + unit * 1.4 + 4
    b.append(g([rect(vx2, y, unit * 4.2, 34, r=4, fill=C["computeW"], stroke=C["compute"],
                     sw=1.2),
                txt(vx2 + unit * 2.1, y + 22, "verify 8 個位置", cls="lbl b",
                    anchor="middle", size=10.5, fill=C["compute"])], s=4))
    b.append(txt(vx2 + unit * 4.2 + 14, y + 22,
                 "block diffusion：一次 forward 同時猜完整塊",
                 cls="lbl", size=10.6, fill=C["ssm"], ds="4"))
    y = 176
    b.append(g([rect(0, y, 1000, 34, fill=C["computeW"], stroke="#d5e4fd"),
                txt(14, y + 22, "差別在成本結構：MTP 的起草成本隨 k 線性成長（T_draft = k × t_step），"
                                "DFlash2 的起草成本與 k 無關（T_draft = t_parallel）。",
                    cls="lbl b", size=11.5, fill=C["compute"])], s=5))
    y += 44
    rows = [("MTP（Qwen3.5 內建）",
             "1 層 decoder，autoregressive 跑 k 次。與目標模型同時訓練，接受率高；"
             "但每一次都要再過 lm_head，k 一大就不划算。", C["spec"], 6),
            ("DFlash2（block diffusion）",
             "獨立的 2 B 起草模型，用非因果 attention 同時看目標模型的 hidden state "
             "與整排 mask token，一次 forward 產生整塊；再用輕量 selector 挑一條路徑。",
             C["ssm"], 6)]
    for i, (k, v, col, st) in enumerate(rows):
        b.append(g([rect(0, y + i * 42, 1000, 38, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y + i * 42, 3, 38, r=0, fill=col, stroke=None),
                    txt(13, y + i * 42 + 16, k, cls="lbl b", size=11, fill=col),
                    txt(13, y + i * 42 + 31, v, cls="lbl", size=10.8)], s=st))
    return svg(W, y + 2 * 42 + 4, b)


# ---------------------------------------------------- 接受率與 τ -----------
def fig_accept():
    import math
    b = []
    x0, y0, w, h = 66, 210, 400, 168
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 12, "↑ 該位置被接受的機率", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 32, "第幾個草稿位置 →", cls="lbl s", anchor="end"))
    for a, col, lab, st in ((0.85, C["ok"], "α = 0.85", 1), (0.72, C["spec"], "α = 0.72", 1),
                            (0.55, C["bad"], "α = 0.55", 1)):
        pts = []
        for i in range(1, 9):
            x = x0 + (i - 1) / 7 * w
            y = y0 - a ** i * h
            pts.append(f"{x:.1f},{y:.1f}")
            b.append(circle(x, y, 3.2, fill=col, s=st))
        b.append(path("M" + " L".join(pts), stroke=col, sw=2, s=st))
        b.append(txt(x0 + w + 6, y0 - a ** 8 * h + 4, lab, cls="lbl b", size=10.4,
                     fill=col, ds=st))
    for i in range(1, 9):
        b.append(txt(x0 + (i - 1) / 7 * w, y0 + 16, f"+{i}", cls="num", anchor="middle"))
    # τ 表
    bx = 560
    b.append(txt(bx, 26, "期望接受長度 τ = Σ αⁱ = (1 − α^(k+1)) / (1 − α)",
                 cls="lbl b", size=11.5, fill=C["spec"], ds="2"))
    heads = ["α", "k=1", "k=3", "k=7", "k=15"]
    for j, hd in enumerate(heads):
        b.append(txt(bx + (0 if j == 0 else 62 + j * 76), 50, hd, cls="lbl xs",
                     anchor="start" if j == 0 else "end", ds="2"))
    b.append(line(bx, 56, bx + 380, 56, stroke=C["line"], sw=1, s=2))
    for r, a in enumerate([0.55, 0.72, 0.80, 0.85, 0.90]):
        yy = 76 + r * 22
        b.append(txt(bx, yy, f"{a:.2f}", cls="num", size=10.6, ds="2"))
        for j, k in enumerate([1, 3, 7, 15]):
            v = M.tau_from_alpha(a, k)
            b.append(txt(bx + 62 + (j + 1) * 76, yy, f"{v:.2f}", cls="num", size=10.6,
                         anchor="end", fill=C["spec"] if v > 3 else C["ink2"], ds="2"))
    b.append(g([rect(bx, 190, 380, 74, fill=C["memW"], stroke="#f0dcb4"),
                txt(bx + 13, 212, "報酬遞減很快", cls="lbl b", size=11.5, fill=C["mem"]),
                txt(bx + 13, 231, "α = 0.80 時，k 從 7 加到 15，τ 只從 4.16", cls="lbl",
                    size=10.6, fill=C["mem"]),
                txt(bx + 13, 248, "長到 4.80（+15%），但驗證成本翻倍。", cls="lbl",
                    size=10.6, fill=C["mem"])], s=3))
    b.append(g([rect(0, 292, 1000, 52, fill=C["specW"], stroke="#f6cfe3"),
                txt(14, 314, "實測參考：DFlash2 在 Qwen3.8-27B 上，concurrency 1 的接受長度是 "
                             "GSM8K 5.46、MATH-500 5.28、MBPP 4.79、HumanEval 4.39；",
                    cls="lbl b", size=11, fill=C["spec"]),
                txt(14, 332, "對應 k = 7、α ≈ 0.80–0.86。EAGLE-3 論文在 LLaMA-3.1-8B 上報告 τ = 6.23。",
                    cls="lbl", size=11, fill=C["spec"])], s=4))
    return svg(W, 356, b)


# ------------------------------------- speedup vs batch（由解析模型算出）----
def fig_spec_batch():
    import math
    b = []
    x0, y0, w, h = 74, 300, 620, 250
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 12, "↑ 相對不用 spec 的輸出速率倍數", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 34, "同時併發序列數 B（對數）→", cls="lbl s", anchor="end"))
    lo, hi = 1, 512
    ymax = 5.0

    def X(v):
        return x0 + math.log2(v / lo) / math.log2(hi / lo) * w

    def Y(v):
        return y0 - min(v, ymax) / ymax * h

    b.append(line(x0, Y(1), x0 + w, Y(1), stroke=C["bad"], sw=1.4, dash="6 5"))
    b.append(txt(x0 + 6, Y(1) - 7, "1.0×　低於這條線就是純浪費", cls="lbl xs", fill=C["bad"]))
    for v in (1, 2, 3, 4, 5):
        b.append(txt(x0 - 8, Y(v) + 4, f"{v}×", cls="num", anchor="end"))
        b.append(line(x0 - 3, Y(v), x0, Y(v), stroke=C["line"], sw=1))
    series = [(M.Q38, 7, C["ssm"], f"27B + DFlash2 (k=7, τ={M.SPEC_TAU['q38']:.2f})", 1),
              (M.Q35, 3, C["spec"], f"122B + MTP (k=3, τ={M.SPEC_TAU['q35']:.2f})", 2)]
    for m, k, col, lab, st in series:
        pts = []
        for e in range(0, 41):
            B = lo * (hi / lo) ** (e / 40)
            pts.append(f"{X(B):.1f},{Y(M.spec_speedup(m, B, k)):.1f}")
        b.append(path("M" + " L".join(pts), stroke=col, sw=2.8, s=st))
        be = M.spec_break_even(m, k)
        if be < hi:
            b.append(circle(X(be), Y(1), 5, fill=col, stroke="#fff", sw=1.8, s=st))
            b.append(txt(X(be), Y(1) + 22, f"損益兩平 B ≈ {be:,.0f}", cls="lbl b",
                         anchor="middle", size=10.4, fill=col, ds=st))
        b.append(txt(X(1.15), Y(M.spec_speedup(m, 1.15, k)) - 10, lab, cls="lbl b",
                     size=11, fill=col, ds=st))
    for v in (1, 4, 16, 64, 256):
        b.append(line(X(v), y0, X(v), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(v), y0 + 18, str(v), cls="num", anchor="middle"))
    knee = M.GPU.knee_tokens("fp8")
    for m, k, col, st in ((M.Q38, 7, C["ssm"], 3),):
        bx = knee / (k + 1)
        b.append(line(X(bx), y0 - h, X(bx), y0, stroke=C["mem"], sw=1.3, dash="4 4", s=st))
        b.append(txt(X(bx) + 6, y0 - h + 16, f"B × (k+1) 撞到 N* = {knee:.0f}",
                     cls="lbl b", size=10.6, fill=C["mem"], ds=st))
        b.append(txt(X(bx) + 6, y0 - h + 32, f"→ B ≈ {bx:,.0f}", cls="lbl", size=10.6,
                     fill=C["mem"], ds=st))
    b.append(g([rect(716, 40, 284, 226, fill=C["card2"]),
                txt(730, 62, "為什麼兩條線形狀不同", cls="lbl b", size=12),
                txt(730, 86, "27B 是 dense 模型：每個 step 讀", cls="lbl", size=10.8),
                txt(730, 102, "的權重固定，B×8 一旦超過 N*", cls="lbl", size=10.8),
                txt(730, 118, "就變成 compute-bound，加速直", cls="lbl", size=10.8),
                txt(730, 134, "線墜落。", cls="lbl b", size=10.8, fill=C["ssm"]),
                line(730, 146, 986, 146, stroke=C["line"], sw=1),
                txt(730, 168, "122B 是 MoE：batch 越大，被碰", cls="lbl", size=10.8),
                txt(730, 184, "到的 expert 越多、要讀的權重也", cls="lbl", size=10.8),
                txt(730, 200, "越多，所以低併發時 spec 的優勢", cls="lbl", size=10.8),
                txt(730, 216, "反而被吃掉；但高併發時 expert", cls="lbl", size=10.8),
                txt(730, 232, "讀取被更多 token 分攤，優勢回", cls="lbl", size=10.8),
                txt(730, 248, "升。", cls="lbl b", size=10.8, fill=C["spec"])], s=4))
    return svg(W, y0 + 44, b)


# --------------------------------------------- 接受率隨 context 衰減 -------
def fig_accept_ctx():
    b = []
    data = [(2048, 138.04, 60.39, [0.935, 0.829, 0.715, 0.585, 0.463, 0.366]),
            (4096, 119.51, 60.49, None),
            (8192, 88.25, 60.10, None),
            (16384, 51.03, 59.13, None),
            (30720, 26.81, 54.55, [0.721, 0.512, 0.395, 0.326, 0.256, 0.140])]
    x0, y0, w, h = 70, 236, 470, 196
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 12, "↑ decode 吞吐（tok/s）", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 32, "context 長度 →", cls="lbl s", anchor="end"))
    import math
    mx = 150

    def X(t):
        return x0 + math.log2(t / 2048) / math.log2(30720 / 2048) * w

    def Y(v):
        return y0 - v / mx * h
    for key, col, lab, st in ((1, C["spec"], "開 MTP（k=6）", 1), (2, C["neutral"], "不開", 2)):
        pts = [f"{X(d[0]):.1f},{Y(d[key]):.1f}" for d in data]
        b.append(path("M" + " L".join(pts), stroke=col, sw=2.6, s=st))
        for d in data:
            b.append(circle(X(d[0]), Y(d[key]), 3.6, fill=col, s=st))
        b.append(txt(X(2048) + 6, Y(data[0][key]) - 10, lab, cls="lbl b", size=10.6,
                     fill=col, ds=st))
    # 交叉點
    b.append(circle(X(12000), Y(59.6), 5.5, fill=C["bad"], stroke="#fff", sw=1.8, s=3))
    b.append(txt(X(12000) + 8, Y(59.6) - 10, "約 12k 之後開 MTP 反而更慢", cls="lbl b",
                 size=10.6, fill=C["bad"], ds="3"))
    for t in (2048, 8192, 30720):
        b.append(line(X(t), y0, X(t), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(t), y0 + 18, f"{t//1024}k", cls="num", anchor="middle"))
    for v in (0, 50, 100, 150):
        b.append(txt(x0 - 8, Y(v) + 4, str(v), cls="num", anchor="end"))
    # 逐位置接受率
    bx = 596
    b.append(txt(bx, 24, "逐位置接受率也整條下降", cls="lbl b", size=12, fill=C["spec"],
                 ds="4"))
    for gi, (ctx, _, _, acc) in enumerate([data[0], data[-1]]):
        yy = 44 + gi * 96
        col = C["ok"] if gi == 0 else C["bad"]
        b.append(txt(bx, yy + 10, f"context ≈ {ctx//1024}k　平均 "
                                  f"{sum(acc)/len(acc)*100:,.1f}%", cls="lbl b",
                     size=10.8, fill=col, ds="4"))
        for i, a in enumerate(acc):
            bw = 56
            b.append(rect(bx + i * (bw + 5), yy + 20 + (1 - a) * 44, bw, a * 44, r=2,
                          fill=col, stroke=None, opacity=.75, s=4))
            b.append(txt(bx + i * (bw + 5) + bw / 2, yy + 78, f"+{i+1}", cls="num",
                         anchor="middle", size=9, ds="4"))
            b.append(txt(bx + i * (bw + 5) + bw / 2, yy + 14, f"{a*100:.0f}",
                         cls="num", anchor="middle", size=9, fill=col, ds="4"))
    b.append(g([rect(0, 268, 1000, 52, fill=C["badW"], stroke="#f5cdc8"),
                txt(14, 290, "資料來源：vLLM issue #47602 在 Qwen3.6-27B 上的實測。"
                             "context 從 2k 長到 30k，平均接受率從 64.9% 掉到 39.1%，",
                    cls="lbl b", size=11, fill=C["bad"]),
                txt(14, 308, "吞吐從「比不開快 129%」變成「比不開慢 51%」。"
                             "長 context 服務一定要重新量接受率，不能沿用短 prompt 的結論。",
                    cls="lbl", size=11, fill=C["bad"])], s=5))
    return svg(W, 332, b)


# -------------------------------------------- spec decoding 的記憶體成本 ---
def fig_spec_mem():
    b = []
    m35, m38 = M.Q35, M.Q38
    b.append(txt(0, 14, "開了 speculative decoding，記憶體帳本會多出三筆", cls="lbl s"))
    y = 26
    items = [
        ("① 起草器的權重", C["spec"],
         [(f"122B · MTP head", m35.spec_draft_params * 2),
          (f"27B · DFlash2", m38.spec_draft_params * 2)],
         "一次性成本，直接從 cache 預算裡扣掉", 1),
        ("② GDN 的 conv state 變長", C["ssm"],
         [(f"122B · {m35.conv_k}−1+{m35.spec_k} = {m35.conv_k-1+m35.spec_k}",
           m35.ssm_conv_bytes(m35.spec_k) - m35.ssm_conv_bytes(0)),
          (f"27B · {m38.conv_k}−1+{m38.spec_k} = {m38.conv_k-1+m38.spec_k}",
           m38.ssm_conv_bytes(m38.spec_k) - m38.ssm_conv_bytes(0))],
         "vLLM 的 conv_state_shape = (conv_dim, conv_kernel−1+num_spec)，每條序列都要多留", 2),
        ("③ 每條序列要多預留 k 個 KV slot", C["mem"],
         [(f"122B · k={m35.spec_k}", m35.spec_k * m35.kv_bytes_per_token("fp8")),
          (f"27B · k={m38.spec_k}", m38.spec_k * m38.kv_bytes_per_token("bf16"))],
         "驗證時 k+1 個位置的 K/V 都要有地方放；被拒絕的再釋放", 3),
    ]
    for title, col, vals, note, st in items:
        b.append(g([txt(0, y + 16, title, cls="lbl b", size=12, fill=col),
                    txt(0, y + 34, note, cls="lbl s", size=10.4)], s=st))
        for i, (lab, v) in enumerate(vals):
            bx = 470 + i * 268
            unit = "GB" if v > 1e8 else ("MiB" if v > 1e5 else "KiB")
            val = (v / 1e9 if unit == "GB" else (v / M.MiB if unit == "MiB" else v / 1024))
            b.append(g([rect(bx, y, 254, 40, r=6, fill="#fff", stroke=col, sw=1.1),
                        txt(bx + 13, y + 17, lab, cls="lbl", size=10.4),
                        txt(bx + 241, y + 27, f"{val:,.2f} {unit}", cls="num",
                            anchor="end", size=13, fill=col)], s=st))
        y += 52
    b.append(g([rect(0, y, 1000, 92, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, y + 22, "算總帳（單卡 B200、gpu_memory_utilization 0.90）：",
                    cls="lbl b", size=12, fill=C["mem"]),
                txt(14, y + 44, f"122B：起草器 {m35.spec_draft_params*2/1e9:,.2f} GB 是固定支出，"
                                f"相當於少放 "
                                f"{m35.spec_draft_params*2/(m35.seq_bytes(8192,'fp8',3)):,.0f} 條 8k 序列。",
                    cls="lbl", size=11, fill=C["mem"]),
                txt(14, y + 62, f"27B：起草器 {m38.spec_draft_params*2/1e9:,.2f} GB，"
                                f"每條序列的 state 也從 {m38.ssm_bytes(0)/M.MiB:,.1f} MiB "
                                f"漲到 {m38.ssm_bytes(7)/M.MiB:,.1f} MiB（+"
                                f"{(m38.ssm_bytes(7)/m38.ssm_bytes(0)-1)*100:,.1f}%）。",
                    cls="lbl", size=11, fill=C["mem"]),
                txt(14, y + 80, "記憶體不是 spec decoding 的主要限制。但 cache 本來就很緊的長 context 場景，"
                                "這幾 GB 會直接吃掉併發數。",
                    cls="lbl b", size=11, fill=C["mem"])], s=4))
    return svg(W, y + 106, b)
