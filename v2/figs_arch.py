# -*- coding: utf-8 -*-
"""CH2–CH4 的圖：vLLM 堆疊、排程、架構堆疊、MoE、Gated DeltaNet。"""
from svgkit import *
import model as M

W = 1000


# ------------------------------------------------------------- vLLM 堆疊 ---
def fig_stack():
    b = []
    rows = [
        ("API server / frontend", "HTTP、chat template、tokenize、SSE 串流回傳",
         "OpenAI 相容端點；request 在這裡被指派 id", C["neutral"], "#f4f6f8", 1),
        ("EngineCore 排程器", "waiting / running 佇列、每輪 token budget、cache 配置",
         "決定「這一輪誰上車、各帶幾個 token」", C["compute"], C["computeW"], 2),
        ("KV cache manager", "block table、free block pool、prefix cache 命中判斷",
         "混合模型還要同時管 SSM state pool", C["mem"], C["memW"], 3),
        ("Executor / Worker", "分散式執行、TP / EP / DP 的集合通訊",
         "單卡時只有一個 worker", C["neutral"], "#f4f6f8", 4),
        ("Model runner", "組 batch metadata、CUDA graph 重播、呼叫模型",
         "把「本輪要算什麼」變成具體張量", C["moe"], C["moeW"], 5),
        ("Model / backend kernels", "attention · GDN · MoE · GEMM · sampling",
         "量化 kernel 就掛在這一層", C["ssm"], C["ssmW"], 6),
    ]
    y, hh = 22, 46
    for i, (t, s2, s3, col, tint, st) in enumerate(rows):
        b.append(g([rect(0, y, 700, hh, r=7, fill=tint, stroke=col, sw=1.1),
                    txt(16, y + 20, t, cls="lbl b", size=12.5, fill=col),
                    txt(16, y + 36, s2, cls="lbl s", size=10.4),
                    txt(716, y + 20, s3, cls="lbl", size=10.6, fill=C["mut"])], s=st))
        if i:
            b.append(arrow(350, y - 8, 350, y - 1, s=st))
        y += hh + 8
    # 資料流動的小圓點
    dpath = "M350,22 L350,%d" % (y - 8)
    b.append(g([flow_dot(dpath, C["compute"], 4, "2.4s"),
                flow_dot(dpath, C["compute"], 4, "2.4s", "0.8s"),
                flow_dot(dpath, C["compute"], 4, "2.4s", "1.6s")], s=6))
    b.append(g([rect(0, y, 700, 34, fill=C["okW"], stroke="#cbe9d5"),
                txt(16, y + 22, "每個 step 都完整走一次：schedule → forward → postprocess",
                    cls="lbl b", size=11.5, fill="#14532d")], s=6))
    return svg(W, y + 46, b)


# -------------------------------------------------- continuous batching ---
def fig_batching():
    b = []
    x0, y0, cw, rh = 118, 26, 46, 26
    steps = 12
    b.append(txt(0, y0 - 8, "靜態 batching：整批一起開始、一起結束", cls="lbl b", size=11.5,
                 fill=C["bad"]))
    lens = [11, 4, 7, 3]
    for r, L in enumerate(lens):
        y = y0 + r * (rh + 5)
        b.append(txt(0, y + 17, f"req {r+1}", cls="lbl s"))
        for i in range(steps):
            fill = C["computeW"] if i < L else C["badW"]
            stroke = C["compute"] if i < L else "#f5cdc8"
            b.append(rect(x0 + i * (cw + 3), y, cw, rh, r=3, fill=fill, stroke=stroke, sw=.9))
        if L < steps:
            b.append(txt(x0 + (L + (steps - L) / 2) * (cw + 3), y + 17,
                         "GPU 空轉", cls="lbl xs", anchor="middle", fill=C["bad"], ds="1"))
    yb = y0 + 4 * (rh + 5) + 16
    b.append(txt(0, yb, "continuous batching：每一輪重新組隊", cls="lbl b", size=11.5,
                 fill=C["ok"], ds="2"))
    yb += 12
    plan = [  # (row, start, len, label)
        (0, 0, 11, "req 1"), (1, 0, 4, "req 2"), (2, 0, 7, "req 3"), (3, 0, 3, "req 4"),
        (1, 4, 8, "req 5"), (3, 3, 6, "req 6"), (2, 7, 5, "req 7"), (3, 9, 3, "req 8"),
    ]
    cols = {"req 1": C["compute"], "req 2": C["moe"], "req 3": C["ssm"], "req 4": C["mem"],
            "req 5": C["spec"], "req 6": C["ok"], "req 7": C["neutral"], "req 8": C["compute"]}
    for r in range(4):
        b.append(txt(0, yb + r * (rh + 5) + 17, f"slot {r+1}", cls="lbl s", ds="2"))
    for r, st, L, lab in plan:
        y = yb + r * (rh + 5)
        col = cols[lab]
        step = 2 if st == 0 else 3
        b.append(g([rect(x0 + st * (cw + 3), y, L * (cw + 3) - 3, rh, r=3,
                         fill="#fff", stroke=col, sw=1.2),
                    rect(x0 + st * (cw + 3), y, 3, rh, r=0, fill=col, stroke=None),
                    txt(x0 + st * (cw + 3) + L * (cw + 3) / 2, y + 17, lab, cls="lbl",
                        anchor="middle", size=10.5, fill=col)], s=step))
    yy = yb + 4 * (rh + 5) + 8
    for i in range(steps):
        b.append(txt(x0 + i * (cw + 3) + cw / 2, yy + 10, f"{i+1}", cls="num",
                     anchor="middle", size=9))
    b.append(txt(x0 - 8, yy + 10, "step", cls="lbl s", anchor="end"))
    b.append(g([rect(0, yy + 22, 980, 34, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, yy + 44, "空出來的位置立刻被新 request 補上。"
                                 "同一次權重讀取被更多 token 分攤，就是 CH1 講的那件事。",
                    cls="lbl b", size=11.5, fill="#14532d")], s=4))
    return svg(W, yy + 66, b)


# ------------------------------------------------------- PagedAttention ---
def fig_paged():
    b = []
    b.append(txt(0, 14, "邏輯上：每個 request 是一條連續的 token 序列", cls="lbl s"))
    seqs = [("req A", C["compute"], C["computeW"], ["The", "future", "of", "AI", "is"]),
            ("req B", C["moe"], C["moeW"], ["Paged", "memory", "avoids"]),
            ("req C", C["ssm"], C["ssmW"], ["Blocks", "map"])]
    y = 26
    cw = 74
    lay = []
    for name, col, tint, toks in seqs:
        b.append(txt(0, y + 19, name, cls="lbl b", size=11, fill=col))
        for i, t in enumerate(toks):
            b.append(cell(64 + i * (cw + 4), y, cw, 28, t, fill=tint, stroke=col,
                          color=col, size=10.5))
            lay.append((name, col, i, t))
        y += 34
    b.append(txt(0, y + 12, "實體上：切成固定大小的 block，散落在 HBM 各處", cls="lbl s", ds="1"))
    # physical pool
    py = y + 22
    pool = [("A0", 0), ("C0", 2), ("free", -1), ("B0", 1), ("A1", 0), ("free", -1),
            ("B1", 1), ("free", -1), ("A2", 0), ("free", -1), ("C1", 2), ("free", -1)]
    colmap = [C["compute"], C["moe"], C["ssm"]]
    tintmap = [C["computeW"], C["moeW"], C["ssmW"]]
    bw = 74
    for i, (lab, owner) in enumerate(pool):
        x = i * (bw + 4)
        if owner < 0:
            b.append(cell(x, py, bw, 30, "free", fill="#fff", stroke=C["line"],
                          color=C["mut2"], size=10, s=1))
        else:
            b.append(cell(x, py, bw, 30, lab, fill=tintmap[owner], stroke=colmap[owner],
                          color=colmap[owner], size=10.5, s=1, weight=650))
    b.append(txt(0, py + 48, "block table 記住「邏輯第幾塊 → 實體第幾塊」", cls="lbl s", ds="2"))
    ty = py + 58
    b.append(g([rect(0, ty, 470, 74, fill=C["card2"]),
                txt(14, ty + 20, "req A → [0, 4, 8]", cls="num", size=11.5, fill=C["compute"]),
                txt(14, ty + 40, "req B → [3, 6]", cls="num", size=11.5, fill=C["moe"]),
                txt(14, ty + 60, "req C → [1, 10]", cls="num", size=11.5, fill=C["ssm"])], s=2))
    b.append(g([rect(486, ty, 514, 74, fill=C["okW"], stroke="#cbe9d5"),
                txt(500, ty + 20, "得到三件事：", cls="lbl b", size=11.5, fill="#14532d"),
                txt(500, ty + 38, "① 幾乎沒有碎片，只浪費最後一塊的尾巴",
                    cls="lbl", size=11, fill="#14532d"),
                txt(500, ty + 55, "② 相同前綴可以共用同一塊（prefix caching / 分支）",
                    cls="lbl", size=11, fill="#14532d"),
                txt(500, ty + 70, "③ request 可以被 preempt 換出，之後再換回來",
                    cls="lbl", size=11, fill="#14532d")], s=3))
    return svg(W, ty + 90, b)


# ------------------------------------------------------------- scheduler ---
def fig_scheduler():
    b = []
    budget = 8192
    b.append(txt(0, 16, f"本輪 token budget = max_num_batched_tokens = {budget:,}",
                 cls="lbl b", size=12.5, fill=C["compute"]))
    x0, y0, wtot, hh = 0, 28, 1000, 40
    b.append(rect(x0, y0, wtot, hh, r=6, fill=C["card2"]))
    segs = [("32 個 decode request × 1 token", 32, C["mem"], C["memW"], 1),
            ("1 個新 request 的 prefill chunk", 4096, C["compute"], C["computeW"], 2),
            ("另一個 prefill 的續傳 chunk", 2048, C["compute"], "#dce8fd", 3),
            ("剩餘 budget（讓給下一輪）", budget - 32 - 4096 - 2048, C["neutral"], "#f1f3f5", 4)]
    x = x0
    for lab, v, col, tint, st in segs:
        wd = v / budget * wtot
        b.append(g([rect(x, y0, wd, hh, r=4, fill=tint, stroke=col, sw=1),
                    txt(x + wd / 2, y0 + 18, f"{v:,}", cls="num", anchor="middle",
                        size=11, fill=col),
                    txt(x + wd / 2, y0 + 32, "token", cls="lbl xs", anchor="middle")], s=st))
        lx = min(max(x + wd / 2, 4), 996)
        anc = "start" if lx < 90 else ("end" if lx > 910 else "middle")
        b.append(txt(4 if anc == "start" else (996 if anc == "end" else lx),
                     y0 + hh + 16 + (14 if wd < 120 else 0), lab, cls="lbl xs",
                     anchor=anc, fill=col, ds=st, size=9.6))
        if wd < 120:
            b.append(line(lx, y0 + hh + 2, lx, y0 + hh + 12, stroke=col, sw=.9, s=st))
        x += wd
    # 佇列
    qy = y0 + hh + 40
    b.append(txt(0, qy, "排程器同時看兩個佇列與兩種資源", cls="lbl s"))
    boxes = [("running 佇列", "已在生成中；優先保住它們的 decode", C["mem"], C["memW"], 1),
             ("waiting 佇列", "還沒開始；FCFS 或 priority", C["neutral"], "#f4f6f8", 1),
             ("KV / SSM cache 餘量", "配不到 slot 就得 preempt 別人", C["ssm"], C["ssmW"], 2),
             ("token budget", "本輪算力與頻寬的預算上限", C["compute"], C["computeW"], 2)]
    bx = 0
    for t, s2, col, tint, st in boxes:
        b.append(g([rect(bx, qy + 10, 238, 54, r=7, fill=tint, stroke=col, sw=1.1),
                    txt(bx + 14, qy + 32, t, cls="lbl b", size=11.5, fill=col),
                    txt(bx + 14, qy + 50, s2, cls="lbl s", size=10)], s=st))
        bx += 254
    b.append(g([rect(0, qy + 76, 1000, 52, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, qy + 96, "決策順序：① 先排 running 的 decode（保 TPOT）"
                                 "② 用剩下的 budget 排 prefill chunk（追 TTFT）",
                    cls="lbl b", size=11.5, fill=C["mem"]),
                txt(14, qy + 116, "③ 若 cache 不足，從尾端 preempt request 換出 "
                                  "④ 產生 batch metadata，交給 model runner",
                    cls="lbl", size=11, fill=C["mem"])], s=4))
    return svg(W, qy + 142, b)


# ---------------------------------------------------- 架構堆疊（三個模型）--
def _stack_col(x, y, m, s=None, cellw=150, ch=17, gap=2.5, maxrows=64, scale=1.0):
    """畫出一個模型的層堆疊（由下往上）。"""
    out = []
    L = m.n_layers
    ch = ch * scale
    for i in range(L):
        is_full = (m.n_gdn == 0) or ((i + 1) % 4 == 0)
        col = C["compute"] if is_full else C["ssm"]
        tint = C["computeW"] if is_full else C["ssmW"]
        yy = y + (L - 1 - i) * (ch + gap)
        out.append(rect(x, yy, cellw, ch, r=2.5, fill=tint, stroke=col, sw=.8))
        if m.n_experts:
            out.append(rect(x + cellw + 3, yy, 26, ch, r=2.5, fill=C["moeW"],
                            stroke=C["moe"], sw=.8))
        else:
            out.append(rect(x + cellw + 3, yy, 26, ch, r=2.5, fill="#f4f6f8",
                            stroke=C["neutral"], sw=.8))
    h = L * (ch + gap) - gap
    return out, h


def fig_arch3():
    b = []
    models = [M.Q332, M.Q38, M.Q35]
    x0, ytop = 214, 34
    wid = 646
    rowh = 92
    for mi, m in enumerate(models):
        y = ytop + mi * rowh
        b.append(txt(0, y + 16, m.label, cls="lbl b", size=12.5))
        b.append(txt(0, y + 33, f"{m.n_layers} 層 · hidden {m.hidden:,} · "
                                f"{'MoE' if m.n_experts else 'Dense FFN'}", cls="lbl s"))
        b.append(txt(0, y + 49, f"{m.quant} · {m.real_weight_bytes/1e9:,.1f} GB",
                     cls="lbl s", size=10, fill=C["mem"]))
        cw = wid / m.n_layers
        for i in range(m.n_layers):
            is_full = (m.n_gdn == 0) or ((i + 1) % 4 == 0)
            col = C["compute"] if is_full else C["ssm"]
            tint = C["computeW"] if is_full else C["ssmW"]
            b.append(rect(x0 + i * cw, y, cw - 1.4, 30, r=2, fill=tint, stroke=col, sw=.7))
            ff = C["moeW"] if m.n_experts else "#f4f6f8"
            fs = C["moe"] if m.n_experts else C["neutral"]
            b.append(rect(x0 + i * cw, y + 33, cw - 1.4, 14, r=2, fill=ff, stroke=fs,
                          sw=.6, s=1))
        b.append(txt(x0 - 8, y + 20, "mixer", cls="lbl xs", anchor="end"))
        b.append(txt(x0 - 8, y + 44, "FFN", cls="lbl xs", anchor="end", ds="1"))
        b.append(txt(x0 + wid + 10, y + 20,
                     f"KV {m.kv_bytes_per_token('bf16')/1024:,.0f} KiB/tok",
                     cls="lbl b", size=10.6, fill=C["mem"], ds="2"))
        b.append(txt(x0 + wid + 10, y + 40,
                     (f"state {m.ssm_bytes(0)/M.MiB:,.0f} MiB/序列" if m.n_gdn else "無 SSM state"),
                     cls="lbl", size=10.6, fill=C["ssm"], ds="2"))
        b.append(txt(x0, y + 62, f"full attention {m.n_full} 層", cls="lbl", size=10,
                     fill=C["compute"]))
        if m.n_gdn:
            b.append(txt(x0 + 132, y + 62, f"Gated DeltaNet {m.n_gdn} 層", cls="lbl",
                         size=10, fill=C["ssm"]))
    y2 = ytop + 3 * rowh + 2
    b.append(g([rect(0, y2, 1000, 30, fill=C["card2"]),
                txt(14, y2 + 20, "顏色：", cls="lbl s"),
                rect(58, y2 + 8, 13, 13, r=2, fill=C["computeW"], stroke=C["compute"]),
                txt(76, y2 + 19, "full attention（要 KV cache）", cls="lbl", size=11),
                rect(268, y2 + 8, 13, 13, r=2, fill=C["ssmW"], stroke=C["ssm"]),
                txt(286, y2 + 19, "Gated DeltaNet（固定大小 state）", cls="lbl", size=11),
                rect(516, y2 + 8, 13, 13, r=2, fill=C["moeW"], stroke=C["moe"]),
                txt(534, y2 + 19, "MoE FFN", cls="lbl", size=11),
                rect(624, y2 + 8, 13, 13, r=2, fill="#f4f6f8", stroke=C["neutral"]),
                txt(642, y2 + 19, "Dense FFN", cls="lbl", size=11)], s=1))
    return svg(W, y2 + 40, b)


# --------------------------------------------------------------- MoE 路由 --
def fig_moe():
    m = M.Q35
    b = []
    b.append(txt(0, 16, "一個 token 進到 MoE 層", cls="lbl s"))
    b.append(node(0, 26, 130, 38, "hidden xₜ", "[3072]", color=C["neutral"], tint="#f4f6f8"))
    b.append(arrow(134, 45, 176, 45))
    b.append(node(180, 26, 130, 38, "router", "gate.weight", color=C["moe"],
                  tint=C["moeW"], s=1))
    b.append(arrow(314, 45, 356, 45, s=1))
    # experts grid
    ex, ey = 380, 20
    cols_, rows_ = 16, 16
    cw, chh, gp = 24, 11.5, 2.2
    import random
    random.seed(7)
    chosen = sorted(random.sample(range(256), 8))
    for i in range(256):
        r, c = divmod(i, cols_)
        x = ex + c * (cw + gp)
        y = ey + r * (chh + gp)
        if i in chosen:
            b.append(rect(x, y, cw, chh, r=1.6, fill=C["moe"], stroke=None, s=2))
        else:
            b.append(rect(x, y, cw, chh, r=1.6, fill="#eceff3", stroke=None, s=1))
    gh = rows_ * (chh + gp)
    b.append(txt(ex + cols_ * (cw + gp) + 16, ey + 20,
                 f"256 個 routed expert", cls="lbl b", size=12, fill=C["moe"]))
    b.append(txt(ex + cols_ * (cw + gp) + 16, ey + 40,
                 f"每個 {m.expert_params/1e6:,.1f} M 參數", cls="lbl s"))
    b.append(txt(ex + cols_ * (cw + gp) + 16, ey + 66, "本 token 只啟用其中 8 個",
                 cls="lbl b", size=11.5, fill=C["moe"], ds="2"))
    b.append(txt(ex + cols_ * (cw + gp) + 16, ey + 84, "（深紫色）＋ 1 個 shared",
                 cls="lbl", size=11, fill=C["moe"], ds="2"))
    y3 = ey + gh + 24
    b.append(g([rect(0, y3, 486, 92, fill=C["computeW"], stroke="#d5e4fd"),
                txt(14, y3 + 22, "算力：只算 8/256", cls="lbl b", size=12, fill=C["compute"]),
                txt(14, y3 + 42, f"每 token 啟用 {m.activated/1e9:,.2f} B 參數",
                    cls="num", size=11),
                txt(14, y3 + 60, f"→ 2 × {m.activated/1e9:,.1f}B = "
                                 f"{2*m.activated/1e9:,.0f} GFLOP / token", cls="num", size=11),
                txt(14, y3 + 80, "型號裡的 A10B 講的就是這個", cls="lbl", size=11,
                    fill=C["compute"])], s=3))
    b.append(g([rect(506, y3, 494, 92, fill=C["memW"], stroke="#f0dcb4"),
                txt(520, y3 + 22, "記憶體：整組 256 個都得放在 HBM", cls="lbl b", size=12,
                    fill=C["mem"]),
                txt(520, y3 + 42, f"{m.n_layers} 層 × 256 expert × "
                                  f"{m.expert_params/1e6:,.1f} M = "
                                  f"{m.n_layers*256*m.expert_params/1e9:,.1f} B 參數",
                    cls="num", size=11),
                txt(520, y3 + 60, f"NVFP4 之下 = "
                                  f"{m.n_layers*256*m.expert_bytes_each/1e9:,.1f} GB",
                    cls="num", size=11),
                txt(520, y3 + 80, "→ 佔了整個 checkpoint 的 69%", cls="lbl", size=11,
                    fill=C["mem"])], s=4))
    b.append(g([rect(0, y3 + 102, 1000, 34, fill=C["badW"], stroke="#f5cdc8"),
                txt(14, y3 + 124, "而且 batch 一大，被選到的 expert 就會涵蓋幾乎全部 256 個，"
                                  "記憶體流量從 2 GB 長到 65 GB。CH7 會用到這件事。",
                    cls="lbl b", size=11.5, fill=C["bad"])], s=5))
    return svg(W, y3 + 148, b)


# --------------------------------------------------- GDN 的一次 state update
def fig_gdn():
    b = []
    m = M.Q38
    b.append(txt(0, 14, "一個 GDN head 的狀態 S 是固定大小的矩陣（dᵥ × dₖ = 128 × 128）",
                 cls="lbl s"))
    # 狀態方塊
    def mat(x, y, sz, label, col, tint, s=None, cells=8, seed=1):
        import random
        random.seed(seed)
        out = [rect(x, y, sz, sz, r=4, fill=tint, stroke=col, sw=1.2)]
        step = sz / cells
        for i in range(cells):
            for j in range(cells):
                v = random.random()
                out.append(rect(x + j * step + 1, y + i * step + 1, step - 2, step - 2,
                                r=1, fill=col, stroke=None, opacity=round(.12 + v * .68, 2)))
        out.append(txt(x + sz / 2, y + sz + 15, label, cls="lbl b", anchor="middle",
                       size=11.5, fill=col))
        return g(out, s=s)

    y0 = 34
    sz = 108
    b.append(mat(0, y0, sz, "Sₜ₋₁　上一步的狀態", C["ssm"], C["ssmW"], seed=3))
    b.append(txt(sz + 22, y0 + 40, "×", cls="lbl b", size=20, fill=C["mut2"], ds="1"))
    b.append(g([rect(sz + 44, y0 + 12, 176, 56, r=6, fill=C["memW"], stroke=C["mem"], sw=1.1),
                txt(sz + 132, y0 + 34, "αₜ (I − βₜ kₜkₜᵀ)", cls="num", anchor="middle",
                    size=13, fill=C["mem"]),
                txt(sz + 132, y0 + 54, "遺忘 + 定點清除", cls="lbl xs", anchor="middle")], s=1))
    b.append(txt(sz + 234, y0 + 40, "+", cls="lbl b", size=20, fill=C["mut2"], ds="2"))
    b.append(g([rect(sz + 256, y0 + 12, 150, 56, r=6, fill=C["okW"], stroke=C["ok"], sw=1.1),
                txt(sz + 331, y0 + 34, "βₜ vₜ kₜᵀ", cls="num", anchor="middle",
                    size=13, fill=C["ok"]),
                txt(sz + 331, y0 + 54, "寫入新的關聯", cls="lbl xs", anchor="middle")], s=2))
    b.append(txt(sz + 420, y0 + 40, "=", cls="lbl b", size=20, fill=C["mut2"], ds="3"))
    b.append(mat(sz + 444, y0, sz, "Sₜ　新的狀態（大小完全沒變）", C["ssm"], C["ssmW"],
                 s=3, seed=9))
    b.append(txt(sz + 580, y0 + 40, "→", cls="lbl b", size=20, fill=C["mut2"], ds="4"))
    b.append(g([rect(sz + 606, y0 + 12, 150, 56, r=6, fill=C["computeW"],
                     stroke=C["compute"], sw=1.1),
                txt(sz + 681, y0 + 34, "oₜ = Sₜ qₜ", cls="num", anchor="middle",
                    size=13, fill=C["compute"]),
                txt(sz + 681, y0 + 54, "讀出這個位置的輸出", cls="lbl xs", anchor="middle")],
               s=4))
    yy = y0 + sz + 36
    rows = [("αₜ ∈ (0,1)", "decay gate：由資料決定要忘掉多少。α→0 等於整張狀態清空。",
             C["mem"], 1),
            ("βₜ ∈ (0,1)", "delta rule 強度：β→1 表示把 kₜ 這個「鍵」原本存的東西整條換掉。",
             C["ok"], 2),
            ("S 的大小", f"{m.dv} × {m.dk} × 4 bytes = "
                        f"{m.dv*m.dk*4/1024:,.0f} KiB / head，"
                        f"和 context 長度完全無關。", C["ssm"], 3),
            ("代價", "S 是有損壓縮。context 越長，被擠掉的細節越多，"
                    "所以每 4 層要放一層 full attention 來補。", C["bad"], 5)]
    for i, (k, v, col, st) in enumerate(rows):
        b.append(g([rect(0, yy + i * 30, 1000, 26, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, yy + i * 30, 3, 26, r=0, fill=col, stroke=None),
                    txt(13, yy + i * 30 + 17, k, cls="num", size=11, fill=col),
                    txt(126, yy + i * 30 + 17, v, cls="lbl", size=11.2)], s=st))
    return svg(W, yy + 4 * 30 + 6, b)


# ------------------------------------------------------ hybrid 一次 decode --
def fig_hybrid(key="q35"):
    m = M.MODELS[key]
    b = []
    b.append(txt(0, 14, f"{m.label}：{m.n_layers} 層 = "
                        f"{m.n_layers//4} × (3 × GDN + 1 × full attention)", cls="lbl b",
                 size=12.5))
    x0, y0 = 24, 30
    group = 4
    ng = m.n_layers // group
    avail = (960 - x0) / ng
    gp = 2.5
    cw = avail / group - gp - 2.2
    chh = 34
    perg = group * (cw + gp) + (avail - group * (cw + gp))
    for gi in range(ng):
        gx = x0 + gi * perg
        for li in range(group):
            is_full = li == group - 1
            col = C["compute"] if is_full else C["ssm"]
            tint = C["computeW"] if is_full else C["ssmW"]
            b.append(rect(gx + li * (cw + gp), y0, cw, chh, r=3, fill=tint, stroke=col, sw=1))
        b.append(txt(gx + group * (cw + gp) / 2 - gp / 2, y0 + chh + 13, f"{gi+1}",
                     cls="num", anchor="middle", size=8.4))
    tot = ng * avail
    b.append(txt(x0, y0 + chh + 30, "資料由左往右穿過每一層", cls="lbl s", ds="1"))
    dpath = f"M{x0},{y0+chh/2} L{x0+tot-10},{y0+chh/2}"
    b.append(g([flow_dot(dpath, C["moe"], 5, "3s"),
                flow_dot(dpath, C["moe"], 5, "3s", "1s"),
                flow_dot(dpath, C["moe"], 5, "3s", "2s")], s=1))
    # 兩種 cache 的更新
    yb = y0 + chh + 46
    b.append(g([rect(0, yb, 486, 108, fill=C["ssmW"], stroke=C["ssm"]),
                txt(14, yb + 22, f"{m.n_gdn} 層 GDN：更新 recurrent state",
                    cls="lbl b", size=12, fill=C["ssm"]),
                txt(14, yb + 44, f"每層 {m.nv} 個 value head × {m.dv}×{m.dk} FP32",
                    cls="num", size=11),
                txt(14, yb + 62, f"= {m.ssm_recurrent_bytes()/M.MiB:,.0f} MiB / 序列（固定）",
                    cls="num", size=11, fill=C["ssm"]),
                txt(14, yb + 84, "讀舊 state → 就地寫回新 state，不會變大",
                    cls="lbl", size=11, fill=C["ssm"])], s=2))
    b.append(g([rect(506, yb, 494, 108, fill=C["memW"], stroke=C["mem"]),
                txt(520, yb + 22, f"{m.n_full} 層 full attention：append KV",
                    cls="lbl b", size=12, fill=C["mem"]),
                txt(520, yb + 44, f"每 token 每層 2 × {m.n_kv} × {m.head_dim}",
                    cls="num", size=11),
                txt(520, yb + 62, f"= {m.kv_bytes_per_token(m.kv_dtype)/1024:,.0f} KiB / token"
                                  f"（{m.kv_dtype.upper()}），會一直長大",
                    cls="num", size=11, fill=C["mem"]),
                txt(520, yb + 84, "讀全部歷史 K/V → 尾端多寫一格",
                    cls="lbl", size=11, fill=C["mem"])], s=3))
    yc = yb + 120
    b.append(g([rect(0, yc, 1000, 34, fill=C["okW"], stroke="#cbe9d5"),
                txt(14, yc + 22, "兩種 cache 得「同步前進」：任何一邊被換出或算錯位置，"
                                 "整條序列後面的輸出就全壞了。hybrid 模型 serving 最難的地方就在這。",
                    cls="lbl b", size=11.5, fill="#14532d")], s=4))
    return svg(W, yc + 48, b)


# ------------------------------------------------------- chunkwise prefill --
def fig_chunkwise():
    b = []
    b.append(txt(0, 14, "GDN 的 prefill 不能一個 token 一個 token 遞迴（太慢），"
                        "改用 chunkwise 平行", cls="lbl s"))
    x0, y0, cw, chh = 0, 28, 52, 30
    nch, per = 4, 4
    for ci in range(nch):
        gx = x0 + ci * (per * (cw + 4) + 28)
        b.append(rect(gx - 5, y0 - 5, per * (cw + 4) + 2, chh + 10, r=6, fill=C["ssmW"],
                      stroke=C["ssm"], sw=1, s=1))
        for i in range(per):
            b.append(cell(gx + i * (cw + 4), y0, cw, chh, f"t{ci*per+i+1}",
                          fill="#fff", stroke=C["line"], size=10))
        b.append(txt(gx + (per * (cw + 4)) / 2 - 2, y0 + chh + 24, f"chunk {ci+1}",
                     cls="lbl b", anchor="middle", size=10.5, fill=C["ssm"], ds="1"))
        if ci:
            b.append(arrow(gx - 24, y0 + chh / 2, gx - 8, y0 + chh / 2, color=C["ssm"], s=2))
            b.append(txt(gx - 16, y0 - 14, "S", cls="num", anchor="middle", size=10,
                         fill=C["ssm"], ds="2"))
    yy = y0 + chh + 44
    rows = [("chunk 內", "用矩陣乘法平行處理，塞滿 Tensor Core（WY representation）",
             C["compute"], 1),
            ("chunk 之間", "只傳遞一個固定大小的狀態 S，序列依賴只剩這一條", C["ssm"], 2),
            ("full attention 層", "同一時間照常做 FlashAttention，並把 prompt 的 K/V 寫進 cache",
             C["mem"], 3),
            ("Prefix caching", "vLLM 對 mamba / GDN 的 prefix caching 仍在開發中；"
                               "混合模型的命中長度取兩種 group 的交集", C["bad"], 4)]
    for i, (k, v, col, st) in enumerate(rows):
        b.append(g([rect(0, yy + i * 32, 1000, 28, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, yy + i * 32, 3, 28, r=0, fill=col, stroke=None),
                    txt(13, yy + i * 32 + 18, k, cls="lbl b", size=11.2, fill=col),
                    txt(146, yy + i * 32 + 18, v, cls="lbl", size=11.2)], s=st))
    return svg(W, yy + 4 * 32 + 6, b)


# ------------------------------------------------- 兩種 cache 的成長曲線 ----
def fig_two_caches():
    import math
    b = []
    x0, y0, w, h = 74, 268, 620, 216
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 4, y0 - h - 14, "↑ 每條序列佔用的記憶體（MiB，對數）", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 34, "context 長度 T（對數）→", cls="lbl s", anchor="end"))
    lo_t, hi_t = 256, 262144
    lo_v, hi_v = 60.0, 24000.0

    def X(t):
        return x0 + math.log10(t / lo_t) / math.log10(hi_t / lo_t) * w

    def Y(v):
        return y0 - math.log10(max(v, lo_v) / lo_v) / math.log10(hi_v / lo_v) * h

    series = [(M.Q332, C["neutral"], "Qwen3-32B（純 Transformer, BF16 KV）", "bf16", 0, 1),
              (M.Q38, C["compute"], "Qwen3.8-27B（BF16 KV）", "bf16", 0, 2),
              (M.Q35, C["mem"], "Qwen3.5-122B-A10B（FP8 KV）", "fp8", 0, 2)]
    for _i, (m, col, lab, kvd, _, st) in enumerate(series):
        pts = []
        for e in range(0, 41):
            t = lo_t * (hi_t / lo_t) ** (e / 40)
            if t > m.max_ctx:
                break
            v = m.seq_bytes(t, kvd, m.spec_k) / M.MiB
            pts.append(f"{X(t):.1f},{Y(v):.1f}")
        b.append(path("M" + " L".join(pts), stroke=col, sw=2.6, s=st))
        tt = min(hi_t, m.max_ctx)
        b.append(txt(X(tt) - 6, Y(m.seq_bytes(tt, kvd, m.spec_k) / M.MiB) - 9 - 14 * _i,
                     lab, cls="lbl b", anchor="end", size=10.4, fill=col, ds=st))
    # 固定的 SSM 水平線
    for m, col, st in ((M.Q38, C["ssm"], 3), (M.Q35, C["ssm"], 3)):
        v = m.ssm_bytes(m.spec_k) / M.MiB
        b.append(line(x0, Y(v), x0 + w, Y(v), stroke=col, sw=1.6, dash="6 5", s=st))
    b.append(txt(x0 + 10, Y(M.Q35.ssm_bytes(M.Q35.spec_k) / M.MiB) + 18,
                 "兩個混合模型的 SSM state 都是 ~150 MiB，而且是水平線（不隨 context 成長）",
                 cls="lbl b", size=10.6, fill=C["ssm"], ds="3"))
    # 交會點
    for m, kvd, col, st in ((M.Q35, "fp8", C["mem"], 4), (M.Q38, "bf16", C["compute"], 4)):
        t = m.crossover_tokens(kvd, m.spec_k)
        v = m.seq_bytes(t, kvd, m.spec_k) / M.MiB
        b.append(circle(X(t), Y(v), 5, fill=col, stroke="#fff", sw=1.8, s=st))
        b.append(txt(X(t), Y(v) - 12, f"{t:,.0f} tok", cls="num", anchor="middle",
                     size=10, fill=col, ds=st))
    for v in (100, 300, 1000, 3000, 10000):
        b.append(line(x0 - 3, Y(v), x0, Y(v), stroke=C["line"], sw=1))
        b.append(txt(x0 - 7, Y(v) + 4,
                     f"{v//1000} GiB" if v >= 1000 else f"{v}", cls="num", anchor="end"))
    for t in (256, 1024, 4096, 16384, 65536, 262144):
        b.append(line(X(t), y0, X(t), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(t), y0 + 18, f"{t//1024}k" if t >= 1024 else str(t), cls="num",
                     anchor="middle"))
    b.append(g([rect(716, 40, 284, 190, fill=C["card2"]),
                txt(730, 62, "交會點的意義", cls="lbl b", size=12),
                txt(730, 84, "context 短的時候，混合模型的", cls="lbl", size=11),
                txt(730, 100, "記憶體幾乎全部是固定的 SSM", cls="lbl", size=11),
                txt(730, 116, "state；長 context 之後才換成", cls="lbl", size=11),
                txt(730, 132, "KV cache 主導。", cls="lbl", size=11),
                line(730, 144, 986, 144, stroke=C["line"], sw=1),
                txt(730, 164, f"122B：{M.Q35.crossover_tokens('fp8', 3):,.0f} token",
                    cls="num", size=11.5, fill=C["mem"]),
                txt(730, 182, f"27B：{M.Q38.crossover_tokens('bf16', 7):,.0f} token",
                    cls="num", size=11.5, fill=C["compute"]),
                txt(730, 204, "→ 短對話的併發上限其實卡在", cls="lbl", size=10.6),
                txt(730, 219, "　 SSM state，不在 KV。", cls="lbl b", size=10.6,
                    fill=C["ssm"])], s=4))
    return svg(W, y0 + 44, b)
