# -*- coding: utf-8 -*-
"""補充圖：tokenizer 實例、causal mask、lm_head、mixer 分工、三種記憶、高併發。"""
import json
from pathlib import Path
from svgkit import *
import model as M

W = 1000
DEMO = json.loads((Path(__file__).resolve().parent.parent / "research"
                   / "tokenize_demo.json").read_text(encoding="utf-8"))


# ------------------------------------------------ 1. 真實的 tokenizer 例子 --
def fig_tokenize_real():
    b = []
    zh = DEMO["zh"]
    b.append(txt(0, 14, "① 使用者輸入", cls="lbl s"))
    b.append(g([rect(0, 22, 420, 36, fill=C["card2"]),
                txt(14, 46, zh["text"], cls="lbl b", size=15)]))
    b.append(txt(432, 46, f"{zh['n_chars']} 個中文字", cls="lbl s"))

    b.append(txt(0, 88, "② tokenizer 切成 token，再查表換成整數 id", cls="lbl s", ds="1"))
    x, y = 0, 96
    for t in zh["tokens"]:
        w = 20 + len(t["s"]) * 15
        b.append(cell(x, y, w, 32, t["s"], fill=C["computeW"], stroke=C["compute"],
                      color=C["compute"], s=1, size=13, weight=650))
        b.append(txt(x + w / 2, y + 48, str(t["id"]), cls="num", anchor="middle", ds="1"))
        x += w + 6
    b.append(txt(x + 10, y + 21, f"→ {zh['n']} 個 token", cls="lbl b", size=12,
                 fill=C["compute"], ds="1"))
    b.append(g([rect(0, 158, 1000, 26, r=5, fill=C["memW"], stroke="#f0dcb4"),
                txt(12, 175, "注意「修正」被切成「修」+「正是」。BPE 只看統計上常一起出現的位元組，"
                             "不管詞的邊界。", cls="lbl", size=11, fill=C["mem"])], s=2))

    y2 = 198
    b.append(txt(0, y2, "③ chat template 再包上角色標記，這才是真正送進模型的東西",
                 cls="lbl s", ds="3"))
    sp = DEMO["special"]
    seq = [("<|im_start|>", sp["<|im_start|>"], C["moe"]), ("user", None, C["neutral"]),
           ("…8 個 token…", None, C["compute"]), ("<|im_end|>", sp["<|im_end|>"], C["moe"]),
           ("<|im_start|>", sp["<|im_start|>"], C["moe"]),
           ("assistant", None, C["neutral"]), ("<think>", None, C["ssm"])]
    x = 0
    for lab, tid, col in seq:
        w = 22 + len(lab) * 7.2
        tint = {C["moe"]: C["moeW"], C["neutral"]: "#f4f6f8",
                C["compute"]: C["computeW"], C["ssm"]: C["ssmW"]}[col]
        b.append(cell(x, y2 + 10, w, 30, lab, fill=tint, stroke=col, color=col,
                      s=3, size=10.5))
        if tid:
            b.append(txt(x + w / 2, y2 + 54, str(tid), cls="num", anchor="middle", ds="3"))
        x += w + 5
    b.append(g([rect(0, y2 + 68, 1000, 46, fill=C["computeW"], stroke="#d5e4fd"),
                txt(12, y2 + 87, "④ 這串 id 的長度 T 才是「context 長度」。"
                                 "13 個字的問句，實際送進去大約 20 個 token。",
                    cls="lbl b", size=11.5, fill=C["compute"]),
                txt(12, y2 + 105, "整份報告裡 KV cache 的公式、記憶體帳本、prefill 成本，"
                                  "分母都是這個 T，不是字數。", cls="lbl", size=11,
                    fill=C["compute"])], s=4))
    return svg(W, y2 + 124, b)


# --------------------------------------------- 2. causal mask 與 KV cache --
def fig_causal():
    b = []
    cw = 44
    x0, y0 = 150, 44
    toks = ["A", "B", "C"]
    b.append(txt(0, 16, "Prefill 完 A B C 之後，causal attention 是一個下三角",
                 cls="lbl b", size=12.5))
    for j, t in enumerate(toks):
        b.append(txt(x0 + j * cw + cw / 2, y0 - 8, f"K_{t}", cls="lbl s", anchor="middle"))
    for i, t in enumerate(toks):
        b.append(txt(x0 - 12, y0 + i * cw + cw / 2 + 4, f"Q_{t}", cls="lbl s", anchor="end"))
        for j in range(3):
            ok = j <= i
            b.append(rect(x0 + j * cw, y0 + i * cw, cw - 2, cw - 2, r=4,
                          fill=C["computeW"] if ok else "#fff",
                          stroke=C["compute"] if ok else C["line2"], sw=1))
            if ok:
                b.append(txt(x0 + j * cw + cw / 2 - 1, y0 + i * cw + cw / 2 + 4, "✓",
                             cls="lbl b", anchor="middle", fill=C["compute"], size=13))
    b.append(txt(x0 + 3 * cw + 22, y0 + 20, "A 只看得到 A", cls="lbl", size=11.5))
    b.append(txt(x0 + 3 * cw + 22, y0 + 20 + cw, "B 看得到 A、B", cls="lbl", size=11.5))
    b.append(txt(x0 + 3 * cw + 22, y0 + 20 + 2 * cw, "C 看得到 A、B、C", cls="lbl", size=11.5))

    # 生出 D 之後
    x1 = 560
    b.append(txt(x1 - 10, 16, "生出 D 之後，矩陣多一列一行", cls="lbl b", size=12.5, ds="1"))
    toks4 = ["A", "B", "C", "D"]
    for j, t in enumerate(toks4):
        b.append(txt(x1 + j * cw + cw / 2, y0 - 8, t, cls="lbl s", anchor="middle", ds="1"))
    for i, t in enumerate(toks4):
        b.append(txt(x1 - 12, y0 + i * cw + cw / 2 + 4, t, cls="lbl s", anchor="end", ds="1"))
        for j in range(4):
            ok = j <= i
            newrow = i == 3
            fill = ("#fff" if not ok else (C["okW"] if newrow else C["computeW"]))
            st = (C["line2"] if not ok else (C["ok"] if newrow else C["compute"]))
            b.append(rect(x1 + j * cw, y0 + i * cw, cw - 2, cw - 2, r=4, fill=fill,
                          stroke=st, sw=1.6 if newrow and ok else 1, s=1))
            if ok:
                b.append(txt(x1 + j * cw + cw / 2 - 1, y0 + i * cw + cw / 2 + 4, "✓",
                             cls="lbl b", anchor="middle",
                             fill=C["ok"] if newrow else C["compute"], size=13, ds="1"))
    # 圈出沒變的區塊
    b.append(rect(x1 - 4, y0 - 4, 3 * cw + 2, 3 * cw + 2, r=7, fill="none",
                  stroke=C["mem"], sw=2.4, dash="6 4", s=2))
    b.append(txt(x1 + 4 * cw + 16, y0 + 1.5 * cw, "這一整塊", cls="lbl b",
                 size=12, fill=C["mem"], ds="2"))
    b.append(txt(x1 + 4 * cw + 16, y0 + 1.5 * cw + 17, "完全沒變", cls="lbl b",
                 size=12, fill=C["mem"], ds="2"))
    b.append(txt(x1 + 4 * cw + 16, y0 + 3.5 * cw + 6, "只有這一列是新的",
                 cls="lbl b", size=11.5, fill=C["ok"], ds="1"))

    y3 = y0 + 4 * cw + 44
    rows = [("為什麼沒變", "A、B、C 不可能突然看得到 D。causal mask 保證未來不會影響過去，"
                        "所以它們算好的 K/V 永遠有效。", C["mem"], 2),
            ("所以只要算新的一列", "Q_D × [K_A, K_B, K_C, K_D]。前面三列一個都不用重算。",
             C["ok"], 3),
            ("KV cache 就是幹這個的", "把 prefill 算好的 K/V 存起來，"
                                   "decode 時只 append 新 token 的 K_D、V_D。", C["compute"], 3),
            ("但它沒有解決另一件事", "Q_D 還是得掃過全部 T 個位置的 K/V。"
                                "所以 decode 的 attention 成本是 O(T)，context 越長越貴。",
             C["bad"], 4)]
    for i, (k, v, col, st) in enumerate(rows):
        b.append(g([rect(0, y3 + i * 32, 1000, 28, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y3 + i * 32, 3, 28, r=0, fill=col, stroke=None),
                    txt(13, y3 + i * 32 + 18, k, cls="lbl b", size=11.2, fill=col),
                    txt(196, y3 + i * 32 + 18, v, cls="lbl", size=11.2)], s=st))
    return svg(W, y3 + 4 * 32 + 4, b)


# ------------------------------------------- 3. 為什麼是 KV 不是 QKV cache --
def fig_qkv_roles():
    b = []
    b.append(txt(0, 16, "下一個 token E 進來的時候，它需要什麼？", cls="lbl b", size=12.5))
    cw, chh = 62, 34
    x0, y0 = 130, 34
    for i, t in enumerate(["A", "B", "C", "D"]):
        b.append(cell(x0 + i * (cw + 6), y0, cw, chh, f"Q_{t}", fill="#fff",
                      stroke=C["line"], color=C["mut2"], size=11.5))
        b.append(txt(x0 + i * (cw + 6) + cw / 2, y0 + chh + 16, "✕", cls="lbl b",
                     anchor="middle", fill=C["bad"], size=14, ds="1"))
    b.append(txt(0, y0 + 22, "歷史 Q", cls="lbl s"))
    b.append(txt(x0 + 4 * (cw + 6) + 12, y0 + 22, "用完就丟，永遠不會再被查詢",
                 cls="lbl b", size=11.5, fill=C["bad"], ds="1"))

    y1 = y0 + 66
    for row, (nm, col, tint) in enumerate((("K", C["mem"], C["memW"]),
                                           ("V", C["ssm"], C["ssmW"]))):
        y = y1 + row * (chh + 8)
        b.append(txt(0, y + 22, f"歷史 {nm}", cls="lbl s"))
        for i, t in enumerate(["A", "B", "C", "D"]):
            b.append(cell(x0 + i * (cw + 6), y, cw, chh, f"{nm}_{t}", fill=tint,
                          stroke=col, color=col, size=11.5, s=2, weight=650))
        b.append(txt(x0 + 4 * (cw + 6) + 12, y + 22,
                     "可被查詢的索引" if nm == "K" else "被查到之後要讀出來的內容",
                     cls="lbl b", size=11.5, fill=col, ds="2"))
    y2 = y1 + 2 * (chh + 8) + 14
    b.append(g([rect(0, y2, 1000, 34, fill=C["computeW"], stroke="#d5e4fd"),
                txt(13, y2 + 22, "Q 是「我現在想查什麼」，一次性的問題；K/V 是「我留給未來查的東西」。"
                                 "所以只有 K 和 V 需要 cache。",
                    cls="lbl b", size=11.5, fill=C["compute"])], s=3))
    y2 += 44
    b.append(g([rect(0, y2, 490, 78, fill=C["card2"]),
                txt(14, y2 + 22, "Attention：append", cls="lbl b", size=12, fill=C["mem"]),
                txt(14, y2 + 42, "舊的 K/V 一個位元組都不動，", cls="lbl", size=11),
                txt(14, y2 + 60, "只在尾巴多一格。memory ∝ T。", cls="lbl", size=11)], s=4))
    b.append(g([rect(510, y2, 490, 78, fill=C["card2"]),
                txt(524, y2 + 22, "GDN：就地覆寫", cls="lbl b", size=12, fill=C["ssm"]),
                txt(524, y2 + 42, "每個 token 都把同一份固定大小的", cls="lbl", size=11),
                txt(524, y2 + 60, "state 改掉。memory 是常數。", cls="lbl", size=11)], s=4))
    return svg(W, y2 + 92, b)


# ------------------------------------------------------ 4. lm_head 的數學 --
def fig_lmhead(key="q35"):
    m = M.MODELS[key]
    V, d = m.vocab, m.hidden
    b = []
    b.append(txt(0, 16, "最後一層算完之後，只剩一個向量", cls="lbl s"))
    b.append(node(0, 26, 150, 44, "h", f"[{d:,}]", color=C["neutral"], tint="#f4f6f8"))
    b.append(arrow(154, 48, 196, 48))
    b.append(g([rect(200, 22, 200, 52, r=7, fill=C["memW"], stroke=C["mem"], sw=1.3),
                txt(300, 44, "lm_head", cls="lbl b", anchor="middle", size=12.5,
                    fill=C["mem"]),
                txt(300, 62, f"W ∈ [{V:,} × {d:,}]", cls="num", anchor="middle", size=10)],
               s=1))
    b.append(arrow(404, 48, 446, 48, s=1))
    b.append(node(450, 26, 170, 44, "logits", f"[{V:,}]", color=C["compute"],
                  tint=C["computeW"], s=1))
    b.append(arrow(624, 48, 666, 48, s=2))
    b.append(node(670, 26, 150, 44, "softmax", "→ 抽一個 token", color=C["moe"],
                  tint=C["moeW"], s=2))

    y = 92
    b.append(g([rect(0, y, 1000, 58, fill=C["card2"]),
                txt(14, y + 24, "logits[i] = Σ", cls="num", size=13, fill=C["compute"]),
                txt(112, y + 24, "h[j] · W[i][j]", cls="num", size=13),
                txt(232, y + 24, f"，i = 0 … {V-1:,}", cls="num", size=13),
                txt(14, y + 46, f"要算出第 i 個分數，就要讀 W 的第 i 列。"
                                f"而 softmax 要對全部 {V:,} 個分數做正規化，一個都不能少。",
                    cls="lbl b", size=11.5, fill=C["mem"])], s=3))
    y += 70
    # 對比 embedding
    for i, (title, desc, col, tint, st) in enumerate([
            ("embed_tokens　同樣是 [V × d]，但只讀 1 列",
             f"查表：拿到 token id 就取出那一列，讀 {d:,} × 2 B = {d*2/1024:.0f} KiB。幾乎免費。",
             C["ok"], C["okW"], 4),
            ("lm_head　整個 [V × d] 都要讀",
             f"{V:,} × {d:,} × 2 B = {V*d*2/1e9:.2f} GB，每個 decode step 一次。",
             C["mem"], C["memW"], 4)]):
        b.append(g([rect(0, y + i * 52, 1000, 46, r=7, fill=tint, stroke=col, sw=1.1),
                    txt(14, y + i * 52 + 20, title, cls="lbl b", size=12, fill=col),
                    txt(14, y + i * 52 + 38, desc, cls="lbl", size=11)], s=st))
    y += 2 * 52 + 6
    b.append(g([rect(0, y, 1000, 52, fill=C["computeW"], stroke="#d5e4fd"),
                txt(14, y + 21, f"差別在於：embedding 是「用 id 找列」，lm_head 是「對每一列都算一次內積」。",
                    cls="lbl b", size=11.5, fill=C["compute"]),
                txt(14, y + 40, f"batch B 時這 {V*d*2/1e9:.2f} GB 只讀一次、被 B 個 token 分攤，"
                                f"又是同一個屋頂線的故事。",
                    cls="lbl", size=11.5, fill=C["compute"])], s=5))
    return svg(W, y + 66, b)


# --------------------------------------------- 5. token mixer / channel ----
def fig_mixers():
    b = []
    bw, bh = 430, 150
    b.append(g([rect(0, 24, bw, bh, r=10, fill=C["computeW"], stroke=C["compute"], sw=1.4),
                txt(bw / 2, 52, "TOKEN MIXER", cls="lbl b", anchor="middle", size=14,
                    fill=C["compute"]),
                txt(bw / 2, 74, "「我要去哪裡拿資訊？」", cls="lbl b", anchor="middle",
                    size=12.5),
                txt(bw / 2, 98, "跨位置搬運。只有它需要記憶體保存歷史。",
                    cls="lbl", anchor="middle", size=11),
                txt(bw / 2, 122, "Attention　·　Gated DeltaNet　·　SSM",
                    cls="lbl b", anchor="middle", size=11.5, fill=C["compute"]),
                txt(bw / 2, 143, "→ KV cache　/　state cache", cls="lbl", anchor="middle",
                    size=10.5, fill=C["mem"])]))
    b.append(g([rect(570, 24, bw, bh, r=10, fill=C["moeW"], stroke=C["moe"], sw=1.4),
                txt(570 + bw / 2, 52, "CHANNEL MIXER", cls="lbl b", anchor="middle",
                    size=14, fill=C["moe"]),
                txt(570 + bw / 2, 74, "「拿到之後要算出什麼？」", cls="lbl b",
                    anchor="middle", size=12.5),
                txt(570 + bw / 2, 98, "同一個位置內，把所有 feature 重新組合成新的 feature。",
                    cls="lbl", anchor="middle", size=11),
                txt(570 + bw / 2, 122, "FFN / SwiGLU　·　MoE", cls="lbl b",
                    anchor="middle", size=11.5, fill=C["moe"]),
                txt(570 + bw / 2, 143, "→ 不需要任何歷史 cache", cls="lbl",
                    anchor="middle", size=10.5, fill=C["ok"])], s=1))
    b.append(txt(500, 100, "→", cls="lbl b", anchor="middle", size=26, fill=C["mut2"], ds="1"))

    y = 190
    b.append(txt(0, y, "舉例：The cat didn't eat the fish because it was sick.",
                 cls="lbl b", size=12.5, ds="2"))
    steps = [("Attention 把資訊搬過來",
              "「it」這個位置從前面拿到：cat 是 animate、是前句主詞、sick 與 eat 有否定關係。",
              C["compute"], 2),
             ("FFN 把它們組合成新的東西",
              "animate + 主詞 + sick 這幾個 feature 一起出現 → 「it 指的大概是 cat」。",
              C["moe"], 3),
             ("所以有人用這組對照來記",
              "Attention = communication（通訊），FFN = computation（運算）。",
              C["ok"], 4)]
    for i, (k, v, col, st) in enumerate(steps):
        b.append(g([rect(0, y + 12 + i * 34, 1000, 30, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y + 12 + i * 34, 3, 30, r=0, fill=col, stroke=None),
                    txt(13, y + 12 + i * 34 + 19, k, cls="lbl b", size=11.2, fill=col),
                    txt(230, y + 12 + i * 34 + 19, v, cls="lbl", size=11.2)], s=st))
    y2 = y + 12 + 3 * 34 + 10
    b.append(g([rect(0, y2, 1000, 34, fill=C["memW"], stroke="#f0dcb4"),
                txt(13, y2 + 22, "換掉 token mixer，cache 的形狀就變了；換掉 channel mixer，"
                                 "只有算力與權重大小變。Qwen 的混合架構動的就是前者。",
                    cls="lbl b", size=11.5, fill=C["mem"])], s=5))
    return svg(W, y2 + 48, b)


# ------------------------------------------------- 6. 三種記憶的比喻 -------
def fig_memory_kinds():
    b = []
    cards = [
        ("Attention", "檔案櫃", C["mem"], C["memW"],
         ["每個 token 一份文件，全部留著", "查詢時掃過全部", "原始資訊無損",
          "櫃子隨 context 變大"], "memory ∝ T", 0),
        ("Gated DeltaNet", "可擦寫的白板", C["ssm"], C["ssmW"],
         ["把 key→value 的關聯壓進一個矩陣", "可以查、可以改、可以擦",
          "白板大小固定", "寫太多會互相干擾"], "memory = 常數", 1),
        ("SSM / Mamba", "腦中的狀態", C["moe"], C["moeW"],
         ["不記逐字逐句，只更新理解", "新事件進來就更新狀態",
          "不是 key→value 字典", "是動態系統的內部狀態"], "memory = 常數", 2),
    ]
    cw = 322
    for i, (name, meta, col, tint, items, foot, st) in enumerate(cards):
        x = i * (cw + 17)
        b.append(g([rect(x, 20, cw, 190, r=10, fill=tint, stroke=col, sw=1.3),
                    txt(x + 16, 46, name, cls="lbl b", size=13.5, fill=col),
                    txt(x + 16, 66, f"像「{meta}」", cls="lbl", size=11.5)],
                   s=st or None))
        for j, it in enumerate(items):
            b.append(g([circle(x + 22, 88 + j * 22, 2.4, fill=col),
                        txt(x + 32, 92 + j * 22, it, cls="lbl", size=10.8)], s=st or None))
        b.append(g([line(x + 16, 182, x + cw - 16, 182, stroke=col, sw=1),
                    txt(x + 16, 200, foot, cls="num", size=11.5, fill=col)], s=st or None))

    y = 228
    b.append(txt(0, y, "同樣一句話塞在 150K token 之前：「Frank 最喜歡的數字是 721938」",
                 cls="lbl b", size=12.5, ds="3"))
    b.append(g([rect(0, y + 10, 490, 62, fill=C["memW"], stroke="#f0dcb4"),
                txt(14, y + 32, "Attention", cls="lbl b", size=12, fill=C["mem"]),
                txt(14, y + 52, "那個 token 的 K/V 還原封不動地在，理論上查得回來。",
                    cls="lbl", size=11)], s=3))
    b.append(g([rect(510, y + 10, 490, 62, fill=C["ssmW"], stroke=C["ssm"]),
                txt(524, y + 32, "GDN", cls="lbl b", size=12, fill=C["ssm"]),
                txt(524, y + 52, "被後面 150K 個 token 反覆改寫，可能還在，也可能被蓋掉。",
                    cls="lbl", size=11)], s=3))
    y2 = y + 84
    b.append(g([rect(0, y2, 1000, 52, fill=C["okW"], stroke="#cbe9d5"),
                txt(13, y2 + 21, "所以 3:1 的混合排列很合理：GDN 提供便宜的壓縮記憶，"
                                 "每 4 層插一層 attention 補回精確查找的能力。",
                    cls="lbl b", size=11.5, fill="#14532d"),
                txt(13, y2 + 40, "GDN 的 gating（忘掉不重要的）與 delta rule（精準改寫）"
                                 "都是在跟這個有限的容量搏鬥。",
                    cls="lbl", size=11.5, fill="#14532d")], s=4))
    return svg(W, y2 + 66, b)


# ------------------------------------------------- 7. delta rule 的直覺 ----
def fig_delta():
    b = []
    b.append(txt(0, 16, "為什麼不能只是「一直往上加」？", cls="lbl b", size=12.5))
    y = 30
    b.append(g([rect(0, y, 470, 96, fill=C["badW"], stroke="#f5cdc8"),
                txt(14, y + 22, "單純累加（最原始的 linear attention）", cls="lbl b",
                    size=11.5, fill=C["bad"]),
                txt(14, y + 44, "S += apple ⊗ red", cls="num", size=11.5),
                txt(14, y + 62, "S += apple ⊗ green", cls="num", size=11.5),
                txt(14, y + 84, "兩筆疊在一起，查 apple 得到 red 和 green 的混合。",
                    cls="lbl", size=11, fill=C["bad"])]))
    b.append(g([rect(510, y, 490, 96, fill=C["okW"], stroke="#cbe9d5"),
                txt(524, y + 22, "Delta rule：先問，再改", cls="lbl b", size=11.5,
                    fill=C["ok"]),
                txt(524, y + 44, "舊值 = S · k_apple      → red", cls="num", size=11.5),
                txt(524, y + 62, "誤差 = green − red", cls="num", size=11.5),
                txt(524, y + 80, "S += β × 誤差 × k_appleᵀ", cls="num", size=11.5)], s=1))
    y2 = y + 110
    b.append(txt(0, y2, "拆開來看整條式子", cls="lbl s", ds="2"))
    b.append(g([rect(0, y2 + 8, 1000, 44, fill=C["card2"]),
                txt(500, y2 + 36,
                    "S_t = S_{t−1} ( α_t ( I − β_t k_t k_tᵀ ) ) + β_t v_t k_tᵀ",
                    cls="num", anchor="middle", size=15, fill=C["ssm"])], s=2))
    y3 = y2 + 62
    parts = [("α_t（decay gate）", "整張 state 乘上一個 0–1 的係數。α→0 等於一次清空，"
                                "適合話題切換。", C["mem"], 3),
             ("I − β_t k k ᵀ（delta）", "只把 k 這個方向原本存的東西擦掉，其他方向不動。",
              C["ok"], 3),
             ("β_t v_t k_tᵀ", "把新的關聯寫進去。β→1 表示整條換掉。", C["compute"], 4),
             ("更新的成本", "一次固定維度的矩陣運算 O(d_k·d_v)，"
                        "不是「把所有舊 key 跑一遍」，state 裡根本沒有 key 的清單。",
              C["moe"], 5)]
    for i, (k, v, col, st) in enumerate(parts):
        b.append(g([rect(0, y3 + i * 32, 1000, 28, r=5, fill="#fff", stroke=C["line2"]),
                    rect(0, y3 + i * 32, 3, 28, r=0, fill=col, stroke=None),
                    txt(13, y3 + i * 32 + 18, k, cls="lbl b", size=11.2, fill=col),
                    txt(230, y3 + i * 32 + 18, v, cls="lbl", size=11.2)], s=st))
    y4 = y3 + 4 * 32 + 8
    b.append(g([rect(0, y4, 1000, 34, fill=C["computeW"], stroke="#d5e4fd"),
                txt(13, y4 + 22, "Mamba2 只有 α，DeltaNet 只有 delta。Gated DeltaNet 兩個都有，"
                                 "所以既能快速清空，也能精準改寫。",
                    cls="lbl b", size=11.5, fill=C["compute"])], s=6))
    return svg(W, y4 + 48, b)


# --------------------------------------------- 8. 高併發下各指標怎麼變 -----
def fig_concurrency(key="q38"):
    import math
    m = M.MODELS[key]
    users = 128
    cands = [8, 16, 24, 32, 48, 64, 96, 128]
    rows = [M.serve_sim(m, users, c) for c in cands]
    b = []
    x0, y0, w, h = 74, 268, 560, 210
    b.append(axes(x0, y0, w, h))
    b.append(txt(x0 - 2, y0 - h - 14, "↑ 相對值（各自正規化）", cls="lbl s"))
    b.append(txt(x0 + w, y0 + 34, "max_num_seqs →", cls="lbl s", anchor="end"))

    def X(i):
        return x0 + i / (len(cands) - 1) * w
    series = [("TTFT", [r["ttft_ms"] for r in rows], C["compute"], 1),
              ("ITL / TPOT", [r["tpot_ms"] for r in rows], C["mem"], 2),
              ("整機 tok/s", [r["tokens_per_s"] for r in rows], C["ok"], 3)]
    for nm, vals, col, st in series:
        mx = max(vals)
        pts = [f"{X(i):.1f},{y0 - v / mx * h:.1f}" for i, v in enumerate(vals)]
        b.append(path("M" + " L".join(pts), stroke=col, sw=2.6, s=st))
        for i, v in enumerate(vals):
            b.append(circle(X(i), y0 - v / mx * h, 3.4, fill=col, s=st))
        b.append(txt(X(len(cands) - 1) - 4, y0 - vals[-1] / mx * h - 11, nm,
                     cls="lbl b", anchor="end", size=10.6, fill=col, ds=st))
    for i, c in enumerate(cands):
        b.append(line(X(i), y0, X(i), y0 + 4, stroke=C["line"], sw=1))
        b.append(txt(X(i), y0 + 18, str(c), cls="num", anchor="middle"))
    # 標出屋頂線轉折點落在哪
    knee = M.GPU.knee_tokens("fp8")
    for i, r in enumerate(rows):
        if r["tokens_per_step"] > knee:
            b.append(line(X(i - .5), y0 - h, X(i - .5), y0, stroke=C["bad"], sw=1.4,
                          dash="5 4", s=4))
            b.append(txt(X(i - .5) + 7, y0 - h + 16, f"tok/step 越過 N* = {knee:.0f}",
                         cls="lbl b", size=10.6, fill=C["bad"], ds="4"))
            b.append(txt(X(i - .5) + 7, y0 - h + 31, "ITL 從這裡開始爬升",
                         cls="lbl", size=10.6, fill=C["bad"], ds="4"))
            break
    # 右欄：48 vs 96 對照
    a48 = next(r for r in rows if r["max_seqs"] == 48)
    a96 = next(r for r in rows if r["max_seqs"] == 96)
    bx = 664
    b.append(g([rect(bx, 30, 336, 250, fill=C["card2"]),
                txt(bx + 14, 54, f"{users} 個人同時上門", cls="lbl b", size=12.5),
                txt(bx + 14, 72, "in 2048 / out 512，27B FP8", cls="lbl s", size=10.4),
                txt(bx + 150, 98, "48", cls="lbl b", anchor="middle", size=12,
                    fill=C["compute"]),
                txt(bx + 262, 98, "96", cls="lbl b", anchor="middle", size=12,
                    fill=C["ssm"]),
                line(bx + 14, 106, bx + 322, 106, stroke=C["line"], sw=1)], s=5))
    metrics = [("running / 排隊", f"{a48['running']:.0f} / {a48['queued']:.0f}",
                f"{a96['running']:.0f} / {a96['queued']:.0f}"),
               ("tok / step", f"{a48['tokens_per_step']:.0f}", f"{a96['tokens_per_step']:.0f}"),
               ("瓶頸", "頻寬", "算力"),
               ("ITL / TPOT", f"{a48['tpot_ms']:.2f} ms", f"{a96['tpot_ms']:.2f} ms"),
               ("TTFT", f"{a48['ttft_ms']/1000:.2f} s", f"{a96['ttft_ms']/1000:.2f} s"),
               ("E2E", f"{a48['e2e_s']:.2f} s", f"{a96['e2e_s']:.2f} s"),
               ("整機 tok/s", f"{a48['tokens_per_s']:,.0f}", f"{a96['tokens_per_s']:,.0f}")]
    for i, (k, v1, v2) in enumerate(metrics):
        yy = 126 + i * 21
        better48 = k in ("ITL / TPOT",)
        b.append(g([txt(bx + 14, yy, k, cls="lbl", size=10.6),
                    txt(bx + 190, yy, v1, cls="num", anchor="end", size=10.6,
                        fill=C["ok"] if better48 else C["ink2"]),
                    txt(bx + 322, yy, v2, cls="num", anchor="end", size=10.6,
                        fill=C["ink2"] if better48 else C["ok"])], s=5))
    y2 = y0 + 44
    b.append(g([rect(0, y2, 1000, 52, fill=C["memW"], stroke="#f0dcb4"),
                txt(13, y2 + 21, "把 max_num_seqs 掐小，唯一真正變好的是 ITL/TPOT，"
                                 "因為每個 step 的 token 數還停在屋頂線左邊。",
                    cls="lbl b", size=11.5, fill=C["mem"]),
                txt(13, y2 + 40, "代價是 80 個人在門外排隊，TTFT 從 1.0 秒變成 3.1 秒，"
                                 "E2E 與整機吞吐也一起變差。",
                    cls="lbl", size=11.5, fill=C["mem"])], s=6))
    return svg(W, y2 + 66, b)
