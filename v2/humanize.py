# -*- coding: utf-8 -*-
"""一次性的文字修潤：拿掉 AI 寫作痕跡。

處理的模式：
  · 「一句話」標籤（公式化開場）
  · 破折號 —— 當作戲劇性揭示的用法（94 處）
  · 「不是…而是」否定式排比
  · 「這正是／這就是」的金句收尾
  · 多餘的強調詞（完全、真正的、唯一的、必須）
  · 三段式排比

執行：python humanize.py   （會就地改寫 v2/ 底下的原始碼，再跑 build.py）
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

# (檔名, 舊字串, 新字串)
R = [
# ---------------------------------------------------------------- 「一句話」
("common.py",
 '''def gist(text, accent=None):
    st = f' style="--accent:var(--{accent})"' if accent else ""
    return f'<div class="gist"{st}><b>一句話</b><span>{text}</span></div>\'''',
 '''def gist(text, accent=None):
    st = f' style="--accent:var(--{accent})"' if accent else ""
    return f'<div class="gist"{st}><span>{text}</span></div>\''''),

# ------------------------------------------------------------ common.py 名詞
("common.py",
 "A10B 指每個 token 只用約 10B —— 算力照 10B 算，但記憶體要放滿 122B。",
 "A10B 指每個 token 只用約 10B。算力照 10B 算，記憶體要放滿 122B。"),
("common.py",
 "整份報告的最佳化都在提高這個數字。",
 "報告裡談的最佳化，多半都是在拉高這個數字。"),
("common.py",
 "一次只產生一個 token，再把它接回輸入產生下一個。生成 T 個 token 就要 T 次 forward，這是 decode 慢的根本原因。",
 "一次只產生一個 token，再把它接回輸入產生下一個。生成 T 個 token 就要跑 T 次 forward，decode 慢的原因就在這裡。"),
("common.py",
 "把文字切成 token 並轉成整數 id",
 "把文字切成 token 並轉成整數 id"),
("common.py",
 "沒有 SLO 就沒有容量數字。",
 "沒講定 SLO，容量數字就沒有意義。"),
("common.py",
 "猜對就一次前進多格，猜錯就丟掉。輸出分布不變。",
 "猜對就一次前進多格，猜錯就丟掉。輸出分布不會變。"),

# ------------------------------------------------------------- figs_arch.py
("figs_arch.py",
 '"空出來的位置立刻被新 request 補上 —— "\n                                 "同一次權重讀取被更多 token 分攤，正是 CH1 的結論。"',
 '"空出來的位置立刻被新 request 補上。"\n                                 "同一次權重讀取被更多 token 分攤，就是 CH1 講的那件事。"'),
("figs_arch.py",
 '"① 幾乎沒有碎片 —— 只浪費最後一塊的尾巴"',
 '"① 幾乎沒有碎片，只浪費最後一塊的尾巴"'),
("figs_arch.py",
 '"而且 batch 一大，被選到的 expert 就會涵蓋幾乎全部 256 個 "\n                                  "—— 記憶體流量會從 2 GB 長到 65 GB。CH7 會用到這件事。"',
 '"而且 batch 一大，被選到的 expert 就會涵蓋幾乎全部 256 個，"\n                                  "記憶體流量從 2 GB 長到 65 GB。CH7 會用到這件事。"'),
("figs_arch.py",
 '("代價", "S 是有損壓縮。context 越長，被擠掉的細節越多 —— "\n                    "所以每 4 層要放 1 層 full attention 來補。", C["bad"], 5)]',
 '("代價", "S 是有損壓縮。context 越長，被擠掉的細節越多，"\n                    "所以每 4 層要放一層 full attention 來補。", C["bad"], 5)]'),
("figs_arch.py",
 '("chunk 之間", "只傳遞一個固定大小的狀態 S —— 這是唯一的序列依賴", C["ssm"], 2),',
 '("chunk 之間", "只傳遞一個固定大小的狀態 S，序列依賴只剩這一條", C["ssm"], 2),'),
("figs_arch.py",
 '"兩種 cache 必須「同步前進」：任何一邊被換出或算錯位置，"\n                                 "整條序列的後續輸出都會壞掉。這是 hybrid 模型 serving 最難的地方。"',
 '"兩種 cache 得「同步前進」：任何一邊被換出或算錯位置，"\n                                 "整條序列後面的輸出就全壞了。hybrid 模型 serving 最難的地方就在這。"'),
("figs_arch.py",
 '"context 短的時候，混合模型的", cls="lbl", size=11),',
 '"context 短的時候，混合模型的", cls="lbl", size=11),'),
("figs_arch.py",
 'txt(730, 204, "→ 短對話的併發上限其實由", cls="lbl", size=10.6),\n                txt(730, 219, "　 SSM state 決定，不是 KV。", cls="lbl b", size=10.6,',
 'txt(730, 204, "→ 短對話的併發上限其實卡在", cls="lbl", size=10.6),\n                txt(730, 219, "　 SSM state，不在 KV。", cls="lbl b", size=10.6,'),

# ------------------------------------------------------------- figs_core.py
("figs_core.py",
 '"新產生的 token 接回輸入尾端 —— 這就是 autoregressive"',
 '"新產生的 token 接回輸入尾端，一輪一輪這樣跑下去"'),
("figs_core.py",
 '"所以「Q 很寬」不花記憶體，只花算力 —— 這是 GQA 的整個賣點。"',
 '"所以「Q 很寬」不花記憶體，只花算力。GQA 的好處就在這裡。"'),
("figs_core.py",
 '"怎麼讓每次讀取權重的成本，被更多 token 分攤 —— "\n                             "continuous batching、chunked prefill、speculative decoding 全都是。"',
 '"怎麼讓每次讀取權重的成本，被更多 token 分攤。"\n                             "continuous batching、chunked prefill、speculative decoding 都在做這件事。"'),
("figs_core.py",
 '"沒有 cache：每一輪都要把前面 T−1 個位置重算 → 總成本 O(T²)"',
 '"沒有 cache：每一輪都要把前面 T−1 個位置重算，總成本 O(T²)"'),

# ------------------------------------------------------------- figs_load.py
("figs_load.py",
 '"TPOT = (E2EL − TTFT) ÷ (輸出 token 數 − 1)　"\n                                 "—— 是「每個 request 自己的平均」，不是所有 ITL 的平均。"',
 '"TPOT = (E2EL − TTFT) ÷ (輸出 token 數 − 1)　"\n                                 "算的是「每個 request 自己的平均」，不是所有 ITL 的平均。"'),
("figs_load.py",
 '"少數人很慢 —— 但他們就是會抱怨的人"',
 '"少數人很慢，而他們就是會來抱怨的人"'),
("figs_load.py",
 '"但全部的人都超過 SLO —— goodput = 0。"',
 '"但全部的人都超過 SLO，goodput = 0。"'),
("figs_load.py",
 '"為什麼不能只做 Stage 1：閉環測試的延遲永遠不會爆炸，"\n                                "因為 client 要等回覆才發下一個 ——", cls="lbl b", size=11.5,',
 '"為什麼不能只做 Stage 1：閉環測試的延遲不會爆炸，"\n                                "因為 client 要等回覆才發下一個，", cls="lbl b", size=11.5,'),
("figs_load.py",
 '"報告紀律：任何一個延遲數字都必須同時寫出「哪個指標、哪個百分位、"\n                             "哪個量測窗、哪個 workload」。"',
 '"報告紀律：任何一個延遲數字都要寫清楚指標、百分位、量測窗與 workload。"'),
("figs_load.py",
 '"「TPOT 30 ms」沒有意義；「在 2k in / 512 out、32 併發、"\n                             "穩定 5 分鐘窗內，p95 TPOT = 30 ms」才有意義。"',
 '"「TPOT 30 ms」沒有意義；「2k in / 512 out、32 併發、"\n                             "穩定 5 分鐘窗內，p95 TPOT = 30 ms」才有。"'),
("figs_load.py",
 '"共通原則：一次只動一個旗標，每次都用同一組 workload 與 SLO 重測，"\n                                "並記錄下來。", cls="lbl b", size=11.5, fill="#14532d"),',
 '"共通原則：一次只動一個旗標，每次用同一組 workload 與 SLO 重測，並記錄下來。",\n                    cls="lbl b", size=11.5, fill="#14532d"),'),
("figs_load.py",
 '"沒有「最佳設定」，只有「在這個 workload 與這組 SLO 之下的最佳設定」。"',
 '"沒有放諸四海的最佳設定，只有「這個 workload 配這組 SLO」的最佳設定。"'),
("figs_load.py",
 '"這代表 8 卡的節點可以跑 8 份獨立的 DP 副本，而不是 1 份 TP=8。"\n                                "沒有 all-reduce、沒有通訊抖動，", cls="lbl", size=11,',
 '"8 卡的節點因此可以跑 8 份獨立的 DP 副本，取代 1 份 TP=8。"\n                                "沒有 all-reduce，也沒有通訊抖動，", cls="lbl", size=11,'),

# ------------------------------------------------------------- figs_spec.py
("figs_spec.py",
 '("保底 token", "拒絕發生的位置，目標模型自己的輸出直接採用 —— "\n                          "所以最差情況也一定前進 1 格，不會比不猜還慢（就 token 數而言）",',
 '("保底 token", "拒絕發生的位置，直接採用目標模型自己的輸出。"\n                          "所以最差情況也會前進 1 格，就 token 數而言不會比不猜慢",'),
("figs_spec.py",
 '"speculative decoding 的整個想法：拿這塊閒置算力去「猜」後面幾個 token，"',
 '"speculative decoding 就是打這塊閒置算力的主意：拿它去猜後面幾個 token，"'),
("figs_spec.py",
 '"再讓目標模型用<同一次>權重讀取一口氣驗證它們。猜對就一次前進好幾格。"',
 '"再讓目標模型用同一次權重讀取一口氣驗證。猜對就一次前進好幾格。"'),
("figs_spec.py",
 '"數學上與不猜時的輸出分布完全相同。"',
 '"數學上與不猜時的輸出分布相同。"'),
("figs_spec.py",
 '"這正是本章要量化的那條界線。"',
 '"本章要量的就是那條界線。"'),
("figs_spec.py",
 '"關鍵差異：MTP 的起草成本隨 k 線性成長（T_draft = k × t_step）；"\n                                "DFlash2 的起草成本與 k 無關（T_draft = t_parallel）。"',
 '"差別在成本結構：MTP 的起草成本隨 k 線性成長（T_draft = k × t_step），"\n                                "DFlash2 的起草成本與 k 無關（T_draft = t_parallel）。"'),
("figs_spec.py",
 '"教訓：長 context 服務一定要重新量接受率，不能沿用短 prompt 的結論。"',
 '"長 context 服務要重新量接受率，短 prompt 的結論搬不過來。"'),
("figs_spec.py",
 '"結論：記憶體不是 spec decoding 的主要限制，但在 cache 已經"\n                                "很緊的長 context 場景，這幾 GB 會直接吃掉併發數。"',
 '"記憶體不是 spec decoding 的主要限制。但 cache 本來就很緊的長 context 場景，"\n                                "這幾 GB 會直接吃掉併發數。"'),
]


def apply(rules, root=HERE):
    missed = []
    per_file = {}
    for fn, old, new in rules:
        per_file.setdefault(fn, []).append((old, new))
    changed = 0
    for fn, pairs in per_file.items():
        p = root / fn
        s = p.read_text(encoding="utf-8")
        for old, new in pairs:
            if old not in s:
                missed.append((fn, old[:60]))
                continue
            s = s.replace(old, new, 1)
            changed += 1
        p.write_text(s, encoding="utf-8")
    return changed, missed


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    c, m = apply(R)
    print(f"applied {c} / {len(R)}")
    for fn, frag in m:
        print("  MISS", fn, frag)
