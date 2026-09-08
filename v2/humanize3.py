# -*- coding: utf-8 -*-
"""第三批修潤：否定式排比、金句收尾、多餘的強調詞。"""
from humanize import apply
import sys, io

R = [
# ------------------------------------------------ 否定式排比「不是…而是」
("slides_part4.py",
 "<div class=\"csub\">容量數字之所以會被吵，多半不是因為量錯，而是因為兩邊講的不是同一個指標、\n同一個百分位或同一個量測窗。這一章把定義釘死。</div>",
 "<div class=\"csub\">容量數字會被吵，多半不是量錯。是兩邊講的指標、百分位或量測窗根本不一樣。\n這一章把定義釘死。</div>"),
("slides_part4.py",
 "<div class=\"csub\">「這台機器能服務幾個人」不是一個數字，而是一個條件式：\n在這個 workload、這組 SLO、這個引擎版本之下，能承諾多少。\n這一章給出一套可重現的流程。</div>",
 "<div class=\"csub\">「這台機器能服務幾個人」沒有單一答案。它是一個條件式：\n在這個 workload、這組 SLO、這個引擎版本之下，能承諾多少。\n這一章給出一套可重現的流程。</div>"),
("slides_part2.py",
 "要強調 continuous batching 不是\n「把 batch 開大」，而是「讓 batch 的成員每一輪都可以換」。",
 "要強調 continuous batching 的重點不在「把 batch 開大」，\n而在「讓 batch 的成員每一輪都可以換」。"),

# ------------------------------------------------------ 金句式的「這就是」
("figs_arch.py", '"這就是型號裡「A10B」的意思"', '"型號裡的 A10B 講的就是這個"'),
("figs_core.py", '"→ 這就是 memory-bound"', '"→ 於是卡在 memory-bound"'),
("slides_part1.py",
 "讀的權重量一模一樣，但前者做了 2,048 倍的運算。這就是為什麼混合 prefill 與 decode\n在同一個 batch（chunked prefill）能同時改善兩邊。",
 "讀的權重量一模一樣，但前者做了 2,048 倍的運算。把 prefill 與 decode 混在同一個 batch\n（chunked prefill）能同時改善兩邊，道理就在這。"),
("slides_part3.py",
 '"這就是為什麼 DFlash2 敢用 k = 7，而 MTP 通常停在 k = 2–4。"',
 '"DFlash2 敢用 k = 7、MTP 通常停在 k = 2–4，差別就在這裡。"'),
("slides_part4.py",
 "ρ = 0.9 時延遲是空載的 10 倍，ρ = 0.95 時是 20 倍。\n這就是為什麼容量規劃要留 20–30% 的餘裕。",
 "ρ = 0.9 時延遲是空載的 10 倍，ρ = 0.95 時是 20 倍。\n容量規劃要留 20–30% 餘裕的理由就在這。"),

# ---------------------------------------------------------- 多餘的強調詞
("slides_part1.py", "<div class=\"csub\">推論的成本結構完全由這一段機制決定。",
 "<div class=\"csub\">推論的成本結構就是在這一段裡定下來的。"),
("slides_part1.py", '{pane("唯一的例外"', '{pane("有一個例外"'),
("slides_part1.py", '"<b>③</b> 這串 id 的長度 T 才是真正的輸入長度。',
 '"<b>③</b> 這串 id 的長度 T 才是實際的輸入長度。'),
("slides_part2.py", "真正的差異在 decode（GDN 省很多）與 prefix caching（GDN 目前吃虧）。",
 "差異在 decode（GDN 省很多）與 prefix caching（GDN 目前吃虧）。"),
("slides_part3.py", "所以它是<b>樂觀估計</b>；真正的 τ 要從引擎的 acceptance metrics 讀。",
 "所以它是<b>樂觀估計</b>；實際的 τ 要從引擎的 acceptance metrics 讀。"),
("slides_part3.py", "不是實測結論。真正的決策要等 CH9 的兩階段測試跑完，用同一組 SLO 比較開／關的 goodput。",
 "不是實測結論。決策要等 CH9 的兩階段測試跑完，用同一組 SLO 比較開／關的 goodput。"),
("slides_part3.py", '"在短 context 時完全主導。"', '"在短 context 時佔掉絕大部分。"'),
("slides_part3.py", 'f"完全在 memory-bound 區，B×(k+1) 遠低於 {KNEE:.0f}。',
 'f"整段都在 memory-bound 區，B×(k+1) 遠低於 {KNEE:.0f}。'),
("slides_part4.py", "只看 throughput 會做出完全錯誤的決策。", "只看 throughput 會做出錯誤的決策。"),
("slides_part4.py", '"<b>網路 + 佇列</b>：這一段完全不在模型裡，但使用者一樣要等。"',
 '"<b>網路 + 佇列</b>：這一段不在模型裡，但使用者一樣要等。"'),
("slides_part2.py", "<td>一次一個 request。權重讀取完全沒有分攤，最浪費。</td>",
 "<td>一次一個 request。權重讀取沒有分攤，最浪費。</td>"),

# ------------------------------------------------------------ 三段式排比
("slides_part1.py",
 "<div class=\"csub\">一個 request 進到 GPU 之後發生什麼事，以及「能服務多少人」該怎麼算</div>",
 "<div class=\"csub\">一個 request 進到 GPU 之後發生什麼事，以及「能服務多少人」該怎麼算</div>"),
("figs_core.py", '"換來三個能力：沒有碎片、前綴可共用、request 可以被換出再換回。"',
 '"換來三個能力：沒有碎片、前綴可共用、request 可以被換出再換回。"'),

# ------------------------------------------------------------ 口氣調整
("slides_part1.py", "這一章結束時，我們會得到一個\n可以拿去解釋後面所有現象的單一數字：B200 上的屋頂線轉折點。",
 "這一章結束時會得到一個數字，後面所有現象都可以拿它來解釋：\nB200 上的屋頂線轉折點。"),
("slides_part1.py", "章節頁只停 15 秒，唸出最後一行的 N* 讓聽眾記住這個數字。",
 "章節頁停 15 秒就好，把最後一行的 N* 唸出來讓聽眾記住。"),
("slides_part1.py", "這一頁是防呆頁。第一版報告最容易被誤讀的地方，就是聽眾把示意曲線當成實測。\n這一版把可信度分級明講，並在每張教學示例圖上都留紅字。",
 "這一頁是防呆用的。第一版最容易被誤讀的地方，就是聽眾把示意曲線當成實測。\n這一版把可信度分級講明，每張教學示例圖上也都留了紅字。"),
("slides_part1.py", "這一頁的用途是把「字數」與「token 數」分開。業務端談的是字數，容量計算用的是 token 數，",
 "這一頁要把「字數」與「token 數」分開。業務端談字數，容量計算用 token 數，"),
("slides_part2.py", "這一章的目的是讓「調參數」變得有跡可循：知道每個旗標作用在哪一層。",
 "這一章要讓「調參數」變得有跡可循：知道每個旗標作用在哪一層。"),
("slides_part2.py", "這一章的價值在「可驗證」。所有參數都由 config 推導，並與 HuggingFace 上的"
 "safetensors dtype 統計核對，誤差 &lt; 0.02%。",
 "這一章的重點在可驗證。所有參數都由 config 推導，再跟 HuggingFace 上的 "
 "safetensors dtype 統計核對，誤差 &lt; 0.02%。"),
("slides_part2.py", "這一章是全報告技術密度最高的部分。若聽眾偏工程，可以在這裡多花時間。",
 "全報告技術密度最高的就是這一章。聽眾偏工程的話可以多花點時間。"),
("slides_part3.py", "這一頁要傳達的態度：量化的精度風險是<b>可量測</b>的，不要用感覺討論。",
 "態度很簡單：量化的精度風險是<b>可量測</b>的，別用感覺討論。"),
("slides_part4.py", "這一章短，但值得要求所有人用同一份定義寫報告。",
 "這一章短，但值得要求所有人照同一份定義寫報告。"),
]

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    c, m = apply(R)
    print(f"applied {c} / {len(R)}")
    for fn, frag in m:
        print("  MISS", fn, frag)
