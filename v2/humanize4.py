# -*- coding: utf-8 -*-
"""第四批修潤：新增章節的破折號、否定式排比與金句收尾。"""
from humanize import apply
import sys, io

R = [
# ------------------------------------------------------------ 破折號 ------
("blog_body.py",
 "要離開 memory-bound 只有一條路，就是提高每個 step 的 token 數 ——\n更大的 batch，或是 {T(\"Speculative decoding\")}。",
 "要離開 memory-bound 只有一條路，就是提高每個 step 的 token 數：\n更大的 batch，或是 {T(\"Speculative decoding\")}。"),
("blog_body.py",
 "這些試算器就變成很好用的\n快速估算工具 —— 換設定時先估一次，再決定要不要花時間實測。",
 "這些試算器就變成很好用的快速估算工具。\n換設定時先估一次，再決定要不要花時間實測。"),
("figs_new.py",
 'f"batch B 的時候這 {V*d*2/1e9:.2f} GB 只讀一次、被 B 個 token 分攤 —— "\n                                f"又是同一個屋頂線的故事。",',
 'f"batch B 的時候這 {V*d*2/1e9:.2f} GB 只讀一次、被 B 個 token 分攤，"\n                                f"又是同一個屋頂線的故事。",'),
("figs_new.py",
 '"不是「把所有舊 key 跑一遍」——state 裡根本沒有 key 的清單。",',
 '"不是「把所有舊 key 跑一遍」，state 裡根本沒有 key 的清單。",'),
("slides_part1.py",
 "<p><b>vocab 是什麼？</b>一張跨語言共用的對照表，但它不是「詞典」——是 byte-level BPE 的\nsubword 表。",
 "<p><b>vocab 是什麼？</b>一張跨語言共用的對照表，但表裡的不是「詞」，\n而是 byte-level BPE 切出來的 subword。"),

# -------------------------------------------------------- 否定式排比 ------
("blog_body.py",
 "<p>最反直覺的是 ITL 那一列：它不是一路變差，而是在屋頂線轉折點之前幾乎不動。",
 "<p>最反直覺的是 ITL 那一列。它不會一路變差；在屋頂線轉折點之前幾乎不動。"),
("slides_part4.py",
 "<p>最關鍵的是 ITL 那一列：它<b>不是</b>一路變差，而是在屋頂線轉折點之前幾乎不動。",
 "<p>最關鍵的是 ITL 那一列。它<b>不會</b>一路變差；在屋頂線轉折點之前幾乎不動。"),

# ---------------------------------------------------------- 金句收尾 ------
("blog_body.py",
 '"這就是 ", T("KV cache"), "。", cite("paged"))',
 '"存起來的那份就叫 ", T("KV cache"), "。", cite("paged"))'),
("blog_body.py",
 "直到這個 request 結束。而這就是併發數的硬上限。\")",
 "直到這個 request 結束。併發數的硬上限就卡在這裡。\")"),
("blog_body.py",
 "所以 MoE 再大也不會增加每條序列的記憶體，只會增加權重。\n這就是後面「MoE 省算力不省記憶體」那句話的來源。</p>",
 "所以 MoE 再大也不會增加每條序列的記憶體，只會增加權重。\n後面「MoE 省算力不省記憶體」那句話就是從這裡來的。</p>"),
("figs_new.py",
 '("這就是 KV cache 的用途", "把 prefill 算好的 K/V 存起來，"',
 '("KV cache 就是幹這個的", "把 prefill 算好的 K/V 存起來，"'),
("figs_new.py",
 '"只有算力與權重大小變。這就是 Qwen 混合架構在做的事。",',
 '"只有算力與權重大小變。Qwen 的混合架構動的就是前者。",'),

# ------------------------------------------------------------ 其他 --------
("blog_body.py",
 '"真正的差異在 decode（省很多）與 prefix caching（目前吃虧）。", a',
 '"差異在 decode（省很多）與 prefix caching（目前吃虧）。", a'),
("blog_body.py",
 '"真正的 τ 要從引擎的 acceptance metrics 讀。", accent="s',
 '"實際的 τ 要從引擎的 acceptance metrics 讀。", accent="s'),
("blog_body.py",
 "p(\"這筆交易在長 context 下非常划算，但它的代價是<b>記憶體</b>：每個 token 都要佔一塊 HBM，\",",
 "p(\"長 context 下這筆交易很划算，但代價是<b>記憶體</b>：每個 token 都要佔一塊 HBM，\","),
]

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    c, m = apply(R)
    print(f"applied {c} / {len(R)}")
    for fn, frag in m:
        print("  MISS", fn, frag)
