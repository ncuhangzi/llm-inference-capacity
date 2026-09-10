# -*- coding: utf-8 -*-
# ==================================================== CH5 量化 ============
chapter("CH5 量化：NVFP4 與 FP8", "moe")

slide("量化：NVFP4 與 FP8", kind="section", accent="moe", body=f"""
<div class="no">CHAPTER 05</div>
<h1>量化：NVFP4 與 FP8</h1>
<div class="csub">這次要上線的是 nvidia/Qwen3.5-122B-A10B-NVFP4 與 Qwen/Qwen3.8-27B-FP8。
量化的影響不只是檔案變小。它決定能不能單卡跑，也決定哪些張量還留在 BF16 當頻寬大戶。</div>
<ul>
 <li>NVFP4 與 FP8 的位元佈局</li>
 <li>checkpoint 實際量化了哪些張量（直接看 weight map）</li>
 <li>省下多少：權重、KV cache、可用 cache 空間</li>
 <li>精度代價：NVIDIA 官方的評測表</li>
 <li>為什麼量化不會移動屋頂線的轉折點</li>
</ul>""",
      notes="<p>這一章是第二版新增的。第一版假設 BF16，記憶體數字全部偏大 2–3 倍。</p>")

slide("NVFP4 與 FP8 的位元佈局", "兩層 scale 是 NVFP4 能在 4 bit 保住精度的關鍵",
      sources=["nvfp4", "nvfp4b", "nvq", "f38"], accent="moe", body=f"""
{gist("NVFP4 = 每 16 個 4-bit 值共用一個 FP8 scale，再乘一個 per-tensor 的 FP32 scale，平均 4.5 bit／值。")}
{fig("nvfp4", F3.fig_nvfp4(), 5, [
 "一個 NVFP4 block 就是 16 個 E2M1 值。E2M1 只有 4 bit，非零大小只有 8 個級距。",
 "每 16 個值共用一個 <b>FP8 E4M3 的 block scale</b>。block 越小，一組值的動態範圍越窄，"
 "4 bit 就越夠用。",
 "再乘上一個 <b>per-tensor 的 FP32 scale</b>（checkpoint 裡的 <code>weight_scale_2</code>）。",
 "把 scale 攤提進去：(16×4 + 8) ÷ 16 = <b>4.5 bit / 值</b>。",
 "對照 FP8：Qwen3.8-27B-FP8 用 128 個值共用一個 scale，平均 8.03 bit / 值。"],
 accent="moe")}""",
      notes="""<p>三個容易搞混的點：<br>
① NVFP4 與 OCP 的 MXFP4 不同：block 是 16 不是 32，scale 是 FP8-E4M3 不是 2 的冪次（E8M0）。
所以 NVFP4 精度較好，但 scale 的儲存開銷是 MXFP4 的兩倍。<br>
② 「4-bit 模型」實際上是 4.5 bit，算記憶體時要用 4.5。<br>
③ FP8 的 blockwise-128 是<b>二維</b>的 128×128 區塊，checkpoint 裡叫 <code>weight_scale_inv</code>。</p>
<p>Blackwell 的張量核心原生支援這個格式，所以不需要在 kernel 裡反量化再算，比軟體量化快的原因就在這。</p>""")

slide("checkpoint 實際量化了哪些張量", "直接看 safetensors 的 weight map，不用猜",
      sources=["nvi", "i38", "nvq", "nv4"], accent="moe", body=f"""
{gist("122B 的 NVFP4 只量化了 routed expert；GDN、attention、shared expert、lm_head、MTP head 全部還是 BF16。")}
{fig("qmap", F3.fig_quantmap(), 3, [
 "122B 的 NVFP4 checkpoint：紫色是 4-bit 的 routed expert，橘色是 FP8 的 block scale，"
 "藍色是完全沒被量化的 BF16 部分。BF16 佔了 21.9%。",
 "27B 的 FP8 checkpoint：青色是 FP8，比例高很多（80%），因為 dense FFN 佔了模型的大宗。",
 "怎麼確定的：只有帶 <code>weight_scale</code> 的張量才是被量化的。"
 "把數量乘出來，與 HuggingFace 回報的 dtype 統計逐一比對。"], accent="moe")}""",
      notes="""<p>這一頁的發現大概是整份報告裡最有價值的，而且可以自己驗。</p>
<p>對 122B 的實務意涵：因為 36 層 GDN 全部維持 BF16，它們在 batch 1 的 decode 佔了
6.37 GB 的記憶體流量，是總量 12.8 GB 的一半。也就是說 <b>NVFP4 只解決了 expert 那一半</b>，
另一半仍然是 BF16 的頻寬成本。想再快，得等 GDN 的 FP8 / FP4 kernel。</p>
<p>對 27B：GDN 的 <code>in_proj_a</code>、<code>in_proj_b</code>、<code>conv1d</code>、
<code>A_log</code>、<code>dt_bias</code> 保持 BF16。這些是決定 decay gate 的參數，
數值敏感度高，量化下去會直接壞掉。</p>""")

slide("省了多少", "權重、KV cache 與可用 cache 空間的三筆帳",
      sources=["nv4", "f38", "nvi", "i38"], accent="mem", body=f"""
{gist(f"122B 從 {gb(Q35.real_bf16_bytes)} GB 降到 {gb(Q35.real_weight_bytes)} GB。"
      f"這正好是「單卡跑得動」與「至少要兩張卡」的分界線。")}
{table(["", "~BF16 權重", "~量化後權重", "~省下", "~單張 B200 180 GB",
        "~剩給 cache（util 0.90，扣 8 GiB activation）"], [
 [f"{Q35.label} <span class='tagc e'>NVFP4</span>",
  f"{gb(Q35.real_bf16_bytes)} GB", f"<b>{gb(Q35.real_weight_bytes)} GB</b>",
  f"{(1-Q35.real_weight_bytes/Q35.real_bf16_bytes)*100:.0f}%",
  "<span class='tagc g'>單卡放得下</span>（BF16 需 2 卡）",
  f"{M.budget(Q35)['cache']:.1f} GiB"],
 [f"{Q38.label} <span class='tagc s'>FP8</span>",
  f"{gb(Q38.real_bf16_bytes)} GB", f"<b>{gb(Q38.real_weight_bytes)} GB</b>",
  f"{(1-Q38.real_weight_bytes/Q38.real_bf16_bytes)*100:.0f}%",
  "<span class='tagc g'>單卡輕鬆</span>", f"{M.budget(Q38)['cache']:.1f} GiB"],
 [f"{Q332.label} <span class='tagc n'>BF16</span>",
  f"{gb(Q332.real_bf16_bytes)} GB", "—", "—",
  "<span class='tagc g'>單卡放得下</span>", f"{M.budget(Q332)['cache']:.1f} GiB"],
], "compact", hi=(0,))}
<div class="cols3" style="margin-top:2px">
 {pane("KV cache 也可以量化", f'''<p style="font-size:13.2px">NVIDIA 的 NVFP4 模型卡在部署指令裡
 直接帶了 <code>--kv-cache-dtype fp8</code>，把 KV 從 2 bytes 降到 1 byte。</p>
 <p style="font-size:13.2px">122B：{Q35.kv_bytes_per_token('bf16')/KiB:.0f} →
 <b>{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB / token</b>，等於同樣的 HBM 能放兩倍的 context。</p>''',
 "mem")}
 {pane("SSM state 不吃量化的紅利", f'''<p style="font-size:13.2px">GDN 的 recurrent state 由
 <code>mamba_ssm_dtype: float32</code> 決定，是 <b>FP32</b>，與權重量化無關。</p>
 <p style="font-size:13.2px">兩個模型都是 {mib(Q35.ssm_recurrent_bytes(),0)} MiB / 序列。
 想省的話要另外設 <code>--mamba-ssm-cache-dtype</code>，但要驗證品質。</p>''', "ssm")}
 {pane("被忽略的一塊：MTP head", f'''<p style="font-size:13.2px">122B 的 MTP head 有
 {Q35.params_mtp/1e9:.2f} B 參數而且<b>沒有被量化</b>，BF16 佔
 {gb(Q35.params_mtp*2)} GB。</p>
 <p style="font-size:13.2px">只要開 speculative decoding，這 {gb(Q35.params_mtp*2)} GB
 就從 cache 預算裡扣掉，相當於少放 {Q35.params_mtp*2/Q35.seq_bytes(8192,'fp8',3):.0f} 條 8k 序列。</p>''',
 "spec")}
</div>""",
      notes="""<p>三張表的重點：<br>
① 權重省下的空間<b>全部</b>變成 cache 空間，而 cache 空間直接等於併發數。<br>
② KV cache 量化是另一個獨立旋鈕，效果同樣大。<br>
③ MTP head 沒被量化這件事在官方文件裡沒有明說，是從 weight map 推出來的。</p>""")

slide("精度代價", "NVIDIA 官方的評測：NVFP4 對 FP8 的差距在 0.6 分以內",
      sources=["nv4"], accent="ok", body=f"""
{gist("五項評測裡四項小輸、一項小贏，最大差距 0.60 分。以 4× 的記憶體節省來說，這個代價很划算。")}
<div class="cols w6-4">
 <div>
  {table(["Benchmark", "~FP8 baseline", "~NVFP4", "~差距"], [
   ["MMMU Pro", "75.90", "75.55", "<span style='color:var(--bad)'>−0.35</span>"],
   ["GPQA Diamond", "87.37", "86.77", "<span style='color:var(--bad)'>−0.60</span>"],
   ["SciCode", "42.16", "41.79", "<span style='color:var(--bad)'>−0.37</span>"],
   ["AA-LCR", "65.50", "67.13", "<span style='color:var(--ok)'>+1.63</span>"],
   ["IFBench", "70.91", "70.80", "<span style='color:var(--bad)'>−0.11</span>"],
  ], "compact")}
  <div class="hint">數據取自 nvidia/Qwen3.5-122B-A10B-NVFP4 模型卡，
  對照組是同一家的 FP8 版本（不是 BF16）。查核日 {K.CHECK}。</div>
 </div>
 <div>
  {pane("要注意的三件事", '''<ul style="font-size:13.2px">
  <li>對照組是 <b>FP8</b> 不是 BF16。FP8 對 BF16 本身也有一點差距，兩段要分開看。</li>
  <li>這些是<b>單輪</b>評測。多輪 agent 任務的誤差會累積，官方沒有提供這類數據。</li>
  <li>量化對<b>長尾行為</b>（罕見格式、少數語言、極長 context）的影響通常比平均分數大。
   自家的驗收集一定要自己跑一次。</li>
  </ul>''', "ok")}
  {pane("27B 的 FP8", f'''<p style="font-size:13.2px">Qwen 官方模型卡對 FP8 版本的說法是
  「performance metrics nearly identical to those of the original model」，
  沒有附評測表。</p>
  <p style="font-size:13.2px">FP8 blockwise-128 在業界已算成熟做法，
  但同樣建議自己跑一次驗收集。</p>''', "ssm")}
 </div>
</div>
{quote("Weights and activations of the linear operators within transformer blocks in MoE are quantized.",
       "nvidia/Qwen3.5-122B-A10B-NVFP4 模型卡")}""",
      notes="""<p>態度很簡單：量化的精度風險是<b>可量測</b>的，別用感覺討論。
建議的驗收流程：拿一組自家的 200–500 題代表性任務，BF16 / FP8 / NVFP4 各跑一次，
看分數差與輸出格式合規率。</p>
<p>引文那句話也解釋了為什麼只有 routed expert 被量化：官方寫得很清楚，
量化範圍是 MoE 內部 transformer block 的線性算子。</p>""")

slide("量化不會移動屋頂線的轉折點", "它讓兩邊同時變快，你的相對位置沒動",
      sources=["b200l", "nvfp4"], accent="compute", body=f"""
{gist(f"BF16、FP8、NVFP4 的 N* 都是 {KNEE:.0f}。量化把讀取和運算<b>同時</b>加速，比值不變。")}
<div class="cols">
 {pane("三種精度、同一個轉折點", f'''
 <div class="formula" style="font-size:13.5px">N* = F · b / (2 · BW)</div>
 {table(["~精度", "~b（B/參數）", "~F（PFLOP/s）", "~N* token/step"], [
   ["BF16", "2", "2.25", f"<b>{GPU.knee_tokens('bf16'):.0f}</b>"],
   ["FP8", "1", "4.5", f"<b>{GPU.knee_tokens('fp8'):.0f}</b>"],
   ["NVFP4", "0.5", "9.0", f"<b>{GPU.knee_tokens('nvfp4'):.0f}</b>"],
 ], "compact")}
 <p style="font-size:13.2px;margin-top:8px">Blackwell 每把位寬砍半就把張量核心吞吐加倍。
 分子的 F 加倍、b 減半，剛好相消。</p>''', "compute")}
 {pane("那量化到底改善了什麼", f'''<ul style="font-size:13.3px">
 <li><b>絕對速度</b>：同一個 batch 下，step 時間變短。122B 從 BF16 的
  {gb(Q35.real_bf16_bytes)} GB 降到 {gb(Q35.real_weight_bytes)} GB，
  單就權重讀取就快 {Q35.real_bf16_bytes/Q35.real_weight_bytes:.1f} 倍。</li>
 <li><b>裝得下</b>：122B 從必須兩卡變成單卡，省掉 TP 的 all-reduce。</li>
 <li><b>cache 空間</b>：省下來的 HBM 全部變成併發數。</li>
 <li><b>沒有改善</b>：你還是 memory-bound，還是需要靠 batch 或 speculative decoding
  才能把算力用起來。</li>
 </ul>''', "mem")}
</div>
{take(f"所以量化與 speculative decoding 是<b>互補</b>而不是替代：量化讓每一步更便宜，"
      f"speculative decoding 讓每一步做更多事。兩個都做才會同時改善 TPOT 與 throughput。")}""",
      notes=f"""<p>這一頁把 CH1 的核心洞見與本章接起來。常見的錯誤預期是
「上了 FP4 之後 GPU 就會被算力綁住」，實際上不會。</p>
<p>唯一會移動 N* 的，是硬體本身的 FLOP/byte 比。B200 是 {GPU.flops['fp8']/GPU.bw:.0f} FLOP/byte
（FP8 dense），所以 N* = {KNEE:.0f}。換到別的卡就要重算。</p>""")

# =============================================== CH6 記憶體帳本 ===========
chapter("CH6 單卡記憶體帳本", "mem")

slide("單卡記憶體帳本", kind="section", accent="mem", body=f"""
<div class="no">CHAPTER 06</div>
<h1>單卡記憶體帳本</h1>
<div class="csub">180 GB 的 HBM 要同時放權重、起草器、KV cache、SSM state 與 activation。
混合模型的關鍵在於：<b>KV cache 與 SSM state 必須分開算</b>，它們的成長曲線完全不同。</div>
<ul>
 <li>兩種 cache、兩種成長曲線</li>
 <li>KV cache 的公式與逐模型數字</li>
 <li>SSM cache 的公式與逐模型數字（含 speculative decoding 的影響）</li>
 <li>交會點：context 多長時 KV 才追上 SSM</li>
 <li>互動試算：B200 單卡的完整帳本</li>
</ul>""",
      notes="<p>這一章就是把 SSM cache 跟 KV cache 分開算。</p>")

slide("兩種 cache、兩種成長曲線", "一條斜線，一條水平線",
      sources=["c35", "c38", "mut", "hyb"], accent="mem", body=f"""
{gist("KV cache 隨 context 線性成長；SSM state 是常數。混合模型在短 context 時，記憶體其實由 SSM 主導。")}
{fig("twocache", F2.fig_two_caches(), 4, [
 "縱軸是每條序列佔用的記憶體，兩軸都是對數刻度。",
 "灰線是 Qwen3-32B（純 Transformer）：全程被 KV cache 支配，斜率最陡。",
 "藍線與橘線是兩個混合模型：短 context 時幾乎是水平的。",
 f"因為它們各有一條固定的 SSM state 水平線（約 {mib(Q35.ssm_bytes(3),0)} MiB），"
 "在短 context 時佔掉絕大部分。",
 "兩條線的交會點就是「KV 開始接管」的位置。"], accent="mem")}""",
      notes=f"""<p>這張圖是這一章的地圖。三個觀察：<br>
① 純 Transformer 沒有起跳成本，但斜率很陡。<br>
② 混合模型有一個約 150 MiB 的固定起跳成本，之後斜率平緩很多。<br>
③ 所以<b>短對話</b>的場景，混合模型的併發上限反而由 SSM state 決定，
不在 KV cache。這一點跟直覺相反，第一版報告也漏掉了。</p>""")

slide("KV cache 的容量計算", "先算每條序列，再考慮並行、分片與配置開銷",
      sources=["c35", "c38", "c332", "nv4"], accent="mem", body=f"""
{gist("只有 full attention 層需要 KV。混合架構把層數從 64 降到 12–16，直接把 KV 砍掉 3/4。")}
<div class="formula">M<sub>KV</sub> = 2 × <em>L<sub>full</sub></em> × <span class="c2">H<sub>KV</sub></span>
 × d<sub>head</sub> × bytes × <span class="c3">T</span>
 　　（2 = K 與 V 各一份）</div>
{table(["模型", "~L_full", "~H_KV", "~d_head", "~KV/token BF16", "~KV/token FP8",
        "~@32k (FP8)", "~@262k (FP8)"], [
 [f"{Q35.label} <span class='tagc e'>NVFP4</span>", Q35.n_full, Q35.n_kv, Q35.head_dim,
  f"{Q35.kv_bytes_per_token('bf16')/KiB:.0f} KiB",
  f"<b>{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB</b>",
  f"{32768*Q35.kv_bytes_per_token('fp8')/MiB:,.0f} MiB",
  f"{262144*Q35.kv_bytes_per_token('fp8')/GiB:,.2f} GiB"],
 [f"{Q38.label} <span class='tagc s'>FP8</span>", Q38.n_full, Q38.n_kv, Q38.head_dim,
  f"<b>{Q38.kv_bytes_per_token('bf16')/KiB:.0f} KiB</b>",
  f"{Q38.kv_bytes_per_token('fp8')/KiB:.0f} KiB",
  f"{32768*Q38.kv_bytes_per_token('fp8')/MiB:,.0f} MiB",
  f"{262144*Q38.kv_bytes_per_token('fp8')/GiB:,.2f} GiB"],
 [f"{Q332.label} <span class='tagc n'>BF16</span>", Q332.n_full, Q332.n_kv, Q332.head_dim,
  f"<b>{Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB</b>",
  f"{Q332.kv_bytes_per_token('fp8')/KiB:.0f} KiB",
  f"{32768*Q332.kv_bytes_per_token('fp8')/GiB:,.2f} GiB", "超過原生 context"],
], "compact", hi=(0, 1))}
<div class="cols3" style="margin-top:2px">
 {pane("粗體的是本次的預設", f'''<p style="font-size:13.2px">122B 依 NVIDIA 模型卡的部署指令用
 <code>--kv-cache-dtype fp8</code>；27B 沒有這個建議，預設 auto（BF16）。
 試算器可以切換兩者比較。</p>''', "mem")}
 {pane("公式之外還要加什麼", '''<ul style="font-size:13px">
  <li>PagedAttention 的最後一塊尾巴（block 內部碎片）</li>
  <li>混合模型撐大的 page size 造成的額外碎片</li>
  <li>preempt 時的搬移緩衝</li>
 </ul><p style="font-size:12.8px;color:var(--mut)">實務上抓公式值的 1.05–1.15 倍。</p>''')}
 {pane("多卡時怎麼變", '''<p style="font-size:13.2px">Tensor parallel 會把 KV head 切開：
 TP=2 時每張卡的 KV/token 減半。但 122B 只有 2 組 KV head，
 <b>TP 最多只能切到 2</b>，再多就得複製。這是 GQA 的副作用。</p>''', "compute")}
</div>""",
      notes="""<p>公式要逐項唸一次，特別是 2 的來源（K 和 V 各一份）。</p>
<p>那個「TP 最多切到 2」很重要：122B 的 num_key_value_heads = 2，
所以 TP=4 或 TP=8 時 KV head 不夠切，vLLM 會複製 KV head，
每張卡的 KV cache 不會再變小。這也是 NVFP4 單卡部署更划算的理由之一。</p>""")

slide("SSM cache 的容量計算", "固定大小的 recurrent 矩陣，加一小段 convolution 歷史",
      sources=["mut", "gdnc", "c35", "c38"], accent="ssm", body=f"""
{gist(f"每條序列固定 {mib(Q35.ssm_recurrent_bytes(),0)} MiB 的 FP32 矩陣，與 context 長度完全無關。"
      f"conv state 很小，但會隨 speculative decoding 的 k 變長。")}
<div class="formula">M<sub>state</sub> = <em>L<sub>GDN</sub></em> × <span class="c2">H<sub>V</sub></span>
 × d<sub>V</sub> × d<sub>K</sub> × 4 B
 M<sub>conv</sub> = <em>L<sub>GDN</sub></em> × conv_dim × (conv_kernel − 1 + <span class="c3">k<sub>spec</sub></span>) × 2 B</div>
{table(["模型", "~L_GDN", "~H_V", "~d_V × d_K", "~recurrent (FP32)", "~conv 無 spec",
        "~conv 有 spec", "~合計 / 序列"], [
 [f"{Q35.label}", Q35.n_gdn, Q35.nv, f"{Q35.dv} × {Q35.dk}",
  f"<b>{mib(Q35.ssm_recurrent_bytes(),1)} MiB</b>",
  f"{mib(Q35.ssm_conv_bytes(0),2)} MiB",
  f"{mib(Q35.ssm_conv_bytes(Q35.spec_k),2)} MiB <span class='tagc p'>k=3</span>",
  f"<b>{mib(Q35.ssm_bytes(Q35.spec_k),1)} MiB</b>"],
 [f"{Q38.label}", Q38.n_gdn, Q38.nv, f"{Q38.dv} × {Q38.dk}",
  f"<b>{mib(Q38.ssm_recurrent_bytes(),1)} MiB</b>",
  f"{mib(Q38.ssm_conv_bytes(0),2)} MiB",
  f"{mib(Q38.ssm_conv_bytes(Q38.spec_k),2)} MiB <span class='tagc p'>k=7</span>",
  f"<b>{mib(Q38.ssm_bytes(Q38.spec_k),1)} MiB</b>"],
], "compact")}
<div class="cols3" style="margin-top:2px">
 {pane("144 MiB 的巧合", f'''<p style="font-size:13.2px">兩個模型的 recurrent state 竟然一樣大：
 122B 是 {Q35.n_gdn}×{Q35.nv} = {Q35.n_gdn*Q35.nv:,}，
 27B 是 {Q38.n_gdn}×{Q38.nv} = {Q38.n_gdn*Q38.nv:,} 個 value head，
 每個都是 128×128×4 B = 64 KiB。</p>
 <p style="font-size:13.2px">好記，但也提醒：<b>SSM state 與模型大小無關</b>，
 只跟 GDN 層數 × head 數有關。</p>''', "ssm")}
 {pane("為什麼是 FP32", f'''<p style="font-size:13.2px">config 的
 <code>mamba_ssm_dtype: "float32"</code>。recurrent 狀態會被反覆乘加上千次，
 用 BF16 累積會漂移。</p>
 <p style="font-size:13.2px">vLLM 有 <code>--mamba-ssm-cache-dtype</code> 可以改成 BF16，
 直接砍半成 {mib(Q35.ssm_recurrent_bytes()/2,0)} MiB，但長 context 的品質要先驗過。</p>''',
 "compute")}
 {pane("spec decoding 的影響", f'''<p style="font-size:13.2px">vLLM 的
 <code>conv_state_shape = (conv_dim, conv_kernel − 1 + num_spec)</code>。</p>
 <p style="font-size:13.2px">27B 開 DFlash2（k=7）之後，conv 長度從 3 變成 10，
 conv state 從 {mib(Q38.ssm_conv_bytes(0),2)} 漲到 {mib(Q38.ssm_conv_bytes(7),2)} MiB。
 佔比不大，但「開 spec 會多吃記憶體」有一部分就是它。</p>''', "spec")}
</div>""",
      notes="""<p>兩條公式都寫出來了。conv state 那條裡的 <code>num_spec</code> 是 CH7 的伏筆。</p>
<p>要強調 SSM state 是「每條序列一份」，不是「每 token 一份」。
所以它跟 <code>--max-num-seqs</code> 直接相乘，跟 <code>--max-model-len</code> 無關。</p>""")

slide("交會點：context 多長時 KV 才追上 SSM", "短對話的併發上限其實由 SSM state 決定",
      sources=["c35", "c38", "mut"], accent="ssm", body=f"""
{gist(f"122B 要到 {Q35.crossover_tokens('fp8', 3):,.0f} token、27B 要到 "
      f"{Q38.crossover_tokens('bf16', 7):,.0f} token，KV cache 才追上固定的 SSM state。")}
{stats([
 (f"{Q35.crossover_tokens('fp8', 3):,.0f}", "tok", "122B（FP8 KV）的交會點", "e"),
 (f"{Q38.crossover_tokens('bf16', 7):,.0f}", "tok", "27B（BF16 KV）的交會點", "s"),
 (f"{Q38.crossover_tokens('fp8', 7):,.0f}", "tok", "27B 若改用 FP8 KV", "m"),
 ("0", "tok", "Qwen3-32B：沒有 SSM state，從第一個 token 就是 KV", "n"),
])}
{table(["情境", "典型 context", "122B 每條序列", "27B 每條序列", "誰主導"], [
 ["短問答 / 分類", "1k", f"{mib(Q35.seq_bytes(1024,'fp8',3),0)} MiB",
  f"{mib(Q38.seq_bytes(1024,'bf16',7),0)} MiB",
  "<span class='tagc s'>SSM state 佔 90%+</span>"],
 ["一般對話", "8k", f"{mib(Q35.seq_bytes(8192,'fp8',3),0)} MiB",
  f"{mib(Q38.seq_bytes(8192,'bf16',7),0)} MiB",
  "<span class='tagc s'>SSM 仍佔多數（122B）</span>"],
 ["RAG / 長文件", "32k", f"{mib(Q35.seq_bytes(32768,'fp8',3),0)} MiB",
  f"{mib(Q38.seq_bytes(32768,'bf16',7),0)} MiB",
  "<span class='tagc m'>KV 開始主導</span>"],
 ["Agent 長對話", "128k", f"{mib(Q35.seq_bytes(131072,'fp8',3),0)} MiB",
  f"{mib(Q38.seq_bytes(131072,'bf16',7),0)} MiB",
  "<span class='tagc m'>KV 完全主導</span>"],
 ["跑滿原生 context", "262k", f"{mib(Q35.seq_bytes(262144,'fp8',3),0)} MiB",
  f"{mib(Q38.seq_bytes(262144,'bf16',7),0)} MiB",
  "<span class='tagc m'>KV 完全主導</span>"],
], "compact")}
{take("實務結論：如果你的 workload 是短對話，把 <code>--max-model-len</code> 調小"
      "<b>不會</b>提高多少併發，因為瓶頸在 SSM state；這時要調的是 "
      "<code>--max-num-seqs</code> 或考慮把 SSM state 降成 BF16。"
      "反之如果是長文件場景，KV 的精度與 context 上限才是主要旋鈕。")}""",
      notes="""<p>這一頁把「該調哪個旗標」直接接到 workload 上，最好用。</p>
<p>可以現場算給聽眾看：122B 在 8k context 下，一條序列 197 MiB，
其中 149 MiB 是 SSM state（76%）。把 max_model_len 從 262k 降到 8k
不會改變這 149 MiB，只會影響 vLLM 預留的上限。</p>""")

slide("互動試算：B200 單卡的完整帳本", "自己調參數看併發上限怎麼變",
      sources=["b200l", "nv4", "f38", "mut"], accent="mem", body=f"""
{gist("權重 + 起草器 + KV + SSM + activation 必須全部塞進 gpu_memory_utilization 圈起來的空間。")}
{widget("budget")}
<div class="cols3" style="margin-top:4px">
 {pane("調哪個旋鈕影響最大", table(["旋鈕", "效果"], [
   ["<code>--kv-cache-dtype fp8</code>", "KV 直接砍半；長 context 場景收益最大"],
   ["<code>--max-num-seqs</code>", "同時乘上 KV 與 SSM，短對話場景的主要旗標"],
   ["關掉 speculative decoding", "省下起草器權重與加長的 conv state"],
 ], "compact"))}
 {pane("這個試算沒有算進去的", '<ul style="font-size:12.8px">'
   '<li>PagedAttention 最後一塊 block 的內部碎片</li>'
   '<li>混合模型撐大 page size 造成的碎片</li>'
   '<li>CUDA context 與 allocator 保留區</li></ul>'
   '<p style="font-size:12.6px;color:var(--mut)">實務上把公式值乘 1.05–1.15 比較保險。</p>',
   "mem")}
 {warn("這是<b>記憶體</b>上限，不等於「能達成 SLO 的併發數」。能承諾的容量由 CH9 的開環測試決定，"
       "那個數字通常小得多。")}
</div>""",
      notes=f"""<p>示範順序建議：<br>
① 預設（122B、FP8 KV、開 spec、8k、32 併發）→ 看 SSM 那條佔多大。<br>
② 把 context 拉到 262k → KV 立刻爆掉，剩餘變負。<br>
③ 關掉 spec → 省下 {gb(Q35.params_mtp*2)} GB。<br>
④ 切到 27B → 權重只剩 {gib(Q38.real_weight_bytes)} GiB，cache 空間變成兩倍多。<br>
⑤ 把 KV 切到 BF16 → 看 27B 的 KV 直接翻倍。</p>
<p>activation 的估計是教學用途；實際由 vLLM 依 max_num_batched_tokens、
CUDA graph 捕捉的 batch 尺寸與 kernel workspace 決定，開機時 log 會印出實際保留量。</p>""")

slide("併發上限與 context 的取捨", "同一張卡，你可以選「多人短對話」或「少人長文件」",
      sources=["c35", "c38"], accent="mem", body=f"""
{gist("HBM 是固定的：併發數 × 每條序列的記憶體 ≤ cache 預算。這是一條雙曲線，沒有免費的午餐。")}
{widget("cache")}
<div class="cols3" style="margin-top:2px">
 {pane("122B @ 單卡 B200", f'''<p style="font-size:13px">cache 預算約 {M.budget(Q35, draft=True)['cache']:.0f} GiB（已扣起草器）</p>
 <table class="compact"><tbody>
 <tr><td>4k</td><td class="n">{M.max_seqs(Q35, 4096, num_spec=3, draft=True):,.0f} 條</td></tr>
 <tr><td>32k</td><td class="n">{M.max_seqs(Q35, 32768, num_spec=3, draft=True):,.0f} 條</td></tr>
 <tr><td>262k</td><td class="n">{M.max_seqs(Q35, 262144, num_spec=3, draft=True):,.0f} 條</td></tr>
 </tbody></table>''', "moe")}
 {pane("27B @ 單卡 B200", f'''<p style="font-size:13px">cache 預算約 {M.budget(Q38, draft=True)['cache']:.0f} GiB（已扣起草器）</p>
 <table class="compact"><tbody>
 <tr><td>4k</td><td class="n">{M.max_seqs(Q38, 4096, num_spec=7, draft=True):,.0f} 條</td></tr>
 <tr><td>32k</td><td class="n">{M.max_seqs(Q38, 32768, num_spec=7, draft=True):,.0f} 條</td></tr>
 <tr><td>262k</td><td class="n">{M.max_seqs(Q38, 262144, num_spec=7, draft=True):,.0f} 條</td></tr>
 </tbody></table>''', "ssm")}
 {warn("這些是<b>記憶體</b>能容納的序列數上限，不是「能達成 SLO 的併發數」。"
       "能承諾的容量還受延遲限制，CH9 的開環測試會找出那個更小的數字。"
       "記憶體上限通常遠大於 SLO 上限。")}
</div>""",
      notes="""<p>最後那個警告很重要：很多人把「max_num_seqs 設多少」當成容量，
其實記憶體上限往往是 SLO 上限的好幾倍。把 max_num_seqs 開到記憶體上限，
只會讓延遲爆掉、佇列變長，goodput 反而下降。</p>
<p>正確做法：記憶體上限決定<b>硬性天花板</b>，SLO 測試決定<b>實際設定值</b>。</p>""")

slide("同一顆模型，兩個引擎怎麼記帳", "vLLM 一個統一池，SGLang 兩個獨立池",
      sources=["own", "hyb", "mut"], accent="mem", body=f"""
{gist("兩邊 log 印的「KV cache」不是同一個東西：vLLM 把 GDN state 混進同一個 pool 用 token 計價，"
      "SGLang 把 KV 與 SSM 分成兩個池分開配。數字不能直接相比。")}
<div class="cols">
 {pane("vLLM：統一 page，一個數字", f'''
 <p style="font-size:13.2px">把 attention 的 <code>block_size</code> 撐大到 page ≥ mamba state，
 兩種 cache 用同樣大小的 page 共享一個 block pool。所以 log 只印一行
 <code>GPU KV cache size: N tokens</code>——<b>GDN state 已經混在裡面</b>。</p>
 <table class="compact"><tbody>
 <tr><td>block size</td><td class="n">832 tokens</td></tr>
 <tr><td>mamba page padding</td><td class="n">1.71%</td></tr>
 <tr><td>池子（TP=2 合計）</td><td class="n">{141.3 * 2:.0f} GiB</td></tr>
 <tr><td>印出來的容量</td><td class="n">3,339,380 tokens</td></tr>
 <tr><td>262K 併發</td><td class="n"><b>12.74×</b></td></tr>
 </tbody></table>
 <p style="font-size:12.6px;color:var(--mut)">用本章的 SSM 公式反算 mamba page，
 只有 block size 832 能對上 1.71% 這個 padding——等於替公式做了一次實機驗算。</p>''', "compute")}
 {pane("SGLang：兩個池，分開印", f'''
 <p style="font-size:13.2px"><code>token_to_kv_pool</code> 按 <b>token</b> 配，
 mamba/SSM pool 按 <b>槽位</b> 配（每槽 = 一條序列的完整 state，與 context 無關）。
 所以會看到兩個獨立的數字。</p>
 <table class="compact"><tbody>
 <tr><td>KV 池</td><td class="n">31 GiB（508,278 tokens）</td></tr>
 <tr><td>SSM 池</td><td class="n">97 GiB</td></tr>
 <tr><td>換算槽位</td><td class="n">≈ {97 * M.GiB / Q38.ssm_bytes(0):.0f} 條</td></tr>
 <tr><td>262K 併發（KV 限）</td><td class="n"><b>508,278 ÷ 262,144 = 1.94</b></td></tr>
 </tbody></table>
 <p style="font-size:12.6px;color:var(--mut)">同一顆 27B、單卡、沒開 spec。
 兩個池<b>不能互相調度</b>，所以真正的上限是兩者取小。</p>''', "ssm")}
</div>
{warn(f"<b>SGLang 這組的配比是歪的。</b>SSM 池預留了 {97 * M.GiB / Q38.ssm_bytes(0):.0f} "
      f"條序列的空間，但 KV 只剩 31 GiB —— 跑滿 262K 時只放得下 <b>1.9 條</b>。"
      f"把槽位砍到接近實際併發（例如 64 條，約 {64 * Q38.ssm_bytes(0) / M.GiB:.1f} GiB），"
      f"省下的還給 KV，同一張卡的 262K 併發可以拉到約 "
      f"{(31 + 97 - 64 * Q38.ssm_bytes(0) / M.GiB) * M.GiB / (262144 * Q38.kv_bytes_per_token('bf16')):.1f} 條。"
      "vLLM 的統一池則是自動按需分配：不用調，但也看不出哪一塊吃掉了記憶體。")}""",
      notes=f"""<p>這一頁是實機資料，來源是使用者提供的兩份啟動 log（vLLM 0.26.1 TP=2、
SGLang 單卡）。它同時是本章公式的<b>實機驗算</b>：用投影片上的 SSM 公式算出的 mamba page
是 1,675,264 B，vLLM 選 block size 832 讓 attention page = 1,703,936 B，
兩者差 1.712%——log 印的就是 1.71%。</p>
<p>要強調的實務結論有兩點：<br>
① 比較兩個引擎時，先問「它的 KV cache 數字含不含 SSM」，再問 TP 與 spec 開關，否則是雞同鴨講。<br>
② SGLang 這種雙池設計，配比要自己按 workload 調。長 context 就少給槽位、多給 KV；
短對話反過來。這組設定顯然是照預設值跑的。</p>""")

# ============================================ CH7 Speculative decoding ====
chapter("CH7 Speculative decoding 對 serving 的影響", "spec")

slide("Speculative decoding 對 serving 的影響", kind="section", accent="spec", body=f"""
<div class="no">CHAPTER 07</div>
<h1>Speculative decoding<br>對 serving 的影響</h1>
<div class="csub">本次上線的設定：<b>Qwen3.5-122B-A10B 用內建 MTP 猜 3 個</b>、
<b>Qwen3.8-27B 用 DFlash2 猜 7 個</b>。這一章把它從「一個開關」拆成
機制、接受率、記憶體、批次上限與 SLO 五個面向。</div>
<ul>
 <li>為什麼可以猜：decode 有閒置算力</li>
 <li>draft → verify → accept / reject 的一輪</li>
 <li>MTP 與 DFlash2 的機制差異與時間軸</li>
 <li>接受長度 τ：報酬遞減得多快</li>
 <li>(k+1)× 的 token 撞上屋頂線 → 併發上限</li>
 <li>長 context 會吃掉接受率（實測案例）</li>
 <li>記憶體成本、與 prefix caching 的衝突、什麼時候該關掉</li>
</ul>""",
      notes="<p>這一整章是第二版新增的，大概也是最實用的一章。</p>")

slide("為什麼可以「猜」", "decode 時算力大量閒置，那塊算力是免費的",
      sources=["spd", "sspec"], accent="spec", body=f"""
{gist(f"32 併發時，27B 只用掉不到 15% 的張量核心算力。speculative decoding 就是把那 85% 拿去猜 token。")}
{fig("specwhy", F3.fig_spec_why(), 3, [
 "decode 的兩種資源使用率差距很大：頻寬被打滿，算力幾乎閒著。",
 "粉紅色那塊算力，不用也是浪費掉。",
 "所以：拿它去猜 k 個 token，再讓目標模型用<b>同一次</b>權重讀取一口氣驗證。",
 "輸出品質不變（rejection sampling 保證），但那塊算力有限，"
 "本章要量的就是那條界線。"], accent="spec")}""",
      notes=f"""<p>把 CH1 的屋頂線接回來：只要 N &lt; {KNEE:.0f}，加 token 幾乎不用付錢。
speculative decoding 就是「用同一次權重讀取，多送幾個 token 進去」的另一種辦法，
跟加大 batch 是同一件事的兩種做法。</p>
<p>差別在於：加大 batch 需要有更多使用者；speculative decoding 在<b>只有少數使用者</b>時
也能把 N 撐大。所以它是低併發、低延遲場景的利器。</p>""")

slide("一輪 draft → verify → accept", "被拒絕之後整串丟掉，但一定會前進至少一格",
      sources=["spd", "eg3"], accent="spec", body=f"""
{gist("起草 k 個 → 目標模型一次驗證 k+1 個位置 → 從左到右接受，遇到第一個拒絕就停，並補上一個保底 token。")}
{fig("speccyc", F3.fig_spec_cycle(), 5, [
 "已經確定的輸出擺在左邊。",
 "<b>起草</b>：用便宜的方式產生 k = 7 個候選 token。",
 "<b>驗證</b>：目標模型一次 forward 同時算出 8 個位置的正確分布。",
 "<b>接受</b>：由左往右逐位置判定。第 4 個開始被拒絕，後面整串丟掉。",
 "拒絕的位置改用目標模型自己的輸出（<b>保底 token</b>），所以一定前進至少 1 格。",
 "這一輪前進 4 格，只花了 1 次起草 + 1 次目標模型 forward。"], accent="spec")}""",
      notes="""<p>三個常見誤解要澄清：<br>
① <b>「猜錯會讓品質變差」</b>：不會。rejection sampling 在數學上保證輸出分布不變，
vLLM 文件的說法是「theoretically lossless up to the precision limits of hardware numerics」。<br>
② <b>「最壞情況會比不猜慢」</b>：就 <b>token 數</b>而言不會（一定前進 1 格），
但就<b>時間</b>而言會，因為多付了起草成本與驗證的額外計算。<br>
③ <b>「被拒絕的位置後面那些草稿還可以留著」</b>：不行，它們是以錯誤前綴為條件產生的。</p>""")

slide("MTP 與 DFlash2：兩種完全不同的起草器",
      "一個是序列式跑 k 次，一個是一次猜完整塊",
      sources=["m35", "df2", "dfp", "dfn", "eg3"], accent="spec", body=f"""
{gist("MTP 的起草成本隨 k 線性成長；DFlash2 的 block diffusion 一次 forward 出整塊，成本與 k 無關。")}
{fig("spectl", F3.fig_spec_timeline(), 6, [
 "同樣要產生草稿，兩種起草器的時間軸完全不同。",
 "<b>122B 的 MTP</b>：1 層 decoder head，autoregressive 跑 k 次。",
 "然後目標模型一次驗證 k+1 = 4 個位置。",
 "<b>27B 的 DFlash2</b>：block diffusion，一次 forward 就把整塊 8 個位置都猜出來。",
 "接著驗證 8 個位置。",
 "所以 MTP 的 T_draft = k × t_step，DFlash2 的 T_draft = t_parallel（與 k 無關）。",
 "DFlash2 敢用 k = 7、MTP 通常停在 k = 2–4，差別就在這裡。"], accent="spec")}""",
      notes="""<p>DFlash 論文的說法：drafting cost 是 T_draft = t_parallel，
相對於 autoregressive drafting 的 T_draft = γ·t_step。
它用非因果 attention 讓每個 query 同時看到目標模型的 hidden state 與所有 mask token，
一次產生整塊草稿，再用一個輕量 selector（selector_rank 256、top-k 16）挑一條連貫的路徑。</p>
<p>另外 DFlash 把目標模型 5 層的 hidden state 注入到起草器每一層的 KV
（config 裡的 <code>target_layer_ids: [5, 19, 33, 47, 61]</code>），
這讓接受長度能隨草稿深度成長，而不是很快飽和。</p>""")

slide("兩個起草器的規格對照", "本次上線設定的完整參數",
      sources=["m35", "df2", "dfp", "c35", "c38"], accent="spec", body=f"""
{gist("122B：內建 MTP、k=3。27B：外掛 DFlash2、block size 8、k=7。兩者的成本結構完全不同。")}
{table(["", f"{Q35.label}", f"{Q38.label}"], [
 ["起草方法", "<b>MTP（multi-token prediction）</b>，EAGLE 系",
  "<b>DFlash2</b>，block diffusion"],
 ["來源", "模型 checkpoint 內含 <code>mtp.*</code>（Qwen 官方訓練）",
  "外部 checkpoint <code>incoai/Qwen3.8-27B-DFlash2</code>"],
 ["起草器規模", f"1 層 MoE decoder，{Q35.params_mtp/1e9:,.2f} B 參數（BF16，未量化）",
  f"5 層 sliding attention，{Q38.spec_draft_params/1e9:,.2f} B 參數（BF16）"],
 ["草稿 token 數 k", "<b>3</b>（官方建議 2–4）", "<b>7</b>（block size 8，扣掉 anchor）"],
 ["起草方式", "autoregressive，跑 k 次 forward", "一次 forward 產生整塊"],
 ["起草成本", "∝ k　（每次都要重讀 lm_head，1.53 GB）",
  f"固定 {Q38.spec_draft_params*2/1e9:,.2f} GB，與 k 無關"],
 ["條件輸入", "目標模型最後一層的 hidden state",
  "目標模型 5 層 hidden state 注入到起草器每層的 KV"],
 ["vLLM 設定", '<code>{"method":"qwen3_next_mtp","num_speculative_tokens":3}</code>',
  '<code>{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}</code>'],
 ["官方實測接受長度", "官方未公布逐 benchmark 數字（本報告取 τ ≈ 2.6 作保守估計）",
  "GSM8K <b>5.46</b>、MATH-500 <b>5.28</b>、MBPP <b>4.79</b>、HumanEval <b>4.39</b>"],
 ["官方實測加速（concurrency 1）", "—",
  "GSM8K <b>3.43×</b>、MATH-500 <b>3.34×</b>、HumanEval <b>3.11×</b>"],
], "compact", hi=(3,))}
{warn("DFlash2 模型卡的吞吐數字是在 concurrency 1 量的，且未載明硬體。"
      "<b>接受長度</b>可以跨硬體沿用（它只跟模型與資料有關），"
      "<b>絕對 tok/s 與加速倍數不行</b>，那會隨 GPU、量化與併發改變。上線前要自己量一次。")}""",
      notes="""<p>這一頁把使用者提供的設定（122B 用 MTP 猜 3、27B 用 DFlash2 猜 7）
變成可查證的規格表。</p>
<p>要特別提醒 k 的選擇邏輯：MTP 因為起草成本 ∝ k，k 大了不划算，
所以官方建議 2–4；DFlash2 因為起草成本固定，把 k 拉到 7 幾乎沒有額外起草成本，
代價是驗證時要多算 token，正好接到下一頁的屋頂線分析。</p>""")

slide("接受長度 τ：報酬遞減得很快", "k 加大不會等比例變好",
      sources=["eg3", "df2", "lat", "sspec"], accent="spec", body=f"""
{gist("每個草稿位置能被接受的機率是 α^i，所以 τ = (1−α^(k+1))/(1−α)，很快就飽和。")}
{fig("accept", F3.fig_accept(), 4, [
 "第 i 個草稿位置要被接受，前面 i−1 個都得先對，所以機率是 α^i。",
 "α 越小衰減越快。α = 0.55 時，第 4 個位置就只剩 9% 的機會。",
 "把每個位置的接受機率加起來就是期望接受長度 τ。",
 "報酬遞減：α = 0.80 時，k 從 7 加到 15，τ 只多 15%，但驗證成本翻倍。",
 "實測參考：DFlash2 在 27B 上的接受長度是 4.4–5.5，對應 α ≈ 0.80–0.86。"],
 accent="spec")}""",
      notes="""<p>公式 τ = Σ_{i=0..k} α^i = (1−α^(k+1))/(1−α)，是 SmartSpec 論文與
「Interpretable Latency Model」論文都用的估計式。</p>
<p>要注意這個式子假設每個位置的接受率相同，實際上會逐位置下降（見兩頁後的實測）。
所以它是<b>樂觀估計</b>；實際的 τ 要從引擎的 acceptance metrics 讀。</p>
<p>vLLM 有 per-request acceptance metrics，上線後應該持續監控 τ，
它是判斷「該不該繼續開 spec」最直接的訊號。</p>""")

slide("(k+1)× 的 token 撞上屋頂線", "這是 speculative decoding 對 serving 最大的影響",
      sources=["sspec", "lat", "b200l"], accent="spec", body=f"""
{gist(f"驗證階段一次要送 B × (k+1) 個 token。一旦超過 N* ≈ {KNEE:.0f}，就從「免費」變成「多花錢」。")}
{fig("specbatch", F3.fig_spec_batch(), 4, [
 "橫軸是併發序列數 B，縱軸是相對不用 spec 的輸出速率倍數。",
 f"<b>27B + DFlash2 (k=7)</b>：B × 8 一旦超過 N* = {KNEE:.0f}（也就是 B ≈ "
 f"{int(KNEE//8)}），驗證變成 compute-bound，加速直線墜落。",
 "<b>122B + MTP (k=3)</b>：形狀完全不同，因為它是 MoE。低併發時 spec 的優勢被"
 "「驗證要碰到更多 expert」吃掉；高併發時 expert 讀取被更多 token 分攤，優勢回升。",
 "紅色虛線是損益兩平線。低於它，speculative decoding 就是純粹在燒算力。"],
 accent="spec")}""",
      notes=f"""<p>這一頁是全章最重要的一張圖，也最直接回答「speculative decoding 對 serving 的影響」。</p>
<p>推導很簡單：驗證階段一次送 N = B × (k+1) 個 token。CH1 已經證明
N &lt; N* 時加 token 幾乎免費、N &gt; N* 時線性收費。所以：<br>
<b>B_安全 ≈ N* / (k+1)</b>。<br>
27B 配 k=7 → B ≈ {int(KNEE//8)}；122B 配 k=3 → B ≈ {int(KNEE//4)}（但 MoE 的修正讓它實際更寬）。</p>
<p>這也解釋了一個常見現象：同一組設定在壓測初期（低併發）看起來很棒，
壓到中段突然變差，就是越過了這條線。</p>
<p>模型的假設要講清楚：這是解析上限，只算權重讀取與張量運算，
不含 attention kernel、MoE routing、all-to-all、取樣與排程開銷。
真實曲線的形狀相同，但轉折會更早、絕對值更低。</p>""")

slide("一次驗證 8 個位置：為什麼可行、什麼時候變貴",
      "驗證是一次迷你 prefill，不是 8 次 decode",
      sources=["spd", "dfp", "sspec"], accent="spec", body=f"""
{gist(f"草稿已經在手上，所以目標模型可以用一般的 causal mask 一次算完 8 個位置的分布——"
      f"共用同一次權重讀取。代價是 B × {Q38.spec_k + 1} 這個數字會去撞 N*。")}
<div class="cols">
 {pane("為什麼 autoregressive 還能一次驗證 8 個", f'''
 <p style="font-size:13.4px">驗證階段目標模型<b>不需要自己生成</b>，它只回答一個問題：
 「如果前綴長這樣，我會吐什麼？」</p>
 <p style="font-size:13.4px">把 anchor 接上 {Q38.spec_k} 個草稿拼成一條序列送進去，
 配 causal mask 一次 forward，第 i 個位置看到的正好是它前面那些 token，
 跟逐步生成看到的<b>完全一樣</b>。</p>
 <p style="font-size:13.4px;color:var(--mut)">所以它是一次<b>迷你 prefill</b>。
 8 個位置的 GEMM 一批做完，權重只讀一次——這就是 N 從 B 變成 B × (k+1) 的由來。
 平均只有 τ ≈ {M.SPEC_TAU["q38"]:.1f} 個位置會被採用，其餘算完就丟。</p>''', "compute")}
 {pane(f"27B 的梯子（k = {Q38.spec_k}）", table(["~B", f"~B × {Q38.spec_k + 1}", "~vs N*", "~位置"], [
   [f"{b}", f"{b * (Q38.spec_k + 1):,}", f"{b * (Q38.spec_k + 1) / KNEE:.2f}×",
    ("<span class='tagc g'>免費區</span>" if b * (Q38.spec_k + 1) < KNEE * .95
     else "<span class='tagc'>碰線</span>" if b * (Q38.spec_k + 1) < KNEE * 1.1
     else "<span class='tagc e'>開始收費</span>")]
   for b in (8, 16, 32, 36, 48, 64, 128)], "compact"), "spec")}
</div>
{warn(f"<b>兩個門檻不要搞混。</b>B ≈ {int(KNEE//8)} 是「額外驗證不再免費」的<b>起點</b>，"
      f"不是「超過就不能開」。真正由賺轉賠是損益兩平點，解析模型算出來是 "
      f"<b>B ≈ {M.spec_break_even(Q38):.0f}</b>：B=32 還有 {M.spec_speedup(Q38, 32):.2f}×、"
      f"B=64 剩 {M.spec_speedup(Q38, 64):.2f}×、B=128 只剩 {M.spec_speedup(Q38, 128):.2f}×。"
      f"實機 max_num_seqs = 60 正好落在這段邊際效益快速遞減的區間，值得逐點 A/B。")}""",
      notes=f"""<p>這一頁補的是聽眾最常卡住的兩個問題。</p>
<p><b>問題一：「autoregressive 不是一個一個生嗎？怎麼一次驗 8 個？」</b>
關鍵在草稿已經產生了，驗證只是把整條候選序列當成 prompt 做一次 causal forward，
同時取出每個位置的 logits。跟 CH1 講 prefill 的道理完全一樣——所以我說它是迷你 prefill。</p>
<p><b>問題二：「那 {int(KNEE//8)} 是不是天花板？」</b>不是。{int(KNEE//8)} 是免費區的邊界，
{M.spec_break_even(Q38):.0f} 才是損益兩平。中間那段還在賺，只是越來越薄。
實機 max_num_seqs 設 60，就落在這一段——所以壓測要在 32 / 48 / 60 三點各做一次開關對照，
不要只測低併發就下結論。</p>""")

slide("互動試算：spec decoding 划不划算", "自己調 k、α 與併發數",
      sources=["sspec", "lat", "df2"], accent="spec", body=f"""
{gist("三個變數決定一切：草稿數 k、逐位置接受率 α、同時併發數 B。")}
{widget("spec")}
<div class="cols3" style="margin-top:4px">
 {pane("建議的操作順序", '<ol style="font-size:12.8px">'
   '<li>先用預設值看 27B + DFlash2 在低併發的 4.4×</li>'
   '<li>把 B 拉到 64 → 加速掉一半</li>'
   '<li>再拉到 200 → TPOT 比不開還差</li>'
   '<li>把 k 降到 3 → 高併發時反而更好</li>'
   '<li>把 α 降到 0.6（模擬長 context）→ 整條塌下來</li></ol>', "spec")}
 {pane("模型的假設", '<ul style="font-size:12.8px">'
   '<li>只算權重讀取與張量運算（roofline 上限）</li>'
   '<li>MoE 的 routing 假設均勻 → 專家碰觸數是上界</li>'
   '<li>每個草稿位置的接受率固定為 α（實際會逐位置下降）</li>'
   '<li>不含 attention kernel、all-to-all、取樣與排程開銷</li></ul>', "mem")}
 {pane("所以怎麼用它", '<p style="font-size:12.8px">當成<b>形狀</b>的指南，不是絕對值的預測。'
   '它能回答「B 大概到哪裡就該關掉」，不能回答「會有幾 tok/s」。</p>'
   '<p style="font-size:12.8px">量過一次實測之後，把「實測 ÷ 解析上限」的比例記下來，'
   '之後就能用它快速估算新設定。</p>', "ok")}
</div>""",
      notes="""<p>示範腳本：<br>
① 預設（27B + DFlash2、k=7、α=0.80、B=16）→ 看到 4.4× 與 TPOT 大幅改善。<br>
② 把 B 拉到 64 → 加速掉到 2.5×。<br>
③ 拉到 200 → 掉到 1× 以下，TPOT 反而比不開還差。<br>
④ 把 k 降到 3 → 在高併發時反而比 k=7 好（因為 (k+1) 較小）。<br>
⑤ 把 α 降到 0.6（模擬長 context）→ 整條曲線塌下來。<br>
⑥ 切到 122B + MTP → 看形狀完全不同。</p>
<p>結論：<b>k 應該隨負載動態調整</b>。vLLM 已有 Dynamic Speculative Decoding
可以依 QPS 調整，SmartSpec 論文的做法是依 goodput 決定每一輪的 k，
高負載時自動歸零。</p>""")

slide("長 context 會吃掉接受率", "實測案例：從加速 129% 變成減速 51%",
      sources=["iss1", "dfp"], accent="bad", body=f"""
{gist("接受率不是常數。同一個模型、同一個 k，context 從 2k 長到 30k，平均接受率從 65% 掉到 39%。")}
{fig("acceptctx", F3.fig_accept_ctx(), 5, [
 "vLLM issue #47602 在 Qwen3.6-27B 上的實測：短 context 時 MTP 帶來 129% 的吞吐提升。",
 "但隨著 context 變長，優勢一路縮小。",
 "約 12k 之後，開 MTP 反而<b>比不開更慢</b>。30k 時慢了 51%。",
 "原因在右邊：逐位置接受率整條下降。第 1 個位置從 93.5% 掉到 72.1%，"
 "第 6 個從 36.6% 掉到 14.0%。",
 "長 context 服務要重新量接受率，短 prompt 的結論搬不過來。"],
 accent="bad")}""",
      notes="""<p>這是全章最該記住的實務警告，而且有公開實測資料撐著。</p>
<p>為什麼會這樣？兩個原因：<br>
① 起草器看到的上下文越長，任務越難、分布越發散，猜中率自然下降。<br>
② context 一長，目標模型自己的每一步也更貴（attention 讀取量隨 T 成長），
但起草器的成本沒有等比例下降，所以相對開銷變高。</p>
<p>DFlash 論文也觀察到同樣現象（base model 的接受長度在 4k 之後開始下降），
他們的解法是針對長 context 微調起草器，能把 16k 的 τ 從 3.61 拉回 6.05。
如果我們的 workload 是長文件，這是一個可以評估的方向。</p>""")

slide("其他三個副作用", "記憶體、prefix caching 與延遲抖動",
      sources=["mut", "iss2", "spd"], accent="spec", body=f"""
{gist("開 spec 不只是多一個開關：它會吃記憶體、干擾 prefix caching，也會讓 ITL 變得不規則。")}
{fig("specmem", F3.fig_spec_mem(), 4, [
 "第一筆：起草器本身的權重。",
 "第二筆：GDN 的 conv state 長度是 conv_kernel − 1 + num_spec，所以 k 越大它越長。",
 "第三筆：每條序列要多預留 k 個 KV slot 給驗證用。",
 "總結：記憶體不是主要限制，但在 cache 已經很緊的長 context 場景，這幾 GB 會直接吃掉併發數。"],
 accent="spec")}
<div class="cols" style="margin-top:2px">
 {pane("干擾 prefix caching", f'''<p style="font-size:13.2px">vLLM issue #38182 回報：
 在 Qwen3.5-35B-A3B 上開 MTP（<code>num_speculative_tokens: 1</code>）之後，
 prefix cache 命中率從 <b>92% 掉到 71%</b>。</p>
 <p style="font-size:13.2px">原因是被拒絕的草稿會讓序列長度不對齊 block 邊界，
 使得可快取的 block 變少。對 agent／多輪對話（本來就靠 prefix caching 吃飯）影響特別大。</p>''',
 "bad")}
 {pane("延遲變得不規則", f'''<p style="font-size:13.2px">沒開 spec 時，每個 step 產生 1 個 token，
 {T("ITL")} 很平穩。開了之後，一輪可能吐 1 個也可能吐 8 個。</p>
 <p style="font-size:13.2px">平均 {T("TPOT")} 改善，但 <b>ITL 的 p99 會變差</b>。
 如果 SLO 是用 ITL p99 定義的，要重新校準門檻，或改用 TPOT。</p>''', "mem")}
</div>""",
      notes="""<p>這三個副作用都不是致命的，但都會讓「開了 spec 之後 benchmark 數字變差」
變得難以解釋。先知道它們存在，排查時就快很多。</p>
<p>特別是 prefix caching 那個：如果你的 workload 前綴重複度高，
關掉 spec 反而可能更快。這一定要 A/B 測。</p>""")

slide("什麼時候該開、什麼時候該關", "一張可以直接用的決策表",
      sources=["sspec", "lat", "spd", "iss1"], accent="spec", body=f"""
{gist("低併發 + 短 context + 高接受率 → 開，而且 k 可以大。高併發或長 context → 調小 k 或關掉。")}
{table(["情境", "建議", "理由"], [
 ["單一使用者互動、demo、內部工具（B ≤ 8）",
  "<span class='tagc g'>全開，k 用最大</span>",
  f"整段都在 memory-bound 區，B×(k+1) 遠低於 {KNEE:.0f}。TPOT 可以改善 2–4×。"],
 [f"中等併發（B ≈ 8–{int(KNEE//8)}）、context &lt; 8k",
  "<span class='tagc g'>開</span>",
  f"仍在 memory-bound 區。27B 配 k=7 可以撐到 B ≈ {int(KNEE//8)}。"],
 [f"高併發（B &gt; {int(KNEE//8)}）、追求整機 throughput",
  "<span class='tagc m'>調小 k，或關掉</span>",
  "驗證階段變成 compute-bound，多猜的 token 都在燒算力。"],
 ["長 context（&gt; 16k）",
  "<span class='tagc r'>先量再決定</span>",
  "接受率會顯著下降，實測案例顯示 30k 時反而慢 51%。"],
 ["前綴重複度高的 agent／多輪對話",
  "<span class='tagc m'>A/B 測 prefix cache 命中率</span>",
  "spec 會拉低命中率，兩個效益可能互相抵消。"],
 ["混合負載、QPS 波動大",
  "<span class='tagc'>用動態 speculative decoding</span>",
  "vLLM 支援依負載調整；SmartSpec 的做法是依 goodput 決定每輪的 k，高負載自動歸零。"],
], "compact")}
<div class="cols3" style="margin-top:2px">
 {pane("上線前一定要量的三個數字", '''<ul style="font-size:13.2px">
  <li>逐位置接受率與 τ（引擎的 acceptance metrics）</li>
  <li>開／關 spec 的 goodput 對照（同一組 SLO）</li>
  <li>prefix cache 命中率的變化</li>
 </ul>''', "ok")}
 {pane("上線後要盯的訊號", '''<ul style="font-size:13.2px">
  <li>τ 隨時間下降 → workload 變了，該重評 k</li>
  <li>ITL p99 惡化但 TPOT 改善 → SLO 定義要調整</li>
  <li>preempt 次數上升 → 起草器吃掉的記憶體造成壓力</li>
 </ul>''', "mem")}
 {pane("本次設定的初步判斷", f'''<p style="font-size:13.2px">122B + MTP k=3：
 <b>建議開啟</b>，MoE 的特性讓它在高併發時仍有優勢。</p>
 <p style="font-size:13.2px">27B + DFlash2 k=7：<b>建議在 B ≤ {int(KNEE//8)} 時開啟</b>；
 高併發批次任務可考慮降到 k=3 或關閉。兩者都要先用自家 workload 量接受率。</p>''', "spec")}
</div>""",
      notes="""<p>這一頁可以直接印出來當 checklist。</p>
<p>要強調最後一格：本報告給的是<b>基於解析模型的初步判斷</b>，
不是實測結論。決策要等 CH9 的兩階段測試跑完，用同一組 SLO 比較開／關的 goodput。</p>""")

slide("上線設定與命令", "可以直接複製的設定",
      sources=["nv4", "df2", "m35", "spd"], accent="spec", body=f"""
{gist("兩組設定都來自官方模型卡，只把草稿數改成本次評估要用的值。")}
<div class="cols">
 {pane("122B · NVFP4 + MTP k=3", code(
"""<span class="c"># 目標模型：NVIDIA 量化版，單卡即可</span>
vllm serve nvidia/Qwen3.5-122B-A10B-NVFP4 \\
  --quantization <span class="s">modelopt_fp4</span> \\
  --kv-cache-dtype <span class="s">fp8</span> \\
  --tensor-parallel-size <span class="n">1</span> \\
  --max-model-len <span class="n">65536</span> \\
  --reasoning-parser qwen3 \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  --speculative-config '<span class="hl">{{
      "method": "qwen3_next_mtp",
      "num_speculative_tokens": 3
  }}</span>'

<span class="c"># 官方模型卡建議 2–4；本次取 3</span>
<span class="c"># 容器：nvcr.io/nvidia/vllm:26.04-py3</span>"""), "moe")}
 {pane("27B · FP8 + DFlash2 k=7", code(
"""<span class="c"># 目標模型：Qwen 官方 FP8 版</span>
vllm serve Qwen/Qwen3.8-27B-FP8 \\
  --max-model-len <span class="n">65536</span> \\
  --kv-cache-dtype <span class="s">auto</span> \\
  --reasoning-parser qwen3 \\
  --speculative-config '<span class="hl">{{
      "method": "dflash",
      "model": "incoai/Qwen3.8-27B-DFlash2",
      "num_speculative_tokens": 7
  }}</span>'

<span class="c"># block size 8 → 7 個草稿 + 1 個 anchor</span>
<span class="c"># SGLang 對應：--speculative-algorithm DFLASH</span>
<span class="c">#              --speculative-num-draft-tokens 8</span>"""), "ssm")}
</div>
{warn("執行前務必先跑一次 <code>vllm serve --help</code> 與 "
      "<code>vllm bench serve --help</code> 核對<b>你安裝的版本</b>的旗標名稱，"
      "speculative decoding 的介面最近幾個版本改動頻繁。"
      "DFlash 在 vLLM 走 Speculators 函式庫，請確認版本支援。")}""",
      notes="""<p>兩段指令的來源：122B 是 NVIDIA 模型卡的部署範例（原本用
<code>--tensor-parallel-size 1</code>，我們保留），加上 Qwen 模型卡的 MTP 設定；
27B 是 incoai 模型卡的 vLLM 範例。</p>
<p><code>--max-model-len</code> 我填 65536 而不是原生的 262144，
因為 CH6 顯示跑滿 262k 只能放個位數的序列。實務上應該依 workload 的
p99 context 長度設定，留一點餘裕即可。</p>""")
