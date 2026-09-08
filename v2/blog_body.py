# -*- coding: utf-8 -*-
# 由 blog.py exec()，共用其命名空間。內容為部落格正文。

# ============================================================== 開場 ======
note(f"""
<h4>先講結論</h4>
<p><b>1.</b> LLM 的 decode 階段卡在記憶體頻寬，不是算力。一次 forward 要把整份權重讀過一遍，
卻只服務 batch 裡那幾個 token。</p>
<p><b>2.</b> 在 B200 上，這件事有一個很乾淨的臨界值：一次 forward 湊到大約
<b>{KNEE:.0f} 個 token</b> 之前，多塞 token 幾乎不用加錢。而且這個數字
<b>不會因為換成 FP8 或 FP4 而改變</b>。</p>
<p><b>3.</b> 混合架構（3 層 Gated DeltaNet 配 1 層 full attention）把 KV cache 砍掉 3/4，
但換來每條序列固定 ~150 MiB 的 SSM state。短對話時，記憶體其實是被這個固定成本吃掉的。</p>
<p><b>4.</b> NVFP4 版的 Qwen3.5-122B-A10B <b>只量化了 routed expert</b>。36 層 GDN、
12 層 attention、lm_head、MTP head 全都還是 BF16，佔了 checkpoint 的 21.9%。
這是我們從 safetensors 的 weight map 直接讀出來的，官方文件沒有明說。</p>
<p><b>5.</b> Speculative decoding 的效益由一個很簡單的式子決定：併發數 × (k+1) 有沒有超過那個
{KNEE:.0f}。27B 配 DFlash2 猜 7 個，安全併發大約到 36；超過就開始賠。</p>
""", "key")

p("這篇是我們在評估 ", "<code>nvidia/Qwen3.5-122B-A10B-NVFP4</code> 與 ",
  "<code>Qwen/Qwen3.8-27B-FP8</code> 這兩個模型要怎麼上線時，整理出來的一份筆記。",
  "原本是投影片，後來發現有些推導在簡報上放不下，就改寫成文章。")

p("我盡量讓每個數字都可以被驗算：架構參數全部來自官方 config，權重大小來自 HuggingFace 上",
  "safetensors 的實際位元組統計，公式都寫在文章裡。文中所有圖表都是可以按的，",
  "捲到畫面中間會自己播一次，也可以用「下一步」自己控制節奏。")

note("""
<p><b>閱讀方式</b>　虛線底的粗體字是專有名詞，滑鼠移過去會出現解釋。
右上角可以關掉動畫自動播放。想看簡報版（82 頁、有講稿）點右上的「簡報版」。</p>
<p><b>數字的可信度</b>　文章裡的數字分三種：<span class="tagc g">實測 / 官方</span>
可以直接引用；<span class="tagc">公式推導</span>公式都印出來了，可以自己算，
標成「解析上限」的實測通常只有它的 30–60%；<span class="tagc m">教學示例</span>
是用排隊理論生出來的示意曲線，不是任何硬體的實測結果。</p>
""")

# ================================================== 1. 一次 forward ======
h2("PART 01", "一次 forward 到底在做什麼", "part1")

p("先把最基本的事情講清楚。語言模型產生文字的方式是 ", T("Autoregressive", "自回歸"),
  "：跑一次 forward，得到「下一個 token」的機率分布，抽一個出來，接回輸入尾端，再跑一次。",
  "要生 500 個 token，就要跑 500 次 forward。", cite("q3r"))

figure("autoreg", F1.fig_autoregress(), 5, [
    "輸入的 token id 先過 embedding，穿過 L 層 decoder，最後由 LM head 投影回詞彙表大小的分數。",
    "<b>Sampling</b> 依 temperature、top-p、top-k 從分數裡抽一個 token。",
    "抽到的 token 接回輸入尾端，再跑一次。自回歸就是這個迴圈。",
    "每一輪只在序列尾端多一格。生 500 個 token 就是 500 次完整 forward。",
    "重點在這個不對稱：<b>本輪只算 1 個 token，卻要把整個模型的權重讀一遍。</b>"],
    "自回歸生成的迴圈。第 5 步那個不對稱，是後面所有事情的起點。")

p("最後那一步是整篇文章的起點。一次 forward 要把模型的每一個權重都從 HBM 搬進計算單元，",
  "但如果同時只有一個使用者，這一趟搬運只換到一個 token。GPU 的張量核心在這個過程裡幾乎全程閒著。")

h3("KV cache：拿記憶體換計算", "kv")

p("Attention 需要拿本輪的 Query 去比對<b>前面每一個位置</b>的 Key 和 Value。",
  "如果每輪都重算前面所有位置，總成本會是 O(T²)。所以大家都把算過的 K/V 存起來，",
  "這就是 ", T("KV cache"), "。", cite("paged"))

figure("kv", F1.fig_kv(), 4, [
    "attention 需要本輪的 q 去比對所有位置的 K/V。",
    "<b>①</b> 前面 7 個位置的 K/V 早就算過了，躺在 cache 裡。",
    "<b>②</b> 本輪只需要新增第 8 個位置的 k₈ 與 v₈。",
    "<b>③</b> 但 attention 要<b>讀取全部 8 格</b>，讀取量隨 context 線性成長。",
    "<b>④</b> 沒有 cache 的話，每輪都要把前面重算一次，總成本變 O(T²)。"],
    "KV cache 把生成成本從 O(T²) 壓到 O(T)，代價是每個 token 都要永久佔一塊 HBM。")

p("這筆交易在長 context 下非常划算，但它的代價是<b>記憶體</b>：每個 token 都要佔一塊 HBM，",
  "直到這個 request 結束。而這就是併發數的硬上限。")

h3("GQA：Q 很多、K/V 很少", "gqa")

p("減少 KV cache 最便宜的辦法，是讓多個 Query head 共用同一組 K/V head，也就是 ",
  T("GQA"), "。Qwen3.5-122B-A10B 用 32 個 Q head 共用 2 組 K/V，比例 16:1。", cite("c35"))

figure("gqa", F1.fig_gqa("q35"), 2, [
    "上排 32 個藍格是 Query head，每個 head 各自算 attention。",
    "下排只有 2 組橘格：16 個 Q head 共用同一組 K/V。",
    "所以每個 token 只需要保存 2 × 2 × 256 = 1,024 個數字。"],
    "GQA 的效果直接寫在 config 裡：num_attention_heads=32、num_key_value_heads=2。",
    wide=False)

formula("M<sub>KV</sub> = 2 × <em>L<sub>full</sub></em> × "
        "<span class='c2'>H<sub>KV</sub></span> × d<sub>head</sub> × bytes × "
        "<span class='c3'>T</span>",
        "2 是 K 和 V 各一份；L<sub>full</sub> 是需要 KV 的層數；H<sub>KV</sub> 是 KV head 數；"
        "T 是目前的 context 長度。整篇文章的 KV 數字都從這條式子來。")

table(["模型", "~L_full", "~H_KV", "~d_head", "~KV / token (BF16)", "~KV / token (FP8)"], [
    [f"{Q35.label}", Q35.n_full, Q35.n_kv, Q35.head_dim,
     f"{Q35.kv_bytes_per_token('bf16')/KiB:.0f} KiB", f"<b>{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB</b>"],
    [f"{Q38.label}", Q38.n_full, Q38.n_kv, Q38.head_dim,
     f"<b>{Q38.kv_bytes_per_token('bf16')/KiB:.0f} KiB</b>", f"{Q38.kv_bytes_per_token('fp8')/KiB:.0f} KiB"],
    [f"{Q332.label}（純 Transformer 對照）", Q332.n_full, Q332.n_kv, Q332.head_dim,
     f"<b>{Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB</b>", f"{Q332.kv_bytes_per_token('fp8')/KiB:.0f} KiB"],
], "compact", hi=(2,))

p("最後一列是重點。Qwen3-32B 每個 token 要 ",
  f"{Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB，是 122B 的 ",
  f"{Q332.kv_bytes_per_token('bf16')/Q35.kv_bytes_per_token('bf16'):.0f} 倍。",
  "差距來自兩件事相乘：層數（64 對 12）與 KV head 數（8 對 2）。", cite("c332"))

h3("Prefill 與 decode 是兩種不同的工作", "prefill")

p("同樣的權重、同樣的層，但 prefill 和 decode 的成本結構是相反的。")

figure("pd", F1.fig_prefill_decode(), 3, [
    "prefill 一次吃完整個 prompt；decode 每輪每條序列只餵 1 個 token。",
    "矩陣形狀完全不同：prefill 是胖矩陣（大 GEMM），decode 退化成瘦長的 GEMV。",
    "所以算術強度差了兩個數量級：prefill 逼近算力上限，decode 卡在頻寬上限。",
    "看清這件事之後，serving 上的最佳化就收斂成同一個動作："
    "<b>把更多 token 塞進同一次權重讀取裡。</b>"],
    "prefill 決定 TTFT，decode 決定 TPOT。兩者要用不同的方式最佳化。")

p("Prefill 一次處理幾千個 token，每個權重讀進來之後被用了幾千次，張量核心可以打滿。",
  "Decode 每輪每條序列只有 1 個 token，同一份權重只服務 batch 大小那麼多次運算。",
  "兩者的 ", T("Arithmetic intensity", "算術強度"), " 差了兩個數量級。", cite("anat"))

# ============================================== 2. roofline ==============
h2("PART 02", f"B200 上的臨界值：{KNEE:.0f} 個 token", "roofline")

p("這是整篇文章唯一需要動筆推的地方，但它只有兩行。")

p("一次 forward 的時間，被兩件事夾住。一邊是把權重從 HBM 搬進來要多久：")

formula("t<sub>頻寬</sub> = <em>P</em> · <span class='c2'>b</span> / <span class='c3'>BW</span>",
        "P 是參數量，b 是每個參數佔幾個 byte，BW 是 HBM 頻寬。")

p("另一邊是張量核心把這些乘加做完要多久。每個 token 對每個參數做一次乘、一次加，所以是 2 次 FLOP：")

formula("t<sub>算力</sub> = <em>N</em> · 2<em>P</em> / <span class='c3'>F</span>",
        "N 是這一次 forward 送進去的 token 總數，F 是峰值算力。")

p("兩者相等的地方，就是 memory-bound 與 compute-bound 的交界。把 P 消掉："
  )

formula("<em>N*</em> = <span class='c3'>F</span> · <span class='c2'>b</span> / "
        "(2 · <span class='c3'>BW</span>)",
        "代入 B200 的 FP8 規格：4.5 PFLOP/s × 1 byte ÷ (2 × 7.7 TB/s) = "
        f"{KNEE:.1f} token / step。")

figure("roof", F1.fig_roofline(), 5, [
    "橫軸是一次 forward 送進去的 token 總數 N，縱軸是每個 token 的成本，兩軸都是對數。",
    f"兩段曲線的交會處就是 <b>N* ≈ {KNEE:.0f}</b>。",
    "左邊 memory-bound：時間等於讀權重的時間，跟 N 無關，多塞 token 幾乎不用加錢。"
    "右邊 compute-bound：時間開始隨 N 線性上升。",
    "所以 batch 1 是最浪費的模式：付了整份權重的錢，只換到 1 個 token。",
    "32 併發配上 DFlash2 的 7 個草稿，一次就送進 256 個 token，剛好貼在轉折點上。"],
    "屋頂線。左半邊是「加 token 幾乎免費」的區間，右半邊開始按 token 收費。")

h3("量化幫不上這個忙", "quant-invariance")

p("這裡有個結果很反直覺。B200 的張量核心，每把位寬砍半就把吞吐加倍：BF16 是 2.25 PFLOP/s、",
  "FP8 是 4.5、FP4 是 9。", cite("b200l"),
  " 把這組數字代回上面那條式子，分子的 F 加倍、分母那側的 b 減半，兩者剛好相消：")

table(["~精度", "~b（byte/參數）", "~F（PFLOP/s）", "~N*"], [
    ["BF16", "2", "2.25", f"<b>{GPU.knee_tokens('bf16'):.0f}</b>"],
    ["FP8", "1", "4.5", f"<b>{GPU.knee_tokens('fp8'):.0f}</b>"],
    ["NVFP4", "0.5", "9.0", f"<b>{GPU.knee_tokens('nvfp4'):.0f}</b>"],
], "compact")

note(f"""
<p>所以「上了 FP4 就不再是 memory-bound」是個誤會。FP4 讓你讀得更快，也讓你算得更快，
你在屋頂線上的<b>相對位置</b>沒有動。</p>
<p>量化真正買到的是三件事：step 時間變短、模型裝得進更少張卡、省下來的 HBM 全部變成 cache 空間。
要離開 memory-bound 只有一條路，就是提高每個 step 的 token 數 ——
更大的 batch，或是 {T("Speculative decoding")}。</p>
<p>反過來說，只要 N &lt; {KNEE:.0f}，你手上就有一塊<b>免費的算力</b>。
這塊算力就是 speculative decoding 的本錢。</p>
""")

# =============================================== 3. vLLM ================
h2("PART 03", "vLLM 怎麼把它變成一個服務", "vllm")

p("模型只回答「這一個 token 是什麼」。要變成能同時服務幾百人的服務，中間那層要解決三個問題：",
  "誰在這一輪上車、KV cache 怎麼配、算完之後誰該離開。", cite("arch", "anat"))

figure("stack", F2.fig_stack(), 6, [
    "一個 request 從上往下穿過六層。",
    "<b>API server</b> 負責 HTTP、chat template、tokenize 與串流回傳。",
    "<b>EngineCore 排程器</b> 決定每一輪誰上車、各帶幾個 token。",
    "<b>KV cache manager</b> 管 block table 與空閒池；混合模型還要多管一個 SSM state pool。",
    "<b>Executor / Worker</b> 處理分散式；單卡時只是一層薄殼。",
    "<b>Model runner</b> 組 batch metadata 並重播 CUDA graph。",
    "<b>Kernels</b> 才是真正動手的地方，量化 kernel 就掛在這一層。"],
    "vLLM 的六層責任邊界。知道每個 CLI 旗標作用在哪一層，調參就不會亂試。")

h3("Continuous batching", "cb")

p("靜態 batching 的問題是要等最慢的那個人講完。", T("Continuous batching"),
  " 每一輪重新組隊：做完的離開，等待的立刻補進來。", cite("anat"))

figure("batch", F2.fig_batching(), 4, [
    "上半是靜態 batching：四個 request 一起送進去，短的做完了也只能等。",
    "紅色格子是純粹的浪費，而且算力和頻寬一起浪費。",
    "下半是 continuous batching：做完就離開，位置立刻讓給新 request。",
    "同一份權重讀取被更多 token 分攤，往屋頂線上更划算的位置移動。"],
    "Continuous batching 的重點不在「把 batch 開大」，而在「讓成員每一輪都可以換」。",
    accent="ok")

h3("PagedAttention", "paged")

p("KV cache 如果要求連續配置，就得為每個 request 預留最壞情況的空間，實際利用率常常只有兩三成。",
  T("PagedAttention"), " 把它切成固定大小的 block，用 block table 對應，跟作業系統的分頁一樣。",
  cite("paged"))

figure("paged", F2.fig_paged(), 3, [
    "邏輯上，每個 request 就是一串連續的 token。",
    "實體上，KV 被切成固定大小的 block，散落在 HBM 各處；用完的回到空閒池。",
    "<b>block table</b> 記住邏輯到實體的對應。",
    "換來三個能力：沒有碎片、前綴可共用、request 可以被換出再換回。"],
    "兩個 request 的 logical block 0 可以指向同一個 physical block，"
    "prefix caching 就是靠這個做的。", accent="mem")

h3("排程器一輪做的四個決定", "sched")

figure("sched", F2.fig_scheduler(), 4, [
    "每一輪有一個 token 預算（<code>--max-num-batched-tokens</code>，示例取 8,192）。",
    "<b>①</b> 先排 running 佇列裡的 decode：32 個 request 只吃掉 32 個 token，非常便宜。",
    "<b>②</b> 剩下的 budget 拿去餵 prefill；長 prompt 被切成 chunk 分次送。",
    "<b>③</b> 若 cache 配不出來，就從尾端 preempt request 換出。",
    "<b>④</b> 最後把 batch metadata 交給 model runner。"],
    "decode 優先於 prefill，等於 TPOT 優先於 TTFT。這兩個指標在排程器裡共用同一個預算。")

p("這個順序有它的道理：已經在串流的使用者如果卡住，體感是「打字停住」，",
  "比多等一點第一個字難忍受得多。想調整這個取捨，旋鈕就是那個 token budget。")

note("""
<p><b>一個實用的觀察指標</b>　如果 <code>num_preempted</code> 持續大於 0，
代表 cache 不夠用。這時該做的是降低 <code>max-num-seqs</code> 或縮短
<code>max-model-len</code>，而不是繼續加壓。preempt 是容量訊號，不是效能訊號。</p>
""", "warn")

# ============================================ 4. 混合架構 ================
h2("PART 04", "混合架構：用固定狀態換掉 KV cache", "hybrid")

p("Qwen3.5 與 Qwen3.8 都不是純 Transformer。它們每 4 層裡只有 1 層是 full attention，",
  "另外 3 層換成 ", T("Gated DeltaNet (GDN)", "Gated DeltaNet"),
  "，一種 linear attention。", cite("c35", "c38"))

figure("arch3", F2.fig_arch3(), 2, [
    "三個模型的層堆疊並排：上排是 token mixer，下排是 FFN。",
    "藍色是 full attention（需要 KV cache），青色是 Gated DeltaNet（固定大小 state）。",
    "所以 KV/token 差了一個數量級。後面的記憶體帳本都從這裡長出來。"],
    "同一個 decoder 骨架，換掉 token mixer 與 channel mixer 兩個插槽，"
    "就得到三種完全不同的成本結構。", accent="moe")

h3("核心想法：把逐字稿換成摘要", "linear")

p("Full attention 保留每一個位置的 K/V，所以什麼都查得到，但檔案越來越厚。",
  "Linear attention 只維護一個固定大小的矩陣，永遠一頁，代價是細節會流失。",
  "混合架構等於「每四頁摘要就附一頁逐字稿」。")

table(["", "Full attention", "Gated DeltaNet"], [
    ["每條序列的記憶體", "O(T)，隨 context 線性成長", "O(1)，常數"],
    ["decode 的讀取量", "O(T)", "O(1)"],
    ["prefill 的計算", "O(T²)，可分塊", "O(T)，chunkwise 平行"],
    ["資訊保真度", "無損", "有損，越長越糊"],
    ["Qwen 的用法",
     f"每 4 層放 1 層（{Q35.n_full}/{Q35.n_layers} 或 {Q38.n_full}/{Q38.n_layers}）",
     f"其餘 3/4（{Q35.n_gdn} 或 {Q38.n_gdn} 層）"],
], "compact")

h3("狀態更新的數學", "gdn-math")

p("GDN 的更新規則長這樣：", cite("gdn"))

formula("S<sub>t</sub> = S<sub>t−1</sub>( <em>α<sub>t</sub></em>"
        "( I − <span class='c2'>β<sub>t</sub></span> k<sub>t</sub>k<sub>t</sub><sup>T</sup> ) )"
        " + <span class='c2'>β<sub>t</sub></span> v<sub>t</sub>k<sub>t</sub><sup>T</sup>"
        "　　o<sub>t</sub> = S<sub>t</sub> q<sub>t</sub>",
        "α ∈ (0,1) 是 decay gate，決定整張狀態要忘掉多少；"
        "β ∈ (0,1) 是 delta rule 強度，決定 k 這個「位址」原本存的東西要換掉多少。")

figure("gdn", F2.fig_gdn(), 5, [
    "狀態 S 是一個固定大小的矩陣，每個 head 一份。",
    "<b>α</b> 是 decay gate：整張狀態乘上一個 0–1 的係數。括號裡的 (I − βkkᵀ) "
    "只針對 k 這個方向做清除。",
    "<b>βvkᵀ</b> 把新的關聯寫進去。",
    "更新完，S 的大小沒有變。整件事的重點就在這裡。",
    "輸出 o = Sq：用 q 去「問」這個摘要矩陣。"],
    "跟 Mamba2 的差別就在括號那一項。Mamba2 只有整體衰減，DeltaNet 只有定點替換，"
    "Gated DeltaNet 兩個都有。", accent="ssm")

p("為什麼要兩個都有？因為它們解決不同的事。α → 0 可以一口氣把整張狀態清空，",
  "適合話題切換；β → 1 可以只改寫某一個「鍵」原本存的內容，適合多輪對話裡改口。",
  "Mamba2 只會前者，DeltaNet 只會後者。")

p("這個式子有一個對工程很重要的性質：gate 只做逐元素乘法，不破壞矩陣乘法的結構。",
  "所以 prefill 時可以分塊平行，塞滿 Tensor Core，速度跟 attention 同級。")

quote("The gating term only performs elementwise multiplication with intermediate variables "
      "without affecting matrix multiply structures.",
      "Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv 2412.06464")

figure("chunk", F2.fig_chunkwise(), 4, [
    "prompt 被切成固定大小的 chunk。",
    "chunk 內部用矩陣乘法平行處理，塞滿 Tensor Core。",
    "chunk 之間只傳遞一個固定大小的狀態 S，序列依賴只剩這一條。",
    "同時，full attention 層照常做 FlashAttention 並寫入 prompt 的 K/V。",
    "注意：vLLM 對 mamba / GDN 的 prefix caching 仍在開發中。"],
    "chunkwise 平行讓混合模型的 prefill 不會因為 GDN 而變慢。"
    "真正的差異在 decode（省很多）與 prefix caching（目前吃虧）。", accent="ssm")

h3("兩種 cache 要同步前進", "two-caches")

figure("hybrid", F2.fig_hybrid("q35"), 4, [
    "Qwen3.5-122B-A10B 的 48 層排成 12 組，每組是「GDN、GDN、GDN、attention」。",
    "資料由左往右穿過每一層。",
    f"{Q35.n_gdn} 層 GDN：讀舊 state → 就地寫回新 state，大小固定 "
    f"{mib(Q35.ssm_recurrent_bytes(),0)} MiB。",
    f"{Q35.n_full} 層 full attention：讀全部歷史 K/V，在尾端多寫一格。",
    "兩者要同步前進。hybrid 模型 serving 最容易出錯的地方就在這裡。"],
    "一次 decode 同時做了兩種性質完全不同的記憶體操作："
    "一種是就地更新固定區塊，一種是 append 到成長中的緩衝區。", accent="ssm")

p("vLLM 為此做了一個 Hybrid KV Cache Manager。它把層依 attention 型態分成 KV cache group，",
  "強制所有 group 的實體 page 大小一致，做法是<b>把 attention 層的 block_size 撐大</b>，",
  "直到塞得下一份 mamba state。官方文件坦承這會讓 block_size 超過 400，效率上有疑慮。",
  cite("hyb", "mut"))

note("""
<p><b>這對容量規劃的三個後果</b></p>
<p>SSM state 是「每條序列一份」，不是「每 token 一份」，所以它直接乘上 max-num-seqs，
跟 max-model-len 無關。</p>
<p>page size 被撐大之後，小 request 的內部碎片變多，實際可用的 cache 會略少於公式值。
實務上把公式值乘 1.05–1.15 比較保險。</p>
<p>GDN 的 prefix caching 還沒完備，所以多輪對話的 TTFT 改善幅度會比純 Transformer 小。
做 benchmark 時要記錄命中率。</p>
""", "warn")

# =============================================== 5. 量化 =================
h2("PART 05", "NVFP4 到底量化了哪些張量", "quant")

p("這一段大概是整篇文章裡最有趣的部分，因為結論跟大家的預期不一樣，而且完全可以自己驗。")

h3("先看格式", "nvfp4-format")

p(T("NVFP4"), " 是 NVIDIA 為 Blackwell 設計的 4 位元浮點格式。",
  "每 16 個 E2M1 值共用一個 FP8-E4M3 的 block scale，再乘上一個 per-tensor 的 FP32 scale。",
  cite("nvfp4", "nvfp4b"))

figure("nvfp4", F3.fig_nvfp4(), 5, [
    "一個 NVFP4 block 就是 16 個 E2M1 值。E2M1 只有 4 bit，非零大小只有 8 個級距。",
    "每 16 個值共用一個 <b>FP8 E4M3 的 block scale</b>。block 越小，一組值的動態範圍越窄，"
    "4 bit 就越夠用。",
    "再乘上一個 <b>per-tensor 的 FP32 scale</b>（checkpoint 裡的 <code>weight_scale_2</code>）。",
    "把 scale 攤提進去：(16×4 + 8) ÷ 16 = <b>4.5 bit / 值</b>。",
    "對照 FP8：Qwen3.8-27B-FP8 用 128 個值共用一個 scale，平均 8.03 bit / 值。"],
    "「4-bit 模型」實際上是 4.5 bit。算記憶體的時候要用 4.5。", accent="moe")

p("跟 OCP 標準的 MXFP4 比，NVFP4 有兩個差別：block 是 16 不是 32，",
  "scale 是真正的 FP8 浮點而不是 2 的冪次（E8M0）。前者讓每組值的動態範圍更窄，",
  "後者讓 scale 可以表示 1.75× 這種比例。代價是 scale 的儲存開銷變成兩倍。")

h3("weight map 說了什麼", "weightmap")

p("接下來是這篇文章的原創部分。我們沒有猜，而是直接把 ",
  "<code>model.safetensors.index.json</code> 抓下來，看哪些張量帶了 ",
  "<code>weight_scale</code>。有 scale 的才是被量化的。", cite("nvi", "i38"))

figure("qmap", F3.fig_quantmap(), 3, [
    "122B 的 NVFP4 checkpoint：紫色是 4-bit 的 routed expert，橘色是 FP8 的 block scale，"
    "藍色是完全沒被量化的 BF16 部分。BF16 佔了 21.9%。",
    "27B 的 FP8 checkpoint：青色是 FP8，比例高很多（80%），因為 dense FFN 佔了大宗。",
    "怎麼確定的：把張量數量乘出來，與 HuggingFace 回報的 dtype 統計逐一比對。"],
    "只有帶 weight_scale 的張量才是被量化的。這張圖完全來自 checkpoint 本身。",
    accent="moe")

note(f"""
<h4>驗算</h4>
<p>48 層 × 256 個 expert × 3 個投影 × 3072 × 1024 =
<b>115,964,116,992</b> 個 4-bit 權重。兩個 4-bit 打包成一個 byte，
得到 <b>57,982,058,496</b> bytes，與 HuggingFace 回報的 U8 統計<b>完全相同</b>。</p>
<p>再除以 group_size 16，得到 <b>7,247,757,312</b> 個 FP8 block scale，
與 F8_E4M3 統計也<b>完全相同</b>。</p>
<p>剩下的 {Q35.real_dtypes['BF16']:,} 個 BF16 參數，就是<b>沒有被量化</b>的部分。</p>
""", "key")

p("所以 122B 的 NVFP4 版本，實際上只量化了 routed expert。", T("Gated DeltaNet (GDN)", "GDN"),
  " 的 36 層、full attention 的 12 層、48 個 shared expert、router、embedding、",
  "lm_head、視覺塔、", T("MTP"), " head，<b>全部還是 BF16</b>。這跟 NVIDIA 模型卡的說法一致：",
  cite("nv4"))

quote("Weights and activations of the linear operators within transformer blocks in MoE are quantized.",
      "nvidia/Qwen3.5-122B-A10B-NVFP4 模型卡")

p("這件事有實際後果。在 batch 1 的 decode 裡，那 36 層 BF16 的 GDN 要讀 6.37 GB，",
  "而整個 step 的權重讀取量是 12.8 GB。也就是說 <b>NVFP4 只解決了一半的頻寬問題</b>，",
  "另一半還在等 GDN 的低精度 kernel。")

h3("省了多少，賠了多少", "quant-cost")

table(["", "~BF16 權重", "~量化後", "~省下", "單張 B200 180 GB"], [
    [f"{Q35.label} <span class='tagc e'>NVFP4</span>",
     f"{gb(Q35.real_bf16_bytes)} GB", f"<b>{gb(Q35.real_weight_bytes)} GB</b>",
     f"{(1-Q35.real_weight_bytes/Q35.real_bf16_bytes)*100:.0f}%",
     "<span class='tagc g'>單卡放得下</span>（BF16 至少要 2 卡）"],
    [f"{Q38.label} <span class='tagc s'>FP8</span>",
     f"{gb(Q38.real_bf16_bytes)} GB", f"<b>{gb(Q38.real_weight_bytes)} GB</b>",
     f"{(1-Q38.real_weight_bytes/Q38.real_bf16_bytes)*100:.0f}%",
     "<span class='tagc g'>單卡輕鬆</span>"],
], "compact", hi=(0,))

p("122B 從 250 GB 降到 83.5 GB，剛好跨過「單卡跑得動」那條線。",
  "NVIDIA 模型卡的部署範例就直接寫 <code>--tensor-parallel-size 1</code>。", cite("nv4"),
  " 這件事的價值不只是省卡：少了 TP 的 all-reduce，decode 的延遲與抖動都會改善，",
  "而且 8 卡的節點可以跑 8 份獨立副本，而不是 1 份 TP=8。")

p("精度的代價，NVIDIA 有公布評測。對照組是他們自己的 FP8 版本，不是 BF16：", cite("nv4"))

table(["Benchmark", "~FP8 baseline", "~NVFP4", "~差距"], [
    ["MMMU Pro", "75.90", "75.55", "<span style='color:var(--bad)'>−0.35</span>"],
    ["GPQA Diamond", "87.37", "86.77", "<span style='color:var(--bad)'>−0.60</span>"],
    ["SciCode", "42.16", "41.79", "<span style='color:var(--bad)'>−0.37</span>"],
    ["AA-LCR", "65.50", "67.13", "<span style='color:var(--ok)'>+1.63</span>"],
    ["IFBench", "70.91", "70.80", "<span style='color:var(--bad)'>−0.11</span>"],
], "compact")

note("""
<p>五項裡四項小輸、一項小贏，最大差距 0.60 分。以 3 倍的記憶體節省來說，這個代價很划算。</p>
<p>但有三件事要注意：對照組是 FP8 不是 BF16，FP8 對 BF16 本身也有差距；這些都是單輪評測，
多輪 agent 任務的誤差會累積；量化對長尾行為（罕見格式、少數語言、極長 context）的影響
通常比平均分數大。自家的驗收集還是要自己跑一次。</p>
""", "warn")

# ============================================== 6. 記憶體帳本 ============
h2("PART 06", "記憶體帳本：兩種 cache，兩條曲線", "memory")

p("混合模型的記憶體要分開算，因為 KV cache 與 SSM state 的成長方式完全不同。",
  "一個隨 context 線性長大，一個是常數。")

formula("M<sub>state</sub> = <em>L<sub>GDN</sub></em> × "
        "<span class='c2'>H<sub>V</sub></span> × d<sub>V</sub> × d<sub>K</sub> × 4 B"
        "<br>M<sub>conv</sub> = <em>L<sub>GDN</sub></em> × conv_dim × "
        "(conv_kernel − 1 + <span class='c3'>k<sub>spec</sub></span>) × 2 B",
        "recurrent 狀態用 FP32（config 的 mamba_ssm_dtype），"
        "conv 歷史的長度會隨 speculative decoding 的草稿數 k 變長。")

table(["模型", "~L_GDN", "~H_V", "~d_V × d_K", "~recurrent (FP32)", "~conv", "~合計 / 序列"], [
    [Q35.label, Q35.n_gdn, Q35.nv, f"{Q35.dv} × {Q35.dk}",
     f"<b>{mib(Q35.ssm_recurrent_bytes(),1)} MiB</b>",
     f"{mib(Q35.ssm_conv_bytes(Q35.spec_k),2)} MiB <span class='tagc p'>k=3</span>",
     f"<b>{mib(Q35.ssm_bytes(Q35.spec_k),1)} MiB</b>"],
    [Q38.label, Q38.n_gdn, Q38.nv, f"{Q38.dv} × {Q38.dk}",
     f"<b>{mib(Q38.ssm_recurrent_bytes(),1)} MiB</b>",
     f"{mib(Q38.ssm_conv_bytes(Q38.spec_k),2)} MiB <span class='tagc p'>k=7</span>",
     f"<b>{mib(Q38.ssm_bytes(Q38.spec_k),1)} MiB</b>"],
], "compact")

p("兩個模型的 recurrent state 竟然一樣大，都是 144 MiB。因為 36×64 = 48×48 = 2,304 個",
  " value head，每個都是 128×128×4 bytes = 64 KiB。巧合，但很好記。", cite("mut"))

figure("twocache", F2.fig_two_caches(), 4, [
    "縱軸是每條序列佔用的記憶體，兩軸都是對數刻度。",
    "灰線是 Qwen3-32B（純 Transformer）：全程被 KV cache 支配，斜率最陡。",
    "藍線與橘線是兩個混合模型：短 context 時幾乎是水平的。",
    "因為它們各有一條固定的 SSM state 水平線（約 150 MiB），在短 context 時完全主導。",
    "兩條線的交會點就是「KV 開始接管」的位置。"],
    "混合模型有一個約 150 MiB 的固定起跳成本，之後斜率平緩很多。", accent="mem")

note(f"""
<h4>一個反直覺的結論</h4>
<p>122B 要到 <b>{Q35.crossover_tokens('fp8', 3):,.0f} 個 token</b>、
27B 要到 <b>{Q38.crossover_tokens('bf16', 7):,.0f} 個 token</b>，
KV cache 才追上固定的 SSM state。</p>
<p>換句話說，如果你的 workload 是短對話，併發上限其實卡在 SSM state，不在 KV cache。
這時候把 <code>--max-model-len</code> 調小<b>不會</b>提高多少併發，
該調的是 <code>--max-num-seqs</code>，或考慮用
<code>--mamba-ssm-cache-dtype bf16</code> 把 state 砍半（但要先驗長 context 的品質）。</p>
""", "key")

h3("自己算一次", "budget-calc")

p("下面這個試算器把整張帳本攤開：權重、起草器、KV、SSM、activation 全部要塞進 ",
  "<code>gpu_memory_utilization</code> 圈起來的空間。可以切模型、切 KV 精度、",
  "拉 context 與併發數看它怎麼變。")

widget("budget", "單卡 B200 的記憶體帳本。權重採用 HuggingFace 上實際的 safetensors 位元組數；"
                 "activation 是教學用的估計值，實際由 vLLM 依 batch 尺寸與 kernel workspace 決定。")

p("值得留意的是「起草器權重」那一條。122B 的 MTP head 有 ",
  f"{Q35.params_mtp/1e9:.2f} B 參數，而且在 NVFP4 checkpoint 裡<b>沒有被量化</b>，",
  f"BF16 佔 {gb(Q35.params_mtp*2)} GB。只要開 speculative decoding，這塊就從 cache 預算裡扣掉。")

widget("cache", "同一張卡，你可以選「多人短對話」或「少人長文件」。這是一條雙曲線，沒有免費的午餐。")

note("""
<p><b>提醒</b>　這些是<b>記憶體</b>能容納的序列數上限，不等於「能達成 SLO 的併發數」。
真正能承諾的容量還受延遲限制，通常小得多。後面 PART 08 會講怎麼量。</p>
""", "warn")

# ========================================= 7. speculative decoding =======
h2("PART 07", "Speculative decoding 什麼時候會賠錢", "spec")

p("回到 PART 02 的結論：只要一次 forward 的 token 數少於 ", f"{KNEE:.0f}",
  "，你就有一塊免費的算力。", T("Speculative decoding"),
  " 就是拿這塊算力去猜後面幾個 token，再讓目標模型用<b>同一次</b>權重讀取一口氣驗證。",
  cite("spd"))

figure("specwhy", F3.fig_spec_why(), 3, [
    "decode 的兩種資源使用率差距很大：頻寬被打滿，算力幾乎閒著。",
    "粉紅色那塊算力，不用也是浪費掉。",
    "所以：拿它去猜 k 個 token，再讓目標模型用同一次權重讀取一口氣驗證。",
    "輸出品質不變（rejection sampling 保證），但那塊算力有限。"],
    "32 併發時，27B 只用掉不到 15% 的張量核心算力。", accent="spec")

h3("一輪是怎麼跑的", "spec-cycle")

figure("speccyc", F3.fig_spec_cycle(), 5, [
    "已經確定的輸出擺在左邊。",
    "<b>起草</b>：用便宜的方式產生 k = 7 個候選 token。",
    "<b>驗證</b>：目標模型一次 forward 同時算出 8 個位置的正確分布。",
    "<b>接受</b>：由左往右逐位置判定。第 4 個開始被拒絕，後面整串丟掉。",
    "拒絕的位置改用目標模型自己的輸出（<b>保底 token</b>），所以一定前進至少 1 格。",
    "這一輪前進 4 格，只花了 1 次起草 + 1 次目標模型 forward。"],
    "驗證用的是 rejection sampling，數學上保證輸出分布與不猜時相同。", accent="spec")

note("""
<p><b>三個常見誤解</b></p>
<p>「猜錯會讓品質變差」：不會。vLLM 文件的說法是
"theoretically lossless up to the precision limits of hardware numerics"。</p>
<p>「最壞情況會比不猜慢」：就 <b>token 數</b>而言不會（一定前進 1 格），
但就<b>時間</b>而言會，因為多付了起草成本與驗證的額外計算。</p>
<p>「被拒絕的位置後面那些草稿還可以留著」：不行。它們是以錯誤前綴為條件產生的。</p>
""")

h3("我們用的兩種起草器", "two-drafters", )

p("這次評估的兩個模型，用的是完全不同的起草方式。122B 用 Qwen 自己訓練、",
  "包在 checkpoint 裡的 ", T("MTP"), " head；27B 用外掛的 ", T("DFlash2"), "。",
  cite("m35", "df2"))

figure("spectl", F3.fig_spec_timeline(), 6, [
    "同樣要產生草稿，兩種起草器的時間軸完全不同。",
    "<b>122B 的 MTP</b>：1 層 decoder head，autoregressive 跑 k 次。",
    "然後目標模型一次驗證 k+1 = 4 個位置。",
    "<b>27B 的 DFlash2</b>：block diffusion，一次 forward 就把整塊 8 個位置都猜出來。",
    "接著驗證 8 個位置。",
    "所以 MTP 的 T_draft = k × t_step，DFlash2 的 T_draft = t_parallel（與 k 無關）。",
    "DFlash2 敢用 k = 7、MTP 通常停在 k = 2–4，差別就在這裡。"],
    "起草成本的結構不同，能用的 k 就不同。", accent="spec")

p("DFlash2 的做法值得多講兩句。它用<b>非因果</b>的 attention mask，",
  "讓每個 query 同時看到目標模型的 hidden state 與整排 mask token，一次 forward 產生整塊草稿，",
  "再用一個輕量 selector（config 裡的 <code>selector_rank: 256</code>、",
  "<code>selector_top_k: 16</code>）挑一條連貫的路徑出來。", cite("dfp", "dfn"))

p("它還把目標模型 5 層的 hidden state 注入到起草器<b>每一層</b>的 KV",
  "（config 的 <code>target_layer_ids: [5, 19, 33, 47, 61]</code>），",
  "而不是只在輸入層融合一次。論文說這讓接受長度能隨草稿深度繼續成長，而不是很快飽和。",
  cite("dfp"))

table(["", f"{Q35.label}", f"{Q38.label}"], [
    ["起草方法", "<b>MTP</b>（multi-token prediction），EAGLE 系", "<b>DFlash2</b>，block diffusion"],
    ["來源", "checkpoint 內含 <code>mtp.*</code>", "外部 <code>incoai/Qwen3.8-27B-DFlash2</code>"],
    ["起草器規模", f"1 層 MoE decoder，{Q35.params_mtp/1e9:,.2f} B（BF16，未量化）",
     f"5 層 sliding attention，{Q38.spec_draft_params/1e9:,.2f} B（BF16）"],
    ["草稿 token 數 k", "<b>3</b>（官方建議 2–4）", "<b>7</b>（block size 8，扣掉 anchor）"],
    ["起草成本", "∝ k，每次都要重讀 lm_head（1.53 GB）",
     f"固定 {Q38.spec_draft_params*2/1e9:,.2f} GB，與 k 無關"],
    ["官方實測接受長度", "未公布逐 benchmark 數字",
     "GSM8K <b>5.46</b>、MATH-500 <b>5.28</b>、MBPP <b>4.79</b>、HumanEval <b>4.39</b>"],
    ["官方實測加速（併發 1）", "—",
     "GSM8K <b>3.43×</b>、MATH-500 <b>3.34×</b>、HumanEval <b>3.11×</b>"],
], "compact", hi=(3,))

note("""
<p><b>這些數字哪些可以搬，哪些不行</b>　DFlash2 模型卡的吞吐是在併發 1 量的，
而且沒有載明硬體。<b>接受長度</b>可以跨硬體參考，它只跟模型與資料有關；
<b>絕對 tok/s 與加速倍數不行</b>，那會隨 GPU、量化與併發改變。</p>
""", "warn")

h3("接受長度 τ 的報酬遞減", "tau")

p("每個草稿位置要被接受，前面所有位置都得先對，所以第 i 個位置的接受機率是 α^i。",
  "把它們加起來就是期望接受長度：", cite("sspec", "lat"))

formula("τ = Σ<sub>i=0..k</sub> α<sup>i</sup> = "
        "(1 − <em>α</em><sup>k+1</sup>) / (1 − <em>α</em>)",
        "α 是逐位置的接受率，k 是草稿數。τ 的上限是 k+1。")

figure("accept", F3.fig_accept(), 4, [
    "第 i 個草稿位置要被接受，前面 i−1 個都得先對，所以機率是 α^i。",
    "α 越小衰減越快。α = 0.55 時，第 4 個位置就只剩 9% 的機會。",
    "把每個位置的接受機率加起來就是期望接受長度 τ。",
    "報酬遞減：α = 0.80 時，k 從 7 加到 15，τ 只多 15%，但驗證成本翻倍。",
    "實測參考：DFlash2 在 27B 上的接受長度是 4.4–5.5，對應 α ≈ 0.80–0.86。"],
    "這個式子假設每個位置的接受率相同，所以它是樂觀估計。"
    "真正的 τ 要從引擎的 acceptance metrics 讀。", accent="spec")

h3("(k+1)× 的 token 撞上屋頂線", "spec-roofline")

p("這是整篇文章最實用的一段。驗證階段一次要送 <b>B × (k+1)</b> 個 token 進去。",
  "PART 02 已經證明，這個數字超過 ", f"{KNEE:.0f}", " 之後就開始按 token 收費。所以：")

formula("B<sub>安全</sub> ≈ <em>N*</em> / (k + 1)",
        f"27B 配 k=7 → B ≈ {KNEE/8:.0f}；122B 配 k=3 → B ≈ {KNEE/4:.0f}（MoE 會讓它更寬，見下）。")

figure("specbatch", F3.fig_spec_batch(), 4, [
    "橫軸是併發序列數 B，縱軸是相對不用 spec 的輸出速率倍數。",
    f"<b>27B + DFlash2 (k=7)</b>：B × 8 一旦超過 N* = {KNEE:.0f}（也就是 B ≈ "
    f"{KNEE/8:.0f}），驗證變成 compute-bound，加速直線墜落。",
    "<b>122B + MTP (k=3)</b>：形狀完全不同，因為它是 MoE。低併發時 spec 的優勢被"
    "「驗證要碰到更多 expert」吃掉；高併發時 expert 讀取被更多 token 分攤，優勢回升。",
    "紅色虛線是損益兩平線。低於它，speculative decoding 就是純粹在燒算力。"],
    "解析上限，只算權重讀取與張量運算。真實曲線的形狀相同，但轉折更早、絕對值更低。",
    accent="spec")

p("MoE 那條線的形狀值得解釋一下。每個 token 從 256 個 expert 裡選 8 個，",
  "N 個 token 之後被碰到的 expert 期望值是 256 × (1 − (1 − 8/256)^N)。",
  "N=1 是 8 個、N=32 是 164 個、N=128 是 251 個、N≈290 就幾乎全部。")

p("所以開了 spec 之後，驗證階段送進去的 token 變多，<b>要讀的 expert 也跟著變多</b>。",
  "低併發時這件事會吃掉 spec 的好處；等到高併發、expert 本來就已經全部要讀了，",
  "spec 多送的 token 就變成純賺。這也是為什麼 122B 那條線會先降後升。")

h3("自己調調看", "spec-calc")

widget("spec", "三個變數決定一切：草稿數 k、逐位置接受率 α、同時併發數 B。"
                "建議的操作順序：先看預設的 4.4×，再把 B 拉到 64、200，"
                "然後把 k 降到 3 看高併發的表現，最後把 α 降到 0.6 模擬長 context。")

h3("接受率不是常數", "accept-decay")

p("最後一個坑，也是最容易踩的。接受率會隨 context 長度掉。",
  "vLLM 有一個 issue 記錄了 Qwen3.6-27B 上的實測：", cite("iss1"))

figure("acceptctx", F3.fig_accept_ctx(), 5, [
    "vLLM issue #47602 的實測：短 context 時 MTP 帶來 129% 的吞吐提升。",
    "但隨著 context 變長，優勢一路縮小。",
    "約 12k 之後，開 MTP 反而<b>比不開更慢</b>。30k 時慢了 51%。",
    "原因在右邊：逐位置接受率整條下降。第 1 個位置從 93.5% 掉到 72.1%，"
    "第 6 個從 36.6% 掉到 14.0%。",
    "長 context 服務要重新量接受率，短 prompt 的結論搬不過來。"],
    "同一個模型、同一個 k，context 從 2k 長到 30k，平均接受率從 64.9% 掉到 39.1%。",
    accent="bad")

p("為什麼？兩個原因。起草器看到的上下文越長，任務越難、分布越發散，猜中率自然下降。",
  "同時 context 一長，目標模型自己的每一步也更貴（attention 讀取量隨 T 成長），",
  "但起草器的成本沒有等比例下降，相對開銷就變高。")

p("DFlash 論文也觀察到同樣現象：base model 的接受長度在 4k 之後開始下降。",
  "他們的解法是針對長 context 微調起草器，能把 16k 的 τ 從 3.61 拉回 6.05。", cite("dfp"))

h3("其他三個副作用", "spec-side")

figure("specmem", F3.fig_spec_mem(), 4, [
    "第一筆：起草器本身的權重。",
    "第二筆：GDN 的 conv state 長度是 conv_kernel − 1 + num_spec，所以 k 越大它越長。",
    "第三筆：每條序列要多預留 k 個 KV slot 給驗證用。",
    "記憶體不是主要限制，但 cache 本來就很緊的長 context 場景，這幾 GB 會直接吃掉併發數。"],
    "第二筆是從 vLLM 的 mamba_utils.py 讀出來的：conv_state_shape 的第二維就是 "
    "conv_kernel − 1 + num_spec。", accent="spec")

p("除了記憶體，還有兩個容易讓 benchmark 數字變得難以解釋的副作用。")

p("<b>Prefix caching 命中率會掉。</b>vLLM 有一個 issue 回報，在 Qwen3.5-35B-A3B 上開 MTP",
  "（<code>num_speculative_tokens: 1</code>）之後，prefix cache 命中率從 92% 掉到 71%。",
  "被拒絕的草稿會讓序列長度不對齊 block 邊界，可快取的 block 就變少。",
  "對本來就靠 prefix caching 吃飯的 agent／多輪對話，影響特別大。", cite("iss2"))

p("<b>ITL 會變得不規則。</b>沒開 spec 時每個 step 產生 1 個 token，", T("ITL"),
  " 很平穩；開了之後一輪可能吐 1 個也可能吐 8 個。平均 ", T("TPOT"),
  " 改善，但 ITL 的 p99 會變差。如果你的 SLO 是用 ITL p99 定義的，要重新校準門檻，",
  "或改用 TPOT。")

h3("所以到底該不該開", "spec-decision")

table(["情境", "建議", "理由"], [
    ["單一使用者互動、demo、內部工具（B ≤ 8）", "<span class='tagc g'>全開，k 用最大</span>",
     f"完全在 memory-bound 區，B×(k+1) 遠低於 {KNEE:.0f}。TPOT 可以改善 2–4×。"],
    [f"中等併發（B ≈ 8–{KNEE/8:.0f}）、context &lt; 8k", "<span class='tagc g'>開</span>",
     "仍在 memory-bound 區。27B 配 k=7 可以撐到 B ≈ 36。"],
    [f"高併發（B &gt; {KNEE/8:.0f}）、追求整機 throughput",
     "<span class='tagc m'>調小 k，或關掉</span>", "驗證變成 compute-bound，多猜的都在燒算力。"],
    ["長 context（&gt; 16k）", "<span class='tagc r'>先量再決定</span>",
     "接受率顯著下降，實測案例顯示 30k 時反而慢 51%。"],
    ["前綴重複度高的 agent／多輪對話", "<span class='tagc m'>A/B 測命中率</span>",
     "spec 會拉低 prefix cache 命中率，兩個效益可能互相抵消。"],
    ["QPS 波動大", "<span class='tagc'>用動態 speculative decoding</span>",
     "vLLM 支援依負載調整；SmartSpec 的做法是依 goodput 決定每輪的 k，高負載自動歸零。"],
], "compact")

p("SmartSpec 那篇論文把這件事講得很清楚：他們用 goodput 當目標函數決定每一輪要猜幾個，",
  "在高負載時自動把 k 降到 0，實測最多降低 3.2× 延遲，而且<b>從不劣化</b>。", cite("sspec"))

# ============================================== 8. 怎麼量 ================
h2("PART 08", "怎麼量，才算數", "measure")

p("前面所有的數字都是解析推導。要變成可以對外承諾的容量，還得實際量一次。",
  "這一段講怎麼量才不會自己騙自己。")

h3("先把指標定義釘死", "metrics")

figure("lat", F4.fig_latency(), 7, [
    "一個 request 的生命週期：網路與排隊、prefill、然後是 N 次 decode。",
    "<b>網路 + 佇列</b>：這一段不在模型裡，但使用者一樣要等。",
    "<b>prefill</b>：把整個 prompt 算完並填好 cache。",
    "<b>decode</b>：一次吐一個 token（開了 spec 之後可能一次吐好幾個）。",
    "<b>TTFT</b>：送出到收到第一個 token，包含排隊與 prefill。",
    "<b>ITL</b>：相鄰兩次串流輸出之間的間隔。母體是「每一次間隔」。",
    "<b>E2EL</b>：送出到收到完整回覆。",
    "<b>TPOT</b> 是每個 request 自己算的平均，母體是「每一個 request」。"],
    "ITL 與 TPOT 最容易被混用。100 個 request 各 500 個 token："
    "ITL 有 49,900 個樣本，TPOT 只有 100 個。所以 p95 會是兩個完全不同的數字。",
    accent="compute")

p("為什麼要這麼囉唆？因為容量數字之所以會被吵，多半不是量錯，",
  "而是兩邊講的指標、百分位或量測窗根本不一樣。", cite("bench"))

h3("兩階段測試", "twostage")

figure("twostage", F4.fig_twostage(), 2, [
    "Stage 1 閉環：固定 N 個 client，每個回一個發一個。",
    "Stage 2 開環：依 Poisson 到達率發送，不管伺服器來不來得及。",
    "為什麼不能只做 Stage 1：閉環自帶背壓，延遲不會爆炸。真實使用者不會等你。"],
    "閉環量「機器最多能做多少」，開環量「能承諾多少」。兩個問題不一樣。", accent="ok")

p("這是負載測試最常見的方法論錯誤。大多數壓測工具預設是閉環：client 收到回覆才發下一個，",
  "等於在幫伺服器限流。系統看起來永遠很穩，因為它根本沒有機會過載。")

widget("sweep2", "教學示例曲線，以排隊理論生成，不是實測結果。"
                 "低於容量時吞吐等於到達率、延遲平穩；一超過，延遲就以 1/(1−ρ) 爆炸。")

p("ρ = 到達率 ÷ 服務率是利用率。M/M/1 的等待時間正比於 1/(1−ρ)，",
  "所以 ρ = 0.9 時延遲是空載的 10 倍，ρ = 0.95 時是 20 倍。",
  "容量規劃要留 20–30% 餘裕的理由就在這。")

figure("slo", F4.fig_slo(), 5, [
    "橫軸是到達率，縱軸是 p95 TTFT。",
    "先畫出 SLO 那條水平線。沒講定 SLO，就談不上容量。",
    "① 粗掃：大步長找出「還好」與「已經爆」之間的區間。",
    "② 細掃：在區間內二分逼近交點。",
    "③ 重複確認：邊界點至少跑 3 次，確認可重現、量測窗內沒有趨勢。"],
    "SLO 容量是一個邊界值，要用搜尋找出來，而且必須驗證可重現。", accent="ok")

h3("從 RPS 換算成人數", "little")

p("最後一步是把 req/s 換成「可以服務幾個人」。這一步的數學很簡單，但前提很多。")

widget("little", "λ = goodput ÷ 平均輸出長度；人數 = λ ÷ 每人的到達率。"
                 "Little's Law（L = λW）在這裡的用途是驗算：算出來的 L 應該小於等於 max-num-seqs。")

note("""
<p><b>四個前提，缺一不可</b>　系統穩定（到達率小於服務率、佇列不成長）；
輸出長度分布與量測窗一致；在該吞吐下 SLO 仍然滿足；使用者行為模型（每小時幾輪）有依據。
任何一項不成立，這個人數就不能引用。</p>
<p>另外要用 <b>goodput</b> 而不是 throughput。系統過載時 throughput 可能還很高，
但沒有一個人滿足 SLO，此時 goodput 是 0。</p>
""", "warn")

h3("實際的命令", "commands")

p("vLLM 的 <code>bench serve</code> 兩種模式都支援。閉環用 ",
  "<code>--max-concurrency</code>，開環用 <code>--request-rate</code> 搭配 ",
  "<code>--burstiness 1.0</code>（Poisson）。", cite("bench", "gp"))

code("""<span class="c"># Stage 1：閉環飽和測試（併發掃描 1 2 4 8 16 32 64 128）</span>
vllm bench serve --backend openai-chat \\
  --base-url http://HOST:8000 --endpoint /v1/chat/completions \\
  --model MODEL --dataset-name random \\
  --random-input-len 2048 --random-output-len 512 --num-prompts 600 \\
  <span class="hl">--max-concurrency 32</span> \\
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \\
  --save-result --result-filename c32.json

<span class="c"># Stage 2：開環 SLO 容量測試（到達率粗掃 → 細掃 → 邊界重複 3 次）</span>
vllm bench serve --backend openai-chat \\
  --base-url http://HOST:8000 --endpoint /v1/chat/completions \\
  --model MODEL --dataset-name random \\
  --random-input-len 2048 --random-output-len 512 --num-prompts 1200 \\
  <span class="hl">--request-rate 12 --burstiness 1.0</span> \\
  <span class="hl">--goodput ttft:800 tpot:40</span> \\
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \\
  --save-result --result-filename r12.json""")

p("兩個對照組是必要的：開 spec 對關 spec（其餘完全相同），以及不同 k。",
  "沒有對照組就無法歸因。另外每一輪之間要重啟服務或清空 prefix cache，",
  "否則後面的測試點會因為快取命中而虛胖。")

# ============================================== 9. 上線設定 ==============
h2("PART 09", "我們最後怎麼設定", "config")

p("把前面的分析收斂成兩組設定。都是從官方模型卡改的，只把草稿數換成我們評估要用的值。",
  cite("nv4", "df2"))

code("""<span class="c"># 122B · NVFP4 + MTP k=3　容器：nvcr.io/nvidia/vllm:26.04-py3</span>
vllm serve nvidia/Qwen3.5-122B-A10B-NVFP4 \\
  --quantization <span class="s">modelopt_fp4</span> \\
  --kv-cache-dtype <span class="s">fp8</span> \\
  --tensor-parallel-size <span class="n">1</span> \\
  --max-model-len <span class="n">65536</span> \\
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \\
  --speculative-config '<span class="hl">{"method":"qwen3_next_mtp","num_speculative_tokens":3}</span>'

<span class="c"># 27B · FP8 + DFlash2 k=7（block size 8 → 7 個草稿 + 1 個 anchor）</span>
vllm serve Qwen/Qwen3.8-27B-FP8 \\
  --max-model-len <span class="n">65536</span> \\
  --kv-cache-dtype <span class="s">auto</span> \\
  --reasoning-parser qwen3 \\
  --speculative-config '<span class="hl">{"method":"dflash",
     "model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}</span>'""")

p("<code>--max-model-len</code> 我填 65536 而不是原生的 262144，",
  "因為前面算過，跑滿 262k 單卡只放得下個位數的序列。",
  "實務上應該依 workload 的 p99 context 長度設定，留一點餘裕就好。")

note("""
<p><b>執行前一定要做的事</b>　跑一次 <code>vllm serve --help</code> 與
<code>vllm bench serve --help</code>，核對你安裝的那個版本的旗標名稱。
speculative decoding 的介面最近幾個版本改動頻繁，DFlash 在 vLLM 走 Speculators 函式庫，
要確認版本支援。</p>
""", "warn")

# ============================================== 10. 收尾 =================
h2("PART 10", "十個可以帶走的結論", "takeaways")

ul([
    "<b>decode 是 memory-bound。</b>一次 forward 讀完整份權重，只服務 B 個 token。"
    "所有最佳化都在提高「每次讀取分攤到的 token 數」。",
    f"<b>B200 的屋頂線轉折點是 N* ≈ {KNEE:.0f} token/step，而且與精度無關。</b>"
    "量化讓兩邊同時變快，不會改變你的相對位置。",
    f"<b>混合架構把 KV cache 砍掉 3/4。</b>122B 的 KV 只有 "
    f"{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB/token（FP8），"
    f"Qwen3-32B 是 {Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB。",
    f"<b>SSM state 是每序列 ~{mib(Q35.ssm_bytes(0),0)} MiB 的固定成本。</b>"
    f"短對話時它主導記憶體，122B 要到 {Q35.crossover_tokens('fp8',3):,.0f} token "
    "KV 才追上它。",
    f"<b>MoE 省算力不省記憶體，而且 batch 一大就要讀滿所有 expert。</b>"
    f"122B 從 batch 1 的 {gb(Q35.step_bytes(1))} GB 長到高併發的 "
    f"{gb(Q35.step_bytes(2000))} GB。",
    "<b>NVFP4 只量化了 routed expert。</b>36 層 GDN、12 層 attention、lm_head、"
    "MTP head 全部還是 BF16，佔 checkpoint 的 21.9% 與 batch-1 頻寬的一半。",
    f"<b>NVFP4 讓 122B 單卡跑得動</b>（{gb(Q35.real_weight_bytes)} GB vs "
    f"BF16 的 {gb(Q35.real_bf16_bytes)} GB），可以用 DP 取代 TP，省掉 all-reduce。",
    f"<b>speculative decoding 的效益由 B × (k+1) 是否超過 N* 決定。</b>"
    f"27B 配 DFlash2 k=7 → 安全併發約 {KNEE/8:.0f}；超過就開始賠。",
    "<b>接受率不是常數。</b>context 從 2k 長到 30k，實測平均接受率從 65% 掉到 39%，"
    "吞吐從 +129% 變成 −51%。",
    "<b>容量是一個條件式，不是一個數字。</b>"
    "f(模型版本, 量化, 引擎設定, workload, SLO, 硬體)。少一個自變數就不能引用。",
], ordered=True)

h3("這篇沒有回答的問題", "limits")

p("誠實列一下邊界。以下這些必須自己量，這篇文章給不了：",
  "兩個模型在你的 workload 上的實際吞吐與延遲曲線；speculative decoding 的實際接受率；",
  "開關 spec 的 goodput 對照；prefix cache 命中率的變化；量化對你自己驗收集的影響；",
  "長 context（&gt; 32k）的接受率與吞吐。")

p("刻意沒有涵蓋的主題：訓練與微調、多模態的容量影響、P/D 分離、多節點與網路拓撲、",
  "成本模型、模型品質的橫向評比。")

note(f"""
<p><b>建議的下一步</b>　用 PART 08 的兩階段流程對兩個模型各跑一次基準測試，
把結果填回 PART 06 與 PART 07 的試算器，看看解析模型與實測差多少。</p>
<p>解析上限通常是實測的 1.7–3 倍。知道自己的比例之後，這些試算器就變成很好用的
快速估算工具 —— 換設定時先估一次，再決定要不要花時間實測。</p>
""", "key")

# ============================================== 附錄 =====================
h2("APPENDIX", "附錄", "appendix")

h3("這些數字怎麼來的（可複現）", "repro")

p("所有架構數字都由官方 config 逐張量推導，再與 HuggingFace 上 safetensors 的 dtype 統計對帳。",
  "三個模型的誤差分別是 ",
  f"{abs(Q35.params_total/(Q35.real_bf16_bytes/2)-1)*100:.3f}%、",
  f"{abs(Q38.params_total/(Q38.real_bf16_bytes/2)-1)*100:.3f}% 與 ",
  f"{abs(Q332.params_total/(Q332.real_bf16_bytes/2)-1)*100:.3f}%。")

code("""<span class="c"># 抓官方 config 與 safetensors 索引</span>
python research/fetch_more.py

<span class="c"># 解析模型的自我檢查（參數量、cache、roofline、spec decoding）</span>
python v2/model.py

<span class="c"># 重新產生這篇文章與簡報版</span>
python v2/blog.py
python v2/build.py""")

table(["模型", "~解析推導參數", "~checkpoint 實際", "~誤差", "~語言模型", "~MTP head", "~視覺塔"], [
    [Q35.label, f"{Q35.params_total:,}", f"{int(Q35.real_bf16_bytes/2):,}",
     f"{abs(Q35.params_total/(Q35.real_bf16_bytes/2)-1)*100:.3f}%",
     f"{Q35.params_lang/1e9:.2f} B", f"{Q35.params_mtp/1e9:.2f} B",
     f"{Q35.params_vision/1e9:.3f} B"],
    [Q38.label, f"{Q38.params_total:,}", f"{int(Q38.real_bf16_bytes/2):,}",
     f"{abs(Q38.params_total/(Q38.real_bf16_bytes/2)-1)*100:.3f}%",
     f"{Q38.params_lang/1e9:.2f} B", f"{Q38.params_mtp/1e9:.2f} B",
     f"{Q38.params_vision/1e9:.3f} B"],
    [Q332.label, f"{Q332.params_total:,}", f"{int(Q332.real_bf16_bytes/2):,}",
     f"{abs(Q332.params_total/(Q332.real_bf16_bytes/2)-1)*100:.3f}%",
     f"{Q332.params_lang/1e9:.2f} B", "—", "—"],
], "compact")

h3("checkpoint revision", "revisions")

code("""nvidia/Qwen3.5-122B-A10B-NVFP4   98915d837c4e7c87ac8296d02e89de19b3207e6d
Qwen/Qwen3.8-27B-FP8            017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
incoai/Qwen3.8-27B-DFlash2      dedf8df68adfb1afeaf7b7480c0a0243108177b4
Qwen/Qwen3.5-122B-A10B          dc4d348443bc740c68e2d77492492c11606384d5
Qwen/Qwen3.8-27B                1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Qwen/Qwen3-32B                  9216db5781bf21249d130ec9da846c4624c16137""")

h3("名詞表", "glossary")

w('<dl class="gl">' + "".join(
    f"<div><dt>{k}</dt><dd>{v}</dd></div>"
    for k, v in sorted(K.GLOSSARY.items(), key=lambda x: x[0].lower())) + "</dl>")

h2("REFERENCES", "參考文獻", "refs")
p("查核日期 ", K.CHECK, "。HuggingFace 上的模型會更新，引用時請連同上面的 revision 一起寫。")
w("@@REFS@@")

w("""<footer class="end">
<p>這篇文章裡的圖表全部是為它重新繪製的 SVG，沒有引用第三方圖片。
所有數字的推導程式碼與原始 config 都在同一個 repo。</p>
<p>投影片版本（82 頁、含講稿與逐頁來源）也在 repo 裡。</p>
</footer>""")
