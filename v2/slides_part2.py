# -*- coding: utf-8 -*-
# ================================================ CH2 vLLM 的執行堆疊 =====
chapter("CH2 vLLM 把它變成一個服務", "compute")

slide("vLLM 把它變成一個服務", kind="section", body="""
<div class="no">CHAPTER 02</div>
<h1>vLLM 把它變成一個服務</h1>
<div class="csub">模型只回答「這一個 token 是什麼」。要變成能同時服務數百人的服務，
中間這一層要解決排程、記憶體分頁與批次組合三個問題。</div>
<ul>
 <li>從 HTTP 到 GPU kernel 的責任邊界</li>
 <li>Continuous batching：每一輪重新組隊</li>
 <li>PagedAttention：把 KV cache 當成作業系統的分頁</li>
 <li>Scheduler 一輪實際在做的四個決定</li>
 <li>Chunked prefill 與 prefix caching</li>
 <li>換模型的時候，哪一層真的會動</li>
</ul>""",
      notes="<p>這一章要讓「調參數」變得有跡可循：知道每個旗標作用在哪一層。</p>")

slide("vLLM 的執行堆疊", "六層責任邊界，換模型時只有最下面兩層需要改",
      sources=["arch", "anat"], body=f"""
{gist("上面三層是「服務」，下面三層是「模型」。看懂這條界線，就知道每個 CLI 旗標會影響什麼。")}
{fig("stack", F2.fig_stack(), 6, [
 "先看整體形狀：一個 request 從上往下穿過六層。",
 "<b>API server</b> 負責 HTTP、chat template、tokenize 與串流回傳。",
 "<b>EngineCore 排程器</b> 是全場最重要的角色：決定每一輪誰上車、各帶幾個 token。",
 "<b>KV cache manager</b> 管 block table 與空閒池；混合模型還要多管一個 SSM state pool。",
 "<b>Executor / Worker</b> 處理分散式；單卡時只是一層薄殼。",
 "<b>Model runner</b> 組 batch metadata 並重播 CUDA graph。",
 "<b>Kernels</b> 才是真正動手的地方，量化 kernel 就掛在這一層。"], accent="compute")}""",
      notes="""<p>對照官方文件的說法：引擎只要有 request 就反覆呼叫 <code>step()</code>，
每個 step 分成三段：schedule、forward pass、postprocess。</p>
<p>調參對照表：<code>--max-num-seqs</code> 與 <code>--max-num-batched-tokens</code> 作用在第二層；
<code>--gpu-memory-utilization</code>、<code>--kv-cache-dtype</code>、<code>--enable-prefix-caching</code>
作用在第三層；<code>--tensor-parallel-size</code> 在第四層；<code>--enforce-eager</code> 在第五層；
<code>--quantization</code> 在第六層。</p>""")

slide("Continuous batching：空出來的位置立刻補上", "同一次權重讀取被更多 token 分攤",
      sources=["anat", "paged"], body=f"""
{gist("靜態 batching 要等最慢的那個人講完；continuous batching 每一輪重新組隊，所以 GPU 幾乎不空轉。")}
{fig("batch", F2.fig_batching(), 4, [
 "上半是靜態 batching：四個 request 一起送進去，短的做完了也只能等。",
 "紅色格子是純粹的浪費，而且算力和頻寬一起浪費。",
 "下半是 continuous batching：做完就離開，位置立刻讓給新 request。",
 "所以同一份權重讀取被更多 token 分攤，右移到屋頂線上更划算的位置。"], accent="ok")}
<div class="hint">下一頁附上 vLLM 官方部落格的原始示意圖作對照。</div>""",
      notes="""<p>這是把 CH1 的結論落地的第一個機制。要強調 continuous batching 的重點不在「把 batch 開大」，
而在「讓 batch 的成員每一輪都可以換」。</p>
<p>實務上它帶來一個副作用：同一個 request 的 ITL 會抖動，因為每一輪的同伴數量不同。
CH8 討論百分位時會回來處理這件事。</p>""")

slide("原始示意圖對照", "vLLM 官方部落格的 batching 策略圖",
      sources=["anat", "paged"], body=f"""
<div class="cols w6-4">
 <div>{img("continuous_batching", "圖片由使用者提供，出自 vLLM 官方部落格 "
   "《Inside vLLM》。三種策略由上到下：individual requests、dynamic batching、"
   "continuous batching。")}</div>
 <div>
  {pane("三種策略的差別", '''<table class="compact"><tbody>
  <tr><td><b>Individual</b></td><td>一次一個 request。權重讀取沒有分攤，最浪費。</td></tr>
  <tr><td><b>Dynamic</b></td><td>湊滿一批才送，整批一起結束。有分攤，但尾端空轉。</td></tr>
  <tr><td><b>Continuous</b></td><td>每輪重組。分攤最大化，GPU 幾乎不空轉。</td></tr>
  </tbody></table>''')}
  {take("這張圖與前一頁的自製動畫講的是同一件事。放在一起是為了讓你之後看到官方文件時能對得上。")}
 </div>
</div>""",
      notes="<p>時間緊的話這一頁可以跳過，前一頁的動畫已經講完了。</p>")

slide("PagedAttention：把 KV cache 當成作業系統的分頁",
      "邏輯連續、實體不連續，碎片幾乎消失",
      sources=["paged", "anat"], body=f"""
{gist("KV cache 切成固定大小的 block，用 block table 對應。碎片沒了，前綴還能共用。")}
{fig("paged", F2.fig_paged(), 3, [
 "邏輯上，每個 request 就是一串連續的 token。",
 "實體上，KV 被切成固定大小的 block，散落在 HBM 各處；用完的 block 回到空閒池。",
 "<b>block table</b> 記住邏輯到實體的對應，就像作業系統的分頁表。",
 "換來三個能力：沒有碎片、前綴可共用、request 可以被換出再換回。"], accent="mem")}""",
      notes="""<p>在 PagedAttention 之前，系統必須為每個 request 預留「最壞情況」的連續空間，
實際利用率常常只有 20–40%。分頁之後浪費只剩下最後一塊的尾巴。</p>
<p>對混合模型有一個重要的後果：GDN 的 state 大小固定，不適合用 token 為單位分頁。
vLLM 的做法是把 attention 層的 block_size 撐大到能塞得下一份 mamba state，
再把兩者當成同一種「頁」來管理。CH4 會展開。</p>""")

slide("原始示意圖對照", "vLLM 官方部落格的 PagedAttention 動畫",
      sources=["paged"], body=f"""
<div class="cols w7-3">
 <div>{img("paged_attention", "圖片由使用者提供，出自 vLLM PagedAttention 發表文。"
   "動畫展示兩個共用同一段 prompt 的 request，如何對應到相同的 physical KV block。")}</div>
 <div>
  {pane("看這張圖時注意兩件事", '''<ul style="font-size:13.5px">
  <li>兩個 request 的 logical block 0 指向<b>同一個</b> physical block，
   prefix caching 就是靠這個做的。</li>
  <li>Logical 與 physical 的順序完全無關；順序只存在 block table 裡。</li>
  </ul>''', "mem")}
  {warn("這張是 GIF，會自動播放，與自製動畫的「下一步」按鈕分開。")}
 </div>
</div>""",
      notes="<p>GIF 保留原始播放速度。若在會議室投影，可以先停在這頁讓聽眾看兩輪。</p>")

slide("Scheduler 一輪做的四個決定", "token budget 怎麼被切開，決定了 TTFT 與 TPOT 的取捨",
      sources=["anat", "arch"], body=f"""
{gist("排程器先保住正在生成的人的 decode，再用剩下的 budget 去餵新來的 prefill。這個順序就是 TPOT 優先於 TTFT。")}
{fig("sched", F2.fig_scheduler(), 4, [
 "每一輪有一個 token 預算（<code>--max-num-batched-tokens</code>，示例取 8,192）。",
 "<b>①</b> 先排 running 佇列裡的 decode：32 個 request 只吃掉 32 個 token，非常便宜。",
 "<b>②</b> 剩下的 budget 拿去餵 prefill；長 prompt 被切成 chunk 分次送。",
 "<b>③</b> 若 cache 配不出來，就從尾端 preempt request 換出。",
 "<b>④</b> 最後把 batch metadata 交給 model runner。"], accent="compute")}
<div class="cols3" style="margin-top:2px">
 {pane("為什麼 decode 優先", '<p style="font-size:13.5px">已經在串流的使用者如果卡住，'
       '體感是「打字停住」，比多等一點 TTFT 難忍受得多。</p>', "mem")}
 {pane("budget 開大開小", '<p style="font-size:13.5px">開大 → prefill 吃更多，TTFT 好、'
       '正在生成的人 ITL 變抖。開小 → 反之。這是最直接的 TTFT/TPOT 旋鈕。</p>', "compute")}
 {pane("preempt 的代價", '<p style="font-size:13.5px">換出的 request 之後要重算或搬回，'
       '對混合模型還要一併處理 SSM state。頻繁 preempt 是容量過載的明顯訊號。</p>', "bad")}
</div>""",
      notes="""<p>這一頁把 CH8/CH9 要量的東西先接起來：TTFT 與 TPOT 不是兩個獨立指標，
它們在排程器裡共用同一個 token budget，是一個蹺蹺板。</p>
<p>觀察指標：如果 <code>num_preempted</code> 持續 &gt; 0，代表 cache 不夠，
應該降 max_num_seqs 或縮 max_model_len，而不是繼續加壓。</p>""")

slide("Chunked prefill 與 prefix caching", "兩個最容易被低估的旗標",
      sources=["anat", "hyb"], body=f"""
{gist("chunked prefill 讓長 prompt 不會卡住別人；prefix caching 讓共用開頭的 prompt 只算一次。")}
<div class="cols">
 {pane("Chunked prefill", f'''
 <p style="font-size:13.8px">把長 prompt 的 prefill 切成小塊，分散到多個排程輪次。</p>
 <ul style="font-size:13.3px">
  <li>沒有它：一個 128k 的 prompt 會獨占好幾個 step，同時在線的人 ITL 出現長尖峰。</li>
  <li>有它：每一輪都能混入 decode，尖峰被抹平。</li>
  <li>副作用：那個長 prompt 自己的 TTFT 會變長，這是刻意的取捨。</li>
  <li>旗標：<code>--long-prefill-token-threshold</code> 與 budget 一起決定切多細。</li>
 </ul>''', "compute")}
 {pane("Prefix caching", f'''
 <p style="font-size:13.8px">不同 request 共用的開頭（system prompt、few-shot、
 對話歷史）只算一次，KV block 直接重用。</p>
 <ul style="font-size:13.3px">
  <li>逐 block 做雜湊；命中就跳過那段 prefill。</li>
  <li>對 agent／多輪對話效果最大，常見命中率 60–90%。</li>
  <li>混合模型：full attention 由左往右找命中，兩種 group 取交集；
   vLLM 對 mamba / GDN 的 prefix caching <b>仍在開發中</b>。</li>
  <li>{T("Speculative decoding")} 會干擾命中率，CH7 有實測案例。</li>
 </ul>''', "mem")}
</div>
{quote("Chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks.",
       "Inside vLLM: Anatomy of a High-Throughput LLM Inference System, 2025-09-05")}""",
      notes="""<p>這兩個功能在新版 vLLM 預設都開著，但值得知道它們的代價。特別是 prefix caching：
它會讓 benchmark 數字大幅變好，如果測試資料的前綴重複度和線上不一樣，容量估計就會失真。
CH9 的測試設定表會要求記錄命中率。</p>""")

slide("換模型的時候，哪一層真的會動？", "分清「同架構換權重」與「換架構」的成本差距",
      sources=["arch", "gdnc"], body=f"""
{gist("同架構換 checkpoint 幾乎零成本；換 token mixer 就要動到 kernel 與 cache 管理，是完全不同的工程量。")}
{table(["層級", "同架構、換 checkpoint（例如換量化版）", "換架構（例如 Transformer → 混合）"], [
 ["REST API", "不動", "不動；只有新的模態或輸出解析器才需要"],
 ["Tokenizer / chat template", "多半沿用，但要核對 <code>tokenizer_config</code> 與模板",
  "多半沿用（Qwen 系列 vocab 相同）"],
 ["模型載入 / 權重對應", "要對應新的 tensor 名稱與量化 metadata", "要新的模型類別與並行切法"],
 ["排程器", "不動", "不動（介面相同）"],
 ["Cache 管理", "只變大小（KV dtype、層數）",
  "<b>要新增一種 cache</b>：SSM state pool、page size 對齊、prefix caching 規則"],
 ["GPU kernels", "要有對應的量化 kernel（例如 <code>modelopt_fp4</code>）",
  "要新的 token-mixer kernel（chunkwise GDN、state update）"],
 ["Speculative decoding", "起草器要與新 checkpoint 相容（vocab、hidden size）",
  "起草器也要能處理新的 cache 形狀"],
], "compact", hi=(4, 5))}
{take("實務結論：從 Qwen3.8-27B 換到 Qwen3.8-27B-FP8 是設定層面的事；"
      "從 Qwen3-32B 換到 Qwen3.5-122B-A10B 則需要引擎本身支援 GDN 與 hybrid cache manager。"
      "評估新模型時，先確認 vLLM 版本支不支援，再談效能。")}""",
      notes="""<p>決策會議上這一頁通常最實用。它把「支援一個新模型」拆成七個具體項目，
工作量與風險比較好估。</p>
<p>特別提醒 speculative decoding 那一列：起草器與目標模型是綁定的。
換了目標模型的量化版本，起草器不一定要重訓，但一定要重新量接受率，CH7 會說明為什麼。</p>""")

# ================================================== CH3 三個模型的架構 ====
chapter("CH3 三個模型的架構帳本", "moe")

slide("三個模型的架構帳本", kind="section", accent="moe", body="""
<div class="no">CHAPTER 03</div>
<h1>三個模型的架構帳本</h1>
<div class="csub">同一個 decoder 骨架，換掉兩個插槽就得到三種完全不同的成本結構。
這一章的每個數字都會回頭跟 checkpoint 的實際位元組數對帳。</div>
<ul>
 <li>Qwen3-32B：純 Transformer 基準線</li>
 <li>Qwen3.8-27B：混合 token mixer + dense FFN</li>
 <li>Qwen3.5-122B-A10B：混合 token mixer + MoE</li>
 <li>MoE 的算力與記憶體錯位：A10B 到底是什麼意思</li>
 <li>參數帳本：我們怎麼驗證這些數字沒有算錯</li>
</ul>""",
      notes="<p>這一章的重點在可驗證。所有參數都由 config 推導，再跟 HuggingFace 上的 "
            "safetensors dtype 統計核對，誤差 &lt; 0.02%。</p>")

slide("三個模型並排", "換掉 token mixer 與 channel mixer 兩個插槽",
      sources=["c332", "c38", "c35"], accent="moe", body=f"""
{gist("同一個骨架，三種取捨：32B 全部 full attention、27B 四層裡放一層、122B 也是四層放一層但 FFN 換成 MoE。")}
{fig("arch3", F2.fig_arch3(), 2, [
 "三個模型的層堆疊並排：左邊細長的方塊是 token mixer，右邊小方塊是 FFN。",
 "顏色說明：藍色是 full attention（需要 KV cache），青色是 Gated DeltaNet（固定大小狀態）。",
 "所以 KV/token 差了一個數量級。後面的記憶體帳本都從這裡長出來。"], accent="moe")}""",
      notes="""<p>Qwen3.8-27B 與 Qwen3.5-122B-A10B 的 layer_types 陣列在 config 裡是
「三個 linear_attention 接一個 full_attention」重複到底，<code>full_attention_interval = 4</code>。
27B 是 64 層 → 48 GDN + 16 full；122B 是 48 層 → 36 GDN + 12 full。</p>
<p>Qwen3-32B 則是 64 層全部 full attention，head_dim 128、KV head 8 組。</p>""")

slide("Qwen3-32B：純 Transformer 基準線", "所有比較的原點",
      sources=["c332", "q3r"], accent="compute", body=f"""
{gist(f"64 層全部 full attention。每個 token 要付 {Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB 的 KV cache，"
      f"比 122B 高 {Q332.kv_bytes_per_token('bf16')/Q35.kv_bytes_per_token('bf16'):.0f} 倍。")}
<div class="cols w6-4">
 <div>
  {table(["項目", "~數值", "說明"], [
   ["層數", f"{Q332.n_layers}", "全部 full attention，沒有 linear attention"],
   ["hidden size", f"{Q332.hidden:,}", ""],
   ["Q / KV head", f"{Q332.n_heads} / {Q332.n_kv}", f"GQA {Q332.n_heads//Q332.n_kv}:1"],
   ["head_dim", f"{Q332.head_dim}", "比 Qwen3.5 系列的 256 小一半"],
   ["FFN intermediate", f"{Q332.moe_per_layer//(3*Q332.hidden):,}", "Dense SwiGLU"],
   ["原生 context", f"{Q332.max_ctx:,}", "config 的 max_position_embeddings"],
   ["總參數", f"{Q332.params_total/1e9:,.2f} B", "解析值與 checkpoint 完全一致"],
   ["BF16 權重", f"{gb(Q332.real_weight_bytes)} GB", "safetensors 實際位元組數"],
  ], "compact")}
 </div>
 <div>
  {stats([(f"{Q332.kv_bytes_per_token('bf16')/KiB:.0f}", "KiB", "每個 token 的 KV cache（BF16）", "m"),
          (f"{Q332.max_ctx*Q332.kv_bytes_per_token('bf16')/GiB:.0f}", "GiB",
           f"單條序列跑滿 {Q332.max_ctx//1024}k context", "m")])}
  {pane("為什麼它的 KV 這麼貴", f'''<div class="formula" style="font-size:12.5px">
  2 × <em>64</em> 層 × <span class="c2">8</span> KV head × <span class="c3">128</span> dim × 2 B
  = {Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB</div>
  <p style="font-size:13.3px;margin-top:7px">關鍵是<b>層數</b>：64 層每一層都要存。
  混合架構把這個 64 降成 12 或 16，就直接把 KV 砍掉 3/4。</p>''', "mem")}
  {warn(f"在 B200 上，光是 {Q332.max_ctx//1024}k context 的一條序列就要 "
        f"{Q332.max_ctx*Q332.kv_bytes_per_token('bf16')/GiB:.0f} GiB。"
        f"扣掉 {gib(Q332.real_weight_bytes)} GiB 的權重後，單卡大約只放得下 8 條。")}
 </div>
</div>""",
      notes="""<p>把 Qwen3-32B 留在報告裡是刻意的：它提供一個「如果不用混合架構會怎樣」的對照。
所有跟長 context 有關的結論，都可以用它來凸顯差距。</p>
<p>注意它沒有 MTP head（checkpoint 裡沒有 mtp.* 張量），所以要做 speculative decoding
必須另外訓練起草器。這也是評估新模型時要問的問題之一。</p>""")

slide("Qwen3.8-27B：混合 token mixer + dense FFN", "本次要上線的 FP8 版本",
      sources=["c38", "m38", "f38"], accent="ssm", body=f"""
{gist(f"64 層裡有 {Q38.n_gdn} 層 GDN、只有 {Q38.n_full} 層 full attention。KV/token 因此降到 "
      f"{Q38.kv_bytes_per_token('bf16')/KiB:.0f} KiB，但多了固定 {mib(Q38.ssm_bytes(0))} MiB 的 SSM state。")}
<div class="cols w6-4">
 <div>
  {table(["項目", "~數值", "說明"], [
   ["層數", f"{Q38.n_layers}", f"{Q38.n_layers//4} × (3 × GDN → 1 × full attention)"],
   ["hidden size", f"{Q38.hidden:,}", ""],
   ["full attention", f"{Q38.n_full} 層", f"Q {Q38.n_heads} / KV {Q38.n_kv}，head_dim {Q38.head_dim}"],
   ["Gated DeltaNet", f"{Q38.n_gdn} 層", f"value head {Q38.nv}、key head {Q38.nk}、head dim {Q38.dv}"],
   ["FFN", f"{Q38.moe_per_layer//(3*Q38.hidden):,}", "Dense SwiGLU（不是 MoE）"],
   ["MTP head", "1 層", "checkpoint 內含 <code>mtp.*</code>，可直接做 spec decode"],
   ["原生 context", f"{Q38.max_ctx:,}", "YaRN 可外推到約 1 M"],
   ["總參數", f"{Q38.params_total/1e9:,.2f} B",
    f"語言 {Q38.params_lang/1e9:,.2f} B + MTP {Q38.params_mtp/1e9:,.2f} B + 視覺 {Q38.params_vision/1e9:,.2f} B"],
   ["<b>FP8 權重</b>", f"<b>{gb(Q38.real_weight_bytes)} GB</b>",
    f"= {gib(Q38.real_weight_bytes)} GiB（BF16 版是 {gb(Q38.real_bf16_bytes)} GB）"],
  ], "compact", hi=(8,))}
 </div>
 <div>
  {stats([(f"{Q38.kv_bytes_per_token('bf16')/KiB:.0f}", "KiB", "KV / token（BF16）", "m"),
          (f"{mib(Q38.ssm_bytes(0),0)}", "MiB", "SSM state / 序列（固定）", "s")])}
  {pane("兩條公式", f'''<div class="formula" style="font-size:12px">
  KV = 2 × <em>{Q38.n_full}</em> × <span class="c2">{Q38.n_kv}</span> × {Q38.head_dim} × 2 B
  = {Q38.kv_bytes_per_token('bf16')/KiB:.0f} KiB / token</div>
  <div class="formula" style="font-size:12px;margin-top:6px">
  state = <em>{Q38.n_gdn}</em> × <span class="c3">{Q38.nv}</span> × {Q38.dv} × {Q38.dk} × 4 B
  = {mib(Q38.ssm_recurrent_bytes(),0)} MiB / 序列</div>
  <p style="font-size:13px;margin-top:7px">上面那條隨 context 長大，下面那條<b>永遠不變</b>。
  CH6 會把兩者放在同一張帳本上。</p>''', "ssm")}
  {pane("這一版與第一版的差別", f'''<p style="font-size:13.3px">第一版報告用 BF16 假設，
  權重算 {gb(Q38.real_bf16_bytes)} GB。改成官方 FP8 版之後降到
  <b>{gb(Q38.real_weight_bytes)} GB</b>，省下 {gb(Q38.real_bf16_bytes-Q38.real_weight_bytes)} GB，
  這些空間全部可以拿去放 cache。</p>''', "compute")}
 </div>
</div>""",
      notes="""<p>27B 這個模型的 head_dim 是 256（很大），但只有 4 組 KV head、16 層 full attention，
所以 KV/token 只有 64 KiB。與 Qwen3-32B 的 256 KiB 相比是 1/4。</p>
<p>要提醒一件容易踩到的事：FP8 checkpoint 指的是<b>權重</b>是 FP8，
KV cache 的精度是另一個獨立的服務端旗標 <code>--kv-cache-dtype</code>，預設是 auto（BF16）。
本報告 27B 的 KV 預設用 BF16 計算，並在試算器提供 FP8 選項。</p>""")

slide("Qwen3.5-122B-A10B：混合 token mixer + MoE", "本次要上線的 NVFP4 版本",
      sources=["c35", "m35", "nv4"], accent="moe", body=f"""
{gist(f"122B 總參數、每個 token 只算 {Q35.activated/1e9:.1f}B。KV/token 只有 "
      f"{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB（FP8），是三個模型裡最省的。")}
<div class="cols w6-4">
 <div>
  {table(["項目", "~數值", "說明"], [
   ["層數", f"{Q35.n_layers}", f"{Q35.n_layers//4} × (3 × (GDN→MoE) → 1 × (Attention→MoE))"],
   ["hidden size", f"{Q35.hidden:,}", "比 27B 小，但層數少、寬度靠 MoE 補"],
   ["full attention", f"{Q35.n_full} 層", f"Q {Q35.n_heads} / KV {Q35.n_kv}（GQA 16:1），head_dim {Q35.head_dim}"],
   ["Gated DeltaNet", f"{Q35.n_gdn} 層", f"value head {Q35.nv}、key head {Q35.nk}、head dim {Q35.dv}"],
   ["MoE", f"{Q35.n_experts} experts", f"每 token 選 {Q35.top_k} 個 + 1 shared，"
                                       f"intermediate {Q35.moe_inter:,}"],
   ["MTP head", "1 層", f"{Q35.params_mtp/1e9:,.2f} B 參數（自己也是一層 MoE）"],
   ["原生 context", f"{Q35.max_ctx:,}", "YaRN 可外推到約 1 M"],
   ["總參數", f"{Q35.params_total/1e9:,.2f} B",
    f"語言 {Q35.params_lang/1e9:,.2f} B ← 型號裡的 122B"],
   ["每 token 啟用", f"{Q35.activated/1e9:,.2f} B", "← 型號裡的 A10B"],
   ["<b>NVFP4 權重</b>", f"<b>{gb(Q35.real_weight_bytes)} GB</b>",
    f"= {gib(Q35.real_weight_bytes)} GiB（BF16 版是 {gb(Q35.real_bf16_bytes)} GB）"],
  ], "compact", hi=(9,))}
 </div>
 <div>
  {stats([(f"{Q35.kv_bytes_per_token('fp8')/KiB:.0f}", "KiB", "KV / token（FP8 KV cache）", "m"),
          (f"{mib(Q35.ssm_bytes(0),0)}", "MiB", "SSM state / 序列（固定）", "s")])}
  {pane("能塞進單張 B200 是重點", f'''<p style="font-size:13.3px">BF16 要
  {gb(Q35.real_bf16_bytes)} GB，一張 180 GB 的卡放不下，至少要兩張。
  NVFP4 之後只要 <b>{gb(Q35.real_weight_bytes)} GB</b>，
  官方模型卡的部署範例就是 <code>--tensor-parallel-size 1</code>。</p>
  <p style="font-size:13.3px">少了 TP 的 all-reduce，decode 的延遲與抖動都會明顯改善。</p>''',
  "compute")}
  {warn(f"但別忘了 MTP head：{Q35.params_mtp/1e9:.2f} B 參數在 NVFP4 checkpoint 裡"
        f"<b>沒有被量化</b>，BF16 佔 {gb(Q35.params_mtp*2)} GB。要做 speculative decoding "
        f"就得把這塊算進帳本。")}
 </div>
</div>""",
      notes="""<p>這一頁要強調三件事：<br>
① 122B 是總參數，A10B 是每 token 啟用量，兩者用在不同的計算裡（記憶體用前者、算力用後者）。<br>
② NVFP4 讓它變成單卡可跑，這是這次評估最重要的工程結論。<br>
③ MTP head 有 2.52 B 參數而且沒被量化，會吃掉 5 GB。第一版報告完全沒提到這件事。</p>""")

slide("MoE：算力與記憶體的錯位", "少算一些 expert，但整組權重都得放在卡上",
      sources=["m35", "c35", "nvi"], accent="moe", body=f"""
{gist("MoE 用「稀疏算力」換「密集記憶體」。它讓模型變聰明而不變慢，但不會讓模型變小。")}
{fig("moe", F2.fig_moe(), 5, [
 "一個 token 進到 MoE 層，先過 router。",
 "router 對 256 個 expert 打分數。",
 "只有分數最高的 8 個（紫色）會被計算，另外加上 1 個所有 token 都用的 shared expert。",
 "算力面：每 token 只啟用 9.0 B 參數，型號裡的「A10B」講的就是這個。",
 "記憶體面：256 個 expert <b>全部</b>都得放在 HBM 裡，佔了整個 checkpoint 的 69%。",
 "而且 batch 一大，被選到的 expert 幾乎涵蓋全部 256 個，記憶體流量從 2 GB 長到 65 GB。"],
 accent="moe")}""",
      notes="""<p>最後一步是這一版新增的關鍵洞見，也是 CH7 的伏筆。用機率算一下：
每個 token 選 8/256，N 個 token 之後被碰到的 expert 期望值是
256 × (1 − (1 − 8/256)^N)。N=1 是 8 個、N=32 是 164 個、N=128 是 251 個、N≈290 就幾乎全部。</p>
<p>這代表 MoE 在 batch 1 極省頻寬（只讀 12.8 GB），但在高併發時要讀滿 76 GB，
step 時間下限變成 9.9 ms。這對 speculative decoding 的效益評估影響很大。</p>
<p>注意：這個模型假設 routing 均勻。真實 routing 有偏好（熱門 expert），
實際碰到的 expert 數會比公式略少，所以這是上界。</p>""")

slide("參數帳本：我們怎麼確定沒算錯", "解析公式 vs checkpoint 實際位元組數",
      sources=["nvi", "i38", "c332"], accent="ok", body=f"""
{gist("每個模型的參數量都用 config 逐張量算一次，再與 HuggingFace 上 safetensors 的 dtype 統計對帳。誤差都在 0.02% 以內。")}
{table(["模型", "~解析推導參數", "~checkpoint 實際參數", "~誤差", "~語言模型", "~MTP head", "~視覺塔"], [
 [Q35.label, f"{Q35.params_total:,}", f"{int(Q35.real_bf16_bytes/2):,}",
  f"{abs(Q35.params_total/(Q35.real_bf16_bytes/2)-1)*100:.3f}%",
  f"{Q35.params_lang/1e9:.2f} B", f"{Q35.params_mtp/1e9:.2f} B", f"{Q35.params_vision/1e9:.3f} B"],
 [Q38.label, f"{Q38.params_total:,}", f"{int(Q38.real_bf16_bytes/2):,}",
  f"{abs(Q38.params_total/(Q38.real_bf16_bytes/2)-1)*100:.3f}%",
  f"{Q38.params_lang/1e9:.2f} B", f"{Q38.params_mtp/1e9:.2f} B", f"{Q38.params_vision/1e9:.3f} B"],
 [Q332.label, f"{Q332.params_total:,}", f"{int(Q332.real_bf16_bytes/2):,}",
  f"{abs(Q332.params_total/(Q332.real_bf16_bytes/2)-1)*100:.3f}%",
  f"{Q332.params_lang/1e9:.2f} B", "—", "—"],
], "compact")}
<div class="cols w6-4" style="margin-top:2px">
 {pane("逐層拆解（每一層的參數量）", table(
   ["模型", "~GDN 層", "~attention 層", "~FFN / MoE 層", "~conv_dim"], [
    [Q35.short, f"{Q35.gdn_per_layer/1e6:,.1f} M", f"{Q35.attn_per_layer/1e6:,.1f} M",
     f"{Q35.moe_per_layer/1e6:,.0f} M", f"{Q35.conv_dim:,}"],
    [Q38.short, f"{Q38.gdn_per_layer/1e6:,.1f} M", f"{Q38.attn_per_layer/1e6:,.1f} M",
     f"{Q38.moe_per_layer/1e6:,.1f} M", f"{Q38.conv_dim:,}"],
    [Q332.short, "—", f"{Q332.attn_per_layer/1e6:,.1f} M",
     f"{Q332.moe_per_layer/1e6:,.1f} M", "—"],
   ], "compact"))}
 {pane("為什麼要做這件事", '''<p style="font-size:13.3px">因為後面每一張記憶體帳本、
 每一條 roofline 都建立在這些數字上。如果參數量算錯 10%，容量結論就會錯 10%。</p>
 <p style="font-size:13.3px">對帳方式：把 config 裡的每個維度乘出來，
 逐個對應 checkpoint 的 tensor 名稱（<code>in_proj_qkv</code>、
 <code>mlp.experts.*</code>、<code>mtp.*</code>…），再與 HF API 回報的
 BF16 / F8_E4M3 / U8 元素數比對。</p>''', "ok")}
</div>""",
      notes="""<p>這一頁是報告的可信度基礎。實際做法寫在附錄 B。三個模型的誤差分別是
0.009%、0.018%、0.000%（Qwen3-32B 完全一致，因為它結構最單純）。</p>
<p>0.02% 的殘差來自 MTP head 內部幾個 norm 與 fc 的細節，不影響任何結論。</p>""")

# ================================================ CH4 Gated DeltaNet ======
chapter("CH4 Gated DeltaNet：用固定狀態換掉 KV cache", "ssm")

slide("Gated DeltaNet：用固定狀態換掉 KV cache", kind="section", accent="ssm", body="""
<div class="no">CHAPTER 04</div>
<h1>Gated DeltaNet</h1>
<div class="csub">混合架構的核心。把「保存全部歷史 K/V」換成「維護一個固定大小的矩陣」，
記憶體從隨 context 線性成長變成常數。代價是這個矩陣屬於有損壓縮。</div>
<ul>
 <li>linear attention 的核心想法</li>
 <li>state update 的數學：decay gate α 與 delta rule β</li>
 <li>與 Mamba2 / DeltaNet 的關係</li>
 <li>hybrid 3:1 的一次 decode：兩種 cache 同步前進</li>
 <li>chunkwise prefill：長 prompt 怎麼吃</li>
 <li>vLLM 的 Hybrid KV Cache Manager 怎麼同時管兩種 cache</li>
</ul>""",
      notes="<p>全報告技術密度最高的就是這一章。聽眾偏工程的話可以多花點時間。</p>")

slide("Linear attention 的核心想法", "把「查表」換成「維護一個摘要」",
      sources=["gdn", "gdnc"], accent="ssm", body=f"""
{gist("full attention 保存每一個位置；linear attention 只保存一個固定大小的摘要矩陣。")}
<div class="cols">
 {pane("Full attention", f'''
 <div class="formula" style="font-size:13px">oₜ = Σ<sub>i≤t</sub> softmax(qₜ·kᵢ) vᵢ</div>
 <ul style="font-size:13.3px;margin-top:8px">
  <li>必須保留<b>每一個</b> kᵢ、vᵢ → KV cache 隨 T 線性成長。</li>
  <li>讀取量也隨 T 線性成長 → 長 context 的 decode 越跑越慢。</li>
  <li>好處：資訊<b>無損</b>。任何位置都能被精確查到。</li>
 </ul>''', "compute")}
 {pane("Linear attention（GDN）", f'''
 <div class="formula" style="font-size:13px">Sₜ = f(Sₜ₋₁, kₜ, vₜ)　·　oₜ = Sₜ qₜ</div>
 <ul style="font-size:13.3px;margin-top:8px">
  <li>只保留一個 dᵥ×dₖ 的矩陣 S → 記憶體<b>固定</b>。</li>
  <li>讀取量也固定 → decode 速度不隨 context 變慢。</li>
  <li>代價：S 是<b>有損壓縮</b>。寫進去的東西會互相覆蓋。</li>
 </ul>''', "ssm")}
</div>
{table(["", "Full attention", "Gated DeltaNet"], [
 ["每條序列的記憶體", "O(T)　隨 context 線性成長", "O(1)　常數"],
 ["decode 的讀取量", "O(T)", "O(1)"],
 ["prefill 的計算", "O(T²)（可分塊）", "O(T)（chunkwise 平行）"],
 ["資訊保真度", "無損", "有損，越長越糊"],
 ["Qwen 的用法", f"每 4 層放 1 層（{Q35.n_full}/{Q35.n_layers} 或 {Q38.n_full}/{Q38.n_layers}）",
  f"其餘 3/4（{Q35.n_gdn} 或 {Q38.n_gdn} 層）"],
], "compact")}
{take("混合架構的想法很直接：用 3/4 的層享受 O(1) 的好處，"
      "留 1/4 的層做無損查表把細節補回來。現在長 context 模型多半這樣做。")}""",
      notes="""<p>可以用一個類比：full attention 像逐字稿，linear attention 像會議摘要。
逐字稿什麼都查得到但越來越厚；摘要永遠一頁，但細節會流失。
混合架構等於「每四頁摘要就附一頁逐字稿」。</p>""")

slide("GDN 的一次 state update", "decay gate 負責遺忘，delta rule 負責改寫",
      sources=["gdn", "gdnc"], accent="ssm", body=f"""
{gist("S ← S·α(I − βkkᵀ) + βvkᵀ：先整體衰減、再把 k 這個位址原本存的東西換掉、最後寫入新值。")}
{fig("gdn", F2.fig_gdn(), 5, [
 "狀態 S 是一個固定大小的矩陣（每個 head 一份）。",
 "<b>αₜ</b> 是 decay gate：整張狀態乘上一個 0–1 的係數，決定要忘掉多少。"
 "括號裡的 (I − βkkᵀ) 只針對 kₜ 這個方向做清除。",
 "<b>βₜvₜkₜᵀ</b> 把新的關聯寫進去。",
 "更新完，S 的大小<b>沒有變</b>。整件事的重點就在這裡。",
 "輸出 oₜ = Sₜ qₜ：用 q 去「問」這個摘要矩陣。"], accent="ssm")}""",
      notes="""<p>原論文的式子：<b>Sₜ = Sₜ₋₁(αₜ(I − βₜkₜkₜᵀ)) + βₜvₜkₜᵀ</b>，
其中 αₜ ∈ (0,1) 是資料相關的 decay gate、βₜ ∈ (0,1) 決定 delta rule 的替換強度。</p>
<p>和 Mamba2 的差別就在括號那一項：Mamba2 只有整體衰減 αSₜ₋₁，
DeltaNet 只有定點替換 (I − βkkᵀ)，Gated DeltaNet 兩個都有，
所以既能「快速清空」（α→0）也能「精準改寫」（β→1）。</p>
<p>實作上 α 由 <code>in_proj_a</code>、β 由 <code>in_proj_b</code> 產生，
每個 value head 一個純量；<code>A_log</code> 與 <code>dt_bias</code> 是可學的偏置。</p>""")

slide("Delta rule 的直覺", "為什麼不能只是「一直往上加」",
      sources=["gdn", "gdnimpl"], accent="ssm", body=f"""
{gist("寫進去之前先問一次「這個 key 現在存的是什麼」，只修正差額。這樣同一個 key 被更新兩次才不會疊在一起。")}
{fig("delta", F5.fig_delta(), 6, [
 "假設 memory 裡已經有「apple → red」，後來又出現「apple → green」。",
 "最原始的 linear attention 只會一直累加，兩筆疊在一起，查 apple 會得到混合的東西。",
 "Delta rule 的做法：先用 k 去查 state，看它現在記得什麼（red），算出誤差（green − red），"
 "再只修正那個差額。",
 "把完整式子拆開來看。",
 "<b>α</b> 管整體遺忘，<b>(I − βkkᵀ)</b> 只擦掉 k 這個方向，<b>βvkᵀ</b> 寫入新值。",
 "更新的成本是一次固定維度的矩陣運算 O(d_k·d_v)，不是「把所有舊 key 跑一遍」。",
 "Mamba2 只有 α，DeltaNet 只有 delta。Gated DeltaNet 兩個都有。"], accent="ssm")}""",
      notes="""<p>這一頁補的是「為什麼要有 delta rule」的動機，第一版只給了式子沒給直覺。</p>
<p>要澄清一個常見誤解：state 裡面<b>沒有</b>一份「key1、key2、key3」的清單。
它只有一個矩陣。所謂「更新和 apple 有關的方向」，數學上是一次 outer product
（v 是 128×1、k 是 1×128，乘出來就是 128×128），很多甚至全部的矩陣元素都會被改到。
但這是一次矩陣運算，不是迴圈掃過舊資料。</p>
<p>所以複雜度是 O(d_k·d_v)，與 T 無關。這跟 attention 的 O(T·d) 是完全不同的 scaling。</p>""")

slide("三種記憶：Attention、GDN、SSM", "同一個問題的三種答案：過去的資訊要怎麼存",
      sources=["gdn", "mamba", "hfq35"], accent="ssm", body=f"""
{gist("檔案櫃、可擦寫的白板、腦中的狀態。差別不在「能不能記」，在「怎麼存」與「存多久不變大」。")}
{fig("memkinds", F5.fig_memory_kinds(), 4, [
 "三者都在回答同一個問題：怎麼把過去 token 的資訊帶到現在。",
 "<b>Attention</b> 像檔案櫃：每個 token 一份文件，全部留著，查詢時掃過全部。",
 "<b>GDN</b> 像可擦寫的白板：把 key→value 的關聯壓進一個固定大小的矩陣，可查、可改、可擦。"
 "<b>SSM</b> 像腦中的狀態：不記逐字逐句，只更新理解。",
 "差別在長 context 時最明顯：Attention 還留著原始的 K/V；GDN 的資訊可能已經被後面的 token 蓋掉。",
 "所以 3:1 的排列很合理：GDN 提供便宜的壓縮記憶，attention 補回精確查找。"],
 accent="ssm")}""",
      notes="""<p>GDN 與 SSM 常被混為一談，它們確實都是「固定大小的 recurrent state」，
但 state 的數學含義不同：</p>
<p><b>GDN</b> 的 state 比較像 associative memory，是一個 key→value 的壓縮字典，
用 q 去查（o = S·q）。<br>
<b>SSM</b>（Mamba 那一系）的 state 比較像動態系統的內部狀態，
h_t = A·h_{t−1} + B·x_t，不是字典。Mamba 的貢獻是讓 A、B 隨輸入動態改變，
所以模型可以學「這個 token 重要，留著；那個不重要，忘掉」。</p>
<p>另外一個有用的說法：模型權重是 slow weights，整個推論過程都不變；
GDN 的 state 是 fast weights，每個 token 都在改。o = S·q 看起來就像一個 linear layer，
只是那個「權重」在 decode 過程中一直被改寫。</p>
<p>還有一個容易被忽略的實務點：GDN 的 O(1) 是<b>對 context 長度</b>而言，
不代表計算量小。每個 decode token 都要把整份 state 從 HBM 讀出來、改完再寫回去。
低 batch 的 GDN decode 一樣可能是 memory-bandwidth-bound。演算法複雜度不等於 GPU 瓶頸。</p>""")

slide("與 Mamba2 / DeltaNet 的關係", "同一條家族樹上的三個節點",
      sources=["gdn", "mut"], accent="ssm", body=f"""
{gist("Mamba2 只會整體遺忘、DeltaNet 只會定點改寫，Gated DeltaNet 把兩種能力合起來。")}
{table(["", "更新規則", "整體清空記憶", "精準改寫某個鍵", "vLLM 的歸類"], [
 ["Mamba2", "Sₜ = αₜSₜ₋₁ + vₜkₜᵀ", "<span class='tagc g'>快</span>（α→0）",
  "<span class='tagc r'>弱</span>", "mamba 家族"],
 ["DeltaNet", "Sₜ = Sₜ₋₁(I − βₜkₜkₜᵀ) + βₜvₜkₜᵀ", "<span class='tagc r'>慢</span>（一次一個鍵）",
  "<span class='tagc g'>強</span>（β→1）", "—"],
 ["<b>Gated DeltaNet</b>", "Sₜ = Sₜ₋₁(αₜ(I − βₜkₜkₜᵀ)) + βₜvₜkₜᵀ",
  "<span class='tagc g'>快</span>", "<span class='tagc g'>強</span>",
  "<b>mamba 家族</b>（走同一套 state 管理）"],
], "compact", hi=(2,))}
<div class="cols w6-4" style="margin-top:2px">
 {pane("為什麼 serving 要在意它屬於 mamba 家族", f'''
 <ul style="font-size:13.3px">
  <li>vLLM 把 GDN 當成 mamba 層處理：走同一個 state pool、同一套
   <code>mamba_ssm_dtype</code> 設定、同一組 shape 計算程式。</li>
  <li>config 裡的 <code>mamba_ssm_dtype: float32</code> 就是在說
   recurrent 矩陣用 FP32 存，CH6 的 144 MiB 就是這樣來的。</li>
  <li>所以看 vLLM 的 log 與 metrics 時，GDN 的資源會被標成 mamba。</li>
 </ul>''', "ssm")}
 {pane("狀態形狀（照 vLLM 的算法）", f'''<div class="formula" style="font-size:11.5px">
 conv_dim = d<sub>k</sub>·n<sub>k</sub>·<span class="c2">2</span> + d<sub>v</sub>·n<sub>v</sub></div>
 <div class="formula" style="font-size:11.5px;margin-top:5px">
 state = (n<sub>v</sub>, d<sub>v</sub>, d<sub>k</sub>)</div>
 <p style="font-size:12.6px;margin-top:6px">
 122B：conv_dim {Q35.conv_dim:,}、state ({Q35.nv}, {Q35.dv}, {Q35.dk})<br>
 27B：conv_dim {Q38.conv_dim:,}、state ({Q38.nv}, {Q38.dv}, {Q38.dk})</p>''', "compute")}
</div>""",
      notes="""<p>這一頁的實務價值：知道 GDN 走 mamba 路徑之後，就知道要去哪裡找設定與問題。
例如 <code>--mamba-ssm-cache-dtype</code> 可以把 recurrent state 從 FP32 降到 BF16，
直接把 144 MiB 砍成 72 MiB。CH6 會提到這個旋鈕，但品質要先驗過。</p>""")

slide("一次 decode 走完全部層", "兩種 cache 必須同步前進",
      sources=["c35", "c38", "hyb"], accent="ssm", body=f"""
{gist("同一個 token 穿過 48（或 64）層時，GDN 層就地更新固定大小的狀態，attention 層在尾端多寫一格 KV。")}
{fig("hybrid", F2.fig_hybrid("q35"), 4, [
 f"Qwen3.5-122B-A10B 的 48 層排成 12 組，每組是「GDN、GDN、GDN、attention」。",
 "資料由左往右穿過每一層。",
 f"{Q35.n_gdn} 層 GDN：讀舊 state → 就地寫回新 state，大小固定 {mib(Q35.ssm_recurrent_bytes(),0)} MiB。",
 f"{Q35.n_full} 層 full attention：讀全部歷史 K/V，在尾端多寫一格，每 token "
 f"{Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB。",
 "兩者要同步前進。hybrid 模型 serving 最容易出錯的地方就在這裡。"], accent="ssm")}""",
      notes="""<p>這頁要建立一個直覺：一次 decode 同時做了兩種完全不同性質的記憶體操作。
一種是「就地更新固定區塊」（隨機讀寫、大小不變），一種是「append 到成長中的緩衝區」。
它們的分頁策略、eviction 策略、prefix caching 能力都不一樣。</p>
<p>可以按頁面上的模型切換看 27B（64 層、16 組）。兩者的 recurrent state 竟然都剛好是 144 MiB，
因為 36×64 = 48×48 = 2,304 個 value head。這個巧合很好記。</p>""")

slide("Chunkwise prefill：GDN 怎麼吃長 prompt", "chunk 內平行、chunk 之間只傳一個狀態",
      sources=["gdn", "gdnc", "hyb"], accent="ssm", body=f"""
{gist("如果 prefill 也一個 token 一個 token 遞迴，長 prompt 會慢到不能用。GDN 改成分塊：塊內用矩陣乘法，塊間只傳狀態。")}
{fig("chunk", F2.fig_chunkwise(), 4, [
 "prompt 被切成固定大小的 chunk。",
 "chunk 內部用矩陣乘法平行處理，塞滿 Tensor Core。",
 "chunk 之間只傳遞一個固定大小的狀態 S，序列依賴只剩這一條。",
 "同時，full attention 層照常做 FlashAttention 並寫入 prompt 的 K/V。",
 "注意：vLLM 對 mamba / GDN 的 prefix caching 仍在開發中，這會影響多輪對話的 TTFT。"],
 accent="ssm")}
{quote("The gating term only performs elementwise multiplication with intermediate variables without affecting matrix multiply structures.",
       "Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv 2412.06464")}""",
      notes="""<p>上面那句引文是 GDN 能跑得快的關鍵：gate 只做逐元素乘法，
不破壞矩陣乘法的結構，所以還是能用 Tensor Core。
這也是為什麼 GDN 的 prefill 吞吐可以和 attention 同級。</p>
<p>對 serving 的意義：混合模型的 prefill 不會因為 GDN 而變慢；
差異在 decode（GDN 省很多）與 prefix caching（GDN 目前吃虧）。</p>""")

slide("vLLM 怎麼同時管兩種 cache", "Hybrid KV Cache Manager 的三個設計決定",
      sources=["hyb", "mut"], accent="mem", body=f"""
{gist("vLLM 把層依 attention 型態分成 KV cache group，強制所有 group 的實體 page 大小一致，再把 mamba state 塞進同樣的 page。")}
<div class="cols3">
 {pane("① KV cache group", '''<p style="font-size:13.2px">同一個 group 內只能有<b>同一種</b>
 attention 型態。混合模型至少兩個 group：full attention 一組、GDN（mamba）一組。</p>
 <p style="font-size:13.2px;color:var(--mut)">分組大小以層數較少的那一種為基準，減少浪費。</p>''',
 "compute")}
 {pane("② 統一 page size", '''<p style="font-size:13.2px">所有 group 的實體 page 必須同樣大。
 做法是<b>把 attention 層的 block_size 撐大</b>，直到
 <code>block_size × kv_hidden ≥ mamba_state_size</code>，再把 mamba state 補齊。</p>
 <p style="font-size:13.2px;color:var(--bad)">官方文件坦承這會導致 block_size 超過 400，
 有效率上的疑慮。</p>''', "mem")}
 {pane("③ Prefix caching 取交集", '''<p style="font-size:13.2px">命中長度是所有 group
 的<b>交集</b>：full attention 由左往右掃到第一個 miss；其他型態再從那個長度往回掃。</p>
 <p style="font-size:13.2px;color:var(--bad)">mamba / GDN 的 prefix caching
 仍在開發中；超過兩種 attention 型態的模型必須關掉 prefix caching。</p>''', "bad")}
</div>
{table(["這對容量規劃的影響", "後果"], [
 ["SSM state 是「每序列一份」，不是「每 token 一份」",
  f"併發上限直接乘上 {mib(Q35.ssm_bytes(3),0)} MiB（122B）或 {mib(Q38.ssm_bytes(7),0)} MiB（27B）。"
  "短對話時，這一項會<b>主宰</b>記憶體。"],
 ["page size 被撐大", "小 request 的內部碎片變多；實際可用的 cache 會略少於公式值。"],
 ["GDN 的 prefix caching 尚未完備",
  "多輪對話的 TTFT 改善幅度會比純 Transformer 小。做 benchmark 時要記錄命中率。"],
 ["preempt 需要同時處理兩種狀態", "換出換回的成本比純 Transformer 高，應盡量避免觸發。"],
], "compact")}""",
      notes="""<p>這一頁是第二版新增的，實務上也最容易踩雷。
第一版只說「vLLM 會管」，沒有講清楚 page size 對齊與 prefix caching 交集的代價。</p>
<p>檢查方式：啟動時 vLLM 會在 log 印出 KV cache group 的配置與 block size。
如果看到很大的 block_size（例如 &gt; 256），就要知道小 request 的碎片會偏高，
公式算出來的併發數要打個折扣。</p>""")
