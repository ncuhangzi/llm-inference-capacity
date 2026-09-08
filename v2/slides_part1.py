# -*- coding: utf-8 -*-
# 由 build.py exec()，共用其命名空間。
# ============================================================ 開場 =========
chapter("導讀", "compute")

slide("LLM 推論與服務容量", kind="cover", body=f"""
<div class="wrap">
  <div class="kicker">Inference · Memory · Capacity</div>
  <h1>LLM 推論與服務容量</h1>
  <div class="csub">一個 request 進到 GPU 之後發生什麼事，以及「能服務多少人」該怎麼算</div>
  <div class="qs">
    <div><b>Q1</b><span>模型的一次 forward，錢花在算力還是頻寬上？</span></div>
    <div><b>Q2</b><span>混合架構把 KV cache 換成固定大小的狀態，記憶體帳本長什麼樣？</span></div>
    <div><b>Q3</b><span>NVFP4 與 FP8 到底量化了哪些張量，省下多少？</span></div>
    <div><b>Q4</b><span>speculative decoding 什麼時候賺、什麼時候該關掉？</span></div>
    <div><b>Q5</b><span>從實測數字到「可以承諾幾個人用」，中間要補哪些前提？</span></div>
  </div>
</div>
<div class="meta">
  <b>評估對象</b>　nvidia/Qwen3.5-122B-A10B-NVFP4（MTP，k=3）　·　Qwen/Qwen3.8-27B-FP8（DFlash2，k=7）　·　Qwen/Qwen3-32B 作為純 Transformer 對照<br>
  <b>執行環境</b>　vLLM on NVIDIA B200 180 GB　·　<b>資料查核日</b>　{K.CHECK}　·
  架構數字取自官方 config 與 checkpoint 的 safetensors 統計；容量數字為公式推導<br>
  <b>操作</b>　<kbd>→</kbd> 換頁　<kbd>Space</kbd> 動畫下一步　<kbd>G</kbd> 名詞表　<kbd>N</kbd> 講稿　<kbd>O</kbd> 目錄　<kbd>?</kbd> 說明
</div>""",
      notes="""<p>開場先問這五個問題，等於宣告這份報告的邊界：我們談的是<b>單一 request 的機制</b>與
<b>整機容量的算法</b>，不談訓練、不談模型品質評測。</p>
<p>第一版報告與這一版最大的差別：所有數字都對齊<b>實際要上線的量化版本</b>
（122B 用 NVFP4、27B 用 FP8），並且把 speculative decoding 從一頁擴充成完整一章。</p>
<p>建議節奏：導讀 3 分鐘，CH1–CH2 共 15 分鐘，CH3–CH4 架構 15 分鐘，CH5 量化 8 分鐘，
CH6 記憶體 10 分鐘，CH7 speculative 12 分鐘，CH8–CH9 量測 15 分鐘，結論 5 分鐘。
時間不夠時可以整章跳過 CH1（假設聽眾已懂 Transformer），附錄留給 Q&amp;A。</p>""")

slide("怎麼讀這份報告", "先講清楚顏色、名詞與數字的可信度分級",
      sources=["nv4", "f38", "df2"], body=f"""
<div class="cols w6-4">
 <div>
  {pane("顏色一律代表同一件事", f'''
  <div class="legend" style="gap:10px 18px;font-size:12.5px">
   <span><i style="background:var(--compute)"></i>算力 / attention / 權重</span>
   <span><i style="background:var(--mem)"></i>記憶體頻寬 / KV cache</span>
   <span><i style="background:var(--ssm)"></i>GDN、SSM 狀態</span>
   <span><i style="background:var(--moe)"></i>MoE / expert</span>
   <span><i style="background:var(--spec)"></i>speculative decoding</span>
   <span><i style="background:var(--ok)"></i>被接受 / 有效產出</span>
   <span><i style="background:var(--bad)"></i>被拒絕 / 浪費</span>
  </div>''')}
  {pane("看不懂的字直接把滑鼠放上去", f'''
  <p style="font-size:14px">像 {T("PagedAttention")}、{T("Gated DeltaNet (GDN)","Gated DeltaNet")}、
  {T("Acceptance length (τ)","接受長度 τ")} 這種<b class="t" data-term="Arithmetic intensity">虛線底的粗體字</b>
  都是專有名詞，滑過會出現解釋，按 <kbd>G</kbd> 可以打開完整名詞表（共 {len(K.GLOSSARY)} 條）。</p>
  <p style="font-size:13.5px;color:var(--mut)">每一頁的頁尾都列出該頁的來源，可直接點開原始 config、
  論文或官方文件；按 <kbd>N</kbd> 的講者筆記也會附同一組連結。</p>''')}
 </div>
 <div>
  {pane("數字的可信度分三級", f'''
  <table class="compact"><tbody>
  <tr><td><span class="tagc g">實測 / 官方</span></td><td>直接來自官方 config、模型卡、
   checkpoint 的 safetensors 位元組統計，或論文與廠商公布的 benchmark。可以直接引用。</td></tr>
  <tr><td><span class="tagc">公式推導</span></td><td>由上面那些數字用明確公式算出來的，
   例如 KV cache 大小、記憶體帳本、roofline 上限。公式都寫在頁面上，可以自己驗算。
   標示為「解析上限」的，實測通常只有它的 30–60%。</td></tr>
  <tr><td><span class="tagc m">教學示例</span></td><td>throughput / 延遲 / 佇列曲線與
   SLO 掃描示意圖。用排隊理論生成，<b>不是任何硬體的實測結果</b>，只用來說明形狀。</td></tr>
  </tbody></table>''', "mem")}
  {warn("凡是標成教學示例的圖，都不可以當成 B200 實測成績對外引用。正式容量報告必須換成自家實測資料。")}
 </div>
</div>""",
      notes="""<p>這一頁是防呆用的。第一版最容易被誤讀的地方，就是聽眾把示意曲線當成實測。
這一版把可信度分級講明，每張教學示例圖上也都留了紅字。</p>
<p>顏色系統也統一了：整份報告只要看到橘色就是「頻寬 / KV」，藍色就是「算力 / 權重」，
粉紅就是 speculative。到後面看複合圖時，不用重新解讀圖例。</p>""")

slide("報告路線", "從一個 token 的產生，一路推到「能服務幾個人」", body="""
<div class="agenda">
 <a data-go="3"><b>01</b><span class="nm">一個 token 是怎麼生出來的</span>
   <span class="ds">autoregressive、KV cache、GQA、prefill vs decode，最後導出 B200 的屋頂線轉折點</span>
   <span class="mn">8 分</span></a>
 <a data-go="11"><b>02</b><span class="nm">vLLM 把它變成一個服務</span>
   <span class="ds">執行堆疊、continuous batching、PagedAttention、每一輪排程實際在做什麼</span>
   <span class="mn">7 分</span></a>
 <a data-go="20"><b>03</b><span class="nm">三個模型的架構帳本</span>
   <span class="ds">純 Transformer、hybrid dense、hybrid MoE；參數如何逐張量驗證</span>
   <span class="mn">10 分</span></a>
 <a data-go="27"><b>04</b><span class="nm">Gated DeltaNet：用固定狀態換掉 KV cache</span>
   <span class="ds">state update 的數學、hybrid 的 3:1 排列、vLLM 怎麼同時管兩種 cache</span>
   <span class="mn">10 分</span></a>
 <a data-go="34"><b>05</b><span class="nm">量化：NVFP4 與 FP8</span>
   <span class="ds">位元佈局、checkpoint 實際量化了哪些張量、精度代價、為什麼轉折點不會移動</span>
   <span class="mn">8 分</span></a>
 <a data-go="40"><b>06</b><span class="nm">單卡記憶體帳本</span>
   <span class="ds">KV cache 與 SSM state 分開算、交會點、B200 180 GB 怎麼分、併發上限</span>
   <span class="mn">10 分</span></a>
 <a data-go="47"><b>07</b><span class="nm">Speculative decoding 對 serving 的影響</span>
   <span class="ds">MTP 與 DFlash2 的機制差異、接受長度、(k+1)× 撞屋頂線、什麼時候該關掉</span>
   <span class="mn">12 分</span></a>
 <a data-go="59"><b>08</b><span class="nm">延遲指標的定義與陷阱</span>
   <span class="ds">TTFT / TPOT / ITL / E2EL、百分位、goodput</span>
   <span class="mn">6 分</span></a>
 <a data-go="63"><b>09</b><span class="nm">兩階段負載測試</span>
   <span class="ds">閉環找飽和、開環找 SLO 容量、邊界搜尋、RPS 換算人數</span>
   <span class="mn">12 分</span></a>
 <a data-go="70"><b>10</b><span class="nm">結論與調校決策樹</span>
   <span class="ds">十個可以帶走的結論，加一張「該轉哪個旋鈕」的流程圖</span>
   <span class="mn">5 分</span></a>
</div>""",
      notes="""<p>點目錄任一列可以直接跳章。全程約 90 分鐘；壓縮到 50 分鐘的話，
建議保留 01（快帶）、04、06、07、09，把 02、03、05 濃縮成各一頁摘要。</p>
<p>這份報告的主線是「同一個 request」：從 tokenizer 一路到負載測試報告，
每一章問的都是同一件事的不同層次：這一輪 forward 的成本被多少 token 分攤了。</p>""")

# ============================================== CH1 一個 token 的誕生 =====
chapter("CH1 一個 token 是怎麼生出來的", "compute")

slide("一個 token 是怎麼生出來的", kind="section", body="""
<div class="no">CHAPTER 01</div>
<h1>一個 token 是怎麼生出來的</h1>
<div class="csub">推論的成本結構就是在這一段裡定下來的。這一章結束時會得到一個數字，後面所有現象都可以拿它來解釋：
B200 上的屋頂線轉折點。</div>
<ul>
 <li>tokenizer 與 chat template：模型真正看到的長度</li>
 <li>autoregressive 迴圈：為什麼生 T 個 token 要 T 次 forward</li>
 <li>decoder layer 的內部與張量形狀</li>
 <li>KV cache 做了什麼、GQA 又省下多少</li>
 <li>prefill 與 decode 是兩種完全不同的工作</li>
 <li>算術強度與 roofline：N* ≈ 292 token / step</li>
</ul>""",
      notes="<p>章節頁停 15 秒就好，把最後一行的 N* 唸出來讓聽眾記住。</p>")

slide("模型真正看到的是一串 id", "長度 T 決定 prefill 的計算量與 KV cache 的大小",
      sources=["c38"], body=f"""
{gist("使用者打了 13 個中文字，模型看到的可能是 20 幾個 token。容量計算的分母是這個 T，不是字數。")}
{fig("tokenize", F1.fig_tokenize(), 3, [
 "使用者輸入一段文字。",
 "<b>① tokenizer</b> 把文字切成 token 並轉成整數 id。中文常見 1–2 字一個 token，"
 "英文約 4 個字元一個 token。",
 "<b>② chat template</b> 再包上角色標記與思考模式的開頭。Qwen3.5/3.8 預設開啟 thinking，"
 "會多出 <code>&lt;think&gt;</code> 這段。",
 "<b>③</b> 這串 id 的長度 T 才是實際的輸入長度。整份報告裡的「context 長度」都指這個 T。"])}
<div class="cols w6-4" style="margin-top:2px">
 {pane("兩個模型的詞彙表一樣大", table(["模型", "~vocab_size", "~embedding 參數", "~BF16 佔用"], [
   ["Qwen3.5-122B-A10B", f"{Q35.vocab:,}", f"{Q35.vocab*Q35.hidden/1e6:,.0f} M", f"{Q35.vocab*Q35.hidden*2/1e9:,.2f} GB"],
   ["Qwen3.8-27B", f"{Q38.vocab:,}", f"{Q38.vocab*Q38.hidden/1e6:,.0f} M", f"{Q38.vocab*Q38.hidden*2/1e9:,.2f} GB"],
   ["Qwen3-32B（對照）", f"{Q332.vocab:,}", f"{Q332.vocab*Q332.hidden/1e6:,.0f} M", f"{Q332.vocab*Q332.hidden*2/1e9:,.2f} GB"],
 ], "compact"))}
 {pane("為什麼要在意 vocab", '''<p style="font-size:13.5px">248,320 × hidden 的
 <code>lm_head</code> 是<b>每一個 decode step 都要完整讀一次</b>的張量。到 CH7 會看到，
 這一塊主宰了 MTP 起草器的成本：起草一次的代價，幾乎就是再讀一次 lm_head。</p>''', "mem")}
</div>""",
      notes="""<p>這一頁要把「字數」與「token 數」分開。業務端談字數，容量計算用 token 數，
中間差 1.5–2 倍很常見；再加上 chat template 與 thinking 標記，短問句的實際 prefill 長度
往往是原文的兩倍以上。</p>
<p>vocab 那張表先鋪梗：248,320 是個很大的詞彙表，lm_head 在 122B 上是 762.8 M 參數、
BF16 佔 1.53 GB，而它在 NVFP4 checkpoint 裡<b>沒有被量化</b>。CH5 與 CH7 都會回來用這個事實。</p>""")

slide("Autoregressive：生 T 個 token 就要 T 次 forward", "decode 之所以慢，原因就在這裡",
      sources=["q3r", "anat"], body=f"""
{gist("模型一次 forward 只能決定「下一個」token。整個 serving 最佳化史，就是想辦法讓每次 forward 更划算。")}
{fig("autoreg", F1.fig_autoregress(), 5, [
 "輸入的 id 先過 embedding，再穿過 L 層 decoder，最後由 LM head 投影回 vocabulary 大小的分數。",
 "<b>Sampling</b> 依 temperature / top-p / top-k 從分數裡抽一個 token 出來。",
 "抽到的 token <b>接回輸入尾端</b>，再跑一次。autoregressive 就是這個迴圈。",
 "每一輪只在序列尾端多一格。生 500 個 token 就是 500 次完整 forward。",
 "關鍵的不對稱：<b>本輪只算 1 個 token，卻要把整個模型的權重讀一遍。</b>"
 "後面所有的事情都從這個不對稱長出來。"])}
<div class="cols3" style="margin-top:2px">
 {pane("推論不做的事", '<p style="font-size:13.5px">沒有 backward、沒有 optimizer state。'
       '權重固定不動，變動的只有 activation 與每個 request 的狀態。</p>')}
 {pane("什麼時候停", '<p style="font-size:13.5px">抽到 EOS、命中 stop string、'
       '或達到 max_tokens。輸出長度事前不知道，這讓容量規劃必須用分布而非單一值。</p>')}
 {pane("有一個例外", f'<p style="font-size:13.5px">{T("Speculative decoding")} 可以讓'
       '一次 forward 前進多格。那是 CH7 的主題，機制上還是同一個迴圈。</p>', "spec")}
</div>""",
      notes="""<p>講到第 5 步時停一下，把「1 個 token vs 整份權重」這個不對稱寫在白板上。
122B NVFP4 的權重是 83.5 GB，B200 頻寬 7.7 TB/s，光讀完就要 10.8 ms。只服務一個人的話，一秒最多 92 個 token，而 GPU 的算力幾乎全閒著。</p>
<p>（實際上 MoE 在 batch 1 只會讀到被路由到的 8/256 個 expert，所以是 12.8 GB、1.66 ms，
一秒 601 個 token 的解析上限。這個修正在 CH3 會補上。）</p>""")

slide("Decoder layer 內部：兩個 mixer", "token mixer 讓位置之間交換資訊，channel mixer 只在單一位置上做事",
      sources=["c35", "q3r"], body=f"""
{gist("每一層都是「先跨位置混一次，再跨通道混一次」。attention 與 GDN 是可以互換的 token mixer；FFN 與 MoE 是可以互換的 channel mixer。")}
{fig("layer", F1.fig_layer(), 5, [
 "一層的骨架：norm → token mixer → norm → channel mixer，兩段都有 residual。",
 "<b>QKV projection</b> 把 hidden 投影成 Query、Key、Value。注意右邊的形狀："
 "Q 有 32 組、K/V 只有 2 組。",
 "<b>QK-norm 與 RoPE</b> 把位置資訊寫進 Q 與 K。Qwen3.5 只旋轉 head_dim 的 1/4。",
 "<b>Attention</b> 用本輪的 Q 去查<b>整段歷史</b>的 K/V。整層裡只有這一步要跨位置讀取。",
 "<b>Output projection</b> 帶 gate（Qwen3.5 的 attn_output_gate），回到 hidden 寬度。",
 "<b>Channel mixer</b>（FFN 或 MoE）每個 token 各自算，位置之間互不相干。"])}""",
      notes="""<p>這頁的重點是建立「可替換的插槽」這個心智模型：後面 CH4 把 token mixer 換成
Gated DeltaNet、CH3 把 channel mixer 換成 MoE，架構就從 Qwen3-32B 變成 Qwen3.5-122B-A10B。
只有 token mixer 需要記憶體保存歷史，cache 的形狀就是這樣決定的。</p>
<p>右側的張量形狀值得逐行唸：hidden 3072、Q 是 32×256、K/V 只有 2×256。
Q 的總寬度 8192 比 hidden 還大，這是 Qwen3.5 的特色（head_dim 256 相當大）。</p>""")

slide("KV cache：把算過的東西存起來", "代價是記憶體隨 context 線性成長",
      sources=["paged", "q3r"], body=f"""
{gist("沒有 KV cache，生成成本是 O(T²)；有了 KV cache 變成 O(T)，但要付出隨 T 成長的記憶體。")}
{fig("kv", F1.fig_kv(), 4, [
 "attention 需要本輪的 q 去比對「所有」位置的 K/V。",
 "<b>①</b> 前面 7 個位置的 K/V 早就算過了，直接躺在 cache 裡。",
 "<b>②</b> 本輪只需要新增第 8 個位置的 k₈ 與 v₈。",
 "<b>③</b> 但 attention 要<b>讀取全部 8 格</b>，讀取量隨 context 線性成長。",
 "<b>④</b> 若不快取，每輪都要把前面所有位置重算一次，總成本變成 O(T²)。"])}
<div class="cols3" style="margin-top:2px">
 {pane("換來的是什麼", '<p style="font-size:13.5px">計算從 O(T²) 降到 O(T)。'
       '長 context 之下這是不可能放棄的。</p>', "ok")}
 {pane("付出的是什麼", '<p style="font-size:13.5px">每個 token 都要永久佔一塊 HBM，'
       '直到 request 結束。這是併發數的<b>硬上限</b>。</p>', "mem")}
 {pane("三種解法", '<p style="font-size:13.5px">① GQA 減少 KV head（本頁下一張）'
       '② FP8 KV cache 砍一半 ③ 換掉 token mixer，讓它不需要 KV，也就是 CH4 的 GDN。</p>', "ssm")}
</div>""",
      notes="""<p>這一頁要讓聽眾意識到 KV cache 是「拿記憶體換計算」的交易，而不是免費的最佳化。
在 262,144 的 context 之下，Qwen3-32B 這種純 Transformer 的 KV cache 會到 64 GiB／條，
單卡放不下兩條。混合架構會出現，就是為了這件事。</p>""")

slide("GQA：Q 很多、K/V 很少", "KV cache 的大小只跟 KV head 數有關，跟 Q head 數無關",
      sources=["c35", "c38", "c332"], body=f"""
{gist("122B 用 32 個 Q head 共用 2 組 K/V，等於把 KV cache 壓成 MHA 的 1/16。這是設定檔裡最便宜的一個決定。")}
{fig("gqa", F1.fig_gqa("q35"), 2, [
 "上排 32 個藍格是 Query head；每個 head 各自去算 attention。",
 "下排只有 2 組橘格：16 個 Q head 共用同一組 K/V。",
 "所以每個 token 只需要保存 2 × 2 × 256 = 1,024 個數字（K 與 V 各 512）。"])}
{table(["模型", "~Q head", "~KV head", "~head_dim", "~共用比例",
        "~假設 MHA 的 KV/token", "~實際 KV/token (BF16)"], [
 [Q35.label, Q35.n_heads, Q35.n_kv, Q35.head_dim, f"{Q35.n_heads//Q35.n_kv}:1",
  f"{2*Q35.n_full*Q35.n_heads*Q35.head_dim*2/KiB:,.0f} KiB",
  f"<b>{Q35.kv_bytes_per_token('bf16')/KiB:,.0f} KiB</b>"],
 [Q38.label, Q38.n_heads, Q38.n_kv, Q38.head_dim, f"{Q38.n_heads//Q38.n_kv}:1",
  f"{2*Q38.n_full*Q38.n_heads*Q38.head_dim*2/KiB:,.0f} KiB",
  f"<b>{Q38.kv_bytes_per_token('bf16')/KiB:,.0f} KiB</b>"],
 [Q332.label + "（純 Transformer）", Q332.n_heads, Q332.n_kv, Q332.head_dim,
  f"{Q332.n_heads//Q332.n_kv}:1",
  f"{2*Q332.n_full*Q332.n_heads*Q332.head_dim*2/KiB:,.0f} KiB",
  f"<b>{Q332.kv_bytes_per_token('bf16')/KiB:,.0f} KiB</b>"],
], "compact", hi=(2,))}
<div class="hint">「假設 MHA」欄是把 KV head 數換成 Q head 數重算的結果，用來看 GQA 省了多少。
122B 的 KV/token 之所以特別小，是 GQA（16:1）與只有 12 層 full attention 兩件事相乘的結果。</div>""",
      notes="""<p>這裡要點出一個容易被忽略的事：head_dim 256 其實很大（Qwen3-32B 只有 128），
但因為 KV head 只有 2 組，最後 KV/token 還是很小。看 KV cache 只看單一參數會誤判，
必須用完整公式。</p>
<p>Qwen3-32B 的 256 KiB/token 與 122B 的 24 KiB/token 差 10.7 倍，同樣的 HBM 能放的 context 也就差 10 倍。
這是這一版最重要的對照之一。</p>""")

slide("Prefill 與 decode：兩種完全不同的工作", "同樣的權重、同樣的層，成本結構卻相反",
      sources=["anat", "q3r"], body=f"""
{gist("prefill 是「很多 token 分攤一次權重讀取」，decode 是「很少 token 分攤一次權重讀取」。前者吃算力，後者吃頻寬。")}
{fig("pd", F1.fig_prefill_decode(), 3, [
 "prefill 一次吃完整個 prompt；decode 每輪每條序列只餵 1 個 token。",
 "矩陣的形狀完全不同：prefill 是胖矩陣（大 GEMM），decode 退化成瘦長的 GEMV。",
 "所以<b>算術強度</b>差了兩個數量級：prefill 逼近算力上限，decode 卡在頻寬上限。",
 "看清這件事之後，serving 上的最佳化就收斂成同一個動作："
 "<b>把更多 token 塞進同一次權重讀取裡。</b>"])}
{table(["", "Prefill", "Decode"], [
 ["本輪 token 數", "prompt 全長（可達數萬）", "batch 大小 B（每序列 1 個）"],
 ["矩陣運算", "大 GEMM，Tensor Core 打滿", "GEMV／瘦 GEMM，Tensor Core 大量閒置"],
 ["瓶頸", "<span class='tagc'>算力</span>", "<span class='tagc m'>記憶體頻寬</span>"],
 ["attention 成本", "O(T²)，但可用 FlashAttention 分塊", "O(T)，隨 context 線性上升"],
 ["對指標的影響", "決定 TTFT", "決定 TPOT 與 ITL"],
 ["調校旋鈕", "chunked prefill、prefix caching", "batch 大小、量化、speculative decoding"],
], "compact")}""",
      notes="""<p>這一頁的概念在全報告裡最重要。之後每一個技術（continuous batching、chunked prefill、
MoE、speculative decoding）都可以用「它把哪一側的浪費補起來」來解釋。</p>
<p>補一個實務數字：prefill 一個 2,048 token 的 prompt 與 decode 一個 token，
讀的權重量一模一樣，但前者做了 2,048 倍的運算。把 prefill 與 decode 混在同一個 batch
（chunked prefill）能同時改善兩邊，道理就在這。</p>""")

slide("屋頂線：B200 上的轉折點是 292 token / step",
      "而且它<span style='color:var(--mem)'>不會</span>因為量化而移動",
      sources=["b200l", "b200"], body=f"""
{gist(f"一次 forward 裡湊到約 {KNEE:.0f} 個 token 之前，時間都花在讀權重上、加 token 幾乎免費；超過之後才開始付算力的錢。")}
{fig("roof", F1.fig_roofline(), 5, [
 "橫軸是「一次 forward 送進去的 token 總數 N」，縱軸是每個 token 的成本。",
 f"兩段曲線的交會處就是轉折點 <b>N* = F·b/(2·BW) ≈ {KNEE:.0f}</b>。",
 "左邊 memory-bound：時間等於讀權重的時間，跟 N 無關，多塞 token 幾乎不用加錢。"
 "右邊 compute-bound：時間開始隨 N 線性上升。",
 "所以 batch 1 是最浪費的模式：付了整份權重的錢，只換到 1 個 token。",
 "而 32 併發配上 DFlash2 的 7 個草稿，一次就送進 256 個 token，"
 "剛好貼在轉折點上。CH7 要算的就是這件事。"])}
<div class="cols3" style="margin-top:2px">
 {pane("量化幫不上這個忙", '<p style="font-size:13.5px">FP8 或 NVFP4 讓兩邊<b>同時</b>變快，'
       '你在屋頂線上的相對位置完全沒動。「上了 FP4 就不再 memory-bound」是很常見的誤會。</p>', "compute")}
 {pane("想離開只有一條路", f'<p style="font-size:13.5px">要離開 memory-bound 只有一條路：'
       f'<b>提高每個 step 的 token 數</b>：更大 batch，或 {T("Speculative decoding")}。</p>', "mem")}
 {pane("反過來說", f'<p style="font-size:13.5px">只要 N &lt; {KNEE:.0f}，就有<b>免費的算力</b>'
       '可以拿去猜 token。speculative decoding 之所以成立，靠的就是這塊。</p>', "spec")}
</div>
""",
      notes=f"""<p>這一頁是第二版新增的，也是整份報告的樞紐。把 N* 的公式推一次：<br>
時間<sub>頻寬</sub> = P·b / BW，時間<sub>算力</sub> = N·2P / F，兩者相等得 N* = F·b/(2·BW)。<br>
代入 B200：4.5e15 × 1 / (2 × 7.7e12) = {KNEE:.1f}。</p>
<p>精度不變性這個結果很漂亮，聽眾通常會意外。它也推翻了一個常見誤會：
「上了 FP4 就不再是 memory-bound」。FP4 讓你讀得更快也算得更快，
但你在 roofline 上的<b>相對位置</b>沒有動。</p>
<p>要注意這是 dense 峰值、且不含 attention kernel 與 MoE routing 的成本，
所以是解析上限。MoE 的專家讀取量會隨 batch 成長，讓 122B 的實際轉折點推遲到約 1,765 token/step，
CH3 與 CH7 會處理這個修正。</p>""")
