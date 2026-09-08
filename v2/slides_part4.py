# -*- coding: utf-8 -*-
# =============================================== CH8 延遲指標 =============
chapter("CH8 延遲指標的定義與陷阱", "compute")

slide("延遲指標的定義與陷阱", kind="section", accent="compute", body="""
<div class="no">CHAPTER 08</div>
<h1>延遲指標的定義與陷阱</h1>
<div class="csub">容量數字會被吵，多半不是量錯。是兩邊講的指標、百分位或量測窗根本不一樣。
這一章把定義釘死。</div>
<ul>
 <li>一個 request 的完整時間軸</li>
 <li>TTFT / TPOT / ITL / E2EL 的精確定義與它們的統計母體</li>
 <li>為什麼平均值沒有用：百分位與長尾</li>
 <li>Throughput 與 goodput 的差別</li>
</ul>""",
      notes="<p>這一章短，但值得要求所有人照同一份定義寫報告。</p>")

slide("一個 request 的完整時間軸", "四個指標各自量的是哪一段",
      sources=["bench", "gp"], accent="compute", body=f"""
{gist("TTFT 量的是「等多久才開始」，TPOT / ITL 量的是「開始之後順不順」，E2EL 是使用者實際等待的總時間。")}
{fig("lat", F4.fig_latency(), 7, [
 "一個 request 的生命週期：網路與排隊、prefill、然後是 N 次 decode。",
 "<b>網路 + 佇列</b>：這一段不在模型裡，但使用者一樣要等。",
 "<b>prefill</b>：把整個 prompt 算完並填好 cache。",
 "<b>decode</b>：一次吐一個 token（開了 spec 之後可能一次吐好幾個）。",
 "<b>TTFT</b>：送出到收到第一個 token。包含排隊與 prefill。",
 "<b>ITL</b>：相鄰兩次串流輸出之間的間隔。母體是「每一次間隔」。",
 "<b>E2EL</b>：送出到收到完整回覆。",
 "<b>TPOT</b> 是每個 request 自己算的平均，母體是「每一個 request」，與 ITL 不同。"],
 accent="compute")}""",
      notes="""<p>ITL 與 TPOT 最容易被混用。舉例：100 個 request，每個 500 個 token。<br>
ITL 的樣本數是 100 × 499 = 49,900；<br>
TPOT 的樣本數是 100。<br>
所以「p95 ITL」與「p95 TPOT」是兩個完全不同的數字，前者會被單一次的長停頓拉高，
後者被平均掉。開了 speculative decoding 之後這個差距會更大。</p>""")

slide("四個指標的精確定義", "以及各自的陷阱",
      sources=["bench", "gp", "anat"], accent="compute", body=f"""
{gist("每個指標都要問三件事：量的是哪一段、母體是什麼、有沒有含服務外部的時間。")}
{table(["指標", "定義", "統計母體", "常見陷阱"], [
 [f"<b>{T('TTFT')}</b>", "送出 request → 收到第一個 token", "每個 request 一個樣本",
  "包含排隊時間。系統過載時 TTFT 會先爆炸，而 TPOT 看起來還好。"
  "只看 TPOT 會誤判系統健康。"],
 [f"<b>{T('TPOT')}</b>", "(E2EL − TTFT) ÷ (輸出 token 數 − 1)", "每個 request 一個樣本",
  "只有一個輸出 token 的 request 沒有 TPOT。短輸出佔比高時，樣本會偏少。"],
 [f"<b>{T('ITL')}</b>", "相鄰兩次串流輸出的時間差", "每一次間隔一個樣本",
  "開了 speculative decoding 之後，一輪可能吐好幾個 token，ITL 分布變成雙峰，"
  "p99 會惡化但平均改善。"],
 [f"<b>{T('E2EL')}</b>", "送出 → 收到完整回覆", "每個 request 一個樣本",
  "與輸出長度強相關。比較不同 workload 的 E2EL 沒有意義。"],
], "compact")}
<div class="cols3" style="margin-top:2px">
 {pane("串流與非串流", '<p style="font-size:13.2px">非串流 API 量不到 TTFT 與 ITL，'
       '只有 E2EL。要量延遲細節就必須用串流端點。</p>', "compute")}
 {pane("客戶端 vs 伺服器端", '<p style="font-size:13.2px">壓測工具量的是客戶端時間，'
       '含網路與 tokenizer。引擎的 metrics 量的是伺服器內部。兩者會差幾十毫秒，'
       '報告要說清楚用哪一個。</p>', "mem")}
 {pane("量測窗", '<p style="font-size:13.2px">暖機期（CUDA graph 捕捉、cache 預熱）'
       '必須排除。建議跑 3–5 分鐘，取中間穩定的 60–120 秒。</p>', "ssm")}
</div>""",
      notes="""<p>這一頁可以直接拿去當團隊的指標定義規範。</p>
<p>「開了 spec 之後 ITL 變雙峰」這件事值得強調：如果 SLO 寫的是「p99 ITL &lt; 50 ms」，
開了 spec 之後可能反而不合格，即使使用者感受更好。這時要把 SLO 改寫成 TPOT。</p>""")

slide("百分位與 goodput", "平均值會騙人，goodput 才是使用者拿到的服務",
      sources=["gp", "bench"], accent="compute", body=f"""
{gist("系統過載時 throughput 可能還很高，但沒有一個人滿足 SLO，此時 goodput = 0。")}
{fig("pct", F4.fig_percentile(), 5, [
 "延遲分布是右偏的長尾。",
 "平均值落在左邊那個峰，看起來很漂亮。",
 "但 p95 與 p99 在尾巴上，而那些人就是會來抱怨的人。",
 "goodput 只計算「同時滿足所有 SLO」的 request。",
 "報告紀律：每個延遲數字都要註明指標、百分位、量測窗與 workload。"],
 accent="compute")}""",
      notes="""<p>vLLM 的 <code>bench serve</code> 支援 <code>--goodput</code> 參數，
可以一次指定多個 SLO（例如 <code>--goodput ttft:500 tpot:40</code>），
輸出 request goodput。這是最接近「使用者體驗」的單一數字。</p>
<p>強調一個實務場景：把併發從 32 加到 128，throughput 可能從 6,000 漲到 9,000 tok/s，
但 p95 TTFT 從 600 ms 變成 4,000 ms，goodput 反而從 5,800 掉到 0。
只看 throughput 會做出錯誤的決策。</p>""")

# =============================================== CH9 負載測試 =============
chapter("CH9 兩階段負載測試", "ok")

slide("兩階段負載測試", kind="section", accent="ok", body="""
<div class="no">CHAPTER 09</div>
<h1>兩階段負載測試</h1>
<div class="csub">「這台機器能服務幾個人」沒有單一答案。它是一個條件式：
在這個 workload、這組 SLO、這個引擎版本之下，能承諾多少。
這一章給出一套可重現的流程。</div>
<ul>
 <li>閉環 vs 開環：為什麼一定要兩階段</li>
 <li>Stage 1：找飽和點（機器的極限）</li>
 <li>Stage 2：找 SLO 容量（可以承諾的量）</li>
 <li>SLO 邊界搜尋的三段式做法</li>
 <li>從 RPS 換算服務人數：Little's Law 與它的前提</li>
 <li>報告該寫什麼</li>
</ul>""",
      notes="<p>這一章要交出來的是一份別人可以重跑的測試報告，而不是一個最高 tok/s 的數字。</p>")

slide("閉環 vs 開環", "兩種測試回答的是完全不同的問題",
      sources=["bench"], accent="ok", body=f"""
{gist("閉環量「機器最多能做多少」，開環量「能承諾多少」。只做閉環，你永遠看不到佇列失控的那條線。")}
{fig("twostage", F4.fig_twostage(), 2, [
 "Stage 1 閉環：固定 N 個 client，每個回一個發一個。",
 "Stage 2 開環：依 Poisson 到達率發送，不管伺服器來不來得及。",
 "為什麼不能只做 Stage 1：閉環自帶背壓，延遲永遠不會爆炸。真實使用者不會等你。"],
 accent="ok")}""",
      notes="""<p>負載測試最常見的方法論錯誤就是這個。閉環測試（大多數工具的預設）
會讓系統看起來一直很穩，因為 client 在幫伺服器限流。</p>
<p>vLLM 的 <code>bench serve</code> 用 <code>--request-rate</code> 控制開環到達率，
用 <code>--max-concurrency</code> 控制閉環併發。兩個都要跑。</p>""")

slide("Stage 1：找飽和點", "控制併發，量 achieved throughput",
      sources=["bench", "anat"], accent="compute", body=f"""
{gist("併發從 1 開始往上加，看 throughput 什麼時候不再成長，那裡就是這台機器的物理極限。")}
{widget("sweep")}""",
      notes=f"""<p>操作方式：把上面切到「Stage 1 閉環」，拖動控制變數看兩張圖怎麼變。</p>
<p>Stage 1 要記錄的：<br>
① throughput 開始飽和的併發數，對照 CH1 的 N* ≈ {KNEE:.0f}；
如果沒開 spec，飽和點的併發數應該接近 N*。<br>
② 飽和之後 TPOT 的成長速率。<br>
③ 有沒有出現 preempt（那代表 cache 不足，不是算力飽和）。</p>
<p>再次提醒：曲線是教學示例，用排隊理論生成。</p>""")

slide("Stage 2：找 SLO 容量", "控制到達率，看佇列與延遲是否穩定",
      sources=["bench", "gp"], accent="mem", body=f"""
{gist("到達率低於容量時，吞吐等於到達率、延遲平穩；一超過，延遲就以 1/(1−ρ) 爆炸。")}
{widget("sweep2")}
<div class="cols3" style="margin-top:2px">
 {pane("穩定的判準（三個都要成立）", '''<ul style="font-size:12.8px">
 <li>佇列長度沒有往上的趨勢</li>
 <li>throughput ≈ 到達率，沒有被削平</li>
 <li>p95 延遲在量測窗內平穩，不隨時間漂移</li>
 </ul>''', "ok")}
 {pane("每個點要跑多久", '''<p style="font-size:13px">暖機 60 秒 + 量測 120 秒是常見起點。
 判斷是否穩定：把量測窗切成前後兩半，如果 p95 差超過 10%，代表還沒穩定。</p>''', "compute")}
 {warn("到達率必須是 Poisson（或真實 trace）而不是等間隔。等間隔會低估佇列效應，讓容量估計偏樂觀。")}
</div>""",
      notes="""<p>把 widget 切到「Stage 2 開環」，拖動到達率。<br>
低於容量時：吞吐跟著漲、延遲平穩。<br>
超過容量時：吞吐被削平（機器已滿）、延遲垂直上升。<br>
紅色虛線就是 SLO 容量。</p>
<p>ρ = 到達率 / 服務率是利用率。M/M/1 的等待時間 ∝ 1/(1−ρ)，
所以 ρ = 0.9 時延遲是空載的 10 倍，ρ = 0.95 時是 20 倍。
容量規劃要留 20–30% 餘裕的理由就在這。</p>""")

slide("SLO 邊界搜尋", "粗掃找範圍，細掃逼近，再重複確認",
      sources=["bench", "gp"], accent="ok", body=f"""
{gist("SLO 容量是一個邊界值，要用搜尋找出來，而且必須驗證可重現。")}
{fig("slo", F4.fig_slo(), 5, [
 "橫軸是到達率，縱軸是 p95 TTFT。",
 "先畫出 SLO 那條水平線。沒講定 SLO，就談不上容量。",
 "① 粗掃：大步長找出「還好」與「已經爆」之間的區間。",
 "② 細掃：在區間內二分逼近交點。",
 "③ 重複確認：邊界點至少跑 3 次，確認可重現、量測窗內沒有趨勢。"],
 accent="ok")}""",
      notes="""<p>常見錯誤：<br>
① 只跑一次就下結論，而邊界點的變異最大。<br>
② 用平均值取代 p95，邊界會被高估。<br>
③ 量測窗太短，系統還沒進入穩定態就記錄。<br>
④ 每個點之間沒有清空 prefix cache，導致後面的點虛胖。</p>""")

slide("從 RPS 換算服務人數", "Little's Law 與它的四個前提",
      sources=["bench"], accent="ok", body=f"""
{gist("λ = goodput ÷ 平均輸出長度；人數 = λ ÷ 每人的到達率。只有在系統穩定時才成立。")}
{widget("little")}""",
      notes="""<p>換算鏈條要一步一步講：<br>
① 從 Stage 2 得到 SLO 容量（req/s）。注意用的是 goodput，不是 throughput。<br>
② 每人每小時發幾次請求 → 每人的到達率。<br>
③ 人數 = 系統到達率 ÷ 每人到達率。</p>
<p>Little's Law（L = λW）在這裡的用途是<b>驗算</b>：算出來的 L
應該小於等於你設定的 max_num_seqs。如果 L 比 max_num_seqs 大很多，
代表設定不一致，數字有問題。</p>
<p>四個前提：系統穩定、輸出長度分布與量測窗一致、SLO 在該吞吐下滿足、
使用者行為模型（每小時輪數）有依據。任何一個不成立，人數就不能引用。</p>""")

slide("報告該寫什麼", "一份可以被別人重跑的容量報告",
      sources=["bench", "gp", "anat"], accent="ok", body=f"""
{gist("容量數字沒有附上這張表的內容，就無法被驗證，也無法被比較。")}
<div class="cols">
 {pane("① 測試設定", '''<table class="compact"><tbody>
 <tr><td>引擎</td><td>vLLM 版本、容器 image、啟動旗標全文</td></tr>
 <tr><td>模型</td><td>repo 名稱 + <b>revision hash</b>、量化方式</td></tr>
 <tr><td>起草器</td><td>repo + revision、method、num_speculative_tokens</td></tr>
 <tr><td>硬體</td><td>GPU 型號、數量、TP/EP/DP 設定、驅動版本</td></tr>
 <tr><td>快取</td><td>kv-cache-dtype、max-model-len、max-num-seqs、
  max-num-batched-tokens、gpu-memory-utilization、prefix caching 開關</td></tr>
 </tbody></table>''', "compute")}
 {pane("② Workload", '''<table class="compact"><tbody>
 <tr><td>輸入長度</td><td>分布（不只是平均）：p50 / p95 / max</td></tr>
 <tr><td>輸出長度</td><td>同上；是否設 max_tokens 上限</td></tr>
 <tr><td>前綴重複度</td><td>prefix cache 命中率（開 spec 前後都要量）</td></tr>
 <tr><td>思考模式</td><td>thinking 開／關，會大幅改變輸出長度</td></tr>
 <tr><td>到達模式</td><td>Poisson / 真實 trace / 等間隔</td></tr>
 </tbody></table>''', "mem")}
</div>
<div class="cols" style="margin-top:2px">
 {pane("③ SLO 與量測", '''<table class="compact"><tbody>
 <tr><td>SLO</td><td>指標 + 百分位 + 門檻，例如「p95 TTFT ≤ 800 ms 且 p95 TPOT ≤ 40 ms」</td></tr>
 <tr><td>量測窗</td><td>暖機時間、量測長度、重複次數</td></tr>
 <tr><td>穩定性證據</td><td>佇列長度時間序列、前後半窗的 p95 差異</td></tr>
 </tbody></table>''', "ok")}
 {pane("④ 結果", '''<table class="compact"><tbody>
 <tr><td>Stage 1</td><td>飽和併發數、峰值 throughput、當時的 TPOT</td></tr>
 <tr><td>Stage 2</td><td>SLO 容量（req/s）、goodput、邊界點的重複結果</td></tr>
 <tr><td>換算</td><td>人數 + 使用者行為假設 + Little's Law 驗算</td></tr>
 <tr><td>對照組</td><td>開／關 speculative decoding、不同 k 的同條件比較</td></tr>
 </tbody></table>''', "spec")}
</div>
{take("濃縮成一行：<b>「容量」= f(模型版本, 量化, 引擎設定, workload, SLO, 硬體)</b>。"
      "少寫任何一個自變數，那個數字就不能被引用。")}""",
      notes="""<p>這一頁可以直接拿去當報告模板的目錄。</p>
<p>特別提醒 revision hash：HuggingFace 上的模型會更新，
不記錄 revision 的話，三個月後重跑可能得到不同結果卻找不到原因。
本報告用的三個 revision 都寫在附錄 F。</p>""")

# =============================================== CH10 結論 ================
chapter("CH10 結論", "compute")

slide("結論", kind="section", body="""
<div class="no">CHAPTER 10</div>
<h1>可以帶走的十個結論</h1>
<div class="csub">以及一張「該轉哪個旋鈕」的決策圖，跟一份誠實的「這份報告沒有回答什麼」。</div>
<ul>
 <li>五個關於機制的結論</li>
 <li>五個關於設定與量測的結論</li>
 <li>從症狀出發的調校決策圖</li>
 <li>必須自己量的六件事</li>
</ul>""",
      notes="<p>最後 5 分鐘。若時間不夠，直接跳到「十個結論」那一頁唸完即可。</p>")

slide("十個結論", "把整份報告壓成十句話", accent="ok", body=f"""
<div class="cols">
 <div>
  {pane("機制", f'''<ol style="font-size:13.2px">
  <li><b>decode 是 memory-bound。</b>一次 forward 讀完整份權重，只服務 B 個 token。
   所有最佳化都在提高「每次讀取分攤到的 token 數」。</li>
  <li><b>B200 的屋頂線轉折點是 N* ≈ {KNEE:.0f} token/step，而且與精度無關。</b>
   量化讓兩邊同時變快，不會改變你的相對位置。</li>
  <li><b>混合架構把 KV cache 砍掉 3/4，但換來固定的 SSM state。</b>
   122B 的 KV 只有 {Q35.kv_bytes_per_token('fp8')/KiB:.0f} KiB/token（FP8），
   Qwen3-32B 是 {Q332.kv_bytes_per_token('bf16')/KiB:.0f} KiB。</li>
  <li><b>SSM state 是每序列 ~{mib(Q35.ssm_bytes(0),0)} MiB 的固定成本。</b>
   短對話時它<b>主導</b>記憶體：122B 要到 {Q35.crossover_tokens('fp8',3):,.0f} token
   KV 才追上它。</li>
  <li><b>MoE 省算力不省記憶體，而且 batch 一大就要讀滿所有 expert。</b>
   122B 從 batch 1 的 {gb(Q35.step_bytes(1))} GB 長到高併發的 {gb(Q35.step_bytes(2000))} GB。</li>
  </ol>''', "compute")}
 </div>
 <div>
  {pane("設定與量測", f'''<ol start="6" style="font-size:13.2px">
  <li><b>NVFP4 只量化了 routed expert。</b>36 層 GDN、12 層 attention、lm_head、
   MTP head 全部還是 BF16，佔了 checkpoint 的 21.9% 與 batch-1 頻寬的一半。</li>
  <li><b>NVFP4 讓 122B 單卡跑得動</b>（{gb(Q35.real_weight_bytes)} GB vs
   BF16 的 {gb(Q35.real_bf16_bytes)} GB），可以用 DP 取代 TP，省掉 all-reduce。</li>
  <li><b>speculative decoding 的效益由 B × (k+1) 是否超過 N* 決定。</b>
   27B 配 DFlash2 k=7 → 安全併發約 {KNEE/8:.0f}；超過就開始賠。</li>
  <li><b>接受率不是常數。</b>context 從 2k 長到 30k，實測平均接受率從 65% 掉到 39%，
   吞吐從 +129% 變成 −51%。長 context 一定要重量。</li>
  <li><b>容量是一個條件式，不是一個數字。</b>
   f(模型版本, 量化, 引擎設定, workload, SLO, 硬體)。少一個自變數就不能引用。</li>
  </ol>''', "ok")}
 </div>
</div>""",
      notes="""<p>如果只能記住三句：<br>
① decode 是 memory-bound，N* ≈ 292。<br>
② 混合模型要分開算 KV 與 SSM，短對話由 SSM 主導。<br>
③ speculative decoding 看 B×(k+1) 有沒有超過 N*。</p>""")

slide("該轉哪個旋鈕", "從症狀出發的決策圖", accent="compute", body=f"""
{gist("先看指標，再動旗標。一次只動一個，每次用同一組 workload 與 SLO 重測。")}
{fig("dec", F4.fig_decision(), 5, [
 "先確認症狀是什麼。別在沒看指標之前就開始調參數。",
 "<b>TTFT 太長</b>：問題在 prefill 或排隊。",
 "<b>TPOT / ITL 太長</b>：問題在 decode 的每步成本或併發太高。",
 "<b>吞吐上不去</b>：問題在沒有把 N 推到 N*，或推過頭了。",
 "<b>常常 preempt</b>：問題在 cache 不夠。這屬於容量問題，不是效能問題。",
 "共通原則：一次只動一個旗標，記錄每次的結果。"], accent="compute")}""",
      notes="""<p>這張圖可以印出來貼在牆上。</p>
<p>第四支要特別講：preempt 是<b>容量</b>訊號，不是效能訊號。
看到 preempt 就別再想「怎麼調快」，該想的是「是不是不該收這麼多請求」。</p>""")

slide("下一步：這份報告沒有回答的問題", "誠實列出邊界", accent="mem", body=f"""
{gist("這份報告給的是機制、公式與可驗證的架構數字。實測數字必須自己跑。")}
<div class="cols">
 {pane("必須自己量的", '''<ul style="font-size:13.3px">
 <li>兩個模型在你的 workload 上的<b>實際</b> throughput / TTFT / TPOT 曲線</li>
 <li>speculative decoding 的<b>實際</b>逐位置接受率與 τ</li>
 <li>開／關 spec 的 goodput 對照（同一組 SLO）</li>
 <li>prefix cache 命中率，以及開 spec 之後的變化</li>
 <li>量化對<b>你的</b>驗收集的影響（不是官方 benchmark）</li>
 <li>長 context（&gt; 32k）的接受率與吞吐</li>
 </ul>''', "mem")}
 {pane("這份報告刻意沒有涵蓋的", '''<ul style="font-size:13.3px">
 <li>訓練與微調</li>
 <li>多模態（視覺塔）的容量影響，只在參數帳本裡列出大小</li>
 <li>P/D 分離（prefill–decode disaggregation）</li>
 <li>多節點部署與網路拓撲</li>
 <li>成本模型（$/百萬 token）</li>
 <li>模型品質的橫向評比</li>
 </ul>''', "neutral")}
</div>
{take("建議的下一步：用 CH9 的兩階段流程，對兩個模型各跑一次基準測試，"
      "並把結果填回 CH6 與 CH7 的試算器，驗證解析模型與實測的差距有多大。"
      "解析上限通常是實測的 1.7–3 倍；知道自己的比例之後，"
      "以後就能用試算器快速估算新設定。")}""",
      notes="""<p>把邊界寫清楚，報告才站得住。</p>
<p>那個「1.7–3 倍」的說法要說清楚是經驗值：roofline 只算權重讀取與張量運算，
不含 attention kernel、MoE routing、all-to-all、取樣、Python 排程與 CUDA graph 之外的開銷。
一旦量過自己的比例，這個模型就變成很好用的快速估算工具。</p>""")

# ================================================== 附錄 ==================
chapter("附錄", "neutral")

slide("附錄", kind="section", accent="neutral", body="""
<div class="no">APPENDIX</div>
<h1>附錄</h1>
<div class="csub">給 Q&amp;A 與後續查證用。</div>
<ul>
 <li>A　名詞表（也可以隨時按 G 打開）</li>
 <li>B　config 與 checkpoint 的原始證據</li>
 <li>C　可以直接複製的命令</li>
 <li>D　GDN 的數學細節</li>
 <li>E　單卡與多卡：TP / EP / DP</li>
 <li>F　來源清單與資料界線</li>
</ul>""", notes="")

_gl = sorted(K.GLOSSARY.items(), key=lambda x: x[0].lower())
_half = (len(_gl) + 1) // 2
for _pi, _part in enumerate((_gl[:_half], _gl[_half:])):
    slide(f"附錄 A：名詞表（{_pi+1}/2）",
          f"{_part[0][0]} – {_part[-1][0]}　·　共 {len(K.GLOSSARY)} 條；"
          f"隨時按 <kbd>G</kbd> 也能打開", accent="neutral", body=f"""
<div style="column-count:3;column-gap:22px;font-size:11.8px;line-height:1.45">
{"".join(f'<div style="break-inside:avoid;margin-bottom:8px">'
         f'<b style="color:var(--ink)">{k}</b><br>'
         f'<span style="color:var(--ink2)">{v}</span></div>' for k, v in _part)}
</div>""",
          notes="<p>這一頁與 G 鍵的名詞表是同一份資料，由同一個字典產生。</p>")

slide("附錄 B：config 與 checkpoint 的原始證據", "所有架構數字的來源",
      sources=["c35", "c38", "nvi", "i38", "nvq"], accent="neutral", body=f"""
<div class="cols">
 {pane("關鍵 config 欄位", code(
"""<span class="c">// Qwen3.5-122B-A10B / text_config</span>
"num_hidden_layers": <span class="n">48</span>,
"full_attention_interval": <span class="n">4</span>,
"hidden_size": <span class="n">3072</span>,
"num_attention_heads": <span class="n">32</span>,
"num_key_value_heads": <span class="n">2</span>,
"head_dim": <span class="n">256</span>,
"partial_rotary_factor": <span class="n">0.25</span>,
"attn_output_gate": <span class="k">true</span>,
"linear_num_key_heads": <span class="n">16</span>,
"linear_num_value_heads": <span class="n">64</span>,
"linear_key_head_dim": <span class="n">128</span>,
"linear_value_head_dim": <span class="n">128</span>,
"linear_conv_kernel_dim": <span class="n">4</span>,
<span class="hl">"mamba_ssm_dtype": "float32"</span>,
"num_experts": <span class="n">256</span>,
"num_experts_per_tok": <span class="n">8</span>,
"moe_intermediate_size": <span class="n">1024</span>,
"shared_expert_intermediate_size": <span class="n">1024</span>,
"mtp_num_hidden_layers": <span class="n">1</span>,
"vocab_size": <span class="n">248320</span>,
"max_position_embeddings": <span class="n">262144</span>"""))}
 {pane("量化設定與 dtype 統計", code(
f"""<span class="c">// nvidia/…-NVFP4 hf_quant_config.json</span>
"quant_algo": <span class="s">"NVFP4"</span>,
"kv_cache_quant_algo": <span class="s">"FP8"</span>,
"group_size": <span class="n">16</span>

<span class="c">// safetensors dtype 統計（HF API）</span>
122B-NVFP4  BF16 {Q35.real_dtypes['BF16']:>15,}
            F8_E4M3 {Q35.real_dtypes['F8_E4M3']:>12,}
            U8   {Q35.real_dtypes['U8']:>15,}
            <span class="hl">total_size {Q35.real_weight_bytes:,} B</span>

27B-FP8     BF16 {Q38.real_dtypes['BF16']:>15,}
            F8_E4M3 {Q38.real_dtypes['F8_E4M3']:>12,}
            <span class="hl">weight files {Q38.real_weight_bytes:,} B</span>

<span class="c">// Qwen3.8-27B / text_config 的差異</span>
"num_hidden_layers": <span class="n">64</span>   <span class="c">// 48 GDN + 16 full</span>
"hidden_size": <span class="n">5120</span>
"num_key_value_heads": <span class="n">4</span>
"linear_num_value_heads": <span class="n">48</span>
"intermediate_size": <span class="n">17408</span>  <span class="c">// dense</span>

<span class="c">// incoai/Qwen3.8-27B-DFlash2</span>
"dflash_config": {{ "block_size": <span class="n">8</span>,
  "selector_rank": <span class="n">256</span>, "selector_top_k": <span class="n">16</span>,
  "target_layer_ids": [<span class="n">5,19,33,47,61</span>] }}"""))}
</div>""",
      notes=f"""<p>驗證方式：把 config 的維度乘出來，逐一對應 checkpoint 的 tensor 名稱，
再與 HF API 回報的 dtype 元素數比對。三個模型的誤差分別是
{abs(Q35.params_total/(Q35.real_bf16_bytes/2)-1)*100:.3f}%、
{abs(Q38.params_total/(Q38.real_bf16_bytes/2)-1)*100:.3f}%、
{abs(Q332.params_total/(Q332.real_bf16_bytes/2)-1)*100:.3f}%。</p>
<p>NVFP4 的驗算最漂亮：48 × 256 × 3 × 3072 × 1024 = 115,964,116,992 個 4-bit 權重，
除以 2 得到 57,982,058,496 個 U8，與統計完全相同；再除以 group_size 16
得到 7,247,757,312 個 FP8 block scale，也完全相同。</p>""")

slide("附錄 C：可以直接複製的命令", "服務啟動與壓測",
      sources=["bench", "nv4", "df2"], accent="neutral", body=f"""
<div class="cols">
 {pane("Stage 1：閉環飽和測試", code(
"""vllm bench serve \\
  --backend openai-chat \\
  --base-url http://HOST:8000 \\
  --endpoint /v1/chat/completions \\
  --model MODEL_NAME \\
  --dataset-name random \\
  --random-input-len 2048 \\
  --random-output-len 512 \\
  --num-prompts 600 \\
  <span class="hl">--max-concurrency 32</span> \\
  --percentile-metrics ttft,tpot,itl,e2el \\
  --metric-percentiles 50,95,99 \\
  --save-result --result-filename c32.json

<span class="c"># 併發掃描：1 2 4 8 16 32 64 128</span>
<span class="c"># 記錄每一點的 throughput 與 p95 TPOT</span>"""))}
 {pane("Stage 2：開環 SLO 容量測試", code(
"""vllm bench serve \\
  --backend openai-chat \\
  --base-url http://HOST:8000 \\
  --endpoint /v1/chat/completions \\
  --model MODEL_NAME \\
  --dataset-name random \\
  --random-input-len 2048 \\
  --random-output-len 512 \\
  --num-prompts 1200 \\
  <span class="hl">--request-rate 12 \\
  --burstiness 1.0</span> \\
  <span class="hl">--goodput ttft:800 tpot:40</span> \\
  --percentile-metrics ttft,tpot,itl,e2el \\
  --metric-percentiles 50,95,99 \\
  --save-result --result-filename r12.json

<span class="c"># burstiness 1.0 = Poisson</span>
<span class="c"># 到達率掃描：粗掃 → 細掃 → 邊界重複 3 次</span>"""))}
</div>
{warn("執行前一定要跑 <code>vllm bench serve --help</code> 核對<b>你安裝的版本</b>的旗標名稱。"
      "另外：每一輪之間若要公平比較，記得重啟服務或清空 prefix cache，"
      "否則後面的測試點會因為快取命中而虛胖。")}""",
      notes="""<p>兩個對照組是必要的：<br>
① 開 spec / 關 spec，其餘完全相同。<br>
② 不同 k（例如 27B 的 k=7 與 k=3）。<br>
沒有對照組就無法歸因。</p>""")

slide("附錄 D：GDN 的數學細節", "單一 head 的更新式",
      sources=["gdn", "gdnc", "mut"], accent="ssm", body=f"""
<div class="cols w6-4">
 <div>
  <div class="formula">S<sub>t</sub> = S<sub>t−1</sub>( <em>α<sub>t</sub></em>
   ( I − <span class="c2">β<sub>t</sub></span> k<sub>t</sub>k<sub>t</sub><sup>T</sup> ) )
   + <span class="c2">β<sub>t</sub></span> v<sub>t</sub>k<sub>t</sub><sup>T</sup></div>
  <div class="formula" style="margin-top:6px">o<sub>t</sub> = S<sub>t</sub> q<sub>t</sub></div>
  <ul style="font-size:13.3px;margin-top:9px">
   <li>S<sub>t</sub> ∈ ℝ<sup>d<sub>v</sub> × d<sub>k</sub></sup>，每個 value head 一份</li>
   <li>α<sub>t</sub> ∈ (0,1)：decay gate，由 <code>in_proj_a</code> 與
    <code>A_log</code>、<code>dt_bias</code> 產生</li>
   <li>β<sub>t</sub> ∈ (0,1)：delta rule 強度，由 <code>in_proj_b</code> 產生</li>
   <li>q、k、v 由 <code>in_proj_qkv</code> 一次產生，再經短 convolution
    （<code>conv1d</code>，kernel = {Q35.conv_k}）</li>
   <li>輸出還會乘上 <code>in_proj_z</code> 產生的 output gate，
    並過 <code>norm</code> 之後由 <code>out_proj</code> 投影回 hidden</li>
   <li>此處省略 normalization 與 head 之間的細節</li>
  </ul>
 </div>
 <div>
  {pane("vLLM 的 state shape", code(
f"""conv_dim = d_k·n_k·<span class="n">2</span> + d_v·n_v

conv_state_shape = (
  conv_dim / tp,
  conv_kernel − <span class="n">1</span> + <span class="hl">num_spec</span>,
)

temporal_state_shape = (
  n_v / tp, d_v, d_k,
)

<span class="c"># 122B: conv_dim = {Q35.conv_dim:,}</span>
<span class="c">#       state = ({Q35.nv}, {Q35.dv}, {Q35.dk})</span>
<span class="c"># 27B : conv_dim = {Q38.conv_dim:,}</span>
<span class="c">#       state = ({Q38.nv}, {Q38.dv}, {Q38.dk})</span>"""), "ssm")}
  {pane("為什麼 chunkwise 能跑滿 Tensor Core", f'''
  <p style="font-size:13px">把 α 的累積乘積提出來之後，gate 只剩下逐元素乘法，
  不破壞矩陣乘法的結構。論文的說法：</p>
  {quote("The gating term only performs elementwise multiplication with intermediate variables without affecting matrix multiply structures.", "Gated Delta Networks, arXiv 2412.06464")}''')}
 </div>
</div>""",
      notes="""<p>如果聽眾問「為什麼不用 Mamba2 就好」：Mamba2 只有整體衰減，
沒辦法針對某個特定的 key 精準改寫。在需要「更新某個事實」的任務上（例如多輪對話裡改口），
delta rule 那一項很重要。</p>""")

slide("附錄 E：單卡與多卡", "TP / EP / DP 各自改變什麼",
      sources=["arch", "nv4"], accent="compute", body=f"""
{fig("par", F4.fig_parallel(), 4, [
 "三種平行化改變的是不同的東西。",
 "<b>Tensor Parallel</b>：切矩陣，每層都要 all-reduce。",
 "<b>Expert Parallel</b>：切專家，每層都要 all-to-all。",
 "<b>Data Parallel</b>：複製整份模型，沒有跨卡通訊。",
 "NVFP4 讓 122B 可以走 DP 取代 TP。這算本次評估最重要的工程結論之一。"],
 accent="compute")}""",
      notes=f"""<p>補充 TP 對 KV cache 的限制：TP 會切 KV head，
但 122B 只有 {Q35.n_kv} 組 KV head，所以 TP &gt; 2 時 KV head 不夠切，
vLLM 會複製它們，每張卡的 KV cache 不會再變小。
TP=8 對 122B 因此特別不划算。</p>
<p>DP 的限制：每張卡都要放得下整份模型與足夠的 cache。
122B NVFP4 是 {gib(Q35.real_weight_bytes)} GiB，留給 cache 約
{M.budget(Q35, draft=True)['cache']:.0f} GiB，
所以單卡副本適合中等 context（≤ 64k）的高併發場景；
需要 256k context 時還是得用 TP 把 cache 空間合起來。</p>""")

slide("附錄 F：來源與資料界線", f"查核日期 {K.CHECK}", accent="neutral", body=f"""
<div class="cols w6-4">
 <div>
  <h3 class="ph">本報告引用的來源（點擊開啟）</h3>
  <div style="display:flex;flex-wrap:wrap;gap:5px">
  {"".join(f'<a href="{u}" target="_blank" rel="noopener" style="font-size:11px;'
           f'color:var(--mut);text-decoration:none;border:1px solid var(--line);'
           f'border-radius:5px;padding:3px 7px;background:#fff">{n}</a>'
           for k, (n, u) in K.SOURCES.items() if u != "#")}
  </div>
 </div>
 <div>
  {pane("checkpoint revision", code(
f"""nvidia/Qwen3.5-122B-A10B-NVFP4
  98915d837c4e7c87ac8296d02e89de19b3207e6d
Qwen/Qwen3.8-27B-FP8
  017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
incoai/Qwen3.8-27B-DFlash2
  dedf8df68adfb1afeaf7b7480c0a0243108177b4
Qwen/Qwen3.5-122B-A10B
  dc4d348443bc740c68e2d77492492c11606384d5
Qwen/Qwen3.8-27B
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Qwen/Qwen3-32B
  9216db5781bf21249d130ec9da846c4624c16137"""), "neutral")}
 </div>
</div>
<div class="cols3" style="margin-top:2px">
 {pane("<span class='tagc g'>實測 / 官方</span>", '''<p style="font-size:12.6px">
 架構參數、參數量、checkpoint 位元組數、量化設定、NVFP4 精度評測、
 DFlash2 的接受長度與加速倍數、vLLM issue 的實測數字、B200 規格。
 全部來自上列來源，可直接引用。</p>''', "ok")}
 {pane("<span class='tagc'>公式推導</span>", '''<p style="font-size:12.6px">
 KV / SSM cache 大小、記憶體帳本、roofline 轉折點、每 step 的位元組流量、
 speculative decoding 的加速估計。公式都寫在頁面上。
 標示「解析上限」者，實測通常是其 30–60%。</p>''')}
 {pane("<span class='tagc m'>教學示例</span>", '''<p style="font-size:12.6px">
 CH9 的所有 throughput / 延遲 / 佇列曲線與 SLO 掃描圖。
 以排隊理論生成，<b>不是任何硬體的實測結果</b>。
 正式報告必須替換成自家實測資料。</p>''', "mem")}
</div>
<div class="hint" style="margin-top:4px">圖片：continuous batching 與 PagedAttention 兩張為使用者提供，
出自 vLLM 官方部落格。其餘所有圖表均為本報告自行繪製的 SVG。</div>""",
      notes=f"""<p>資料界線是這份報告最重要的自我約束。查核日期 {K.CHECK}；
HuggingFace 上的模型會更新，引用時務必連同 revision 一起寫。</p>""")
