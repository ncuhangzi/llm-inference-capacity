from pathlib import Path
import base64, json

ROOT=Path(__file__).parent
SOURCES={
'q3':('Qwen3-32B config','https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json'),
'q35':('Qwen3.5-122B-A10B 模型卡','https://huggingface.co/Qwen/Qwen3.5-122B-A10B'),
'c35':('Qwen3.5-122B-A10B config','https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json'),
'q38':('Qwen3.8-27B 模型卡','https://huggingface.co/Qwen/Qwen3.8-27B'),
'c38':('Qwen3.8-27B config','https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json'),
'c36':('Qwen3.6-27B config','https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json'),
'arch':('vLLM Architecture Overview','https://docs.vllm.ai/en/v0.15.0/design/arch_overview/'),
'anatomy':('Inside vLLM','https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html'),
'runner':('Model Runner V2','https://vllm-project.github.io/2026/03/24/mrv2.html'),
'gdn':('Gated Delta Networks 論文','https://arxiv.org/html/2412.06464v1'),
'gdncode':('vLLM Qwen GDN implementation','https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'),
'state':('vLLM state shape calculator','https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py'),
'hybrid':('Hybrid KV Cache Manager','https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/'),
'bench':('vLLM bench serve CLI','https://docs.vllm.ai/en/latest/cli/bench/serve/'),
'metrics':('vLLM benchmark 計算實作','https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py'),
'b200':('NVIDIA DGX B200 User Guide','https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html'),
'discussion':('使用者提供：計算 LLM 同時負載','https://chatgpt.com/share/6a9ff5da-77cc-83ee-bcc6-801d56bb7d8f'),
'q3report':('Qwen3 Technical Report','https://arxiv.org/abs/2505.09388'),
'paged':('vLLM PagedAttention','https://blog.vllm.ai/2023/06/20/vllm.html'),
}
slides=[]
def slide(title, sub, body, notes='', sources=(), kind='', demo=''):
    slides.append(dict(title=title,sub=sub,body=body,notes=notes,sources=list(sources),kind=kind,demo=demo))
def cols(a,b): return f'<div class="cols"><div>{a}</div><div>{b}</div></div>'
def ul(*items): return '<ul>'+''.join(f'<li>{x}</li>' for x in items)+'</ul>'
def table(headers, rows): return '<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def flow(items): return '<div class="flow">'+''.join(f'<div class="node">{x}</div>'+('<span class="arrow">→</span>' if i<len(items)-1 else '') for i,x in enumerate(items))+'</div>'
def formula(s): return f'<div class="formula">{s}</div>'
def callout(s): return f'<p class="takeaway">{s}</p>'
def demo(name): return f'<div class="demo" data-demo="{name}"></div>'

slide('LLM 推論與服務容量','模型運算、狀態記憶體與負載測試',
 '<div class="coverline">一個 request 進來後，GPU 做了什麼？</div><div class="coverline muted">當 request 變多，瓶頸出現在哪裡？</div><p class="covermeta">Qwen3 · Qwen3.5-122B-A10B · Qwen3.8-27B<br>vLLM serving / NVIDIA B200 180 GB</p>',
 '建議主講約 50 分鐘。模型基礎快速帶過，將時間留給混合架構、cache 與負載測試。空白鍵逐步演示，方向鍵換頁，N 顯示講者筆記，O 顯示目錄。資料查核日期 2026-09-08。',kind='cover')
slide('報告路線','同一個 request，從模型內部看到服務外部',
 '<div class="agenda">'+''.join(f'<div><b>0{i+1}</b><span>{x}</span><small>{t}</small></div>' for i,(x,t) in enumerate([('LLM 與 vLLM 執行堆疊','10 分鐘'),('標準 Transformer 與 MoE','10 分鐘'),('Qwen 混合架構與 cache','15 分鐘'),('B200 記憶體與效能限制','5 分鐘'),('兩階段負載測試與容量','15 分鐘')]))+'</div>',
 '主簡報 34 頁，後面附錄可供問答。本文所有吞吐與 SLO 曲線都是教學示例，不代表任何硬體的實測成績。型號參數與 cache 算式則依官方設定推導。')
slide('Autoregressive inference','每次 forward 產生下一個 token 的分布',
 flow(['Token IDs','Embedding','Decoder layers × L','Final norm<br>LM head','Sampling'])+callout('Prompt 的最後位置產生第一個 output token；下一輪再將這個 token 送入模型。')+ul('推論時更新 activation 與 request state，模型權重通常固定。','基準流程一次 decode 一個 token；speculative decoding 另外處理。'),
 '這裡使用標準自回歸文字生成路徑，不討論訓練的 backward。LM head 將 hidden state 投影至 vocabulary logits。temperature、top-k、top-p 等控制抽樣。EOS、stop string 或長度上限決定何時結束。',['q3report'])
slide('vLLM 的執行堆疊','API 到 GPU 的主要責任邊界',
 '<div class="stack">'+''.join(f'<div><b>{a}</b><span>{b}</span></div>' for a,b in [('API server / frontend','HTTP、chat template、tokenization、串流回傳'),('Engine / scheduler','request 狀態、每輪 token budget、cache 配置'),('Executor / workers','分散式執行、各 GPU 的工作協調'),('Model runner','準備 tensors 與 metadata、graph、呼叫 model'),('Model / backend / kernels','模型 layer、attention / GDN / MoE、GPU 算子')])+'</div>',
 '概念圖聚焦 vLLM V1 系列共通責任，並非每個版本的程序拓樸。frontend 與 engine core 可在不同 process，worker 通常對應一張 GPU。MRV2 改變 runner 內部組織，報告不綁定未核對的預設 runner 選項。',['arch','runner'])
slide('Serve 不同模型，改動在哪裡？','Checkpoint、架構與硬體 backend 是不同層次的變更',
 table(['層次','同架構的新 checkpoint','加入新架構'],[
 ['REST API','多半共用','多半共用；新增模態或輸出語意可能擴充'],
 ['Tokenizer / parser','詞表、template、reasoning / tool parser','依模型格式調整'],
 ['Model / loader','設定、權重、quantization metadata','新 module、weight mapping、parallel 分割'],
 ['Runner / cache','通常沿用既有支援','視 state、metadata、graph 需求整合'],
 ['GPU kernels','選用相容 backend','缺少算子或 layout 才新增／最佳化']]),
 '不要把換模型說成必定改 runner。已有 Qwen3.5 架構支援時，新 checkpoint 可能只需要配置及解析器設定。模型差異也可能出現在 scheduler 的 cache policy，而不只 model.forward。',['arch','gdncode'])
slide('Continuous batching','每輪調度重新組合仍在執行的 requests',
 '<img class="provided" src="@@BATCH@@" alt="使用者提供的 individual、dynamic 與 continuous batching 對照圖"><p class="caption">使用者提供素材，圖上署名 Baseten。示意排程利用率，不代表等比例速度提升。</p>',
 '圖中的 dynamic batching 是整批收集後執行的示意名稱。Continuous batching 在 iteration 邊界讓完成者離開、新 request 加入，減少等待最慢序列的空槽。圖本身沒有表示 chunked prefill 或 prefill/decode 混合的完整細節。',['anatomy'])
slide('PagedAttention','Logical blocks 對應非連續的 physical KV blocks',
 '<img class="provided" src="@@PAGED@@" alt="使用者提供的 PagedAttention 動畫，顯示共享 prompt 與 KV block 對應"><p class="caption">使用者提供 GIF，保留原始動畫。共享前綴及寫入分歧需維持正確的 block ownership。</p>',
 'PagedAttention 解決 KV 的配置與存取問題，並不改變模型的 attention 定義。按 block 分配減少大型連續預留造成的浪費，末頁仍可能有內部碎片。Prefix caching 是利用這套配置重用前綴計算的另一層策略。GIF 自動循環，PDF 只會保留其中一幀。',['paged'])
slide('每一輪 scheduler 做什麼？','以 token budget 與 cache 可用量決定本輪工作',
 flow(['等待 / 執行中 requests','選取 token chunks','配置 cache slots','Runner forward','抽樣與回傳'])+
 cols(ul('新 request 可以進入 prefill。','長 prompt 可切成多個 prefill chunks。'),ul('執行中 request 繼續 decode。','資源不足可能等待或 preempt。'))+
 callout('一輪 batch 可以包含 prefill tokens 與 decode tokens；它們的運算形狀與成本不同。'),
 'Chunked prefill 的 chunk 是 serving 排程單位，後面的 GDN chunkwise 是數值演算法，兩者不可混用。切小 prefill chunk 可能減少 decode 被拖慢的時間，但增加排程或 launch overhead。實際取捨要測量。',['anatomy'])
slide('Qwen3-32B：標準 Transformer 基準','64 層 full attention，搭配 dense SwiGLU FFN',
 flow(['RMSNorm','QKV projection<br>QK norm / RoPE','GQA attention','Output projection<br>+ residual'])+
 flow(['RMSNorm','Gate / up projection','SiLU × up','Down projection<br>+ residual'])+
 table(['Hidden size','Q heads','KV heads','Head dim','FFN intermediate'],[['5120','64','8','128','25,600']]),
 'Qwen3 的 QK normalization 及 RoPE 在 attention 前作用於對應 tensors。Q head 數不一定由 hidden_size / head_dim 得到，本例 64×128=8192，投影後再回到 5120。不要將 Llama 某一版的維度假定套到 Qwen3。',['q3','q3report'])
slide('Prefill 與 decode','相同 layers，不同 token 維度與歷史資料存取',
 demo('tokens'),
 '演示刻意用 4 個 prompt tokens 與 3 個 output tokens。Prefill 以因果遮罩處理多個已知位置，並寫入各層 K/V。第一個 output 由 prompt 最後位置產生；第二個 output 的 forward 輸入是第一個 output。最後抽樣出的 token 若立刻結束，未必再被 forward，因此不必已存在 KV 中。',['q3report','anatomy'],demo='tokens')
slide('Decode 如何使用 KV cache','每層只新增目前位置的 K/V，再讀取歷史 K/V',
 demo('kv'),
 'Attention 是 softmax(q K^T / sqrt(d)) V，這裡省略 mask、head indexing 與實作融合。每個 layer 有自己的 cache。快取的是歷史 K/V，不是歷史 Q，也不是完整的模型答案。GQA 多個 query heads 共用較少 KV heads。',['q3report'],demo='kv')
slide('Prefill 與 decode 的效能形狀','瓶頸取決於 batch、context、kernel 與精度',
 table(['','Prefill','Decode'],[
 ['輸入位置數','多個 prompt tokens','基準為每個 request 一個新 token'],
 ['矩陣運算','大 GEMM，較易提高算術強度','小 batch 常受權重讀取與 launch 限制'],
 ['Attention','長序列全注意力計算量快速增加','每步掃讀更長歷史 KV'],
 ['服務觀察','影響 TTFT，可能干擾 decode','影響 ITL / TPOT 與輸出吞吐']])+callout('高 concurrency 能攤平部分權重讀取成本，同時也可能拉長每個人的 token 間隔。'),
 'Full attention prefill 的注意力計算對序列長度約二次成長，但 FlashAttention 不需將完整分數矩陣寫回 HBM。不能因此說所有 prefill 都 compute-bound、所有 decode 都 memory-bound；需 profiler 才能確認。MoE、長 context、大 batch 都可能改變瓶頸。',['anatomy'])
slide('Dense 與 MoE 的差別','MoE 改變 FFN 的計算路徑',
 demo('moe'),
 '這是 8 選 2 的教學示意，並非 122B 的實際 expert 數。Dense 每個 token 使用同一個 FFN，MoE 依 token 的 hidden state 選擇 experts，組合加權輸出。一般 attention / recurrent state 屬於 token mixer，MoE 屬於 FFN，兩個維度可獨立組合。',['q3report','q35'],demo='moe')
slide('MoE 的 activated parameters 與記憶體','少算一些 experts，仍需部署整組權重',
 cols('<div class="big">122B<span>total parameters</span></div>', '<div class="big amber">10B<span>activated parameters / token</span></div>')+
 ul('單個 token 只選部分 experts；同一個 batch 的 tokens 可能觸及更多 experts。','一般全 GPU 部署仍要存放所有 experts；offload 是另一項取捨。','Expert parallelism 引入 token dispatch / combine 與負載不均問題。'),
 '122B-A10B 的 A10B 是整體啟用參數的模型命名近似，不等於只有 10B 權重佔 HBM。每層 256 routed experts、8 routed 加 1 shared。激活量不能直接拿來計算整個 batch 的權重流量或預測 tokens/s。',['q35'])
slide('Qwen3.5 之後的混合堆疊','Token mixer 與 FFN 類型分開比較',
 table(['模型','Full attention','Gated DeltaNet','FFN'],[
 ['Qwen3-32B','64 層','無','Dense'],
 ['Qwen3.5-122B-A10B','12 層','36 層','MoE'],
 ['Qwen3.8-27B','16 層','48 層','Dense']])+ '<div class="layerpattern"><span>GDN</span><span>GDN</span><span>GDN</span><span class="attn">Attention</span></div>'+callout('混合模型每四層包含一層 full attention；每個 mixer 後仍有 FFN。'),
 'Qwen3.8-27B config 仍使用 Qwen3_5ForConditionalGeneration。與 Qwen3.6-27B 的主要層數、維度和配置相同，這裡直接用 3.8 的官方 config，不需要假設未知模型架構。多模態 encoder 另在一頁交代。',['q3','c35','c38','c36'])
slide('Gated DeltaNet 與 Mamba 的關係','共享 recurrent-state 思路，更新規則有所不同',
 cols('<h3>Gated DeltaNet</h3>'+ul('輸入產生 q、k、v 與控制 gates。','以 decay 與 delta rule 更新 state。','Qwen 混合模型使用這種 mixer。'),'<h3>Serving 的命名</h3>'+ul('vLLM 將 GDN 放在 mamba 相關模組下。','設定可能出現 mamba_ssm_dtype。','本報告稱它為 GDN recurrent state，並對應常說的 SSM cache。')),
 'SSM cache 是部署討論中的廣義用法。GDN 並不等於 Mamba2，不能由程式路徑帶有 mamba 就推論模型是 Mamba。GDN 論文將 gating 與 delta update 結合；不同架構的 convolution、normalization、gate 與 state shape 仍需逐一檢查。',['gdn','gdncode','c38'])
slide('GDN 的一次 state update','固定大小矩陣承接歷史資訊',
 demo('state'),
 '單一 head 的概念公式採 S 為 d_v × d_k。先將舊 state 乘 decay α，再以目前 k 查出預測值，用 β(v−prediction)k^T 寫入修正。最後 S q 讀出結果。這是前向 activation state 更新，不是在訓練模型權重。圖中矩陣色彩只顯示狀態改變，沒有語意或實測數值。',['gdn'],demo='state')
slide('Hybrid prefill 與 decode','GDN state 和 full-attention KV 必須一起正確延續',
 table(['階段','GDN layers','Full-attention layers'],[
 ['Prefill','chunkwise 計算 prompt 並保留終點 state','因果 attention 並寫入 prompt K/V'],
 ['Decode','讀取舊 state，更新為新 state','讀取歷史 KV，追加目前 K/V'],
 ['保存','recurrent matrix + convolution state','每層、每位置的 K/V'],
 ['隨 context 增長','最小 live state 大致固定','live KV 線性增長']])+callout('GDN prefill 可以使用 chunkwise 平行演算法；不必逐 token 呼叫 Python 迴圈。'),
 'GDN state 固定是指單條 sequence 的最小 live recurrent state，並不保證整個 serving cache 都固定。Prefix checkpoints、speculative rollback、allocator padding、暫存 tensors 都可增加記憶體。GDN 演算法 chunk 與 scheduler prefill chunk 是不同概念。',['gdn','hybrid','state'])
slide('兩個 Qwen 模型的一次 decode','沿著所有 layers 前進，兩種 cache 都會更新',
 demo('hybrid'),
 '動畫顯示一個四層 group，122B 重複 12 組、27B 重複 16 組。每個 layer 還包含 norm、residual 與 projection，為了閱讀省略於動畫，並非不執行。MoE 的 router 在各層各 token 重新決定。請按模型切換比較。',['c35','c38','q35'],demo='hybrid')
slide('KV cache 與 recurrent state 的生命週期','Live state 與可重用 prefix 是不同的儲存需求',
 flow(['新 request','建立／命中前綴','每輪延伸','完成或取消','回收／保留可重用資料'])+
 table(['機制','需要保存什麼','注意事項'],[
 ['Full attention','前綴各位置 K/V','共享 block、引用與 eviction'],
 ['GDN / SSM cache','對應前綴終點的 recurrent + conv state','只有最新 state，無法任意回到較早 prefix'],
 ['Hybrid prefix hit','各類 layer 都能接續的共同位置','支援程度、checkpoint policy 依引擎版本'],
 ['Speculative decode','可驗證與回復的狀態','額外 buffer、候選 token、commit / rollback']]),
 'Hybrid KV manager 設計文件本身標明根據特定 commit 撰寫，包含 work-in-progress 註記。此頁不將舊文件的限制當成所有最新版本的結論。部署時記錄版本、cache policy 與實際 startup log。MTP 訓練不表示 serve 時自動啟用 speculative decoding。',['hybrid','state'])
slide('KV cache 的容量計算','先算每條 sequence，再考慮並行、分片與配置開銷',
 formula('M<sub>KV</sub> = 2 × L<sub>attention</sub> × T × H<sub>KV</sub> × d<sub>head</sub> × bytes')+
 table(['模型','KV heads × head dim','BF16 KV / token','T = 32,768'],[
 ['Qwen3-32B','8 × 128','256 KiB','8 GiB'],
 ['Qwen3.5-122B-A10B','2 × 256','24 KiB','0.75 GiB'],
 ['Qwen3.8-27B','4 × 256','64 KiB','2 GiB']])+ '<p class="caption">模型全體、單條 sequence、無 prefix sharing。只計 full-attention KV；未含 GDN state、padding、runtime。1 GiB = 2³⁰ bytes。</p>',
 '2 代表 K 與 V。L 只計 full-attention layers。T 是已經 forward 並保留的 token 位置數。這是邏輯容量，TP 分片、KV head 複製與 block 配置可能改變每張 GPU 的佔用，不能總是一律除以 GPU 數。計算由官方 config 推導。',['q3','c35','c38'])
slide('SSM cache 的容量計算','GDN 的 recurrent matrix 與短 convolution state',
 formula('M<sub>state</sub> = L<sub>GDN</sub> × H<sub>V</sub> × d<sub>V</sub> × d<sub>K</sub> × 4 bytes')+
 table(['模型','FP32 recurrent matrices','BF16 conv history','合計最小 live state'],[
 ['122B-A10B','36 × 64 × 128 × 128 × 4 = 144 MiB','約 2.53 MiB','約 146.53 MiB / sequence'],
 ['27B','48 × 48 × 128 × 128 × 4 = 144 MiB','約 2.81 MiB','約 146.81 MiB / sequence']])+callout('並行序列增加時，state 仍會增加；「固定」只描述單序列對 context 長度的關係。'),
 'conv history 依 vLLM shape calculator，以 channels = 2 H_K d_K + H_V d_V，history = kernel_size−1 = 3，BF16 2 bytes，且 num_spec=0。state precision 採 config 的 float32。實際配置可能保留更多 convolution slots 或 checkpoints；這是最小邏輯 live state 估算。',['state','c35','c38'])
slide('Context 長度與 cache 成長','在相同 BF16 KV、FP32 recurrent state 假設下比較',
 demo('memory'),
 '互動圖以三個模型的官方維度做容量推導，並非量測。T 從 1024 到 32768，避免超出 Qwen3-32B config 的原始 context 上限。Hybrid 包含前頁 conv state。短 context 時固定 state 佔比較高，長 context 時 full-attention KV 逐漸主導。',['q3','c35','c38','state'],demo='memory')
slide('B200 180 GB 的記憶體帳本','權重與 cache 必須共享 HBM 預算',
 table(['模型','BF16 權重約值','1 byte / parameter 約值','單張 180 GB 的初步判讀'],[
 ['Qwen3-32B','64 GB','32 GB','仍需留 cache 與 runtime'],
 ['Qwen3.8-27B','54 GB','27 GB','BF16 有空間，容量取決於 workload'],
 ['Qwen3.5-122B-A10B','244 GB','122 GB','BF16 權重已超單卡；需分片或量化']])+
 formula('HBM = weights + KV + recurrent / conv state + activations + runtime + reserve')+
 '<p class="caption">GB 採 10⁹ bytes。以型號參數數量粗估，未計 scales、未量化層、vision / MTP 額外項及 runtime。不能當作載入成功保證。</p>',
 'NVIDIA DGX B200 8 張 GPU 合計 1440 GB，對應每張 180 GB。不同 Blackwell 產品及標示方式不要混淆。這裡不使用峰值 FLOPS 推算實際 RPS，也不把 1 byte/parameter 當作每種 FP8 checkpoint 的精確檔案大小。',['b200','q35','q38'])
slide('GPU 資源與可觀察的症狀','指標提供線索，profiler 與 queue / cache telemetry 才能定位',
 table(['壓力來源','可能觀察到','下一步查證'],[
 ['權重讀取 / 小 batch','低 concurrency 的 TPOT 偏高','memory traffic、GEMM shape、launch gaps'],
 ['長 context / cache','HBM 增加、attention 時間增加','KV 用量、context 分布、attention profile'],
 ['Prefill 干擾 / 排隊','TTFT 或 ITL tail 上升','waiting requests、prefill chunks、GPU trace'],
 ['MoE 分散式通訊','dispatch / combine 時間增加','expert load、collective timing、拓樸'],
 ['Client / network / parser','client latency 與 server 指標分歧','送出時間、串流 chunk、CPU / network']])+callout('ITL 先超標表示輸出節奏先失守；單靠 ITL 無法斷言 GPU decode 是唯一瓶頸。'),
 '這也修正分享討論中的過強解讀：ITL mean first 是症狀，不等於已確認 decode capacity 是根因。新 prefill、kernel contention、network buffering、推測解碼的成批輸出都可能影響 client ITL。TP、EP、DP 應依硬體拓樸與瓶頸比較。',['anatomy','gdncode'])
slide('多模態與 MTP 的位置','文字主線之外的額外工作',
 flow(['Image / video','Vision encoder','視覺 embeddings','同一文字 decoder'])+
 cols('<h3>多模態</h3>'+ul('前處理、encoder 與視覺 tokens 增加成本。','文字 benchmark 無法直接代表影像／影片容量。'),'<h3>MTP / speculative decoding</h3>'+ul('候選 token 提案後，由 target model 驗證。','接受率、驗證成本與 state rollback 決定收益。')),
 '本報告模型具有 vision encoder，但主線用純文字。MTP heads 經過訓練，實際是否使用由 serving 設定、模型及 backend 支援決定。後面 metrics 範例假設單 token 串流，真實服務需注意 speculative chunks。',['q35','q38'])
slide('Latency 的時間邊界','Client 看到的 token 時間，包含服務外部成本',
 demo('timeline'),
 '設送出 t0=0，token 到達 t1=.8,t2=.83,t3=.88,t4=.91,t5=1.0 秒，完成 t_done=1.02。TTFT=.8，最後 token latency=1.0，E2EL 若定義為完成則1.02，TPOT 依 benchmark end-to-end duration 計算為55ms，而 token intervals 平均50ms。刻意保留20ms收尾來說明 endpoint 定義差異。',['metrics'],demo='timeline')
slide('ITL、TPOT 與 percentile','先決定統計母體，再討論 p95 / p99',
 table(['指標','每筆 request / interval','彙總方式'],[
 ['TTFT','t_first − t_send','每筆 request 一個樣本'],
 ['ITL','t_i − t_(i−1)','各 token／stream chunk 間隔；確認 backend'],
 ['TPOT','(E2EL − TTFT) / (N_out − 1)','每筆 request 的平均，N_out > 1'],
 ['E2EL','t_done − t_send','包含完成語意的 client latency']])+callout('所有 intervals 的平均，與「各 request 平均再平均」通常不同。p95 ITL 也不等於 p95 TPOT。'),
 '一個 SSE chunk 不保證只有一個 token，空 role delta、reasoning 及 parser buffering 都可能改變 first-token 的觀察方式。TPOT N=1 分母為0，必須說明工具排除或特殊處理方式。對 variable OSL，mean(TTFT)+(mean N−1)×mean(TPOT) 也不一定等於 mean E2EL，因 N 與 TPOT 可能相關。',['metrics'])
slide('Stage 1：Closed-loop saturation test','控制 concurrency，量測 achieved throughput',
 demo('closed'),
 '固定 C 個槽位，完成後立即補新 request，無 think time。這是 client outstanding 上限，不等於 GPU 每輪真正 batch size。CLI：request-rate inf，max-concurrency C。warmup 與 drain 期間達不到滿 C，應另觀察穩態。RPS 是成功完成數除以相同量測期間。',['discussion','bench'],demo='closed')
slide('Stage 2：Open-loop SLO capacity test','控制到達率，觀察 queue 與 latency 是否穩定',
 demo('open'),
 '示意採固定處理時間的單服務槽位來突出到達率與完成率差異，不是 vLLM 的 GPU 模擬器。真實 vLLM 可批次執行多筆。Finite request-rate 配合 arrival process，通常不設會觸發的 client concurrency cap；若觸發必須報告 client delay 與實際送出率。',['discussion','bench'],demo='open')
slide('Offered load、throughput 與 goodput','三種速率回答不同問題',
 table(['同一觀測窗 Δt = 100 s','數量','速率'],[
 ['實際送出 requests','150','1.50 req/s'],
 ['成功完成 requests','140','1.40 req/s'],
 ['完成且符合每筆 SLO 的 requests','110','1.10 good req/s']])+formula('Goodput = 符合所選每筆 SLO 的成功 requests / Δt')+
 callout('測試窗內尚未完成的 requests、失敗與 timeout，必須保留在報告中。'),
 '以上為同一觀測窗示例，未假設剩餘10筆全部失敗。有限批次 benchmark 可能以發送開始到最後完成作為總 duration，包含 drain，不能把它與 arrival window 的 offered RPS 當成同分母。Goodput 的 per-request 門檻也不等於下一頁 run-level aggregate SLO All Pass。',['metrics','discussion'])
slide('SLO 邊界搜尋','粗掃找範圍，再細掃並重複確認',
 demo('slo'),
 '採用分享討論的 SLO：TTFT mean≤600ms、p95≤900ms，ITL mean≤150ms、p95≤300ms。曲線全部為示意。某次全綠只代表該次樣本達標；在邊界重跑並觀察隊列趨勢及 timeout。二分搜尋是近似策略，batching、cache 與隨機性可能使結果不完全單調。',['discussion'],demo='slo')
slide('由 RPS 換算服務人數','在穩定系統與一致邊界下使用平均值',
 demo('users'),
 'Little’s Law C=X R 適用穩定系統，量測窗及 arrival/departure 範圍須一致。Closed user population、每人最多一筆 outstanding、無平行 tool fan-out 時，U=X(R+Z)。若40秒已是送出到送出的週期，U=X×40，不要再加R。全體用戶人口=active users/peak active fraction。不可把goodput率搭配所有requests的平均延遲直接套Little’s Law。',['discussion'],demo='users')
slide('負載測試報告的完整結論','Workload 與 SLO 定義決定容量數字的意義',
 '<div class="statement">在指定模型版本、GPU 拓樸、精度與 workload 下，<br>透過 concurrency sweep 找到效率曲線，<br>再以 arrival-rate sweep 找到穩定的 SLO 邊界。</div>'+ul('模型架構決定每步運算，以及要保存的歷史狀態。','vLLM 排程與 cache 管理決定如何將多筆工作放進 GPU。','Production 容量再納入真實流量、burst、headroom 與故障情境。'),
 '主簡報結束。最後的交付結論應包含：model revision、vLLM commit/container digest、GPU及互連、TP/EP/DP、權重/KV/state精度、ISL/OSL分布、prefix hit、thinking/MTP設定、樣本數與時長、失敗率、SLO百分位、邊界與operating point。附錄提供實際命令與算式。')

slide('附錄 A：測試設定與量測窗','可重現性比單一最高 tok/s 更有用',
 table(['分類','必須記錄'],[
 ['模型與引擎','model revision、vLLM version / commit、container digest、backend'],
 ['硬體與並行','GPU 型號 / 數量、TP / EP / DP / PP、互連、功率設定'],
 ['精度與最佳化','weight / KV / state dtype、prefix cache、chunked prefill、MTP'],
 ['Workload','ISL / OSL 分布、prompt prefix、thinking、stop / EOS、seed'],
 ['測試與統計','warmup、arrival window、drain、樣本數、重跑次數、errors'],
 ['Telemetry','scheduled / sent / completed、queue、cache、CPU/GPU traces']]),
 '固定 num_prompts 可以比較樣本，但高 concurrency 需足夠輪次。固定時間方便觀察 open-loop queue buildup，低 RPS 時延長測試以增加 tail 樣本。最少數百筆不代表足以穩健估計 p99，應報告樣本數與重複波動。')
slide('附錄 B：vLLM bench 命令範例','先核對安裝版本的 vllm bench serve --help',
 '<pre>vllm bench serve \\\n  --backend openai-chat --base-url http://HOST:8000 \\\n  --endpoint /v1/chat/completions --model MODEL \\\n  --dataset-name random --random-input-len 2048 \\\n  --random-output-len 512 --num-prompts 1000 \\\n  --percentile-metrics ttft,tpot,itl,e2el \\\n  --metric-percentiles 50,95,99 --save-result</pre>'+
 cols('<h3>Stage 1 追加</h3><pre>--request-rate inf\n--max-concurrency 32</pre>','<h3>Stage 2 追加</h3><pre>--request-rate 1.0\n--burstiness 1.0</pre>'),
 '以上是 Linux shell 範例，HOST/MODEL 必須替換，非本機實際執行結果。將 Stage1 的32改成 sweep 各值；Stage2 改 finite RPS。random prompt 不等於 production chat/template/token 分布，實際 token 數以回傳和tokenizer校對。EOS、thinking和max_tokens都會影響OSL。goodput支援哪些門檻依版本確認；這裡未把ITL aggregate SLO硬塞進goodput旗標。',['bench'])
slide('附錄 C：記憶體配置試算','單卡教學預算，不含 allocator、分片與 kernel 的實際限制',
 demo('budget'),
 '單張 B200 180 GB，示例使用率上限90%=162GB。扣除粗估權重與手動runtime預留，再除以每sequence的cache邏輯容量，只是記憶體上界。FP8權重不代表KV也FP8，本頁固定BF16 KV / FP32 state。BF16 122B單卡不能容納。這個計算不能取代vLLM startup profile，尤其MoE量化格式、vision/MTP與memory planner影響實際值。',['b200','c35','c38','state'],demo='budget')
slide('附錄 D：GDN 單一 head 的更新式','令 S 為 dᵥ × dₖ 矩陣，省略 norm、conv 與 output gate',
 formula('S̄<sub>t</sub> = α<sub>t</sub> S<sub>t−1</sub>')+formula('r<sub>t</sub> = v<sub>t</sub> − S̄<sub>t</sub> k<sub>t</sub>')+formula('S<sub>t</sub> = S̄<sub>t</sub> + β<sub>t</sub> r<sub>t</sub> k<sub>t</sub><sup>T</sup>　　o<sub>t</sub> = S<sub>t</sub> q<sub>t</sub>')+
 '<p>α 控制衰減，β 控制修正幅度。這是 request state 的 forward update。</p>',
 '等價展開為 α S_prev (I−β k k^T)+β v k^T。不同程式可能採用轉置layout，不影響元素總量。Qwen實際包含short convolution、q/k normalization、門控RMSNorm與output projection。本公式是用於說明記憶體更新的核心，非完整layer實作。',['gdn','gdncode'])
slide('附錄 E：單 GPU 與多 GPU','TP、EP 與 DP 改變不同部分',
 table(['策略','主要分配的內容','成本與限制'],[
 ['Tensor parallelism','同層矩陣與 attention heads','collectives、head 可分性、部分 KV 可能複製'],
 ['Expert parallelism','不同 experts','token dispatch / combine、routing 不均'],
 ['Data parallelism','完整模型 replicas','分流與 cache locality，每 replica 需權重'],
 ['Pipeline parallelism','不同 layers','pipeline bubbles、跨 stage activation 傳輸']])+callout('總 HBM 足夠只是起點；吞吐、尾延遲與跨卡通訊需要一起測。'),
 '這裡提供概念，不指定某個尚未量測的最佳TP值。兩張180GB在容量上可以容納約244GB BF16權重，但還有cache/runtime與實際支援條件；不能只用總量宣稱某部署必定成功。',['arch'])
slide('附錄 F：來源與資料界線','查核日期 2026-09-08',
 '<div class="sourcegrid">'+''.join(f'<a href="{u}" target="_blank" rel="noopener">{n}</a>' for n,u in SOURCES.values())+'</div><p class="caption">官方 config / 模型卡用於架構參數；容量由公式推導。吞吐曲線與動畫使用教學示例。圖片由使用者提供。</p>',
 '來源連結需要網路，簡報本身與動畫可離線。main/latest頁面會更新，請在正式實測報告另外固定model revision與engine commit。使用者分享討論是需求素材而非權威技術證明，其中公式的限制與過強診斷已在主簡報補正。')

template=(ROOT/'template.html').read_text(encoding='utf-8')
for token,name,mime in [('@@BATCH@@','continuous_batching.png','image/png'),('@@PAGED@@','paged_attention.gif','image/gif')]:
    uri='data:'+mime+';base64,'+base64.b64encode((ROOT/'assets'/name).read_bytes()).decode()
    for s in slides: s['body']=s['body'].replace(token,uri)
data=json.dumps(slides,ensure_ascii=False).replace('</','<\\/')
template=template.replace('@@SLIDES@@',data).replace('@@SOURCES@@',json.dumps(SOURCES,ensure_ascii=False))
(ROOT/'LLM推論報告.html').write_text(template,encoding='utf-8')
(ROOT/'research'/'slides.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf-8')
notes=['# LLM 推論與服務容量\n\n講者筆記，資料查核日期 2026-09-08。\n']
for i,s in enumerate(slides,1):
    notes.append(f'## {i:02d}. {s["title"]}\n\n{s["sub"]}\n\n{s["notes"]}\n')
    for src in s['sources']:
        n,u=SOURCES[src];notes.append(f'- [{n}]({u})')
    notes.append('')
(ROOT/'講者筆記.md').write_text('\n'.join(notes),encoding='utf-8')
print(f'Built {len(slides)} slides')
