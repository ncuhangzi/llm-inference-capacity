# -*- coding: utf-8 -*-
"""來源清單、名詞表、投影片組裝工具。"""
from __future__ import annotations
import model as M

CHECK = "2026-09-08"

# ------------------------------------------------------------------ 來源 ---
SOURCES = {
    # 官方模型卡與 config
    "c35": ("Qwen3.5-122B-A10B config.json", "https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json"),
    "m35": ("Qwen3.5-122B-A10B 模型卡", "https://huggingface.co/Qwen/Qwen3.5-122B-A10B"),
    "nv4": ("nvidia/Qwen3.5-122B-A10B-NVFP4 模型卡", "https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4"),
    "nvq": ("NVFP4 hf_quant_config.json", "https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4/blob/main/hf_quant_config.json"),
    "nvi": ("NVFP4 safetensors index", "https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4/blob/main/model.safetensors.index.json"),
    "c38": ("Qwen3.8-27B config.json", "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json"),
    "m38": ("Qwen3.8-27B 模型卡", "https://huggingface.co/Qwen/Qwen3.8-27B"),
    "f38": ("Qwen3.8-27B-FP8 模型卡", "https://huggingface.co/Qwen/Qwen3.8-27B-FP8"),
    "i38": ("Qwen3.8-27B-FP8 safetensors index", "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/main/model.safetensors.index.json"),
    "c332": ("Qwen3-32B config.json", "https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json"),
    "q3r": ("Qwen3 Technical Report (arXiv 2505.09388)", "https://arxiv.org/abs/2505.09388"),
    # speculative decoding
    "df2": ("incoai/Qwen3.8-27B-DFlash2 模型卡", "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2"),
    "dfp": ("DFlash: Block Diffusion for Flash Speculative Decoding", "https://arxiv.org/abs/2602.06036"),
    "dfn": ("NVIDIA：Boost Inference up to 15x with DFlash", "https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/"),
    "dfs": ("vLLM Speculators：DFlash 演算法文件", "https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/"),
    "eg3": ("EAGLE-3 (arXiv 2503.01840)", "https://arxiv.org/abs/2503.01840"),
    "spd": ("vLLM Speculative Decoding 文件", "https://docs.vllm.ai/en/latest/features/speculative_decoding/"),
    "sspec": ("SmartSpec：以 goodput 最佳化 speculative decoding", "https://arxiv.org/abs/2406.14066"),
    "lat": ("An Interpretable Latency Model for Speculative Decoding", "https://arxiv.org/abs/2605.15051"),
    "iss1": ("vLLM #47602：MTP 接受率隨 context 衰減", "https://github.com/vllm-project/vllm/issues/47602"),
    "iss2": ("vLLM #38182：MTP 降低 prefix cache 命中率", "https://github.com/vllm-project/vllm/issues/38182"),
    # vLLM 內部
    "arch": ("vLLM Architecture Overview", "https://docs.vllm.ai/en/latest/design/arch_overview/"),
    "anat": ("Inside vLLM: Anatomy of a High-Throughput LLM Inference System", "https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html"),
    "paged": ("vLLM PagedAttention 發表文", "https://blog.vllm.ai/2023/06/20/vllm.html"),
    "hyb": ("vLLM Hybrid KV Cache Manager 設計文件", "https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/"),
    "mut": ("vLLM mamba_utils.py（state shape 計算）", "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py"),
    "gdnc": ("vLLM Qwen GDN linear attention 實作", "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
    "bench": ("vllm bench serve CLI 文件", "https://docs.vllm.ai/en/latest/cli/bench/serve/"),
    "gp": ("vLLM goodput metric（PR #9338）", "https://github.com/vllm-project/vllm/pull/9338"),
    # 演算法論文
    "gdn": ("Gated Delta Networks (arXiv 2412.06464)", "https://arxiv.org/abs/2412.06464"),
    # 量化與硬體
    "nvfp4": ("NVFP4 格式說明（Modal LLM Almanac）", "https://modal.com/llm-almanac/block-quants"),
    "nvfp4b": ("NVFP4：Blackwell microscaling 4-bit float", "https://zeroentropy.dev/concepts/nvfp4/"),
    "b200": ("NVIDIA Blackwell / HGX B200 datasheet", "https://www.nvidia.com/en-us/data-center/hgx/"),
    "b200l": ("Lenovo ThinkSystem HGX B200 180GB 產品指南（規格表）", "https://lenovopress.lenovo.com/lp2226.pdf"),
    # 使用者提供
    "own": ("使用者提供：本次評估的模型版本與 spec 設定", "#"),
}

# ---------------------------------------------------------------- 名詞表 ---
GLOSSARY = {
    "Activated parameters": "MoE 模型每產生一個 token 真正參與運算的參數量。Qwen3.5-122B-A10B 的 122B 是總量，A10B 指每個 token 只用約 10B。算力照 10B 算，記憶體要放滿 122B。",
    "Arithmetic intensity": "每從記憶體讀 1 byte 能做幾次浮點運算。prefill 很高（吃算力），decode 很低（吃頻寬）。報告裡談的最佳化，多半都是在拉高這個數字。",
    "Autoregressive": "一次只產生一個 token，再把它接回輸入產生下一個。生成 T 個 token 就要跑 T 次 forward，decode 慢的原因就在這裡。",
    "Acceptance length (τ)": "speculative decoding 每一輪平均被接受的 token 數（含最後那個「保底」token）。τ 越大越省。k 個草稿的上限是 k+1。",
    "BF16": "bfloat16，16 位元浮點。與 FP16 同寬但指數位更多，訓練與推論的預設精度。1 個參數佔 2 bytes。",
    "Block diffusion": "DFlash2 用的起草方式：把未來一整段位置全部遮起來，用一次 forward 同時「去噪」猜出整塊 token，而不是一個一個猜。",
    "Chunked prefill": "把很長的 prompt 切成小塊，分散到多個排程輪次，避免一個長 prompt 讓其他人的 decode 卡住。",
    "Continuous batching": "每一輪排程都重新組合正在跑的 requests：做完的離開、等待的立刻補進來，不用等整個 batch 一起結束。",
    "CUDA graph": "把一串固定的 GPU kernel 呼叫錄下來重播，省掉每次啟動 kernel 的 CPU 開銷。decode 步驟很短，這個開銷佔比很高。",
    "Decode": "生成階段。每一輪每條序列只餵 1 個新 token，讀取整份權重與全部歷史 KV，屬於 memory-bound。",
    "DFlash2": "z-lab / IncoAI 的 block-diffusion 起草模型。用非因果 attention 同時看目標模型的 hidden state 與 mask token，一次 forward 產生整塊草稿。本報告 27B 使用 block size 8、7 個草稿 token。",
    "EAGLE / EAGLE-3": "以目標模型的 hidden state 為條件的輕量起草器。EAGLE-3 融合目標模型低／中／高層特徵，並在訓練時模擬多步生成。Qwen 的 MTP head 屬於同一系。",
    "E2EL": "End-to-end latency，從送出 request 到收到完整回覆的總時間。",
    "Expert parallelism (EP)": "把 MoE 的專家分散到不同 GPU。省記憶體，但每一層都要做 all-to-all 把 token 送到專家所在的卡。",
    "FP8 (E4M3)": "8 位元浮點，4 位指數 3 位小數。1 個參數 1 byte。Qwen3.8-27B-FP8 用 128 為單位的 blockwise 縮放。",
    "Full attention": "標準的 softmax attention，需要保存整段歷史的 K/V。Qwen 混合架構每 4 層放 1 層。",
    "Gated Attention": "Qwen3.5 系列的 attention 變體，輸出乘上一個學出來的 gate（config 的 attn_output_gate=true），所以 q_proj 的輸出寬度是兩倍。",
    "Gated DeltaNet (GDN)": "一種 linear attention。用固定大小的矩陣狀態 S 取代 KV cache，靠 decay gate α 與 delta rule β 更新。記憶體不隨 context 成長。",
    "Goodput": "只計算「滿足 SLO」的產出速率。相對於 throughput 把所有輸出都算進去，goodput 才反映使用者真正拿到的服務。",
    "GQA": "Grouped-Query Attention。多個 Q head 共用一組 K/V head，直接把 KV cache 縮小成 1/(head 比例)。",
    "HBM": "GPU 上的高頻寬記憶體。B200 單卡 180 GB、約 7.7 TB/s。權重、KV cache、SSM state、activation 全部共用這塊。",
    "ITL": "Inter-token latency，串流時相鄰兩次輸出之間的間隔。使用者感受到的「順不順」。",
    "KV cache": "把每個位置算過的 K/V 存起來，避免每輪重算。大小 = 2 × full-attention 層數 × KV head 數 × head_dim × 位元組數 × token 數，隨 context 線性成長。",
    "Little's Law": "L = λ × W。穩定系統裡，平均同時在系統內的 request 數 = 到達率 × 平均停留時間。把 RPS 換算成「同時服務幾個人」的依據。",
    "MoE": "Mixture of Experts。把 FFN 換成很多個小 expert，每個 token 只選幾個來算。省算力，不省記憶體。",
    "MTP": "Multi-Token Prediction。模型自帶的額外一層 head，訓練時就學著預測後面幾個 token，可直接當 speculative decoding 的起草器。Qwen3.5/3.8 的 checkpoint 內含 mtp.* 權重。",
    "NVFP4": "NVIDIA Blackwell 的 4 位元浮點格式。每 16 個 E2M1 值共用一個 FP8-E4M3 的 block scale，再乘上一個 per-tensor 的 FP32 scale，平均 4.5 bit／值。",
    "PagedAttention": "把 KV cache 切成固定大小的 block，用 block table 把邏輯位置對應到不連續的實體 block，像作業系統的分頁一樣，消除記憶體碎片。",
    "Percentile (p95/p99)": "延遲分布的第 95／99 百分位。平均值會被少數快的請求拉低，SLO 必須看百分位。",
    "Prefill": "把整段 prompt 一次算完並填好 KV cache 的階段。token 多、算術強度高，屬於 compute-bound。",
    "Prefix caching": "不同 request 共用的開頭（system prompt、few-shot）只算一次，KV block 直接重用。",
    "Rejection sampling": "speculative decoding 的驗證方式：逐位置決定接受或拒絕草稿，數學上讓最終輸出分布與不猜時完全一致。",
    "RoPE": "Rotary Position Embedding。把位置資訊以旋轉的方式寫進 Q/K。Qwen3.5 只旋轉 head_dim 的 1/4（partial_rotary_factor 0.25）。",
    "Roofline": "把「頻寬上限」與「算力上限」畫在一起的模型。兩條線交會處就是轉折點，決定你在 memory-bound 還是 compute-bound。",
    "SLO": "Service Level Objective。事先講定的延遲目標，例如「p95 TTFT < 800 ms 且 p95 TPOT < 40 ms」。沒講定 SLO，容量數字就沒有意義。",
    "Speculative decoding": "先用便宜的方式猜 k 個 token，再讓目標模型一次驗證。猜對就一次前進多格，猜錯就丟掉。輸出分布不會變。",
    "SSM cache / recurrent state": "GDN 等 linear attention 層要保存的狀態：一個 dᵥ×dₖ 的矩陣（FP32）加一小段 convolution 歷史。大小固定，與 context 長度無關。",
    "Tensor parallelism (TP)": "把每一層的矩陣切開放在多張卡上，每層都要做一次 all-reduce。降低單卡記憶體與延遲，代價是通訊。",
    "TPOT": "Time Per Output Token，第一個 token 之後的平均每 token 時間。等於「使用者看到的打字速度」的倒數。",
    "TTFT": "Time To First Token，送出 request 到收到第一個 token 的時間。包含排隊與 prefill。",
    "vLLM": "本報告使用的推論伺服器。負責排程、KV cache 分頁管理、continuous batching 與 kernel 呼叫。",
}


def term(name, text=None):
    """產生帶 tooltip 的名詞。"""
    return f'<b class="t" data-term="{name}">{text or name}</b>'


T = term

# ---------------------------------------------------------------- 章節 -----
CHAPTERS = []
SLIDES = []


def chapter(name, accent="compute"):
    CHAPTERS.append({"name": name, "accent": accent, "at": len(SLIDES)})


def slide(title, sub="", body="", notes="", sources=(), kind="", accent=None, lt=""):
    ch = CHAPTERS[-1] if CHAPTERS else {"name": "", "accent": "compute"}
    SLIDES.append({
        "title": title, "lt": lt, "sub": sub, "body": body, "notes": notes,
        "sources": list(sources), "kind": kind, "accent": accent or ch["accent"],
        "ch": ch["name"],
    })


# ------------------------------------------------------------- 版面工具 ----
def fig(name, svg_markup, steps, caps, accent=None, bare=False, style="", grow=True):
    """把 SVG 包成可逐步播放的圖。caps 為每一步的說明文字。"""
    cap = "".join(f'<p data-s="{i}">{c}</p>' for i, c in enumerate(caps))
    cls = "fig bare" if bare else "fig"
    css = (f"--accent:var(--{accent});" if accent else "") + ("" if grow else "flex:none;") + style
    st = f' style="{css}"' if css else ""
    return (f'<div class="{cls}" data-fig="{name}" data-steps="{steps}"{st}>'
            f'{svg_markup}<div class="figbar"></div><div class="figcap">{cap}</div></div>')


def gist(text, accent=None):
    st = f' style="--accent:var(--{accent})"' if accent else ""
    return f'<div class="gist"{st}><span>{text}</span></div>'


def take(text):
    return f'<div class="takeaway">{text}</div>'


def warn(text):
    return f'<div class="warn">{text}</div>'


def quote(text, cite):
    return f'<div class="quote"><q>{text}</q><cite>{cite}</cite></div>'


def pane(title, body, accent=None):
    cls = "pane acc" if accent else "pane"
    st = f' style="--accent:var(--{accent})"' if accent else ""
    h = f"<h3>{title}</h3>" if title else ""
    return f'<div class="{cls}"{st}>{h}{body}</div>'


def stats(items):
    """items: [(value, unit, key, cls)]"""
    cells = "".join(
        f'<div class="stat {c}"><div class="v">{v}{f"<small>{u}</small>" if u else ""}</div>'
        f'<div class="k">{k}</div></div>' for v, u, k, c in items)
    return f'<div class="stats">{cells}</div>'


def table(head, rows, cls="", hi=()):
    th = "".join(f'<th class="{ "n" if h.startswith("~") else "" }">{h.lstrip("~")}</th>'
                 for h in head)
    body = []
    for i, r in enumerate(rows):
        tds = "".join(f'<td class="{ "n" if h.startswith("~") else "" }">{c}</td>'
                      for h, c in zip(head, r))
        body.append(f'<tr class="{"hi" if i in hi else ""}">{tds}</tr>')
    return (f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def bars(rows, total=None, unit=""):
    """rows: [(label, [(value,color)], display)]"""
    tot = total or max(sum(v for v, _ in segs) for _, segs, _ in rows)
    out = []
    for lab, segs, disp in rows:
        seg = "".join(f'<i style="width:{v/tot*100:.2f}%;background:var(--{c})"></i>'
                      for v, c in segs)
        out.append(f'<div class="bar"><span class="t">{lab}</span>'
                   f'<span class="track">{seg}</span><span class="v">{disp}</span></div>')
    return f'<div class="bars">{"".join(out)}</div>'


def legend(items):
    return ('<div class="legend">' +
            "".join(f'<span><i style="background:var(--{c})"></i>{t}</span>' for t, c in items) +
            "</div>")


def code(text, lang=""):
    return f"<pre>{text}</pre>"


def num(v, d=0):
    return f"{v:,.{d}f}"


def gib(b, d=2):
    return f"{b / M.GiB:,.{d}f}"


def gb(b, d=2):
    return f"{b / 10**9:,.{d}f}"


def mib(b, d=1):
    return f"{b / M.MiB:,.{d}f}"


def kib(b, d=0):
    return f"{b / M.KiB:,.{d}f}"


# ------------------------------------------------- 匯出給前端的模型常數 ----
def model_js():
    """把解析模型的常數輸出成 JS，讓互動試算與投影片數字同源。"""
    out = {}
    for k, m in M.MODELS.items():
        out[k] = {
            "label": m.label, "short": m.short, "repo": m.repo, "quant": m.quant,
            "kvDtype": m.kv_dtype, "hidden": m.hidden, "L": m.n_layers,
            "nFull": m.n_full, "nGdn": m.n_gdn, "nKv": m.n_kv, "headDim": m.head_dim,
            "nv": m.nv, "dv": m.dv, "dk": m.dk, "convK": m.conv_k, "convDim": m.conv_dim,
            "nExperts": m.n_experts, "topK": m.top_k,
            "expertBytes": m.expert_bytes_each, "denseRead": m.dense_read_bytes,
            "tokenCompute": m.token_compute_s, "weightBytes": m.real_weight_bytes,
            "bf16Bytes": m.real_bf16_bytes, "maxCtx": m.max_ctx,
            "specK": m.spec_k, "specLabel": m.spec_label,
            "draftBytes": (m.spec_draft_params * 2) if m.spec_k else 0,
            "draftSerial": bool(m.spec_draft_serial),
            "tau": M.SPEC_TAU.get(k, 1.0), "alpha": M.SPEC_ALPHA.get(k, 0.0),
            "paramsTotal": m.params_total, "activated": m.activated,
            "expertParams": m.expert_params,
            "mtpDenseParams": (m.attn_per_layer + 3 * m.hidden * m.shared_inter
                               + m.n_experts * m.hidden + 2 * m.hidden * m.hidden
                               + m.vocab * m.hidden) if m.key == "q35" else 0,
        }
    hw = {"hbm": M.GPU.hbm_bytes, "bw": M.GPU.bw, "flops": M.GPU.flops,
          "knee": M.GPU.knee_tokens("fp8")}
    import json as _j
    return ("const MODELS=" + _j.dumps(out, ensure_ascii=False) + ";\n"
            "const HW=" + _j.dumps(hw, ensure_ascii=False) + ";\n")
