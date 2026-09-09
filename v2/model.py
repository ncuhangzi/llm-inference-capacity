# -*- coding: utf-8 -*-
"""LLM 推論容量的解析模型 (analytic model).

所有數字都由官方 config / safetensors 位元組數推導，並與 HuggingFace 上實際
checkpoint 的 dtype 統計交叉驗證。查核日期 2026-09-08。

單位約定：
  * GB  = 10^9 bytes（磁碟與廠商規格慣用）
  * GiB = 2^30 bytes（GPU 記憶體配置慣用）
"""
from __future__ import annotations
import math

GiB = 1024 ** 3
MiB = 1024 ** 2
KiB = 1024

CHECK_DATE = "2026-09-08"

# ---------------------------------------------------------------- hardware ---
class GPU:
    """NVIDIA HGX B200 單顆 GPU 規格（官方 datasheet，dense 值）。"""
    name = "NVIDIA B200 (HGX, 1 GPU)"
    hbm_bytes = 180 * 10**9          # 180 GB HBM3e
    bw = 7.7 * 10**12                # 7.7 TB/s
    flops = {                        # dense tensor-core FLOP/s
        "bf16": 2.25e15,
        "fp8": 4.5e15,
        "nvfp4": 9.0e15,
    }
    # 每個參數的位元組數（含 block scale 攤提）
    bytes_per_param = {"bf16": 2.0, "fp8": 1.0, "nvfp4": 0.5 + 1 / 16}

    @classmethod
    def knee_tokens(cls, prec="fp8"):
        """roofline 轉折點：一個 forward step 需要幾個 token，計算時間才追上讀權重時間。

        N* = F * b / (2 * BW)   —— Blackwell 每降一半位寬就把算力加倍，
        因此 bf16 / fp8 / nvfp4 得到同一個 N*。
        """
        b = 2.0 if prec == "bf16" else (1.0 if prec == "fp8" else 0.5)
        return cls.flops[prec] * b / (2 * cls.bw)


# ------------------------------------------------------------- primitives ---
def gdn_layer(h, nk, nv, dk, dv, conv_k):
    """Gated DeltaNet(linear_attn)一層的參數，鍵名對應 checkpoint 實際 tensor 名。"""
    kdim, vdim = nk * dk, nv * dv
    conv_dim = 2 * kdim + vdim          # vLLM: head_k*nk*2 + head_v*nv
    return {
        "in_proj_qkv": conv_dim * h,
        "in_proj_z": vdim * h,
        "in_proj_a": nv * h,
        "in_proj_b": nv * h,
        "out_proj": h * vdim,
        "conv1d": conv_dim * conv_k,
        "A_log": nv,
        "dt_bias": nv,
        "norm": dv,
    }, conv_dim


def attn_layer(h, nq, nkv, hd, out_gate=True):
    """Gated Attention 一層。attn_output_gate=True 時 q_proj 輸出兩倍寬（含 gate）。"""
    return {
        "q_proj": nq * hd * (2 if out_gate else 1) * h,
        "k_proj": nkv * hd * h,
        "v_proj": nkv * hd * h,
        "o_proj": h * nq * hd,
        "q_norm": hd,
        "k_norm": hd,
    }


def vision_tower(out_h, depth=27, h=1152, inter=4304, patch=16, tp=2, ch=3,
                 npos=2304, merge=2):
    per = (3 * h * h + 3 * h) + (h * h + h) + (h * inter + inter) + (inter * h + h) + 4 * h
    return {
        "patch_embed": ch * tp * patch * patch * h + h,
        "pos_embed": npos * h,
        "blocks": depth * per,
        "merger": (h * merge * merge) * out_h + out_h + out_h * out_h + out_h + 2 * (h * merge * merge),
    }


# ----------------------------------------------------------------- models ---
class Model:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    # --- 快取 -------------------------------------------------------------
    def kv_bytes_per_token(self, kv_dtype="fp8"):
        """KV cache：只有 full-attention 層需要，隨 token 數線性成長。

        M_KV/token = 2 (K,V) * L_full * n_kv * d_head * bytes
        """
        b = {"bf16": 2, "fp16": 2, "fp8": 1}[kv_dtype]
        return 2 * self.n_full * self.n_kv * self.head_dim * b

    def ssm_recurrent_bytes(self):
        """GDN recurrent state：固定大小，與 context 長度無關。

        M_state = L_gdn * n_v * d_v * d_k * 4 (FP32, mamba_ssm_dtype)
        """
        return self.n_gdn * self.nv * self.dv * self.dk * 4

    def ssm_conv_bytes(self, num_spec=0):
        """GDN short-conv state：長度 = conv_kernel - 1 + num_spec（vLLM mamba_utils）。"""
        return self.n_gdn * self.conv_dim * (self.conv_k - 1 + num_spec) * 2

    def ssm_bytes(self, num_spec=0):
        return self.ssm_recurrent_bytes() + self.ssm_conv_bytes(num_spec)

    def seq_bytes(self, tokens, kv_dtype="fp8", num_spec=0):
        return tokens * self.kv_bytes_per_token(kv_dtype) + self.ssm_bytes(num_spec)

    def crossover_tokens(self, kv_dtype="fp8", num_spec=0):
        """context 要多長，KV cache 才追上固定的 SSM state。"""
        k = self.kv_bytes_per_token(kv_dtype)
        return self.ssm_bytes(num_spec) / k if k else float("inf")

    # --- 權重流量 ---------------------------------------------------------
    def experts_touched(self, n_tokens):
        if not self.n_experts:
            return 0
        p = 1 - self.top_k / self.n_experts
        return self.n_experts * (1 - p ** n_tokens)

    def step_bytes(self, n_tokens):
        """一個 forward step 需要從 HBM 讀多少權重位元組（roofline 分母）。"""
        b = self.dense_read_bytes
        if self.n_experts:
            b += self.experts_touched(n_tokens) * self.n_layers * self.expert_bytes_each
        return b

    def step_compute_s(self, n_tokens):
        return n_tokens * self.token_compute_s

    def step_time_s(self, n_tokens):
        return max(self.step_bytes(n_tokens) / GPU.bw, self.step_compute_s(n_tokens))

    def knee_tokens(self, hi=1 << 16):
        """實際轉折點：考慮 MoE 專家讀取量隨 batch 成長。"""
        lo = 1.0
        if self.step_compute_s(hi) < self.step_bytes(hi) / GPU.bw:
            return float(hi)
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.step_compute_s(mid) < self.step_bytes(mid) / GPU.bw:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def build_q35():
    h, L, nfull, ngdn, V = 3072, 48, 12, 36, 248320
    nk, nv, dk, dv, ck = 16, 64, 128, 128, 4
    E, me, se = 256, 1024, 1024
    g, conv_dim = gdn_layer(h, nk, nv, dk, dv, ck)
    a = attn_layer(h, 32, 2, 256)
    gdn_p, attn_p = sum(g.values()), sum(a.values())
    expert_p = 3 * h * me                       # gate+up+down per expert
    shared_p = 3 * h * se
    gate_p = E * h
    moe_p = E * expert_p + shared_p + gate_p + h
    embed = V * h
    lang = ngdn * gdn_p + nfull * attn_p + L * moe_p + L * 2 * h + 2 * embed + h
    mtp = attn_p + moe_p + 2 * h + h * 2 * h + 3 * h
    vis = sum(vision_tower(h).values())

    # NVFP4 只量化 routed experts（由 weight map 驗證：只有 experts 有 weight_scale）
    bpp = GPU.bytes_per_param["nvfp4"]
    expert_bytes_each = expert_p * bpp
    quant_params = L * E * expert_p
    bf16_params = lang + mtp + vis - quant_params
    weight_bytes = quant_params * bpp + bf16_params * 2
    # 每 step 必讀（非 routed experts）；embed_tokens 只查表故不計，lm_head 全讀
    dense_read = (ngdn * gdn_p + nfull * attn_p + L * (shared_p + gate_p + 2 * h)
                  + embed) * 2
    tok_c = (2 * (ngdn * gdn_p + nfull * attn_p + L * (shared_p + gate_p) + embed) / GPU.flops["bf16"]
             + 2 * (L * 8 * expert_p) / GPU.flops["nvfp4"])
    return Model(
        key="q35", short="122B-A10B", label="Qwen3.5-122B-A10B",
        repo="nvidia/Qwen3.5-122B-A10B-NVFP4", quant="NVFP4", quant_label="NVFP4 (W4A4)",
        kv_dtype="fp8", hidden=h, n_layers=L, n_full=nfull, n_gdn=ngdn,
        n_heads=32, n_kv=2, head_dim=256, rope_dim=64,
        nk=nk, nv=nv, dk=dk, dv=dv, conv_k=ck, conv_dim=conv_dim,
        ffn="MoE", n_experts=E, top_k=8, moe_inter=me, shared_inter=se,
        vocab=V, max_ctx=262144,
        params_lang=lang, params_mtp=mtp, params_vision=vis, params_total=lang + mtp + vis,
        gdn_per_layer=gdn_p, attn_per_layer=attn_p, moe_per_layer=moe_p,
        expert_params=expert_p, expert_bytes_each=expert_bytes_each,
        quant_params=quant_params, bf16_params=bf16_params,
        weight_bytes=weight_bytes, real_weight_bytes=83474870752,
        real_dtypes={"BF16": 9122380016, "F8_E4M3": 7247757312, "U8": 57982058496},
        bf16_weight_bytes=(lang + mtp + vis) * 2, real_bf16_bytes=250173007840,
        dense_read_bytes=dense_read, token_compute_s=tok_c,
        activated=(ngdn * gdn_p + nfull * attn_p + L * (8 * expert_p + shared_p + gate_p)
                   + L * 2 * h + embed + h),
        spec_method="qwen3_next_mtp", spec_label="MTP (Qwen3.5 內建，EAGLE 系)",
        spec_k=3, spec_draft_params=mtp, spec_draft_serial=True,
        gdn_break=g, attn_break=a,
    )


def build_q38():
    h, L, nfull, ngdn, V, inter = 5120, 64, 16, 48, 248320, 17408
    nk, nv, dk, dv, ck = 16, 48, 128, 128, 4
    g, conv_dim = gdn_layer(h, nk, nv, dk, dv, ck)
    a = attn_layer(h, 24, 4, 256)
    gdn_p, attn_p = sum(g.values()), sum(a.values())
    mlp_p = 3 * h * inter
    embed = V * h
    lang = ngdn * gdn_p + nfull * attn_p + L * mlp_p + L * 2 * h + 2 * embed + h
    mtp = attn_p + mlp_p + 2 * h + h * 2 * h + 3 * h
    vis = sum(vision_tower(h).values())

    # FP8 blockwise(128)：mlp / self_attn 投影 / GDN 的 qkv,z,out — 與 weight map 完全吻合
    fp8_params = (L * mlp_p + nfull * (attn_p - 2 * 256)
                  + ngdn * (g["in_proj_qkv"] + g["in_proj_z"] + g["out_proj"])
                  + (mlp_p + attn_p - 2 * 256))
    bf16_params = lang + mtp + vis - fp8_params
    weight_bytes = fp8_params + bf16_params * 2
    dense_read = (ngdn * (g["in_proj_qkv"] + g["in_proj_z"] + g["out_proj"])
                  + nfull * (attn_p - 512) + L * mlp_p) * 1 \
        + (ngdn * (g["in_proj_a"] + g["in_proj_b"] + g["conv1d"] + g["A_log"] + g["dt_bias"] + g["norm"])
           + nfull * 512 + L * 2 * h + embed) * 2
    tok_c = (2 * (L * mlp_p + nfull * (attn_p - 512)
                  + ngdn * (g["in_proj_qkv"] + g["in_proj_z"] + g["out_proj"])) / GPU.flops["fp8"]
             + 2 * (embed + ngdn * (g["in_proj_a"] + g["in_proj_b"])) / GPU.flops["bf16"])
    return Model(
        key="q38", short="27B", label="Qwen3.8-27B",
        repo="Qwen/Qwen3.8-27B-FP8", quant="FP8", quant_label="FP8 blockwise-128 (W8A8)",
        kv_dtype="bf16", hidden=h, n_layers=L, n_full=nfull, n_gdn=ngdn,
        n_heads=24, n_kv=4, head_dim=256, rope_dim=64,
        nk=nk, nv=nv, dk=dk, dv=dv, conv_k=ck, conv_dim=conv_dim,
        ffn="Dense SwiGLU", n_experts=0, top_k=0, moe_inter=inter, shared_inter=0,
        vocab=V, max_ctx=262144,
        params_lang=lang, params_mtp=mtp, params_vision=vis, params_total=lang + mtp + vis,
        gdn_per_layer=gdn_p, attn_per_layer=attn_p, moe_per_layer=mlp_p,
        expert_params=0, expert_bytes_each=0,
        quant_params=fp8_params, bf16_params=bf16_params,
        weight_bytes=weight_bytes, real_weight_bytes=30866866928,
        real_dtypes={"BF16": 3082220272, "F8_E4M3": 24699207680},
        bf16_weight_bytes=(lang + mtp + vis) * 2, real_bf16_bytes=55562855904,
        dense_read_bytes=dense_read, token_compute_s=tok_c,
        activated=lang - embed,
        spec_method="dflash", spec_label="DFlash2 (block diffusion drafter)",
        spec_k=7, spec_draft_params=1924404480, spec_draft_serial=False,
        spec_draft_repo="incoai/Qwen3.8-27B-DFlash2", spec_block=8,
        gdn_break=g, attn_break=a,
    )


def build_q332():
    h, L, V, inter = 5120, 64, 151936, 25600
    a = attn_layer(h, 64, 8, 128, out_gate=False)
    attn_p = sum(a.values())
    mlp_p = 3 * h * inter
    lang = L * (attn_p + mlp_p + 2 * h) + 2 * V * h + h
    dense_read = lang * 2
    return Model(
        key="q332", short="32B", label="Qwen3-32B",
        repo="Qwen/Qwen3-32B", quant="BF16", quant_label="BF16",
        kv_dtype="bf16", hidden=h, n_layers=L, n_full=L, n_gdn=0,
        n_heads=64, n_kv=8, head_dim=128, rope_dim=128,
        nk=0, nv=0, dk=0, dv=0, conv_k=0, conv_dim=0,
        ffn="Dense SwiGLU", n_experts=0, top_k=0, moe_inter=inter, shared_inter=0,
        vocab=V, max_ctx=40960,
        params_lang=lang, params_mtp=0, params_vision=0, params_total=lang,
        gdn_per_layer=0, attn_per_layer=attn_p, moe_per_layer=mlp_p,
        expert_params=0, expert_bytes_each=0,
        quant_params=0, bf16_params=lang,
        weight_bytes=lang * 2, real_weight_bytes=65524246528,
        real_dtypes={"BF16": 32762123264},
        bf16_weight_bytes=lang * 2, real_bf16_bytes=65524246528,
        dense_read_bytes=dense_read,
        token_compute_s=2 * (lang - V * h) / GPU.flops["bf16"],
        activated=lang - V * h,
        spec_method=None, spec_label="無內建 MTP", spec_k=0, spec_draft_params=0,
        spec_draft_serial=True,
        gdn_break={}, attn_break=a,
    )


Q35, Q38, Q332 = build_q35(), build_q38(), build_q332()
MODELS = {m.key: m for m in (Q35, Q38, Q332)}


# ------------------------------------------------------ speculative model ---
# 期望接受長度 τ（含 bonus token）。q38 取 incoai 模型卡四個 benchmark 的平均，
# q35 取 vLLM / EAGLE 系 MTP 在 k=3 時的保守估計。
SPEC_TAU = {"q35": 2.6, "q38": 4.98, "q332": 1.0}
# 逐位置接受率（用來畫 α^i 衰減圖，並反推 τ）
SPEC_ALPHA = {"q35": 0.72, "q38": 0.80}


def tau_from_alpha(alpha, k):
    """τ = Σ_{i=0..k} α^i = (1-α^(k+1))/(1-α)"""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def draft_cost(m, batch, k):
    """一次 iteration 中「起草」階段的 (bytes, compute_s)。"""
    if m.key == "q35":
        # 內建 MTP：1 層 decoder，autoregressive 跑 k 次。
        # NVFP4 checkpoint 沒有量化 mtp.*，所以整顆 head 都是 BF16，
        # 而且每次 draft 都要重讀 lm_head 的 vocab 投影。
        per_pass_p = (m.attn_per_layer
                      + m.experts_touched(batch) * m.expert_params
                      + 3 * m.hidden * m.shared_inter + m.n_experts * m.hidden
                      + 2 * m.hidden * m.hidden          # mtp.fc
                      + m.vocab * m.hidden)              # lm_head
        act_p = (m.attn_per_layer + m.top_k * m.expert_params
                 + 3 * m.hidden * m.shared_inter + 2 * m.hidden * m.hidden
                 + m.vocab * m.hidden)
        return k * per_pass_p * 2, k * batch * 2 * act_p / GPU.flops["bf16"]
    if m.key == "q38":
        # DFlash2：獨立 2B BF16 drafter，block diffusion 一次算完整個 block。
        p = m.spec_draft_params
        n_pos = batch * (k + 1)
        return p * 2, n_pos * 2 * (p - m.vocab * m.hidden * 0) / GPU.flops["bf16"] * 0.83
    return 0.0, 0.0


def spec_iteration(m, batch, k=None, tau=None):
    """回傳 dict：一次 draft+verify iteration 的時間拆解與產出 token 數。"""
    k = m.spec_k if k is None else k
    tau = min(SPEC_TAU[m.key] if tau is None else tau, k + 1)
    verify_n = batch * (k + 1)
    v_bytes, v_comp = m.step_bytes(verify_n), m.step_compute_s(verify_n)
    d_bytes, d_comp = draft_cost(m, batch, k)
    t_verify = max(v_bytes / GPU.bw, v_comp)
    t_draft = max(d_bytes / GPU.bw, d_comp)
    base = m.step_time_s(batch)
    return {
        "k": k, "tau": tau, "batch": batch, "verify_n": verify_n,
        "verify_bytes": v_bytes, "verify_compute_s": v_comp, "t_verify": t_verify,
        "draft_bytes": d_bytes, "draft_compute_s": d_comp, "t_draft": t_draft,
        "t_iter": t_verify + t_draft, "tokens": batch * tau,
        "base_step_s": base,
        "tpot_spec": (t_verify + t_draft) / tau, "tpot_base": base,
        "tps_spec": batch * tau / (t_verify + t_draft), "tps_base": batch / base,
        "verify_bound": "compute" if v_comp > v_bytes / GPU.bw else "memory",
    }


def spec_speedup(m, batch, k=None, tau=None):
    r = spec_iteration(m, batch, k, tau)
    return r["tps_spec"] / r["tps_base"]


def spec_break_even(m, k=None, tau=None, hi=8192):
    lo, hi = 1.0, float(hi)
    if spec_speedup(m, hi, k, tau) > 1:
        return float("inf")
    if spec_speedup(m, lo, k, tau) < 1:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if spec_speedup(m, mid, k, tau) > 1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------ HBM budget ---
def budget(m, util=0.90, activation_gib=8.0, draft=False):
    """單卡 B200 記憶體帳本，回傳 GiB。"""
    total = GPU.hbm_bytes / GiB
    usable = total * util
    w = m.real_weight_bytes / GiB
    d = (m.spec_draft_params * 2 / GiB) if draft else 0.0
    cache = usable - w - d - activation_gib
    return {"total": total, "usable": usable, "weights": w, "draft": d,
            "activation": activation_gib, "cache": cache}


def max_seqs(m, tokens, kv_dtype=None, num_spec=0, **kw):
    kv_dtype = kv_dtype or m.kv_dtype
    per = m.seq_bytes(tokens, kv_dtype, num_spec) / GiB
    return budget(m, **kw)["cache"] / per


# ------------------------------------------------------------- reporting ---
def fmt_gb(b):
    return f"{b / 10**9:,.2f} GB"


def fmt_gib(b):
    return f"{b / GiB:,.2f} GiB"


def selftest():
    lines = []
    lines.append(f"roofline knee (precision-invariant): "
                 f"bf16={GPU.knee_tokens('bf16'):.1f} fp8={GPU.knee_tokens('fp8'):.1f} "
                 f"nvfp4={GPU.knee_tokens('nvfp4'):.1f} tokens/step")
    for m in (Q35, Q38, Q332):
        real = m.real_bf16_bytes / 2
        lines.append(f"\n=== {m.label} [{m.quant}]")
        lines.append(f"  params analytic {m.params_total:,}  vs real {int(real):,}  "
                     f"ratio {m.params_total / real:.5f}")
        lines.append(f"  lang {m.params_lang/1e9:.2f}B  mtp {m.params_mtp/1e9:.2f}B  "
                     f"vision {m.params_vision/1e9:.3f}B  activated {m.activated/1e9:.2f}B")
        lines.append(f"  weight bytes analytic {fmt_gb(m.weight_bytes)}  "
                     f"real {fmt_gb(m.real_weight_bytes)} = {fmt_gib(m.real_weight_bytes)}")
        for kvd in ("bf16", "fp8"):
            lines.append(f"  KV/token [{kvd}] {m.kv_bytes_per_token(kvd)/KiB:.0f} KiB   "
                         f"@{m.max_ctx} ctx = {fmt_gib(m.max_ctx*m.kv_bytes_per_token(kvd))}")
        if m.n_gdn:
            lines.append(f"  SSM recurrent {m.ssm_recurrent_bytes()/MiB:.1f} MiB   "
                         f"conv(k=0) {m.ssm_conv_bytes(0)/MiB:.2f} MiB   "
                         f"conv(k={m.spec_k}) {m.ssm_conv_bytes(m.spec_k)/MiB:.2f} MiB")
            lines.append(f"  SSM total(spec) {m.ssm_bytes(m.spec_k)/MiB:.1f} MiB  ->  "
                         f"= KV of {m.crossover_tokens(m.kv_dtype, m.spec_k):,.0f} tokens "
                         f"[{m.kv_dtype}]")
        lines.append(f"  step bytes @1tok {fmt_gb(m.step_bytes(1))}  @290 {fmt_gb(m.step_bytes(290))}"
                     f"  @2000 {fmt_gb(m.step_bytes(2000))}")
        lines.append(f"  token compute {m.token_compute_s*1e6:.3f} us   "
                     f"real knee {m.knee_tokens():.0f} tokens/step")
        lines.append(f"  batch-1 ceiling {1/m.step_time_s(1):.0f} tok/s   "
                     f"batch-64 {64/m.step_time_s(64):.0f} tok/s aggregate")
        b = budget(m, draft=bool(m.spec_k))
        lines.append(f"  budget: usable {b['usable']:.1f} GiB - w {b['weights']:.1f} "
                     f"- draft {b['draft']:.1f} - act {b['activation']:.1f} = cache {b['cache']:.1f} GiB")
        for t in (4096, 32768, 262144):
            if t <= m.max_ctx:
                lines.append(f"    @{t:>6} ctx: {m.seq_bytes(t, m.kv_dtype, m.spec_k)/MiB:8.1f} MiB/seq"
                             f"  -> {max_seqs(m, t, num_spec=m.spec_k, draft=bool(m.spec_k)):6.0f} seqs")
        if m.spec_k:
            lines.append(f"  spec: k={m.spec_k} tau={SPEC_TAU[m.key]:.2f} "
                         f"alpha={SPEC_ALPHA[m.key]} tau(alpha)={tau_from_alpha(SPEC_ALPHA[m.key], m.spec_k):.2f}")
            for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                r = spec_iteration(m, B)
                lines.append(f"    B={B:>4}  x{spec_speedup(m,B):5.2f}  "
                             f"draft {fmt_gb(r['draft_bytes']):>9} / {r['t_draft']*1e3:6.2f} ms   "
                             f"verify {fmt_gb(r['verify_bytes']):>9} / {r['t_verify']*1e3:6.2f} ms "
                             f"[{r['verify_bound']}]  tpot {r['tpot_spec']*1e3:6.2f} vs {r['tpot_base']*1e3:6.2f} ms")
            lines.append(f"  break-even batch {spec_break_even(m):.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(selftest())


# ---------------------------------------------------- 高併發的穩定態模擬 ---
def serve_sim(m, n_users, max_seqs, in_len=2048, out_len=512, budget=8192,
              spec_k=0, tau=1.0):
    """封閉式的穩定態近似，用來看 max_num_seqs 怎麼在 TTFT 與 TPOT 之間換。

    模型只有三個假設：
      1. running 佇列固定 B = min(併發需求, max_num_seqs)，其餘在 waiting 排隊。
      2. 穩定態下「進來的人」等於「出去的人」，所以每個 step 分給 prefill 的
         token 數 P 必須剛好餵飽 decode 的消耗：P / in_len = B·τ / out_len。
      3. 一個 step 的時間走 roofline：max(讀權重, 張量運算)。

    回傳的每個欄位都可以手算驗證，不含任何擬合參數。
    """
    B = max(1.0, min(n_users, max_seqs))
    queued = max(0.0, n_users - B)

    # 穩定態下每個 step 要撥給 prefill 的 token 數
    p_need = B * tau * in_len / out_len
    p_tok = min(p_need, max(0.0, budget - B * (spec_k + 1)))
    starved = p_need > p_tok + 1e-9          # budget 不夠餵 prefill

    n_step = B * (spec_k + 1) + p_tok
    t_mem = m.step_bytes(n_step) / GPU.bw
    t_cmp = m.step_compute_s(n_step)
    t_step = max(t_mem, t_cmp)

    tpot = t_step / tau                       # 每個使用者的每 token 時間
    decode_s = out_len * tpot
    prefill_steps = in_len / max(p_tok, 1e-9)
    prefill_s = prefill_steps * t_step
    service_s = prefill_s + decode_s          # 進到 running 之後的總服務時間

    rate = B / service_s                      # req/s：穩定態的完成率
    queue_s = queued / rate if rate > 0 else float("inf")
    ttft = queue_s + prefill_s
    e2e = ttft + decode_s
    tps = B * tau / t_step                    # 整機 decode 輸出

    return {
        "users": n_users, "max_seqs": max_seqs, "running": B, "queued": queued,
        "prefill_tokens_per_step": p_tok, "tokens_per_step": n_step,
        "budget_starved": starved,
        "t_step_ms": t_step * 1e3, "bound": "compute" if t_cmp > t_mem else "memory",
        "util_mem": min(1.0, t_mem / t_step), "util_cmp": min(1.0, t_cmp / t_step),
        "tpot_ms": tpot * 1e3, "itl_ms": t_step * 1e3,
        "prefill_ms": prefill_s * 1e3, "queue_ms": queue_s * 1e3,
        "ttft_ms": ttft * 1e3, "e2e_s": e2e, "rate_rps": rate,
        "tokens_per_s": tps,
    }


def sweep_max_seqs(m, n_users, candidates=(16, 32, 48, 64, 96, 128, 192, 256), **kw):
    return [serve_sim(m, n_users, c, **kw) for c in candidates]


def goodput(rows, ttft_ms=800.0, tpot_ms=40.0):
    """在同一組 SLO 下，哪些設定過關；回傳通過者的整機輸出。"""
    out = []
    for r in rows:
        ok = r["ttft_ms"] <= ttft_ms and r["tpot_ms"] <= tpot_ms
        out.append({**r, "slo_ok": ok, "goodput": r["tokens_per_s"] if ok else 0.0})
    return out
