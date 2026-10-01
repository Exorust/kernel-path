# Agent C raw report: gated delta rule / chunkwise / linear attention resources

Provenance: subagent (Opus), run 2026-10-01, 15 tool-call budget, free text-first sources.
Access path: curl on raw.githubusercontent.com, api.github.com, arxiv.org/html, huggingface.co raw READMEs and config.json, HF and arXiv APIs.
Caveats: primary sources only, no community sampling; FLA tolerance values UNVERIFIED. Verbatim below.

---
## Findings

(AS_OF 2026-10-01. All quotes read literally via curl; no WebFetch summarization used.)

### Q1 — Delta rule / gated delta rule as a recurrence, and why it is an online least-squares update

**#1. Songlin Yang, "DeltaNet Explained (Part I)" — https://sustcsonglin.github.io/blog/2024/deltanet-1/** (academic, author of both papers; Dec 3 2024, 14 min read)
Read: "What is Delta Rule?" → "What is DeltaNet?" → "Why is DeltaNet Superior at In-context Retrieval Compared to Linear Attention?". Starts from pedagogy, ships runnable NumPy.
> "What is Delta Rule? The Delta Rule is a fundamental error-correction learning principle in neural networks. Its core idea is beautifully simple: adjust the model's parameters based on the difference (delta) between what we want (target) and what we actually get (prediction). To understand this intuitively, imagine teaching a child to aim at a target. If they shoot too far to the left, you'd tell them to adjust right; too far right, adjust left. The size of the adjustment depends on how far they missed - a concept directly reflected in the Delta Rule."
Scalar reference inline: `error = y[i] - np.dot(x[i], w)` ... `w += lr * error * x[i]`. Est. 15 min.

**#2. Gated DeltaNet paper §2.3 + §3.1 — https://arxiv.org/html/2412.06464v1** (academic). ~20 min. The recurrence derived twice: erase/write, and SGD.
> "The delta update rule (Widrow et al., 1960; Schlag et al., 2021b) dynamically erases the value ($v_t^{old}$) associated with the current input key ($k_t$) and writes a new value ($v_t^{new}$), which is a linear combination of the current input value and the old value. This process updates a key-value association pair at each time step, where the scalar $\beta_t\in(0,1)$ determines the extent to which the old association is replaced by the new one"
> "the hidden state $S$ can be interpreted as a weight matrix, with the delta rule optimizing the objective $L(S_t)=\frac{1}{2}\|S_t k_t - v_t\|^2$ via online stochastic gradient descent (SGD): $S_{t+1} = S_t - \beta_t\nabla_S L(S_t) = S_t - \beta_t(S_t k_t - v_t)k_t^\intercal = S_t(I-\beta_t k_t k_t^\intercal) + \beta_t v_t k_t^\intercal$ ... where $\beta_t$ represents the (adaptive) learning rate. From this perspective, the gated delta rule can be viewed as incorporating an adaptive weight decay term $\alpha_t$ into the SGD update"
> Eq. 8: "$S_t = S_{t-1}\left(\alpha_t(I-\beta_t k_t k_t^\intercal)\right) + \beta_t v_t k_t^\intercal$ ... where the data-dependent gating term $\alpha_t\in(0,1)$ controls state decay."
Table 1: LA / Mamba2 / Longhorn / DeltaNet / Gated DeltaNet as five online-learning objectives + five update rules.
**Notation conflict:** GDN applies the transition on the RIGHT (`S_{t-1}(α_t(I − β_t k_t k_tᵀ))`); Kimi Linear restates it on the LEFT with `k_t v_tᵀ`. Same math, transposed. The brief's form matches GDN/FLA.

**#3. DeltaNet paper — https://arxiv.org/abs/2406.06484** (academic; v6 15 Jan 2025). Citation anchor and Householder framing; the blog supersedes it.
> "This work describes a hardware-efficient algorithm for training linear transformers with the delta rule, which exploits a memory-efficient representation for computing products of Householder matrices."

### Q2 — Chunkwise-parallel formulation

**#1. Songlin Yang, "DeltaNet Explained (Part II)" — https://sustcsonglin.github.io/blog/2024/deltanet-2/** (17 min read; ~35 min with algebra). Single best explainer: shows the failed alternative first and gives the reason the triangular solve exists. TOC verbatim:
> "Parallel Scan for DeltaNet: A Failed Attempt / From Delta Updates to Matrix Multiplication Form / Defining the Associative Operator / Parallel Scan for DeltaNet / What's Wrong with Parallel Scan for DeltaNet? / A Chunkwise Algorithm for DeltaNet / Chunkwise Parallel Form for Linear Attention / WY representation for DeltaNet / Chunkwise Parallel Form for DeltaNet / UT Transform Through the Lens of Graph Theory / Speed comparison"
> "the original DeltaNet treated DeltaNet as a pure RNN which required O(L) sequential steps, which is inefficient on modern hardware such as GPUs with massive parallel processing capabilities. We thus seek strategies to parallelize DeltaNet across sequence length to enable hardware-efficient training."
> "We implemented both the recurrent and chunkwise parallel versions of DeltaNet using Triton. ... To ensure fair comparison across configurations, we kept the total sequence elements constant at 16,384 by adjusting batch sizes accordingly. As we can see in the figure above, our chunkwise parallel approach consistently outperforms the recurrent baseline. More importantly, this performance advantage grows more pronounced under two key conditions: as sequences get longer and as head dimensions increase."
> "In graph theory, for a weighted directed graph, the adjacency matrix $A$ captures direct connections - entry $A[i,j]$ represents the edge weight from node $j$ to node $i$. When we compute $(I-A)^{-1}$, each entry $[i,j]$ gives the sum of weights of all possible paths from $j$ to $i$."
> "$(I - A_{[t]})$ is also lower triangular with ones on the diagonal. This special structure allows us to efficiently compute its inverse through forward substitution: $T_{[t]} = (I - A_{[t]})^{-1}$ ... This avoids the need for general matrix inversion, making the computation much more efficient. After obtaining $T_{[t]}$, which captures all accumulated influence paths between positions, we proceed with the final multiplication: $W_{[t]} = T_{[t]}\operatorname{diag}(\beta_{[t]})K_{[t]}$, $U_{[t]}=T_{[t]}\operatorname{diag}(\beta_{[t]})V_{[t]}$"
Caveat: Part II covers UNGATED DeltaNet. Compose with GDN §3.2 for the γ terms.

**#2. Gated DeltaNet paper §3.2 + Appendix A.1 — https://arxiv.org/html/2412.06464v1** (~25 min). Authoritative gated chunkwise form = the kernel contract. Eq. (11), (12):
> "$S_{[t+1]} = \gamma_{[t]}^C S_{[t]} + \left(U_{[t]} - \operatorname{Diag}(\gamma_{[t]})W_{[t]}S_{[t]}^\intercal\right)^\intercal \operatorname{Diag}\left(\frac{\gamma_{[t]}^C}{\gamma_{[t]}}\right)K_{[t]}$"
> "$O_{[t]} = \operatorname{Diag}(\gamma_{[t]})Q_{[t]}S_{[t]}^\intercal + (Q_{[t]}K_{[t]}^\intercal \odot \Gamma_{[t]})\left(U_{[t]} - \operatorname{Diag}(\gamma_{[t]})W_{[t]}S_{[t]}^\intercal\right)$"
> "We can see that the key distinction lies in the replacement of the value block $V_{[t]}$ with the 'pseudo'-value term $U_{[t]} - \operatorname{Diag}(\gamma_{[t]})W_{[t]}S_{[t]}^\intercal$. This modification resembles Eq. 6-7, but notably incorporates decay-awareness."
> "$W_{[t]} = A^W_{[t]}\operatorname{Diag}(\beta_{[t]})K_{[t]}$, $A^W_{[t]} = \left(I - \operatorname{lower}(\operatorname{Diag}(\beta_{[t]})K_{[t]}K_{[t]}^\intercal)\right)^{-1}$ ... $U_{[t]} = A^U_{[t]}\operatorname{Diag}(\beta_{[t]})V_{[t]}$, $A^U_{[t]} = \left(I - \operatorname{lower}\left(\operatorname{Diag}(\beta_{[t]})(\Gamma_{[t]} \odot K_{[t]}K_{[t]}^\intercal)\right)\right)^{-1}$ where $\operatorname{lower}(\cdot) := \operatorname{tril}(\cdot,-1)$; and the inverse of a lower triangle matrix can be calculated efficiently by forward substitution."
> "Similar to Mamba2, the gating term (colored in blue) only performs elementwise multiplication with (intermediate) variables without affecting matrix multiply structures, enabling tensor core GPU optimization. As shown in Fig. 3, Gated DeltaNet maintains the same speed as DeltaNet"
Appendix A.1: ~1-page induction proof of `S_t = Σ_i (γ_t/γ_i) u_i k_iᵀ`. "We proof this by mathmetical induction." (sic). Typo: `P_t = Π α_t (I − β_i k_i k_iᵀ)` should index α by i.

**#3. `naive_chunk_gated_delta_rule` in FLA** — the small worked example: 60 lines of PyTorch, run with C=4 and print intermediates.

### Q3 — Readable reference code (paths confirmed 2026-10-01)

**#1. `fla/ops/gated_delta_rule/naive.py` — READ THIS FIRST.** https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/naive.py — 5,290 bytes, 161 lines, pure PyTorch. `naive_recurrent_gated_delta_rule` (L13-64) and `naive_chunk_gated_delta_rule` (L67-161). Everything cast to fp32. Est. 45 min.
- recurrence L54-59, `g` is log-space: `h = h.clone() * g[:, :, i].exp()[..., None, None]` / `b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)` / `b_v = b_v * b_beta[..., None]` / `h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)` / `o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)`.
- cumulative decay L127-129: `decay = decay.squeeze(-1).cumsum(-1)` / `decay_exp = decay.exp()[..., None]` / `L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()`.
- forward substitution L130-133 (`# note that diagonal is masked.` L121): `attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)` / `for i in range(1, chunk_size): attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)` / `attn = attn + torch.eye(chunk_size, ...)`.
- pseudo-value and carry L148-153: `v_prime = (k_cumdecay[:, :, i]) @ S` / `v_new = v_i - v_prime` / `o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S` / `o[:, :, i] = o_inter + attn @ v_new` / `S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new`.

**#2. `fla/ops/gated_delta_rule/chunk.py` + `wy_fast.py`** (Triton). https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/chunk.py — 20,663 bytes, 596 lines. Dir (API, 2026-10-01): `__init__.py` 719, `backends/`, `chunk.py` 20663, `chunk_fwd.py` 17183, `fused_recurrent.py` 17056, `gate.py` 9441, `naive.py` 5290, `wy_fast.py` 11564.
Read: L70-71 `# obtain WY representation. u is actually the new v.` / `# fused kkt + solve_tril + recompute_w_u`; `chunk_gated_delta_rule_fwd` (L33); `chunk_gated_delta_rule_bwd` (L126); `class ChunkGatedDeltaRuleFunction` (L253); `chunk_gated_delta_rule` (L397). L499-501: `# for variable-length inputs, the batch size B is expected to be 1 and cu_seqlens is required` / `# for a batch with 4 sequences, cu_seqlens with 5 start/end positions are expected`. Est. 1 h forward only. The four-stage split kkT → triangular solve → recompute W,U → chunk scan is the kernel skeleton.

**#3. `mlx_lm/models/gated_delta.py`** — https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gated_delta.py — 647 lines, 21,102 bytes, `# Copyright © 2025 Apple Inc.` Alongside `qwen3_next.py` (17,853 B), `kimi_linear.py` (21,488 B), `kimi_k3.py`, `kimi_k25.py`.
Recurrent per-token Metal scan, NOT chunkwise: six kernel variants via `_make_gated_delta_kernel(has_mask, vectorized)`, `_make_gated_delta_kernel_xtree()` (L152), `_make_gated_delta_packed_kernel()` (L247), dispatched in `_gated_delta_kernel_impl` (L429). Scalar vs vector: `decay = g[..., None, None]` vs `decay = g[..., None, :]` at L411/L413; docstring L568-573: "Supports both scalar and vectorized gating. ... g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)". Est. 45 min.

**#4. FlashQLA (QwenLM/FlashQLA)** — https://github.com/QwenLM/FlashQLA, README raw 5,749 bytes. TileLang, production, not teaching. README only.
> "FlashQLA applies **reasonable operator fusion and performance optimization** to the forward and backward passes of GDN Chunked Prefill, achieving **2-3× forward speedup** and **2× backward speedup** over the FLA Triton kernel across multiple scenarios on NVIDIA Hopper and Blackwell."
> "Rather than following the step-by-step decomposition into independent kernels, nor fusing the entire computation flow into a single kernel, we take CP and backward requirements into account, use TileLang to build several key fused kernels, and manually implement warpgroup specialization to overlap data movement, Tensor Core computation, and CUDA Core computation."
Signature `chunk_gated_delta_rule(q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens)`, shapes `q,k: [B,T,H_q,K]`, `v: [B,T,H_v,V]`, `g,beta: [B,T,H_v]`, `initial_state: [B,H_v,K,V]` = cross-library contract. "[2026-07] ... now serves as a backend for flash-linear-attention's GDN". Est. 10 min.

**#5. `fla/ops/kda/naive.py`** — https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/kda/naive.py — 6,339 bytes. `naive_recurrent_kda`: "g (torch.Tensor): Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``", "``HV`` must be divisible by ``H``." ~15 min. kda dir: `chunk.py` 22812, `chunk_bwd.py` 20513, `chunk_fwd.py` 4568, `chunk_intra.py` 36577, `chunk_intra_token_parallel.py` 5524, `fused_recurrent.py` 18203, `gate.py` 15758, `naive.py` 6339, `wy_fast.py` 11928.

### Q4 — Vector (per-channel) gating (KDA) vs scalar gating

**#1. Kimi Linear tech report — https://arxiv.org/html/2510.26692v1** (academic; arXiv API PUBLISHED 2025-10-30). Read §1, §3.1, §6.1, §6.2, Appendix C Listings 8(a)/8(b). Est. 50 min.
> "While GDN, similar to Mamba2 [16], employs a coarse head-wise forget gate, KDA introduces a channel-wise variant in which each feature dimension maintains an independent forgetting rate, akin to Gated Linear Attention (GLA) [114]. This fine-grained design enables more precise regulation of the finite-state RNN memory"
> Eq. 1: "$S_t = (I - \beta_t k_t k_t^\top)\operatorname{Diag}(\bm{\alpha}_t)S_{t-1} + \beta_t k_t v_t^\top \in \mathbb{R}^{d_k\times d_v}$; $\bm{o}_t = S_t^\top q_t$"
> "a key strength of RoPE is its fine-grained positional encoding, achieved by assigning different rotation frequencies to each pair of dimensions, which functions analogously to a Nonuniform Fourier Transform [7,41] along the feature dimension. Standard GDN, however, employs a per-head scalar decay and lacks this per-dimensional diversity, which motivates us to propose KDA with a learnable channel-wise gate."
> "The per-channel decay $\bm{\alpha}^h_t$ is parameterized via a low-rank projection ($W_\alpha^{\downarrow}$ and $W_\alpha^{\uparrow}$ with rank equal to the head dimension) and a decay function $f(\cdot)$ similar to those used in GDN and Mamba [111, 16]."
> "Despite incorporating a more fine-grained decay mechanism, Kimi Linear introduces negligible latency overhead compared to GDN-H during prefilling."
> "By binding both variables $\bm{a}$ and $\bm{b}$ to $\bm{k}$, KDA effectively alleviates this bottleneck—reducing the number of second-level chunk matrix computations from four to two, and further eliminating three additional matrix multiplications. As a result, the operator efficiency of KDA improves by roughly 100% compared to the DPLR formulation."
> "We further benchmark the kernel speed in Fig. 2, showing that KDA achieves nearly $2\times$ the speed of DPLR for sequence lengths up to $64\text{k}$."
Highest-value single page for a kernel writer: Appendix C Listings 8(a) `chunk_dplr` and 8(b) `chunk_kda`, two ~30-line PyTorch functions diffed. `chunk_kda` builds two C×C matrices (`Aqk, Akk`), output `o[:,:,i]=(q_i*g_i.exp()) @ S + Aqk @ (u_i - w_i @ S)`, state `S += (k_i*decay).transpose(-1,-2) @ v_i`. Same forward-substitution loop. Est. 20 min.

**#2. `fla/ops/kda/` vs `fla/ops/gated_delta_rule/`** — gate.py 9,441 B vs 15,758 B; `chunk_intra.py` (36,577 B) exists only on the KDA side.

**#3. Kimi-Linear model card** — https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct (official).
> "At its core is Kimi Delta Attention (KDA)—a refined version of [Gated DeltaNet](https://arxiv.org/abs/2412.06464) that introduces a more efficient gating mechanism to optimize the use of finite-state RNN memory."

### Q5 — Numerics: gate cumprods, log space, tolerance vs bit-exactness

No standalone short note exists (NOT_FOUND). Three primary substitutes:

**#1. Kimi Linear §3.2 + §6.2** — https://arxiv.org/html/2510.26692v1. ~10 min.
> "However, such fine-grained decay introduces numerical precision issues during division operations (e.g., the intra-chunk computation in Eq. 9). To address this, prior work such as GLA [114] performs computations in the logarithmic domain and introduces secondary chunking in full precision. This approach, however, prevents full utilization of half-precision matrix multiplications and significantly reduces operator speed."
> "the reciprocal of the cumulative decay term $1/\Gamma$ in chunkwise form (Eq. 9) can introduce numerical instability. While we can resolve this issue by secondary chunking [113], it incurs additional computation and I/O overhead. By fixing $\bm{a}=\bm{b}=\bm{k}$ in the DPLR formulation, KDA removes the need for two secondary chunking steps"
Mechanism: intra-chunk term `Tril((Γ ⊙ Q)(K/Γ)ᵀ)`; `K/Γ` divides by a product of r gates.

**#2. The log-space contract in FLA, as code.** ~20 min.
- https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/gate.py (9,441 B). `naive_gdn_gate` L21-42: "Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``", L46 `return (-A_log.float().exp() * F.softplus(g)).to(output_dtype)`. `g` is `log α`, ≤ 0 by construction. `gdn_gate_chunk_cumsum_scalar_kernel` (L63) loads `.to(tl.float32)` (L94-97), `b_o = tl.cumsum(b_gate, axis=0)` (L100): fp32 cumsum of non-positive logs, never a running product. L149-151 comments: `# gate = -exp(A_log) * softplus(g + bias)` / `# d(gate)/d(g) = -exp(A_log) * sigmoid(g + bias) (softplus' = sigmoid)` / `# d(gate)/d(A_log) = -exp(A_log) * softplus(g + bias) = gate`.
- `naive.py`: `map(lambda x: x.transpose(1,2).contiguous().to(torch.float32), [q,k,v,beta,g])`; every decay use is `.exp()` of a difference of cumsums. mlx-lm `gated_delta.py` L19-20: `def compute_g(A_log, a, dt_bias): return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))`, plus `compute_lower_bound_g(A_log, a, dt_bias, lower_bound)` at L24 (a gate floor).

**#3. `mlx_lm/models/gated_delta.py` header + kernel docstrings (L10-15, 152-163, 247-267)** — best thing found on bit-exactness vs tolerance. Est. 15 min.
> "For the shapes it supports, the packed kernel is bitwise-identical by construction to an explicit-tree comparator kernel that the tests pin it against (see _make_gated_delta_packed_kernel). Every other shape (masks, vector gates, Dk != 128) and the MLX_GDN_PACKED=0 kill-switch use the original simd_sum kernels, unchanged."
> "This is the unpacked comparator for the packed kernel: it replaces the two simd_sum calls with the ascending butterfly (shuffle_xor 1,2,4,8,16) written out in source, so the reduction order is a contract of this file rather than of the simd_sum lowering. On current Apple GPUs this is the same tree simd_sum lowers to, so it is bit-identical to the generic kernel there."
> "The reduction reproduces the unpacked comparator (_make_gated_delta_kernel_xtree) bitwise BY CONSTRUCTION: both use the same explicitly-written ascending butterfly (shuffle_xor 1,2,4,8,16) rather than relying on how simd_sum lowers. The butterfly's first three levels combine partials that live in a single packed lane (IEEE addition is commutative, so the local pairwise tree is bit-identical), and the last two levels map onto shuffle_xor(1) and shuffle_xor(2) within the four-lane row group. Each 4-element partial keeps the comparator's sequential order, so y and the state are bit-identical to it on any device."
Lesson: bit-exactness across two implementations only if the reduction tree is written out in source; ship a kill-switch plus a slow comparator kernel as test oracle. Counterpoint: FlashQLA README "without sacrificing numerical precision" is unquantified; UNVERIFIED until `tests/test_gdr_unit.py` is read.

### Q6 — Models on this path in 2026 (blast radius)

**Qwen3-Next — Gated DeltaNet.** https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct (README raw):
> "- **Hybrid Attention**: Replaces standard attention with the combination of **Gated DeltaNet** and **Gated Attention**, enabling efficient context modeling for ultra-long context length."
> "- Hybrid Layout: 12 \* (3 \* (Gated DeltaNet -> MoE) -> 1 \* (Gated Attention -> MoE))"
> "- Gated DeltaNet: - Number of Linear Attention Heads: 32 for V and 16 for QK"
config.json: `"model_type": "qwen3_next"`, `"full_attention_interval": 4`, `"linear_key_head_dim": 128`, `"linear_num_key_heads": 16`, `"linear_num_value_heads": 32`, `"linear_value_head_dim": 128`, `"num_hidden_layers": 48`, `"max_position_embeddings": 262144`. 36 of 48 layers GDN.

**Qwen3.5 — Gated DeltaNet.** https://huggingface.co/Qwen/Qwen3.5-9B (README raw):
> "- **Efficient Hybrid Architecture**: Gated Delta Networks combined with sparse Mixture-of-Experts deliver high-throughput inference with minimal latency and cost overhead."
> "    - Hidden Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))"
config.json: `"model_type": "qwen3_5"`, `"num_hidden_layers": 32`, `"layer_types"` = `["linear_attention"×3,"full_attention"]×8`, same 128/16/32/128 head config, `"mamba_ssm_dtype": "float32"`. HF downloads 2026-10-01: Qwen3.5-9B 9,282,840; Qwen3.5-4B 7,539,039; Qwen3.5-0.8B 2,523,451. FlashQLA README: "head configurations used by the Qwen3.5 / Qwen3.6 family h_k,v ∈ {64, 48, 32, 24, 16, 8}".

**Kimi Linear — KDA.** https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct:
> "**Hybrid Architecture:** A 3:1 KDA-to-global MLA ratio reduces memory usage while maintaining or surpassing the quality of full attention."
> "We open-source the KDA kernel in [FLA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda), and release two versions model checkpoints trained with 5.7T tokens."
Paper abstract: "reducing KV cache usage by up to 75% and achieving up to 6 times decoding throughput for a 1M context."

**Convergent layout:** three families, 3 linear : 1 full attention, dk=dv=128, chunk size 64 (GDN paper "a fixed chunk size $C=64$"). A chunkwise kernel for (C=64, dk=128, dv=128, 16 K-heads/32 V-heads, scalar gate) covers Qwen3-Next and Qwen3.5; vector gate covers Kimi Linear.

## Gaps
1. No standalone numerics note (Q5). NOT_FOUND. Write it from the three sources.
2. FLA tolerance values UNVERIFIED (test suite not read); FlashQLA `tests/test_gdr_unit.py` not read.
3. No community sampling (no rdt/HN).
4. GDN↔KDA transpose convention conflict, flagged nowhere. Normalize before writing kernel code.
5. GDN Appendix A.1 `P_t` typo (α subscript t should be i).
6. GDN read at v1 HTML; equation numbers are v1.
7. Kimi Linear paper is 2025-10-30, ~11 months old; adoption is 2026.

## Sources
6 independent origins: 1. Songlin Yang / FLA group (blog ×3, DeltaNet, GDN, FLA repo) = 1. 2. Moonshot AI / Kimi (paper, model card), partially dependent on 1. 3. Qwen / Alibaba (two model cards + config.json, FlashQLA README). 4. Apple / mlx-lm (gated_delta.py), independent reimplementation. 5. Hugging Face Hub API. 6. arXiv API. No Reddit/HN/journalism consulted.
