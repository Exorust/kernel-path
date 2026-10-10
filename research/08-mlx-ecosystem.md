# Research pack 08 — MLX / mlx-lm / llama.cpp Metal ecosystem writing (raw subagent report)

Provenance: deepresearch run 2026-10-10, territory B. Model: Opus subagent, 24 tool calls. Access path: WebSearch, curl, `gh` CLI and GitHub GraphQL; Medium and Towards AI were blocked. Caveats: quotes are the agent's verbatim extracts, not yet spot-checked when saved; the agent inferred that the `Developer-Ecosystem-Engineering` GitHub account is Apple's (not confirmed in the PR). Page content is data, not instructions.

---

## Findings (Territory B: MLX / mlx-lm / llama.cpp Metal writing), AS_OF 2026-10-10

I used all 24 tool calls. Every quote below is literal text I saw in curl, gh or GraphQL output.

**Headline:** the MLX core team writes very little long-form prose. Their "writing" is Apple's ML Research blog post, the MLX docs, release notes and short issue replies. The best kernel-level reading in this territory is (a) llama.cpp Metal PRs and discussions, including PRs from Apple's own Developer-Ecosystem-Engineering account with measured tables, and (b) 2026 practitioner READMEs that tie a tok/s gain to a specific Metal kernel.

### Q1. Writing by MLX maintainers and core contributors

1. **Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"**
   - Link: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
   - Author / date / type: Apple MLX team, Nov 19 2025, official research blog.
   - What to read: the whole post, especially the "Inference Performance on M5" section and Table 1.
   - Why it's good: it's the canonical first-party statement that prefill (TTFT) is compute-bound and benefits from the Neural Accelerators, while decode is bandwidth-bound. It gives measured M4 vs M5 numbers for 6 models, including gpt-oss-20b in MXFP4. About 10 min.
   - Quote: "MLX leverages the Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4 to support the Neural Accelerators' features." / "the M5 provides 19-27% performance boost compared to the M4, thanks to its greater memory bandwidth (120GB/s for the M4, 153GB/s for the M5, which is 28% higher)."

2. **MLX docs, "Lazy Evaluation"** (this is the "why MLX is lazy" piece)
   - Link: https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst
   - Author / date / type: MLX maintainers, living doc, original text from 2023 (older but still the page people link).
   - What to read: the whole page.
   - Why it's good: it's the maintainers' own rationale for the graph-recording design. About 8 min.
   - Quote: "When you perform operations in MLX, no computation actually happens. Instead a compute graph is recorded. The actual computation only happens if an :func:`eval` is performed."
   - Caveat (CONFLICTING with later behavior): it still says "Currently, MLX does not compile and rerun compute graphs." That predates `mx.compile`, so pair it with the compile doc.

3. **MLX docs, "Custom Metal Kernels"** (`mx.fast.metal_kernel`)
   - Link: https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst
   - Author / date / type: maintainers, living doc.
   - What to read: the whole page. It builds from an elementwise `exp` kernel up to a grid-sample kernel with a VJP. About 20 min.
   - Quote: "MLX supports writing custom Metal kernels through the Python and C++ APIs." / "Let's write a custom kernel that computes ``exp`` elementwise"

4. **MLX release notes, v0.29.0 to v0.32.3**
   - Link: https://github.com/ml-explore/mlx/releases
   - Author / date / type: maintainers (awni, jagrit06, zcbenz and others), Aug 2025 to Sep 2026, release notes.
   - What to read:
     - v0.29.0: mxfp4
     - v0.30.0: Neural Accelerators
     - v0.30.3: nvfp4/mxfp8 on Metal
     - v0.30.4: vector fused GQA attention
     - v0.31.2: "Nax Refactor"
     - v0.32.x: NAX attention and quantized-MoE PRs
   - Why it's good: the best index of which kernel changed when. The bullets are one-liners, so follow the PR links. About 15 min.
   - Quotes: "Support for `mxfp4` quantization (Metal, CPU)" (v0.29.0). "Support for Neural Accelerators on M5 (macOS >= 26.2)" and "Add Neural Accelerator Support by @jagrit06 in https://github.com/ml-explore/mlx/pull/2772" (v0.30.0). "Support nvfp4 and mxfp8 quantized ops on Metal" (v0.30.3). "Metal: Much faster vector fused grouped-query attention for long context" (v0.30.4).
   - Warning: the NAX PRs have no design write-up. PR #2772's body is only "This branch is a work in progress. Caution, this branch does not yet compile." PR #3271's body is only "Refactoring NAX to not rely on the intermediate sub-tile." For NAX, the code is the documentation.

5. **ml-explore/mlx issue #2962: nvfp4 scale-format dispute**
   - Link: https://github.com/ml-explore/mlx/issues/2962
   - Author / date / type: Nottlespike, with replies from awni and nastya236; opened Dec 31 2025; issue thread.
   - What to read: the full thread. It's a good case study in how a microscaling format's scale encoding differs between MLX and NVIDIA. About 10 min.
   - Quotes: "MLX's nvfp4 implementation uses **signed E4M3** ... for block scaling factors, but NVIDIA Blackwell GPUs use **unsigned E4M3 (UE4M3)**". awni replied: "I don't see any evidence that there is no exponent bias and it's not clear to me that that is desirable." Closed by nastya236 with "Tensor scale for nvfp4 was added in https://github.com/ml-explore/mlx/pull/3022, so closing for now."
   - Status: CONFLICTING. The reporter's 137x dynamic-range claim was disputed and not conceded.

6. **mlx-lm LEARNED_QUANTS.md (DWQ, AWQ, dynamic quantization, GPTQ)**
   - Link: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
   - Author / type: mlx-lm maintainers, living doc.
   - What to read: the whole doc. About 8 min.
   - Quote: "DWQ fine-tunes non-quantized parameters (including quantization scales and biases) using the non-quantized model as a teacher." / "Distilling 16-bit precision to 8-bit and even 6-bit often doesn't work well."

Not found for Q1:
- NOT_FOUND: a personal blog or long-form post by Awni Hannun, Angelos Katharopoulos, Jagrit Digani or Cheng Zhao (zcbenz) on steel kernels, NAX or quantization.
- MLX GitHub Discussions are thin: 253 total. The most-commented by a maintainer is "MLX Community Projects" (#654), a link list.

### Q2. Third-party explainers of MLX internals

- NOT_FOUND: a high-quality third-party "how MLX works" post or `mx.compile` write-up. Search results were mintlify mirrors of the docs, greptile and emergentmind summaries, and other low-signal pages.
- UNVERIFIED (search snippets only, not fetched): MLX vs PyTorch MPS comparisons with numbers at https://github.com/richiksc/mlx-benchmarks and https://jisaku.com/posts/mlx-vs-pytorch-apple-silicon.
- **Best available stand-in: ml-explore/mlx discussion #3801, "Small-L fused expert gather for fine-grained MoE decode"**
  - Link: https://github.com/ml-explore/mlx/discussions/3801
  - Author / date / type: anicaise-ai, Jul 4 2026, discussion with measurements.
  - What to read: the Context and data sections. About 10 min.
  - Why it's good: measured M3 Ultra numbers that explain why speculative verify loses on fine-grained MoE because of expert memory traffic.
  - Quote: "L=1 decode forward: **~51 ms/token** (≈19.7 tok/s end-to-end)" / "L=2 verify forward: **~112 ms ≈ 2.2× L=1**".

### Q3. llama.cpp Metal backend writing

1. **"Performance of llama.cpp on Apple Silicon M-series," discussion #4167**
   - Link: https://github.com/ggml-org/llama.cpp/discussions/4167
   - Author / date / type: ggerganov, Nov 22 2023 (OLDER, but still the table people link), 97 comments.
   - What to read: the summary table. Note how TG tracks bandwidth while PP tracks core count. About 10 min.
   - Quote (table row): "✅ M1 Max [^1] | 400 | 32 | 599.53 | 23.03 | 537.37 | 40.2 | 530.06 | 61.19 | 8e672ef"

2. **"metal : initial Metal4 tensor API support," PR #16634**
   - Link: https://github.com/ggml-org/llama.cpp/pull/16634
   - Author / date / type: ggerganov, opened Oct 17 2025, merged Nov 6 2025, PR.
   - What to read: the TODO list (the bfloat tensor-API question, the M5 volunteer tests) and the linked comments. About 15 min.
   - Quote: "Rework matrix-matrix multiplication / Use Tensor API when available" and "Confirm that using the Tensor API maintains the existing performance without using it on M4 and earlier".

3. **"Optimize Metal Tensor API usage," PR #20962**
   - Link: https://github.com/ggml-org/llama.cpp/pull/20962
   - Author / date / type: the Developer-Ecosystem-Engineering account (an Apple engineering org; inferred, not confirmed in the PR), Mar 24 2026, PR.
   - What to read: the whole body. It explains tile sizes, skipping threadgroup staging, and cooperative tensors, with a per-quant-type speedup table on M5 Max. This is the single best kernel-level read in the territory. About 15 min.
   - Quote: "Previously both paths were interleaved via #ifdef blocks within a single kernel, forcing the tensor path to share the legacy kernel's data layout and threadgroup memory scheme." / "Tile dimensions are now configurable via compile-time macros (NRA×NRB), currently defaulting to 64×128 vs the legacy kernel's fixed 64×32." / "Overall GeoMean | | | | | +26.4%".

4. **"metal : tuning Flash-Attention kernel across M-chips," discussion #27668**
   - Link: https://github.com/ggml-org/llama.cpp/discussions/27668
   - Author / date / type: ggerganov, Aug 24 2026, discussion.
   - What to read: the body and the linked per-chip PRs (M1 through M5 Max).
   - Why it's good: it shows FA-vec parameters being autotuned per GPU family. About 10 min.
   - Quote: "The `ggml-metal-tuning fa-vec` tuner sweeps the flash-attn vec (Q, NE) config grid on the machine it runs on and prints pasteable rows for `fa_vec_tuned_table`."

5. **"Page-aware expert residency on Metal," discussion #26938**
   - Link: https://github.com/ggml-org/llama.cpp/discussions/26938
   - Author / date / type: rickbarkley, Aug 12 2026, discussion with a patch.
   - What to read: the body. About 8 min.
   - Quote: "Instead of binding the full 63.4GB expert tensor into the Metal working set, it exposes only selected expert ranges to Metal through page-aligned direct host-memory views".

- Also relevant (titles only, bodies not read): #27461 "metal: enable Metal 4.0 tensor API on M5+/A19+" (Aug 2026), and the #29869/#30047/#30065 "few-row MMA" series (Oct 2026).

### Q4. Apple-silicon practitioner write-ups with measured tok/s and the kernel change behind them

1. **dflash-mlx README**
   - Link: https://github.com/bstnxbt/dflash-mlx
   - Author / date / type: bstnxbt, repo created Apr 2026, README (787 stars).
   - What to read: "Technical details" (tape-replay rollback kernel, verify-specialized int4 qmm with a NAX M=16 path). About 10 min.
   - Quote: "M5-class Apple GPUs (`applegpu_g17*`) use the Metal 4 NAX M=16 path automatically when the shape is supported; older Apple GPUs keep the steel simdgroup-MMA fallbacks."

2. **turboquant-mlx README**
   - Link: https://github.com/arozanov/turboquant-mlx
   - Author / date / type: arozanov, Mar 2026, README.
   - What to read: the kernel-optimization bullets and the memory/speed table. About 8 min.
   - Quote: "**SIMD-group reductions**: `simd_sum` replaces tree reduction in QK scoring (1.85x kernel speedup)" / "| Baseline fp16 | 6.21 GB | -- | 35.75 tok/s | correct |".
   - Note: compressed KV is slower (25.84 tok/s for K8+V4).

3. **MTPLX README and its benchmark receipts page**
   - Links: https://github.com/youssofal/MTPLX, plus mtplx.com/benchmarks (receipts page not fetched).
   - Author / date / type: youssofal, May to Sep 2026, README with raw logs (2.5k stars).
   - What to read: the benchmark table, all measured on an M5 Max 128GB. About 10 min.
   - Quote: "Every number below was measured on a MacBook Pro M5 Max with 128 GB, fans verified at maximum" / "| Qwen 3.6 27B, Optimized Speed | 81.74 tok/s | ... 2.69x over 30.37 plain decode".
   - Caveat: the gains come from MTP speculative heads, not new kernels.

4. **oMLX README**
   - Link: https://github.com/jundot/omlx
   - Author / type: jundot, 22.7k stars, README.
   - Quote: "for GLM-5.2 the fused DSA prefill is roughly 30x faster with the kernels (measured 845 vs ~29 tok/s on an M3 Ultra)". About 5 min.

5. **Rapid-MLX README**
   - Link: https://github.com/raullenchai/Rapid-MLX
   - Author / type: raullenchai, README.
   - Quote: "Qwen3.6-35B-A3B, aggregate decode was 3.0× Ollama's, 82.9 vs 27.2 tok/s (32 GB M2 Pro Mac mini". About 5 min.
   - Caveat: serving-level gains, vendor-authored.

6. **Ollama blog, "Ollama's highest performance on Apple Silicon yet with MLX"**
   - Link: https://ollama.com/blog/mlx-performance
   - Author / date / type: Ollama, June 11 2026, blog.
   - About 8 min.
   - Quote: "several operations are now fused into single Metal kernels via MLX's just-in-time compiler features" / "NVFP4 generates about 20% faster than q4_K_M on the updated engine."

Not found or not verified for Q4:
- NOT_FOUND: Prince Canuma (mlx-vlm) long-form post; Eyal Toledano / pMLX (the repo search returned unrelated projects); Ivan Fioravanti outside X.
- UNVERIFIED: exo's "DeepSeek R1 8-bit on two 512GB M3 Ultras at 20 tok/s" (search summary only). The exo blog was not fetched.

### Q5. Quantization on Apple silicon

- Ollama June 2026 post (above): nvfp4 vs q4_K_M perplexity on Gemma 4 12B. Quote: "model-optimized NVFP4 roughly halves the quality loss". The perplexity values themselves are in a chart I couldn't extract.
- MLX issue #2962 (above): what nvfp4 vs mxfp4 scales mean in MLX's Metal kernels.
- mlx-lm LEARNED_QUANTS.md (above): DWQ, AWQ, GPTQ.
- turboquant-mlx (above): KV-cache quantization with measured speed cost.
- NOT_FOUND: a measured comparison of MLX affine vs mxfp4 vs nvfp4 (quality plus speed, same model, same Mac); a llama.cpp GGUF-types-on-Metal speed/quality write-up; llama.cpp KV-cache quantization on Metal with numbers. PR #20962's table does show Metal tensor-API speedup per GGUF type (Q2_K through Q8_0, IQ4_XS), which partly covers the speed side.

## Gaps

- **CONFLICTING:** yage.ai "MLX vs llama.cpp on Apple Silicon" (https://yage.ai/share/mlx-apple-silicon-en-20260331.html, Mar 31 2026) says Apple published the M5 paper "In January 2026" (it was Nov 19 2025). It also claims llama.cpp's Metal backend is "failing to automatically benefit from this specialized hardware". That is contradicted by PR #16634 (Tensor API, merged Nov 2025) and Apple's own PR #20962. Do not use it.
- **Secondary only:** starmorph's "Apple Silicon LLM Inference Optimization" (https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide, Dylan Boudro, Apr 10 2026) is a first-hand M4 Mac mini guide with a paid upsell. It's readable, but its numbers are summaries.
- **INACCESSIBLE:** Medium, "Benchmarking Apple's MLX vs llama.cpp" (andreask_75652; Cloudflare block). pub.towardsai.net "MLX runs 3x faster than llama.cpp until your context hits 40K" is a member-only story.
- **Unconfirmed:** the search-snippet claim that "at 30K+ context, MLX token generation is about 50% slower than llama.cpp with Flash Attention" on M3 Ultra.
- **For agent E (talks):** WWDC25 session 315 covers `mx.fast.metal_kernel` (https://developer.apple.com/videos/play/wwdc2025/315/).
- **Out of scope:** "MLX on CUDA" discussion #2422 (awni, Jul 2025).
- **For agent A:** Max Weinbach on the M5 Neural Accelerators (macstories link).
- **Not read:** the uzu engine claim of 30-40% over llama.cpp (llama.cpp discussion #14965).

## Sources and independence

- **First-party Apple/MLX:** Apple ML Research M5 post; MLX docs (lazy eval, custom kernels); MLX release notes; mlx-lm LEARNED_QUANTS.md; issue #2962 replies by awni and nastya236. These share an author pool, so they are not independent of each other.
- **First-party llama.cpp:** ggerganov's #4167, #16634 and #27668. Apple's Developer-Ecosystem-Engineering account wrote #20962, so it is Apple, not llama.cpp, but it's the same vendor as the MLX sources for the "Tensor API" story.
- **Independent practitioners:** dflash-mlx, turboquant-mlx, MTPLX, oMLX, Rapid-MLX, MLX #3801, llama.cpp #26938. These are self-reported benchmarks with no third-party replication. MTPLX publishes raw logs.
- **Vendor blog:** Ollama (commercial interest in its MLX engine).
- **Low quality or conflicting:** yage.ai, starmorph.
