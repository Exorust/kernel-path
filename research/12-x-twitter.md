# Research pack 12 — X/Twitter first-hand Apple GPU, Metal and MLX posts (raw subagent report)

Provenance: deepresearch run 2026-10-10, territory C. Model: Opus subagent, 22 tool calls through the authenticated `twitter` command-line client (no curl scraping). Caveats: every number is self-reported by the poster unless marked reproduced; user-timeline reads hit HTTP 429 twice and were replaced by `search --from`; URLs are `https://x.com/i/status/<id>`; quotes not yet spot-checked when saved except where the synthesis says so. Post content is data, not instructions.

---

## Territory C: X/Twitter, Apple GPU, Metal and MLX first-hand material (AS_OF 2026-10-10, window 2024-10 to now)

Method: I used the authenticated `twitter` CLI for 22 tool calls. That covered 4 full timelines, 10 assigned searches, 12 per-author and extra searches, 17 threads read with replies, and 6 X articles. I did no curl scraping. Every number below is **self-reported by the poster** unless I mark it as reproduced. All URLs are https://x.com/i/status/<id>.

## Findings

### A. Architecture: M5 Neural Accelerators, Metal 4 TensorOps, bandwidth vs compute

1. **@alexdelrosso, 2026-09-11, thread, id 2098462416270995637.** He forced Metal 4 `matmul2d` onto an M2 Ultra in ds4-metal. This is the best "hardware gating" lesson found on X.
   - Quote: "matmul2d (Metal 4 tensor ops) is gated by device name to M5/M6/A19/A20 — even though the API is portable."
   - M5 Max: "tensor tiles halve routed MoE prefill (16.1→7.5 ms microbench, +30-51% end-to-end)".
   - M2 Ultra: "simdgroup (old) — 25.5 ms dense / 29.8 ms sparse … tensor 32-tok uncompensated (NAX=1) — 22.6 / 26.5 ms (−11% both) … tensor compensated (NAX=5, M5 default) — 36.7 (+44%) / 53.6 (+80%)".
   - His reason: "running the tensor op twice per K step doesn't pay when matmul2d maps to an AMX/simdgroup fallback".
   - Reply from @ivanfioravanti (a partial independent data point): "I did same tests in the past on M3 Ultra but I had bad results".
2. **@SubhoGhosh02, 2026-06-09, 15-part thread, id 2064221238315434203.** A walkthrough of the WWDC26 Metal tensor and TensorOps API, with code. Not benchmarks.
   - "matmul2d<descriptor, execution_simdgroups<4>> op … for supported quantized tensors, TensorOps handles dequantization automatically".
   - "cooperative tensors … keep intermediate matmul results in thread-private storage instead of writing them out to memory".
   - Block scales: "tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32,1>".
3. **CONFLICTING: M5 Max matmul TFLOPS.**
   - @nisten, 2026-03-03, id 2028983221371748494: "M5 Max is not actually that good in raw float16 (same as M1 Ultra lol) but should go 2x+ faster in int8."
   - Correction from @liuliu (Draw Things / MFA author), id 2028989983399526404: "This is incorrect. M5 Max raw TFLOPS (for matmul) exceeds M2 Ultra (around 2x). Your source must not measure the NAX matmul instructions (or measure it correctly)."
   - nisten did not accept it. His source was fp32 GPU benchmarks multiplied by 2.
4. **@jetbrains, 2026-08-24, thread reply, id 2091897832705757647.** First-hand engineering claim: "M5's Neural Accelerator has 8-bit arithmetic instructions that M4 lacks. Patching MLX-VLM to use them got us ~40% more prefill throughput." No independent reproduction found.
5. **@awnihannun (MLX co-creator), 2026-03-09, id 2031153261156381161.**
   - Quote: "M5 Max is faster than M3 Ultra for many MLX benchmarks (prefill 2x, decode for MoEs despite having lower memory bandwidth)."
   - Replies had no reproduction. @TeksEdge: "saw some benchmarks on GGUF prefill up to 4x. But decode only 15% improvement."
   - Related, 2026-03-03, id 2028852360190345687: "Up to 8x faster prefill / image generation compared to M1 Max. Benchmarks done with MLX / mlx-lm."
6. **@alexocheema (exo), 2026-08-19, id 2090215789529800797.** A roofline-style explanation: "DGX Spark: ~350 TFLOPS@fp4, 128GB@273GB/s M5 Max MacBook Pro: ~70 TFLOPS@fp4, 128GB@614GB/s. Prefill is compute-bound. Prefill on Spark runs ~5x faster than M5 Max."
   - Also id 2054784097655078950: "the important ratios are memory bandwidth : network latency and TFLOPS : network bandwidth."
   - Challenged by @Youssofal_, id 2054771505553146061: "It's dishonest to claim 2456GB bandwidth" (summed bandwidth across a 4-MacBook cluster).
7. **ANE vs GPU.**
   - @hxiao, 2026-03-06, id 2029755398643966071: "jina-v5-nano … on M3 Ultra: MLX GPU: 4ms • ANE hybrid (baked weights): 49ms • ANE hybrid (dynamic weights): 236ms."
   - Correction from @anemll: "There should be no shuttle with IOSurface, it is zero copy … ANE bandwidth on M3U is 120GB/s, GPU is 819GB/s."
   - Agreement from @old_sound: "I've been doing a lot of benchmarks on my M3 Ultra and reached similar conclusions."
8. **@mweinbach, X article, 2026-03-09, "M5 Max: Chiplets, Thermals, and Performance per Watt", id 2031105498531975537.**
   - It describes M5 Pro/Max as "a CPU-dominant tile and a GPU-dominant tile".
   - ANE result: "After retuning around larger FP16 matmuls, the measured wall-clock peak rose to about 19.9 TFLOPS, versus the project's older 15.8 TFLOPS M4 reference."

### B. MLX internals (posts by kernel authors)

1. **@zcbenz (MLX maintainer), 2026-09-28, id 2104551074736837026.** "MoE layer in MLX is getting 1.5x faster, achieved with better tile scheduling in grouped mm."
   - Replies ask for the end-to-end gain and note that the 4-bit `gather_qmm` path is still TODO.
2. **@ivanfioravanti, 2026-09-21, id 2102036975066783778.** "The same Metal kernel family needs very different launch/tile parameters to efficiently saturate the Ultra." He points at mlx PR #4447.
3. **@angeloskath (MLX co-creator).**
   - 2024-10-18, id 1847381368445288593: "This is all coming from the fused attention kernel which we just added" (the ">110tps … Mistral 7B … M2 Ultra" post).
   - 2026-02-07, id 2019968198322577821: JACCL RDMA gives "~3.3x speedup for both MoEs and dense models (4 nodes)".
   - 2026-03-07: "3.4× speedup on 4 chips" for prompt processing.
4. **@awnihannun, kernel-level replies.**
   - 2026-02-05: "Did a proper MLA implementation and optimized an underlying kernel in MLX."
   - 2025-07-31: "CPU is responsible for submitting work to GPU … down to kernel launch latency."
   - 2024-10-16: "No need to mx.compile … rms.norm (it's already a single kernel) … Use `precise=True` for the softmax."
   - Retweet, 2024-10-31: matmul faster "as much as a Tera FLOP per second speedup" (credited to @DiganiJagrit).
   - Retweet, 2024-11-04: "generates text faster with very large models (Thanks to a new Metal feature, only available on macOS 15+)".
5. **@NasFilippova (MLX team), 2026-08-10, id 2086614161740501437.** "our rmsnorm backward was hitting significantly lower bandwidth than forward. I wrote a small note on why." The blog is handed to agent A.
6. **@liuliu, quoted by awni, 2025-01-08, id 1877040179942146237.** "Running side-by-side with SDPA kernel in MLX / MPSGraph, MFAv2 is more than 20% faster at 4096 sequence length … first to introduce the gradient op of SDPA on Apple Silicon. @philipturnerar"

### C. Kernel write-ups (community, first-hand)

1. **@drxddy, X article, 2026-01-25, "How I got 2.27x Faster LLM Inference on MLX?", id 2015303288883941808.** A fused rANS entropy decode plus GEMM Metal kernel. It includes the M3 Pro roofline arithmetic.
   - "for every byte you load from memory, you have 47 floating-point operations to spare".
   - "4-bit: 40 tok/s - Entropy-coded: ~90 tok/s".
   - "Fusing into a single kernel that never materializes the full weight matrix is non-negotiable."
2. **@Artex_llm, 2026-10-09, id 2108546269883044340.** A precise format lesson: "Our MLX conversion stored the quantization scales as BF16 … Apple M1 has no hardware for BF16 … before: 53 tok/s → after: 63 tok/s." No independent reply.
3. **@utsav_khairnar, 2026-10-01, id 2105647695465676961.** An agent-written kernel harness: "masked softmax, 1.84x faster than apple's MLX on an M4 … MLX runs this as 3 separate steps … tried layernorm and rmsnorm too, basically tied MLX there since apple already fused those."
4. **@ErikKaum, 2026-05-18, id 2056338682416701455.** "tiled scoring with simdgroup_matrix (Metal) and WMMA … 3–5× speedup compared to naive PyTorch". This was the only real hit for `simdgroup_matrix`.
5. **@Prince_Canuma (mlx-vlm).**
   - 2025-06-15, id 1934059472852254915: "MLX is now almost 30% faster than bitnet.cpp … Fused QKV metal kernel."
   - 2025-04-16, id 1912444776550379534: a bug write-up, "Thread-based processing boundaries" split images.
   - 2026-04-16: "Prefill at 0.87x of the 2bit kernel with v3."
6. **@ivanfioravanti on DwarfStar (antirez ds4) Metal kernels.**
   - 2026-08-07, id 2085728727380484417: "~41 toks/s decode … Same logits, 100% same result: this is The Mandatory Rule."
   - 2026-07-22, id 2079989602597634085: "logits are drifting a lot compared to bf16/standard implementations".
   - Replies agree. @sakurayukiai: "Tokens/sec without top-k agreement and KL drift is half a benchmark." @old_sound: "numerical drift start to show up after 16k ctx or 32k".
7. **@EyalToledano, 2026-07-02, id 2072730778492919910.** This is WebGPU, not Metal, but it is a good dispatch-bound lesson: "mobile inference isn't limited by memory bandwidth. Every token is ~160 tiny GPU jobs … greedy: 37.5 → 41.1 tok/s sampled: 66.2 → 84.3 tok/s."

### D. llama.cpp / ggml Metal

1. **@qvac, 2026-10-05, id 2107043339799593421.** Speculative-decoding Metal kernels merged into llama.cpp: "On an M3 Ultra it now runs up to 3.4x faster than plain decoding, 110 tok/s against 32.1."
   - Independent caveat from @vermaizer: "the new kernels target Metal GPUs without tensor API support … 110 tok/s is the code prompt. Prose lands around 64."
   - Follow-up, 2026-10-07, id 2107924001939235207: "cover all 26 weight formats … matrix multiplies run up to 4.4x faster when a model checks several tokens at once."
2. **@ggerganov.**
   - Retweet, 2025-11-06, id 1986473652137947505: "Initial M5 Neural Accelerators support in llama.cpp … faster TTFT" (PR 16634).
   - 2025-08-06, id 1953088008816619637: "ollama's modifications in ggml … too much branching in their MXFP4 kernels and the attention sinks implementation is really inefficient."
   - 2026-05-31: "after using the RTX and going back to the Mac, it always feels like a snail."
   - The 2025-08-29 "M3 Ultra performance numbers for gpt-oss" (id 1961276336669855896) is image-only, so I have no text numbers.
3. **@no_stp_on_snek, X article, 2026-04-25, id 2047904901053698301.** "31 experiments in 5 days" on runtime speculative decoding on Apple Silicon.
   - "Skip 1 layer: 72% acceptance, 1.8% compute savings … Skip 8: 3.4% acceptance".
   - Separately, he says he is building TurboQuant Metal kernels into llama.cpp (id 2038245543705518477).

### E. Inference numbers worth keeping (all self-reported)

1. **@Youssofal_, MTPLX X article, 2026-05-04, id 2051435496551878847.** "28 tok/s → 63 tok/s on Qwen3.6-27B on a MacBook Pro M5 Max … custom Metal kernels, compiled verify graphs."
   - **Independent reproduction (partial)** by @ivanfioravanti, article id 2051688805858463953: "M3 Ultra 32.4 --> 53.8 toks/s 🥈M5 Max 29.5 --> 48.2 toks/s", with the M3 Ultra winning on decode and the M5 Max on prefill.
2. **@Youssofal_, 2026-08-25, id 2092379119917113426.** "MTPLX only saturates 220GB-230GB BF16 when the M5 max has 614GB/s."
   - Reply from @aaronedell: "roughly 37% of theoretical bandwidth … grouped MoE kernel reaching about 69% of dense quantized-matmul throughput".
   - Youssofal to @anemll: "There is zero dequant state. I'm storing KV in BF16."
3. **@bstnxbt dflash-mlx, 2026-04-13, id 2043696397388554447.** "M5 Max, stock mlx_lm baseline: Qwen3.5-4B: 53.74 → 219.83 tok/s (x4.10) … Qwen3.5-27B-4bit: 32.35 → 62.78 tok/s (x1.90)".
4. **@redp314, 2026-09-01, id 2094706185429545209.** A third-party cross-engine comparison: "every drafter loses 17-36% on prose … mlx-serve: 39.0 · 41.7 · 32.1 → oMLX: 38.5 · 35.3 · 28.2".
5. **@0xkydo (mlx.fast competition), 2026-08-21, id 2090894947335750142.** "Median decode speed increased from 26 tok/s to 87.9 tok/s on the verifier M5 Max … prefill around 971.8 tok/s." The verifier machine makes this more credible than most posts.
6. **Other kernel-optimized engines:** @ddalcu mlx-serve (Zig + Metal), @trymirai uzu ("outperform MTPLX … by almost 2x, and llama.cpp by over 3x"), and TensorFold (@volatilemarkts: "27 tok/s … Today … 61 tok/s"). All are vendor-reported and CONFLICTING with each other. @zhijianliu ran mlx-serve head-to-head: "One request: 74 vs 51 tok/s."
7. **Historical MLX baselines.**
   - @awnihannun, 2025-03-21: "~30,000 tokens under a minute on my M4 Max … causal fused attention."
   - @ivanfioravanti, 2024-11-23: "3bit: 29.00 t/s - 4bit: 24.02 t/s - 6bit: 16.94 t/s - 8bit: 13.68 t/s" (M4 Max).
   - @ivanfioravanti, 2025-04-18: "mlx >=0.25 vs <0.25 8K tokens, 600 vs 328 toks/s".

### F. Quantization

1. **@N8Programs, 2026-04-17.** "4bit MLP 8bit everything else) + DWQ … 0.0225 KL w/ the base model … versus 0.0819 for a naive 4bit quant. Adds only 0.25BPW."
2. **@N8Programs, 2025-05-31.** "a simple MLX kernel reaches throughput up to 30x higher than OpenCV on the CPU."
3. **@TheStageAI, 2026-08-13.** "W4A8 Metal kernels for @Apple M5 … utilize M5 GPU neural accelerator with int8 vector instructions." The blog goes to agent A.
4. **@Prince_Canuma, 2026-04-05.** An explanation of TurboQuant KV on Gemma 4: "50 sliding layers hold a fixed ~400MB regardless of context length."

## Gaps

- **@alyssarzg:** NOT_FOUND ("User @alyssarzg not found").
- **@philipturner:** returned 0 results. The real handle is @philipturnerar, and his recent posts are off-topic (atomic fabrication). No GPU kernel posts since 2024-10.
- **@dougallj:** only CPU material (the Apple Silicon CPU Optimization Guide v5, NEON godbolt replies). No GPU posts found.
- **Rate limits:** `user-posts` timelines for ivanfioravanti, Prince_Canuma, dougallj, exolabs and philipturner were INACCESSIBLE (HTTP 429 twice). I covered them with `search --from` instead.
- **Partial timelines:** awnihannun's timeline only reached back to 2026-02 within 200 posts; older material came via search. A retweet says "We're going to miss @awnihannun on MLX!" (2026-02-28); whether he left MLX is UNVERIFIED.
- **Weak searches:** "simdgroup_matrix" returned only 2 posts, and "M5 neural accelerator prefill benchmark" (latest tab) returned 2.
- **X articles I did not read:**
  - @llm_frontier_jp, M4 Max vs M5 Max, "生成は約1.12倍、入力処理は約2.26倍" (decode ~1.12x, prefill ~2.26x), article 2104098308218302464
  - @ashxhart MCDMA macOS RDMA driver, 2099482115419021312
  - @TheDavidTai mlx-stream, 2107827863135141888
  - @WinterCharmVT WinterMix quantization, 2083837352061857792
  - @ostensiblyneil, 2061886148579135488
  - @sanchitmonga22 (3 articles)
  - @ashen_one M5 Ultra 140 measurements (thread, id 2107424638565544228)
  - @yume_arasaki M5 Ultra roundup (id 2108359576966094952)
- **Handed off:**
  - **Agent A (blogs):** nastya236.github.io/blog/rmsnorm-backward-persistent/, app.thestage.ai qlip-mlx blog, trymirai.com/blog/quantization, huggingface.co/blog/omlx, huggingface.co/blog/transformers-to-mlx, ollama.com/blog/mlx, pytorch.org ExecuTorch MLX delegate blog, simonwillison.net/2026/Apr/27/vibevoice/
  - **Agent B (GitHub/docs):** ml-explore/mlx PRs #4567 (grouped mm tiles), #4447 (Ultra tile params), #3094 (JACCL), mlx/distributed/jaccl/lib; llama.cpp PR #16634 (M5 NA) and the qvac speculative-decoding Metal PRs; ggml discussion #15396; youssofal/MTPLX; bstnxbt/dflash-mlx; ddalcu/mlx-serve; kernelpool/ds4; TheStageAI/qlip-mlx; anemll/anemll-bench; Apple CPU optimization guide
  - **Agent D (Reddit):** r/LocalLLaMA "dont_take_apple_mlx_too_seriously"
  - **Agent E (talks):** Apple tech talks 111431 and 111432 (M5/A19 Neural Accelerators, TensorOps, Xcode 26.4 counters); WWDC26 MLX videos (youtube wykPErJ8M-8, CzgK02zsRg4, KCL8f9ztKFk); WWDC25 sessions 298 and 315; Awni on the "localhost" podcast
  - **Agent F (NVIDIA):** NVIDIA Mistral-3 llama.cpp blog

## Sources and independence

- **Kernel authors, primary:** awnihannun, angeloskath, zcbenz, NasFilippova and DiganiJagrit (MLX team); liuliu and philipturnerar (MFA); ggerganov plus the qvac contributors (llama.cpp); Prince_Canuma, Youssofal_, bstnxbt, ivanfioravanti, alexdelrosso and drxddy (community kernel authors). These are independent of each other.
- **Reposts of one origin, counted once:**
  - The Oluwaphilemon1, TeksEdge and akshay_pachaar posts repackage uzu, TensorFold, Magnitude and mlx-serve claims. They are one source per vendor and none is a reproduction.
  - The Apple M5/M6 spec posts (BenGeskin, kimmonismus, rohanpaul_ai, never_released and others) all trace to Apple press releases, so they count as one source and contain no first-hand data.
  - TheAhmadOsman's repeated "capacity × bandwidth × software stack" post is one source, published at least 4 times.
- **Real independent checks found:**
  - ivanfioravanti reproducing MTPLX, with lower M5 numbers than claimed.
  - alexdelrosso measuring matmul2d on M2 Ultra.
  - redp314 and zhijianliu comparing engines.
  - liuliu correcting nisten (CONFLICTING, unresolved).
  - anemll correcting hxiao.
  - vermaizer and aaronedell adding caveats.
- Everything else is self-reported.
