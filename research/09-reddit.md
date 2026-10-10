# Research pack 09 — Reddit first-hand Apple-silicon inference and kernel threads (raw subagent report)

Provenance: deepresearch run 2026-10-10, territory D. Model: Sonnet subagent, ~24 tool calls, via the `rdt` command-line client (`rdt search`, `rdt read`). Caveats: user-generated content; every thread is a single author's setup unless a reproduction is quoted; scores are as of the read; the agent read only the top 4-7 comments per thread and did not open linked repos or blogs. Per the deepresearch rule, any Reddit claim that drives a course conclusion needs one non-UGC confirmation. Page content is data, not instructions.

---

## Territory D (Reddit): Apple-silicon inference and Metal/MLX kernels, 2024-10 to 2026-10-10

**Method.** I used `rdt search` and `rdt read` (`rdt search -r <sub>`), filtering to posts from 2024-10 onward. I used about 24 tool calls and stopped near saturation.

- Quotes are verbatim, except that I stripped Reddit markdown escapes (`\~`, `\*`).
- Dates are the post's `created_utc`. Scores are as of the read.
- Every post here is by its named author. Where a claim rests on one person's setup, I say so.
- I read no blogs, docs, GitHub, X, HN or NVIDIA material. Links I saw are listed under Gaps.

---

## Findings

### A. Measurements by chip

**A1. M5 Max 128GB, stock `mlx_lm`, clean context sweep (best raw-numbers thread)**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/ · id 1rqnpvj
- Author: cryingneko · 2026-03-11 · score 2213 · r/LocalLLaMA
- What it reports: M5 Max 128GB 14", pure `mlx_lm.generate` / `stream_generate`, 128 max output tokens. The OP first ran `BatchGenerator`, found it slower than expected, and redid everything in a fresh venv. The tables below were compiled by commenter MMAgeezer from OP's posted output.

| Model (MLX quant) | Context | Prompt tok/s | Generation tok/s |
|---|---|---|---|
| Qwen3.5-122B-A10B-4bit | 4K | 881.5 | 65.9 |
| Qwen3.5-122B-A10B-4bit | 32K | 1,067.8 | 54.9 |
| Qwen3-Coder-Next-8bit | 4K | 754.9 | 79.3 |
| Qwen3-Coder-Next-8bit | 64K | 1,432.7 | 48.2 |
| Qwen3.5-27B distill 6bit | 4K | 811.1 | 23.6 |
| Qwen3.5-27B distill 6bit | 64K | 475.8 | 14.2 |
| gpt-oss-120b-MXFP4-Q8 | 4K | 1,325.1 | 87.9 |
| gpt-oss-120b-MXFP4-Q8 | 32K | 2,537.4 | 64.5 |

- Quote (OP): "I initially ran the tests using BatchGenerator, but the speeds weren't quite what I expected. I ended up setting up a fresh Python virtual environment and re-running everything with pure mlx_lm using stream_generate"
- Confirming comment: OP: "I tested again with pure mlx_lm. I think it's safe to say these are the properly measured speeds." (score 144)
- Flag: single author, single machine. Methodology is plain `mlx_lm`, which makes it the cleanest baseline here.

**A2. M5 Max, llama.cpp Metal KV-cache sweep to 1M context**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1sy7srk/ · id 1sy7srk
- Author: Defilan · 2026-04-28 · score 41
- What it reports: MacBook Pro M5 Max 128GB, TheTom's TurboQuant Metal fork of llama.cpp, Qwen3.6-35B-A3B Q8. Settings: `llama-bench`, 3 reps per cell, flash-attn on, depths 0 to 1M, f16 / q8_0 / turbo3 / turbo4 KV cache.
- Generation tok/s at 128K depth: f16 44.4, q8_0 40.7, turbo3 36.0, turbo4 37.7.
- Prompt-processing tok/s at 128K depth: f16 321, q8_0 245, turbo3 253, turbo4 206.
- At 1M depth only turbo3 fits: 6.5 gen and 30 PP tok/s.
- Quote: "At 128K the 3-bit cache catches up to the 8-bit cache on prefill (turbo3 253 vs q8_0 245). Smaller cache means less bandwidth pressure during attention. The bandwidth-bound regime favors turbo3 once contexts grow past about 100K on this hardware."
- Correcting comment: milpster (34): "Cool, but no perplexity calculations?" GCoderDCoder (9) also asks about output quality at each level. The thread reports speed only, with no quality check.
- Flag: single author. The depth sweep took 8 hours of wall-clock time.

**A3. M3 Ultra 512GB vs dual DGX Spark, Qwen3.5-397B**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1s4lmep/ · id 1s4lmep
- Author: trevorbg · 2026-03-26 · score 429 · 242 comments
- What it reports: MLX 6-bit, 323GB model. 30 to 40 tok/s generation. Prefill "30+ seconds on a big system prompt". The dual Sparks (vLLM TP=2, INT4 AutoRound) reach 27 to 28 tok/s.
- Quote: "The biggest selling point is memory bandwidth at roughly 800 GB/s. That bandwidth is what makes token generation feel smooth on such a massive model in a single box."
- Quote: "The weakness is raw compute. Prefill is slow (30+ seconds on a big system prompt with tool definitions)"
- Flag: single author. The comparison is MLX on Mac against vLLM on CUDA.

**A4. M3 Ultra 512GB, DeepSeek R1 671B Q4 on MLX (early, 2025-03)**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/ · id 1j9vjf1
- Author: ifioravanti · 2025-03-12 · score 619
- What it reports: "18.43 tokens/sec" generation on a p5js prompt.
- The top comment, from tengo_harambe (146), asks for prompt-processing time on a prompt of at least 4000 tokens. I did not read the follow-up. Treat the post as decode-only and single-author.

**A5. M5 vs DGX Spark vs Strix Halo vs RTX 6000**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1tfzsd6/ · id 1tfzsd6
- Author: Signal_Ad657 · 2026-05-17 · score 855
- What it reports: standardized tests published to a GitHub repo. I did not read the repo.
- Quote: "Tokens per second per piece of hardware follows that math and curve pretty well."
- Best correction (ttkciar, 277): "When a model and its context fits in the RTX6000, the RTX6000 will outperform M5. The more the model and context overflows the RTX6000's VRAM, the worse it will perform, whereas the M5's performance will hold steady."
- Flag: the numbers live in the repo, which I did not read. Thermal claims such as "cruised in the 80c range" are single-author.

### B. MLX internals and custom kernels

**B1. DFlash speculative decoding native in MLX (M5 Max): an optimization log with failures**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1simszl/ · id 1simszl
- Author: No_Shift_4543 · 2026-04-11 · score 329 · r/LocalLLaMA
- What it reports: M5 Max 64GB. A draft model proposes 16 tokens per cycle and the target verifies them in one pass.

| Model | DFlash tok/s | Baseline tok/s | Speedup |
|---|---|---|---|
| Qwen3.5-9B bf16 | 85 | 26 | 3.3x |
| Qwen3.5-27B 8bit | 35 | 14 | 2.5x |
| Qwen3.5-27B 4bit | 44 | 24 | 1.9x |

- Quote (kernel lessons): "Custom Metal kernels (batched-GEMV, fused gated SiLU, custom SDPA) all came back 0.5 to 0.8x slower than stock MLX steel GEMM. Ended up reverting all of them."
- Quote: "Qwen3.5-9B uses head_dim=256, which MLX's steel_attention didn't support. A 2-line patch unlocked the fast SDPA path."
- Quote: "Restructured the pipeline from 2 GPU→CPU syncs per cycle to 1. At 80+ tok/s each sync costs ~0.5ms."
- Quote: "Verify cost is almost flat from 4 to 16 tokens (57ms vs 59ms). Weight loading dominates, not token count."
- Quote: "On quantized models, the optimization landscape flips: the draft (bf16) becomes slower than the verify (int4/int8)."
- Follow-up: https://www.reddit.com/r/LocalLLaMA/comments/1skesyq/ (id 1skesyq, 2026-04-13, score 114). The OP rewrote the benchmark methodology, fixed numerical issues and open-sourced it; the title now reads "4.1x on Qwen3.5-9B".
- Independent reproduction in that follow-up: putrasherni on an M4 Max 128GB measured Qwen3.5-27B DFlash at 1.51x to 2.42x. Q4: 27.30 to 43.05 tok/s (1.59x). Q8: 15.15 to 36.07 tok/s (2.42x). Acceptance was about 88 to 90%.
- Mixed reproduction: Its-all-redditive (M5 Max 128GB, 4B model) reports "~200 t/s generation vs the expected ~150 t/s".
- Note that the 3.3x headline in the first post was later revised, and the independent M4 Max numbers are lower. Reproduction is partial, not full.

**B2. Fused-kernel optimization of a model on M3 Ultra (ds4 fork): decode and prefill by memory-bandwidth fraction**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1wbkpnw/ · id 1wbkpnw
- Author: IngeniousIdiocy · 2026-09-09 · score 126
- What it reports: GLM-5.3-Flash Q4 on an M3 Ultra, measured against the chip's own measured memory-bandwidth and matmul ceilings.
- Quote: "Main ds4 was single-stream serially decoding GLM-5.3-Flash at about 59 percent of the M3 Ultra's measured memory bandwidth. We set an 80 percent target. The big weight-streaming kernels were already efficient, but dozens of small kernels sat between them, each paying latency costs while much of the GPU was idle. We fused that work into larger dispatches... Result: 29 → 40 t/s at short context, 24 → 38 at 62k, fewer Metal kernels per token, and about 81 percent of the measured bandwidth ceiling."
- Quote (prefill): "We batched more work together, reused prepared weights, widened the loads, and merged passes over the same data. Result: 366 → 550 t/s, a 50% improvement... moving the whole pipeline from roughly 51 to 72 percent of the chip's measured matmul ceiling."
- Quote (long context): "End to end at 300k: stock 21.6 t/s, ours 37.4."
- Comment: niacolhealth (5) on the speculative drafter: "Most speculative decoders never check whether they're earning their keep." This is praise, not correction.
- Flag: single author. The README is on GitHub. The author says the branch is M3-Ultra-only.

**B3. Writing M1-native Metal kernels so a bf16/MPP-dependent engine runs on M1 Max**
- URL: https://www.reddit.com/r/LocalLLM/comments/1woq7cd/ · id 1woq7cd
- Author: Erp4759 · 2026-09-23 · score 194 · r/LocalLLM (cross-posted to r/oMLX as 1wovgy5)
- What it reports: M1 Max 32 GPU cores, 64GB. Qwen3.8-27B 4-bit rises from 18.9 to 39.1 tok/s on npanj's 5-prompt benchmark, against 60.7 tok/s on npanj's M5 Pro.
- Quote: "Splash's kernels rely on bfloat16 arithmetic, which the M1 does not have, and on Apple's MPP matrix library, which is slow on this GPU. So I wrote my own kernels for those parts."
- Quote: "each 4-bit weight is turned into an exact half-precision number with a small bit trick, each activation becomes an exact fp32 number, and the M1's matrix instructions multiply and add them in fp32 without leaving registers."
- Quote: "Apple rates its GPU at 10.4 TFLOPS FP32, my microbenchmark gets 8.6-9.9 TFLOPS out of its matrix instructions, and its 400 GB/s memory bandwidth is more than an M4 Pro's 273 GB/s. What held it back was software built for newer chips."
- Confirming comment: tragdor85 (7) says it is "Certainly faster than OMLX and MTPLX. And more consistent."
- Part 2: https://www.reddit.com/r/LocalLLM/comments/1wqngu9/ (id 1wqngu9, 2026-09-26, score 67). Attention and MoE expert layers rewritten. The 35B-A3B model goes from 99 to 144 tok/s. The post claims "Attention itself: 1.85-1.95x faster per layer".
- Quote from Part 2: "Prefill is where MLX is still ahead, especially on long prompts."
- Flag: single author. Many commenters report their own M1 Max speeds, which is partial confirmation.

**B4. W8A8 activation quantization as custom Metal kernels registered as MLX primitives (M5 Pro)**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1tn2p61/ · id 1tn2p61
- Author: Enough-Astronaut9278, who says they work on inference tooling at Mininglamp AI · 2026-05-25 · score 0
- What it reports: M5 Pro 64GB, 307 GB/s, 4516-token context. Prefill goes from 2.839s (W8A16, MLX) to 2.519s (W8A8, "Cider"). Decode is 80.1 versus 79.5 tok/s.
- Quote: "At M=4096 the per-channel path runs 1.84x faster than W8A16 on the same shape."
- Quote: "INT8 TensorOps only compile on M5 and above. pip install on M4 still works, just falls back to the regular path."
- Flag: single author with a vendor interest. The headline prefill gain is only 11%. The author supplied perplexity numbers (Qwen3-8B: FP16 9.73, W8A8 per-channel 9.76) after a commenter asked about accuracy.

**B5. Stage quantized weights as FP8 or int8 before the matmul (M5 / "M6" matrix path)**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1x13fo2/ · id 1x13fo2
- Author: Brilliant-Hall1387 · 2026-10-08 · score 9
- What it reports: modified MLX quantized matmul (QMM) that stages weights to int8 or FP8 instead of FP16.
- Quote: "Regular MLX QMM (quantized matmul) stages operands (dequant) to FP16 before matrix multiplication. But if you modify MLX to stage to FP8 instead, you can use the 2x faster FP8 matrix path on the new Apple Silicon M6 hardware!"
- Quote: "Benefits apply to prefill (+40% prefill Qwen3-8B or +50% Qwen3.8-27B). Decode is bandwidth limited so not much difference."
- Quote: "worst-case perplexity increase under ~1%."
- Flag: single author. The poster discloses "the blog is from Precisit, where I work". The patch is an experimental fork, not merged upstream (mlx#4627). The FP8 path requires an M6 and macOS 27. I cannot verify that hardware. The blog and fork belong to agents A and B.

**B6. Hand-written Metal kernels in Mojo, GPT-2 training on M4 Max, with MLX as the baseline**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1v3hfy5/ · id 1v3hfy5
- Author: ulmentflam · 2026-07-22 · score 11
- What it reports: M4 Max, B=4, T=1024, GPT-2 124M, six arms interleaved. MLX bf16 gets 10,077 tok/s, llm.mojo bf16 8,138, PyTorch MPS fp32 4,930.
- Quote: "Yes, MLX wins. Apple's own framework is 1.24x faster than my bf16 path... The Metal bf16 matmul I ride runs at only ~1.1x its fp32 speed, while MLX's bf16 uses the tensor cores for ~2x."
- Quote: "the M4 Max throttles after about 8 seconds of sustained GPU load (I watched MPS step times climb from ~877 ms to 1500 to 2500 ms within a few steps)."
- Quote: "Casting threadgroup pointers to the generic address space silently reads device memory (attention softmaxed over all-zero scores and produced uniform weig[hts])". The source text breaks off at "weig", so I completed it from the next fragment.
- Quote: "the scalar flash-attention kernels tuned for NVIDIA ran at under 1 percent of FLOP peak on Apple GPUs, so GEMM-decomposed attention was 8 to 10x faster."
- This is the strongest "explanation by someone who did the work" for a reading course on Apple GPU kernels. Flag: single author, with tests listed as AI-assisted but kernels hand-written.

**B7. Metal kernel work for SSD-streamed MoE: autoresearch loop on M5 Max**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1s7g8ov/ · id 1s7g8ov
- Author: Equivalent-Buy1706 · 2026-03-29 · score 164
- What it reports: Qwen3.5-397B-A17B Q3-GGUF experts, streamed from SSD on a pure C/Metal engine (building on flash-moe by Dan Woods and Anemll's fork). Result: 10.61 → 20.34 tok/s decode and 5.52 tok/s prefill.
- The main steps were 16 IO threads, temporal expert prediction (+55%), Q3 experts, and pre-encoding the GPU command buffer one step ahead (CMD2). A fused Q/K/V kernel added about 0.8 tok/s.
- Quote: "27% of experts activated for token N get activated again for token N+1. Prefetch them during GPU compute so the SSD read is already done when the next token needs them. This dropped expert I/O from 56% of per-token time to nearly nothing."
- Quote: "SSD I/O was the bottleneck, then GPU encoding overhead, then projection kernels. Classic shifting bottleneck problem."
- Quote: "NAX/ANE offloading: tile padding overhead cancelled every gain"
- Correcting comments: val_in_tech (53): "Why pretend. 4t/s prefill = utterly useless." AXYZE8 (6): "The math ain't mathin". That comment points at the claim of a 209GB model "4x larger than the 128GB RAM" (209/128 is about 1.6x).
- Flag: single author. The post is Claude-assisted by the author's own statement. Treat the size claim as UNVERIFIED.

**B8. MoE SSD streaming on a Mac mini M5 64GB (open question, with numbers)**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1wy5fma/ · id 1wy5fma
- Author: turtleninja99 · 2026-10-05 · score 8
- What it reports: 17.5 tok/s decode, 390 tok/s prefill, hot-cache hit rate 75%, lookahead fetch accuracy 72%.
- Quote: "The decode part 27% of the time is still the gpu waiting for experts to stream in from the ssd"
- Comment (sdroege_, 7): "You could train a small model to predict which experts will be used ahead of time". Another comment (PinkysBrein, 3) suggests distilling a model whose gate uses the layer n-2 output.
- This independently shows the same bottleneck as B7. Single author, partly a question.

**B9. Dissenting data point on kernels**
- A kernel-writing report on a Mac Studio 512GB: https://www.reddit.com/r/LocalLLaMA/comments/1vje00y/ (technaturalism, 2026-08-08, score 24). The post says a coding model wrote a custom IQ1_0 Metal kernel in about 50 minutes, giving "About 4 t/s decode and 20 prefill".
- Single author, and it is a model-generated kernel. A commenter (HVACcontrolsGuru) points to their own MLX kernel repo, "hyperion". That repo link goes to agent B.

### C. MLX vs llama.cpp

**C1. GGUF prefill beats MLX on long context, M3 Ultra 512GB**
- URL: https://www.reddit.com/r/LocalLLaMA/comments/1rwaq47/ · id 1rwaq47
- Author: BitXorBit · 2026-03-17 · score 22
- Quote: "unsloth/qwen3.5 models prompt processing is way way better than MLX on large context and the bigger the size - the gap getting bigger. tokens generation? unlike llama.cpp that keeps stable TG, on mlx the TG decrease with the size of the context window."
- Correction (Ok_Technology_5962, 18): "First error is using lmstudio with mlx. They dont have caching enabled for mlx. Try oMLX". This changes the finding: part of the prefill gap is a prompt-cache issue in LM Studio's MLX runner, not the MLX kernels.
- Confirming (the_real_druide67, 5, M4 Pro 64GB, Qwen3.5-35B-A3B): "On short context (code prompts), MLX via LM Studio gives me ~80 tok/s vs ~30 tok/s with Ollama GGUF — MLX is 2.6x faster for generation. But on large context fills, the prefill time tells a different story: MLX TTFT ~14s vs GGUF TTFT ~4s."
- Flag: both measure through a stack (LM Studio / Ollama), not raw kernels. This is CONFLICTING with the Splash M1 thread (see Gaps).

**C2. KV-cache quantization on llama.cpp Metal.** See A2 (1sy7srk).

### D. Quantization, speculative decoding, other

- Speculative decoding is not free on Mac: https://www.reddit.com/r/LocalLLaMA/comments/1voh7oo/ (id 1voh7oo, 2026-08-14, score 2). The OP says "3 out of 5 speculative decoding configs tested were SLOWER than plain decoding on a Mac" and "Best case was 1.61x at K=6". A commenter, Gregory-Wolf, notes the cited paper is about RL ("REASONMAXXER"), so the link between post and paper is weak. Treat as UNVERIFIED and low-signal. The B1 finding that 4-bit changes the draft/verify balance is the better evidence.
- A claimed iPhone-as-second-GPU prefill split (a Mac and a phone each holding layers) is at https://www.reddit.com/r/LocalLLaMA/comments/1wvz1ex/ (id 1wvz1ex, 2026-10-02, score 2332, by StayLameBro). It reports 24GB M4 Pro prefill gains: "8k context: Mac alone 132 tok/s → Mac + iPhone 177 tok/s (+35%)" and "16k context: Mac alone 109 tok/s → Mac + iPhone 157 tok/s (+44%)". The same post claims "The A19 Pro's GPU has matrix units (Metal 4 tensor ops), and they make the phone's half 2.4x faster than the same phone without them." The OP added a disclaimer that the on-phone figures covered only the layers it holds. Single author; the score is popularity, not verification.
- Other MLX stacks, titles only: MTPLX (https://www.reddit.com/r/LocalLLaMA/comments/1t3zuvy/, YoussofAl, "2.24x faster TPS"), an MLX tool reporting "464 tok/s on M4 Max" (https://www.reddit.com/r/mlxAI/comments/1qelduy/, waybarrios), and the Splash engine.

### E. Subreddit and sub-coverage notes
- r/mlxAI exists: top posts are small (scores 5 to 28). They are mostly project announcements, with little first-hand kernel discussion.
- r/MachineLearning: only https://www.reddit.com/r/MachineLearning/comments/1seexk0/ (id 1seexk0) is relevant. It is a question thread. Best answers: sshkhr16 (19): "MLX: great / JAX/PyTorch: not so much"; KingRandomGuy (9): "The original jax-metal plugin is unmaintained, unfortunately." Low-signal opinion with no numbers.

---

## Gaps

**Subreddits and searches with nothing usable**
- r/CUDA: "MLX Metal" returned NOT_FOUND.
- r/hardware: "Apple Silicon inference" returned only news items. NOT_FOUND for first-hand measurements.
- r/GraphicsProgramming: only path-tracer and rendering projects, plus a cross-post of an MLX linear-algebra library announcement. NOT_FOUND for kernel explanations.
- r/apple: https://www.reddit.com/r/apple/comments/1s8m3wz/ (Ollama and MLX, score 1035) is a news repost. Skipped as a vendor/news repost.
- r/macapps: not searched. NOT_FOUND by omission.
- The "flash attention Metal" and "unified memory bandwidth" searches on r/LocalLLaMA were mostly off-topic. Only B6 and the A2 sweep were on target.
- Using `-s top` sorted by popularity brought off-topic hits; `-s relevance` worked better.

**Leads identified by title only, NOT read (UNVERIFIED)**
- https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/ (MTP, score 1238; platform not confirmed as Mac)
- https://www.reddit.com/r/LocalLLaMA/comments/1rzkw4x/ (M5 Max 128G performance tests)
- https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/ and https://www.reddit.com/r/LocalLLaMA/comments/1s4bzo2/ (TurboQuant llama.cpp)
- https://www.reddit.com/r/LocalLLaMA/comments/1vt0g7a/ (ANE+GPU prefill)
- https://www.reddit.com/r/LocalLLaMA/comments/1s1t9on/ and https://www.reddit.com/r/LocalLLaMA/comments/1szaw8n/ (oQ and MLX KLD comparison)
- https://www.reddit.com/r/MacPro2019LocalAI/comments/1u7z9jw/ (Custom Metal Flash Attention on Intel Macs)
- https://www.reddit.com/r/mlxAI/comments/1wbu5gm/ (VeloxQuant-MLX)
- https://www.reddit.com/r/LocalLLaMA/comments/1hne97k/ (DeepSeek-V3 on an M4 Mac mini cluster, 2024-12)

**Contradictions**
1. Custom kernels versus stock MLX. B1 found custom batched-GEMV, fused-SiLU and SDPA kernels came back "0.5 to 0.8x slower than stock MLX steel GEMM". B2 and B3 report large wins from fused or custom kernels. The likely difference is the baseline: B1 competes with MLX's tuned steel GEMM on the same ops, while B2/B3 beat engines that were not tuned for those shapes or that lacked bf16 or matrix support. This is my reading, not stated in the threads. CONFLICTING.
2. MLX versus llama.cpp prefill. C1 says llama.cpp (GGUF) prefill is far better on large context. B3 part 2 says "Prefill is where MLX is still ahead". The workloads, runners and quants differ, and C1 had a missing prompt cache in LM Studio. The two threads do not agree on who wins long-context prefill. CONFLICTING.
3. The DFlash 3.3x claim (B1) was later re-measured as 4.1x for 9B in a rewrite, while independent M4 Max runs show 1.5x to 2.4x for 27B. These are different models and chips, so no contradiction is confirmed, but the headline number is not reproduced for 27B.
4. B7's "4x larger than 128GB RAM" is flagged by a commenter. UNVERIFIED.
5. Vendor-origin: the Splash engine numbers (B3 and the 1wk9hze thread) come from Inco's own announcement and from Splash-friendly benchmarkers. The 1wk9hze thread (https://www.reddit.com/r/LocalLLaMA/comments/1wk9hze/, ResearchCrafty1804, score 224) is a vendor repost with no quantization given; a commenter points out "q4_0 btw". I excluded it.

**Limitations of my coverage**
- Search results show only the top 25 per query, and I did not paginate.
- I read only the top 4 to 7 comments per thread by score, so some corrections may be missed.
- I did not read linked blogs or repos.
- I could not verify the hardware names (M5, "M6", macOS 27, A19 Pro) or model names (Qwen3.8, GLM 5.3) beyond what the posts state. I treated them as stated.
- Some posts are bot-featured ("Your post is getting popular and we just featured it on our Discord!"). Ignore that automated comment.

**Stray links for other agents**
- For agent B (MLX docs and GitHub): https://github.com/precisit/mlx/tree/staged-8bit-qmm (mlx#4627); `steel_attention` head_dim=256 patch (B1); https://github.com/jscott3201/hyperion (MLX kernels).
- For agent A (blogs): https://precisit.com/en/blog/apple-matrix-formats/
- For agent B (GitHub): https://github.com/incoai/splash, https://github.com/npanj/splash-plus, https://github.com/danveloper/flash-moe, https://github.com/Anemll/flash-moe, https://github.com/TheTom/llama-cpp-turboquant, https://github.com/Mininglamp-AI/cider, https://github.com/Light-Heart-Labs/MMBT-Messy-Model-Bench-Tests, https://github.com/Aryagm/dflash-mlx, https://github.com/skeggsguy/Flash-next-ssd, https://github.com/IngeniousIdiocy/ds4/blob/glm53-m3ultra/README.md
- For agent E (talks): the DFlash paper (arXiv 2602.06036), and a Dave2D M3 Ultra review referenced in https://www.reddit.com/r/LocalLLaMA/comments/1j8r2nr/

---

## Sources and independence

- The threads below are independent first-hand reports from different authors on different chips. The main exception is the Splash engine, where B3 builds on Inco's engine and on npanj's benchmark.
- B1 (No_Shift_4543), A1 (cryingneko), A2 (Defilan), A3 (trevorbg), B2 (IngeniousIdiocy), B3 (Erp4759), B6 (ulmentflam), B7 (Equivalent-Buy1706), B8 (turtleninja99), C1 (BitXorBit) are independent of each other. B7 builds on the flash-moe code, so it is not independent of that code or of Dan Woods's work.
- Partial independent confirmations: the putrasherni and Its-all-redditive reproductions of B1; the_real_druide67 of C1; tragdor85 of B3.
- Single-author with vendor or personal interest: B4 (Mininglamp), B5 (Precisit), B3's source engine (Inco).
