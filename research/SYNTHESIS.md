# Famous and niche sources for the kernel-path course — synthesis

Run: 2026-10-10. Six territory agents (packs `07`–`12` in this folder). Brief: find the well-known explainers and the first-hand niche reports on Apple GPUs, Metal, MLX, Apple-silicon inference, and (lightly) NVIDIA kernels, 2024-10 onward, for the 33 day pages in `weeks/`. Interview answers: bar = readable and correct; last 24 months; Apple-heavy; videos only with transcripts.

## TL;DR

1. **The complaint was right.** Of ~60 sources the agents rated worth reading, 50 are not cited anywhere in `weeks/` today (checked by grep). The course leaned on Apple docs and papers; the pieces practitioners actually learn from are a small, recent (mostly 2025–2026) set of blog worklogs, GitHub PRs with measured tables, Reddit optimization logs, and WWDC transcripts.
2. **There is an Apple-GPU reading canon, and it is about 15 pieces.** Architecture: Zakharko's A19/M5 neural-accelerator microbenchmarks, dougallj's G13 ISA reference, Chips and Cheese M2 Pro. Matmul: Medley's "Fast Matrix Multiply on an Apple GPU" plus Bulatov's m5-gemm Metal-4 port. Attention: ThunderMittens, metal-flash-attention. Metal 4 / M5: WWDC26 session 330, WWDC26 232, Apple's "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU", llama.cpp PR #20962. MLX: the lazy-eval and custom-kernel docs, WWDC25 315, release notes v0.29–v0.32. Profiling: vllm-metal's profiling page, GPU MODE lecture 31 slides.
3. **The "insane improvement" stories with a kernel-level explanation live on Reddit and in READMEs, not blogs**: the DFlash log (custom kernels lost 0.5–0.8x to stock steel GEMM; a 2-line head_dim patch unlocked the fast path), the ds4 fused-kernel log (59% → 81% of measured bandwidth, 29 → 40 tok/s), the M1 Splash kernels (18.9 → 39.1 tok/s by writing fp32 matrix kernels for a chip without bf16), llm.mojo (scalar NVIDIA-style attention at <1% of peak; GEMM-decomposed attention 8–10x faster).
4. **Three things the course currently states need softening**: "Apple has no profiler" (a working if clunky Xcode/.gputrace workflow is documented); "unified memory makes Macs good at inference" (true for decode, false for prefill: ~26 FP16 TFLOPS on M3 Ultra vs 4x that on a DGX Spark); "MLX is faster than llama.cpp" (decode yes, prefill and long context CONFLICTING).
5. **Gaps that no amount of searching closed**: no stand-alone MSL-gotchas post, no independent "how I profiled my Metal kernel" post, no personal long-form writing by any MLX maintainer, no podcast transcript, no YouTube transcripts (yt-dlp blocked), no measured M3 Ultra vs H100 same-model head-to-head.

## Themes

### 1. Apple GPU architecture (feeds w1/d1, w1/d2, w1/d7)
- **Best new source:** Zakharko, *Investigating the GPU Neural Accelerators on Apple A19/M5* (tzakharko.github.io). Measured "1024 FLOPS per GPU core per cycle for FP16 matrix-multiply accumulate", optimal tile "around 32 × 32", 7.4 TFLOPS on a 5-core A19. Single author, first-hand, verified. Page header has a "2015" typo (repo created 2025-10-15).
- **ISA-level why:** dougallj's G13 reference: "Using fewer registers … allows more SIMD-groups to fit in the physical register file (higher occupancy)". Older (M1) but still the reference. Already cited 3x in the course; the `docs.html` page itself is not.
- **Measured caches:** Chips and Cheese M2 Pro (2023, older): "a tiny 8 KB L1 cache", system-level cache "234 ns latency". Chips and Cheese has no M3/M4/M5 GPU article (NOT_FOUND).
- **Dynamic Caching:** no technical third-party analysis exists (NOT_FOUND, two agents).
- **ANE vs GPU accelerator confusion:** HN commenter woadwarrior01 separates the two ("NAX cores are matmult accelerators, closer to tensor cores") — single UGC claim, but WWDC26 330 independently confirms the accelerator is "located directly in each shader core" (non-UGC confirmation, verified).
- What would change this: an M5-generation Chips and Cheese or Asahi write-up; none exists as of 2026-10-10.

### 2. Matmul on Apple (feeds w2/d3, w3/d2)
- **Medley, percisely.xyz/gemm** (M2 Air, repo 2025-10-07): "~2.5 trillion 32-bit floating point operations a second … 4000 × 4000"; "Achieving MPS-level performance requires using an undocumented instruction, simdgroup_async_copy." Verified. This is the siboehm-style ladder for Apple.
- **Bulatov, m5-gemm** (2026-05): the Metal 4 sequel. "on Metal 4, the article's inline-asm trick is permanently closed"; `max_total_threads_per_threadgroup` hint was "the single biggest practical win"; 13.5 TFLOPS vs MPS 11.7 at 4096² fp32 on M5 Max; Apple's "+30% bandwidth" "applies to the base M5 only". Verified. **Pair the two pages**: readers on macOS 26 will hit compile errors following Medley alone.
- **ThunderMittens** (Hazy, 2024-11-28): "~200GB/s memory bandwidth and ~6.5 TFLOPs of compute" on M2 Pro; "Swizzling isn't worth the hassle"; "Limited support for bf16." Verified.
- **llama.cpp PR #20962** (Developer-Ecosystem-Engineering, merged 2026-04-25, verified): tile 64×128 vs legacy 64×32, no threadgroup staging on the tensor path, "Overall GeoMean +26.4%" across quant types on M5 Max. The best kernel-level PR body in the territory.
- **GPU MODE lecture 31** (malfet, 2024-10-14): GEMV ladder from float4 dot → mat4 → access pattern; slides only (transcript INACCESSIBLE).

### 3. Attention on Apple (feeds w2/d4)
- metal-flash-attention README (Turner, 2024-09): "4400 gigainstructions per second on M1 Max (83% ALU utilization)". Already cited 4x.
- llama.cpp discussion #27668 (ggerganov, 2026-08-24): per-chip FA-vec autotuning, "sweeps the flash-attn vec (Q, NE) config grid". New.
- llm.mojo Reddit log (ulmentflam, 2026-07, verified): NVIDIA-style scalar FA "ran at under 1 percent of FLOP peak on Apple GPUs, so GEMM-decomposed attention was 8 to 10x faster"; also "Casting threadgroup pointers to the generic address space silently reads device memory". UGC; the first claim is consistent with ThunderMittens' "ALU ops are too precious" (non-UGC).
- DFlash log: head_dim=256 "2-line patch unlocked the fast SDPA path" — matches the dispatch rules already on w1/d6 from `scaled_dot_product_attention.cpp`.

### 4. Metal 4, M5 neural accelerators, Neural Engine (feeds w1/d7, w3/d2)
- **WWDC26 330** "Optimize custom machine learning operations with Metal tensors" (verified transcript): accelerator "located directly in each shader core … designed to accelerate dense compute-bound work such as the prefill stage of an LLM"; "You may also need to write Metal kernels if you're contributing to an ML framework such as MLX or llama.cpp." Watch 0:21–2:30, 5:17–9:31.
- **WWDC26 232** (Angelos, verified): "Neural Accelerators make matrix multiplication four times faster on M5 compared to M4" — Apple's own number, unreplicated.
- **Apple ML Research M5 post** (2025-11-19): prefill compute-bound, decode bandwidth-bound; M4 120 vs M5 153 GB/s; 6-model table. Not cited in the course today.
- MLX release notes v0.30.0 "Support for Neural Accelerators on M5 (macOS >= 26.2)"; PR #2772's body is a stub — "for NAX, the code is the documentation."
- Reddit B5 (Precisit, vendor): staging quantized weights to FP8/int8 instead of FP16 gives +40–50% prefill on "M6". Hardware unverifiable; keep as a pointer only.
- Neural Engine: eiln.github.io ANE reverse-engineering posts (HN 2026-09, 241 pts) — pointer for the ANE half of w1/d7.

### 5. MLX internals (feeds w1/d6, w3/d2, w3/d5)
- The maintainers write almost no prose (NOT_FOUND for personal posts by Hannun, Katharopoulos, Digani, Zhao). Their writing is: lazy-eval doc (still says "MLX does not compile and rerun compute graphs" — stale, pair with the compile doc), custom-kernels doc, release notes, WWDC25 315 (verified: "Each of these nodes corresponds to a separate GPU Kernel launch … Compiling the graph uses all of these separate Kernels into single fused Kernel").
- Third-party "how MLX works": NOT_FOUND. Best stand-in: MLX discussion #3801 (M3 Ultra MoE verify cost ~2.2x L=1).
- nvfp4 scale dispute, issue #2962: a readable case study in microscaling formats; CONFLICTING and unresolved.
- Tiny-LLM (skyzh) is a course-shaped repo for serving on Apple silicon (HN 297 pts).

### 6. llama.cpp Metal (feeds w3/d1)
- #4167 table (already cited), PR #16634 Metal4 tensor API (already cited once), **PR #20962** (new, best), **#27668** FA tuning (new), #26938 page-aware expert residency (new), HN comment: "tensor API disabled for pre-M5 and pre-A19 devices" (the gate, verified in a llama-bench log).

### 7. Practitioner optimization logs — the "insane improvements" pages (feeds w3/d2–d5, w4/d4)
| Log | Chip | Result | Kernel-level lesson | Status |
|---|---|---|---|---|
| DFlash in MLX (Reddit 1simszl → 1skesyq) | M5 Max 64 GB | 26 → 85 tok/s (9B bf16), later re-measured 4.1x | custom GEMV/SiLU/SDPA kernels lost to stock steel; 2 syncs → 1 saves ~0.5 ms each; verify cost flat 4→16 tokens | partial independent repro on M4 Max: 1.5–2.4x for 27B |
| ds4 GLM-5.3-Flash fork (Reddit 1wbkpnw) | M3 Ultra | 29 → 40 tok/s decode, 366 → 550 prefill | fuse dozens of small kernels between the big streaming ones; measure against *measured* bandwidth/matmul ceilings (59% → 81%) | single author |
| Splash M1 kernels (Reddit 1woq7cd, 1wqngu9) | M1 Max | 18.9 → 39.1 tok/s; 99 → 144 (35B-A3B) | M1 has no bf16 and MPP is slow there; write fp32 simdgroup-matrix kernels with exact 4-bit→half bit trick | commenters confirm M1 speeds |
| llm.mojo GPT-2 training (Reddit 1v3hfy5) | M4 Max | MLX bf16 1.24x faster than hand kernels | bf16 matmul without tensor path ~1.1x fp32; throttles after ~8 s | single author |
| SSD-streamed MoE (Reddit 1s7g8ov, 1wy5fma; HN flash-moe, turbo-fieldfare) | M5 Max / M5 mini | 10.6 → 20.3 tok/s | "shifting bottleneck": SSD I/O → encode overhead → projection kernels; pre-encode command buffer one step ahead; "NAX/ANE offloading: tile padding overhead cancelled every gain" | 209 GB "4x of 128 GB" claim flagged wrong by a commenter |
| MTPLX, oMLX, Rapid-MLX, turboquant-mlx READMEs | various | 2.69x (MTP heads), "30x fused DSA prefill", 3x vs Ollama, 1.85x via `simd_sum` | gains mostly from speculative/serving layers, not new kernels, except turboquant's `simd_sum` and oMLX's fused DSA | self-reported |

Pattern across the table (my inference, not stated in any thread): custom kernels win when the baseline lacks a hardware path (M1 bf16, MPP on M1, unsupported head_dim) or when dozens of small dispatches sit between big ones; they lose when competing head-on with steel GEMM on supported shapes. That is the course's "three questions" (w4/d4) with evidence attached.

### 8. Profiling reality (feeds w4/d1, w4/d2)
- vllm-metal profiling page: per-kernel GPU time via `MTL_CAPTURE_ENABLED` + .gputrace in Xcode; "Apple's 'Command Line Tools for Xcode' alone are not enough"; "xctrace … doesn't surface MLX kernel names"; replay "will lock up your machine" on a 20 GiB capture.
- GPU MODE L31 slide: "How to profile Metal from command line: Short answer: you can not" (2024-10).
- burn.dev "When profilers mislead you" (2025-12): following ncu hints produced a pointless optimization — a good NVIDIA-side counterweight for w6/d5.
- Verdict: "no profiler" → "no scriptable per-kernel profiler; a GUI capture workflow exists." Two independent sources agree.

### 9. NVIDIA additions (feeds w5, w6)
- jax-ml scaling book "How to Think About GPUs" (2025-08): chapter-length SM/memory/NVLink/roofline treatment — w5/d1.
- Chips and Cheese B200 (2025-12): 148 SMs, TMEM, "software doesn't have to care about the multi-die setup" — w5/d5.
- Modular Blackwell matmul worklog (Mojo, vendor) — w5/d5.
- gau-nernst FA on RTX 5090 (2025-08): "there are many excellent blogposts on writing fast matmul kernels, but there is none for attention" — w6/d1.
- MPK paper (OSDI'26) + Ada-MK ("launch overhead … 14.6% of end-to-end") + Josef Chen (CUDA Graphs 1.259x on H100, 1.028x on L4) — w6/d2.
- TechCrunch Sakana walk-back; CRFM HN thread MSE 0.056 "misleading" to call FP32 — w4/d3, w6/d5.
- Unsearched (NOT_FOUND by omission): Lei Mao, PMPP companions, Triton tutorials, standalone ncu walkthroughs, DeepSeek kernel write-ups.

### 10. X/Twitter (pack 12; feeds w1/d7, w1/d8, w2/d1, w2/d5, w3/d1, w3/d4, w3/d5, w4/d1, w4/d3)
- The maintainers post on X what they do not blog: zcbenz "MoE layer in MLX is getting 1.5x faster, achieved with better tile scheduling in grouped mm" (2026-09-28, verified); awnihannun "M5 Max is faster than M3 Ultra for many MLX benchmarks (prefill 2x, decode for MoEs despite having lower memory bandwidth)" (verified); angeloskath on the fused attention kernel (2024-10) and JACCL 3.3x on 4 nodes.
- **Best first-hand kernel lesson on X:** alexdelrosso forcing Metal 4 `matmul2d` onto an M2 Ultra: "gated by device name to M5/M6/A19/A20 — even though the API is portable" (verified); the fallback path is 11% faster uncompensated and 44–80% slower with the M5-default compensation. Confirms the HN "tensor API disabled for pre-M5" line from the hardware side.
- **Verification culture:** ivanfioravanti's "Same logits, 100% same result: this is The Mandatory Rule" after an earlier run with drifting logits; reply "Tokens/sec without top-k agreement and KL drift is half a benchmark." This is the course's correctness-first message said by a practitioner.
- **Independent checks found on X** (rare, keep): ivanfioravanti reproducing MTPLX at lower numbers (M5 Max 29.5 → 48.2 vs the claimed 28 → 63); liuliu correcting nisten on M5 Max fp16 TFLOPS (unresolved, turns on whether the benchmark used accelerator instructions); anemll correcting hxiao on ANE bandwidth (120 GB/s vs 819 GB/s GPU on M3 Ultra).
- Self-reported but kernel-specific: drxddy's fused entropy-decode GEMM (47 flops per byte roofline on M3 Pro; 40 → ~90 tok/s); Artex_llm's bf16 scales on M1 (53 → 63 tok/s); utsav_khairnar's agent-written masked softmax 1.84x vs MLX, tied on already-fused norms; qvac's speculative-decoding Metal kernels in llama.cpp (3.4x on code, ~2x on prose per a reply).
- Gaps: alyssarzg not on X; philipturner's handle is @philipturnerar and off-topic since 2024; dougallj posts CPU only; five timelines rate-limited (429) and covered by search instead; eight X articles unread (listed in pack 12); whether Awni left MLX in Feb 2026 is UNVERIFIED.

## Contradictions (map, not average)
1. **Custom kernels vs stock MLX.** DFlash: slower. ds4/Splash: 1.4–2x faster. Resolution candidate: different baselines (see theme 7). Not stated by any source.
2. **MLX vs llama.cpp prefill.** Reddit C1 (M3 Ultra, via LM Studio): GGUF prefill far better at long context — but a commenter notes LM Studio had no MLX prompt cache. Splash part 2: "Prefill is where MLX is still ahead." arXiv 2511.05502 (M2 Ultra): MLX best decode, MLC-LLM best TTFT. yage.ai (secondary): MLX "1.4–1.8x" over raw llama.cpp, lags on TTFT. CONFLICTING; the course should say so rather than pick.
3. **DFlash headline.** 3.3x → "4.1x" after rewrite; independent M4 Max 1.5–2.4x on a different model. Not a contradiction, but the headline is not reproduced.
4. **M5 "+30% bandwidth".** Apple newsroom vs m5-gemm: base M5 only; Pro/Max smaller. An HN comment lists M5 Max at 460 GB/s (Joeri, UGC); the course's 614 GB/s comes from Apple's March 2026 newsroom page (`research/w1.md`), so the HN figure is wrong, not conflicting.
7. **M5 Max fp16 matmul TFLOPS** (X): nisten "same as M1 Ultra" vs liuliu "exceeds M2 Ultra (around 2x)". Unresolved; the likely cause is whether the benchmark used the accelerator instructions.
5. **nvfp4 scale format** (MLX #2962): reporter vs maintainer, unresolved.
6. **Lazy-eval doc** says MLX does not compile graphs; `mx.compile` exists. Stale doc, not a real conflict.

## Negative space
- Searched, not found: standalone MSL gotchas post; independent Metal profiling write-up; Dynamic Caching analysis; "Metal for CUDA programmers"; MLX maintainer long-form; podcasts; AI-kernels GPU MODE lecture; thermal-throttling measurements beyond one Reddit anecdote; M3 Ultra vs H100 same-model.
- Inaccessible: Medium (Cloudflare), Towards AI (paywall), Apple developer forums (bot check), all YouTube transcripts (yt-dlp), Red Hat Machete article (403 to curl, from the earlier run).
- Not read, leads only: arXiv 2512.02189 Blackwell microbenchmarks; CUTLASS tcgen05 guide; Pallas Blackwell matmul; richiksc/mlx-benchmarks; eight Reddit threads by title; GPU MODE L29 materials; WWDC24 10159/10160 transcripts; Apple Tech Talks not enumerated.
- X/Twitter: covered (pack 12); five timelines hit rate limits and were covered by search; eight articles unread.

## Steelman
Strongest objection: "A reading list of 2026 Reddit logs and READMEs is not a canon; it is a snapshot that will rot in a year, and most numbers are single-author and self-reported." True. What would settle it: a second independent measurement for each theme-7 row. Mitigation applied: every UGC claim used for a course conclusion has a non-UGC anchor (ThunderMittens for the ALU argument, WWDC26 330 for the accelerator location, PR #20962 for the tensor path), and the pages should cite the lesson, not the headline tok/s.

## Counter-review of this draft (three issues found)
1. Theme 7's "pattern across the table" is my inference from four single-author threads; labeled as such above, must stay labeled on the pages.
2. "Apple-heavy canon of ~15" counts two older pieces (dougallj 2021-era, Chips and Cheese 2023) against a 24-month window; kept because every agent and HN thread still points to them; marked older on the pages.
3. The 460 vs 614 GB/s M5 Max figure: checked; 614 is Apple's newsroom number in `research/w1.md`, the 460 is a wrong HN comment. Resolved.
4. Spot-checks run: 8 web quotes (percisely, m5-gemm ×3, Zakharko ×3, PR #20962 author and table, WWDC26 330 ×2 and 232, ThunderMittens ×2, scaling book) and 3 X posts (alexdelrosso, zcbenz, awnihannun) and 3 Reddit quotes (DFlash ×2, llm.mojo ×2) all match the live source verbatim. Only one in eight sources was spot-checked; the rest are the agents' word.

## Placement map (what goes on which day)
- w1/d1 The chip: Apple M5 MLX post; m5-gemm bandwidth note; HN M5 thread bandwidth table (UGC); exo "26 TFLOPS vs 4x" (contrarian).
- w1/d2 The core: Zakharko (architecture section); dougallj docs.html; Chips and Cheese M2 Pro; ThunderMittens "Understanding The Apple GPUs".
- w1/d3 Metal path: Metal by Example compute intro (older); vllm-metal profiling (capture mechanics); GPU MODE L31 slides.
- w1/d4 MSL: m5-gemm (max_total_threads hint, async_copy closed); ThunderMittens (simdgroup_matrix layout, bf16); llm.mojo generic-address-space gotcha.
- w1/d5 Threads/memory: Medley §I; dougallj occupancy quote; Zakharko tile-size result.
- w1/d6 MLX: lazy-eval doc + compile doc pairing; custom-kernels doc; WWDC25 315 10:15–17:30; release notes index; Tiny-LLM.
- w1/d7 Metal 4 / M5 / ANE: WWDC26 330, 232; WWDC25 262, 205; Zakharko results; eiln ANE posts; HN woadwarrior01 NAX-vs-ANE comment (UGC, flagged).
- w1/d8 Who writes kernels: HN Ollama-on-MLX thread; antirez/ds4 thread; Tiny-LLM thread (simonw's "who to follow"); PR #20962 author.
- w2/d1 Roofline: ThunderMittens (bandwidth-rich, compute-poor); scaling-book rooflines; exo prefill-is-compute-bound.
- w2/d2 Elementwise/reduction: turboquant `simd_sum` 1.85x; GPU MODE L31 GEMV ladder.
- w2/d3 Matmul: Medley + m5-gemm (paired); PR #20962; Zakharko 32×32.
- w2/d4 Attention: #27668; llm.mojo FA lesson; DFlash head_dim patch; metal-flash-attention README.
- w2/d5 Quantized matvec: LEARNED_QUANTS.md; #2962; Ollama nvfp4 post; Precisit FP8 staging (pointer).
- w3/d1 llama.cpp: PR #20962; #27668; #26938; HN "tensor API disabled for pre-M5" line.
- w3/d2 steel/NAX: release notes v0.30–0.32; dflash-mlx README (NAX M=16 path); Apple M5 post Table 1; Reddit B4 W8A8 (vendor).
- w3/d3 Linear attention: nothing new found (NOT_FOUND).
- w3/d4 MoE/paging: SSD-streaming logs (1s7g8ov, 1wy5fma), flash-moe and turbo-fieldfare HN threads, #26938, MLX #3801.
- w3/d5 Launch overhead: ds4 fused-kernel log; DFlash sync-count lesson; SSD log's "pre-encode one step ahead"; Josef Chen CUDA Graphs paper (NVIDIA side).
- w4/d1 Feedback loop: Splash M1 story; llm.mojo throttling note; L31 "you can not".
- w4/d2 Profiling: vllm-metal page; WWDC26 388 (games, low priority); burn.dev.
- w4/d3 AI kernels: TechCrunch Sakana; CRFM HN MSE comment; Gimlet HN baseline critique; Reddit B9 (model-written IQ1_0 kernel).
- w4/d4 Directing an AI: theme-7 table as the evidence base for the three questions.
- w5/d1: scaling-book GPUs chapter. w5/d3: (nothing new). w5/d4: Modular Blackwell part 1 (as a preview). w5/d5: Chips and Cheese B200; Modular; arXiv 2512.02189 (unread lead).
- w6/d1: gau-nernst fa-5090; cloudrift (vendor, part 1 math). w6/d2: MPK paper; Ada-MK; Josef Chen; HN MPK thread. w6/d5: burn.dev; KernelBench-Verified quotes already present.

## Source mix and next steps
- Rough mix: official (Apple, WWDC, MLX docs) ~25%; GitHub PRs/READMEs ~30%; personal blogs/labs ~15%; Reddit/HN UGC ~25%; papers ~5%.
- With 2x budget: (1) an X sweep if pack 12 is thin; (2) read the eight title-only Reddit leads and the three unread Blackwell references; (3) one refutation agent on the theme-7 pattern; (4) enumerate Apple Tech Talks for the "M5 machine learning talk" that WWDC26 330 cites.
- Provenance caveats: dates for Medley and Zakharko come from repo metadata; the PR #20962 author is an org account inferred to be Apple; all tok/s are self-reported.
