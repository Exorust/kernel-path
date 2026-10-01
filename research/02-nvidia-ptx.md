# Agent B raw report: NVIDIA Triton/PTX-level resources

Provenance: subagent (Sonnet), run 2026-10-01, 15 tool-call budget, free text-first sources.
Access path: WebFetch/curl on the pages quoted; HN Algolia for discovery only.
Caveats: Q1, Q2, Q4, Q6 returned NOT_FOUND or UNVERIFIED by the agent's own account; a
follow-up agent (03-nvidia-ptx-gaps.md) covers them. Verbatim below.

---
## Findings

### Q1 — Triton zero-to-kernel, beyond official tutorials (worked walkthroughs showing PTX/IR)

1. **Triton Puzzles (srush)** — https://github.com/srush/Triton-Puzzles — README, ~30-60 min to work through the puzzle set.
   Quote: "This set of puzzles is meant to teach you how to use Triton from first principles in an interactive fashion." (as of 2026-10-01, community/academic — Sasha Rush, Cornell/HuggingFace)
   Why: Best-known free interactive Triton primer beyond the official docs, progresses from indexing basics to Flash Attention and quantized-kernel puzzles.
   Caveat (verified): the fetched summary explicitly states it does **not** show PTX output — it teaches the Triton language model, not what the compiler decided.

   **Gap**: no verified resource that dumps `kernel.asm["ptx"]` for a matmul/fused-softmax kernel and explains compiler decisions. NOT_FOUND.

### Q2 — Reading PTX and SASS

1. **Modal GPU Glossary** — https://modal.com/gpu-glossary — UNVERIFIED: TOC confirms PTX, SASS, TMA entries; no verbatim body quote obtained (guessed sub-page 404'd).
   **Gap**: no verified source on cuobjdump / Godbolt CUDA mode / which PTX ISA sections matter for ld.global.v4, cp.async, ldmatrix, mma.sync. NOT_FOUND.

### Q3 — Hopper specifics (TMA, mbarrier, wgmma, clusters) — strongest section, 3 ranked

1. **Colfax Research — "Delving into the Hopper Hierarchy" (wgmma tutorial)** — https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/ — full post (~25-35 min).
   Quote: "No series of CUDA® tutorials is complete without a section on GEMM (GEneral Matrix Multiplication)." Shows explicit `wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16` PTX with register constraints, TiledMMA/CuTe code; Part 1 of 3. (industry/academic)

2. **Colfax Research — "A Hopper TMA Deep Dive"** — https://research.colfax-intl.com/tutorial-hopper-tma/ — full post (~25-35 min).
   Quote: "TMA (Tensor Memory Accelerator) is a new feature introduced in the NVIDIA Hopper™ architecture for doing asynchronous memory copy between a GPU's global memory (GMEM) and the shared memory (SMEM) of its threadblocks (i.e., CTAs)." Complete worked kernel code for TMA load, store, store-reduce, load-multicast with cluster sync; linked runnable repo.

3. **Pranjal Shankhdhar — "Outperforming cuBLAS on H100: A Worklog"** — https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog — full worklog (~45-60 min).
   Quote: "TMA is a new hardware piece introduced in Hopper architecture. It is a faster way to load tiles of multi-dimensional matrices between GMEM and SMEM." Covers TMA, wgmma, clusters with practical code, naive-to-tuned narrative. Read after the two Colfax posts.

   Supplementary: **DeepGEMM README** — https://github.com/deepseek-ai/DeepGEMM — quote: "The library is designed for simplicity, with only a limited number of core kernel functions." Read the source after the Colfax posts; README does not teach mechanics.

   Not verified this session: ThunderKittens posts, GPU MODE Hopper lectures, FlashMLA README, Triton Gluon docs. UNVERIFIED leads.

### Q4 — Blackwell

1. **Colfax Research — "CUTLASS Tutorial: NVFP4 Blockscaled GEMM on NVIDIA RTX PRO Blackwell GPUs (SM12x)"** — https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
   Quote: "SM12x does not use `tcgen05` or TMEM; instead, like with SM8x it uses warp-level `mma.sync` instructions." tcgen05/TMEM are SM100 (B200/B300) features. CONFLICTING with the brief's framing; no verified SM100 tcgen05 explainer found. NOT_FOUND.

### Q5 — Nsight Compute

1. **NVIDIA Nsight Compute Kernel Profiling Guide** — https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html — ComputeWorkloadAnalysis, SchedulerStats, MemoryWorkloadAnalysis sections (~30-40 min).
   Quotes: "Pipelines with very high utilization might limit the overall performance." / "Having many skipped issue slots indicates poor latency hiding." / "Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy)." (official)
   **Gap**: no verified guide for running ncu headless on Modal / cloud H100 (perf-counter permissions). NOT_FOUND.

### Q6 — Int4 weight-only GEMV/GEMM (Marlin) and inline PTX

1. **Marlin README (IST-DASLab)** — https://github.com/IST-DASLab/marlin — full README (~15-20 min).
   Quote: "This means that the full performance benefits of weight-only quantization should, in principle, extend to batchsizes 4-8x larger than what is currently achieved by existing kernels." Explains the FLOP-to-byte argument (25-50 MACs/weight budget) and the striped-partitioning/double-buffering layout.
   **Gap**: inline PTX references (Triton `tl.inline_asm_elementwise`, CUDA `asm volatile`) unverified. NOT_FOUND.

## Gaps
- Q1 PTX-dump-and-explain walkthrough: NOT_FOUND. Q2 largely unmet. Q3 candidates ThunderKittens/GPU MODE/FlashMLA/Gluon unverified. Q4 CONFLICTING (SM12x post ≠ tcgen05). Q5 Modal ncu: NOT_FOUND. Q6 inline PTX: NOT_FOUND.
- GPU MODE lectures directory listing was fetched but only speculative labels came back; discarded.

## Sources
- Colfax Research (3 posts, 1 institutional source). cudaforfun.substack.com (independent individual). IST-DASLab/marlin (official repo). deepseek-ai/DeepGEMM (official). srush/Triton-Puzzles (community/academic). docs.nvidia.com/nsight-compute (official). modal.com/gpu-glossary (weakly verified). HN Algolia (discovery only).
