# kernel-path

Six weeks, about two hours a day, to understand the Apple GPU stack and the kernels people write for it well enough to build your own by directing an AI. Four weeks on Apple silicon, the first of them eight days, and two on NVIDIA for contrast and for the kernels that made news there.

Each day is one page in `weeks/`: a narrative with a citation on every number, a worked example, the numbers to remember, an ordered two-hour reading plan with minutes and what to look for in each source, and seven questions, two of which need the deeper sources. You write your answers at the bottom of the page and run `/learn read weeks/wN/dM.md` to have them checked and turned into cards. Nothing to install until you choose to run something.

## The thirty days

### Week 1 · The Apple stack, big picture (eight days)

- **[Day 1 · The chip](weeks/w1/d1.md)** — One system on a chip, one memory pool, four engines. Bandwidth by tier from 153 GB/s on the M5 to 614 on the M5 Max, and a worked decode floor from bandwidth alone.
- **[Day 2 · The GPU core](weeks/w1/d2.md)** — Simdgroups of 32, large registers and small caches, threadgroup memory limits, occupancy, and how Dynamic Caching changed it from the M3 on. A worked example of how many threads fill ten cores.
- **[Day 3 · Metal, the path from source to GPU](weeks/w1/d3.md)** — Apple's `add_arrays` in C and in MSL, line by line, then the seven objects every launch passes through, and the per-launch cost measured on an M5.
- **[Day 4 · MSL, the language](weeks/w1/d4.md)** — C++17 with things removed: types, literals, vectors, casts, the four address spaces, attributes, and a generated kernel read line by line. What `double` and `printf` do when you try them.
- **[Day 5 · Threads and memory inside a kernel](weeks/w1/d5.md)** — How a thread learns where it is, `simd_sum` and shuffles, the one-writer rule, threadgroup scratch with a barrier, and a row-sum kernel worked in full.
- **[Day 6 · MLX](weeks/w1/d6.md)** — Lazy graphs, the chain from a Python op to a Metal kernel, `mx.compile` and `mx.fast`, and a custom kernel call mapped argument by argument to days 2 to 5.
- **[Day 7 · Metal 4, the M5 accelerators, and the Neural Engine](weeks/w1/d7.md)** — Two engines, two paths. MTLTensor, the machine-learning encoder, Metal Performance Primitives, and MLX's NAX kernels, against Core ML's 10x on the Neural Engine with no kernel control.
- **[Day 8 · Who writes Apple kernels today](weeks/w1/d8.md)** — Apple's MLX team, llama.cpp, MPS and MPP, and a small community whose best repositories stop at the M2, with a table of which file to open for each question.

### Week 2 · Kernel families and the ideas that carry across hardware

- **[Day 1 · Roofline on Apple](weeks/w2/d1.md)** — Memory-bound, compute-bound, launch-bound. The A100 ridge at 208 operations per byte against about 27 on an M1 Max, and what that does to every kernel.
- **[Day 2 · Elementwise, reduction, scan, softmax](weeks/w2/d2.md)** — The NVIDIA classics, 30x on a reduction and 5x from access pattern alone, read beside MLX's own reduce and softmax kernels.
- **[Day 3 · Matrix multiply](weeks/w2/d3.md)** — Boehm's ten-rung ladder from 1.3 to 93.7 percent of cuBLAS, and Apple's guide saying staging and pipelining are unnecessary on an M5.
- **[Day 4 · Attention](weeks/w2/d4.md)** — FlashAttention's byte argument, FlashAttention-2's partitioning lesson, Metal's missing atomics, and MLX's prefill and decode split at query length 8.
- **[Day 5 · Quantized matvec](weeks/w2/d5.md)** — 4-bit formats, the pre-scaling trick both engines share, Marlin's batch budget, and the MTPLX stall at M=3 to 6 that MLX then fixed.

### Week 3 · Apple case studies with large speedups

- **[Day 1 · llama.cpp's Metal backend](weeks/w3/d1.md)** — The June 2023 pull request, 90 percent of bandwidth five days later, and the table where decode follows bandwidth and prefill follows cores until the M5.
- **[Day 2 · MLX steel and the M5 neural accelerators](weeks/w3/d2.md)** — 4x compute with 1.3x bandwidth, split-K at 1.62x on large-K matmuls, and a dispatch cliff where one row cost a third of the speed.
- **[Day 3 · Linear attention on MLX](weeks/w3/d3.md)** — A 56x that was measured against the wrong baseline, a 2x kernel that was 6.4 percent end to end, and an independent 2x from keeping state in registers.
- **[Day 4 · Expert paging and MoE on Apple](weeks/w3/d4.md)** — 60 percent of experts on SSD at the same speed as all in memory, the knee below 40 percent, and the kernels that stop working on empty expert rows.
- **[Day 5 · Launch overhead at scale](weeks/w3/d5.md)** — Hazy's 1,350-to-770 arithmetic, 231 command buffers per token on MLX with the GPU busy 68 percent, and three authors who say a kernel needs a schedule.

### Week 4 · Building kernels with an AI, on Apple

- **[Day 1 · The feedback-loop problem on Metal](weeks/w4/d1.md)** — 62 issues that say "silently", a validation layer that hid a compiler bug, and the three parts of a loop you have to build yourself.
- **[Day 2 · Profiling on Apple](weeks/w4/d2.md)** — Over 150 counters in Xcode against 3 sets in the API, the new gpudebug and metalperftrace tools, and why the roofline is your profiler.
- **[Day 3 · AI-generated kernels, the evidence](weeks/w4/d3.md)** — KernelBench's under-20 percent, Sakana's withdrawn 100x, the ReLU hack, the drop from 1.43 to 0.88, and NVIDIA's speed-of-light scoreboard.
- **[Day 4 · Directing an AI](weeks/w4/d4.md)** — A 232x worklog steered every few hours, a project where every custom kernel lost, the three questions, and the seven parts of a kernel brief.
- **[Day 5 · Capstone brief](weeks/w4/d5.md)** — Five open MLX gaps checked on 2026-10-08, two that closed during the course, and the brief you write for one of them. No code.

### Week 5 · NVIDIA, the big picture

- **[Day 1 · NVIDIA architecture against Apple](weeks/w5/d1.md)** — SMs, warps, 228 KB of shared memory, 3.35 TB/s, tensor cores at 15x the ordinary cores, with each fact beside its Apple counterpart.
- **[Day 2 · CUDA, PTX, SASS, Triton, CUTLASS](weeks/w5/d2.md)** — Four layers from source to machine code, which one the record holders write at, and what each layer costs an AI-written kernel.
- **[Day 3 · The matmul ladder on NVIDIA](weeks/w5/d3.md)** — Six weekends from 1.3 to 93.7 percent of cuBLAS, Volkov's occupancy result, and the same kernel at 4 percent on an H100.
- **[Day 4 · Hopper: TMA, wgmma, the H100 worklog](weeks/w5/d4.md)** — Two new units, a climb from 32 to 764 TFLOPS in one file, and the tile that spilled registers and lost five times its speed.
- **[Day 5 · Low precision and Blackwell](weeks/w5/d5.md)** — FP8 and microscaling formats, tensor memory at 256 KB per SM, the instruction that broke every Hopper kernel, and the two Blackwells.

### Week 6 · NVIDIA kernels that made news

- **[Day 1 · FlashAttention 1 to 4](weeks/w6/d1.md)** — Four versions, four chips, four problems: bytes, partitioning, asynchrony, and the exponential unit that stopped keeping up.
- **[Day 2 · Megakernels](weeks/w6/d2.md)** — 78 percent of bandwidth in one kernel, an 81-kernel rebuild at 97.4 percent of it, and the regime where fusing the whole model wins and loses.
- **[Day 3 · Serving kernels: FlashInfer, Marlin](weeks/w6/d3.md)** — The library under every engine, the quantized matmul that holds 3.87x to batch 32, and its successor when the hardware changed.
- **[Day 4 · Linear attention on NVIDIA](weeks/w6/d4.md)** — FLA's chunked kernels, FlashQLA's 2 to 3x that is 1.2x at a chat prompt, and why the same algorithm lost on a Mac.
- **[Day 5 · Profiling, AI kernels, capstone brief 2](weeks/w6/d5.md)** — Nsight's speed-of-light section, the four harness properties a trustworthy timing needs, and the NVIDIA twin of your Apple brief.

## Your glossary, placed

The 56 pages of [metalworking](https://metalworking.vercel.app/) are assigned to the days they belong to; each day's reading plan names its pages. The map, page → day:

**machine** · [unified-memory](https://metalworking.vercel.app/machine/unified-memory/) → w1d1 · [gpu-core](https://metalworking.vercel.app/machine/gpu-core/) → w1d1, w1d2 · [simdgroup](https://metalworking.vercel.app/machine/simdgroup/) → w1d2, w1d5 · [registers](https://metalworking.vercel.app/machine/registers/) → w1d2 · [threadgroup-memory](https://metalworking.vercel.app/machine/threadgroup-memory/) → w1d2, w1d5 · [occupancy](https://metalworking.vercel.app/machine/occupancy/) → w1d2, w2d1 · [f16](https://metalworking.vercel.app/machine/f16/) → w1d4 · [amx](https://metalworking.vercel.app/machine/amx/) → w1d7 · [neural-accelerators](https://metalworking.vercel.app/machine/neural-accelerators/) → w1d7, w3d2 · [special-paths](https://metalworking.vercel.app/machine/special-paths/) → w2d2

**metal** · [metal-the-api](https://metalworking.vercel.app/metal/metal-the-api/) → w1d3 · [command-buffers](https://metalworking.vercel.app/metal/command-buffers/) → w1d3, w3d5 · [compilation-pipeline](https://metalworking.vercel.app/metal/compilation-pipeline/) → w1d3 · [dispatch-geometry](https://metalworking.vercel.app/metal/dispatch-geometry/) → w1d3, w1d5 · [msl](https://metalworking.vercel.app/metal/msl/) → w1d4 · [function-constants](https://metalworking.vercel.app/metal/function-constants/) → w1d4 · [synchronization](https://metalworking.vercel.app/metal/synchronization/) → w1d5 · [simdgroup-matrix](https://metalworking.vercel.app/metal/simdgroup-matrix/) → w1d3, w2d3 · [simdgroup-async-copy](https://metalworking.vercel.app/metal/simdgroup-async-copy/) → w2d3 · [mtltensor-and-mpp](https://metalworking.vercel.app/metal/mtltensor-and-mpp/) → w1d7 · [mps](https://metalworking.vercel.app/metal/mps/) → w1d8 · [profiling](https://metalworking.vercel.app/metal/profiling/) → w1d8, w4d2 · [disassembly](https://metalworking.vercel.app/metal/disassembly/) → w4d2

**mlx** · [mlx-overview](https://metalworking.vercel.app/mlx/mlx-overview/) → w1d6 · [lazy-evaluation](https://metalworking.vercel.app/mlx/lazy-evaluation/) → w1d6 · [how-an-op-becomes-a-kernel](https://metalworking.vercel.app/mlx/how-an-op-becomes-a-kernel/) → w1d6 · [mx-compile](https://metalworking.vercel.app/mlx/mx-compile/) → w1d6, w3d5 · [mx-fast](https://metalworking.vercel.app/mlx/mx-fast/) → w1d6, w4d4 · [steel](https://metalworking.vercel.app/mlx/steel/) → w1d8, w2d3 · [quantization](https://metalworking.vercel.app/mlx/quantization/) → w2d5 · [distributed](https://metalworking.vercel.app/mlx/distributed/) → w1d6

**techniques** · [roofline](https://metalworking.vercel.app/techniques/roofline/) → w2d1 · [arithmetic-intensity](https://metalworking.vercel.app/techniques/arithmetic-intensity/) → w2d1 · [decode-vs-prefill](https://metalworking.vercel.app/techniques/decode-vs-prefill/) → w2d4 · [kv-cache](https://metalworking.vercel.app/techniques/kv-cache/) → w2d4 · [online-softmax](https://metalworking.vercel.app/techniques/online-softmax/) → w2d2 · [cooperative-load](https://metalworking.vercel.app/techniques/cooperative-load/) → w1d5, w2d2 · [tiling](https://metalworking.vercel.app/techniques/tiling/) → w2d3 · [register-blocking](https://metalworking.vercel.app/techniques/register-blocking/) → w2d3 · [double-buffering](https://metalworking.vercel.app/techniques/double-buffering/) → w2d3 · [fusion-and-epilogues](https://metalworking.vercel.app/techniques/fusion-and-epilogues/) → w3d5 · [flash-attention](https://metalworking.vercel.app/techniques/flash-attention/) → w2d4, w6d1

**kernels** · [gemm-tiled](https://metalworking.vercel.app/kernels/gemm-tiled/) → w2d3 · [gemm-double-buffered](https://metalworking.vercel.app/kernels/gemm-double-buffered/) → w2d3 · [gemm-async-ghost](https://metalworking.vercel.app/kernels/gemm-async-ghost/) → w2d3 · [steel-blockloader](https://metalworking.vercel.app/kernels/steel-blockloader/) → w1d8, w2d3 · [steel-blockmma](https://metalworking.vercel.app/kernels/steel-blockmma/) → w2d3 · [steel-gemm-fused](https://metalworking.vercel.app/kernels/steel-gemm-fused/) → w2d3 · [nax-gemm](https://metalworking.vercel.app/kernels/nax-gemm/) → w1d7, w3d2 · [steel-attention](https://metalworking.vercel.app/kernels/steel-attention/) → w2d4 · [mfa-codegen](https://metalworking.vercel.app/kernels/mfa-codegen/) → w2d4, w1d8 · [llamacpp-attention](https://metalworking.vercel.app/kernels/llamacpp-attention/) → w3d1, w1d8

**war-stories** · [three-questions](https://metalworking.vercel.app/war-stories/three-questions/) → w4d4 · [the-failures](https://metalworking.vercel.app/war-stories/the-failures/) → w4d1, w1d8 · [cheap-tricks](https://metalworking.vercel.app/war-stories/cheap-tricks/) → w2d5, w4d4 · [sparse-v](https://metalworking.vercel.app/war-stories/sparse-v/) → w3d4

All thirty pages are written. Raw research behind the pages is in `research/`, one report per week. The previous version of this repository, a ten-week build plan, is on the `v1` branch.
