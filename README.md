# kernel-path

Six weeks, thirty minutes a day, to understand the Apple GPU stack and the kernels people write for it well enough to build your own by directing an AI. Four weeks on Apple silicon, two on NVIDIA for contrast and for the kernels that made news there.

Each day is one page in `weeks/`: a short narrative with a citation on every number, the one primary source to read, the numbers to remember, and five questions. You write your answers at the bottom of the page and run `/learn read weeks/wN/dM.md` to have them checked and turned into cards. Nothing to install until you choose to run something.

## The thirty days

### Week 1 · The Apple stack, big picture

- **[Day 1 · The chip](weeks/w1/d1.md)** — One system on a chip, one memory pool, four engines. Bandwidth by tier from 153 GB/s on the M5 to 614 on the M5 Max, and why there is no datacenter part.
- **[Day 2 · The GPU core](weeks/w1/d2.md)** — Simdgroups of 32, large registers and small caches, threadgroup memory limits, and how Dynamic Caching changed occupancy from the M3 on.
- **[Day 3 · Metal](weeks/w1/d3.md)** — The seven objects from source to GPU, simdgroup_matrix as Apple's tensor core, and the 2026 guide that drops two CUDA habits.
- **[Day 4 · MLX](weeks/w1/d4.md)** — Lazy graphs, how an op becomes a kernel, mx.compile and mx.fast, and the one door for your own kernels: mx.fast.metal_kernel.
- **[Day 5 · Who writes Apple kernels today](weeks/w1/d5.md)** — Apple's MLX team, llama.cpp, MPS and MPP, and a small community whose best repositories stop at the M2. What is actually missing.

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

All thirty pages are written. Raw research behind the pages is in `research/`, one report per week. The previous version of this repository, a ten-week build plan, is on the `v1` branch.
