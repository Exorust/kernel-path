# kernel-path

Ten weeks, 30 minutes a day, to become a kernel director on Apple silicon first and NVIDIA second.

A kernel director reads a profile, names the bound, specifies the tile and memory plan, predicts the number, directs an AI to write the code, and knows when the result is lying. This repo is the roadmap for getting there, plus the record of every prediction I made and what the hardware said back. Public, but written for me. If you have finished [time-to-first-token](https://github.com/Exorust/kernel-engineering) or equivalent and know what a roofline is, it will work for you too.

Fifty sessions feed two kernels, which together are one thing: the linear-attention layer of the Qwen3.5 / Qwen3-Next / Kimi Linear model family as it runs on MLX. Weeks 1 to 5 build the decode path, a quantized matvec. Weeks 6 to 10 build the prefill path, a chunkwise gated-delta kernel. Both get built on the Mac first, ported to an H100 second, and compared.

---

## What you will have at the end

- A fused int4 matvec in `mx.fast.metal_kernel` benchmarked against MLX's own at M=1, 4, 8, and the same kernel in Triton benchmarked against Marlin on an H100
- A chunkwise gated-delta prefill kernel for the vector-gated (Kimi Delta Attention) path, gated at 2x over the mlx-lm sequential scan at T=4096, and a Triton counterpart profiled against FLA
- Ten Thursday files written by hand, from memory, in MSL, CUDA, or PTX
- Fifty prediction files in `sessions/`, each with a number I wrote before the code existed
- One small merged pull request or accepted issue in mlx-lm, and the chunkwise kernel shipped

## What this does not make you

Someone who can hand-write flash attention on a whiteboard. Attention is the third arc, after week 10. The Thursday rungs cover the primitives an interviewer can reasonably ask for from memory: reductions, a simdgroup_matrix multiply, a dequant, a tiled load, one inline PTX instruction.

## Time and cost

| | |
|---|---|
| Session | 30 minutes |
| Per week | 5 sessions + 2 buffer days (catch-up only, never new material) |
| Total | 10 weeks, 50 sessions, 25 hours |
| Apple hardware | the Mac you own (this plan was run on an M5, 32 GB) |
| NVIDIA hardware | Modal H100, about $4 an hour, per-second billing, ~14 sessions, estimate $30 to $40 total; one optional B200 session |

## The weekly rhythm

Every week has the same shape. The "understand, apply, build" order is deliberate: a concept read on Monday gets a 20-line microbenchmark on Tuesday before it is trusted inside the real kernel on Wednesday.

| Day | Session | Output |
|---|---|---|
| Mon | **Understand.** One concept, one source, via `/learn read` | corrected blurt, 5 to 10 cards |
| Tue | **Apply.** A microbenchmark of that concept. I predict, the AI writes, I explain the delta | `sessions/wNN-tue.md` with a measured number |
| Wed | **Build.** Fold it into the week's kernel, profile it | kernel version + profile capture |
| Thu | **Hand-write.** The week's rung from memory, in MSL, CUDA, or PTX. No AI. Diff against Wednesday | my file, timed |
| Fri | **Judge.** Code closed, explain the kernel back, cards from the misses, one row in the numbers table | `~/learning/kernel-path/cards.md`, README row |

Thursday is protected. See `CLAUDE.md`.

## Setup

```bash
cd ~/myproj/kernel-path
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx numpy
python bench.py            # self-check, then: python -c "import bench; bench.probe()"
pip install modal && modal setup     # week 4
```

`bench.py` is the arc 1 timer: interleaved A/B pairs, median of per-pair ratios, variance warning above 8 percent CV, roofline column against the probed copy bandwidth. Arc 2 uses the full chunkwise harness at `~/myproj/apple_job_track/code/chunkwise/harness.py`, which adds fp64 parity and byte-level determinism checks.

---

## Arc 1 · the decode path: quantized matvec (weeks 1 to 5)

Decode is bandwidth-bound; TTFT already taught why. A matvec over int4 weights moves 4 bits per weight instead of 16 and dequantizes in registers. The whole game is percent of peak bandwidth, and MLX's stock kernel has a known soft spot at small batch (M=3 to 6).

### Week 1 · the machine and bf16 GEMV

**Deliverable.** A bf16 GEMV in `mx.fast.metal_kernel` with a stated percent of this machine's copy bandwidth, and my first Thursday file.

| Day | Session |
|---|---|
| Mon | *understand* · Apple GPU execution model: threadgroups, simdgroups of 32, threadgroup memory, unified memory. Source: [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) end to end, then the threads-and-threadgroups page of the [Metal docs](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups). Microarchitecture numbers: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks). |
| Tue | *apply* · Copy kernel. Sweep threadgroup width 32 to 1024 and elements per thread 1 to 8. Predict which combination reaches peak and why. Run `bench.probe()` first. |
| Wed | *build* · bf16 GEMV, W[N,K] · x[K], in `metal_kernel`. Baseline: `mx.matmul`. Report percent of copy bandwidth. Profile: `mx.metal.start_capture` then Xcode's shader profiler ([capturing a Metal workload](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode)). |
| Thu | *hand* · `thu_vadd.metal`, then `thu_reduce.metal` using `simd_sum`. From memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 2 · int4 and fused dequant

**Deliverable.** A fused int4 GEMV that matches `mx.quantized_matmul` numerically and within 10 percent of its speed at M=1.

| Day | Session |
|---|---|
| Mon | *understand* · MLX affine quantization: `bits`, `group_size`, per-group scale and bias, and the packed layout. Source: [mx.quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) and [mx.quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) docs, then the packing code in `mlx/backend/metal/kernels/quantized.h` (read only). |
| Tue | *apply* · Dequant-only kernel: int4 to bf16 for one row. Predict bytes in, bytes out, and time from the week 1 bandwidth. |
| Wed | *build* · Fused dequant GEMV, group_size 64, 4 bits. Baseline: `mx.quantized_matmul`. Parity: max abs error against `mx.dequantize` then `mx.matmul`. |
| Thu | *hand* · `thu_dequant.metal`: unpack and dequantize one group of 64 by hand. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 3 · small-M and occupancy

**Deliverable.** A batched quantized matvec at M=1, 4, 8 that matches or beats MLX at each, with an explanation of where the stock kernel stalls.

| Day | Session |
|---|---|
| Mon | *understand* · Read MLX's quantized matvec kernels in `quantized.h` (the `qmv` family) for how they split K and assign rows to simdgroups. Registers per thread vs occupancy on Apple GPUs: the register file section of metal-benchmarks. |
| Tue | *apply* · Occupancy microbench: same kernel with 32, 64, 128 live floats per thread. Predict the cliff. |
| Wed | *build* · Batched qmv. Baseline: `mx.quantized_matmul` at M=1, 4, 8. The M=3 to 6 stall reported by MTPLX is the target; a 10-line fix gave them 2.24x. |
| Thu | *hand* · `thu_tiled_load.metal`: coalesced tiled load of a [64,64] bf16 tile into threadgroup memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 4 · NVIDIA port

**Deliverable.** A Triton int4 GEMV on an H100 benchmarked against Marlin, and the PTX it compiles to, read and annotated.

| Day | Session |
|---|---|
| Mon | *understand* · [Modal GPU guide](https://modal.com/docs/guide/gpu) hello world on H100. Then the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) vector add and matmul, and dump the PTX (`kernel.asm["ptx"]`). Reference for what good looks like: [Marlin](https://github.com/IST-DASLab/marlin). |
| Tue | *apply* · Vectorized vs scalar global loads, then `cp.async`. Sources: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) sections on `ld.global.v4` and `cp.async`. Predict the bandwidth ratio. |
| Wed | *build* · Triton int4 GEMV, group_size 64. Baseline: Marlin's kernel at M=1 and M=8. Profile with [Nsight Compute](https://docs.nvidia.com/nsight-compute/) on Modal. |
| Thu | *hand* · `thu_ptx_load.cu`: a CUDA kernel with one inline PTX `ld.global.v4.b32`, from memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 5 · compare and ship

**Deliverable.** The arc 1 comparison table, the writeup, and the entry ticket to mlx-lm.

| Day | Session |
|---|---|
| Mon | *understand* · Same kernel, two profilers. What Nsight Compute shows that Xcode's shader profiler does not, and the reverse. |
| Tue | *apply* · Fill the comparison table below: both platforms, percent of roofline, M=1, 4, 8. |
| Wed | *build* · Arc 1 writeup: what was tried, what failed, the table, the honest ceiling. |
| Thu | *hand* · `thu_gemv.cu`: bf16 GEMV in CUDA from memory, timed. |
| Fri | *ship* · One small pull request or benchmark-backed issue to [mlx-lm](https://github.com/ml-explore/mlx-lm), written by me, no AI footer. This opens the contributor gate for week 10. Then the arc blurt. |

---

## Arc 2 · the prefill path: chunkwise gated delta (weeks 6 to 10)

mlx-lm runs every linear-attention model through one file, [gated_delta.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py), whose kernel is a token-sequential scan: parallel over batch, heads, and head-dim, serial over T. The chunkwise formulation ([FLA](https://github.com/fla-org/flash-linear-attention), [Gated Delta Networks](https://arxiv.org/abs/2412.06464)) turns a length-T scalar scan into T/C dense matmul steps. A working version of this kernel already exists in `~/myproj/apple_job_track/code/chunkwise/chunk_kernel_v7.py`, built by AI sessions I steered. This arc rebuilds it from the math up so that I own it; v7 is the answer key, not the starting point.

### Week 6 · the math

**Deliverable.** An fp64 chunked reference and a frozen tolerance contract, plus the sequential scan written by hand as the baseline.

| Day | Session |
|---|---|
| Mon | *understand* · Gated delta rule recurrence. Chunkwise algebra: cumulative decays in log space, the unit-lower-triangular solve (UT transform), chunk-end state carry. Why the scan loses past T=1024. Sources: the paper above, FLA's `chunk.py` in `fla/ops/gated_delta_rule/`, and [FlashQLA](https://github.com/QwenLM/FlashQLA) as prior art. |
| Tue | *apply* · Time mlx-lm's scan at T=512, 1024, 4096, 8192 for the vector-gate shape (B=1, Hk=16, Hv=32, Dk=Dv=128). Predict the scaling exponent. |
| Wed | *build* · fp64 chunked reference in numpy, NMSE tolerance per dtype frozen before any Metal code. Contract: fp32 accumulation, fast math off, no atomics, log-space decays. |
| Thu | *hand* · `thu_scan.metal`: the sequential gated-delta scan, one simdgroup per (head, value row), from memory. This is the baseline everything is measured against. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 7 · simdgroup_matrix and the intra-chunk solve

**Deliverable.** A single-chunk kernel (C=64) that matches the fp64 reference within tolerance.

| Day | Session |
|---|---|
| Mon | *understand* · `simdgroup_matrix<T,8,8>`: load, multiply, store, and the layout it wants. The register cliff: what happens to occupancy when a simdgroup holds too many 8x8 tiles. Source: the simdgroup matrix section of the [Metal Shading Language spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), and [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) for a production use. |
| Tue | *apply* · 8x8 simdgroup_matrix throughput microbench. Predict TFLOPs vs the machine's published peak. |
| Wed | *build* · Single-chunk kernel: intra-chunk decays, UT solve, outputs, chunk-end state. Gate against the fp64 reference. |
| Thu | *hand* · `thu_sgmm.metal`: one 8x8 simdgroup_matrix multiply-accumulate, from memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 8 · cross-chunk carry

**Deliverable.** The full chunkwise kernel. Gate: 2x over the sequential scan at T=4096 on the vector-gate path, or a published negative result with the mechanism.

| Day | Session |
|---|---|
| Mon | *understand* · State carry between chunks, barrier cost, double-buffered chunk staging. Read the workbench's own rejection notes: v4 lost to barriers, v5 to the register cliff (`m3-m4-progress.md`). |
| Tue | *apply* · Barrier cost vs chunk size microbench. Predict the crossover C. |
| Wed | *build* · Full kernel. Run the chunkwise `harness.py all` (parity, determinism, paired bench vs the mlx-lm scan). Then diff my directed kernel against `chunk_kernel_v7.py` and write down every place v7 made a different choice and why. |
| Thu | *hand* · `thu_nobarrier.metal`: a reduction across a chunk with no threadgroup barrier, from memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 9 · NVIDIA

**Deliverable.** FLA's chunkwise kernel profiled on an H100, and a simplified Triton chunkwise kernel I directed, with the Hopper PTX vocabulary read.

| Day | Session |
|---|---|
| Mon | *understand* · FLA's `chunk_gated_delta_rule` Triton kernel structure. PTX ISA sections: `cp.async.bulk.tensor` (TMA), `mbarrier`, `wgmma`. For what people build with them: [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). |
| Tue | *apply* · Run FLA's kernel on the H100 at the week 6 shapes. Nsight Compute it: which pipe is the bound. |
| Wed | *build* · Direct a simplified Triton chunkwise kernel (scalar gate, one chunk size). Baseline: FLA. It will lose; the point is explaining by how much and why. |
| Thu | *hand* · `thu_mbarrier.cu`: a CUDA kernel with one inline PTX `mbarrier` init-arrive-wait or one `wgmma` fence, from memory. |
| Fri | *judge* · Blurt, cards, README row. |

### Week 10 · compare and ship

**Deliverable.** The arc 2 comparison table, the writeup, and the kernel shipped.

| Day | Session |
|---|---|
| Mon | *understand* · One session, labeled as reverse-engineered, with [dougallj/applegpu](https://github.com/dougallj/applegpu): disassemble the week 8 kernel and find the instruction the register cliff comes from. Not a tool to lead with in an Apple conversation. |
| Tue | *apply* · Fill the arc 2 table. Optional: one B200 session on Modal to see `tcgen05` and tensor memory in the PTX FLA emits. |
| Wed | *build* · Arc 2 writeup: chunkwise vs scan on M5, FLA vs directed Triton on H100, what each platform's profiler said. |
| Thu | *none* · Friday covers the arc. |
| Fri | *ship* · If the kernel is small enough to review, a pull request to mlx-lm. Otherwise it lands in [kda-metal](https://github.com/Exorust/kda-metal) with an issue in mlx-lm linking it and crediting FLA, FlashQLA, and the closed prior PRs #1241 and #1389. Then the arc blurt. |

---

## Numbers

Filled in on Fridays. Machine, MLX version, and date on every row. Empty until measured.

### Arc 1 · quantized matvec

| Week | Kernel | Platform | Baseline | Speedup | % roofline | CV | Machine, version, date |
|---|---|---|---|---|---|---|---|
| 1 | bf16 GEMV | M5 | mx.matmul | | | | |
| 2 | int4 GEMV M=1 | M5 | mx.quantized_matmul | | | | |
| 3 | int4 GEMV M=4 | M5 | mx.quantized_matmul | | | | |
| 3 | int4 GEMV M=8 | M5 | mx.quantized_matmul | | | | |
| 4 | int4 GEMV M=1 | H100 | Marlin | | | | |
| 4 | int4 GEMV M=8 | H100 | Marlin | | | | |

### Arc 2 · chunkwise gated delta, vector gate, B=1 Hk=16 Hv=32 Dk=Dv=128

| Week | Kernel | Platform | Baseline | T | Speedup | NMSE vs fp64 | CV | Machine, version, date |
|---|---|---|---|---|---|---|---|---|
| 6 | hand scan (Thu) | M5 | mlx-lm scan | 4096 | | | | |
| 7 | single chunk | M5 | fp64 ref | 64 | | | | |
| 8 | full chunkwise | M5 | mlx-lm scan | 4096 | | | | |
| 8 | full chunkwise | M5 | mlx-lm scan | 8192 | | | | |
| 9 | directed Triton | H100 | FLA | 4096 | | | | |

### Thursday files

| Week | File | Compiled first try | Time vs Wednesday's AI version |
|---|---|---|---|
| 1 | thu_vadd.metal, thu_reduce.metal | | |
| 2 | thu_dequant.metal | | |
| 3 | thu_tiled_load.metal | | |
| 4 | thu_ptx_load.cu | | |
| 5 | thu_gemv.cu | | |
| 6 | thu_scan.metal | | |
| 7 | thu_sgmm.metal | | |
| 8 | thu_nobarrier.metal | | |
| 9 | thu_mbarrier.cu | | |

## After week 10

Arc 3 is attention decode, grouped-query, bf16, both platforms, against MLX's fused attention and FlashAttention. Same rhythm. Not planned in detail until arc 2 ships.

## Related

- [metalworking](https://github.com/Exorust/metalworking): the reading track this plan assumes for Apple GPU background
- [kda-metal](https://github.com/Exorust/kda-metal): the hand-written Kimi Delta Attention step kernel, 2x on M5, the portfolio piece
- [kernel-engineering](https://github.com/Exorust/kernel-engineering): the TTFT serving roadmap that precedes this one
