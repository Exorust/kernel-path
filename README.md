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

Every Tuesday, Wednesday, and Thursday already has its file in `sessions/` (`w01-tue.md` through `w10-wed.md`). Each opens with **the question**: what is being asked, the given shapes and numbers, the arithmetic to do before running anything, and what the request to Claude must contain. Open the file, answer in place, then start.

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

**By Friday you can say:** a Metal kernel is a grid of threads, grouped 32 at a time into simdgroups that run in lockstep, grouped again into threadgroups that share a small fast memory. A matrix-vector product (GEMV) on this machine is limited by how fast memory can be read, about 93 GB/s on an M5, and the only score that matters is what percent of that your kernel reaches.

**Words used this week.** *Thread*: one copy of your kernel running on one element. *Simdgroup*: 32 threads that execute the same instruction at the same time; they can pass values to each other without touching memory. *Threadgroup*: a bundle of simdgroups that can share a small on-chip memory. *Grid*: all the threads of one kernel launch. *Unified memory*: the CPU and GPU use the same RAM, so there is no copy step, but the bandwidth limit still applies. *GEMV*: matrix times vector, the shape of every decode step.

**Mon · How does an Apple GPU run my code?**
1. 10 min. Read the dispatch-model part of the [WWDC22 "Scale compute workloads across Apple GPUs" transcript](https://developer.apple.com/videos/play/wwdc2022/10159/) from about 04:55: grid of threadgroups, threadgroups split into SIMD-groups, up to 1024 threads and 32K threadgroup memory per threadgroup. Then your own [gpu-core](https://metalworking.vercel.app/machine/gpu-core/) and [simdgroup](https://metalworking.vercel.app/machine/simdgroup/). Goal: draw thread, simdgroup, threadgroup, grid as four nested boxes.
2. 12 min. Read the first example in the [MLX custom Metal kernels doc, .rst source](https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/custom_metal_kernels.rst) (the raw file, not the rendered page, which buries the text under a 1500-line sidebar). The one rule that trips CUDA people: you pass only the kernel *body*, the signature is generated. Find the two lines that set `grid` and `threadgroup`. Work out how many simdgroups that launch has.
3. 8 min. `/learn read` on what you just read: blurt it back with the pages closed, write the cards.
Skip today: [threadgroup-memory](https://metalworking.vercel.app/machine/threadgroup-memory/) and [unified-memory](https://metalworking.vercel.app/machine/unified-memory/). They are references for when a number surprises you later this week.

**Tue · How wide should a threadgroup be?**

The experiment: a copy kernel reads one array and writes it to another. It does no math, so the only cost is moving bytes, and the best possible result is the copy roof from `bench.probe()`. Two knobs get swept and GB/s is measured for every combination:
- *Threadgroup width*: threads per threadgroup, 32 (one simdgroup) up to 1024 (Monday's limit).
- *Elements per thread*: how many array elements one thread copies, 1 to 8. At 1, each thread does one load and one store. At 4, each thread loads a `float4`, 16 bytes in one instruction.

1. 5 min. Copy `sessions/TEMPLATE.md` to `sessions/w01-tue.md` and fill in "Before" only. Bound: you know it for a copy. Prediction: one pair, for example "width 256, 4 per thread, about 85 percent of roof", plus two or three sentences of why, built from Monday's ingredients: the threads-per-core rule, Apple's advice on threadgroup size, and the outstanding-loads mechanism. Being wrong with a clear reason is the good outcome.
2. 5 min. Run `python -c "import bench; bench.probe()"` once. That number is the machine's copy roof. Compare it with the two On-Chip Memory tables in the [metal-benchmarks README](https://github.com/philipturner/metal-benchmarks), which put Apple next to Ampere and RDNA in one table. Those tables stop at M2, so expect the M5 to differ.
3. 15 min. Tell Claude to write the copy kernel in `mx.fast.metal_kernel` with width and elements-per-thread as parameters, and a sweep over both using `bench.paired`. Read the table it prints.
4. 5 min. Fill in "After" in your own words: where the prediction was wrong, and what [dispatch-geometry](https://metalworking.vercel.app/metal/dispatch-geometry/) says about why.

**Wed · The first real kernel: bf16 GEMV**
1. 5 min. `sessions/w01-wed.md`. Shape: W is [N, K] in bf16, x is [K]. Predict the time from bytes moved (N times K times 2) divided by Tuesday's roof.
2. 5 min. Read [mx.fast](https://metalworking.vercel.app/mlx/mx-fast/) up to the metal_kernel section, so the Ask uses the right words.
3. 15 min. Ask Claude for the GEMV, one simdgroup per output row, `simd_sum` for the dot product. Baseline: `mx.matmul`. Run `bench.paired` with `bytes_moved` set, and read the roofline percent.
4. 5 min. Capture it. The [MLX Metal debugger doc](https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/metal_debugger.rst) states the two preconditions people miss: run with `MTL_CAPTURE_ENABLED=1`, then `mx.metal.start_capture("w01.gputrace")`. Open in Xcode, find the kernel's duration in the Dependencies view. [profiling](https://metalworking.vercel.app/metal/profiling/) explains what Xcode will and will not show you. If the timing method itself feels shaky, [aurelbzo/mlx-metal-kernels](https://github.com/aurelbzo/mlx-metal-kernels) is a learning repo on the same API with a harness that reports medians and median absolute deviation.

**Thu · Hand-write, no AI**
1. 15 min. `kernels/w01/thu_vadd.metal`: add two arrays, one element per thread. From memory. Only the [msl](https://metalworking.vercel.app/metal/msl/) page is open.
2. 15 min. `kernels/w01/thu_reduce.metal`: sum an array using `simd_sum`, one simdgroup per 32 elements. Say "done" and Claude runs both and reports whether they compile and match.
If `thu_vadd` does not compile in 15 minutes, stop there. The compiler error goes in the Friday cards.

**Fri · Judge**
1. 10 min. `/learn review`, then `/learn session` on `kernels/w01/` with every file closed. Explain why the GEMV is one simdgroup per row and where the time goes.
2. 10 min. Cards from the misses into `~/learning/kernel-path/cards.md`.
3. 10 min. One row in the numbers table below: GEMV speedup vs `mx.matmul`, percent of roof, variance, machine and date. Tick week 1 in `progress.md`.

### Week 2 · int4 and fused dequant

**By Friday you can say:** MLX stores a 4-bit weight as a nibble inside a 32-bit word, eight per word, with one scale and one bias per group of 64. A fused kernel reads the packed words, multiplies the nibbles against a pre-scaled activation so it never has to shift them out, and adds the bias term once per group from a running sum of the activation. The result moves one quarter of the bytes of a bf16 matvec and should be close to four times faster at M=1.

**Words used this week.** *Affine quantization*: store `w ≈ s·q + β` where q is a small integer, s the scale, β the bias. *Group*: the run of 64 weights that share one s and one β. *Packing*: eight 4-bit values in one uint32, first value in the four least significant bits. *Fused dequant*: converting q back to a float inside the matvec kernel instead of writing a bf16 matrix out first. *Nibble*: a 4-bit value.

**Mon · How does MLX store an int4 weight?**
1. 10 min. Read the [mx.quantize docstring](https://raw.githubusercontent.com/ml-explore/mlx/main/python/src/ops.cpp) (search for `def quantize`, around line 4800). It has the α, β, s formula, the packing order, and a table of the four modes (affine, mxfp4, nvfp4, and the default marked with `*`). Then your own [quantization](https://metalworking.vercel.app/mlx/quantization/).
2. 12 min. In Python: quantize a [64, 128] bf16 matrix with `mx.quantize(w, group_size=64, bits=4)`, print the shapes of the three outputs, and unpack the first uint32 by hand with shifts and masks until you recover the first eight weights through `mx.dequantize`.
3. 8 min. `/learn read`: blurt the format back, cards.

**Tue · How many bytes does dequant move?**

The experiment: a kernel that reads packed int4 weights and writes them out as bf16, nothing else. It is the dequant half of Wednesday's kernel on its own. The question is whether the time is explained by bytes alone: one input byte produces four output bytes, so the kernel writes four times what it reads, and the roof from week 1 gives a predicted time from the total.

1. 5 min. `sessions/w02-tue.md`. Predict bytes in and bytes out for dequantizing one [N, K] int4 matrix to bf16, and the time from week 1's roof.
2. 5 min. Read the `block_q4_0` struct in [llama.cpp's ggml-common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h): 32 weights, one fp16 scale, no bias, 18 bytes. That is the contrast case: scale-only vs MLX's scale-and-bias.
3. 15 min. Ask Claude for a dequant-only kernel in `metal_kernel`, one thread per packed word. Baseline: `mx.dequantize`. Run `bench.paired` with `bytes_moved`.
4. 5 min. "After". [decode-vs-prefill](https://metalworking.vercel.app/techniques/decode-vs-prefill/) if you want to re-anchor why bytes are the whole story at M=1.

**Wed · The fused int4 GEMV**

The build: the week 1 bf16 matvec, but with the weight matrix stored as int4 groups and converted to bf16 inside the kernel, in registers, right before the multiply. Nothing bf16-sized ever touches memory. Baseline is MLX's own `mx.quantized_matmul`, and the two numbers that matter are the speedup over your week 1 kernel and the gap to MLX.

1. 5 min. `sessions/w02-wed.md`. Predict the speedup over week 1's bf16 GEMV from the byte ratio alone, then predict the gap to `mx.quantized_matmul`.
2. 10 min. Read `load_vector` in [MLX's quantized.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h). The trick: it pre-scales the activation by 1/4, 1/16, 1/64 so the packed nibbles never need shifting out, and it accumulates `sumy` so the bias term costs one multiply per group. [fusion-and-epilogues](https://metalworking.vercel.app/techniques/fusion-and-epilogues/) names the general pattern.
3. 12 min. Ask Claude for the fused GEMV, group_size 64, 4 bits, using that pre-scale trick. Baseline: `mx.quantized_matmul`. Parity: max abs error against `mx.dequantize` then `mx.matmul`.
4. 3 min. "After", including the roofline percent.

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w02/thu_dequant.metal`: unpack and dequantize one group of 64 from eight uint32 words, with scale and bias, writing 64 bf16 values. From memory. Say "done".

**Fri · Judge**
1. 10 min. `/learn review`, then `/learn session` on `kernels/w02/` with files closed. Explain why the activation is pre-scaled and where the bias term goes.
2. 10 min. Cards from the misses.
3. 10 min. README row, tick week 2.

### Week 3 · small-M and occupancy

**By Friday you can say:** a matvec kernel assigns rows to simdgroups and splits K across lanes. At M=1 the kernel is bandwidth-bound and the stock MLX kernel is near the roof. At M=3 to 6 the stock kernel stalls because the work per weight load is too small to hide latency and too large for the matvec layout, and a batched variant that reuses each loaded weight across M activations fixes it. Occupancy, the number of simdgroups resident per core, is what you trade for registers.

**Words used this week.** *M*: the number of activation rows, equal to the number of sequences decoding at once. *Occupancy*: how many simdgroups a core keeps resident to hide memory latency. *Register pressure*: live values per thread; past a cliff the compiler spills or occupancy drops. *K-split*: cutting the reduction dimension across lanes and summing with simd_sum. *Coalesced load*: adjacent threads reading adjacent addresses in one transaction.

**Mon · How does MLX's own matvec work, and where does it stall?**
1. 15 min. Read `qmv_fast` in [quantized.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h): how many rows per simdgroup, how K is split across lanes, where `simd_sum` happens. Write the thread-to-data mapping as a one-line sketch.
2. 7 min. Your own [occupancy](https://metalworking.vercel.app/machine/occupancy/) and [registers](https://metalworking.vercel.app/machine/registers/). Then the register-file row of the [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) On-Chip Memory table (about 208 KB per core on M1/M2).
3. 8 min. `/learn read`, cards.

**Tue · Where is the register cliff?**

The experiment: the week 1 copy kernel again, but each thread is forced to hold an array of 32, then 64, then 128 floats live in registers while it works. More registers per thread means fewer simdgroups fit on a core at once, so occupancy drops. At some count the compiler spills to memory or occupancy falls below what hides latency, and the copy slows down. That count is the cliff.

1. 5 min. `sessions/w03-tue.md`. Predict at how many live floats per thread the kernel slows down, and why.
2. 20 min. Ask Claude for the same copy kernel with 32, 64, 128 live floats per thread held in a register array, timed with `bench.paired`. Read the sweep.
3. 5 min. "After".

**Wed · Batched qmv at M=1, 4, 8**

The build: the week 2 int4 matvec extended so that one loaded weight word is used against M activation rows instead of one. At M=1 it should match week 2. At M=4 and M=8 the stock MLX kernel reportedly stalls, because it reloads weights per row, and a kernel that reuses each loaded word across rows should pull ahead. Baseline is `mx.quantized_matmul` at each M.

1. 5 min. `sessions/w03-wed.md`. Predict time at M=1, 4, 8 for the stock kernel and for a batched kernel that loads each packed word once and uses it for all M rows.
2. 5 min. Second opinion on the layout: `kernel_mul_mv_q4_0_f32` and `block_q_n_dot_y` in [llama.cpp's mul_mv.metal](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/kernels/mul_mv.metal). The war story [cheap-tricks](https://metalworking.vercel.app/war-stories/cheap-tricks/) is the same lesson from your own glossary.
3. 15 min. Ask Claude for the batched kernel. Baseline: `mx.quantized_matmul` at M=1, 4, 8. The M=3 to 6 stall reported by MTPLX is the target; a 10-line fix gave them 2.24x.
4. 5 min. "After", three rows.

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w03/thu_tiled_load.metal`: coalesced load of a [64, 64] bf16 tile from device memory into threadgroup memory, each thread loading a contiguous float4. From memory. [cooperative-load](https://metalworking.vercel.app/techniques/cooperative-load/) and [steel-blockloader](https://metalworking.vercel.app/kernels/steel-blockloader/) are the reference pages allowed open. Say "done".

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session` on `kernels/w03/`. Explain the stall and the fix without the code.
2. 10 min. Cards.
3. 10 min. README rows for M=4 and M=8, tick week 3.

### Week 4 · NVIDIA port

**By Friday you can say:** Triton compiles a Python kernel through three intermediate representations to PTX, NVIDIA's portable assembly, and then to SASS, the real machine code. You can dump all five with `compiled_kernel.asm`, read the PTX for the vector loads and async copies you expected, and tell from the SASS whether the compiler did what you asked. The int4 GEMV on an H100 loses to Marlin because Marlin's weight layout was designed so that the dequant and the tensor-core load line up.

**Words used this week.** *PTX*: NVIDIA's intermediate assembly, stable across GPU generations. *SASS*: the per-architecture machine code PTX compiles to. *cp.async*: an instruction that copies global memory to shared memory without staging through registers. *num_stages*: how many tiles Triton's software pipeline keeps in flight. *Marlin*: the reference int4 GEMM kernel from IST Austria that holds its speed up to batch 16 to 32.

**Mon · What did Triton actually emit?**
1. 5 min. Modal hello world on an H100 from the [Modal GPU guide](https://modal.com/docs/guide/gpu). Save the function decorator; it is reused for six weeks.
2. 20 min. [GPU MODE Lecture 29, Triton Internals](https://github.com/gpu-mode/lectures/tree/main/lecture_029), slides "Example: add_kernel (Artifacts)" and "(JIT Compiled)". Run the deck's `vector_add.py` on the H100, print `compiled_kernel.asm.keys()`, open the PTX, and find the load and store. Then the "Triton GPU Passes" slides: the line `add_pipeline(pm, opt.num_stages)` is the whole lesson that num_stages is a knob to a pipelining pass, not a hardware fact. All five IRs side by side are in [the linked gist](https://gist.github.com/kapilsh/e8f09e8ed4f2f3bcfe13dd4bd099e270).
3. 5 min. `/learn read`, cards. If Triton itself is cold, the first three [Triton Puzzles](https://github.com/srush/Triton-Puzzles) are the warm-up on a buffer day.

**Tue · Vectorized loads and cp.async**

The experiment: a Triton copy kernel on the H100, run at three block sizes. For each, dump the PTX and check whether the load came out as `ld.global.v4` (16 bytes per instruction) or as four scalar loads, then compare the GB/s. A second variant uses `cp.async`, which copies global memory straight into shared memory without passing through registers. Baseline is `torch.clone`.

1. 5 min. `sessions/w04-tue.md`. Predict the bandwidth ratio of 16-byte vector loads over 4-byte scalar loads, and what cp.async adds.
2. 10 min. PTX ISA syntax blocks only: [9.7.10.8 ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld) for `.v4`, and [9.7.10.28.3.1 cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) for `cp-size = {4, 8, 16}` plus commit_group and wait_group. Then [CUDA Binary Utilities §2.1.1](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) for `cuobjdump -sass -ptx`.
3. 12 min. Ask Claude for a Triton copy kernel at three block sizes, dump the PTX, and confirm the `.v4` load appears. Time it against `torch.clone`.
4. 3 min. "After".

**Wed · Triton int4 GEMV vs Marlin**

The build: the week 2 int4 matvec written in Triton for the H100, same group size and packing. Baseline is Marlin, the reference int4 kernel whose weight layout was designed so the dequant lines up with the tensor-core load. Your kernel will lose; the deliverable is the measured gap at M=1 and M=8 and a one-paragraph reason, backed by the PTX.

1. 5 min. `sessions/w04-wed.md`. Predict the gap to Marlin at M=1 and M=8 and name the reason.
2. 10 min. [Marlin README](https://github.com/IST-DASLab/marlin): the FLOP-per-byte argument (why a naive int4 kernel loses its advantage past batch 1 to 2) and the striped partitioning plus double buffering that extend it to "batchsizes 4-8x larger".
3. 12 min. Ask Claude for the Triton int4 GEMV, group_size 64, same format as week 2. Baseline: Marlin at M=1 and M=8. Dump the PTX and find the dequant.
4. 3 min. "After". Inline asm in Triton, if the dequant needs it: [tl.inline_asm_elementwise](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html), note `pack` and that Triton uses `$n` where CUDA uses `%n`.

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w04/thu_ptx_load.cu`: a CUDA kernel with one inline `ld.global.v4.b32`, from memory. Reference allowed open: [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) §1.1 for the constraint syntax and the `%%` escape, §1.2 Pitfalls for `asm volatile` and the memory clobber. Say "done"; Claude compiles it on Modal and reports.

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session` on `kernels/w04/`. Explain, PTX closed, what the compiler did with your block size.
2. 10 min. Cards.
3. 10 min. README rows, tick week 4.

### Week 5 · compare and ship

**By Friday you can say:** the same kernel profiled on both platforms gives different kinds of evidence. Nsight Compute shows pipe utilization, warp stall reasons, and memory workload with hardware counters. Xcode's shader profiler shows timing and occupancy but has no counter-level view, so on Apple you reason from the roofline and from experiments. You know how to run Nsight headless and what breaks on a rented GPU.

**Words used this week.** *Nsight Compute (ncu)*: NVIDIA's kernel profiler with hardware counters. *Warp stall reason*: why a warp did not issue an instruction in a cycle. *ERR_NVGPUCTRPERM*: the error when a container lacks permission to read GPU counters. *Gigainstructions*: instruction throughput, a better score than GFLOPS on Apple GPUs.

**Mon · Two profilers, one kernel**
1. 15 min. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), three sections: ComputeWorkloadAnalysis, SchedulerStats, MemoryWorkloadAnalysis. "Having many skipped issue slots indicates poor latency hiding" is the sentence to remember.
2. 10 min. [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html): `--set full --export`, and the gotcha "if --page is not used but --export is, no results will be printed". Then [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM): containers need `--cap-add=SYS_ADMIN` or host-side caps; the old `NVreg_RestrictProfilingToAdminUsers=0` regkey is now the legacy method. Whether Modal's sandbox allows counters is unknown; Tuesday finds out.
3. 5 min. `/learn read`, cards. Your own [profiling](https://metalworking.vercel.app/metal/profiling/) for the Apple side.

**Tue · Fill the table**

The session: no new kernel. Run Nsight Compute on the week 4 kernel on the H100, rerun the week 3 kernel on the Mac with the machine quiet, and put the numbers side by side in the arc 1 table. The first thing that may happen on Modal is a permissions error from the profiler. That is a finding, not a failure; record it and fall back to event timing.

1. 5 min. `sessions/w05-tue.md`. Predict the percent of roofline on each platform before you look.
2. 20 min. Run `ncu --set full --export w04 --page details` on the week 4 kernel on Modal. If it fails with ERR_NVGPUCTRPERM, record that as the finding and fall back to `torch.cuda.Event` timing plus the roofline. On the Mac, rerun the week 3 kernel at M=1, 4, 8 with fans pinned and apps closed, and capture it.
3. 5 min. Fill the arc 1 comparison table below.

**Wed · Arc 1 writeup**
1. 5 min. Read "Quantifying Performance" in the [metal-flash-attention README](https://github.com/philipturner/metal-flash-attention): why it reports gigainstructions instead of GFLOPS, and that Apple has no native fp32 atomics. Your own [the-failures](https://metalworking.vercel.app/war-stories/the-failures/) before writing anything.
2. 25 min. Write `writeups/arc1.md`: what was tried, what failed, the table, the honest ceiling, both machines named with dates.

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w05/thu_gemv.cu`: bf16 GEMV in CUDA, one warp per row, warp shuffle reduction, from memory. Say "done"; Claude compiles and times it on Modal.

**Fri · Ship**
1. 15 min. One small pull request or benchmark-backed issue to [mlx-lm](https://github.com/ml-explore/mlx-lm), written by you, no AI footer. This opens the contributor gate for week 10.
2. 15 min. Arc blurt: `/learn session` across weeks 1 to 5, cards, tick week 5.

---

## Arc 2 · the prefill path: chunkwise gated delta (weeks 6 to 10)

mlx-lm runs every linear-attention model through one file, [gated_delta.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py), whose kernel is a token-sequential scan: parallel over batch, heads, and head-dim, serial over T. The chunkwise formulation turns a length-T scalar scan into T/C dense matmul steps. A working version of this kernel already exists in `~/myproj/apple_job_track/code/chunkwise/chunk_kernel_v7.py`, built by AI sessions I steered. This arc rebuilds it from the math up so that I own it; v7 is the answer key, not the starting point.

Blast radius, verified from the model cards and config files: Qwen3-Next (36 of 48 layers), the Qwen3.5 family (3 linear layers to 1 full-attention layer), and Kimi Linear (3:1 KDA to MLA) all use this path with dk = dv = 128, 16 key heads, 32 value heads, chunk size 64. A kernel for that shape with a scalar gate covers Qwen; the vector gate covers Kimi.

### Week 6 · the math

**By Friday you can say:** the delta rule updates a d_k by d_v state matrix S so that S·k_t moves toward v_t, which is one step of online least squares. The gate α_t decays S before the update. The chunkwise form splits T into chunks of C, computes all C outputs of a chunk with dense matmuls plus one unit-lower-triangular solve, and carries the state between chunks sequentially, so the sequential work drops from T steps to T/C steps. You know which notation convention you are using.

**Words used this week.** *State S*: the d_k by d_v matrix the layer carries instead of a KV cache. *Delta rule*: `S ← S + β (v − S k) kᵀ`, correct the stored value for key k by the error. *Gate α (or g)*: a per-token decay in (0, 1), stored in log space. *Chunk*: C consecutive tokens, C = 64 in every production model. *UT transform*: the unit-lower-triangular solve that turns C sequential corrections into one matrix. *Scan*: the sequential per-token loop the current mlx-lm kernel runs.

**Mon · What is the delta rule?**
1. 12 min. [DeltaNet Explained, Part I](https://sustcsonglin.github.io/blog/2024/deltanet-1/) by Songlin Yang, sections "What is Delta Rule?" through "Why is DeltaNet Superior at In-context Retrieval". Run the three-line NumPy update it gives.
2. 10 min. [Gated DeltaNet paper §3.1](https://arxiv.org/html/2412.06464v1): the SGD framing, `L(S) = ½‖S k − v‖²`, Eq. 8 with the gate, and Table 1 which lines up five linear-attention variants as five objectives. Your own [arithmetic-intensity](https://metalworking.vercel.app/techniques/arithmetic-intensity/) and [tiling](https://metalworking.vercel.app/techniques/tiling/) are the background for Tuesday.
3. 8 min. `/learn read`, cards.

**Tue · Why does the scan lose, and what replaces it?**

The experiment: time mlx-lm's existing sequential kernel at four sequence lengths and see how the time grows. The kernel runs one loop over all T tokens per (head, value row), so the GPU's parallelism is fixed regardless of T and the time should grow linearly or worse. This is the baseline number the whole arc has to beat.

1. 5 min. `sessions/w06-tue.md`. Predict how the mlx-lm scan's time scales with T at the vector-gate shape (B=1, Hk=16, Hv=32, Dk=Dv=128), and the exponent.
2. 15 min. [DeltaNet Explained, Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/): "Parallel Scan for DeltaNet: A Failed Attempt" first, then "A Chunkwise Algorithm for DeltaNet" and "UT Transform Through the Lens of Graph Theory", which explains the triangular inverse as a sum over paths. This part is ungated; Wednesday adds the gates.
3. 8 min. Ask Claude to time mlx-lm's scan at T = 512, 1024, 4096, 8192. Your own [lazy-evaluation](https://metalworking.vercel.app/mlx/lazy-evaluation/) explains why every iteration must be evaluated.
4. 2 min. "After".

**Wed · The reference and the contract**

The build: not a Metal kernel yet. A slow, exact reference in float64 numpy that implements the chunkwise algorithm from naive.py, and a written tolerance that every later kernel is tested against. Chunking changes the order of additions, so results will not be bit-identical to the sequential scan; the contract says how far off is acceptable, per data type, before any optimization starts.

1. 5 min. `sessions/w06-wed.md`. Write down the convention you will use for the whole arc: GDN applies the transition on the right, `S_t = S_{t-1}(α(I − βkkᵀ)) + βvkᵀ`; Kimi Linear writes it on the left with `kvᵀ`. Same math, transposed. Pick GDN's, because FLA and mlx-lm use it.
2. 12 min. [FLA's naive.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/naive.py), 161 lines, both functions. Lines 127 to 133 are the cumsum of log gates and the forward-substitution loop; lines 148 to 153 are the pseudo-value and the carry. Run it at C=4 and print every intermediate. Then GDN §3.2 Eq. 11 and 12 to match the names, and Appendix A.1 for the γ-ratio induction (its `P_t` line has an α subscript typo; the induction above it is right).
3. 10 min. Ask Claude for an fp64 numpy chunked reference built from naive.py, and freeze the tolerance: NMSE per dtype against the fp64 sequential reference, fp32 accumulation, fast math off, log-space decays.
4. 3 min. "After".

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w06/thu_scan.metal`: the sequential gated-delta scan, one simdgroup per (head, value row), lane-held state, from memory. This is the baseline everything is measured against. Reference allowed open: the three docstrings in [mlx-lm's gated_delta.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gated_delta.py) (lines 568 to 573 give the scalar vs vector gate shapes) and `compute_g` at line 19. Say "done".

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session`. Explain the chunkwise form on paper with C=2.
2. 10 min. Cards.
3. 10 min. README row for the hand scan, tick week 6.

### Week 7 · simdgroup_matrix and the intra-chunk solve

**By Friday you can say:** `simdgroup_matrix<T, 8, 8>` is an 8 by 8 tile held across the 32 lanes of a simdgroup, two elements per lane, multiplied by one instruction. A chunk of C=64 with d=128 is a 64 by 128 block, so the intra-chunk matmuls are tiled from these 8 by 8 pieces, and the number of tiles one simdgroup holds live is the register budget. Apple's own guide names the three decisions: threadgroup tile, simdgroup tile, walk order.

**Words used this week.** *simdgroup_matrix*: Metal's 8 by 8 matrix fragment type with a hardware multiply. *Fragment*: the per-lane slice of a tile, two elements here. *Intra-chunk*: the work inside one chunk that needs no state from other chunks. *WY / UT representation*: writing the product of C rank-1 updates as two small matrices W and U. *Register cliff*: the tile count past which the compiler spills to memory.

**Mon · Apple's own tiling chapter**
1. 20 min. [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), March 2026, §2.2 to 2.3 (pages 6 to 9): threadgroup tile size, simdgroup tile size, Morton walk order, accumulation-loop synchronization, static extents. Then §4.1 for the theoretical analysis. Your own [simdgroup-matrix](https://metalworking.vercel.app/metal/simdgroup-matrix/) and [register-blocking](https://metalworking.vercel.app/techniques/register-blocking/).
2. 10 min. `/learn read`, cards.

**Tue · How fast is one 8x8 multiply?**

The experiment: a kernel that does nothing but `simdgroup_matrix` 8 by 8 multiply-accumulates in a loop, with no memory traffic to speak of. It measures the machine's matrix-multiply ceiling in TFLOPs so you know the compute roof for Wednesday's chunk kernel, the same way week 1's copy measured the bandwidth roof.

1. 5 min. `sessions/w07-tue.md`. Predict TFLOPs for a kernel that does nothing but 8 by 8 multiply-accumulates, against the machine's published peak.
2. 10 min. `BaseMMAFrag<T, 8, 8>` in [MLX's steel mma.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/steel/gemm/mma.h): `kElemsPerFrag = 64 / 32`, `kElemRows = 1, kElemCols = 2` is the fact that makes the layout click. Your own [nax-gemm](https://metalworking.vercel.app/kernels/nax-gemm/) for the M5 tensor path.
3. 12 min. Ask Claude for the throughput microbench. Read the number against the prediction.
4. 3 min. "After".

**Wed · The single-chunk kernel**

The build: the chunkwise algorithm for exactly one chunk of 64 tokens with no state carried in or out. It computes the log-space decays inside the chunk, solves the unit-lower-triangular system, and produces the 64 outputs plus the chunk-end state, with the matmuls tiled from 8 by 8 pieces. It is checked against Wednesday's float64 reference, not timed against the scan yet.

1. 5 min. `sessions/w07-wed.md`. Predict the time for one C=64 chunk from the matmul FLOPs and the roof.
2. 10 min. [Kimi Linear paper, Appendix C](https://arxiv.org/html/2510.26692v1): Listings 8(a) `chunk_dplr` and 8(b) `chunk_kda`, two 30-line PyTorch functions diffed. 8(b) builds only two C by C matrices, `Aqk` and `Akk`, and is the structure to copy for the vector gate. Your own [steel-blockmma](https://metalworking.vercel.app/kernels/steel-blockmma/) for how steel tiles a block.
3. 12 min. Ask Claude for the single-chunk kernel: intra-chunk log-space decays, the UT solve, outputs, chunk-end state, tiled with simdgroup_matrix. Gate against the fp64 reference.
4. 3 min. "After".

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w07/thu_sgmm.metal`: one 8 by 8 simdgroup_matrix load, multiply-accumulate, store, from memory. Reference allowed open: [simdgroup-matrix](https://metalworking.vercel.app/metal/simdgroup-matrix/) and [bkvogel/metal_performance_testing](https://github.com/bkvogel/metal_performance_testing) `mat_mul_optimized_nv.metal`, a CUDA-sample port, for the shape of a tiled Metal kernel. Say "done".

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session`. Explain how 64 elements sit on 32 lanes.
2. 10 min. Cards.
3. 10 min. README row, tick week 7.

### Week 8 · cross-chunk carry

**By Friday you can say:** the full kernel is four stages, kkᵀ, triangular solve, recompute W and U, then the chunk scan that carries S. Gates live in log space as a cumulative sum of non-positive numbers, and every use is the exponential of a difference, never a running product, because a product of 64 gates underflows fp32. Two implementations are bit-exact only if the reduction tree is written out in source. You either reached 2x at T=4096 or you can say which stage ate the time.

**Words used this week.** *Carry*: passing the chunk-end state to the next chunk. *Barrier*: a threadgroup-wide wait; expensive, and the reason v4 lost. *Double buffering*: loading chunk i+1 while computing chunk i. *Log-space gate*: storing log α and summing, so the cumulative product is a cumsum. *Secondary chunking*: splitting a chunk again in full precision to avoid dividing by tiny cumulative gates. *Kill gate*: the pre-agreed number below which the result is published as a negative.

**Mon · The four-stage skeleton**
1. 12 min. [FLA's chunk.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/chunk.py), `chunk_gated_delta_rule_fwd` at line 33 only, and the comment at lines 70 to 71: `kkt + solve_tril + recompute_w_u`. That split is the kernel skeleton; the Triton is Agent-B territory you can skim.
2. 8 min. [FLA's gate.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/gate.py): `g = -A_log.exp() * softplus(g + dt_bias)`, so g is log α and never positive; the kernel does an fp32 cumsum, never a product. Your own [double-buffering](https://metalworking.vercel.app/techniques/double-buffering/) and [gemm-double-buffered](https://metalworking.vercel.app/kernels/gemm-double-buffered/) for the staging pattern.
3. 5 min. Read the workbench's own rejection notes in `~/myproj/apple_job_track/work/m3-m4-progress.md`: v4 lost to barriers, v5 to the register cliff.
4. 5 min. `/learn read`, cards.

**Tue · Barrier cost vs chunk size**

The experiment: a kernel that loops over chunks of C tokens and hits one threadgroup barrier per chunk, with the chunk's work held constant per token. Smaller C means more barriers per token. The sweep over C = 16, 32, 64, 128 shows at what chunk size the barrier stops being the dominant cost, which is the C the full kernel should use.

1. 5 min. `sessions/w08-tue.md`. Predict the crossover chunk size where one barrier per chunk stops mattering.
2. 8 min. [Kimi Linear §3.2 and §6.2](https://arxiv.org/html/2510.26692v1): the `K/Γ` term divides by a cumulative product, GLA's fix was log-space plus secondary chunking in full precision at a speed cost, and KDA's a=b=k removes two of those steps. Your own [synchronization](https://metalworking.vercel.app/metal/synchronization/).
3. 14 min. Ask Claude for the barrier microbench at C = 16, 32, 64, 128.
4. 3 min. "After".

**Wed · The full kernel and the gate**

The build: week 7's single-chunk kernel extended to loop over all chunks, carrying the state from each chunk into the next. This is the kernel the arc exists for. It runs through the chunkwise harness for parity against float64, byte-level determinism across runs, and paired timing against the mlx-lm scan at T = 4096 and 8192. The agreed gate is 2x at T = 4096 on the vector-gate path.

1. 5 min. `sessions/w08-wed.md`. Predict the speedup over the scan at T=4096 and T=8192, and which stage dominates.
2. 10 min. The header and kernel docstrings of [mlx-lm's gated_delta.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gated_delta.py), lines 10 to 15, 152 to 163, 247 to 267: bit-exact by construction means the shuffle_xor butterfly is written out in source, with a kill-switch and a slow comparator kernel kept as the oracle. Your own [simdgroup-async-copy](https://metalworking.vercel.app/metal/simdgroup-async-copy/) and [gemm-async-ghost](https://metalworking.vercel.app/kernels/gemm-async-ghost/).
3. 12 min. Ask Claude for the full kernel. Run the chunkwise `harness.py all` (parity, determinism, paired bench against the mlx-lm scan). Then diff against `chunk_kernel_v7.py` and write down every place v7 chose differently and why.
4. 3 min. "After". If the number is below 2x at T=4096, the Friday row says so and names the stage.

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w08/thu_nobarrier.metal`: a reduction across a chunk using only simd shuffles, no threadgroup barrier, from memory. Say "done".

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session`. Explain why the gates are summed and not multiplied.
2. 10 min. Cards.
3. 10 min. README rows for T=4096 and T=8192, tick week 8.

### Week 9 · NVIDIA

**By Friday you can say:** on Hopper the fast path is TMA copying whole tiles from global to shared memory asynchronously, mbarriers to know when a tile has landed, and wgmma issuing a 64 by N by 16 matmul from a whole warpgroup with operands in shared memory. FLA's Triton kernel uses the same four stages as week 8, and Nsight shows which pipe is the bound. Your simplified Triton port loses to FLA and you can say by how much and where.

**Words used this week.** *TMA (Tensor Memory Accelerator)*: Hopper's hardware tile copier, `cp.async.bulk.tensor` in PTX. *mbarrier*: a shared-memory barrier that async copies signal when complete. *wgmma*: warpgroup matrix multiply-accumulate, four warps issuing one asynchronous MMA. *Warpgroup*: four consecutive warps, 128 threads. *CTA*: a thread block, in PTX vocabulary.

**Mon · TMA, read with a worked kernel**
1. 20 min. [Colfax Research, "A Hopper TMA Deep Dive"](https://research.colfax-intl.com/tutorial-hopper-tma/), the load, store, and multicast kernels. The runnable repo is linked from the post.
2. 5 min. PTX ISA syntax block for [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
3. 5 min. `/learn read`, cards.

**Tue · wgmma, then profile FLA**

The experiment: run FLA's production Triton chunkwise kernel, the one your week 8 kernel is the Metal counterpart of, on the H100 at the arc's shapes, under Nsight Compute if permitted. The point is to see which pipe a well-optimized version of this algorithm saturates on NVIDIA hardware before you direct your own.

1. 5 min. `sessions/w09-tue.md`. Predict which Nsight pipe FLA's chunk kernel saturates at the week 6 shapes.
2. 12 min. [Colfax, "Delving into the Hopper Hierarchy" (wgmma)](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/): the explicit `wgmma.mma_async.sync.aligned.m64n64k16` and its register constraints. PTX ISA syntax blocks for [ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) and [wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async).
3. 10 min. Ask Claude to run FLA's `chunk_gated_delta_rule` on the H100 at the week 6 shapes under `ncu`, if counters are permitted, else timed. Read the stall reasons.
4. 3 min. "After".

**Wed · Direct a Triton chunkwise kernel**

The build: a deliberately simplified version of the week 8 algorithm in Triton, scalar gate only, one fixed chunk size. Baseline is FLA. It will be slower; the deliverable is the measured ratio and the reason, which should come from Tuesday's profile.

1. 5 min. `sessions/w09-wed.md`. Predict the ratio to FLA for a scalar-gate, single-chunk-size Triton kernel.
2. 8 min. [FlashQLA README](https://github.com/QwenLM/FlashQLA): the shared `chunk_gated_delta_rule(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)` contract and the claim of "2-3× forward speedup" over FLA on Hopper through warpgroup specialization in TileLang. Not a teaching repo; read it for what "good" costs. Skim [Shankhdhar's cuBLAS worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) for the TMA and wgmma steps told as a story.
3. 14 min. Ask Claude for the simplified Triton kernel. Baseline: FLA. It will lose; the point is explaining by how much and why.
4. 3 min. "After".

**Thu · Hand-write, no AI**
1. 30 min. `kernels/w09/thu_mbarrier.cu`: a CUDA kernel with one inline PTX mbarrier init, arrive, and wait, or one wgmma fence, from memory. Reference allowed open: the [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) doc. Say "done"; Claude compiles on Modal.

**Fri · Judge**
1. 10 min. `/learn review`, `/learn session`. Explain TMA plus mbarrier without the diagram.
2. 10 min. Cards.
3. 10 min. README row, tick week 9.

### Week 10 · compare and ship

**By Friday you can say:** Blackwell datacenter parts replace wgmma with tcgen05.mma, launched by one thread, accumulating into a dedicated Tensor Memory of 512 columns by 128 lanes per CTA; the consumer Blackwell parts have none of this. On Apple the only way below the compiler is a reverse-engineered disassembler that targets the M1 and may not decode M5 output. Arc 2 is written up and shipped.

**Words used this week.** *tcgen05*: the Blackwell SM100 tensor-core instruction family. *Tensor Memory (TMEM)*: on-chip accumulator memory separate from registers and shared memory. *SM100 vs SM12x*: datacenter Blackwell (B200) vs consumer Blackwell (RTX Pro); only SM100 has TMEM. *applegpu*: dougallj's reverse-engineered Apple GPU disassembler.

**Mon · One look below the Apple compiler**
1. 20 min. [dougallj/applegpu](https://github.com/dougallj/applegpu) README and `disassemble.py`. Try it on the week 8 kernel's binary from the Xcode capture. It targets the G13 (M1); if it fails to decode M5 code, that is the finding. Your own [disassembly](https://metalworking.vercel.app/metal/disassembly/) first. Labeled as reverse-engineered; not a tool to lead with in an Apple conversation.
2. 10 min. `/learn read`, cards.

**Tue · The arc 2 table, and one Blackwell look**

The session: no new kernel. Fill the arc 2 table from the week 8 and week 9 numbers, then spend the second half reading what Blackwell changed, because the wgmma instruction you learned in week 9 is already deprecated on the datacenter Blackwell part.

1. 15 min. Fill the arc 2 table below from the week 8 and week 9 numbers.
2. 15 min. [Colfax, "Writing GEMM Kernels Using Tensor Memory for Blackwell"](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/), Part 1 sections "An Overview of Blackwell MMA" and "Tensor Memory" only, then the one paragraph of [PTX ISA §9.7.18.1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory) with the 512 by 128 by 32-bit geometry. Optional: one B200 session on Modal to see `tcgen05` in the PTX FLA emits.

**Wed · Arc 2 writeup**
1. 5 min. Your own [three-questions](https://metalworking.vercel.app/war-stories/three-questions/), once more, before writing.
2. 25 min. `writeups/arc2.md`: chunkwise vs scan on the M5, FLA vs directed Triton on the H100, what each profiler showed, credits to FLA, FlashQLA, and the closed prior pull requests #1241 and #1389.

**Thu · none.** Friday covers the arc.

**Fri · Ship**
1. 20 min. If the kernel is small enough to review, a pull request to mlx-lm. Otherwise it lands in [kda-metal](https://github.com/Exorust/kda-metal) with an issue in mlx-lm linking it. Written by you, no AI footer, benchmark table included.
2. 10 min. Arc blurt across weeks 6 to 10, cards, tick week 10.

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

- [research/SYNTHESIS.md](research/SYNTHESIS.md): where every external link in this plan came from, with verbatim quotes, the contradictions to watch for, and what was searched and not found. Raw agent reports in `research/0*.md`.

- [metalworking](https://metalworking.vercel.app/): the Apple GPU glossary this plan assumes. Each week lists its pre-read pages; the [machine](https://metalworking.vercel.app/machine/gpu-core/) and [mlx](https://metalworking.vercel.app/mlx/mlx-overview/) sections are the week 0 read if any of it is new
- [kda-metal](https://github.com/Exorust/kda-metal): the hand-written Kimi Delta Attention step kernel, 2x on M5, the portfolio piece
- [kernel-engineering](https://github.com/Exorust/kernel-engineering): the TTFT serving roadmap that precedes this one
