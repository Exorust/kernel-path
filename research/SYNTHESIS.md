# Resources for kernel-path: synthesis

As of 2026-10-01. Four subagent reports, 60 tool calls total, free text-first sources only, every quote
read literally from the page. Raw reports with verbatim quotes: `01-apple-metal-mlx.md`,
`02-nvidia-ptx.md`, `03-linear-attention.md`, `04-nvidia-ptx-gaps.md`. Seven load-bearing quotes were
re-fetched and matched by the orchestrator (MPP PDF title and section names, MLX custom-kernel rst,
Songlin Yang Part II, FLA naive.py L130-133, Colfax TMA, mlx-lm bitwise comment, Marlin README).

## TL;DR

Sixteen resources carry the whole plan. The ones that change what the plan should do:

1. **Apple published a GEMM-tiling guide in March 2026.** The 13-page Metal Performance Primitives (MPP)
   Programming Guide, §2.2–2.3, is Apple itself explaining threadgroup tile, simdgroup tile, Morton walk
   order, and accumulation-loop sync. Nothing like it existed before. It becomes the week 7 Monday read.
2. **The best delta-rule explainer is a blog, not the paper.** Songlin Yang's "DeltaNet Explained"
   Parts I and II are the week 6 reads. Part II shows the failed parallel-scan attempt first, then derives
   the chunkwise form and explains the triangular inverse as path sums on a graph. The GDN paper §3.2 then
   adds the gates.
3. **FLA's `naive.py` is the worked example.** 161 lines of fp32 PyTorch, two functions, the whole
   chunkwise algorithm including the forward-substitution loop in three lines. Run it at C=4 and print.
4. **The GPU MODE "Triton Internals" deck is the PTX-dump walkthrough.** Lecture 29 by Kapil Sharma
   shows `compiled_kernel.asm.keys()`, `cuobjdump -sass -ptx`, and `add_pipeline(pm, opt.num_stages)`,
   which is the single artifact that teaches "num_stages is a knob to a software-pipelining pass".
5. **Colfax Research's tutorials are the Hopper and Blackwell spine.** TMA deep dive, wgmma tutorial, and
   the Blackwell Tensor Memory tutorial, same authors and same CuTe vocabulary across all three.
6. **Profiling on a rented H100 has a permissions trap** that no Modal doc covers. NVIDIA's
   ERR_NVGPUCTRPERM page says containers need `--cap-add=SYS_ADMIN` or host-side caps, and the widely copied
   `NVreg_RestrictProfilingToAdminUsers=0` fix is now the legacy method slated for removal.

## The pick list, by week

Each entry: resource, what to read, minutes. "Own" = the reader's own metalworking glossary, already linked.

| Week | Day | Resource | Read exactly | Min |
|---|---|---|---|---|
| 1 | Mon | [WWDC22 10159 transcript](https://developer.apple.com/videos/play/wwdc2022/10159/) | dispatch model from ~04:55: grid → threadgroup → SIMD-group, 1024 threads, 32K threadgroup memory | 10 |
| 1 | Mon | [MLX custom_metal_kernels.rst](https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/custom_metal_kernels.rst) | first example; "Only pass the body ... signature is generated automatically" | 12 |
| 1 | Tue | [metal-benchmarks README](https://github.com/philipturner/metal-benchmarks) | the two On-Chip Memory tables (M1/M2 era; no M5 data) | 10 |
| 1 | Wed | [aurelbzo/mlx-metal-kernels](https://github.com/aurelbzo/mlx-metal-kernels) | README timing method: medians, MAD, rotated order | 10 |
| 1 | Wed | [MLX metal_debugger.rst](https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/metal_debugger.rst) | `MTL_CAPTURE_ENABLED=1`, `mx.metal.start_capture`, Dependencies view | 5 |
| 2 | Mon | [mx.quantize docstring](https://raw.githubusercontent.com/ml-explore/mlx/main/python/src/ops.cpp) ~L4796 | α, β, s formula; 8 nibbles per uint32, first element in the 4 LSBs; mode table | 10 |
| 2 | Tue | [llama.cpp ggml-common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h) | `block_q4_0`: 32 elements, one fp16 scale, 18 bytes, no bias. The contrast case | 10 |
| 2 | Wed | [MLX quantized.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h) | `load_vector` only: pre-scale x by 1/4, 1/16, 1/64 so nibbles never shift; `sumy` for the bias | 15 |
| 3 | Mon | MLX quantized.h | `qmv_fast`: rows per simdgroup, K split | 20 |
| 3 | Mon | metal-benchmarks README | register file ~208 KB/core row; occupancy vs registers | 10 |
| 3 | Wed | [llama.cpp mul_mv.metal](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/kernels/mul_mv.metal) | `kernel_mul_mv_q4_0_f32`, `block_q_n_dot_y`; second opinion on small-M | 15 |
| 4 | Mon | [GPU MODE Lecture 29 Triton Internals](https://github.com/gpu-mode/lectures/tree/main/lecture_029) | slides "add_kernel (Artifacts)", "(JIT Compiled)"; run `vector_add.py`, print `asm.keys()` | 25 |
| 4 | Mon | [Triton Puzzles](https://github.com/srush/Triton-Puzzles) | puzzles 1–3 only, as warm-up if Triton is cold | opt |
| 4 | Tue | [PTX ISA 9.7.10.8 ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld), [9.7.10.28.3.1 cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) | syntax blocks only: `.v4`, `cp-size = {4, 8, 16}`, commit_group / wait_group | 15 |
| 4 | Tue | [CUDA Binary Utilities §2.1.1](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | `cuobjdump -sass -ptx`, `nvdisasm -gi` | 5 |
| 4 | Wed | [Marlin README](https://github.com/IST-DASLab/marlin) | the FLOP-per-byte argument, "batchsizes 4-8x larger", striped partitioning | 15 |
| 4 | Thu | [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) | §1.1 constraints, `%%` escaping, §1.2 Pitfalls | 15 (reference while hand-writing) |
| 4 | Wed/Thu | [tl.inline_asm_elementwise](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html) | signature, `pack`, the `$n` vs `%n` difference | 10 |
| 5 | Mon | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | ComputeWorkloadAnalysis, SchedulerStats, MemoryWorkloadAnalysis | 25 |
| 5 | Mon | [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) + [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM) | `--set full --export`, "if --page is not used but --export is, no results will be printed"; container needs `--cap-add=SYS_ADMIN` | 10 |
| 5 | Wed | [metal-flash-attention README](https://github.com/philipturner/metal-flash-attention) | "Quantifying Performance" only: gigainstructions not gigaflops; no native fp32 atomics | 10 |
| 6 | Mon | [DeltaNet Explained Part I](https://sustcsonglin.github.io/blog/2024/deltanet-1/) | "What is Delta Rule?" through in-context retrieval; the 3-line NumPy | 15 |
| 6 | Mon | [GDN paper §3.1](https://arxiv.org/html/2412.06464v1) | the SGD framing: `L(S) = ½‖S k − v‖²`, Eq. 8, Table 1 | 10 |
| 6 | Tue | [DeltaNet Explained Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/) | "A Failed Attempt" → "Chunkwise Parallel Form" → "UT Transform Through the Lens of Graph Theory" | 25 |
| 6 | Wed | [FLA naive.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/naive.py) | both functions; L127-133 cumsum + forward substitution; L148-153 carry. Run at C=4 | 20 |
| 6 | Wed | GDN paper §3.2 + App. A.1 | Eq. 11–12, the pseudo-value `U − Diag(γ)W Sᵀ`; the γ-ratio induction (α subscript typo) | 15 |
| 6 | Thu | [mlx-lm gated_delta.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gated_delta.py) | docstrings only, L568-573 scalar vs vector shapes; `compute_g` L19 | 10 (reference for the hand-written scan) |
| 7 | Mon | [MPP Programming Guide PDF](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) | §2.2–2.3 pp. 6–9: tile sizes, Morton walk, accumulation sync; §4.1 | 25 |
| 7 | Tue | [MLX steel mma.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/steel/gemm/mma.h) | `BaseMMAFrag<T,8,8>`: `kElemsPerFrag = 64/32`, `kElemRows = 1, kElemCols = 2` | 15 |
| 7 | Wed | [Kimi Linear App. C](https://arxiv.org/html/2510.26692v1) | Listings 8(a) `chunk_dplr` vs 8(b) `chunk_kda`; the two C×C matrices | 20 |
| 7 | Thu | [bkvogel/metal_performance_testing](https://github.com/bkvogel/metal_performance_testing) | `mat_mul_optimized_nv.metal`, the CUDA-sample port; reference while hand-writing | 10 |
| 8 | Mon | [FLA chunk.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/chunk.py) | L33 fwd only: `kkt → solve_tril → recompute_w_u → scan` | 20 |
| 8 | Mon | [FLA gate.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/gate.py) | `g` is log α ≤ 0; fp32 cumsum, never a running product | 10 |
| 8 | Tue | Kimi Linear §3.2 + §6.2 | the `K/Γ` instability, secondary chunking, why a=b=k removes it | 10 |
| 8 | Wed | mlx-lm gated_delta.py L10-15, 152-163, 247-267 | bit-exact-by-construction: write the butterfly out in source; kill-switch + comparator kernel | 15 |
| 9 | Mon | [Colfax TMA deep dive](https://research.colfax-intl.com/tutorial-hopper-tma/) | full post; the load / store / multicast kernels | 30 |
| 9 | Tue | [Colfax wgmma tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | `wgmma.mma_async.sync.aligned.m64n64k16` and its register constraints | 25 |
| 9 | Tue | PTX ISA [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor), [ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix), [wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async) | syntax blocks only | 10 |
| 9 | Wed | [FlashQLA README](https://github.com/QwenLM/FlashQLA) | signature contract; "2-3× forward speedup over the FLA Triton kernel"; TileLang, not a teaching repo | 10 |
| 9 | Wed | [Shankhdhar, Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | skim the TMA + wgmma + clusters steps; the narrative version of the two Colfax posts | 20 |
| 10 | Mon | [dougallj/applegpu](https://github.com/dougallj/applegpu) | README `disassemble.py`; targets G13/M1, may not decode M5 output | 20 |
| 10 | Tue | [Colfax Blackwell Tensor Memory tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) | Part 1 "Overview of Blackwell MMA" → "Tensor Memory"; wgmma deprecated, tcgen05.mma, one thread launches | 25 |
| 10 | Tue | [PTX ISA 9.7.18 Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory) | §9.7.18.1: 512 columns × 128 lanes × 32 bits per CTA on sm_100 | 5 |

## Contradictions and traps (keep these in the plan text)

- **Transposed conventions.** GDN writes `S_t = S_{t-1}(α(I − βkkᵀ)) + βvkᵀ` (transition on the right).
  Kimi Linear writes `S_t = (I − βkkᵀ)Diag(α)S_{t-1} + βkvᵀ` (left, outer product transposed). Same math.
  Pick one convention in week 6 Wednesday before any Metal code, or `W S` vs `W Sᵀ` costs a day.
- **32K vs 60 KB.** WWDC22 says threads share up to 32K of threadgroup memory per threadgroup;
  metal-benchmarks measures ~60 KB shared memory per core. Different quantities, both plausible.
- **Blackwell has two architectures.** SM12x (RTX Pro, consumer) has no tcgen05 and no Tensor Memory and
  uses `mma.sync`. SM100 (B200/B300) is the tcgen05 part. Colfax's NVFP4 posts are SM12x and are out of scope.
- **Nsight permissions advice is mid-migration.** The regkey method still works but is "Legacy" since
  driver R610; the caps method (`/dev/nvidia-caps`) replaces it. Inside a container the fix is
  `--cap-add=SYS_ADMIN` or host-side enablement. Whether Modal's sandbox allows either: NOT_FOUND.
- **GDN Appendix A.1 typo.** `P_t = Π α_t(...)` should index α by i. The induction above it is correct.
- **llama.cpp moved its Metal kernels.** `ggml-metal.metal` no longer exists; kernels are under
  `ggml/src/ggml-metal/kernels/`. Blogs pointing at the old path are stale.
- **Triton's `asm` access.** The lecture uses `compiled_kernel.asm.keys()`; both `asm["ptx"]` and attribute
  access reach the same dict.

## Single-source findings, kept

- MPP guide as "Apple's GEMM chapter": one agent, one fetch, orchestrator re-verified the PDF (6,099,349 bytes,
  dated 2026-03-16, section titles present). Content quality beyond the TOC is the agent's judgment.
- aurelbzo/mlx-metal-kernels as the only free `mx.fast.metal_kernel` learning repo: one agent, one search.
  A better one may exist; `meirm/VeloxQuant-MLX` was a snippet-only lead, unverified.
- FlashQLA numerics claim "without sacrificing numerical precision": unquantified, one source.

## Negative space

- No free, current blog explains `mx.fast.metal_kernel` beyond the official doc. NOT_FOUND after search.
- No standalone note on gate-cumprod underflow in linear-attention kernels. Everything is in paper
  sections or code comments. Writing one (2 pages) from GDN §3.2, Kimi §3.2, and FLA gate.py would be new.
- No verified explainer of how Triton chooses `num_warps` or vectorization width. The "Coalescing" pass is
  the nearest named thing.
- No Modal documentation on Nsight Compute. Null result on a title index; page bodies unchecked.
- No GPU MODE Blackwell lecture. CUTLASS lectures exist (15, 36).
- developer.apple.com was inaccessible to obscura and WebFetch; curl with a Safari user agent works. The
  Xcode "Analyzing the performance of your Metal app" page was never read; profiling on Apple remains the
  thinnest territory, provisional.
- FLA's test tolerances (rtol/atol vs naive.py) were not read. The reader's week 6 tolerance contract has no
  external anchor yet.
- No Reddit or HN sampling was done in any territory. All findings rest on primary artifacts. A third-party
  explainer blog could exist that none of the agents saw.
- Paid or metered items were excluded by rule: the Medium "44 to 376 GFLOP/s" matmul post was not read.

## Steelman

The strongest objection: this list is primary-source heavy because the agents were told text-first and
free-only, and a 30-minute session may be better served by one good video than by a PTX ISA section. The
evidence that would settle it is a run of week 4 Monday with the Lecture 29 deck versus the recorded
lecture. If the deck alone does not get `asm.keys()` printed in 25 minutes, swap the medium for that day.

## Counter-review (blind pass over claims and quotes)

1. "Sixteen resources carry the whole plan" is the orchestrator's count, not an agent finding; the pick list
   has 40 rows across 16 distinct origins. Stated as such now.
2. The MPP guide's value rests on section titles plus one agent's read. The orchestrator verified the PDF
   exists and the titles match, not that §2.3 is readable in 25 minutes. Flagged single-source above.
3. Agent B's first report rated Colfax posts "~25-35 min" each; agent B2 rated the Blackwell one
   "1.5-2 h for Part 1". The pick list uses 25-30 min per Colfax post for the parts named, not the whole post.
4. The WWDC22 session is 2022 and metal-benchmarks is M1/M2 era; both predate the M5 by three generations.
   They are kept because nothing newer and free explains the execution model, and the MPP guide (2026) covers
   the tiling side. Threadgroup and simdgroup sizes have not changed; register-file numbers may have.
5. Quote drift check: the mlx-lm "bitwise-identical by construction" quote spans a comment line break in the
   file; the agent joined it. Verified present at L10-12.

## Source mix

Roughly 60 percent official (Apple, NVIDIA, MLX, Triton, Qwen, Moonshot, model cards), 25 percent
academic (Yang, GDN, DeltaNet, Kimi Linear, Marlin), 15 percent community (GPU MODE, Colfax, Shankhdhar,
philipturner, bkvogel, aurelbzo, dougallj). Zero journalism, zero forum.

## With 2x budget

Read the Xcode Metal debugger page with curl and a Safari UA; fetch Kapil Sharma's Triton internals part 2;
read FLA's test file for the tolerance numbers; run one HN Algolia pass per territory for third-party
explainers; verify whether Modal's sandbox permits `ncu` by trying it (one 5-minute H100 session).
