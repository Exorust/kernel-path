# Agent A raw report: Apple GPU / Metal / MLX kernel-writing resources

Provenance: subagent (Opus), run 2026-10-01, 15 tool-call budget, free text-first sources.
Access path: curl / obscura / pdftotext / raw.githubusercontent.com / api.github.com, as stated per item.
Caveats: Q5 (profiling) provisional, developer.apple.com pages partly INACCESSIBLE to the fetchers;
no community (Reddit/HN) cross-check performed. Verbatim below.

---
## Findings

(AS_OF 2026-10-01. Every quote below was read literally via curl/obscura/pdftotext, not inferred. All items are free, no sign-up.)

### Q1 — Apple GPU execution model for a CUDA person

**1. philipturner/metal-benchmarks README** — https://github.com/philipturner/metal-benchmarks (raw: https://raw.githubusercontent.com/philipturner/metal-benchmarks/main/README.md) · community
What to read: "Overview" + the two **On-Chip Memory** tables + "ALU Layout" and "Instruction Throughputs". ~35-45 min.
Why: the only free doc that puts Apple's numbers in the same table as Pascal/Turing/Ampere/Ada and RDNA. Verbatim:
> "| Per Core | Apple 7, 8 | Intel Gen9 | Vega | RDNA 1, 2 | RDNA 3 | Pascal | Turing | Ampere, Ada |" … "| Register File | ~208 KB | 224 KB | 256 KB | 256 KB | 384 KB | 256 KB | 256 KB | 256 KB |", "| Shared Memory | ~60 KB | 64 KB …"
> "Pieces of silicon die spanning ~512 KB of registers/L1 and 32 bytes/cycle of L2 bandwidth."
> "On bandwidth bound tasks, the GPU has no advantage over the CPU. Because its I/O bus width is the same (32 bytes)."
Caveat: tables are M1/M2-era ("Apple 7, 8"). No M3/M4/M5 register-file data.

**2. WWDC22 session 10159, "Scale compute workloads across Apple GPUs"** — https://developer.apple.com/videos/play/wwdc2022/10159/ · official. Free transcript on the page.
What to read: the dispatch-model stretch from ~04:55 in the transcript. ~10 min. Verbatim:
> "A workload is dispatched in the form of a 3D grid of threadgroups." / "Threadgroups are uniformly distributed to the GPU cores" / "A single threadgroup is further broken down into SIMD-groups," / "Threadgroups can have up to 1024 threads per threadgroup" / "and threads can share up to 32K of threadgroup memory." / "Threadgroups are dispatched to GPU Clusters"
Contradiction to sit with: Apple says 32K threadgroup memory per threadgroup; metal-benchmarks measures "~60 KB" shared memory per core. Different quantities.

### Q2 — Writing a custom kernel from Python with `mx.fast.metal_kernel`

**1. The MLX doc in reStructuredText source** — https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/custom_metal_kernels.rst · official. ~20 min.
Why this URL: the rendered page is Sphinx with a ~1500-line nav sidebar; the .rst is clean and ahead of the published 0.32.3 docs. Verbatim:
> "Only pass the body of the Metal kernel in ``source``. The function signature is generated automatically."
> "All the attributes defined in Table 5.8 of the `Metal Shading Language Specification <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_ are supported."
> "By default :func:`fast.metal_kernel` compiles kernels with ``compile_options={\"math_mode\": \"safe\"}`` so special values follow IEEE behavior, for example ``exp(-inf) == 0``."
> "For optimal performance, each thread group dimension should be less than or equal to the corresponding grid dimension."

**2. aurelbzo/mlx-metal-kernels** — https://github.com/aurelbzo/mlx-metal-kernels · community. README, then `02_hash_encoder/`. ~45 min, runnable.
Why: the only free repo found that is explicitly a learning project on `mx.fast.metal_kernel` with a disciplined timing harness and committed CSVs plus a custom VJP. Verbatim:
> "The forward encoder launches one `mx.fast.metal_kernel` per resolution level. Each thread handles one 3D point, hashes its eight grid corners, and interpolates learned features."
> "Measured comparisons against MLX baselines on an Apple M1 Pro."
> "The script checks numerical agreement before timing, synchronizes each sample, rotates measurement order, and reports medians and median absolute deviations."
No third resource: NOT_FOUND. Medium piece "From 44 GFLOP/s to 376 GFLOP/s" → UNVERIFIED and metered.

### Q3 — `simdgroup_matrix` 8x8 tiles and Metal GEMM tiling

**1. Apple, "Metal Performance Primitives (MPP) Programming Guide, Version 1", dated 2026-03-16** — https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf (13 pages, no sign-up) · official. Read §2.2–2.3 (pp. 6–9), ~25 min.
Why: Apple itself writing the GEMM-tiling chapter nobody had before. Highest-value item in the set. Verbatim:
> "2.2 Key Distinctions of Apple Silicon GPUs ... 2.3 Key Optimizations for GEMM Kernels ... 2.3.1 Threadgroup Tile Size ... 2.3.2 Simdgroup Tile Size ... 2.3.3 Threadgroup Walk Order ... 2.3.4 Accumulation Loop Synchronization ... 2.3.5 Static Tensor Extents"
> "Simdgroup: Each simdgroup owns a tile of the output matrix"
> "Threadgroup: A group of adjacent simdgroup tiles combine to form a threadgroup tile"
> "matmul2d<desc, execution_simdgroup> op;"
> "4.1 Theoretical Analysis of GEMM Performance" (appendix, p. 12)
Also "Figure 2. Morton order curve" and "Figure 3. Cooperative tensor storage".

**2. MLX steel GEMM, `mma.h`** — https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/steel/gemm/mma.h (dir: gemm.h, loader.h, mma.h, params.h, transforms.h, nax.h) · official. Read `BaseMMAFrag<T,8,8>` then `loader.h`. ~1 h. Verbatim:
> `static_assert(kFragRows_ == 8, "Only 8 x 8 fragment matrices are currently supported");`
> `STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;`
> `STEEL_CONST int kElemRows = 1; STEEL_CONST int kElemCols = 2;`

**3. bkvogel/metal_performance_testing** — https://github.com/bkvogel/metal_performance_testing · community, 2022. Progressive naive→tiled→optimized ladder, CUDA-sample port. ~40 min. Verbatim:
> "The fastest one is currently able to reach roughly 1/2 the performance of Apple's optimized GPU `MPSMatrixMultiplication` kernel on an M1 Max."
> "`mat_mul_optimized_nv.metal`: This version uses shared threadgroup memory with a tiled algorithm. I directly ported it to Metal from the corresponding CUDA kernel in NVIDIA's cuda-samples"

**4. philipturner/metal-flash-attention README** — https://github.com/philipturner/metal-flash-attention · community. "Quantifying Performance" section only (~15 min). Verbatim:
> "The end result is a consistent 4400 gigainstructions per second on M1 Max (83% ALU utilization), at infinite sequence length and infinite head dimension."
> "Instead of gigaflops, I use gigainstructions to understand how well the shader is performing."
> "Apple hardware lacks native FP32 atomics (`metal::atomic<float>` is emulated)."

### Q4 — Affine int4 group quantization and quantized matvec on Metal

**1. MLX `quantize` docstring** — https://raw.githubusercontent.com/ml-explore/mlx/main/python/src/ops.cpp lines ~4796–4880 (rendered at the mlx.core.quantize API page) · official. ~10 min. Verbatim:
> "every ``group_size`` elements in a row of ``w`` are quantized together"
> "α = max_i w_i … β = min_i w_i … s = (α − β)/(2^b − 1) … ŵ_i = round((w_i − β)/s)."
> "for 4-bit quantization we fit 8 elements in an unsigned 32 bit integer where the 1st element occupies the 4 least significant bits, the 2nd bits 4-7 etc."
> Mode table rows: "affine  32, 64*, 128   2, 3, 4*, 5, 6, 8   same as input   yes" / "mxfp4  32*  4*  e8m0  no" / "nvfp4  16*  4*  e4m3  no" (`*` = default).

**2. MLX `quantized.h`** — https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h · official. Read `load_vector` → `qmv_fast`. ~1 h.
Why: `load_vector` pre-scales the activation by 1/4, 1/16, 1/64 so packed nibbles never need shifting out, and accumulates `sumy` for the bias term in the same pass. Verbatim:
> `MLX_MTL_CONST int SIMD_SIZE = 32; MLX_MTL_CONST int QUAD_SIZE = 4;`
> `template <int bits, int wsize = 8> inline constexpr short get_pack_factor() { return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits); }`
> `sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]; x_thread[i] = x[i]; x_thread[i + 1] = x[i + 1] / 4.0f; x_thread[i + 2] = x[i + 2] / 16.0f; x_thread[i + 3] = x[i + 3] / 64.0f;`

**3. llama.cpp Q4_0** — layout: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h ; kernel: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/kernels/mul_mv.metal · community/industry. ~45 min. Verbatim:
> `#define QK4_0 32` / `typedef struct { ggml_half d; // delta` / `uint8_t qs[QK4_0 / 2]; // nibbles / quants } block_q4_0;`
> mul_mv.metal: `inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il)` / `kernel void kernel_mul_mv_q4_0_f32(`
Freshness warning: single-file `ggml-metal.metal` no longer exists; kernels live in `ggml/src/ggml-metal/kernels/` (mul_mv.metal, mul_mm.metal, dequantize.h, …). Blogs pointing at `ggml-metal.metal` are stale.
NOT_FOUND: a free, current blog explaining the int4 layout with diagrams.

### Q5 — Profiling on Apple GPUs

**1. MLX "Metal Debugger" doc** — https://raw.githubusercontent.com/ml-explore/mlx/main/docs/src/dev/metal_debugger.rst · official. ~10 min. Verbatim:
> "You can build MLX with the ``MLX_METAL_DEBUG`` option … Records source during Metal compilation, for later inspection while debugging."
> "To capture a GPU trace you must run the application with ``MTL_CAPTURE_ENABLED=1``."
> "mx.metal.start_capture(trace_file)" … "You can open and replay the GPU trace in Xcode. The ``Dependencies`` view has a great overview of all operations."
> Links: https://developer.apple.com/documentation/xcode/metal-debugger

**2. dougallj/applegpu** — https://github.com/dougallj/applegpu, docs https://dougallj.github.io/applegpu/docs.html · community. ~20 min on the README. Verbatim:
> "This is a project working on reverse engineering the Apple G13 GPU architecture (as used by the M1), and hacking together documentation, a disassembler, an emulator and an assembler along the way. This is a somewhat messy work in progress"
> "$ python3 disassemble.py code.bin" → `0501040d00c43200     device_load      2, 0, i32, pair, 4, r0_r1, u2_u3, 0, lsl 1`
Caveat: targets G13/M1, not M3+.

**3. Apple's Xcode GPU-tooling docs** — INACCESSIBLE for verbatim quoting this run (WebFetch returned title only; obscura timed out). Only assertable URL: https://developer.apple.com/documentation/xcode/metal-debugger (linked by MLX docs). Re-verify before shipping.

### Q6 — Apple's own best material

1. **MPP Programming Guide, Version 1, 2026-03-16, 13 pp.** — https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf — HTTP 200, application/pdf, 6099349 bytes. Cover: "Metal Performance Primitives (MPP) Programming Guide / Version 1 / Developer / 2026-03-16". Read §2 end-to-end.
2. **WWDC22 session 10159** — transcript, not video. `<h1>Scale compute workloads across Apple GPUs</h1>`.
3. **MSL Specification PDF, Table 5.8** — pointer verified via MLX docs; table contents UNVERIFIED.

## Gaps
- No good free blog for `mx.fast.metal_kernel`. Lead (unverified): `meirm/VeloxQuant-MLX`.
- developer.apple.com hostile to obscura/WebFetch; plain `curl` with a Safari UA works and returns transcript spans.
- Q5 weakest: no verbatim Apple text on what the Shader Profiler / GPU counters expose; follow-up: curl the metal-debugger page and the "Optimize GPU renders with Metal debugger" WWDC session.
- Contradiction: 32K threadgroup memory per threadgroup (WWDC22) vs ~60 KB shared memory per core (metal-benchmarks).
- Staleness: metal-benchmarks stops at M2; applegpu targets M1/G13. MPP (Mar 2026) and MLX main are current.
- No community cross-check (no rdt/HN) — budget exhausted.

## Sources (independence)
1. philipturner (metal-benchmarks + metal-flash-attention) = 1. 2. Apple official (WWDC22, MPP PDF, MSL spec) = 1. 3. ml-explore/MLX (rst docs, ops.cpp, quantized.h, steel/mma.h) = 1, and Apple-authored so not fully independent of 2. 4. aurelbzo/mlx-metal-kernels. 5. bkvogel/metal_performance_testing. 6. ggml-org/llama.cpp. 7. dougallj/applegpu. WebSearch used for discovery only.
