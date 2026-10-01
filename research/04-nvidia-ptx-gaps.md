# Agent B2 raw report: NVIDIA gaps follow-up (PTX dump, Triton internals, Blackwell SM100, headless ncu, inline PTX)

Provenance: follow-up subagent (Opus), run 2026-10-01, 14 of 15 tool calls, free text-first sources.
Access path: raw.githubusercontent.com, docs.nvidia.com (curl/grep), developer.nvidia.com, research.colfax-intl.com, triton-lang.org, modal.com/docs/llms.txt.
Caveats: num_warps and Triton vectorization still unverified; Kapil Sharma blog part 2 linked but unfetched; Modal has no ncu guide (null result on a title index). Verbatim below.

---
## Findings

### Gap 1 — PTX: dump it and read it

**1A (community): GPU MODE Lecture 29 "Triton Internals" — Kapil Sharma**
- https://github.com/gpu-mode/lectures/tree/main/lecture_029 (source: https://raw.githubusercontent.com/gpu-mode/lectures/main/lecture_029/presentation.md 18 KB; `vector_add.py` 2.4 KB runnable)
- Read: slides "Example: add_kernel (Artifacts)" and "Example: add_kernel (JIT Compiled)".
- VERBATIM: "`cuobjdump  add_kernel.cubin -sass -ptx`", "`nvdisasm -gi  add_kernel.cubin`", "File is elf formatted: `readelf -a add_kernel.cubin`"
- "- Artifacts that are dumped on the disk can also be retrieved directly from the kernel" → `print(compiled_kernel.asm.keys())` / `# dict_keys(['llir', 'ttgir', 'ttir', 'ptx', 'cubin'])`
- "- Several layers of code generation / Python DSL ➡️ IR ➡️ PTX ➡️ CUBIN/fatbinary ➡️ launcher.so"
- "`MLIR_ENABLE_DUMP=1 python vector_add.py`"; artifacts "add_kernel.ptx", "add_kernel.ttgir", "add_kernel.ttir", "add_kernel.cubin"
- Gist with all five IRs: https://gist.github.com/kapilsh/e8f09e8ed4f2f3bcfe13dd4bd099e270
- Time 40-60 min. Only free resource verified that connects `compiled.asm` to `cuobjdump -sass` with a 20-line vector add.

**1B (official): CUDA Binary Utilities** — https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html (v13.4)
- Read §1.2, §2.1.1 "Dumping SASS, ELF, and PTX (-sass, -elf, -ptx)", §3.1.4 "(-g, -gi, -gp)", §3.1.2 "(-cfg, -bbcfg)". ~20 min.
- VERBATIM: "cuobjdump extracts information from CUDA binary files (both standalone and those embedded in host binaries) and presents them in human readable format. The output of cuobjdump includes CUDA assembly code for each kernel, CUDA ELF section headers, string tables, relocators and other CUDA specific sections. It also extracts embedded ptx text from host binaries."
- VERBATIM: "nvdisasm extracts information from standalone cubin files ... nvdisasm also does control flow analysis to annotate jump/branch targets and makes the output easier to read."
- §4 "Instruction Set Reference", "4.3. Hopper Instruction Set" = SASS opcode tables.

**PTX ISA sections (official)** base https://docs.nvidia.com/cuda/parallel-thread-execution/index.html, titles verbatim 2026-10-01:
| Topic | Section + title | Anchor |
|---|---|---|
| `ld` (.v2/.v4) | "9.7.10.8. Data Movement and Conversion Instructions: ld" | `#data-movement-and-conversion-instructions-ld` |
| `ld.global.nc` | "9.7.10.9. Data Movement and Conversion Instructions: ld.global.nc" | `#data-movement-and-conversion-instructions-ld-global-nc` |
| `cp.async` | "9.7.10.28.3.1. Data Movement and Conversion Instructions: cp.async" | `#data-movement-and-conversion-instructions-cp-async` |
| commit/wait | "9.7.10.28.3.2 … cp.async.commit_group", "9.7.10.28.3.3 … cp.async.wait_group / cp.async.wait_all" | `#data-movement-and-conversion-instructions-cp-async-commit-group`, `…-cp-async-wait-group` |
| TMA | "9.7.10.28.5.3. Data Movement and Conversion Instructions: cp.async.bulk.tensor" | `#data-movement-and-conversion-instructions-cp-async-bulk-tensor` |
| `ldmatrix` | "9.7.16.5.15. Warp-level matrix load instruction: ldmatrix" | `#warp-level-matrix-instructions-ldmatrix` |
| `mma.sync` | "9.7.16. Warp Level Matrix Multiply-Accumulate Instructions" → "9.7.16.5. Matrix multiply-accumulate operation using mma instruction" | `#warp-level-matrix-instructions`, `#warp-level-matrix-instructions-for-mma` |
| `wgmma` | "9.7.17.5. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Operation using wgmma.mma_async instruction" | `#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async` |
- cp.async syntax: "cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size} [dst], [src], cp-size{, src-size}{, cache_policy} ;" with "cp-size = { 4, 8, 16 }".
- ldmatrix: "Collectively load one or more matrices from shared memory for mma instruction", "ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];" ".shape = {.m8n8, .m16n16};" ".num = {.x1, .x2, .x4};".

### Gap 2 — Triton internals

**Verified: GPU MODE Lecture 29 "Triton Internals", Kapil Sharma.** README verbatim: "## Lecture 29: Triton Internals / - Speaker: [Kapil Sharma](https://www.kapilsharma.dev/) / - Code/presentation in the [lecture_029](./lecture_029/) folder". Also "## Lecture 14: Practitioner's Guide to Triton — Date: 2024-04-13, Speaker: [Umer Adil]", "## Lecture 34: Low Bit Triton Kernels", "## Lecture 78: Iris: Multi-GPU Programming in Triton". No Blackwell lecture.
- Read slides "Triton GPU Passes" (two) and "Triton Passes". VERBATIM pass list: "- Coalescing / - F32 dot product optimization / - CTA planning / - Thread locality / - Matrix multiplication acceleration / - Optimization of dot operands / - Data de-duplication / - Instructions reordering / - TMA lowering, etc."
- `make_ttgir` slide: `passes.ttgpuir.add_remove_layout_conversions(pm)`, `passes.ttgpuir.add_accelerate_matmul(pm)`, `passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)`, `passes.ttgpuir.add_pipeline(pm, opt.num_stages)`.
- "Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators" (TritonGPUAccelerateMatmul slide).
- `num_warps`: NOT PRESENT in the deck. ~1 h.

**Kapil Sharma blog** — site index lists "Triton Kernels - Fused Softmax" (2025-09-18), "Triton Kernels - Fused Softmax - 2 — Worklog: Performance debugging Triton Kernel" (2025-09-20), "Triton Kernels - RMS Norm" (2025-09-17), "Learn CUTLASS the hard way!" (2025-11-01), part 2 (2025-12-31). Deck links "[Part 2](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/)" → UNVERIFIED, unfetched.

**Triton official docs** — nav: "Triton Semantics", "Triton MLIR Dialects and Ops", Programming Guide: "Introduction", "Related Work", "Debugging Triton", "Floating-Point Sanitizer (FpSan)"; top-level "Gluon" (Overview / Tutorials / Examples / API Reference). No "Triton internals" page. "Debugging Triton" (https://triton-lang.org/main/programming-guide/chapter-3/debugging.html): "To enable the interpreter mode, set the environment variable TRITON_INTERPRET to 1. This setting causes all Triton kernels to bypass compilation and be simulated by the interpreter using numpy equivalents of Triton operations." Nothing on PTX/layouts/num_warps: NOT PRESENT.

### Gap 3 — Blackwell SM100 / tcgen05 / tensor memory

**3A (industry): Colfax, "CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs"** — https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- VERBATIM: "Note that the consumer Blackwell architecture (compute capability 12.0) differs from the data center Blackwell architecture (compute capability 10.0) in some major ways, notably lacking Tensor Memory. We'll only discuss data center Blackwell in these posts."
- VERBATIM: "If you try to run a CUTLASS Hopper GEMM kernel on a Blackwell GPU, the first thing that you'll notice is that it doesn't work. The Hopper WGMMA instruction (in PTX, wgmma.mma_async) has been deprecated on Blackwell. To replace it, Blackwell introduced the tcgen05.mma instruction for MMA. In CUTLASS, tcgen05.mma is referred to as UMMA".
- VERBATIM: "Tensor Core dedicated memory called Tensor Memory to be used for UMMA accumulation." / "Two adjacent CTAs within an SM cluster, called a CTA pair, can work on the UMMA together across two SMs." / "Unlike WGMMA, only one thread is used to launch UMMA. Even if using two CTAs, only one thread in one CTA launches UMMA."
- "Operand A can be in TMEM or SMEM / Operand B must be in [SMEM]" (truncated; verify B line).
- Read Part 1, "An Overview of Blackwell MMA" → "Tensor Memory". Part map: "[Part 2] explains how to work with clusters, including new considerations around TMA multicast and Blackwell's CTA-pair concept. [Part 3] describes MMA with lower precision datatypes, and how Blackwell natively supports block-scaling with MMAs." 1.5-2 h.
- Other SM100 Colfax titles: "CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA® Blackwell GPUs", "CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs", "CUTLASS Tutorial: Sub-byte GEMM on NVIDIA® Blackwell GPUs", "Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs". NVFP4 posts are SM120 consumer: out of scope.

**3B (official): PTX ISA §9.7.18** — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory
- VERBATIM §9.7.18.1: "The 5th generation TensorCore has dedicated on-chip memory that is specialized for use by TensorCore operations. This Tensor Memory is organized as a two-dimensional matrix where the horizontal rows are called lanes and the vertical columns are called columns. On architecture sm_100a / sm_100f, the 5th generation TensorCore's Tensor Memory has a two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size."
- Sections: "9.7.18.1.1. Tensor Memory Addressing", "9.7.18.1.2. Tensor Memory Allocation", "9.7.18.7.1. … tcgen05.alloc, tcgen05.dealloc, tcgen05.relinquish_alloc_permit", "9.7.18.8.3 … tcgen05.ld", "9.7.18.8.4 … tcgen05.st", "9.7.18.8.5 … tcgen05.wait", "9.7.18.9.2 … tcgen05.cp", "9.7.18.10.10.1. … tcgen05.mma", "9.7.18.10.10.3 … tcgen05.mma.ws", "9.7.18.11.1 … tcgen05.fence", "9.7.18.12.1 … tcgen05.commit", "9.7.18.10.7. Block Scaling for tcgen05.mma". Read §9.7.18.1 + §9.7.18.7.1 + §9.7.18.8.3 (~30 min).

### Gap 4 — Nsight Compute headless on a cloud GPU

**4A (official): Nsight Compute CLI User Guide** — https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- `--set full`: "Identifier of section set to collect. If not specified, the basic set is collected. The full set of sections can be collected with --set full. ... Use --list-sets to see which set is the default."
- §5.3.4: "Using the --import option, saved reports can be imported into the command line profiler. When using this flag, most other options are not available, except for certain result filterting options." [sic]
- §5.3.5: "Using the --import and --export options together, along with supported filtering options, you can export desired results from one report to another."
- Headless gotcha: "Note that if --page is not used but --export is, no results will be printed to the console."
- 30-45 min. No permissions section on this page.

**4B (official): "ERR_NVGPUCTRPERM: Permission issue with Performance Counters"** — https://developer.nvidia.com/ERR_NVGPUCTRPERM
- VERBATIM: "ERR_NVGPUCTRPERM: The user running <tool_name/application_name> does not have permission to access NVIDIA GPU Performance Counters or the Hardware Event System on the target device."
- VERBATIM cause: "Your system administrator or a recent NVIDIA driver installation has disabled access to GPU Performance Counters for regular users due to Security Notice: NVIDIA Response to 'Rendered Insecure: GPU Side Channel Attacks are Practical' - November 2018. Your tool is affected by this restriction when using driver versions 419.17+ on Windows or 418.43+ on Linux."
- VERBATIM container case: "Linux Desktop: Launch the tool with sudo or as a user with the CAP_SYS_ADMIN capability set. Starting in driver version R565, the CAP_PERFMON capability will also allow access. When profiling within a container, access must be enabled on the host, or the container must be started with the appropriate permissions by passing --cap-add=SYS_ADMIN as an admin user." / "Note that CAP_PERFMON will not work in secure execution mode unless profiling within a container as described above."
- VERBATIM regkey: "To allow access for any user, create a file with the .conf extension containing options nvidia NVreg_RestrictProfilingToAdminUsers=0 in /etc/modprobe.d." / "A reboot may be required for the change to take effect."
- Currency: "Historically, profiling access was controlled via kernel registry keys. Starting with driver version R610, NVIDIA introduces a capabilities-based permission system that provides more granular control over profiling and tracing access." / "In a future release, the regkey-based method will be removed. The NVIDIA capabilities method described below replaces it." Headline "Quick Start: Grant Full Profiling Access to All Users (R610+)" iterating `profiler-device profiler-context trace-device` over `/proc/driver/nvidia-caps/sys-minors`, `chmod a+r /dev/nvidia-caps/nvidia-cap$minor`, `setfacl -m u:userA:r /dev/nvidia-caps/nvidia-cap4324`. Regkey path is "Legacy Kernel Regkey-Based Method". 20 min.

**Modal: NOT_FOUND.** https://modal.com/docs/llms.txt has no Nsight / ncu / GPU-profiling entry; https://modal.com/docs/guide/gpu no nsight/ncu match. Title index only; "Modal docs have no Nsight Compute guide," not "Modal cannot run ncu."

### Gap 5 — Inline PTX

**5a (official): `triton.language.inline_asm_elementwise`** — https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html
- VERBATIM signature: `triton.language.inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: dtype | Sequence[dtype], is_pure: bool, pack: int, _semantic=None)`
- "Execute inline assembly over a tensor. Essentially, this is map where the function is inline assembly. The input tensors args are implicitly broadcasted to the same shape."
- "Each invocation of the inline asm processes pack elements at a time. Exactly which set of inputs a block receives is unspecified. Input elements of size less than 4 bytes are packed into 4-byte registers." / "This op does not support empty dtype – the inline asm must return at least one tensor, even if you don't need it."
- Gluon note: "In Gluon, args may also contain shared or tensor memory descriptors. Each descriptor contributes one i32 address per invocation, independently of pack, and does not participate in broadcasting. Descriptor operands require is_pure=False".
- Runnable example (uint8 unpack + `cvt.rn.f32.s32` + `max.f32`) uses `$0..$8`; Triton uses `$n`, CUDA C++ uses `%n`. ~15 min.

**5b (official): "Inline PTX Assembly in CUDA"** — https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html (v13.4)
- "Assembler statements, asm(), provide a way to insert arbitrary PTX code into your CUDA program. A simple example is: asm("membar.gl;");"
- `asm("template-string" : "constraint"(output) : "constraint"(input));` / "Each %n in the template string is an index into the following list of operands, in text order. So %0 refers to the first operand, %1 to the second operand, and so on. Since the output operands are always listed ahead of the input operands, they are assigned the smallest indices."
- "The '=' modifier in '=r' specifies that the register is written to. There is also available a '+' modifier that specifies the register is both read and written"
- "If you want the % in a ptx instruction, then you should escape it with double %%, e.g.: asm("mov.u32 %0, %%clock;" : "=r"(x));"
- Read §1.1, then §1.2 Pitfalls in full: "1.2.1. Namespace Conflicts", "1.2.2. Memory Space Conflicts", "1.2.3. Incorrect Optimization", "1.2.4. Incorrect PTX", "1.3. Error Checking". 30 min.

## Gaps
- `num_warps`: no verified explainer. Vectorization: UNVERIFIED (nearest is the "Coalescing" pass).
- Kapil Sharma "deep-dive-into-triton-internals-2": linked, unfetched.
- No Triton docs "internals" page; nearest "Triton MLIR Dialects and Ops", "Triton Semantics", "Gluon" (unread).
- GPU MODE Blackwell lecture: NOT_FOUND. CUTLASS lectures: "Lecture 15: CUTLASS" (Eric Auld), "Lecture 36: CUTLASS and Flash ATtention 3" [sic].
- Nsight permissions: ERR_NVGPUCTRPERM page is load-bearing and alone; pre-R610 guides mislead.
- Colfax "Operand B must be in …" truncated.
- Not checked: Compiler Explorer CUDA mode, docs.nvidia.com/cutlass Blackwell pages, Modal page bodies.

## Sources
7 origins: docs.nvidia.com (PTX ISA, Inline PTX, Binary Utilities, Nsight CLI) = 1; developer.nvidia.com/ERR_NVGPUCTRPERM = same corporate origin; gpu-mode/lectures (Kapil Sharma); kapilsharma.dev (same author, not independent); research.colfax-intl.com (independent editorially; agrees with PTX ISA on SM100-only TMEM = two-origin confirmation); triton-lang.org; modal.com/docs (null).
