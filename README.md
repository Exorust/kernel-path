# kernel-path

Six weeks, thirty minutes a day, to understand the Apple GPU stack and the kernels people write for it well enough to build your own by directing an AI. Four weeks on Apple silicon, two on NVIDIA for contrast and for the kernels that made news there.

Each day is one page in `weeks/`: a short narrative with a citation on every number, the one primary source to read, the numbers to remember, and five questions. You write your answers at the bottom of the page and run `/learn read weeks/wN/dM.md` to have them checked and turned into cards. Nothing to install until you choose to run something.

| Week | Theme | Days |
|---|---|---|
| 1 | The Apple stack, big picture | the chip · the GPU core · Metal · MLX · who writes Apple kernels today |
| 2 | Kernel families and the ideas that carry across hardware | roofline on Apple · elementwise, reduction, scan, softmax · matrix multiply · attention · quantized matvec |
| 3 | Apple case studies with large speedups | llama.cpp's Metal backend · MLX steel and the M5 neural accelerators · linear attention on MLX · expert paging and MoE · launch overhead at scale |
| 4 | Building kernels with an AI, on Apple | the feedback-loop problem · profiling on Apple · AI-generated kernels, the evidence · directing an AI · capstone brief |
| 5 | NVIDIA, the big picture | architecture against Apple · CUDA, PTX, Triton, CUTLASS · the matmul ladder · Hopper: TMA and wgmma · low precision and Blackwell |
| 6 | NVIDIA kernels that made news | FlashAttention 1 to 4 · megakernels · serving kernels · linear attention on NVIDIA · profiling, AI kernels, second capstone brief |

Pages are being written week by week; a day with only a title is not written yet. Raw research behind the pages is in `research/`, one report per week. The previous version of this repository, a ten-week build plan, is on the `v1` branch.
