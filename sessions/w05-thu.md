# w05-thu · bf16 GEMV in CUDA (hand-written)

## The spec

**`kernels/w05/thu_gemv.cu`**, 30 minutes, CUDA C++.

`gemv(const __half* W, const __half* x, float* y, int N, int K)` with N = K = 4096. One warp per output row: 32 lanes, lane l accumulates columns l, l+32, l+64, ... in a float, then a warp reduction with `__shfl_xor_sync` (offsets 16, 8, 4, 2, 1), and lane 0 writes y[row].

Launch shape: block of 128 threads = 4 warps = 4 rows per block; grid of N/4 blocks.

Plus a `main` that fills W and x, runs it, and compares against a CPU loop for the first 8 rows.

This is the same kernel as week 1 Wednesday in the other vocabulary. No reference open.

## Rules
No AI. Claude does not write, fix, or suggest code today. Only the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
