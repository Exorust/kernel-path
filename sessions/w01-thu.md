# w01-thu · Vector add, then a simdgroup reduction (hand-written)

## The spec

Two files, 15 minutes each, as `mx.fast.metal_kernel` bodies (body only; MLX generates the signature).

**`kernels/w01/thu_vadd.metal`.** Inputs `a`, `b`: fp32 arrays of 2^20 elements. Output `out` = a + b. One element per thread. You need: the thread's index in the grid, two loads, one store. Then decide `grid` and `threadgroup` for the Python call and write them in a comment at the top.

**`kernels/w01/thu_reduce.metal`.** Input `a`: fp32, 2^20 elements. Output `out`: 2^15 elements, where out[i] is the sum of a[32i .. 32i+31]. One simdgroup per output element: each lane loads one value, `simd_sum` adds the 32, and exactly one lane writes. You need: the lane index, the simdgroup's index, and a guard so only lane 0 stores.

Reference page allowed open: https://metalworking.vercel.app/metal/msl/

If `thu_vadd` does not compile within 15 minutes, stop and record the error. Do not start the second file.

## Rules
No AI. Claude does not write, fix, or suggest code today.

**Weeks 1 to 3 ramp.** Before starting, read Wednesday's AI-written kernel in `kernels/` for 5 minutes, then close it. While you write, `docs/metal-cheatsheet.md` is open, plus any reference page named above. From week 4, no cheat sheet.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
