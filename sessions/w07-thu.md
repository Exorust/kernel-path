# w07-thu · One 8x8 simdgroup_matrix multiply (hand-written)

## The spec

**`kernels/w07/thu_sgmm.metal`**, 30 minutes, as a `metal_kernel` body.

Inputs `a`, `b`: fp32 [8, 8]. Input `c`: fp32 [8, 8]. Output `out` = a·b + c.

One simdgroup. You need: three `simdgroup_matrix<float, 8, 8>` declarations, `simdgroup_load` for each from its buffer, `simdgroup_multiply_accumulate`, and `simdgroup_store` to `out`. Decide the grid and threadgroup sizes and write them in a comment.

Check in Python against `a @ b + c`.

Reference pages allowed open: https://metalworking.vercel.app/metal/simdgroup-matrix/ and `mat_mul_optimized_nv.metal` in bkvogel/metal_performance_testing.

## Rules
No AI. Claude does not write, fix, or suggest code today. Only the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
