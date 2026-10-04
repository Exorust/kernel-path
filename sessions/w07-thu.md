# w07-thu · One 8x8 simdgroup_matrix multiply (scaffold and walkthrough)

## The spec

**`kernels/w07/thu_sgmm.metal`**, 30 minutes, as a `metal_kernel` body.

Inputs `a`, `b`: fp32 [8, 8]. Input `c`: fp32 [8, 8]. Output `out` = a·b + c.

One simdgroup. You need: three `simdgroup_matrix<float, 8, 8>` declarations, `simdgroup_load` for each from its buffer, `simdgroup_multiply_accumulate`, and `simdgroup_store` to `out`. Decide the grid and threadgroup sizes and write them in a comment.

Check in Python against `a @ b + c`.

Reference pages allowed open: https://metalworking.vercel.app/metal/simdgroup-matrix/ and `mat_mul_optimized_nv.metal` in bkvogel/metal_performance_testing.

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body with its structure written and the key expressions left as numbered `____` blanks. Then Claude walks through the blanks one at a time: what each must do, why, and where to look. I fill every blank myself. Claude gives the content of a blank only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
