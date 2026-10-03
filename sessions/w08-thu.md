# w08-thu · A reduction with no barrier (hand-written)

## The spec

**`kernels/w08/thu_nobarrier.metal`**, 30 minutes, as a `metal_kernel` body.

Input `a`: fp32 [N, 64], N rows of one chunk each. Output `out`: fp32 [N], the sum of each row.

One simdgroup per row. Each lane loads two values (lane l takes columns l and l+32) and adds them. Then reduce across the 32 lanes by hand with `simd_shuffle_xor` at offsets 16, 8, 4, 2, 1, adding at each step. Do not call `simd_sum`. Lane 0 stores. No `threadgroup_barrier` anywhere and no threadgroup memory.

Write in a comment why the order of those five steps makes the result identical across runs.

No reference page today.

## Rules
No AI. Claude does not write, fix, or suggest code today. Only the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
