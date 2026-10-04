# w08-thu · A reduction with no barrier (scaffold and walkthrough)

## The spec

**`kernels/w08/thu_nobarrier.metal`**, 30 minutes, as a `metal_kernel` body.

Input `a`: fp32 [N, 64], N rows of one chunk each. Output `out`: fp32 [N], the sum of each row.

One simdgroup per row. Each lane loads two values (lane l takes columns l and l+32) and adds them. Then reduce across the 32 lanes by hand with `simd_shuffle_xor` at offsets 16, 8, 4, 2, 1, adding at each step. Do not call `simd_sum`. Lane 0 stores. No `threadgroup_barrier` anywhere and no threadgroup memory.

Write in a comment why the order of those five steps makes the result identical across runs.

No reference page today.

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body with its structure written and the key expressions left as numbered `____` blanks. Then Claude walks through the blanks one at a time: what each must do, why, and where to look. I fill every blank myself. Claude gives the content of a blank only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
