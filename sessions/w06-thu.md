# w06-thu · The sequential scan, the baseline (hand-written)

## The spec

**`kernels/w06/thu_scan.metal`**, 30 minutes, as a `metal_kernel` body. Scalar gate only.

Shapes: q, k are [T, Dk]; v is [T, Dv]; g, beta are [T]; Dk = Dv = 128; one head. State S is [Dv, Dk], starting at zero. Output y is [T, Dv].

Grid (32, Dv, 1), threadgroup (32, 4, 1): one simdgroup per value row. Each lane keeps Dk/32 = 4 state floats in a local `float4` for the whole kernel.

Per token t, for this simdgroup's row r:
1. decay: s = s · g[t]
2. kv = simd_sum(dot(s, k_lane))    (this row's prediction, S[r, :]·k_t)
3. delta = (v[t, r] − kv) · beta[t]
4. s = s + delta · k_lane
5. out = simd_sum(dot(s, q_lane)); lane 0 stores y[t, r]

Write the loop over t inside the kernel. Nothing touches memory for S until the end.

Reference allowed open: the docstrings in mlx-lm `gated_delta.py` (lines 568 to 573) and `compute_g` at line 19. Not the kernel source.

## Rules
No AI. Claude does not write, fix, or suggest code today. Only the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
