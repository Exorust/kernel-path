# w07-wed · The single-chunk kernel

## The question

One chunk of C=64 tokens, no state coming in. Count the work, then predict the time. Which of the four stages costs the most?

**Given.** C=64, Dk=Dv=128, one head, scalar gate. Stages: (a) K·Kᵀ, a [64,128]·[128,64] product, lower triangle only; (b) the unit-lower-triangular solve on the 64 by 64 result; (c) W = T·Diag(β)·K and U = T·Diag(β)·V, two [64,64]·[64,128] products; (d) outputs and chunk-end state from Q, K, U.

**Work out before running.**
1. FLOPs for each of (a) to (d). A [m,k]·[k,n] product is 2·m·k·n.
2. Number of 8x8 tiles in a 64 by 128 block and in a 64 by 64 block.
3. Which stages are 8x8-tileable as written and which one is sequential by nature. For the sequential one, how many steps.
4. Total FLOPs, and the time at Tuesday's measured GFLOPS.
5. Where the log-space decays enter: which matrices get multiplied elementwise by exp of a difference of cumulative sums.

**The Ask must name:** the four stages, the convention from week 6, simdgroup_matrix tiling for (a), (c), (d), the forward-substitution form for (b), fp32 accumulation, and the fp64 reference as the gate.

## Before (written by me, before any code exists)
- The [three questions](https://metalworking.vercel.app/war-stories/three-questions/): can I delete work? unlock an existing fast path? cut dispatch/sync overhead?
- Bound (memory / compute / latency / launch) and why:
- Plan (who owns what: thread, simdgroup, threadgroup; what lives in registers vs threadgroup memory):
- Prediction, with the arithmetic shown:
- What would make me wrong:

## Ask (what I told Claude to write, one paragraph)

## After
- Measured (time, x vs baseline, CV %, roofline %):
- Gap between prediction and measurement, explained in my words:
- One thing I would try next:
