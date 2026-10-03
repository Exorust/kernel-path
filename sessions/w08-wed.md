# w08-wed · The full kernel and the 2x gate

## The question

The whole chunkwise kernel against mlx-lm's scan. Does it reach 2x at T=4096 on the vector-gate path, and which stage eats the time if it does not?

**Given.** Shape B=1, Hk=16, Hv=32, Dk=Dv=128, C=64. Baseline numbers from `sessions/w06-tue.md`. Single-chunk time from `sessions/w07-wed.md`. Barrier cost from Tuesday. The gate was fixed before any code: 2x or better at T=4096, or the result is published as a negative with its mechanism.

**Work out before running.**
1. Predicted time at T=4096 and T=8192: chunks times (single-chunk time + carry + barrier).
2. Predicted speedup over the scan at both lengths.
3. The stage you expect to dominate, and what you would change first if it does.
4. For the vector gate: what replaces the scalar exp(decay difference) in the intra-chunk mask, and why that is [C, Dk] instead of [C].
5. Whether the result will be bit-exact across runs, and what in the kernel guarantees it.

**After, in addition to the usual:** every place `chunk_kernel_v7.py` made a different choice from your directed kernel, one line each, with the reason v7's choice exists.

**The Ask must name:** the four stages plus carry, the staging plan between chunks, log-space decays, the numerics rules from week 6, and `harness.py all` against the mlx-lm scan.

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
