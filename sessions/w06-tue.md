# w06-tue · Why does the scan lose?

## The question

mlx-lm's gated-delta kernel walks the sequence one token at a time. How does its time grow with sequence length T, and what does that say about how much of the GPU it uses?

**Given.** Vector-gate shape: B=1, Hk=16 key heads, Hv=32 value heads, Dk=Dv=128. The kernel launches grid (32, Dv, B·Hv): one simdgroup per (value head, value row), each lane holding Dk/32 = 4 state floats in registers. Per token it does a decay and dot, a delta, and an update and dot.

**Work out before running.**
1. Threads launched: 32 · Dv · B · Hv. Is that above or below the 10K to 20K needed to saturate 10 cores? Does it depend on T?
2. If per-token work is constant, time is proportional to T^1. Predict the measured exponent between T=512 and T=8192, and say what would push it above 1.
3. Bytes read per token for the vector gate: g is [B, T, Hv, Dk], and every value row of a head needs the same gate. How many times is each gate value read?
4. Predicted time at T=4096 in milliseconds.

**The Ask must name:** the shape, T in {512, 1024, 4096, 8192}, l2-normalized q and k (unnormalized inputs diverge past T about 70), evaluating every iteration, a log-log fit for the exponent, and the vector-gate path of `gated_delta_update`.

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
