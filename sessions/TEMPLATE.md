# wNN-<mon|tue|wed|thu|fri> · <title>

## Before (written by me, before any code exists)
- The [three questions](https://metalworking.vercel.app/war-stories/three-questions/), answered before writing any kernel:
  1. Can I delete work?
  2. Can I unlock an existing fast path?
  3. Can I cut dispatch/sync overhead?
- Bound: memory / compute / latency / launch. Because:
- Plan: tile shape, threads per group, what lives in registers vs threadgroup memory:
- Prediction: ___ us  (roofline: ___ GB/s or TFLOPs implied, ___ % of peak)
- What would make me wrong:

## Ask (what I told Claude to write, one paragraph)

## After
- Measured: ___ us, x___ vs baseline, CV ___ %, roofline ___ %
- Gap between prediction and measurement, explained in my words:
- One thing I would try next:
