# w02-wed · The fused int4 GEMV

## The question

Fuse the dequant into the matvec so no bf16 matrix is ever written. How much faster than week 1's bf16 GEMV should it be, and how close to `mx.quantized_matmul`?

**Given.** Same W as Tuesday: 8 MB packed + 1 MB of scales and biases. x is [4096] bf16. Week 1 Wednesday's measured time for the 32 MB bf16 GEMV is in `sessions/w01-wed.md`. Roof: 93 GB/s.

**The trick you read in `load_vector`.** For a group, y = Σ (s·q_i + β)·x_i = s·Σ q_i·x_i + β·Σ x_i. So the bias costs one multiply per group if you keep a running sum of x, and the nibbles can be used in place if x is pre-scaled by 1, 1/16, 1/256 ... per nibble position inside a word.

**Work out before running.**
1. Speedup over week 1 from the byte ratio alone (32 MB vs 9 MB).
2. Why the real speedup will be lower: name the extra work per weight that the bf16 kernel did not do.
3. Whether the kernel is still bandwidth-bound, or whether 4-bit unpacking makes it compute-bound. Give the arithmetic intensity argument in one sentence.
4. Predicted time, percent of roof, and the ratio to `mx.quantized_matmul`.

**The Ask must name:** the format, the two-term identity above, x pre-scaling held in registers, fp32 accumulation, baseline `mx.quantized_matmul`, parity against `mx.dequantize` then `mx.matmul`, and `bytes_moved=9 MB`.

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
