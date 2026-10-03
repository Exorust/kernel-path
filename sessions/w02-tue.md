# w02-tue · How many bytes does dequant move?

## The question

You will time a kernel that only converts packed int4 weights back to bf16. How long should it take, and which side, reading or writing, dominates?

**Given.** W is [4096, 4096], 4 bits, group_size 64. Packed weights: 8 nibbles per uint32, so 4096·4096/8 words = 8 MB. Scales and biases: one bf16 pair per group of 64 = 4096·64 groups · 4 bytes ≈ 1 MB. Output: 32 MB of bf16. Roof: 93 GB/s.

**Work out before running.**
1. Bytes in, bytes out, total.
2. Floor time from the total and the roof.
3. One thread per packed word produces 8 outputs. How many threads is that, and is it enough to saturate 10 cores?
4. Predicted time and percent of roof, and the predicted ratio to `mx.dequantize`.
5. For contrast: llama.cpp's `block_q4_0` is 32 weights with one fp16 scale and no bias in 18 bytes. Bits per weight for that format vs MLX's (4 bits + 32 bits per 64 weights). Which is denser?

**The Ask must name:** the shape and format, one thread per uint32, the unpack by shift and mask, `w = scale·q + bias`, `mx.dequantize` as baseline with a max-abs-error check, and `bytes_moved` from item 1.

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
