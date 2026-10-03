# w01-wed · The first real kernel: bf16 GEMV

## The question

Write the plan for a matrix-vector product y = W·x and predict its time before it exists. How close to the copy roof does it get, and does it beat `mx.matmul`?

**Given.** W is [4096, 4096] in bf16 = 32 MB. x is [4096] bf16 = 8 KB, small enough to stay cached. y is [4096]. Every weight is read exactly once, so bytes moved is about 32 MB. Roof: about 120 GB/s (`machine.json`, re-probe before the session).

**Work out before running.**
1. The floor time: 32 MB divided by about 120 GB/s (`machine.json`, re-probe before the session). Show the number in microseconds.
2. The thread mapping: how many output rows one simdgroup owns, how the 4096 columns are split across its 32 lanes, and where `simd_sum` goes. Draw it as one line, for example "simdgroup = 1 row, lane l reads columns l, l+32, l+64 ...".
3. Total threads launched with that mapping, and whether that meets 10K to 20K threads in flight on 10 cores.
4. Your predicted time and percent of roof. State whether you expect to beat `mx.matmul` and by how much.
5. One reason the kernel could land well under the roof (accumulating in fp32 vs bf16? x not staying in registers? too few elements per load?).

**The Ask must name:** the shape, the mapping from item 2, fp32 accumulation, `mx.matmul` as baseline, a max-abs-error check against it, `bytes_moved=32 MB`, and `mx.metal.start_capture` on one run.

## How to break this down

**What it computes.** Output row r is W[r,0]·x[0] + W[r,1]·x[1] + ... over 4096 columns. 4096 separate sums of 4096 terms each.

**Three decisions make the plan.**
1. Who owns one output row? Usually one simdgroup: 32 threads cooperate on one sum.
2. How do 32 threads split 4096 columns? Layout A: thread l takes columns l, l+32, l+64, ... so at every step the 32 threads read 32 neighbouring values. Layout B: thread l takes one contiguous run of 128 columns, so at every step the 32 threads read values 128 apart. Tuesday's table has an opinion on this.
3. How do 32 partial sums become one number? Each thread keeps its own; `simd_sum` adds them; one thread writes.

**Three numbers to work out.**
- Floor time: 32 MB read once, divided by the roof in `machine.json`.
- Threads launched: 4096 rows times 32. Compare with 10 to 20 thousand for 10 cores.
- Is memory still the limit? Tuesday only moved bytes. This also does 16.7 million multiply-adds. Decide whether that arithmetic becomes the new limit.

## Before (written by me, before any code exists)
- The [three questions](https://metalworking.vercel.app/war-stories/three-questions/): can I delete work? unlock an existing fast path? cut dispatch/sync overhead?
- Bound (memory / compute / latency / launch) and why: "Neither" (my answer, 2026-10-03). Correction of fact (Claude): the kernel needs about 125 billion ops/s at the floor against an estimated 3,300+ available, 4 percent, so math is far from its ceiling and memory is the bound.
- Plan (who owns what: thread, simdgroup, threadgroup; what lives in registers vs threadgroup memory): one simdgroup per row; layout A, "I think A works better" (lane l reads columns l, l+32, ...), chosen from Tuesday's table; each lane keeps its partial sum in a register; simd_sum; lane 0 writes. Threads: 131,072, "It's enough".
- Prediction, with the arithmetic shown: floor "260 microseconds" (32 MB / 120 GB/s = 267). Percent of roof: not predicted. Versus mx.matmul: "It will match afaik".
- What would make me wrong:

## Ask (what I told Claude to write, one paragraph)
Walked through the four questions with Claude, then Claude built from the plan above: [4096, 4096] half-precision W, one simdgroup per row, layout A, fp32 accumulation, simd_sum, baseline `mx.matmul`, parity against an fp32 reference, plus layout B as a second kernel to test the layout choice.

## After
- Measured (time, x vs baseline, CV %, roofline %): run 2026-10-03, `kernels/w01/wed_gemv.py` (local only), M5 32 GB, roof 122 GB/s, floor for 33.6 MB = 276 us.

  | Kernel | Time at [4096, 4096] | vs mx.matmul | Percent of roof | Max abs error vs fp32 |
  |---|---|---|---|---|
  | layout A | 570 to 613 us (CV 3 to 4%) | 0.86x (slower) | 45 to 48 | 1.5e-06 |
  | layout B | 638 us (CV 8.1%, flagged) | 0.80x (slower) | 43 | 1.3e-06 |
  | mx.matmul, half | about 476 to 527 us | 1.00 | about 52 to 58 | 1.9e-03 |

  A against B directly: 0.95x, inside the noise. No measurable layout difference at this shape.

  Height sweep, same kernel, K = 4096:

  | Rows | Bytes | layout A | mx.matmul |
  |---|---|---|---|
  | 64 | 0.5 MB | 205 us | 189 us |
  | 1,024 | 8.4 MB | 260 us | 246 us |
  | 4,096 | 33.6 MB | 606 us | 476 us |
  | 16,384 | 134.2 MB | 1,605 us | 1,312 us |

  Marginal rate from 4,096 to 16,384 rows: layout A 101 GB/s, mx.matmul 120 GB/s. Note (Claude): W is IEEE half (`mx.float16`), not bfloat16; the plan said "bf16" loosely. Same 2 bytes per weight.
- Gap between prediction and measurement, explained in my words:
- One thing I would try next:
