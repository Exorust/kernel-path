# w10-wed · Arc 2 writeup

## The question

Write `writeups/arc2.md`. Outline here first. What does the chunkwise kernel do for prefill on Apple silicon, and what did the NVIDIA comparison teach?

**Answer these, two or three sentences each, before writing prose.**
1. The one-sentence result: speedup over the mlx-lm scan at T=4096 and 8192, vector gate, machine and date. If the gate was missed, say so first.
2. The lineage: each kernel version, its number, and why the next one exists.
3. What was rejected and the mechanism (barriers, register cliff, anything new).
4. The contract: tolerance against fp64, determinism across runs, numerics rules. What a reviewer can rerun.
5. The honest ceiling: how much of end-to-end prefill this kernel is, so the model-level speedup is stated and not implied.
6. H100 side: FLA vs directed Triton, what Nsight showed, what the Mac has no equivalent for.
7. Credits: FLA, FlashQLA, the closed mlx-lm pull requests #1241 and #1389, and that an AI wrote the code under direction.
8. What goes upstream on Friday and in what form.

No "Before" prediction today. The "After" is the commit hash of the writeup.

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
