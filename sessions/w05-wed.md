# w05-wed · Arc 1 writeup

## The question

Write `writeups/arc1.md`. This file is the outline; fill it here first, then expand there. What did five weeks of quantized matvec work actually show?

**Answer these, two or three sentences each, before writing prose.**
1. The one-sentence result, with the best measured number and its machine and date.
2. What was tried, in order, as a list of kernel versions with their numbers.
3. What failed, and the mechanism for each failure. A failure with no mechanism is not ready to publish.
4. The honest ceiling: how far the best kernel is from the roof on each platform, and what is left.
5. What Nsight showed that Xcode could not, and the reverse.
6. Every caveat a skeptical reader would raise: cache-resident working sets, thermal state, format differences against Marlin, single machine.

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
