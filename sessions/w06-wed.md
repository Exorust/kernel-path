# w06-wed · The reference and the contract

## The question

Before any Metal code: fix the notation, build the slow correct answer, and state how wrong the fast kernel is allowed to be. What exactly will the kernel be checked against?

**Decide and write down.**
1. The convention. GDN: `S_t = S_{t-1}·(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`. Kimi Linear writes the same rule transposed. Write the one you will use for the whole arc, and the shape of S under it.
2. From FLA's `naive.py` run at C=4: the shape of `attn` after the forward-substitution loop, and in one sentence what entry [i, j] means.
3. Which FLA variable is the pseudo-value `U − Diag(γ)·W·Sᵀ` from GDN Eq. 11, and which line carries S to the next chunk.
4. Why chunkwise cannot be bit-identical to the scan. One sentence.
5. The tolerance: an NMSE bound against the fp64 sequential reference for fp32 state, and a separate bound for bf16 inputs. Pick numbers now and do not change them after optimizing.
6. The numerics rules: accumulation dtype, fast math on or off, where log space is used, what is never divided by.

**The Ask must name:** an fp64 numpy port of both `naive.py` functions, scalar and vector gate, the convention from item 1, a test at T=256 and C=64 comparing chunked to sequential, and the tolerances from item 5 as asserts.

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
