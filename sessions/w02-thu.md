# w02-thu · Dequantize one group of 64 (hand-written)

## The spec

**`kernels/w02/thu_dequant.metal`**, 30 minutes, as a `metal_kernel` body.

Inputs: `w`, uint32, 8 words (one group: 64 nibbles, first value in the 4 least significant bits of word 0); `scale` and `bias`, one fp32 each. Output: `out`, 64 fp32 values with out[i] = scale · q[i] + bias.

Use 64 threads, one output each. Each thread must work out which word holds its nibble (index / 8), which nibble inside the word (index % 8), shift and mask, convert to float, scale, add bias, store.

Check you will run in Python afterwards: `mx.quantize` a random [1, 64] row with group_size 64 and bits 4, feed its outputs in, compare against `mx.dequantize`.

The packing rule is stated above; the syntax is on the cheat sheet.

## Rules
No AI. Claude does not write, fix, or suggest code today.

**Weeks 1 to 3 ramp.** Before starting, read Wednesday's AI-written kernel in `kernels/` for 5 minutes, then close it. While you write, `docs/metal-cheatsheet.md` is open, plus any reference page named above. From week 4, no cheat sheet.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The line I could not write from memory:
