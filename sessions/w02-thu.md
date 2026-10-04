# w02-thu · Dequantize one group of 64 (scaffold and walkthrough)

## The spec

**`kernels/w02/thu_dequant.py`**, 30 minutes, as a `metal_kernel` body.

Inputs: `w`, uint32, 8 words (one group: 64 nibbles, first value in the 4 least significant bits of word 0); `scale` and `bias`, one fp32 each. Output: `out`, 64 fp32 values with out[i] = scale · q[i] + bias.

Use 64 threads, one output each. Each thread must work out which word holds its nibble (index / 8), which nibble inside the word (index % 8), shift and mask, convert to float, scale, add bias, store.

Check you will run in Python afterwards: `mx.quantize` a random [1, 64] row with group_size 64 and bits 4, feed its outputs in, compare against `mx.dequantize`.

The packing rule is stated above; the syntax is on the cheat sheet.

## Rules
Scaffold and walkthrough. Claude writes the scaffold first: the Python call or `main`, the test data, the correctness check, and the kernel body with its structure written and the key expressions left as numbered `____` blanks. Then Claude walks through the blanks one at a time: what each must do, why, and where to look. I fill every blank myself. Claude gives the content of a blank only after I have made an attempt at it. `docs/metal-cheatsheet.md`, Wednesday's kernel, and the reference pages named above may be open.
Say "done" and Claude runs the file and reports: compiled or the verbatim compiler error, output matches or not, time.

## After
- Compiled first try? If not, the error and what I had wrong:
- Matches the reference output?
- Time vs Wednesday's AI version:
- The step I needed the most help on:
