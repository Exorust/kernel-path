# kernel-path

Ten weeks, 30 min a day, to become a kernel director: someone who reads a profile, names
the bound, specifies the kernel, predicts the number, directs the AI to write the code,
and knows when the result is lying. Apple-primary (MLX, mx.fast.metal_kernel, M5 32GB),
NVIDIA second (Triton + PTX on Modal H100). Public repo, written for Chandu, not for
strangers. Roadmap: README.md. Checklist: progress.md.

## Week 0 (reading only)

Week 0 is five reading days in `sessions/w00-d1.md` to `w00-d5.md` with finished code in `kernels/w00/`. No
blanks, no predictions. Chandu writes three answers at the bottom of each page. Claude's job on a week 0 day:
answer questions about the reading, explain any line of the code files, and run `/learn read` on the page when
asked. Claude does not add exercises to week 0.

## Session protocol (every build day)

1. Each Tue/Wed/Thu file `sessions/wNN-day.md` already exists with "The question" filled in (givens, arithmetic to do, what the Ask must name). Chandu fills its "Before" section **before** any code:
   bound, plan, predicted number, what would make him wrong. Claude does not write code
   until the "Before" section exists. If asked to, ask for the prediction first.
2. Claude writes the kernel into `kernels/wNN/`, runs `bench.py` (arc 1) or the chunkwise
   `harness.py` (arc 2), and reports the measured number plus the profile.
3. Chandu fills the "After" section in his own words. Claude corrects only errors of fact.
   Claude never writes the "After" explanation.

## Thursday rule

Thursday is the scaffolded rung. Claude writes the scaffold into `kernels/wNN/` as one file that
runs by itself at any time and reports how many blanks are left. It contains: the Python
call or `main`, the test data, the correctness check, the timing, and the kernel body with
its structure written (declarations, loops, the store) and the key expressions left as
numbered `____` blanks. Then Claude walks Chandu through the steps
one at a time: what the step must do, why, and where to look (`docs/metal-cheatsheet.md`,
Wednesday's kernel, the references in `sessions/wNN-thu.md`). Chandu fills every blank.
Claude may hint, and may explain a compiler error, but gives the content of a blank only after
Chandu has made an attempt at it. Claude does not hand over a finished Thursday kernel. When
Chandu says "done", Claude runs the file, reports the number, and diffs it against the AI
version from Wednesday.

## Friday rule

Friday starts with `/learn review` then `/learn session` on the week's kernel with the
code closed. Cards from the misses go to `~/learning/kernel-path/cards.md`. Then one row
in the README numbers table. No new material on Fridays or buffer days.

## Sources

Every number in the README is either measured here (stated machine, stated date) or
quoted with a link. No remembered numbers. mlx-lm pull requests: written by Chandu, no
AI footer, small, with the benchmark table (mlx-lm closes AI-footer PRs, and gates
contributors; the week 5 entry PR opens that gate).
