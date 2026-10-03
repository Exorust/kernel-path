# kernel-path

Ten weeks, 30 min a day, to become a kernel director: someone who reads a profile, names
the bound, specifies the kernel, predicts the number, directs the AI to write the code,
and knows when the result is lying. Apple-primary (MLX, mx.fast.metal_kernel, M5 32GB),
NVIDIA second (Triton + PTX on Modal H100). Public repo, written for Chandu, not for
strangers. Roadmap: README.md. Checklist: progress.md.

## Session protocol (every build day)

1. Each Tue/Wed/Thu file `sessions/wNN-day.md` already exists with "The question" filled in (givens, arithmetic to do, what the Ask must name). Chandu fills its "Before" section **before** any code:
   bound, plan, predicted number, what would make him wrong. Claude does not write code
   until the "Before" section exists. If asked to, ask for the prediction first.
2. Claude writes the kernel into `kernels/wNN/`, runs `bench.py` (arc 1) or the chunkwise
   `harness.py` (arc 2), and reports the measured number plus the profile.
3. Chandu fills the "After" section in his own words. Claude corrects only errors of fact.
   Claude never writes the "After" explanation.

## Thursday rule (protected, do not negotiate)

Thursday is the hand-written rung. Claude does **not** write, fix, complete, or suggest
kernel code on a Thursday session, even when asked. Claude may: point to the spec in `sessions/wNN-thu.md`, run the
file when Chandu says "done", report the number, and diff it against the AI version from
Wednesday. If a Thursday file does not compile, Claude reports the compiler error verbatim
and stops.

## Friday rule

Friday starts with `/learn review` then `/learn session` on the week's kernel with the
code closed. Cards from the misses go to `~/learning/kernel-path/cards.md`. Then one row
in the README numbers table. No new material on Fridays or buffer days.

## Sources

Every number in the README is either measured here (stated machine, stated date) or
quoted with a link. No remembered numbers. mlx-lm pull requests: written by Chandu, no
AI footer, small, with the benchmark table (mlx-lm closes AI-footer PRs, and gates
contributors; the week 5 entry PR opens that gate).
