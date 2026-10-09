# kernel-path (v2)

Six weeks of reading, about two hours a day, one page per day in `weeks/wN/dM.md`, to understand the Apple GPU stack and the
kernels people write for it, then NVIDIA for contrast, well enough to direct an AI to build kernels.
The ten-week build plan this replaced is on the `v1` branch.

## How a day works
Chandu reads the page (narrative, primary source, numbers), then writes answers to the five questions
in the "Your answers" section of the same file. He then runs `/learn read weeks/wN/dM.md`. Claude:
- grounds every probe in the page and its cited sources, never in general knowledge;
- marks the written answers, correcting only what is wrong, and writes cards to
  `~/learning/kernel-path/cards.md`;
- does not add exercises, code, or predictions to the pages. Weeks 4 and 6 end in a written brief
  (`weeks/w4/brief.md`, `weeks/w6/brief.md`), no code.

## Sources
Every number on a page carries a link. The research packs behind the pages are in `research/wN.md`
with verbatim quotes and the gaps each agent reported. If a claim is challenged, check the pack, then
the source; do not answer from memory. Dates matter: several gaps closed while the course was written
(MLX attention backward merged 2026-09-29; small-batch qmv merges 2026-10-04 and 10-06). Re-check any
"open gap" on the day it is used.
