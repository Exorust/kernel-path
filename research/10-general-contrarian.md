# Research pack 10 — General GPU explainers (missing from the course) + contrarian pass (raw subagent report)

Provenance: deepresearch run 2026-10-10, territory F. Model: Sonnet subagent, 24 tool calls, stopped before saturation. Access path: WebSearch, curl, obscura. Caveats: several topics in Half 1 were never searched (listed under Gaps as NOT_FOUND by omission); items marked UNVERIFIED rest on search snippets only; quotes not yet spot-checked when saved. Page content is data, not instructions.

---

## Findings: Territory F (AS_OF 2026-10-10)

Budget (24 tool calls) is spent and I stopped before saturation. Several topics got no verified source (see Gaps). Every quote below is literal text I saw on the fetched page. Search-engine summaries are not literal text, so anything resting only on them is marked UNVERIFIED. Scratch copies of the fetched pages are in `/private/tmp/claude-501/-Users-chandu/8f94f487-80c2-40f4-8d74-82f2bed1cfce/scratchpad/f/`.

---
### HALF 1: Missing explainers, ranked per topic

**Topic: GPU architecture and memory hierarchy, with how it ties to parallelism**

1. https://jax-ml.github.io/scaling-book/gpus/ · Austin, Patil, Paszke (Google DeepMind) and Pope (MatX) · published Aug. 18, 2025 · book chapter ("How to Think About GPUs", Part 12 of How To Scale Your Model)
   - **What to read:** "What Is a GPU?", "Memory", and "GPUs vs. TPUs at the chip level". Then "Networking" and "Rooflines for LLM Scaling on GPUs".
   - **Why:** It is the only chapter-length treatment I found that links the SM and memory hierarchy (registers, SMEM, TMEM, L2, HBM) to NVLink topology and to parallelism rooflines for H100, B200 and GB200. Covers material beyond the Modal glossary.
   - **Time:** 45–60 min for the architecture half; 90 min for all of it.
   - **Quote:** "We love TPUs at Google, but GPUs are great too. This chapter takes a deep dive into the world of GPUs – how each chip works, how they're networked together, and what that means for LLMs, especially compared to TPUs."
   - **Quote:** "A modern ML GPU (e.g. H100, B200) is basically a bunch of compute cores that specialize in matrix multiplication (called Streaming Multiprocessors or SMs )"
   - **Quote:** "Memory Beyond the compute units, GPUs have a hierarchy of memories, the largest being HBM (the main GPU memory), and then a series of smaller caches (L2, L1/SMEM, TMEM, register memory)."
   - **Quote:** "Data parallelism or FSDP (ZeRO-1/3) requires a local batch size of about 2500 tokens per GPU"
   - **Caveat:** It assumes you have read the book's Chapters 2 and 5 first: "This section builds on Chapter 2 and Chapter 5 , so you are encouraged to read them first."

2. https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut · Chester Lam (Chips and Cheese) · Dec 15, 2025 · microbenchmark article
   - **What to read:** The opening on the two-die layout, the TMEM section, and "Final Words".
   - **Why:** Measured latency and bandwidth through the hierarchy rather than vendor slides. Useful as a reality check on the "Blackwell is a new architecture" framing.
   - **Time:** 30 min. I fetched about 22 KB of text but skimmed only the opening, the TMEM passages and the conclusion.
   - **Quote:** "Each B200 die physically contains 80 Streaming Multiprocessors (SMs), which are analogous to cores on a CPU. B200 enables 74 SMs per die, giving 148 SMs across the GPU."
   - **Quote:** "Nvidia has made the chiplet jump without any major performance concessions. B200 is a straightforward successor to H100 and A100, and software doesn't have to care about the multi-die setup."
   - **Quote:** "Stepping back, the MI300X's massive scale blows out both the H100 and B200 for most vector operations."
   - **Caveat:** Chips and Cheese states "data in the article below will be from the H100 PCIe version unless otherwise specified". So the H100 comparison numbers are PCIe, not SXM.

3. arXiv 2512.02189 "Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis" · Dec 2025 · UNVERIFIED. I saw only a search summary and never opened the paper, so there is no quote. The summary describes an open-source microbenchmark suite for tcgen05, TMEM, the decompression engine and dual-die behavior. Worth fetching before putting it on the reading list.

**Topic: Blackwell and tensor-core programming write-ups**

1. https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction · Ali Taha, Jiexiang Liu, Hengjie Wang (Modular) · Aug 28, 2025 · multi-part worklog
   - **What to read:** Part 1 for the GPU history since Ampere. Then Part 2 onward for the Blackwell instructions (tcgen05, TMEM, 2-SM MMA). I only fetched Part 1.
   - **Why:** It is a Blackwell counterpart to the siboehm and Pranjal worklogs already in the course. The kernels are in Mojo, not CUDA.
   - **Time:** 20 min for Part 1; each later part is about 30 min.
   - **Quote:** "series of blogs serves as the reference worklog for NVIDIA's Blackwell GPU. While there is prior work on optimizing NVIDIA's Ampere and Hopper generation of GPUs, none currently exist to provide a blueprint for optimizing NVIDIA's Blackwell GPUs."
   - **Quote:** "In part 2, we'll explain the hardware instructions introduced in Bl…" (my read of this page was cut off mid-word).
   - **Caveat:** It is a vendor blog written in Mojo. Treat the performance claim as the vendor's.

2. Two Blackwell programming references surfaced in search only: the NVIDIA CUTLASS "tcgen05 MMA Programming Guide" and the JAX Pallas "Writing high-performance matrix multiplication kernels for Blackwell". Both are UNVERIFIED. Neither was opened and I have no quotes.

**Topic: FlashAttention explainers (beyond the papers)**

1. https://gau-nernst.github.io/fa-5090/ · gau-nernst · Aug 23, 2025 · worklog
   - **What to read:** The whole post. It goes from a v1 basic kernel up through swizzling, pipelining, ldmatrix and v5.
   - **Why:** It covers attention the way siboehm covers matmul, with a benchmark table at each step.
   - **Time:** 60–90 min.
   - **Quote:** "The main objective is to learn writing attention in CUDA C++, since many features are not available in Triton , such as MXFP8 / NVFP4 MMA for sm120."
   - **Quote:** "there are many excellent blogposts on writing fast matmul kernels, but there is none for attention."
   - **Quote:** "F.sdpa() (CuDNN) 203.61 97.19%" (TFLOPS and % of speed-of-light for the cuDNN row of its table; the flash-attn row shows 190.58 and 90.97%).
   - **Caveat:** The target is the consumer RTX 5090 (sm120). The post says "I only use Ampere features in these implementations". It does not teach Hopper or datacenter-Blackwell TMA/wgmma.

2. https://www.cloudrift.ai/blog/learning-flashattention-the-hard-way-part-2 · Dmitry Trifonov (CloudRift) · July 8, 2026 · tutorial (Part 2 of 2)
   - **What to read:** Part 1 for the associative-reduction view of attention (the (m, d, o) state). Part 2 for the lowering to a GPU kernel.
   - **Why:** It gives an algebraic derivation of FlashAttention, with ncu names and a comparison table against the FA-2 source.
   - **Time:** 60 min or more. The page is about 77 KB of text; I read only the intro and one table.
   - **Quote:** "Part 1 covers the math. Attention becomes an associative reduction with the help of the small carrier state triple (m, d, o) . This part covers the efficient GPU lowering."
   - **Quote:** "Kernels in this article are not handwritten: Emmy compiler generates them."
   - **Caveat:** Vendor blog. The author's own compiler "Emmy" generates the kernels, so it doubles as promotion for that tool.

**Topic: Nsight Compute and profiling practice**

1. https://burn.dev/blog/when-profilers-mislead-you · Genna Wingert (Tracel / Burn) · Dec 3, 2025 · case study
   - **What to read:** The "Attempting to Optimize" section and what follows.
   - **Why:** A practitioner follows the Nsight Compute hints and the post is titled "Learning from a Pointless Optimization". It works as a counterweight to "just fix what ncu says".
   - **Time:** 15 min.
   - **Quote:** "I decided to take the easiest and most obvious path: Running Nsight Compute [ 7 ] and fixing whatever issues it points out."
   - **Quote:** "I noted the profiler hints were primarily about shared memory bank conflicts when storing the output, as well as uncoalesced global writes, and a runtime of 2.88ms."
   - **Caveat:** I read only the setup, not the result. Its section list ends with "What did we learn?" and "Addendum (it can go right)".

2. NVIDIA "Kernel Profiling Guide" (docs.nvidia.com/nsight-compute) and the ALCF "Introduction to Kernel Performance Analysis with NVIDIA Nsight Compute" training page surfaced in search. Neither was opened, so both are UNVERIFIED. No dedicated third-party ncu walkthrough was verified. See Gaps.

**Topic: Megakernel and compiler explainers (non-Hazy)**

1. https://arxiv.org/abs/2512.22219 (HTML at arxiv.org/html/2512.22219) · Cheng et al., 21 authors including Tianqi Chen and Zhihao Jia · Dec 2025 · paper, listed on the OSDI'26 program · Mirage Persistent Kernel (MPK)
   - **What to read:** The introduction (CUDA Graphs versus mega-kernels) and §6 Evaluation.
   - **Why:** It is the compiler and generality counterpart to the hand-written Hazy megakernel. It covers multi-GPU and B200, and runs on A100, H100 and B200.
   - **Time:** 40 min for the intro plus evaluation.
   - **Quote:** "We introduce Mirage Persistent Kernel (MPK), the first compiler and runtime system that automatically transforms multi-GPU model inference into a single high-performance mega-kernel."
   - **Quote:** "achieving up to 1.7$\times$ lower end-to-end inference latency"
   - **Quote:** "However, CUDA Graphs are largely static: any changes to control flow, tensor shapes, or data dependencies require re-instantiating or modifying the captured graph, limiting their flexibility for the dynamic workloads commonly seen in model inference."

**Topic: ThunderKittens**

1. https://hazyresearch.stanford.edu/blog/2024-05-12-tk · Spector, Singhal, Arora, Ré (Hazy Research) · May 12, 2024 (OLDER, but still the canonical TK introduction) · blog
   - **What to read:** The H100 SM anatomy and the "keep the tensor core fed" argument.
   - **Why:** It sets up the tile abstraction that the later Hazy megakernel post builds on.
   - **Time:** 26 min, the page's own estimate ("26 min read").
   - **Quote:** "Great, how do I make it go brr? Keep the tensor core fed. That's it."
   - **Quote:** "An H100 GPU has 989 TFLOPs of half-precision matrix multiply compute, and ~60 TFLOPs of "everything else"."

**Topic: DeepSeek kernel write-ups (DeepGEMM, FlashMLA, DeepEP): UNVERIFIED**

I found secondary candidates but fetched none of them:
- kevinyu.substack.com/p/deepseek-open-source-week-a-technical ("DeepSeek Open Source Week: A Technical Journey")
- pstaidecrypted.substack.com ("Opening the vault: DeepSeek releases most (but not all) of its secrets")
- gigazine and bitdeer summaries

I have no quotes, so I am not ranking them. The primary sources are the READMEs in the deepseek-ai GitHub repos, which I did not fetch.

---
### HALF 2: Counter-evidence, per claim

**(a) "Unified memory makes Macs good for LLM inference"**

- **Verdict:** The best evidence is the exo team's own numbers. It is not a hostile source: exo builds Mac clusters.
- https://blog.exolabs.net/nvidia-dgx-spark/ · exo labs · date not visible on the page (my memory puts it around Oct 2025, UNVERIFIED) · vendor engineering blog. Read it for the section "Prefill is compute-bound" and the Spark-plus-Mac split. About 7 KB, 10 min.
  - **Quote:** "The Mac Studio has 512GB of unified memory at 819 GB/s, but the GPU only has ~26 TFLOPs of FP16 performance. The DGX Spark has 4x the compute, the Mac Studio has 3x the memory bandwidth."
  - **Quote:** "You send a prompt. You wait. Nothing appears. This is the prefill phase , and it determines TTFT."
  - **Quote:** "Prefill is compute-bound".
  - **Why it counters the claim:** Unified memory gives capacity and bandwidth. The compute gap, which is what prefill needs, is about 4x against a small Spark. That compute gap is the "prefill slowness" argument.
  - **Caveat:** The Spark has 273 GB/s of bandwidth, far below an H100's. This does not measure an H100 or RTX against a Mac.
- https://arxiv.org/abs/2511.05502 · Rajesh et al. · submitted Oct 9, 2025 · paper. M2 Ultra with 192 GB. 5 min for the abstract.
  - **Quote:** "Although Apple Silicon inference frameworks still trail NVIDIA GPU-based systems such as vLLM in absolute performance, they are rapidly maturing into viable, production-grade solutions for private, on-device LLM inference."
  - **Caveat:** This is a mild counter. The abstract gives no numbers against NVIDIA.
- https://yage.ai/share/mlx-apple-silicon-en-20260331.html · yage.ai · Mar 31, 2026 · secondary aggregator that cites Reddit threads (a weak source). 10 min.
  - **Quote:** "This is because MLX spent 94% of its time on prefill."
  - **Quote:** "However, in chat scenarios (short prompts + short replies), where most of the time per request is spent on prefill, the overall speed of MLX may be slower than GGUF."
  - **Caveat:** The underlying data is a single M1 Max Reddit post, secondhand.
- **Spec-only bandwidth gap, low reliability:** dasroot.net/posts/2026/04/rtx-5090-vs-mac-studio-local-inference/ · Apr 2026.
  - **Quote:** "The RTX 5090, for instance, offers a memory bandwidth of 1,792 GB/s , significantly outperforming the M4 Max's 546 GB/s and the newer M5 Max's 614 GB/s ."
  - **Caveat:** The same page calls the RTX 5090 "based on Ada Lovelace architecture", which is wrong (it is Blackwell). I would not cite this source.
- **Pro-Mac side, for balance:** creativestrategies.com/mac-studio-m3-ultra-ai-workstation-review/ · Max Weinbach · Mar 11, 2025. The title is "The only real AI workstation today". Its argument is capacity: "so to load QwQ, which by itself is only 19GB...". I saw no prefill or thermal data in my pass. Consider it the claim side, not a counter.
- **Thermal throttling:** NOT_FOUND. I found only SEO-style content (aiindigo.com, omr.it.com) and never fetched it, so it is UNVERIFIED and unusable.
- **M3 Ultra versus a single H100 or RTX on the same model, measured:** NOT_FOUND. I have no head-to-head.
- The llama.cpp Apple Silicon table (github.com/ggml-org/llama.cpp/discussions/4167) is fetched and has M3 Ultra rows. I did not extract the prompt-processing versus generation columns into a comparison. It could be mined for the prefill ratio.

**(b) "Apple GPUs have no profiler / weak tooling"**

- **Verdict:** Weakened. A working profiling workflow exists and is documented for an MLX-based serving stack. It is clunky, not absent.
- https://docs.vllm.ai/projects/vllm-metal/en/latest/profiling/ · vllm-metal project docs · undated · docs. 10 min.
  - **Quote:** "Performance | Per-dispatch GPU time, memory bandwidth, occupancy. Click Profile… once to populate; replays the trace (a few minutes)."
  - **Quote:** "MLX labels its dispatches with op names, so search hits them directly."
  - **Quote (the warning, which supports the "weak tooling" side):** "It will lock up your machine on a default-config trace, where the engine has 20+ GiB of paged-attention KV state captured".
  - **Quote:** "Apple's "Command Line Tools for Xcode" alone are not enough — they don't include the GPU Frame Debugger that loads .gputrace bundles."
  - **Quote:** "xctrace ... doesn't surface MLX kernel names".
  - **Reading:** Per-kernel profiling is real, but needs about 10 GB of Xcode plus manual capture, and replay can hang. Tooling is weaker than ncu in ergonomics, not nonexistent.
- Apple docs on the Metal shader profiler, counter statistics and WWDC20/21 sessions surfaced in search. I did not open them, so I have no literal text and they are UNVERIFIED. One claim from the search summary, per-line shader cost, comes from an unverified summary.

**(c) "AI-written kernels beat libraries"**

- https://arxiv.org/abs/2607.16241 · Zhang et al. (Azalia Mirhoseini among the authors) · Jun 26, 2026 · paper. ALREADY IN THE COURSE as KernelBench-Verified, so I list it only as the headline counter-evidence.
  - **Quote:** "the best-performing model (GPT-5.5) achieves a 0.88x geometric mean speedup, significantly lower than the 1.43x speedup observed under the standard evaluation protocol. No model consistently outperforms PyTorch when evaluated against realistic baselines."
  - **Quote:** "28% of GPU kernels generated by the best model increase peak GPU memory usage."
- https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/ · Kyle Wiggers · Feb 21, 2025 · news. 5 min. Not previously in the course.
  - **Quote:** "Users on X quickly discovered that Sakana's system actually resulted in worse-than-average model training performance. According to one user , Sakana's AI resulted in a 3x slowdown — not a speedup."
  - **Quote:** "In a postmortem published Friday, Sakana admitted that the system has found a way to "cheat" (as Sakana described it) and blamed the system's tendency to "reward hack""
  - **Caveat:** This is early 2025 and the models were older. KernelBench-Verified shows the problem persisting at a smaller scale.
- Sakana's corrected number (the aggregate speedup falling from 3.13x to 1.49x) is UNVERIFIED. It appeared only in a search summary, not in text I fetched.
- Contrary signal: the cloudrift post above shows a compiler-generated kernel next to FA-2's hand-written source. That is a compiler, not an LLM, and I did not read the verdict, so I make no claim from it.

**(d) Megakernels: critiques beyond the PDL reconstruction**

- **Verdict:** I found no direct, hostile megakernel critique. These are bounding and portability constraints from the literature itself plus one adjacent measurement.
- https://arxiv.org/abs/2605.11581 · Ada-MK (Dong et al.) · submitted May 12, 2026 · paper
  - **Quote:** "kernel launch overhead alone can account for 14.6% of end-to-end inference time." This caps the upside: launch overhead is the main thing a megakernel removes.
  - **Quote:** "hand-tuned solutions are tightly coupled to specific architectures and lack portability, while auto-compiled approaches introduce runtime dynamic scheduling whose branch penalties are unacceptable in latency-critical settings."
  - **Caveat:** This is a pro-megakernel paper acknowledging limits, aimed at "resource-constrained GPUs such as NVIDIA Ada".
- https://arxiv.org/abs/2605.30571 · Josef Chen, "Memory-Bound but Not Bandwidth-Limited" · May 2026 (by arXiv ID) · paper. Adjacent evidence about batch-1 decode on H100.
  - **Quote:** "On H100 at ctx=2048, CUDA Graphs improves decode latency by 1.259x across N=10 fresh sessions"
  - **Quote:** "On L4, the same intervention gives only 1.028x."
  - **Quote:** "This isolates a launch-side overhead that becomes visible on fast GPUs but remains mostly hidden on slower, bandwidth-bound GPUs."
  - **Reading (my inference, not the paper's claim):** the launch-side gain is only large on fast GPUs. Per the paper, CUDA Graphs already recover a measured share of it. It does not test megakernels.
- https://arxiv.org/html/2512.22219 (MPK itself) · its design needs batch-size specialization. **Quote:** "MPK generates multiple t t Graphs specialized for representative batch sizes, using powers of two up to the maximum batch size." That is an operational cost the megakernel framing underplays.
- NOT_FOUND: an independent third-party critique, for example about register or shared-memory pressure, debuggability, or comparison against CUDA Graphs plus PDL.

**(e) MLX versus llama.cpp, measured, 2025–2026**

- **Verdict: CONFLICTING.** MLX wins decode. Prefill and long-context results go the other way in secondhand reports.
- arXiv 2511.05502 (M2 Ultra 192 GB, Qwen-2.5, prompts up to 100,000 tokens, Oct 2025). Primary measurement, though the abstract gives no numbers.
  - **Quote:** "MLX achieves the highest sustained generation throughput, while MLC-LLM delivers consistently lower TTFT for moderate prompt sizes"
  - **Quote:** "llama.cpp is highly efficient for lightweight single-stream use, Ollama emphasizes developer ergonomics but lags in throughput and TTFT"
- yage.ai (Mar 31, 2026), secondary:
  - **Quote:** "The actual advantage of MLX over raw llama.cpp is roughly 1.4-1.8x."
  - **Quote:** "Ollama's Go wrapper layer consumes about 50% of the performance. This means that the community's claim of MLX being 3x faster than Ollama is largely due to Ollama's own architectural overhead, rather than MLX completely outclassing llama.cpp."
  - **Quote:** "MLX actually lags behind llama.cpp in time-to-first-token (TTFT)."
  - The per-source numbers behind these (the 89.4 vs 43.5 tok/s on M4 Max, GitHub #14861) are cited from Reddit and a GitHub issue I did not open. UNVERIFIED at primary.
- UNVERIFIED search-summary-only claims, not usable without confirmation: that on an M5 Max llama.cpp was "4.3x faster average time-to-first-token (67ms vs 289ms)", and that on an M3 Ultra MLX generation was "about 50% slower" than llama.cpp with Flash Attention at 30K+ context. The source of the second was a paywalled Towards AI piece.
- The Andreas Kunar Medium post (Dec 2023) is INACCESSIBLE via curl (Cloudflare). Via a proxy fetch I saw only its note: "UPDATE April 2025: Please note that this 1 1/2+ years old article is now a bit outdated". Exclude it.

---
## Gaps
- **Half 1 not covered:** Lei Mao's CUDA posts, Lilian Weng or similar, PMPP companion material, Triton tutorials, "GPU memory hierarchy explained" posts, a standalone Nsight Compute walkthrough blog, and FlashAttention-from-scratch explainers (only gau-nernst and cloudrift found). I never searched for these specifically. NOT_FOUND by omission, not by a negative search result.
- DeepSeek kernel write-ups (DeepGEMM, FlashMLA, DeepEP): UNVERIFIED (see above).
- ThunderKittens beyond the 2024 post: nothing newer. The TK search was polluted by irrelevant results.
- Half 2 (a): thermal throttling and M3 Ultra versus H100/RTX same-model measurements are NOT_FOUND. The best available is exo's compute and bandwidth contrast.
- Half 2 (d): no independent megakernel critique found. What I have is bounding evidence only.
- Half 2 (e): MLX versus llama.cpp has few primary measurements with numbers I could read. Most numbers are secondhand.
- I did not extract the llama.cpp #4167 M3 Ultra numbers into a prefill versus decode ratio.

## Sources (independence noted)
- jax-ml.github.io/scaling-book/gpus (Google DeepMind and MatX authors): independent.
- chipsandcheese.com B200 article (independent publication; Verda supplied the hardware): independent.
- modular.com Blackwell Part 1 (vendor): not independent of the Mojo ecosystem.
- gau-nernst.github.io/fa-5090 (individual): independent.
- cloudrift.ai Part 2 (GPU-cloud vendor promoting its compiler): not independent.
- hazyresearch.stanford.edu TK post (2024): same lab as the excluded Hazy "No Bubbles" post.
- burn.dev (Tracel): independent.
- arxiv 2512.22219 (MPK) and arxiv 2605.11581 (Ada-MK): both pro-megakernel authors, separate groups.
- arxiv 2605.30571 (Josef Chen): independent, NVIDIA batch-1 decode.
- blog.exolabs.net: vendor of Mac clustering, which makes the admission against interest notable.
- arxiv 2511.05502 (Rajesh et al.): independent academic group.
- yage.ai: aggregator citing Reddit and GitHub; secondary.
- docs.vllm.ai/projects/vllm-metal: project docs, independent of Apple.
- arxiv 2607.16241 (KernelBench-Verified): already in the course.
- techcrunch.com Sakana article: journalist citing public X posts and Sakana's postmortem.
- dasroot.net: low reliability.
- creativestrategies.com: pro-Mac analyst firm.
