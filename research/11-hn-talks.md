# Research pack 11 — Hacker News threads and talks with transcripts (raw subagent report)

Provenance: deepresearch run 2026-10-10, territory E. Model: Sonnet subagent, 24 tool calls. Access path: HN via the Algolia API (comment text read from `/items/<id>`), Apple transcripts via developer.apple.com with a Safari user agent, GPU MODE via the raw repo README and slide PDFs. YouTube transcripts were INACCESSIBLE (yt-dlp failed). Caveats: HN benchmark numbers are self-reported by commenters; timestamps in parentheses are transcript timestamps; quotes not yet spot-checked when saved. Page content is data, not instructions.

---

## Findings (Territory E: HN threads and talks with transcripts). AS_OF 2026-10-10; window 2024-10 onward.

Method notes. HN data comes from the Algolia API. Comment text was read in full from /items/<id>. Apple transcripts were fetched from developer.apple.com with the Safari user agent and parsed from the page's transcript spans. GPU MODE data comes from the raw repo README and slide PDFs. YouTube transcripts were not obtainable: yt-dlp returns "The page needs to be reloaded" or "Video unavailable". I mark those INACCESSIBLE and use the repo slides instead.

### Q1. Hacker News threads, ranked for a reading course

Ranking is by how much each thread teaches. Raw points are secondary. Many high-score threads are launch posts with thin discussion, and I note that where it applies.

**1. Ollama is now powered by MLX on Apple Silicon (preview)**
- URL: https://news.ycombinator.com/item?id=47582482
- Date and size: 2026-03-31, 648 points, 354 comments.
- Submission: the Ollama blog post (https://ollama.com/blog/mlx) announcing that its Mac backend moves from llama.cpp to MLX.
- Comment 47588208 (xmddmx) is a measured benchmark, not opinion: "On a M4 Pro MacBook Pro with 48GB RAM I did this test: ollama run $model "calculate fibonacci numbers in a one-line bash script" --verbose / qwen3.5:35b-a3b-q4_K_M 6.6 30.0 / qwen3.5:35b-a3b-nvfp4 13.2 66.5 / qwen3.5:35b-a3b-int4 59.4 84.4". The columns are PromptEvalRate and EvalRate.
- Comment 47583733 (Yukonv): "MLX powered inference makes a big difference, especially on M5 as their graphs point out. What really has been a game changer for my workflow is using https://omlx.ai/ that has SSD KV cold caching. No longer have to worry about a session falling out of memory and needing to prefill again. Combine that with the M5 Max prefill speed means more time is spend on generation than waiting for 50k+ content window to process."

**2. 1.5 TB of VRAM on Mac Studio, RDMA over Thunderbolt 5**
- URL: https://news.ycombinator.com/item?id=46319657
- Date and size: 2025-12-18, 617 points, 228 comments.
- Submission: the Jeff Geerling blog post on a four-Mac Studio cluster.
- Comment 46320088 (behnamoh) is a concrete wish list for M5 Max/Ultra: "Neural accelerators to get prompt prefill time down. I don't expect RTX 6000 Pro speeds, but something like 3090/4090 would be nice. … +1TB/s bandwidth. For the past 3 generations, the speed has been 800GB/s…"
- Comment 46320247 (rahimnathwani) is a scaling puzzle: "DeepSeek V3.1 is a 671B parameter model whose weights take up 700GB RAM … I would have expected that going from one node (which can't hold the weights in RAM) to two nodes would have increased inference speed by more than the measured 32% (21.1t/s -> 27.8t/s)."

**3. macOS 26.2 enables fast AI clusters with RDMA over Thunderbolt**
- URL: https://news.ycombinator.com/item?id=46248644
- Date and size: 2025-12-12, 540 points, 291 comments.
- Submission: the Apple developer release note.
- Comment 46251858 (0manrho) on link limits: "Thunderbolt5's stated "80Gbps" bandwidth comes with some caveats. … If Apple's silicon follows spec, then that means you're most likely limited to PCIe4x4 ~=64Gbps bandwidth per TB port".
- Comment 46250023 (reilly3000), on a $50k budget: "Apple M3 Ultra Cluster ($50k): Maximizes capacity (3TB). … albeit at low speeds (~15 t/s). NVIDIA RTX 6000 Workstation ($50k): Maximizes throughput (>80 t/s) … hard-capped at 384GB VRAM". The figures are his own and UNVERIFIED.
- This pairs with the WWDC26 session 233 below, which confirms the RDMA/JACCL design.

**4. DeepSeek 4 Flash local inference engine for Metal (antirez/ds4)**
- URL: https://news.ycombinator.com/item?id=48050751
- Date and size: 2026-05-07, 499 points, 159 comments.
- Submission: a GitHub repo, a hand-written Metal inference engine for one model. It is a good case study in the "single-model engine" approach.
- Comment 48052517 (antirez, the author): "A random, funny, interesting and telling data point: my MacBook M3 Max while DS4 is generating tokens at full speed peaks 50W of energy usage…"
- Comment 48052940 (kgeist) proposes the design idea: "what if we started building ultra-optimized inference engines tailored to an exact GPU+model combination? … If you remove enough abstractions and code directly to the exact hardware/model, you can probably optimize things qu[ite a lot]". The text is cut at about 800 characters in my read.
- Comments 48056801 and 48060640 (lhl) give a kernel-tuning method from AMD hardware, not Apple: "Grid_Size / Workgroup_Size >= CU count … < 0.3 = massively undersubscribed. Fix grid FIRST. Micro-optimization will NOT help." It carries over as a method but is not Apple-specific.

**5. Flash-MoE: Running a 397B Parameter Model on a Laptop**
- URL: https://news.ycombinator.com/item?id=47476422
- Date and size: 2026-03-22, 398 points, 119 comments.
- Submission: the GitHub repo danveloper/flash-moe.
- Comment 47479299 (tarruda) pastes a llama-bench log with a load-bearing line: "ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices". This shows the Metal tensor path in llama.cpp is gated to M5 and A19 hardware.
- Comment 47479961 (tarruda): "I did try 4-bit MLX with other Qwen 3.5 models and yes it is significantly faster. I still prefer llama.cpp due to it being a one in all package".

**6. Show HN: Open-source engine running Gemma 4 26B in 2 GB RAM on any M-series Mac (turbo-fieldfare)**
- URL: https://news.ycombinator.com/item?id=49098510
- Date and size: 2026-07-29, 919 points, 342 comments.
- Submission: the GitHub repo drumih/turbo-fieldfare, which streams MoE experts from SSD.
- Comment 49101376 (Joeri) lists bandwidths. The rumored rows are marked with * in his post: "M1: 68 GB/s / M2 pro: 200 GB/s / M2 max: 400 GB/s / M2 ultra: 800 GB/s / M5: 153 GB/s / M5 pro: 307 GB/s / M5 max: 460 GB/s / M6: 200 GB/s* / M7: 240 GB/s*".
- Comment 49105161 (pwython): "Got 48 tok/s decode at 1.9 GB RSS (2.4 GB peak) … The ~2.0 GB/s SSD number quoted for M4 is the base chip. This M4 Max does ~7 GB/s. Page cache seems to be why it beats the M5 Pro."

**7. Apple M5 chip (announcement thread)**
- URL: https://news.ycombinator.com/item?id=45591799
- Date and size: 2025-10-15, 1263 points, 1395 comments. It is mostly generic. Two comments are useful.
- Comment 45592586 (modeless): "Nvidia DGX Spark has 273 GB/s … All the neural accelerators in the world won't make it competitive in speed with discrete GPUs that all have way more bandwidth."
- Comment 45606539 (hedgehog) has a per-generation table with a "Memory Bandwidth" column: M1 68.25, M2 100, M3 100, M4 120 GB/s. Comment 45598632 (jacobolus) reformats it and gives M5 as 153 GB/s.

**8. Retrospectively Reverse-Engineering Apple's Neural Engine**
- URL: https://news.ycombinator.com/item?id=49670032
- Date and size: 2026-09-12, 241 points, 36 comments.
- Submission: https://eiln.github.io/posts/ane.html. A companion post, "Getting 50 GB/S Back from the Apple Neural Engine" (https://news.ycombinator.com/item?id=49636479, 218 points, 36 comments), covers DMA.
- Comment 49670764 (woadwarrior01): "M4 introduced a fast path for INT8 weights and activations (w8a8). M5 Ultra, M6 and A20 have two ANEs. … NAX cores are matmult accelerators, closer to tensor cores in NVIDIA GPUs." This separates the ANE from the GPU-side Neural Accelerators (NAX). It is a single commenter's claim and UNVERIFIED.
- Related earlier thread: "Inside the M4 Apple Neural Engine, Part 1" (https://news.ycombinator.com/item?id=47208573, 2026-03-01, 376 points, 117 comments). Its best comment is a pointer, not a fact: LatencyKills, "I worked on the Xcode team for years and know the lengths Apple goes to make this stuff difficult to figure out."

**9. Dissecting the Apple M1 GPU, the end (Alyssa Rosenzweig)**
- URL: https://news.ycombinator.com/item?id=45034537
- Date and size: 2025-08-27, 845 points, 224 comments.
- Submission: https://rosenzweig.io/blog/asahi-gpu-part-n.html. Its content belongs to agent A.
- Comment 45035023 (Tiberium) quotes the Asahi report: "we were carrying over 1200 patches downstream … that number is now below 1000 for the first time in many years".
- Related threads:
  - "An Update on Apple M1/M2 GPU Drivers" (LWN), https://news.ycombinator.com/item?id=42011239, 2024-10-31, 625 points, 266 comments. The comments I sampled were about graphics, not compute.
  - "AAA Gaming on Asahi Linux", https://news.ycombinator.com/item?id=41799068, 944 points, 365 comments.
  - "Asahi Linux on M3", https://news.ycombinator.com/item?id=49586698, 2026-09-06, 563 points, 378 comments. I did not read its comments.

**10. Apple's MLX adding CUDA support**
- URL: https://news.ycombinator.com/item?id=44565668
- Date and size: 2025-07-14, 548 points, 203 comments.
- Submission: MLX GitHub PR #1983.
- Comment 44566002 (nxobject): "this set of patches also adds support for "Linux [platforms] with CUDA 12 and SM 7.0 (Volta) and up"".
- Comment 44567263 (lukev) asks whether MLX programs can run on CUDA hardware, but not CUDA programs on Mac. The 72-reply subtree is the part worth reading.

**11. Tiny-LLM: a course of serving LLM on Apple Silicon**
- URL: https://news.ycombinator.com/item?id=43820022
- Date and size: 2025-04-28, 297 points, 35 comments.
- Submission: https://github.com/skyzh/tiny-llm. It is directly relevant to a reading course.
- Comment 43823457 (simonw): "The two best people to follow for MLX stuff are Apple's Awni Hannun … and community member Prince Canuma who's responsible for both mlx-vlm and mlx-audio".
- Comment 43822815 (fsiefken): "For 4 bit deepseek-r1-distill-llama-70b on a Macbook Pro M4 Max with the MLX version on LM Studio: 10.2 tok/sec on power and 4.2 tok/sec on battery / low power. For 4 bit gemma-3-27b-it-qat … 26.37 tok/sec on power and … 9.7".

**12. Apple Silicon costs more than OpenRouter**
- URL: https://news.ycombinator.com/item?id=48168198
- Date and size: 2026-05-17, 355 points, 302 comments.
- Submission: the williamangel.net blog post on energy and cost.
- Comment 48181931 (avidphantasm): "Gemma 4 31B Instruct Q6K … prompt eval time = 32545.36 ms / 5625 tokens (172.84 tokens per second) eval time … (15.33 tokens per second)". The "LM Studio MLX (Apple M5)" runtime appears in his config list, so I take this to be an M5-class machine.
- Comment 48168888 (jmyeet): the market-segmentation argument. 5090 is 32GB at ~$3000, RTX 6000 Pro is 96GB at ~$10,000. His prices are UNVERIFIED.

**13. Kernel-engineering threads (not Apple-specific)**
- **Speeding up PyTorch inference on Apple devices with AI-generated Metal kernels** (Gimlet Labs): https://news.ycombinator.com/item?id=45118111, 2025-09-03, 187 points, 30 comments.
  - Useful as a pointer, because it is the only HN thread on Metal kernel generation.
  - Comment 45119741 (turbo_wombat) is a fair criticism of the baseline: "They are comparing unoptimized PyTorch inference, something you would never deploy on a device, to a model with custom kernels."
  - Comment 45118897 (pbronez): "they're using proprietary models to write kernels for a set of ~250 open Pytorch modules."
- **Surprisingly fast AI-generated kernels we didn't mean to publish yet** (Stanford CRFM): https://news.ycombinator.com/item?id=44139454, 2025-05-30, 417 points, 217 comments.
  - The numerical-error critique is a good lesson on trusting generated kernels.
  - Comment 44143738 (threeducks): "I ran their matrix multiplication code … and got a mean squared error of approximately 0.056 … I think this error is large enough that referring to it as FP32 is misleading."
  - Comment 44141134 (david-gpu) suspects a narrow-case comparison against cuDNN.
- **Compiling LLMs into a MegaKernel** (MPK): https://news.ycombinator.com/item?id=44321672, 2025-06-19, 314 points, 76 comments.
  - Comment 44322700 (refulgentis): "With CUDA Graphs … if you have Matmul -> AllReduce, the AllReduce kernel cannot start until the entire Matmul kernel has finished. … With a megakernel, they break these ops into fine-grained "tasks" scheduled across SMs."
  - A companion thread: "Look Ma, No Bubbles: Designing a Low-Latency Megakernel for Llama-1B", https://news.ycombinator.com/item?id=44111673, 236 points, 31 comments.
- **Fp8 runs ~100 tflops faster when the kernel name has "cutlass" in it**: https://news.ycombinator.com/item?id=45458948, 2025-10-03, 338 points, 166 comments.
  - Comment 45463803 (frogblast): "GPU compilers are hard: Once you've done the basics, trying to do further transforms in a mature compiler will almost always produced mixed results."

**14. Other threads I considered and dropped or demoted**
- Ask HN "Resources for general purpose GPU development on Apple's M* chips?" (https://news.ycombinator.com/item?id=42509730, 2024-12-25, 149 points, 82 comments). It is a good reading-course seed, though the top comments are thin.
  - barkingcat: "There is no general purpose GPU development on Apple M series. There is Metal development."
  - feznyng: "llama.cpp … uses metal for accelerated inference on Apple silicon."
  - morphle links applegpu and AsahiLinux/gpu reverse-engineering repos, so it is a pointer thread.
- Kimi K2 on 2 M3 Ultras (https://news.ycombinator.com/item?id=46262734, 234 points, 121 comments). Its sampled comments were off-topic.
- Apple Silicon GPU Support in Mojo (https://news.ycombinator.com/item?id=45326388, 2025-09-21, 156 points, 69 comments). Archit3ch mentions Metal.jl and KernelAbstractions.jl.
- Launch HN RunAnywhere (https://news.ycombinator.com/item?id=47326101, 240 points, 153 comments). I did not read its comments, so UNVERIFIED as a source of facts.
- Queries that returned NOT_FOUND above a points threshold of about 30–80:
  - "M5 GPU neural accelerator" as a standalone submission. The announcement thread above is the closest.
  - "Mac Studio inference". Only the small Ask HN https://news.ycombinator.com/item?id=46907001 (57 points, 35 comments) surfaced.
  - "llama.cpp Metal" as a dedicated thread.
  - "unified memory inference" as a dedicated thread.
  - "Metal 4" as a dedicated thread.
  - A WWDC25 thread about Metal compute.

### Q2. Apple talks with transcripts (2024–2026)

Verified from transcript pages. The number in parentheses is the transcript timestamp.

**WWDC26 (June 2026), the most current material**

1. **Optimize custom machine learning operations with Metal tensors**
   - https://developer.apple.com/videos/play/wwdc2026/330/
   - Speaker: Shiyao, GPU Software Engineer. Watch 0:21–2:30 and 5:17–9:31 (quantized matmul with TensorOps).
   - Chapters: Apple's ML software stack 0:21; Managing quantized data 2:25; Multi-plane tensors 4:23; Quantized matrix multiplication 5:17; Building advanced ops 9:31; Integrating custom ops into Core AI 13:35.
   - (1:33) "it takes full advantage of the neural accelerator in the M5 chip family. The neural accelerator is a new hardware block in M5, located directly in each shader core. It sits alongside the other GPU pipelines and is designed to accelerate dense compute-bound work such as the prefill stage of an LLM."
   - (1:02) "You may also need to write Metal kernels if you're contributing to an ML framework such as MLX or llama.cpp."
   - (6:10) "We can use quantization to further reduce memory traffic".
   - It also says it builds on the basics from "the M5 machine learning talk". I could not identify that session in the WWDC26 list, so it is UNVERIFIED.

2. **Run local agentic AI on the Mac using MLX**
   - https://developer.apple.com/videos/play/wwdc2026/232/
   - Speaker: Angelos, MLX team. Watch 5:39–6:53 ("Making agents fast").
   - (6:12) "The M5 chip introduces dedicated Neural Accelerators, and MLX can target them for exactly this kind of work. Specifically, Neural Accelerators make matrix multiplication four times faster on M5 compared to M4."
   - (6:41) "Taking advantage of Neural Accelerators requires no special arguments or code changes on your part, MLX selects the best kernel for the available hardware and it just works."
   - (5:55) "Agentic sessions usually comprise hundreds of thousands of tokens and most of those are not generated."

3. **Explore distributed inference and training with MLX**
   - https://developer.apple.com/videos/play/wwdc2026/233/
   - Speaker: Tatiana, MLX research scientist. Watch 2:09–4:32 (RDMA and JACCL) and 10:33–15:53 (model parallelism).
   - (2:32) "Starting in macOS 26.2, Remote Direct Memory Access protocol, shortly RDMA, is supported over Thunderbolt 5. RDMA moves data directly from one machine's memory to another's, avoiding most CPU and operating system overhead."
   - (3:17) "JACCL is an open-source collective communication library built by Apple."

4. **Explore numerical computing in Swift with MLX**
   - https://developer.apple.com/videos/play/wwdc2026/328/
   - Speaker: David Koski. I read only the chapter list (Mandelbrot 4:28, heat distribution 6:34), not the transcript text, so there is no verbatim line.

5. **Build real-time neural rendering pipelines with Metal**
   - https://developer.apple.com/videos/play/wwdc2026/359/
   - Speaker: Yulia. Watch 13:40–20:55 (inline neural networks with tensorOps).
   - (1:26) "the TensorOps API provides the building blocks to design and run custom models directly in your shaders, enabling you to fully leverage the neural accelerator".

6. **Find and fix performance issues in your Metal games**
   - https://developer.apple.com/videos/play/wwdc2026/388/
   - GPU profiling: metrics 1:51, trace collection 3:32, analyzing traces 6:38.
   - (2:10) "GPU time, and Frame Interval tell you the pacing of your frames and how utilized the GPU is."
   - This is games-oriented, so it is low priority for kernel work.

7. Titles I listed but did not read:
   - Meet Core AI (324)
   - Dive into Core AI model authoring and optimization (325)
   - Integrate on-device AI models into your app using Core AI (326)

**WWDC25**

1. **Get started with MLX for Apple silicon**
   - https://developer.apple.com/videos/play/wwdc2025/315/
   - Speaker: Awni Hannun. Watch 10:15–17:30 (accelerating MLX).
   - (11:15) "Each of these nodes corresponds to a separate GPU Kernel launch under the hood. Compiling the graph uses all of these separate Kernels into single fused Kernel. This can save memory bandwidth and graph execution overhead".
   - (13:18) "These Kernels are written in Metal … You build the Kernel by passing in a source string of Metal code".
   - (13:46) "It creates a node in the computation graph and is lazily evaluated."
   - (13:55) "Large models need lots of memory and lots of memory bandwidth to be fast."

2. **Explore large language models on Apple silicon with MLX**
   - https://developer.apple.com/videos/play/wwdc2025/298/
   - Speaker: Angelos. Chapters: MLX LM introduction 3:07, quantization 8:42, fine-tuning 11:39. I did not extract a verbatim line.

3. **Combine Metal 4 machine learning and graphics**
   - https://developer.apple.com/videos/play/wwdc2025/262/
   - Speaker: Preston Provins. Watch 2:52–6:21 (tensors).
   - (2:55) "This year Metal 4 introduces MTLTensor, a new resource that lets you apply machine learning to data with unprecedented ease."

4. **Discover Metal 4**
   - https://developer.apple.com/videos/play/wwdc2025/205/
   - Watch 13:52–17:26.
   - (14:30) "Metal tensor are multi-dimensional data containers."

**WWDC24**
- Train your machine learning and AI models on Apple GPUs: https://developer.apple.com/videos/play/wwdc2024/10160/ (PyTorch improvements 4:16, ExecuTorch 11:26, JAX 13:19).
- Bring your machine learning and AI models to Apple silicon: https://developer.apple.com/videos/play/wwdc2024/10159/ (model compression 2:47, stateful model 13:35, transformer optimization 17:08).
- I confirmed these exist and read their chapters only. No verbatim lines were extracted.

**WWDC26 session list.** I scanned the WWDC26 page's 138 session titles for ML, Metal, and performance topics. The relevant sessions are the six listed above plus the three Core AI titles. The "Machine Learning & AI Group Lab" (8016) and Power and Performance Group Lab (8003) exist. They are labs, and I did not fetch them.

### Q3. GPU MODE and other kernel-engineering talks

The GPU MODE README (https://raw.githubusercontent.com/gpu-mode/lectures/main/README.md) has no dates. I got the dates for the three lectures below from yt-dlp metadata.

1. **Lecture 31: Beginners Guide to Metal Kernels**
   - Speaker: Nikita Shulga ("malfet", PyTorch MPS maintainer). The README links his name to the wrong GitHub profile, and the slides say "malfet".
   - Date: 2024-10-14 (just inside the window; YouTube length 1:32:08).
   - YouTube: https://www.youtube.com/watch?v=cGtiaJjLkAI. Transcript INACCESSIBLE.
   - Slides: https://github.com/gpu-mode/lectures/tree/main/lecture_031. The folder has the PDF and gemm_perf_studies.mm.
   - This is the only GPU MODE lecture on Apple or Metal. It is the most relevant item in the whole territory.
   - What to watch: the GEMV case study, "classic SIMD" (float4 dot), then "mat4 SIMD", then "optimize memory access pattern". The slides name the MLX gemv.metal kernel as the reference.
   - Slide lines, verbatim:
     - "Laptop those slides are stream from is a pretty powerful machine • 110 Gflop of CPU perf vs 7 Tflop of GPU perf".
     - "Threads further organized into simd groups (something like CUDA warps)".
     - "How to profile Metal from command line: Short answer: you can not".
     - "On Mac there are no better tool that Xcode :) • Use MTL_CAPTURE_ENABLED environment variable to enable capture".
     - "Metal Performance Shaders Graph (MPSGraph) is a graph-like compiler that fuses multiple shaders together".
   - The WWDC26 session 330 now adds the M5 tensor path that this 2024 lecture predates.

2. **Lecture 79: Mirage (MPK): Compiling LLMs into Mega Kernels**
   - Speakers: Mengdi Wu, Xinhao Cheng. Date: 2025-09-14 (1:12:52).
   - YouTube: https://www.youtube.com/watch?v=sXDdRCy137c. Transcript INACCESSIBLE.
   - Slides: lecture_079/Mirage-2025-09-13.pdf. My pdftotext grep returned no keyword hits (likely image slides), so there is no verbatim slide line.
   - Pairs with the HN megakernel threads above.
   - CUDA-focused, not Apple.

3. **Lecture 29: Triton Internals**
   - Speaker: Kapil Sharma. Date: 2024-09-28 (1:04:49). That is before the window; older, but it is still listed in the README.
   - YouTube: https://www.youtube.com/watch?v=njgow_zaJMw. Transcript INACCESSIBLE.
   - Materials: lecture_029/presentation.md and presentation.pdf. I did not read them, so there is no verbatim line.

4. **Other README lectures relevant to kernels** (titles and speakers from the README only; no dates, transcripts or slides read, so UNVERIFIED as content):
   - Lecture 1: Profiling and Integrating CUDA kernels in PyTorch.
   - Lecture 12: Flash Attention.
   - Lecture 14: Practitioner's Guide to Triton.
   - Lecture 16: On Hands profiling.
   - Lecture 18: Fused Kernels.
   - Lecture 36: CUTLASS and Flash Attention 3.
   - Lecture 40: Flash Infer (Zihao Ye).
   - Lecture 75: GPU Programming Fundamentals + ThunderKittens (William Brandon, Simran Arora).
   - Lecture 104: Gluon: Tile-Based GPU Programming (Peter Bell, Mario Lezcano, Keren Zhou).
   - Lecture 106: HF kernels.
   - Lecture 114: PyCuTe (Cris Cecka).
   - The README ends at Lecture 114, and I did not check whether it is current. No AI-generated-kernels lecture is in the README titles (NOT_FOUND). The Gimlet and CRFM HN threads are the closest.

### Q4. Podcasts and interviews with MLX or Apple GPU engineers
NOT_FOUND. HN Algolia queries for "Awni Hannun", "MLX podcast", and "Apple MLX interview" returned no podcast submissions. A WebSearch returned nothing usable. yt-dlp ytsearch returned one irrelevant ThursdAI episode (mPQiK5yRYBw, 2024-era). The only Awni material with a transcript I could actually read is WWDC25 session 315 (above). Podcast and YouTube interviews, if any exist, are INACCESSIBLE through the tools available here.

## Gaps
- YouTube transcripts for GPU MODE L29, L31 and L79 are INACCESSIBLE (yt-dlp fails). Only slides and repo files were read, and none of the L29 files were read at all.
- I did not read the WWDC26 328 and 388 transcripts beyond chapters and keyword lines. The WWDC24 transcripts, WWDC25 298, and Core AI sessions 324–326 were chapter-listed only.
- "M5 machine learning talk" cited in WWDC26 session 330 was not found in the session list. It may be a Tech Talk or a WWDC25 session. UNVERIFIED.
- Apple Tech Talks (developer.apple.com/videos/tech-talks) were not enumerated.
- I did not check for Metal Performance Primitives or TensorOps sessions elsewhere.
- Comment text was read for only about 20 threads. Several high-score launch threads (RunAnywhere 47326101, Magnitude 49911995, Cactus) were not read.
- Hand-offs by owner:
  - Agent A (blogs): https://rosenzweig.io/blog/asahi-gpu-part-n.html, https://eiln.github.io/posts/ane.html, https://eiln.github.io/posts/ane-dma.html, https://maderix.substack.com/p/inside-the-m4-apple-neural-engine, https://gimletlabs.ai/blog/ai-generated-metal-kernels, https://crfm.stanford.edu/2025/05/28/fast-kernels.html, https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-stu (URL cut off in the Algolia output), https://www.williamangel.net/blog/2026/05/17/offline-llm-energy-use.ht (URL cut off in the Algolia output), https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles, https://zhihaojia.medium.com/compiling-llms-into-a-megakerne (cut off in the Algolia output).
  - Agent B (MLX docs and GitHub): the MLX CUDA PR #1983, https://github.com/skyzh/tiny-llm, https://github.com/antirez/ds4, https://github.com/danveloper/flash-moe, https://github.com/drumih/turbo-fieldfare, https://github.com/Anemll/Anemll, https://github.com/maderix/ANE, plus the slide-referenced MLX kernel mlx/backend/metal/kernels/gemv.metal.
  - Agent C (X): the Kimi K2 on 2 M3 Ultras thread points to a tweet by @awnihannun (status/19437235999714431, truncated in HN output); simonw's note that Awni Hannun and Prince Canuma are the people to follow for MLX.
  - Agent F (NVIDIA blog explainers): the HN threads on megakernels, FlashAttention 4 (Modal), and the Triton cutlass-name PR belong with the CUDA material.

## Sources and independence
- **Independent primary sources:**
  - Apple WWDC transcript pages, under developer.apple.com/videos/play/wwdc2024, wwdc2025 and wwdc2026.
  - The GPU MODE repo: README, lecture_031 slide PDF, and the lecture_079 PDF.
  - YouTube metadata, which gave dates and durations for L29, L31, L79 only.
- **HN items:** the Algolia API. HN commenters often cite blogs and tweets that other agents cover, so HN facts are secondhand relative to those sources. Benchmark numbers in comments are self-reported, and I did not reproduce any of them.
- **HN thread IDs used:** 47582482, 46319657, 46248644, 48050751, 47476422, 49098510, 45591799, 49670032, 49636479, 47208573, 45034537, 42011239, 41799068, 49586698, 44565668, 43820022, 48168198, 45118111, 44139454, 44321672, 44111673, 45458948, 42509730, 46262734, 45326388, 47326101, 46907001.
- **Apple URLs used:**
  - wwdc2026: 330, 232, 233, 328, 359, 388, plus 324–326 by title.
  - wwdc2025: 315, 298, 262, 205.
  - wwdc2024: 10160, 10159.
- **Non-independence flags:**
  - WWDC26 232 and 233 and WWDC25 315 are all from the same Apple MLX team, so they are one viewpoint.
  - The "4x faster matmul" claim in session 232 is Apple's own and is not independently verified. The 46320088 comment on RDMA and the 47583733 comment on M5 prefill are user reports, not Apple data.
