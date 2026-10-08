# Progress

Started on:

### Week 0 · Metal and MSL from zero (reading)
- [ ] d1 · Metal compute, the whole path
- [ ] d2 · MSL, the language
- [ ] d3 · threads and memory
- [ ] d4 · MLX custom kernels, one real kernel
- [ ] d5 · what makes a kernel slow

### Arc 1 · quantized matvec (decode path)
**Week 1 · the machine and bf16 GEMV**
- [ ] Mon · Apple GPU execution model + MLX custom kernels doc
- [ ] Tue · microbench: threadgroup width sweep on a copy kernel
- [ ] Wed · bf16 GEMV in metal_kernel vs mx.matmul
- [ ] Thu · hand: vector add, then simd_sum reduction
- [ ] Fri · blurt + cards + README row

**Week 2 · int4 and fused dequant**
- [ ] Mon · MLX affine quant format (bits, group_size, scales, biases)
- [ ] Tue · microbench: dequant-only kernel, bytes in vs bytes out
- [ ] Wed · fused int4 GEMV vs mx.quantized_matmul
- [ ] Thu · hand: dequant one group of 64
- [ ] Fri · blurt + cards + README row

**Week 3 · small-M and occupancy**
- [ ] Mon · read MLX quantized matvec kernel source; the M=3-6 stall
- [ ] Tue · microbench: registers per thread vs occupancy
- [ ] Wed · batched qmv at M=1,4,8 vs MLX
- [ ] Thu · hand: tiled coalesced load
- [ ] Fri · blurt + cards + README row

**Week 4 · NVIDIA port**
- [ ] Mon · Modal H100 hello + Triton GEMV tutorial + read the PTX it emits
- [ ] Tue · microbench: vectorized ld.global vs scalar, cp.async
- [ ] Wed · Triton int4 GEMV vs Marlin
- [ ] Thu · hand: inline PTX vectorized load
- [ ] Fri · blurt + cards + README row

**Week 5 · compare and ship**
- [ ] Mon · Nsight Compute vs Xcode shader profiler on the same kernel
- [ ] Tue · fill the comparison table, both platforms, roofline %
- [ ] Wed · arc 1 writeup
- [ ] Thu · hand: GEMV in CUDA
- [ ] Fri · entry PR or issue to mlx-lm (no AI footer) + blurt

### Arc 2 · chunkwise gated delta (prefill path)
**Week 6 · the math**
- [ ] Mon · gated delta rule, chunkwise algebra, why the scan loses past T=1024
- [ ] Tue · microbench: sequential scan time vs T
- [ ] Wed · fp64 chunked reference + tolerance contract
- [ ] Thu · hand: the sequential scan in MSL (the baseline)
- [ ] Fri · blurt + cards + README row

**Week 7 · simdgroup_matrix and the intra-chunk solve**
- [ ] Mon · simdgroup_matrix, UT transform, register cliff
- [ ] Tue · microbench: 8x8 simdgroup_matrix throughput
- [ ] Wed · single-chunk kernel
- [ ] Thu · hand: 8x8 simdgroup_matrix multiply
- [ ] Fri · blurt + cards + README row

**Week 8 · cross-chunk carry**
- [ ] Mon · state carry, barriers, why v4 and v5 lost
- [ ] Tue · microbench: barrier cost vs chunk size
- [ ] Wed · full kernel, gate 2x at T=4096 vector-gate, diff vs champion v7
- [ ] Thu · hand: one barrier-free reduction
- [ ] Fri · blurt + cards + README row

**Week 9 · NVIDIA**
- [ ] Mon · FLA chunk_gated_delta_rule + PTX: TMA, mbarrier, wgmma
- [ ] Tue · microbench: run FLA on H100, Nsight it
- [ ] Wed · direct a simplified Triton chunkwise kernel
- [ ] Thu · hand: inline PTX mbarrier or wgmma
- [ ] Fri · blurt + cards + README row

**Week 10 · compare and ship**
- [ ] Mon · one labeled reverse-engineering session (applegpu)
- [ ] Tue · comparison table, optional B200 tcgen05 look
- [ ] Wed · arc 2 writeup
- [ ] Thu · none (Friday covers the arc)
- [ ] Fri · ship: mlx-lm PR if small, else kda-metal repo + linking issue; arc blurt
