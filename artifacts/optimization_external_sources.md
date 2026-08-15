# External Architecture Research Sources

## Transformer

Source: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., arXiv:1706.03762.

The paper states that the Transformer uses attention without recurrence or convolution and emphasizes parallelizability and lower training time compared with the recurrent sequence models available at the time. This source is used only for the qualitative trade-off: Transformer training exposes parallel sequence computation, while autoregressive decoding still requires sequential token generation and a KV-cache/attention implementation for efficient inference. The CCT comparison must not copy the paper’s historical task metrics as evidence about this repository.

## Selective state-space comparison

Source: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), Gu and Dao, arXiv:2312.00752.

The authors describe selective SSMs whose parameters depend on the input, enabling content-dependent propagation and forgetting, and a hardware-aware parallel algorithm in recurrent mode. They report linear sequence-length scaling and fast inference relative to Transformers in their own implementation and model-scale experiments. These are paper-level claims, not CCT evidence. The relevant implication for CCT is that an efficient recurrent/SSM design requires more than an O(1) recurrence: it needs a fused, hardware-aware kernel, a content-selective state update, and a training path that preserves parallel efficiency. CCT currently has a simple scalar recurrence but not a fused scan, selective parameterization, or incremental state API.

## Recurrent baseline

Source: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Cho et al., arXiv:1406.1078.

The paper presents a recurrent encoder-decoder that compresses a sequence into a fixed-length representation and decodes another sequence recurrently. For this audit, the relevant comparison is that recurrent models have a compact online state and constant per-token state footprint, but sequential dependence limits parallelism and a fixed-size state can create an information bottleneck. CCT’s current state-memory advantage over dense attention is real at the implementation level, but its recurrent computation is still repeated from the beginning of the context for every generated token, so the present implementation does not yet realize the full online-state advantage.

## Structured state-space comparison

Source: [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), Gu, Goel, and Ré, arXiv:2111.00396.

The S4 paper frames long-range sequence modeling through state-space systems and emphasizes that naïve state-space computation can be too expensive, motivating structured parameterization and efficient computation. It reports strong long-sequence results and faster generation in its own experiments. The audit implication is important: asymptotic state-space notation alone does not establish practical efficiency. The implementation must expose stable parameterization, efficient scan/convolution kernels, and a training/inference path whose memory movement matches the claimed complexity. CCT’s current diagonal SSM baseline is much simpler than S4 and should not be described as an S4-equivalent system.

## Hardware-aware attention

Source: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., arXiv:2205.14135.

The authors argue that attention performance depends on reads and writes between memory levels, not just arithmetic complexity, and propose tiling to reduce high-bandwidth-memory traffic. The reported speedups are implementation- and hardware-specific. For CCT, the transferable principle is that replacing allocating scalar loops with contiguous buffers, tiling, fusion, and hardware-appropriate kernels is a prerequisite for credible throughput claims. A theoretically smaller recurrence can still lose if it performs excessive allocations and memory traffic; the CCT gprof profile confirms this is currently material.
