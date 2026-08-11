# Stage 2 Design Research Notes

## Primary references

[1] Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752. https://arxiv.org/abs/2312.00752

The paper motivates input-dependent state-space parameters for content-selective propagation and forgetting, and describes a hardware-aware parallel algorithm for recurrent-mode training. These are design references, not proof that the CCT implementation inherits the same results.

[2] Albert Gu, Karan Goel, and Christopher Ré, *Efficiently Modeling Long Sequences with Structured State Spaces*, arXiv:2111.00396. https://arxiv.org/abs/2111.00396

The paper introduces structured state-space parameterization and efficient computation for long-range sequence modeling. CCT Stage 2 therefore requires an independent reference recurrence and raw scaling measurements rather than relying on asymptotic labels alone.

[3] Aakash Sunil Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, ICLR 2026 Proceedings. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html

The official abstract reports three relevant directions: a more expressive discretized recurrence, optional complex-valued state, and MIMO structure intended to improve quality without increasing decode latency. CCT Stage 2 implements the real diagonal reference path first; complex state is explicitly deferred behind a configuration flag, and MIMO is represented by independent input/output projections.

## Engineering consequence

The CCT gate must measure loop/scan equivalence, gradient correctness, state stability, checkpoint recovery, algorithmic task performance, parameter count, decode memory, and raw scaling slopes. No claim of universal superiority over Transformers is made from these references.
