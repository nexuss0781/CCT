# Production NLP Roadmap Source Findings

## NIST Generative AI Profile

Source: [NIST AI RMF: Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence), NIST AI 600-1, published July 26, 2024 and updated April 8, 2026.

NIST describes the profile as a cross-sectoral companion to AI RMF 1.0 for generative AI. It is intended to help organizations incorporate trustworthiness considerations into the design, development, use, and evaluation of AI products, services, and systems. The roadmap therefore treats risk identification, measurement, management, and governance as lifecycle work rather than a final deployment checklist.

## LoRA

Source: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu et al., arXiv:2106.09685v2.

The paper frames NLP as large-scale pretraining followed by adaptation to particular tasks or domains. LoRA freezes pretrained weights and injects trainable low-rank matrices, reducing the number of trainable parameters and memory requirements for downstream adaptation. The roadmap uses this as a candidate fine-tuning path, but requires comparison against full fine-tuning and other parameter-efficient methods on held-out quality, safety, regression, and serving-cost metrics; the source does not justify assuming equal quality for every domain or architecture.

## Direct Preference Optimization

Source: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), Rafailov et al., arXiv:2305.18290v3.

The paper presents DPO as a preference-optimization approach that avoids a separately trained reward model and reinforcement-learning loop by using a classification-style loss derived from the RLHF objective. The roadmap treats DPO as one candidate post-training experiment after supervised fine-tuning, not as a universal replacement: preference-data quality, judge consistency, refusal behavior, regression tests, and domain-specific human review remain required.

## HELM

Source: [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/), Stanford CRFM.

HELM describes itself as a reproducible and transparent framework with many scenarios, metrics, and models, including multimodal and model-graded evaluations. The roadmap therefore requires a registry of capability, quality, safety, robustness, privacy, latency, cost, and calibration evaluations rather than optimizing only next-token loss or a single benchmark.

## Compute-optimal pretraining

Source: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al., arXiv:2203.15556.

The study investigates model size and token count under a fixed compute budget and reports that model size and training tokens should be scaled together in its experimental regime. The roadmap consequently requires pilot scaling curves, token-quality accounting, deduplication and contamination controls, and an explicit compute/data allocation decision rather than selecting model size first and filling data later. The result is treated as a planning hypothesis to validate for CCT-ASE, not a universal law for its different architecture.

## Production inference serving

Source: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al., SOSP 2023.

The paper identifies dynamic KV-cache memory as a major serving constraint and proposes paging and sharing strategies implemented in vLLM. The roadmap therefore separates the CCT model runtime from the production serving plane and requires explicit batching, cache, admission-control, latency, throughput, memory, and tail-latency measurements. CCT-ASE’s recurrent state may reduce some sequence-state costs, but it must be benchmarked against modern attention serving systems rather than assumed to win in production.
