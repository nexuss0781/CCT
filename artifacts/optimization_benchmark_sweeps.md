# CCT Optimization Benchmark Sweeps

All measurements use the native `cct_optimization_benchmark` executable, compact vocabulary (513 slots), tokenizer hash `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`, real repository text artifacts, and one CPU process. These are comparative engineering measurements, not language-quality claims.

## Context-length sweep

Contract: embedding=hidden=16, steps=10, batch=4, 32 train sequences, 16 evaluation sequences, 128 generated tokens, 5 timing repeats.

| Context | Model | Train tok/s | Eval tok/s | End-to-end decode tok/s | State bytes |
|---:|---|---:|---:|---:|---:|
| 32 | CCT | 29,847.95 | 83,664.44 | 5,828.09 | 256 |
| 32 | GRU | 30,745.35 | 81,840.71 | 5,492.94 | 128 |
| 32 | Diagonal SSM | 33,022.72 | 80,973.83 | 5,545.80 | 128 |
| 32 | Dense attention | 29,776.50 | 52,504.41 | 2,641.04 | 8,320 |
| 64 | CCT | 32,997.58 | 82,645.03 | 3,295.03 | 256 |
| 64 | GRU | 34,575.23 | 80,487.09 | 3,046.43 | 128 |
| 64 | Diagonal SSM | 37,906.12 | 81,365.51 | 3,210.56 | 128 |
| 64 | Dense attention | 30,307.24 | 38,384.85 | 981.42 | 16,512 |
| 128 | CCT | 35,953.54 | 83,680.16 | 2,308.72 | 256 |
| 128 | GRU | 37,608.46 | 82,383.44 | 2,151.48 | 128 |
| 128 | Diagonal SSM | 40,479.49 | 83,298.91 | 2,289.26 | 128 |
| 128 | Dense attention | 26,083.93 | 25,708.67 | 513.86 | 32,896 |
| 256 | CCT | 37,784.01 | 84,929.28 | 2,303.05 | 256 |
| 256 | GRU | 38,448.95 | 81,197.38 | 2,143.08 | 128 |
| 256 | Diagonal SSM | 42,333.00 | 83,415.02 | 2,274.89 | 128 |
| 256 | Dense attention | 19,249.80 | 14,905.36 | 506.52 | 65,664 |

The decode measurements show that the current recurrent implementations are not actually constant-time per generated token: CCT end-to-end decode falls from 5,828 tok/s at context 32 to about 2,309 tok/s at context 128 because each token recomputes a windowed forward pass. Dense attention loses more sharply with context and its declared state bytes grow linearly with context.

## Width sweep

Contract: context=128, steps=10, batch=2, 32 train sequences, 16 evaluation sequences, 128 generated tokens, 5 timing repeats.

| Width | Model | Parameters | Train tok/s | Eval tok/s | End-to-end decode tok/s | State bytes |
|---:|---|---:|---:|---:|---:|---:|
| 16 | CCT | 18,001 | 29,232.72 | 84,107.71 | 2,323.93 | 256 |
| 16 | GRU | 18,513 | 30,148.70 | 81,165.49 | 2,144.85 | 128 |
| 16 | Diagonal SSM | 17,201 | 31,915.78 | 85,104.16 | 2,266.60 | 128 |
| 16 | Dense attention | 17,697 | 17,909.66 | 24,508.63 | 491.54 | 32,896 |
| 32 | CCT | 37,537 | 17,693.99 | 55,486.63 | 988.68 | 512 |
| 32 | GRU | 39,585 | 11,699.76 | 36,610.69 | 701.63 | 256 |
| 32 | Diagonal SSM | 34,401 | 14,320.13 | 39,989.56 | 802.84 | 256 |
| 32 | Dense attention | 36,417 | 7,356.32 | 10,730.73 | 195.38 | 65,792 |
| 64 | CCT | 82,753 | 8,453.17 | 29,671.38 | 1,917.76 | 1,024 |
| 64 | GRU | 90,945 | 8,307.86 | 26,256.75 | 424.90 | 512 |
| 64 | Diagonal SSM | 70,337 | 11,123.80 | 35,183.21 | 541.23 | 512 |
| 64 | Dense attention | 78,465 | 3,357.23 | 3,684.68 | 65.10 | 131,584 |

The width sweep confirms that all current implementations are strongly scalar and memory/loop bound at these sizes. Exact per-token decode values are sensitive to short-run timer resolution and output length, but the order-of-magnitude gap for dense attention at width 64 is stable. The more important structural result is that CCT’s parameter count is dominated by the embedding and output projection as width grows, while GRU adds a quadratic hidden-to-hidden gate cost and dense attention adds quadratic context interactions.
