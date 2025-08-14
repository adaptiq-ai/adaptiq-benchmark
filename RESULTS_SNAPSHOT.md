# Results Snapshot — Bench 1/5

| Metric | Baseline | AdaptiQ | Δ (%) | p-value |
|-------:|---------:|--------:|-----:|:-------:|
| Latency (s) | 13.94 | 11.85 | −15.0% | < 0.001 |
| Cost (USD/img) | 0.0099 | 0.0086 | −13.6% | < 0.001 |
| Tokens (count) | 8,347 | 7,459 | −10.6% | 0.366 (NS) |
| Quality (CLIP) | 91.18 | 91.01 | ΔCLIP = −0.17 | Target ≥ 0 |

**Stability (variance)** — StdDev(tokens): Baseline ≈ 1,278 vs AdaptiQ ≈ 457 → **≈2.8× lower variance** (mean difference NS).
