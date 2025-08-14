## AdaptiQ — Benchmark 1/5 (v0.1.0-bench1)

**TL;DR.** Paired benchmark of two identical CrewAI agents on GPT-4.1 (reasoning) + FLUX-1.1-pro (image): Baseline vs AdaptiQ runtime RL.  
At equal quality target (**Target: ΔCLIP ≥ 0; observed ΔCLIP = −0.17**), AdaptiQ delivers:
- **−15.0% latency**, **−13.6% cost**, **−10.6% tokens (NS, p=0.366)**
- **≈2.8× lower token variance** → more stable policy & context control

### What’s included
- PDF report (N=99 pairs; CLIP N=93)
- Per-image logs (`metrics.csv`) to recompute all plots
- One-command report generator (rebuild figures + PDF)
- Config: reward weights, prompts, runtime params

### Reproduce
```bash
uv run python flow_test.py --pairs 100 --save runs/bench1
bash scripts/make_report.sh runs/bench1 reports/bench1
# Windows: ./scripts/make_report.ps1 -InputDir runs/bench1 -OutDir reports/bench1
```

### Models (pinned)
- Reasoning: `openai:gpt-4.1-YYYY-MM-DD`  <!-- fill exact ID -->
- Image: `black-forest-labs/FLUX.1.1-pro`

### Cite this release
**DOI:** 10.5281/zenodo.16876744  
Badge: `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16876744.svg)](https://doi.org/10.5281/zenodo.16876744)`

License: Apache-2.0.
