### From-Scratch Stratified Attention Ablation Summary

- Baseline: `original_gpt2` (4 layers, 256 hidden)
- Ablations: Stratified attention with varying S, top_k, α, aux losses
- Dataset: 2,000 samples (1,800 train / 200 val), 50 epochs

#### Key Outcomes
- Best eval loss: 0.6652 (S=3, top_k=1, α=0.02, aux=5e-5/5e-5)
- Baseline eval loss: 0.6956 → absolute −0.0304
- Best perplexity: 1.9449 vs baseline 2.0049
- Generation: 5/5 success across configs, diversity ≈ 0.48, avg length ≈ 42 words

#### Detailed Results

| Config | S | top_k | α | aux (entropy/balance) | Eval Loss | PPL | Gen Success | Diversity |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Baseline (`original_gpt2`) | - | - | - | - | 0.6956 | 2.0049 | 5/5 | 0.474 |
| strat_s3_top1_a001_e1e4_b1e4 | 3 | 1 | 0.01 | 1e-4 / 1e-4 | 0.6654 | 1.9453 | 5/5 | 0.483 |
| strat_s4_top1_a001_e1e4_b1e4 | 4 | 1 | 0.01 | 1e-4 / 1e-4 | 0.6654 | 1.9453 | 5/5 | 0.483 |
| strat_s3_top2_a001_e1e4_b1e4 | 3 | 2 | 0.01 | 1e-4 / 1e-4 | 0.6654 | 1.9452 | 5/5 | 0.483 |
| strat_s3_top1_a0005_e1e4_b1e4 | 3 | 1 | 0.005 | 1e-4 / 1e-4 | 0.6655 | 1.9455 | 5/5 | 0.483 |
| strat_s3_top1_a002_e5e5_b5e5 | 3 | 1 | 0.02 | 5e-5 / 5e-5 | 0.6652 | 1.9449 | 5/5 | 0.483 |

Notes:
- All stratified variants outperform baseline on eval loss and perplexity with tight spread.
- Generation quality metrics are stable across configs; improvements are subtle but consistent in perplexity.

#### Best Configuration
- strat_s3_top1_a002_e5e5_b5e5
  - S=3, top_k=1, α=0.02, aux=(5e-5, 5e-5)

#### Next Steps
1) Extend sweep: S ∈ {2,3,4,6}, top_k ∈ {1,2,3}, α ∈ {0.005, 0.01, 0.02, 0.04}, aux ∈ {(2e-5,2e-5),(5e-5,5e-5),(1e-4,1e-4)}
2) Seed repeats: run best 3× with different seeds to confirm stability
3) Validate best on HF Trainer + Accelerate with a larger dataset (5k–10k samples) and multi-GPU
4) Evaluate on generation benchmarks (perplexity, BLEU/ROUGE, diversity, average length)
