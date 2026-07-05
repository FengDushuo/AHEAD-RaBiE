| Method | Claim | MAE | RMSE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Strict superblend | held-out baseline | 1.667 +/- 0.350 | 2.213 +/- 0.490 | 0.531 +/- 0.180 | 0.546 +/- 0.176 |
| Chemistry-spike prior | held-out baseline | 0.795 +/- 0.139 | 0.989 +/- 0.141 | 0.917 +/- 0.038 | 0.886 +/- 0.057 |
| Few-shot calibrated chemistry prior | main held-out result | 0.388 +/- 0.087 | 0.492 +/- 0.112 | 0.981 +/- 0.011 | 0.967 +/- 0.025 |
| Few-shot calibrated chemistry prior | leave-one-dopant-out | 0.390 +/- 0.236 | 0.427 +/- 0.238 |  |  |
| Bidirectional chemistry prior | full-data reference, not held-out | 0.386 | 0.436 | 0.989 | 0.986 |
| Bidirectional chemistry prior | post-hoc upper bound, not held-out | 0.216 | 0.240 | 0.998 | 0.996 |
