folds: [0, 1, 2]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 34.1 ± 1.5 | 28.8 ± 1.5 | 0.9763 ± 0.0214 |
| single_left | 43.3 ± 3.4 | 33.7 ± 3.3 | 0.9203 ± 0.0141 |
| single_right | 34.3 ± 3.1 | 26.1 ± 2.5 | 0.9640 ± 0.0153 |
| fuse_mean | 24.7 ± 2.4 | 24.3 ± 2.3 | 0.9983 ± 0.0016 |
| fuse_median | 26.6 ± 2.3 | 25.1 ± 2.3 | 0.9964 ± 0.0029 |
| best_single_oracle | 32.5 ± 1.8 | 26.2 ± 2.5 | 0.9791 ± 0.0088 |
| learned (fixed steps) | 23.5 ± 1.8 | 22.8 ± 1.7 | 0.9970 ± 0.0024 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 29.3 | 27.2 | 28.7 | 53.6 |
| single_left | 33.8 | 28.9 | 32.4 | 84.4 |
| single_right | 24.3 | 26.5 | 25.0 | 71.0 |
| fuse_mean | 21.5 | 26.7 | 23.0 | 35.4 |
| fuse_median | 22.2 | 26.8 | 23.5 | 40.8 |
| best_single_oracle | 24.5 | 26.7 | 25.1 | 60.1 |
| learned (fixed steps) | 21.6 | 25.0 | 22.6 | 33.0 |

learned mean view weights (front/left/right): [0.513, 0.226, 0.261] ± [0.052, 0.019, 0.035]

### view corruption: drop (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 23.5 | 22.8 | nan | 24.7 | nan |
| front | 25.6 | 23.1 | 0.000 | 26.9 | 26.9 |
| left | 27.6 | 23.4 | 0.000 | 28.2 | 28.2 |
| right | 29.5 | 27.0 | 0.000 | 30.9 | 30.9 |

### view corruption: zero (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 23.5 | 22.8 | nan | 24.7 | nan |
| front | 27.7 | 23.3 | 0.019 | 84.1 | 26.9 |
| left | 28.8 | 23.7 | 0.016 | 86.0 | 28.2 |
| right | 29.6 | 27.8 | 0.018 | 83.1 | 30.9 |

### view corruption: noise (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 23.5 | 22.8 | nan | 24.7 | nan |
| front | 30.3 | 27.0 | 0.210 | 59.3 | 26.9 |
| left | 28.5 | 25.4 | 0.100 | 59.2 | 28.2 |
| right | 30.9 | 28.2 | 0.121 | 59.2 | 30.9 |