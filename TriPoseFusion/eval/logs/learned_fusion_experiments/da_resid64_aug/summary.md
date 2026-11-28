folds: [0, 1, 2]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 34.1 ± 1.5 | 28.8 ± 1.5 | 0.9763 ± 0.0214 |
| single_left | 43.3 ± 3.4 | 33.7 ± 3.3 | 0.9203 ± 0.0141 |
| single_right | 34.3 ± 3.1 | 26.1 ± 2.5 | 0.9640 ± 0.0153 |
| fuse_mean | 24.7 ± 2.4 | 24.3 ± 2.3 | 0.9983 ± 0.0016 |
| fuse_median | 26.6 ± 2.3 | 25.1 ± 2.3 | 0.9964 ± 0.0029 |
| best_single_oracle | 32.5 ± 1.8 | 26.2 ± 2.5 | 0.9791 ± 0.0088 |
| learned (fixed steps) | 21.6 ± 1.9 | 19.3 ± 2.1 | 0.9973 ± 0.0017 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 29.3 | 27.2 | 28.7 | 53.6 |
| single_left | 33.8 | 28.9 | 32.4 | 84.4 |
| single_right | 24.3 | 26.5 | 25.0 | 71.0 |
| fuse_mean | 21.5 | 26.7 | 23.0 | 35.4 |
| fuse_median | 22.2 | 26.8 | 23.5 | 40.8 |
| best_single_oracle | 24.5 | 26.7 | 25.1 | 60.1 |
| learned (fixed steps) | 18.4 | 24.8 | 20.2 | 29.6 |

learned mean view weights (front/left/right): [0.491, 0.238, 0.271] ± [0.045, 0.018, 0.029]

### view corruption: drop (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 21.6 | 19.3 | nan | 24.7 | nan |
| front | 23.1 | 19.9 | 0.000 | 26.9 | 26.9 |
| left | 24.6 | 20.2 | 0.000 | 28.2 | 28.2 |
| right | 24.7 | 21.6 | 0.000 | 30.9 | 30.9 |

### view corruption: zero (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 21.6 | 19.3 | nan | 24.7 | nan |
| front | 24.4 | 20.4 | 0.029 | 84.1 | 26.9 |
| left | 25.5 | 21.1 | 0.027 | 86.0 | 28.2 |
| right | 25.8 | 22.7 | 0.028 | 83.1 | 30.9 |

### view corruption: noise (mean over 3 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 21.6 | 19.3 | nan | 24.7 | nan |
| front | 28.4 | 24.1 | 0.223 | 59.3 | 26.9 |
| left | 26.7 | 22.6 | 0.121 | 59.2 | 28.2 |
| right | 28.3 | 24.2 | 0.141 | 59.2 | 30.9 |