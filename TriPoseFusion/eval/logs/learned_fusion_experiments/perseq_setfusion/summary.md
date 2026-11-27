folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 132.0 ± 31.4 | 42.3 ± 8.7 | 0.5383 ± 0.0874 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 86.0 | 23.2 | 54.6 | 278.6 |
| single_left | 83.3 | 23.0 | 53.2 | 255.6 |
| single_right | 80.0 | 22.4 | 51.2 | 243.2 |
| fuse_mean | 81.2 | 22.0 | 51.6 | 233.5 |
| fuse_median | 81.2 | 22.0 | 51.6 | 239.6 |
| best_single_oracle | 80.2 | 22.6 | 51.4 | 233.7 |
| learned (fixed steps) | 73.0 | 24.6 | 48.8 | 207.8 |

uncertainty calibration (mean over folds): spearman(scale, error)=0.738, error most-confident decile 16.3 mm vs least-confident 271.9 mm (all 148.4 mm)

learned mean view weights (front/left/right): [0.143, 0.315, 0.542] ± [0.039, 0.055, 0.076]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 132.0 | 42.3 | nan | 148.6 | nan |
| front | 132.7 | 42.7 | 0.000 | 147.9 | 147.9 |
| left | 135.8 | 46.5 | 0.000 | 153.1 | 153.1 |
| right | 140.7 | 45.9 | 0.000 | 155.1 | 155.1 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 132.0 | 42.3 | nan | 148.6 | nan |
| front | 133.2 | 43.3 | 0.011 | 175.1 | 147.9 |
| left | 136.3 | 46.5 | 0.020 | 176.9 | 153.1 |
| right | 140.5 | 47.6 | 0.025 | 177.0 | 155.1 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 132.0 | 42.3 | nan | 148.6 | nan |
| front | 133.3 | 46.3 | 0.094 | 166.8 | 147.9 |
| left | 135.8 | 52.5 | 0.197 | 166.8 | 153.1 |
| right | 141.4 | 65.5 | 0.375 | 166.8 | 155.1 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 26.7 | 28.4 | 30.7 | 26.6 | 30.7 |