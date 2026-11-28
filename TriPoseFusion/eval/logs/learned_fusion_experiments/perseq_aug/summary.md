folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 149.2 ± 25.5 | 44.4 ± 8.6 | 0.5034 ± 0.0612 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 82.7 | 21.8 | 52.3 | 282.8 |
| single_left | 80.1 | 21.4 | 50.8 | 250.2 |
| single_right | 78.2 | 21.1 | 49.6 | 240.8 |
| fuse_mean | 78.2 | 20.7 | 49.4 | 232.0 |
| fuse_median | 78.4 | 20.8 | 49.6 | 238.3 |
| best_single_oracle | 78.0 | 21.4 | 49.7 | 235.9 |
| learned (fixed steps) | 82.1 | 51.0 | 66.5 | 218.6 |

learned mean view weights (front/left/right): [0.318, 0.335, 0.347] ± [0.009, 0.007, 0.011]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.2 | 44.4 | nan | 148.6 | nan |
| front | 148.4 | 41.1 | 0.000 | 147.9 | 147.9 |
| left | 153.9 | 49.8 | 0.000 | 153.1 | 153.1 |
| right | 155.3 | 47.9 | 0.000 | 155.1 | 155.1 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.2 | 44.4 | nan | 148.6 | nan |
| front | 150.8 | 45.1 | 0.038 | 175.1 | 147.9 |
| left | 155.2 | 51.1 | 0.036 | 176.9 | 153.1 |
| right | 157.2 | 51.3 | 0.038 | 177.0 | 155.1 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.2 | 44.4 | nan | 148.6 | nan |
| front | 150.2 | 52.7 | 0.223 | 166.8 | 147.9 |
| left | 152.1 | 55.4 | 0.239 | 166.8 | 153.1 |
| right | 153.0 | 57.1 | 0.253 | 166.8 | 155.1 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 22.4 | 23.4 | 25.7 | 21.7 | 22.3 |