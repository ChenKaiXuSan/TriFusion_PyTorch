folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 149.3 ± 25.8 | 44.2 ± 8.8 | 0.5071 ± 0.0629 |

learned mean view weights (front/left/right): [0.322, 0.334, 0.344] ± [0.003, 0.006, 0.007]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.3 | 44.2 | nan | 148.6 | nan |
| front | 148.4 | 41.1 | 0.000 | 147.9 | 147.9 |
| left | 153.5 | 49.7 | 0.000 | 153.1 | 153.1 |
| right | 155.2 | 47.8 | 0.000 | 155.1 | 155.1 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.3 | 44.2 | nan | 148.6 | nan |
| front | 160.1 | 47.4 | 0.113 | 175.1 | 147.9 |
| left | 163.9 | 54.0 | 0.112 | 176.9 | 153.1 |
| right | 165.8 | 53.0 | 0.113 | 177.0 | 155.1 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 149.3 | 44.2 | nan | 148.6 | nan |
| front | 154.1 | 59.2 | 0.263 | 166.8 | 147.9 |
| left | 155.5 | 61.3 | 0.273 | 166.8 | 153.1 |
| right | 156.3 | 63.2 | 0.286 | 166.8 | 155.1 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 23.4 | 22.6 | 24.8 | 22.3 | 22.2 |