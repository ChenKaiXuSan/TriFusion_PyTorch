folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 135.8 ± 28.1 | 41.3 ± 7.9 | 0.5168 ± 0.0764 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 86.0 | 23.2 | 54.6 | 278.6 |
| single_left | 83.3 | 23.0 | 53.2 | 255.6 |
| single_right | 80.0 | 22.4 | 51.2 | 243.2 |
| fuse_mean | 81.2 | 22.0 | 51.6 | 233.5 |
| fuse_median | 81.2 | 22.0 | 51.6 | 239.6 |
| best_single_oracle | 80.2 | 22.6 | 51.4 | 233.7 |
| learned (fixed steps) | 80.8 | 24.8 | 52.8 | 210.0 |

learned mean view weights (front/left/right): [0.12, 0.378, 0.503] ± [0.039, 0.047, 0.021]

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 135.8 | 41.3 | nan | 148.6 | nan |
| front | 137.3 | 41.6 | 0.013 | 175.1 | 147.9 |
| left | 135.8 | 43.7 | 0.028 | 176.9 | 153.1 |
| right | 142.3 | 44.6 | 0.013 | 177.0 | 155.1 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 135.8 | 41.3 | nan | 148.6 | nan |
| front | 137.4 | 43.9 | 0.070 | 166.8 | 147.9 |
| left | 137.7 | 50.8 | 0.177 | 166.8 | 153.1 |
| right | 142.2 | 54.0 | 0.212 | 166.8 | 155.1 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 35.2 | 33.7 | 36.8 | 44.7 | 31.3 |