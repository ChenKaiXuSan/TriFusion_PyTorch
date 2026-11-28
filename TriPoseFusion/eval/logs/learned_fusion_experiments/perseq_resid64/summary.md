folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 130.9 ± 29.7 | 40.9 ± 8.4 | 0.5460 ± 0.0814 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 86.0 | 23.2 | 54.6 | 278.6 |
| single_left | 83.3 | 23.0 | 53.2 | 255.6 |
| single_right | 80.0 | 22.4 | 51.2 | 243.2 |
| fuse_mean | 81.2 | 22.0 | 51.6 | 233.5 |
| fuse_median | 81.2 | 22.0 | 51.6 | 239.6 |
| best_single_oracle | 80.2 | 22.6 | 51.4 | 233.7 |
| learned (fixed steps) | 74.0 | 26.3 | 50.1 | 203.1 |

learned mean view weights (front/left/right): [0.323, 0.332, 0.344] ± [0.006, 0.009, 0.009]

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 130.9 | 40.9 | nan | 148.6 | nan |
| front | 163.8 | 49.4 | 0.249 | 175.1 | 147.9 |
| left | 172.2 | 62.7 | 0.249 | 176.9 | 153.1 |
| right | 185.3 | 71.0 | 0.251 | 177.0 | 155.1 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 |
|---|---|
| single_front | 26.9 |
| single_left | 31.9 |
| single_right | 24.0 |
| fuse_mean | 23.1 |
| fuse_median | 23.5 |
| learned | 36.0 |