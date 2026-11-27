folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 880.7 ± 163.6 | 384.0 ± 185.8 | 0.1176 ± 0.0326 |
| single_left | 855.6 ± 172.6 | 336.0 ± 175.3 | 0.1120 ± 0.0408 |
| single_right | 868.0 ± 168.1 | 345.8 ± 175.8 | 0.1080 ± 0.0428 |
| fuse_mean | 865.3 ± 168.1 | 355.4 ± 181.3 | 0.1127 ± 0.0394 |
| fuse_median | 864.5 ± 170.4 | 356.0 ± 182.8 | 0.1137 ± 0.0402 |
| best_single_oracle | 846.4 ± 166.2 | 348.8 ± 183.8 | 0.1141 ± 0.0409 |
| learned (fixed steps) | 766.9 ± 201.5 | 317.3 ± 168.9 | 0.1770 ± 0.0416 |

learned mean view weights (front/left/right): [0.405, 0.319, 0.276] ± [0.024, 0.04, 0.032]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 766.9 | 317.3 | nan | 865.3 | nan |
| front | 760.5 | 283.8 | 0.000 | 860.0 | 860.0 |
| left | 775.5 | 317.2 | 0.000 | 872.2 | 872.2 |
| right | 765.9 | 312.3 | 0.000 | 865.8 | 865.8 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 766.9 | 317.3 | nan | 865.3 | nan |
| front | 915.2 | 487.7 | 0.547 | 928.4 | 860.0 |
| left | 923.5 | 495.0 | 0.502 | 935.6 | 872.2 |
| right | 916.7 | 492.7 | 0.491 | 931.5 | 865.8 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE (mm) | learned PA (mm) | weight on corrupted view | mean-fusion MPJPE, same corruption (mm) | mean excl. view, oracle (mm) |
|---|---|---|---|---|---|
| clean | 766.9 | 317.3 | nan | 865.3 | nan |
| front | 819.7 | 395.7 | 0.629 | 871.5 | 860.0 |
| left | 811.9 | 388.0 | 0.581 | 871.5 | 872.2 |
| right | 815.9 | 389.9 | 0.544 | 871.5 | 865.8 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 28.7 | 31.5 | 37.4 | 31.6 | 35.6 |