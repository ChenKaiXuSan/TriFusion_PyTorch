folds: [0, 1, 2, 3, 4]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.8807 ± 0.1636 | 0.3840 ± 0.1858 | 0.1176 ± 0.0326 |
| single_left | 0.8556 ± 0.1726 | 0.3360 ± 0.1753 | 0.1120 ± 0.0408 |
| single_right | 0.8680 ± 0.1681 | 0.3458 ± 0.1758 | 0.1080 ± 0.0428 |
| fuse_mean | 0.8653 ± 0.1681 | 0.3554 ± 0.1813 | 0.1127 ± 0.0394 |
| fuse_median | 0.8645 ± 0.1704 | 0.3560 ± 0.1828 | 0.1137 ± 0.0402 |
| best_single_oracle | 0.8464 ± 0.1662 | 0.3488 ± 0.1838 | 0.1141 ± 0.0409 |
| learned (fixed steps) | 0.7669 ± 0.2015 | 0.3173 ± 0.1689 | 0.1770 ± 0.0416 |

learned mean view weights (front/left/right): [0.405, 0.319, 0.276] ± [0.024, 0.04, 0.032]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.7669 | 0.3173 | nan | 0.8653 | nan |
| front | 0.7605 | 0.2838 | 0.000 | 0.8600 | 0.8600 |
| left | 0.7755 | 0.3172 | 0.000 | 0.8722 | 0.8722 |
| right | 0.7659 | 0.3123 | 0.000 | 0.8658 | 0.8658 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.7669 | 0.3173 | nan | 0.8653 | nan |
| front | 0.9152 | 0.4877 | 0.547 | 0.9284 | 0.8600 |
| left | 0.9235 | 0.4950 | 0.502 | 0.9356 | 0.8722 |
| right | 0.9167 | 0.4927 | 0.491 | 0.9315 | 0.8658 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.7669 | 0.3173 | nan | 0.8653 | nan |
| front | 0.8197 | 0.3957 | 0.629 | 0.8715 | 0.8600 |
| left | 0.8119 | 0.3880 | 0.581 | 0.8715 | 0.8722 |
| right | 0.8159 | 0.3899 | 0.544 | 0.8715 | 0.8658 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 28.7 | 31.5 | 37.4 | 31.6 | 35.6 |