folds: [0, 1, 2, 3, 4]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.1723 ± 0.0249 | 0.0588 ± 0.0106 | 0.4586 ± 0.0620 |
| single_left | 0.1606 ± 0.0288 | 0.0457 ± 0.0106 | 0.4730 ± 0.0603 |
| single_right | 0.1524 ± 0.0253 | 0.0454 ± 0.0102 | 0.4774 ± 0.0600 |
| fuse_mean | 0.1486 ± 0.0256 | 0.0440 ± 0.0092 | 0.4972 ± 0.0546 |
| fuse_median | 0.1515 ± 0.0241 | 0.0458 ± 0.0101 | 0.4848 ± 0.0579 |
| best_single_oracle | 0.1447 ± 0.0216 | 0.0462 ± 0.0117 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 0.1320 ± 0.0314 | 0.0423 ± 0.0087 | 0.5383 ± 0.0874 |

| method | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 0.0860 | 0.0232 | 0.0546 | 0.2786 |
| single_left | 0.0833 | 0.0230 | 0.0532 | 0.2556 |
| single_right | 0.0800 | 0.0224 | 0.0512 | 0.2432 |
| fuse_mean | 0.0812 | 0.0220 | 0.0516 | 0.2335 |
| fuse_median | 0.0812 | 0.0220 | 0.0516 | 0.2396 |
| best_single_oracle | 0.0802 | 0.0226 | 0.0514 | 0.2337 |
| learned (fixed steps) | 0.0730 | 0.0246 | 0.0488 | 0.2078 |

uncertainty calibration (mean over folds): spearman(scale, error)=0.738, error most-confident decile 0.0163 vs least-confident 0.2719 (all 0.1484)

learned mean view weights (front/left/right): [0.143, 0.315, 0.542] ± [0.039, 0.055, 0.076]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1320 | 0.0423 | nan | 0.1486 | nan |
| front | 0.1327 | 0.0427 | 0.000 | 0.1479 | 0.1479 |
| left | 0.1358 | 0.0465 | 0.000 | 0.1531 | 0.1531 |
| right | 0.1407 | 0.0459 | 0.000 | 0.1551 | 0.1551 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1320 | 0.0423 | nan | 0.1486 | nan |
| front | 0.1332 | 0.0433 | 0.011 | 0.1751 | 0.1479 |
| left | 0.1363 | 0.0465 | 0.020 | 0.1769 | 0.1531 |
| right | 0.1405 | 0.0476 | 0.025 | 0.1770 | 0.1551 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1320 | 0.0423 | nan | 0.1486 | nan |
| front | 0.1333 | 0.0463 | 0.094 | 0.1668 | 0.1479 |
| left | 0.1358 | 0.0525 | 0.197 | 0.1668 | 0.1531 |
| right | 0.1414 | 0.0655 | 0.375 | 0.1668 | 0.1551 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 26.7 | 28.4 | 30.7 | 26.6 | 30.7 |