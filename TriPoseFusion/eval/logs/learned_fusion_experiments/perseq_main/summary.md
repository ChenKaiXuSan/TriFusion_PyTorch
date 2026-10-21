folds: [0, 1, 2, 3, 4]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.1723 ± 0.0249 | 0.0588 ± 0.0106 | 0.4586 ± 0.0620 |
| single_left | 0.1606 ± 0.0288 | 0.0457 ± 0.0106 | 0.4730 ± 0.0603 |
| single_right | 0.1524 ± 0.0253 | 0.0454 ± 0.0102 | 0.4774 ± 0.0600 |
| fuse_mean | 0.1486 ± 0.0256 | 0.0440 ± 0.0092 | 0.4972 ± 0.0546 |
| fuse_median | 0.1515 ± 0.0241 | 0.0458 ± 0.0101 | 0.4848 ± 0.0579 |
| best_single_oracle | 0.1447 ± 0.0216 | 0.0462 ± 0.0117 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 0.1493 ± 0.0258 | 0.0442 ± 0.0088 | 0.5071 ± 0.0629 |

learned mean view weights (front/left/right): [0.322, 0.334, 0.344] ± [0.003, 0.006, 0.007]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1493 | 0.0442 | nan | 0.1486 | nan |
| front | 0.1484 | 0.0411 | 0.000 | 0.1479 | 0.1479 |
| left | 0.1535 | 0.0497 | 0.000 | 0.1531 | 0.1531 |
| right | 0.1552 | 0.0478 | 0.000 | 0.1551 | 0.1551 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1493 | 0.0442 | nan | 0.1486 | nan |
| front | 0.1601 | 0.0474 | 0.113 | 0.1751 | 0.1479 |
| left | 0.1639 | 0.0540 | 0.112 | 0.1769 | 0.1531 |
| right | 0.1658 | 0.0530 | 0.113 | 0.1770 | 0.1551 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1493 | 0.0442 | nan | 0.1486 | nan |
| front | 0.1541 | 0.0592 | 0.263 | 0.1668 | 0.1479 |
| left | 0.1555 | 0.0613 | 0.273 | 0.1668 | 0.1531 |
| right | 0.1563 | 0.0632 | 0.286 | 0.1668 | 0.1551 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 23.4 | 22.6 | 24.8 | 22.3 | 22.2 |