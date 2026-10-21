folds: [0, 1, 2, 3, 4]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.1723 ± 0.0249 | 0.0588 ± 0.0106 | 0.4586 ± 0.0620 |
| single_left | 0.1606 ± 0.0288 | 0.0457 ± 0.0106 | 0.4730 ± 0.0603 |
| single_right | 0.1524 ± 0.0253 | 0.0454 ± 0.0102 | 0.4774 ± 0.0600 |
| fuse_mean | 0.1486 ± 0.0256 | 0.0440 ± 0.0092 | 0.4972 ± 0.0546 |
| fuse_median | 0.1515 ± 0.0241 | 0.0458 ± 0.0101 | 0.4848 ± 0.0579 |
| best_single_oracle | 0.1447 ± 0.0216 | 0.0462 ± 0.0117 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 0.1309 ± 0.0297 | 0.0409 ± 0.0084 | 0.5460 ± 0.0814 |

| method | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 0.0860 | 0.0232 | 0.0546 | 0.2786 |
| single_left | 0.0833 | 0.0230 | 0.0532 | 0.2556 |
| single_right | 0.0800 | 0.0224 | 0.0512 | 0.2432 |
| fuse_mean | 0.0812 | 0.0220 | 0.0516 | 0.2335 |
| fuse_median | 0.0812 | 0.0220 | 0.0516 | 0.2396 |
| best_single_oracle | 0.0802 | 0.0226 | 0.0514 | 0.2337 |
| learned (fixed steps) | 0.0740 | 0.0263 | 0.0501 | 0.2031 |

learned mean view weights (front/left/right): [0.323, 0.332, 0.344] ± [0.006, 0.009, 0.009]

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1309 | 0.0409 | nan | 0.1486 | nan |
| front | 0.1638 | 0.0494 | 0.249 | 0.1751 | 0.1479 |
| left | 0.1722 | 0.0627 | 0.249 | 0.1769 | 0.1531 |
| right | 0.1853 | 0.0710 | 0.251 | 0.1770 | 0.1551 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 |
|---|---|
| single_front | 26.9 |
| single_left | 31.9 |
| single_right | 24.0 |
| fuse_mean | 23.1 |
| fuse_median | 23.5 |
| learned | 36.0 |