folds: [0, 1, 2, 3, 4]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.1723 ± 0.0249 | 0.0588 ± 0.0106 | 0.4586 ± 0.0620 |
| single_left | 0.1606 ± 0.0288 | 0.0457 ± 0.0106 | 0.4730 ± 0.0603 |
| single_right | 0.1524 ± 0.0253 | 0.0454 ± 0.0102 | 0.4774 ± 0.0600 |
| fuse_mean | 0.1486 ± 0.0256 | 0.0440 ± 0.0092 | 0.4972 ± 0.0546 |
| fuse_median | 0.1515 ± 0.0241 | 0.0458 ± 0.0101 | 0.4848 ± 0.0579 |
| best_single_oracle | 0.1447 ± 0.0216 | 0.0462 ± 0.0117 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 0.1492 ± 0.0255 | 0.0444 ± 0.0086 | 0.5034 ± 0.0612 |

| method | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 0.0827 | 0.0218 | 0.0523 | 0.2828 |
| single_left | 0.0801 | 0.0214 | 0.0508 | 0.2502 |
| single_right | 0.0782 | 0.0211 | 0.0496 | 0.2408 |
| fuse_mean | 0.0782 | 0.0207 | 0.0494 | 0.2320 |
| fuse_median | 0.0784 | 0.0208 | 0.0496 | 0.2383 |
| best_single_oracle | 0.0780 | 0.0214 | 0.0497 | 0.2359 |
| learned (fixed steps) | 0.0821 | 0.0510 | 0.0665 | 0.2186 |

learned mean view weights (front/left/right): [0.318, 0.335, 0.347] ± [0.009, 0.007, 0.011]

### view corruption: drop (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1492 | 0.0444 | nan | 0.1486 | nan |
| front | 0.1484 | 0.0411 | 0.000 | 0.1479 | 0.1479 |
| left | 0.1539 | 0.0498 | 0.000 | 0.1531 | 0.1531 |
| right | 0.1553 | 0.0479 | 0.000 | 0.1551 | 0.1551 |

### view corruption: zero (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1492 | 0.0444 | nan | 0.1486 | nan |
| front | 0.1508 | 0.0451 | 0.038 | 0.1751 | 0.1479 |
| left | 0.1552 | 0.0511 | 0.036 | 0.1769 | 0.1531 |
| right | 0.1572 | 0.0513 | 0.038 | 0.1770 | 0.1551 |

### view corruption: noise (mean over 5 folds)

| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |
|---|---|---|---|---|---|
| clean | 0.1492 | 0.0444 | nan | 0.1486 | nan |
| front | 0.1502 | 0.0527 | 0.223 | 0.1668 | 0.1479 |
| left | 0.1521 | 0.0554 | 0.239 | 0.1668 | 0.1531 |
| right | 0.1530 | 0.0571 | 0.253 | 0.1668 | 0.1551 |

### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)

| method | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| single_front | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| single_left | 31.9 | 31.9 | 31.9 | 31.9 | 31.9 |
| single_right | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |
| fuse_mean | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| fuse_median | 23.5 | 23.5 | 23.5 | 23.5 | 23.5 |
| learned | 22.4 | 23.4 | 25.7 | 21.7 | 22.3 |