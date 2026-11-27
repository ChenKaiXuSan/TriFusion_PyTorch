folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 130.3 ± 28.7 | 38.7 ± 9.3 | 0.5359 ± 0.0774 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 86.0 | 23.2 | 54.6 | 278.6 |
| single_left | 83.3 | 23.0 | 53.2 | 255.6 |
| single_right | 80.0 | 22.4 | 51.2 | 243.2 |
| fuse_mean | 81.2 | 22.0 | 51.6 | 233.5 |
| fuse_median | 81.2 | 22.0 | 51.6 | 239.6 |
| best_single_oracle | 80.2 | 22.6 | 51.4 | 233.7 |
| learned (fixed steps) | 74.5 | 23.5 | 49.0 | 202.1 |

learned mean view weights (front/left/right): [0.334, 0.334, 0.334] ± [0.0, 0.0, 0.0]