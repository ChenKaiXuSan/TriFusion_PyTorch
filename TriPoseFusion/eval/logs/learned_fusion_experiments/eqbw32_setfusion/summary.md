folds: [0, 1, 2, 3, 4]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 172.3 ± 24.9 | 58.8 ± 10.6 | 0.4586 ± 0.0620 |
| single_left | 160.6 ± 28.8 | 45.7 ± 10.6 | 0.4730 ± 0.0603 |
| single_right | 152.4 ± 25.3 | 45.4 ± 10.2 | 0.4774 ± 0.0600 |
| fuse_mean | 148.6 ± 25.6 | 44.0 ± 9.2 | 0.4972 ± 0.0546 |
| fuse_median | 151.5 ± 24.1 | 45.8 ± 10.1 | 0.4848 ± 0.0579 |
| best_single_oracle | 144.7 ± 21.6 | 46.2 ± 11.7 | 0.4884 ± 0.0607 |
| learned (fixed steps) | 129.5 ± 30.0 | 40.1 ± 9.2 | 0.5449 ± 0.0769 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 86.0 | 23.2 | 54.6 | 278.6 |
| single_left | 83.3 | 23.0 | 53.2 | 255.6 |
| single_right | 80.0 | 22.4 | 51.2 | 243.2 |
| fuse_mean | 81.2 | 22.0 | 51.6 | 233.5 |
| fuse_median | 81.2 | 22.0 | 51.6 | 239.6 |
| best_single_oracle | 80.2 | 22.6 | 51.4 | 233.7 |
| learned (fixed steps) | 73.1 | 22.2 | 47.7 | 202.9 |

uncertainty calibration (mean over folds): spearman(scale, error)=0.704, error most-confident decile 13.0 mm vs least-confident 244.8 mm (all 145.9 mm)

learned mean view weights (front/left/right): [0.162, 0.389, 0.449] ± [0.022, 0.059, 0.056]