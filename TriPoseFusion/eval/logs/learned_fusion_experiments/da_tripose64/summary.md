folds: [0, 1, 2]

| method | MPJPE (mm) | PA-MPJPE (mm) | PCK@100mm |
|---|---|---|---|
| single_front | 34.1 ± 1.5 | 28.8 ± 1.5 | 0.9763 ± 0.0214 |
| single_left | 43.3 ± 3.4 | 33.7 ± 3.3 | 0.9203 ± 0.0141 |
| single_right | 34.3 ± 3.1 | 26.1 ± 2.5 | 0.9640 ± 0.0153 |
| fuse_mean | 24.7 ± 2.4 | 24.3 ± 2.3 | 0.9983 ± 0.0016 |
| fuse_median | 26.6 ± 2.3 | 25.1 ± 2.3 | 0.9964 ± 0.0029 |
| best_single_oracle | 32.5 ± 1.8 | 26.2 ± 2.5 | 0.9791 ± 0.0088 |
| learned (fixed steps) | 23.7 ± 2.1 | 21.6 ± 2.1 | 0.9954 ± 0.0034 |

| method (group MPJPE, mm) | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 29.3 | 27.2 | 28.7 | 53.6 |
| single_left | 33.8 | 28.9 | 32.4 | 84.4 |
| single_right | 24.3 | 26.5 | 25.0 | 71.0 |
| fuse_mean | 21.5 | 26.7 | 23.0 | 35.4 |
| fuse_median | 22.2 | 26.8 | 23.5 | 40.8 |
| best_single_oracle | 24.5 | 26.7 | 25.1 | 60.1 |
| learned (fixed steps) | 22.5 | 24.7 | 23.1 | 31.7 |

learned mean view weights (front/left/right): [0.436, 0.272, 0.292] ± [0.039, 0.012, 0.029]