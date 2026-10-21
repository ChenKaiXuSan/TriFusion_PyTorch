folds: [0, 1, 2]

| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |
|---|---|---|---|
| single_front | 0.0341 ± 0.0015 | 0.0288 ± 0.0015 | 0.9763 ± 0.0214 |
| single_left | 0.0433 ± 0.0034 | 0.0337 ± 0.0033 | 0.9203 ± 0.0141 |
| single_right | 0.0343 ± 0.0031 | 0.0261 ± 0.0025 | 0.9640 ± 0.0153 |
| fuse_mean | 0.0247 ± 0.0024 | 0.0243 ± 0.0023 | 0.9983 ± 0.0016 |
| fuse_median | 0.0266 ± 0.0023 | 0.0251 ± 0.0023 | 0.9964 ± 0.0029 |
| best_single_oracle | 0.0325 ± 0.0018 | 0.0262 ± 0.0025 | 0.9791 ± 0.0088 |
| learned (fixed steps) | 0.0217 ± 0.0016 | 0.0199 ± 0.0016 | 0.9959 ± 0.0038 |

| method | head | shoulders/neck | body | hands |
|---|---|---|---|---|
| single_front | 0.0293 | 0.0272 | 0.0287 | 0.0536 |
| single_left | 0.0338 | 0.0289 | 0.0324 | 0.0844 |
| single_right | 0.0243 | 0.0265 | 0.0250 | 0.0710 |
| fuse_mean | 0.0215 | 0.0267 | 0.0230 | 0.0354 |
| fuse_median | 0.0222 | 0.0268 | 0.0235 | 0.0408 |
| best_single_oracle | 0.0245 | 0.0267 | 0.0251 | 0.0601 |
| learned (fixed steps) | 0.0190 | 0.0242 | 0.0205 | 0.0301 |

uncertainty calibration (mean over folds): spearman(scale, error)=0.507, error most-confident decile 0.0106 vs least-confident 0.0344 (all 0.0217)

learned mean view weights (front/left/right): [0.398, 0.239, 0.363] ± [0.054, 0.014, 0.041]