### seq-mean 口径（逐序列均值，n=20）
| 方法 | canonicalizer | MPJPE | PA-MPJPE | MPJPE(PA 同掩码) | PCK@0.10 |
|---|---|---|---|---|---|
| single front | robust | 0.1995 | 0.0726 | 0.1995 | 0.4215 |
| single left | robust | 0.2056 | 0.0647 | 0.2056 | 0.4311 |
| single right | robust | 0.1881 | 0.0627 | 0.1881 | 0.4347 |
| best single (oracle, per-sequence) | robust | 0.1736 | 0.0653 | 0.1736 | 0.4406 |
| fuse mean (= uniform gate) | robust | 0.1858 | 0.0595 | 0.1858 | 0.4524 |
| fuse median | robust | 0.1859 | 0.0619 | 0.1859 | 0.4423 |
| TriPoseFusion (self-supervised) | robust | 0.1857 | 0.0593 | 0.1857 | 0.4528 |

### fold-agg 口径（eval_trifusion fold 级聚合）+ 关节分组
| 方法 | MPJPE | PA-MPJPE | MPJPE(PA 同掩码) | PCK@0.10 | head | shoulders_neck | body | hands |
|---|---|---|---|---|---|---|---|---|
| single front | 0.2042 | 0.0758 | 0.2042 | 0.4210 | 0.102 | 0.033 | 0.067 | 0.343 |
| single left | 0.2064 | 0.0679 | 0.2064 | 0.4304 | 0.106 | 0.034 | 0.070 | 0.348 |
| single right | 0.1908 | 0.0659 | 0.1908 | 0.4340 | 0.099 | 0.033 | 0.066 | 0.317 |
| fuse mean (= uniform gate) | 0.1888 | 0.0632 | 0.1888 | 0.4511 | 0.101 | 0.033 | 0.067 | 0.317 |
| fuse median | 0.1886 | 0.0654 | 0.1886 | 0.4417 | 0.100 | 0.032 | 0.066 | 0.315 |
| TriPoseFusion (self-supervised) | 0.1887 | 0.0629 | 0.1887 | 0.4514 | 0.101 | 0.033 | 0.067 | 0.316 |
