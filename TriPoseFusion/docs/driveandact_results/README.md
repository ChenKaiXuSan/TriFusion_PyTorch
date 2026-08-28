# Drive&Act 公开数据集评测结果（2026-08-28）

回应评审 xS2o "Mandatory Evaluation on Drive&Act"。脚本
`eval/eval_driveandact_baselines.py`，完整数字见同目录 `driveandact_baselines.json`。

## 设置

- **数据**：Drive&Act 官方 split_0 **test** 受试者 vp5 / vp11 / vp13，每人 2 个 run
  （共 6 段、约 126 分钟），按 1 fps 采样 → 7,855 个同步时刻，三视角
  （front=inner_mirror/ids_1，left=a_column_driver/ids_5，right=a_column_co_driver/ids_2），
  按 `.timestamps` 对齐（容差 ±21 ms）。NIR 灰度 1280×1024。
- **输入**：每视角逐帧 SAM3D-Body（sam-3d-body-dinov3，与主实验同一模型、同一设置），
  23,583 帧中 23,577 检出（99.97%）。
- **GT**：官方 `openpose_3d`（BODY-25，多视角 OpenPose 三角化伪 GT，ids_1 相机系，米制）。
  膝/踝置信度恒为 0（车内被遮挡），左耳仅 4% 有效。
- **协议**：与论文主实验一致——逐视角肩颈 canonicalize → 融合 → `compute_metrics`
  （修复后的 Umeyama PA）。关节取 GT 有标注的 **body14**
  （nose, 2 eye, 2 ear, 2 shoulder, 2 elbow, 2 wrist, 2 hip, neck）与 **upper12**（去髋）。
  两处因 BODY-25 定义必须做的调整：① down 轴用 midHip−neck（BODY-25 neck≈双肩中点，
  shoulder_mid−neck 仅 ~2 cm 会退化）；② 预测侧 neck 同样取双肩中点。
  GT 髋/肩/颈任一无效的帧整帧剔除：6,673 / 7,855 帧进入统计。neck 为 canonical
  原点，误差恒为 0（与主实验相同）。

## 结果（pooled，单位 mm）

| 方法 | body14 MPJPE | body14 PA-MPJPE | upper12 MPJPE | upper12 PA-MPJPE |
|---|---|---|---|---|
| single front | 61.1 | 35.8 | 59.5 | 27.5 |
| single left | 64.8 | 37.8 | 62.9 | 31.9 |
| single right | 54.1 | 34.0 | 50.6 | 27.3 |
| **fuse mean** | **55.8** | **31.3** | **53.6** | **23.8** |
| fuse median | 57.8 | 32.9 | 55.4 | 24.6 |

按受试者（fuse mean，body14）：vp5 52.4/51.2 · vp11 55.4/57.7 · vp13 62.1/55.5 mm
（两 run 各自 MPJPE）；PA 28.7–33.5 mm，跨人稳定。

fuse mean body14 per-joint MPJPE：shoulder 28–40 · elbow 48–59 · nose 60 ·
hip 68 · eye 73–76 · wrist 75–79 · lEar 92（仅 4% 有效）。
PCK@50mm 42%，@100mm 94%，AUC(0–150mm) 0.626。

## 与评审论点的关系

- 评审引用的 Drive&Act SOTA（22.6–30.4 mm MPJPE）是**在 Drive&Act 上全监督训练**的
  方法；本结果是 **零样本 SAM3D + 免训练融合**，PA-MPJPE 31.3 mm（上半身 23.8 mm）
  已在同一量级，canonical 系 MPJPE 55.8 mm。
- 融合一致优于最佳单视角：PA −8%（body14）/ −13%（upper12），与主实验结论方向一致。
- 论文主实验 0.412 m 的绝对量级来自 52 关节协议（含 42 个手部关节）与自建伪 GT，
  并非方法在标准协议下的表现——本表即为证据。

## 注意事项 / 局限

- GT 为 OpenPose 三角化伪 GT（非 mocap），误差下限约厘米级。
- SAM3D（MHR 默认体型）输出的躯干尺度比 GT 大约 18%（肩宽 0.348 vs 0.296 m），
  该尺度差留在未对齐 MPJPE 内，PA 消除。
- 1 fps 采样；模型（TriPoseFusion checkpoint）零样本评测需 30 fps 连续窗口，
  待补抽密集片段后另行报告。
- 数据与中间产物：`/work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval/`
  （frames / sam3d / gt），管线脚本在 `../pipeline/`。

## DLT 三角化基线（绝对坐标系，回应 LAYQ "direct 2D triangulation baseline"）

脚本 `eval/eval_driveandact_triangulation.py`，数字见 `driveandact_triangulation.json`。
用各 run 自带 `calibration.json`（内参 + k1/k2 畸变 + 四元数外参；已验证 R,t 为
world→cam，世界系 = ids_1 相机系）对三视角 SAM3D 2D 做去畸变 DLT 三角化。
**absolute** 列直接在 GT 世界系比较、不做任何对齐或规范化——与 Drive&Act 上
SOTA 报告的 MPJPE **严格同口径**；**canonical** 列与上表协议相同。
帧集与上表一致（6,673 帧），neck 取三角化后双肩中点。合成自检：还原误差 1.9e-4 m。

| 视角组合 | abs body14 MPJPE / PA | abs upper12 MPJPE / PA | canon body14 MPJPE / PA |
|---|---|---|---|
| **front+left+right** | **32.5 / 29.0** | **22.2 / 18.0** | **46.8 / 29.0** |
| left+right | 35.9 / 30.2 | 25.4 / 19.2 | 47.8 / 30.2 |
| front+left | 36.5 / 30.2 | 30.4 / 19.7 | 48.1 / 30.2 |
| front+right | 45.5 / 33.2 | 31.0 / 22.3 | 55.6 / 33.2 |

按 run（3 视角 abs body14 MPJPE）：vp11 28.8 / 28.9 · vp13 33.1 / 30.9 · vp5 37.4 / 33.6 mm。
标定/同步 sanity：GT 3D 投影到各视角与 SAM3D 2D 的像素残差中位 15.5 px（@1280×1024）。

要点：
- **零样本 SAM3D + 免训练几何三角化，在 Drive&Act 官方 test 受试者上绝对 MPJPE
  32.5 mm（上半身 22.2 mm）**，已落在评审引用的全监督 SOTA 区间（22.6–30.4 mm）边缘，
  上半身甚至优于该区间下限。评审"412 mm 说明系统输出不可用"的推断不成立：
  量级差异来自 52 关节手部协议与自建伪 GT，而非关键点级融合范式本身。
- 三角化的 canonical 系 MPJPE（46.8 mm）优于 canonical 系 mean 融合（55.8 mm）；
  PA 两者相当（29.0 vs 31.3 mm）——几何约束主要修正的是全局尺度/位置，
  而非形状。该基线依赖精确标定，论文场景（无标定、杆载/车载自由布局）不可用，
  这正是免标定关键点融合的设计动机。
- 去掉任一视角误差上升 10–40%（front+right 最差 45.5 mm），说明三视角冗余有实际贡献。

## TriPoseFusion checkpoint 零样本评测（30 fps 密集片段）

脚本 `eval/eval_driveandact_model.py`，数字见 `driveandact_model_full.json` /
`driveandact_model_robust_canon.json`。模型按训练同样的 **16 帧连续 30 fps 窗口**推理，
数据为同 6 个 run 各 3 段 × 20 s 连续片段（`tripose_eval_dense/`，32,400 帧，
SAM3D 检出 32,399），共 665 个窗口 / 10,640 帧。checkpoint 为论文原始（修复前代码）
fold-0 权重，**未在 Drive&Act 上训练或微调**。

关节协议 **hs10**：模型输出为 52 关节（无肘/髋），与 BODY-25 的公共集为
nose, 2 eye, 2 ear, 2 shoulder, 2 wrist, neck（预测侧 neck 取双肩中点）。GT 无髋无法
对齐到模型 canonical 系，故主指标 PA-MPJPE，另报刚性对齐（旋转+平移、无尺度）MPJPE。
基线在模型自身 `_canonicalize_pose` 系内计算，与模型完全同协议。

| 方法（hs10，mm） | PA-MPJPE | rigid-MPJPE |
|---|---|---|
| **TriPoseFusion full（零样本）** | **23.7** | 32.5 |
| TriPoseFusion robust_canon（零样本） | 24.7 | 33.8 |
| canonicalize + mean 融合（免训练） | 23.1 | 31.6 |
| single front | 26.9 | 34.2 |
| single left | 31.9 | 45.0 |
| single right | 24.0 | 29.0 |

按 run（full 模型 PA）：vp11 19.8 / 21.8 · vp13 23.5 / 26.5 · vp5 26.4 / 24.1 mm。

要点（如实）：
- **零样本迁移成立**：未见过 Drive&Act 的模型在官方 test 受试者上 PA-MPJPE 23.7 mm，
  优于任一单视角（最佳 24.0 mm），与评审引用的全监督 SOTA 区间（22.6–30.4 mm）同量级。
- **学习模块相对免训练 canonicalize+mean 融合没有增益**（23.7 vs 23.1 mm）——与论文
  消融（uniform_gate 最优）及评审判断一致：canonicalization 是有效成分。
- **门控塌缩在 Drive&Act 上复现**：两个 ckpt 的逐视角 alpha 均值均为 0.3333/0.3333/0.3333。
  这是修复前的权重（熵正则零梯度），重训（λ_gate 修复版）完成后可用同一脚本
  约 10 分钟内重评。
- hs10 的 PA 低于 body14 的 PA（23 vs 31 mm）是因为 hs10 不含肘/髋这两类难关节，
  两表数字不可直接互比。
