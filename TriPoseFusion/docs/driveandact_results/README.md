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
