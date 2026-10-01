# RCG-Fusion 实现说明

## 实现状态

三个模块均已实现为可独立测试、可端到端运行的代码：

| 模块 | 代码 | 训练产物 |
|---|---|---|
| 联盟有效任务骨干 | `rcg/rcg_fusion.py::CoalitionAwareBackbone` | `backbone.pt` |
| 多教师交叉拟合贡献监督 | `generate_oof_teachers`、`build_teacher_targets` | `oof_teacher_outputs.parquet`、`aggregated_targets.parquet`、`edge_targets.parquet`、`shapley_targets.parquet` |
| 关系感知风险预测器 | `CoalitionRiskPredictor`、`risk_objective` | `risk_predictor.pt` |
| Conformal 安全选择 | `fit_joint_conformal`、`ConformalSafeSelector` | `conformal.json`、`predictions.parquet` |

旧版 `rcg/contribution.py` 保留为先导负结果对照。新方法没有覆盖旧模型或历史运行。

每个方法种子的 `target_audit.json` 额外报告训练内与 OOF 损失差、贡献符号一致率、
删除贡献与 Shapley 的相关性和符号一致率，用于创新一的协议验收。

## 数据与默认配置

运行器统一支持：

- `mosi`：`data/mosi/prepared/*.npz`，文本/音频/视觉；
- `mosei`：`data/mosei/prepared/*.npz`，文本/音频/视觉；
- `cremad`：`data/cremad/pretrained/cremad_wav2vec2_r3d18.npz`，视觉/音频，演员级五折；
- `avmnist`：自动识别 `data/avmnist/prepared/*.npz` 或现有社区镜像
  `data/avmnist_hf/prepared/*.npz`，图像/音频。

正式运行默认使用五个方法种子、五个教师种子、五折 OOF、100 个任务训练 epoch 和 100 个风险训练 epoch。任务骨干与风险预测器均以 early stopping 提前结束。

主选择策略固定为：

```text
alpha = 0.10
margin = 0.01 nats
```

`--tune-policy` 只用于消融。它在 selection 上选择策略，最终 conformal 分位数仍只在 calibration 上拟合。测试标签不进入温度、模型、策略或区间拟合。

## 正式运行

```powershell
cd 'E:\papers\信息融合\rcg-study'

.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline `
  --dataset mosi --data data --output runs\rcg-fusion-mosi

.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline `
  --dataset mosei --data data --output runs\rcg-fusion-mosei --batch-size 128

.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline `
  --dataset cremad --data data --output runs\rcg-fusion-cremad

.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline `
  --dataset avmnist --data data --output runs\rcg-fusion-avmnist --batch-size 128
```

首次正式运行计算量较大：三模态数据每个外层数据折要训练 25 个 OOF 教师骨干，再训练 5 个正式骨干和风险预测器。中断后使用完全相同的命令可复用已经完整生成的 `oof_targets.npz` 和已完成方法种子。若协议改变，必须换新的输出目录。

## 快速验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline `
  --dataset mosi --data data --output runs\rcg-fusion-smoke `
  --seeds 11 --teacher-seeds 11 --folds 2 `
  --epochs 1 --risk-epochs 1 --batch-size 128
```

烟雾运行只验证接口、训练和产物，不能用于报告方法效果。

## 关键实现约束

1. OOF 样本不会进入生成其目标的教师训练集；MOSI/MOSEI 按视频、CREMA-D 按演员分组。
2. 空联盟不进入模型；仅用训练类别先验定义 Shapley 的空联盟价值。
3. 风险预测器测试接口只接收特征、联盟概率和掩码，不接收标签。
4. conformal nonconformity 对同一样本全部“相对完整融合的损失改进量”取最大标准化残差，
   形成贡献改进的联合区间。早期绝对风险区间实验过宽，因此正式实现校准相关误差能够抵消的风险差。
5. 只有候选联盟的改进下界超过安全边际时才切换，否则回退完整融合。若整模态缺失改变了
   回退联盟，当前实现保守地直接回退，因为干净 calibration 没有为新的参考联盟提供配对保证。
6. 当前精确枚举适用于二至三模态；没有声称解决模态数量较大时的指数复杂度。

## 结果判读

`metrics_by_seed.csv` 包含风险 MAE/Spearman、最优联盟 Top-1/2、贡献伤害 AUROC/AUPRC、完整与选择后悔、有害切换率、区间联合覆盖率、Accuracy、Macro-F1 和 NLL。

每个种子还生成 `stress_metrics.csv` 和 `stress_predictions.parquet`，覆盖单模态高斯噪声
`0.25/0.5/1/2`、特征遮蔽 `0.25/0.5/0.75`、三个固定扰动种子和整模态缺失。
缺失条件只允许选择仍可用模态组成的联盟，并以“全部仍可用模态”作为安全回退。

方法进入论文主结果前，仍应执行计划中的门槛：风险 Spearman 至少 0.70、伤害 AUROC 比最佳原生分数提高至少 5 个百分点、平均后悔相对最佳动态融合基线下降至少 20%，并且干净性能下降不超过 0.5 个百分点。代码完成不代表这些经验门槛已经通过。
