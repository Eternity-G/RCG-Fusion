# Reliability–Contribution Gap：冻结特征观测实验

实现目标：在 MOSI/MOSEI 上比较**单模态可靠性**与**固定模型条件删除贡献**，控制校准、缺失输入与重复采样伪象。项目不把这些实验称为现实因果归因，也不预先保证发现成立。

## RCG-Fusion 方法实现

观测实验之后的完整方法已实现为“多教师交叉拟合贡献监督 → 关系感知联盟风险预测 → conformal 安全回退”流水线。正式命令、数据接口、产物和验收门槛见 [RCG_FUSION_IMPLEMENTATION.md](RCG_FUSION_IMPLEMENTATION.md)。核心代码位于 `rcg/rcg_fusion.py`，端到端入口为：

```powershell
.\.venv\Scripts\python.exe -m rcg.rcg_fusion_pipeline --dataset mosi --data data --output runs\rcg-fusion-mosi
```

MOSI 五种子首轮方法验收没有达到预注册门槛；这部分结果不得作为正向论文结果。
实际指标、失败诊断和下一步判断见 [RCG_FUSION_IMPLEMENTATION_REPORT.md](RCG_FUSION_IMPLEMENTATION_REPORT.md)。

修订后的“后验结构化条件收益 + Conformal Risk Control”也已独立实现并完成 MOSI
五种子验收。它解决了零切换问题，但候选安全 AUROC、总体有害切换和后悔下降没有
达到门槛。代码入口、正式结果和停止理由见
[BENEFIT_CRC_IMPLEMENTATION_REPORT.md](BENEFIT_CRC_IMPLEMENTATION_REPORT.md)。

第二次修订将不稳定的独立收益头替换为“关系标签后验 + 解析贡献积分”。该版本只学习
`P(Y|X)`，再由标签条件对数概率比严格计算期望收益、有益概率、期望增益和损害：

```powershell
.\.venv\Scripts\python.exe -m rcg.posterior_analytic_pipeline `
  --dataset mosi --data data --base-run runs\rcg-fusion-mosi-v4 `
  --output runs\rcg-posterior-analytic-mosi-v2
```

MOSI 已参与方案诊断，因此该入口的 MOSI 结果属于开发证据；方法冻结后的 MOSEI 和
CREMA-D 才能承担无偏确认。流水线仅在创新二四项数值门槛全部通过时允许 `--run-crc`。
当前五种子主配置通过其中三项，Top-20% 安全精确率未达标，因此已停止在 MOSI；完整
结果与消融见 [POSTERIOR_ANALYTIC_IMPLEMENTATION_REPORT.md](POSTERIOR_ANALYTIC_IMPLEMENTATION_REPORT.md)。

创新三的“top-3联盟后验凸包投影 + 完整融合收缩 + CRC”也已独立实现：

```powershell
.\.venv\Scripts\python.exe -m rcg.projected_fusion_pipeline `
  --dataset mosi --data data --base-run runs\rcg-fusion-mosi-v4 `
  --posterior-run runs\rcg-posterior-analytic-mosi-v2b `
  --output runs\rcg-projected-fusion-mosi-v1
```

MOSI阶段A没有达到覆盖率、后悔下降和相对硬切换改善门槛，因此没有继续运行无偏数据集
或完整主实验。结果与精确凸包敏感性核查见
[PROJECTED_FUSION_IMPLEMENTATION_REPORT.md](PROJECTED_FUSION_IMPLEMENTATION_REPORT.md)。

针对投影融合收益过小的问题，创新三最终开发路径改为贡献概率自适应的有界后验收缩：

```powershell
.\.venv\Scripts\python.exe -m rcg.posterior_shrinkage_pipeline `
  --dataset mosi --data data --base-run runs\rcg-fusion-mosi-v4 `
  --posterior-run runs\rcg-posterior-analytic-mosi-v2 `
  --output runs\rcg-posterior-shrinkage-mosi-v3 `
  --alpha .75 --adaptive-gamma 2
```

该策略使用 `alpha(x)=0.75*pi(x)^2`，其中 `pi(x)` 是解析动作有益概率。MOSI 五种子
Accuracy 提高0.91个百分点，NLL下降0.00729；MOSEI分别提高0.23个百分点和下降
0.00400。两者配对簇CI均排除零，平均截断损害分别为1.44%和1.06%。实现、理论边界
和协议污染说明见
[POSTERIOR_SHRINKAGE_IMPLEMENTATION_REPORT.md](POSTERIOR_SHRINKAGE_IMPLEMENTATION_REPORT.md)。

四数据集正式扩展现已完成 CREMA-D 演员级五折和 AV-MNIST。虽然四个数据集的 NLL
均获得配对改善，固定插值 Pareto 对照显示 `alpha=0.75*pi^2` 只在 MOSI 上具有明确
差异化优势，当前第三创新尚未通过跨数据集验收。完整数值、PNG 图和下一版门控方向见
[MAIN_EXPERIMENT_PROGRESS.md](MAIN_EXPERIMENT_PROGRESS.md)。在第三创新修订前，项目暂停
扩大鲁棒性和原生骨干实验，避免围绕未成立的策略堆叠结果。

面向10%相对提升的重构已实现第一阶段：OOF软联盟分布、成对偏序、稳定改善标签以及
解析先验加列表式残差路由器。MOSI五种子显示 Top-2 候选覆盖提高，但 Top-1 与成对
排序没有通过门槛；五教师 oracle 完全一致率仅11.54%。代码、命令和停止判断见
[LISTWISE_ROUTER_IMPLEMENTATION_REPORT.md](LISTWISE_ROUTER_IMPLEMENTATION_REPORT.md)。

## 完整强基线观测流水线

当前扩展实现了全量 CREMA-D 与统一强基线协议：

```powershell
# 可断点续跑的 7,442 条 CREMA-D 冻结表示
.\.venv\Scripts\python.exe scripts\extract_cremad_pretrained.py

# 默认五种子；统一运行 concat、TMC、QMF、PDF、I²MoE
.\.venv\Scripts\python.exe -m rcg.strong_observation --dataset mosi  --data data --output runs\strong-observation
.\.venv\Scripts\python.exe -m rcg.strong_observation --dataset mosei --data data --output runs\strong-observation
.\.venv\Scripts\python.exe -m rcg.strong_observation --dataset cremad --data data\cremad\pretrained --output runs\strong-observation

# 10,000 次簇 bootstrap、四张表和仅 PNG 的 300 dpi 图片
.\.venv\Scripts\python.exe -m rcg.observation_analysis --root runs\strong-observation --output runs\observation-final\analysis
.\.venv\Scripts\python.exe scripts\build_observation_artifacts.py --strong-root runs\strong-observation --analysis runs\observation-final\analysis --output runs\observation-final
```

每次运行同时保存紧凑的 `samples.parquet` 和满足实验产物接口的长表 `coalitions.parquet`。执行状态见 [OBSERVATION_EXPERIMENT_CHECKLIST.md](OBSERVATION_EXPERIMENT_CHECKLIST.md)，统一适配与原方法的边界见 [BASELINE_ADAPTATION_AUDIT.md](BASELINE_ADAPTATION_AUDIT.md)。

## 环境与运行

Windows / Python 3.11 / CUDA 12.4 / PyTorch 2.6.0。选择 CUDA 12.4 是为了兼容检查到的 NVIDIA 560.81 驱动。其余直接依赖固定在 pyproject.toml；实际完整环境导出为 requirements-resolved.txt。

```powershell
cd 'E:\papers\信息融合\rcg-study'
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m pytest -q

# Full-anchored listwise candidate mixer and stable posterior aggregation
.\.venv\Scripts\python.exe -m rcg.anchored_mixer_pipeline --dataset mosi --data data --base-run runs\rcg-fusion-mosi-v4 --posterior-run runs\rcg-posterior-analytic-mosi-v2 --router-run runs\rcg-listwise-mosi-v1 --output runs\rcg-anchored-mixer-mosi-v3
.\.venv\Scripts\python.exe -m rcg.stable_ensemble_pipeline --dataset mosi --data data --base-run runs\rcg-fusion-mosi-v4 --posterior-run runs\rcg-posterior-analytic-mosi-v2 --output runs\rcg-stable-ensemble-mosi-v2
.\.venv\Scripts\python.exe -m rcg all --dataset mosi
.\.venv\Scripts\python.exe -m rcg all --dataset mosei
```

也可以分阶段运行（同一解释器）：

```powershell
.\.venv\Scripts\python.exe -m rcg download --dataset mosi
.\.venv\Scripts\python.exe -m rcg prepare --dataset mosi
.\.venv\Scripts\python.exe -m rcg run --dataset mosi
.\.venv\Scripts\python.exe -m rcg analyze --dataset mosi
.\.venv\Scripts\python.exe -m rcg plot --dataset mosi
```

先导运行使用独立输出目录，避免和正式实验混淆：

```powershell
.\.venv\Scripts\python.exe -m rcg run --dataset mosi --seeds 11 --run runs/mosi-pilot
.\.venv\Scripts\python.exe -m rcg analyze --dataset mosi --run runs/mosi-pilot --bootstrap 10000
.\.venv\Scripts\python.exe -m rcg plot --dataset mosi --run runs/mosi-pilot
```

断点恢复：使用相同配置和代码重复 run，已完成种子跳过，已训练模型复用；只有完整写出的 evaluation_complete.json 才视为评估完成。配置、数据审计或源码改变后必须新建 --run 目录。下载采用 8 MiB 分块、4 个工作线程、失败重试，读取 pickle 前强制校验 MMSA 官方 SHA-256。下载缓存会额外占磁盘；建议预留 25 GB。

CUDA 大文件下载不稳定时，可以运行 `python scripts/fetch_torch.py`，它从官方 PyTorch 域名分块获取固定 wheel 并核验官方哈希，然后用 uv/pip 安装 `.wheels/torch-2.6.0+cu124-cp311-cp311-win_amd64.whl`。该备用脚本仅适用于 Windows CPython 3.11 x64。

## 冻结的实验协议

- 输入：MMSA `aligned_50.pkl` 中浮点 `text`（BERT 表示）、`audio`、`vision`；`text_bert` 仅用作有效时间步的元数据，不是模型特征。
- 删除中性标签；沿有效时间步均值池化。所核验文件的音视频在 BERT [CLS]/[SEP] 位置是零占位，三模态一致排除首尾特殊位置和 padding。非有限帧排除并审计；无有效帧输出零并报告数量。
- 仅训练集拟合逐维均值／标准差；零方差使用 1。原 train/valid/test 不变；检查样本 ID 和原视频无交集。
- 原 valid 按原视频以固定 split_seed=20260927 对半分成 selection/calibration；selection 早停，calibration 拟合各单模态温度及分数的 90% 分位阈值。测试数据不决定阈值或训练参数。
- MLP：输入→128→ReLU→Dropout(.2)→2；AdamW(lr=.001, weight_decay=.01)，batch=64，最多100轮、早停10轮，以干净 selection NLL 选择模型。
- 模型：三个单模态头、等权概率平均、置信度加权、温度校准后置信度加权、带缺失训练的拼接模型、仅干净输入训练的拼接对照、TMC 统一特征适配。拼接用原始标准化池化特征加3个存在标记。缺失训练50%完整，50%均匀取六种非空不完整组合。
- TMC：Softplus 非负证据、alpha=evidence+1、DS 组合、各模态和融合的 digamma CE + KL。KL 系数10个epoch线性增长到1。DS 数学与作者实现相同；小型预测头容量统一是本项目适配，不能与原论文表格直接比较。
- 删除：晚期融合剩余权重重新归一化；TMC 仅组合剩余证据；拼接清零对应输入并更新存在标记。未经缺失训练的 concat_clean 只作伪象对照。
- 种子11/22/33/44/55；噪声种子101/202/303。高斯SD .25/.5/1/2，特征遮蔽.25/.5/.75；每次仅扰动一种模态。另3种完全缺失，以及固定目标、其余两个模态共同受高斯扰动的36种条件，总计103种。扰动在池化并标准化后的特征空间进行。
- 同一随机场跨严重度、方法、训练种子复用。完整／删除输入共享其他模态的相同扰动。缺失目标不计贡献。每个存在模态都保存 full 与 without 概率。
- 可靠性：原始max-softmax、温度校准max-softmax、归一化负熵和TMC原生1−K/sum(alpha)。二分类负熵和max-softmax排序等价，不计为独立发现。
- 拼接模型的可靠性来自相应单模态探针，不是它的原生权重；同时报告 raw 与 calibrated_probe 两种诊断，不修改拼接预测本身。

## 统计与判读

贡献 `u = CE(without,y) - CE(full,y)`。HCR = `P(u < -epsilon | r >= clean_calibration_q90)`，主 epsilon=.01 nats，敏感性0/.05。高置信平均损害为高置信子集中 mean(max(-u,0))。AUROC/AUPRC用1-r检测有害，单类别时为NA。

每种模态／方法／条件分别计算 Spearman、HCR、损害、负向翻转、AUROC/AUPRC、覆盖率；任务指标单独去除删除模态维度的重复行。ECE为10个等宽置信度区间，CSV保留计数；Brier定义为两类平方误差之和。

五个训练种子各自先汇总三次扰动，报告均值与样本标准差。主容差的CI另以全部已训练种子与扰动的 pooled 条件比率为估计量，对原视频做10000次配对簇bootstrap：一段视频的所有片段／种子／扰动／模态一起重采样。CI固定已训练种子，不冒充训练过程总体不确定性。重采样权重跨比较共享。采用两个预先指定的检验族：校准late−原始late HCR、各方法stress−clean HCR；分别做近似零中心bootstrap双侧检验与Holm校正。

不足30个不同高置信测试样本的条件只描述：实现采用更保守的**任一训练种子不足30即标记**。不依靠其他种子或扰动次数补足独立样本。CI仍供描述，禁止门槛判断／显著性标记。

项目门槛：校准后HCR区间下界>5%，同时有稳定损害或负向翻转，并在概率与学习式融合两种机制重复，再考虑继续。脚本仅标记HCR这一分项，不自动宣判研究假设成立。上下文实验逐样本检查目标分数恒定，统计clean正贡献与stress负贡献之间的双向变号（均超出.01）。

## 输出

- `data/<dataset>/prepared/audit.json`：实际数量、类别、视频、有效帧、维度、哈希及验证划分。
- `runs/<dataset>/manifest.json`：环境、完整配置、数据审计、源码哈希。
- `seed_*/checkpoint.pt`、`training.json`、`thresholds.json`：模型、早停、温度、固定阈值。
- `seed_*/*.parquet`：逐样本预测、可靠性、删除损失、视频ID及种子。
- `analysis/diagnostics_by_seed.csv`／`diagnostics_summary.csv`：全部条件与容差。
- `analysis/cluster_intervals.csv`：主容差的视频簇CI，独立样本计数与描述标志。
- `analysis/paired_*_comparisons.csv`：直接差值CI及校正后的检验。
- `analysis/task_metrics_*`：Accuracy、Macro-F1、NLL、Brier、ECE及单模态对照。
- `analysis/context_*`：全样本上下文变号率与轨迹，未按期望结果筛选总体。
- `figures/`：四类图（HCR两种扰动各一张），CSV源数据、SVG/PDF/TIFF/PNG、图注与QA说明。
- `REPORT.md`：自动生成的实际结果与解释限制。

## 文献与来源

- [MMSA数据及哈希](https://github.com/thuiar/MMSA#2-datasets)。镜像内容必须精确匹配官方哈希；数据权益仍归原作者。
- [TMC作者代码](https://github.com/Han-Zongbo/TMC)，[论文](https://arxiv.org/abs/2102.02051)。本项目独立实现其公开数学公式，并使用统一预测头。
- [SHAPE，IJCAI2022](https://www.ijcai.org/proceedings/2022/0425.pdf)：已有模态贡献研究，不声称首次提出模态归因。
- [QMF，ICML2023](https://proceedings.mlr.press/v202/zhang23ar.html)、[PDF，ICML2024](https://proceedings.mlr.press/v235/cao24c.html)、[AEMRL，IJCAI2026](https://www.ijcai.org/proceedings/2026/128)：后续论文级扩展，本轮未实现。

暂不训练contribution gate、重新抽特征、端到端编码器或构造按测试标签选取的冲突供体。单元测试的合成数据仅用于软件验收，不进入研究图表或报告。

## 联盟一致贡献扩展

第一轮观测完成后，可复用 `concat_masked` 检查全部七个非空模态联盟。该模型的训练采样已经覆盖完整输入、三个单模态及三个双模态组合；空联盟使用训练集类别先验，仅作为精确 Shapley 基准。原始运行保持不变，扩展结果写入独立目录：

```powershell
.\.venv\Scripts\python.exe -m rcg coalition-all --dataset mosi --base-run runs/mosi --run runs/mosi-coalition
.\.venv\Scripts\python.exe -m rcg coalition-all --dataset mosei --base-run runs/mosei --run runs/mosei-coalition
```

逐样本 `coalitions.parquet` 保存八个联盟的概率和损失、完整上下文删除贡献、精确 Shapley 贡献、oracle 联盟及完整融合后悔值。`analysis/clean_cluster_intervals.csv` 使用原视频簇 bootstrap 报告干净测试集的后悔值和高可靠有害率；`analysis/context_switches.csv` 检验目标模态可靠性不变时贡献是否变号。Oracle 和测试标签只用于离线评价，不构成可部署选择器。

## 关系贡献预测器先导

`contribution` 命令实现按原视频的五折交叉拟合联盟监督、关系联盟损失预测和校准集保守选择。该入口当前用于方法筛选；单种子先导没有降低测试后悔值，不能作为正向论文结果：

```powershell
.\.venv\Scripts\python.exe -m rcg contribution --dataset mosi --base-run runs/mosi --run runs/mosi-contribution-pilot-v4 --seeds 11
.\.venv\Scripts\python.exe -m rcg contribution --dataset mosei --base-run runs/mosei --run runs/mosei-contribution-pilot-v2 --seeds 11
```

完整负结果与停止条件见 `METHOD_PILOT_REPORT.md`。

## 独立任务先导

AV-MNIST 使用 `pranavmr/AV-MNIST` 社区镜像的 24,000 条记录，固定分层划分，冻结的 28×28 灰度像素与 130 维音频频谱摘要。该先导用于检验现象能否跨出语言情感任务，不与 MultiBench 官方划分或端到端方法直接比较：

```powershell
.\.venv\Scripts\python.exe scripts\download_avmnist.py
.\.venv\Scripts\python.exe scripts\avmnist_pilot.py all
```

CREMA-D 从官方项目的 GitLab 镜像获取 LFS 视频。划分按演员互斥，标签取文件名中的表演意图情绪。完整数据可直接运行；本轮较快的预注册先导固定使用五个句子代码，共 3,633 条样本：

```powershell
cd data\cremad\repository
git lfs fetch gitlab master --include="VideoFlash/*_DFA_*.flv,VideoFlash/*_IEO_*.flv,VideoFlash/*_IOM_*.flv,VideoFlash/*_ITH_*.flv,VideoFlash/*_ITS_*.flv" --exclude=""
git lfs checkout
cd ..\..\..
.\.venv\Scripts\python.exe scripts\cremad_pilot.py all --sentences DFA IEO IOM ITH ITS
```

视觉输入是一秒处的单张 32×32 灰度帧，音频输入是 130 维频谱摘要。该配置只承担跨任务机制筛查；若现象成立，论文实验仍需完整语料、预训练音视频编码器和公开协议复验。

五种子先导已经完成。AV-MNIST 的高可靠负 Shapley 贡献低于 0.4%，而 CREMA-D 音频为 19.91%，演员簇 95% CI 为 [5.03%, 30.55%]；后者处在 5% 预注册门槛边界，不能作为稳健通过。完整判断见 `CROSS_TASK_GO_NO_GO.md`，CREMA-D 逐种子记录和校准参数见 `runs/cremad-pilot/`。
