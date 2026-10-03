# E13 规则选取案例表

所有案例由脚本中的冻结规则自动产生；完整逐样本内部量见Parquet源记录。

## 成功纠错：按CE收益降序

| 数据集 | 排名 | 样本 | 标签 | Base→Final | CE收益 | p(y) Base→Final | α | 路由Top-3 | 动作权重 |
|---|---:|---|---:|---|---:|---:|---:|---|---|
| MOSI | 1 | `etzxEpPuc6I$_$12` | 1 | 0→1 | +0.371 | 0.368→0.534 | 0.360 | text>text+audio>text+audio+vision | {"full":0.2023,"posterior":0.325,"anchor":0.1732,"support1":0.1571,"support2":0.1425} |
| MOSI | 2 | `pLTX3ipuDJI$_$4` | 0 | 1→0 | +0.345 | 0.366→0.516 | 0.114 | text+audio+vision>text+vision>text | {"full":0.2099,"posterior":0.3026,"anchor":0.1744,"support1":0.157,"support2":0.1561} |
| MOSI | 3 | `c7UH_rxdZv4$_$32` | 0 | 1→0 | +0.343 | 0.427→0.602 | 0.205 | vision>text+audio+vision>text+vision | {"full":0.2009,"posterior":0.374,"anchor":0.1518,"support1":0.1372,"support2":0.1361} |
| MOSI | 4 | `d6hH302o4v8$_$32` | 0 | 1→0 | +0.335 | 0.477→0.666 | 0.230 | text>text+vision>text+audio+vision | {"full":0.2057,"posterior":0.3481,"anchor":0.1582,"support1":0.1479,"support2":0.14} |
| MOSI | 5 | `c7UH_rxdZv4$_$1` | 0 | 1→0 | +0.323 | 0.445→0.615 | 0.295 | text+audio+vision>text+vision>text+audio | {"full":0.1947,"posterior":0.3057,"anchor":0.1731,"support1":0.1634,"support2":0.1632} |
| MOSEI | 1 | `28006$_$0` | 0 | 1→0 | +0.300 | 0.378→0.510 | 0.145 | vision>text+audio>text+audio+vision | {"full":0.0946,"posterior":0.4346,"anchor":0.1856,"support1":0.2158,"support2":0.0694} |
| MOSEI | 2 | `c5VEvmutmVg$_$15` | 0 | 1→0 | +0.238 | 0.464→0.588 | 0.171 | audio>audio+vision>vision | {"full":0.061,"posterior":0.4444,"anchor":0.1804,"support1":0.1648,"support2":0.1493} |
| MOSEI | 3 | `46615$_$9` | 0 | 1→0 | +0.146 | 0.451→0.522 | 0.233 | vision>audio>audio+vision | {"full":0.0657,"posterior":0.5269,"anchor":0.1272,"support1":0.1657,"support2":0.1146} |
| MOSEI | 4 | `ozA7pRW4gFM$_$9` | 1 | 0→1 | +0.138 | 0.481→0.552 | 0.212 | text>text+vision>text+audio | {"full":0.0994,"posterior":0.4908,"anchor":0.1388,"support1":0.1581,"support2":0.1129} |
| MOSEI | 5 | `mgsvwAVQAQo$_$0` | 0 | 1→0 | +0.133 | 0.464→0.530 | 0.311 | audio>audio+vision>text+audio | {"full":0.0824,"posterior":0.3875,"anchor":0.1788,"support1":0.1997,"support2":0.1515} |
| CREMA-D | 1 | `1044_ITH_HAP_XX` | 3 | 1→3 | +0.793 | 0.118→0.261 | 0.170 | visual+audio>visual>audio | {"full":0.1986,"posterior":0.5906,"anchor":0.1327,"support1":0.0781} |
| CREMA-D | 2 | `1036_MTI_FEA_XX` | 2 | 3→2 | +0.680 | 0.185→0.366 | 0.157 | visual+audio>visual>audio | {"full":0.1647,"posterior":0.7132,"anchor":0.0755,"support1":0.0466} |
| CREMA-D | 3 | `1044_IOM_HAP_XX` | 3 | 1→3 | +0.614 | 0.206→0.381 | 0.133 | audio>visual+audio>visual | {"full":0.2282,"posterior":0.5318,"anchor":0.1816,"support1":0.0584} |
| CREMA-D | 4 | `1044_WSI_HAP_XX` | 3 | 1→3 | +0.609 | 0.282→0.519 | 0.087 | visual+audio>audio>visual | {"full":0.2564,"posterior":0.633,"anchor":0.0761,"support1":0.0345} |
| CREMA-D | 5 | `1032_IWL_FEA_XX` | 2 | 3→2 | +0.601 | 0.225→0.410 | 0.281 | visual+audio>audio>visual | {"full":0.1405,"posterior":0.6965,"anchor":0.1143,"support1":0.0488} |
| AV-MNIST | 1 | `2328` | 4 | 9→4 | +1.624 | 0.147→0.743 | 0.372 | image+audio>image>audio | {"full":0.2706,"posterior":0.6731,"anchor":0.0542,"support1":0.0022} |
| AV-MNIST | 2 | `9918` | 4 | 7→4 | +1.284 | 0.241→0.871 | 0.175 | image+audio>image>audio | {"full":0.2766,"posterior":0.5148,"anchor":0.1937,"support1":0.0149} |
| AV-MNIST | 3 | `14872` | 6 | 0→6 | +1.039 | 0.316→0.892 | 0.682 | image>image+audio>audio | {"full":0.0141,"posterior":0.8923,"anchor":0.0906,"support1":0.0029} |
| AV-MNIST | 4 | `8588` | 3 | 8→3 | +1.013 | 0.288→0.793 | 0.364 | image+audio>image>audio | {"full":0.1301,"posterior":0.7696,"anchor":0.0992,"support1":0.001} |
| AV-MNIST | 5 | `16578` | 0 | 2→0 | +0.898 | 0.321→0.787 | 0.738 | image+audio>audio>image | {"full":0.1401,"posterior":0.81,"anchor":0.0466,"support1":0.0034} |

## 负向翻转：按CE损害降序

| 数据集 | 排名 | 样本 | 标签 | Base→Final | CE收益 | p(y) Base→Final | α | 路由Top-3 | 动作权重 |
|---|---:|---|---:|---|---:|---:|---:|---|---|
| MOSI | 1 | `vyB00TXsimI$_$3` | 1 | 1→0 | -0.469 | 0.538→0.336 | 0.119 | text+vision>text>text+audio+vision | {"full":0.2211,"posterior":0.2912,"anchor":0.1724,"support1":0.1639,"support2":0.1514} |
| MOSI | 2 | `yvsjCA6Y5Fc$_$11` | 1 | 1→0 | -0.449 | 0.589→0.376 | 0.085 | text>text+vision>text+audio+vision | {"full":0.2162,"posterior":0.2941,"anchor":0.1731,"support1":0.1597,"support2":0.1569} |
| MOSI | 3 | `d6hH302o4v8$_$20` | 1 | 1→0 | -0.339 | 0.523→0.372 | 0.166 | text>text+vision>text+audio+vision | {"full":0.2119,"posterior":0.2898,"anchor":0.1796,"support1":0.1688,"support2":0.1499} |
| MOSI | 4 | `vyB00TXsimI$_$17` | 1 | 1→0 | -0.291 | 0.507→0.379 | 0.234 | text+audio>text+audio+vision>text+vision | {"full":0.2193,"posterior":0.3159,"anchor":0.1774,"support1":0.1519,"support2":0.1355} |
| MOSI | 5 | `vyB00TXsimI$_$6` | 1 | 1→0 | -0.290 | 0.582→0.436 | 0.482 | text+audio+vision>audio>text+audio | {"full":0.2014,"posterior":0.3796,"anchor":0.1553,"support1":0.1395,"support2":0.1242} |
| MOSEI | 1 | `221153$_$5` | 1 | 1→0 | -0.280 | 0.652→0.493 | 0.160 | audio>vision>audio+vision | {"full":0.0677,"posterior":0.4588,"anchor":0.173,"support1":0.1914,"support2":0.1092} |
| MOSEI | 2 | `221153$_$13` | 1 | 1→0 | -0.220 | 0.524→0.420 | 0.276 | vision>text>text+audio | {"full":0.1042,"posterior":0.4327,"anchor":0.1871,"support1":0.1871,"support2":0.089} |
| MOSEI | 3 | `_q7DM8WkzAQ$_$9` | 0 | 0→1 | -0.193 | 0.532→0.439 | 0.194 | text>text+vision>text+audio | {"full":0.1064,"posterior":0.4897,"anchor":0.1547,"support1":0.1606,"support2":0.0887} |
| MOSEI | 4 | `111881$_$17` | 0 | 0→1 | -0.177 | 0.539→0.452 | 0.334 | audio+vision>audio>vision | {"full":0.0579,"posterior":0.4587,"anchor":0.1962,"support1":0.1925,"support2":0.0947} |
| MOSEI | 5 | `59673$_$7` | 0 | 0→1 | -0.169 | 0.544→0.460 | 0.227 | text+vision>text+audio+vision>vision | {"full":0.1099,"posterior":0.413,"anchor":0.1812,"support1":0.1677,"support2":0.1281} |
| CREMA-D | 1 | `1050_IWW_NEU_XX` | 4 | 4→3 | -1.027 | 0.367→0.131 | 0.187 | audio>visual+audio>visual | {"full":0.1604,"posterior":0.5845,"anchor":0.2324,"support1":0.0227} |
| CREMA-D | 2 | `1023_ITS_NEU_XX` | 4 | 4→2 | -0.804 | 0.319→0.143 | 0.155 | audio>visual+audio>visual | {"full":0.2379,"posterior":0.5773,"anchor":0.1226,"support1":0.0622} |
| CREMA-D | 3 | `1079_IEO_FEA_HI` | 2 | 2→0 | -0.690 | 0.494→0.248 | 0.111 | visual+audio>visual>audio | {"full":0.2859,"posterior":0.4944,"anchor":0.1332,"support1":0.0864} |
| CREMA-D | 4 | `1007_ITH_DIS_XX` | 1 | 1→3 | -0.678 | 0.309→0.157 | 0.184 | audio>visual+audio>visual | {"full":0.0533,"posterior":0.6806,"anchor":0.2562,"support1":0.0099} |
| CREMA-D | 5 | `1082_MTI_NEU_XX` | 4 | 4→0 | -0.663 | 0.462→0.238 | 0.378 | audio>visual+audio>visual | {"full":0.1109,"posterior":0.8032,"anchor":0.0767,"support1":0.0092} |
| AV-MNIST | 1 | `18120` | 2 | 2→7 | -2.022 | 0.401→0.053 | 0.400 | image+audio>image>audio | {"full":0.1447,"posterior":0.7156,"anchor":0.137,"support1":0.0027} |
| AV-MNIST | 2 | `8205` | 7 | 7→9 | -1.450 | 0.429→0.101 | 0.621 | image+audio>audio>image | {"full":0.1022,"posterior":0.8126,"anchor":0.0789,"support1":0.0063} |
| AV-MNIST | 3 | `1005` | 3 | 3→2 | -1.431 | 0.697→0.167 | 0.321 | image+audio>audio>image | {"full":0.1548,"posterior":0.7568,"anchor":0.0856,"support1":0.0029} |
| AV-MNIST | 4 | `11933` | 3 | 3→0 | -0.726 | 0.712→0.345 | 0.438 | image+audio>audio>image | {"full":0.2122,"posterior":0.7494,"anchor":0.0346,"support1":0.0038} |
| AV-MNIST | 5 | `15677` | 2 | 2→3 | -0.327 | 0.653→0.471 | 0.790 | image>image+audio>audio | {"full":0.1001,"posterior":0.8117,"anchor":0.0863,"support1":0.0019} |

## 高可靠负贡献：按损害幅度降序

| 数据集 | 排名 | 样本 | 模态 | 可靠性/阈值 | 条件贡献 | α | 路由Top-3 | 联盟真实类概率 |
|---|---:|---|---|---:|---:|---:|---|---|
| MOSI | 1 | `nbWiPyCm4g0$_$1` | vision | 0.646/0.580 | -0.340 | 0.381 | vision>text+vision>text+audio+vision | text:0.650; audio:0.554; vision:0.354; text+audio:0.669; text+vision:0.446; audio+vision:0.306; text+audio+vision:0.476 |
| MOSI | 2 | `cM3Yna7AavY$_$11` | vision | 0.656/0.580 | -0.188 | 0.077 | text+vision>text+audio+vision>text | text:0.880; audio:0.387; vision:0.344; text+audio:0.883; text+vision:0.734; audio+vision:0.269; text+audio+vision:0.732 |
| MOSI | 3 | `nzpVDcQ0ywM$_$15` | audio | 0.692/0.655 | -0.139 | 0.234 | text+audio+vision>text+vision>text | text:0.177; audio:0.308; vision:0.591; text+audio:0.149; text+vision:0.244; audio+vision:0.579; text+audio+vision:0.212 |
| MOSI | 4 | `etzxEpPuc6I$_$13` | vision | 0.610/0.580 | -0.130 | 0.142 | text+vision>text+audio+vision>text | text:0.475; audio:0.486; vision:0.390; text+audio:0.461; text+vision:0.400; audio+vision:0.361; text+audio+vision:0.405 |
| MOSI | 5 | `nzpVDcQ0ywM$_$19` | audio | 0.719/0.655 | -0.130 | 0.142 | text+audio+vision>text+audio>text+vision | text:0.144; audio:0.281; vision:0.534; text+audio:0.120; text+vision:0.161; audio+vision:0.481; text+audio+vision:0.142 |
| MOSEI | 1 | `252097$_$10` | text | 0.989/0.983 | -3.609 | 0.000 | text+audio>text>text+vision | text:0.011; audio:0.481; vision:0.699; text+audio:0.013; text+vision:0.015; audio+vision:0.569; text+audio+vision:0.015 |
| MOSEI | 2 | `1zXAYdPdzy8$_$2` | text | 0.990/0.983 | -3.184 | 0.000 | text>text+audio>text+audio+vision | text:0.010; audio:0.330; vision:0.382; text+audio:0.011; text+vision:0.013; audio+vision:0.325; text+audio+vision:0.013 |
| MOSEI | 3 | `cml9rShionM$_$3` | text | 0.987/0.983 | -3.052 | 0.197 | text+audio>text>text+audio+vision | text:0.013; audio:0.391; vision:0.557; text+audio:0.016; text+vision:0.019; audio+vision:0.417; text+audio+vision:0.020 |
| MOSEI | 4 | `92291$_$3` | text | 0.984/0.983 | -3.017 | 0.363 | text>text+audio>text+vision | text:0.016; audio:0.648; vision:0.850; text+audio:0.022; text+vision:0.031; audio+vision:0.769; text+audio+vision:0.038 |
| MOSEI | 5 | `ddWHTdJz2O8$_$1` | vision | 0.950/0.807 | -0.533 | 0.555 | text+vision>text+audio+vision>vision | text:0.160; audio:0.313; vision:0.050; text+audio:0.132; text+vision:0.075; audio+vision:0.136; text+audio+vision:0.077 |
| CREMA-D | 1 | `1043_DFA_FEA_XX` | visual | 0.916/0.770 | -3.739 | 0.044 | visual+audio>visual>audio | visual:0.010; audio:0.701; visual+audio:0.017 |
| CREMA-D | 2 | `1006_IEO_ANG_HI` | visual | 0.832/0.770 | -3.427 | 0.034 | visual+audio>visual>audio | visual:0.018; audio:0.847; visual+audio:0.028 |
| CREMA-D | 3 | `1043_TSI_FEA_XX` | visual | 0.907/0.770 | -3.351 | 0.123 | visual+audio>audio>visual | visual:0.012; audio:0.291; visual+audio:0.010 |
| CREMA-D | 4 | `1083_IWL_HAP_XX` | visual | 0.849/0.770 | -3.273 | 0.029 | visual+audio>visual>audio | visual:0.014; audio:0.272; visual+audio:0.010 |
| CREMA-D | 5 | `1083_WSI_HAP_XX` | visual | 0.874/0.770 | -3.191 | 0.093 | visual+audio>visual>audio | visual:0.013; audio:0.241; visual+audio:0.010 |

## AV-MNIST完整联盟保留：先按alpha升序

| 数据集 | 排名 | 样本 | 标签 | Base→Final | CE收益 | p(y) Base→Final | α | 路由Top-3 | 动作权重 |
|---|---:|---|---:|---|---:|---:|---:|---|---|
| AV-MNIST | 1 | `21674` | 6 | 6→6 | +0.000 | 1.000→1.000 | 0.000 | image+audio>audio>image | {"full":0.2896,"posterior":0.3054,"anchor":0.245,"support1":0.1599} |
| AV-MNIST | 2 | `6763` | 5 | 5→5 | +0.000 | 1.000→1.000 | 0.000 | image+audio>image>audio | {"full":0.3,"posterior":0.29,"anchor":0.2582,"support1":0.1518} |
| AV-MNIST | 3 | `2791` | 7 | 7→7 | +0.000 | 1.000→1.000 | 0.000 | image+audio>image>audio | {"full":0.2821,"posterior":0.29,"anchor":0.2571,"support1":0.1708} |
| AV-MNIST | 4 | `4281` | 7 | 7→7 | +0.000 | 1.000→1.000 | 0.000 | image+audio>image>audio | {"full":0.3209,"posterior":0.3158,"anchor":0.2734,"support1":0.0899} |
| AV-MNIST | 5 | `2704` | 7 | 7→7 | +0.000 | 1.000→1.000 | 0.000 | image+audio>image>audio | {"full":0.3112,"posterior":0.3061,"anchor":0.2342,"support1":0.1485} |
