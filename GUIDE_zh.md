# 模型操作指南

## 1. 文件夹约定

| 文件夹 | 作用 | 是否会被脚本修改 |
|---|---|---|
| `06_Datasets_Raw_Adopted` | 采用的原始数据（下载原样） | 否，只读 |
| `02_Data_Processed` | 第 1–2 步导出的中间数据 | 是 |
| `04_Results_Analysis` | 所有结果表格和图，每步一个子文件夹 | 是 |
| `07_GitHub_Upload` | 代码（`python/`、`matlab/`），即上传 GitHub 的内容 | 否 |

所有路径都在 `python/paths.py` 里。项目换位置时只改其中的 `PROJECT` 一行。

## 2. 环境

```
conda create -n cat401 python=3.11
conda activate cat401
pip install -r requirements.txt
```

## 3. 运行顺序

在 `07_GitHub_Upload` 文件夹里依次运行（Anaconda Prompt）：

| 步骤 | 命令 | 作用 | 大致耗时 |
|---|---|---|---|
| 1 | `python python/01_export_netcdf_to_csv.py` | NetCDF 转 CSV（唯一需要 netCDF4 的一步） | 1–2 分钟 |
| 2 | `python python/02_rain_attenuation_exploration.py` | 天气状态、降雨事件、衰减、主/备配对 | 1–2 分钟 |
| 3 | `python python/03_measured_performance.py` | 各频段正常/降雨性能 | 1 分钟 |
| 4 | `python python/04_fallback_model.py` | 第一版备份模型（可跳过） | 30 秒 |
| 5 | `python python/05_validation_and_risk.py` | 易损性曲线与验证 | 3–5 分钟 |
| 6 | `python python/06_bivariate_im_risk.py` | 双指标灾害、年度风险 | 3–5 分钟 |
| 7 | `python python/07_real_pair_fallback.py` | 真实配对策略、恢复、敏感性 | 1 分钟 |
| 8 | `python python/08_physics_calibration.py` | 物理标定（需先跑第 5 步） | 1 分钟 |

第 2 步会生成约 250 MB 的 `link_rain_5min.pkl`，第 3–8 步都依赖它。不想放在项目里可以先设置
`set CAT401_CACHE=E:\cache` 再运行。

## 4. 常改的参数

| 想改什么 | 位置 |
|---|---|
| 晴天容量余量、剩余容量、检测延迟、回切等待 | `07_real_pair_fallback.py` 顶部 `BASE`、`S_GRID` |
| 剩余容量档位、损失模型 | `06_bivariate_im_risk.py` 顶部 `S_LIST`、`MODELS` |
| 训练/测试分割日期 | `05_validation_and_risk.py`、`08_physics_calibration.py` 中的 `SPLIT` |
| 噪声底（信号强度换算成信噪比） | 各脚本顶部 `NOISE` |
| 雨量站质控阈值 | `02_rain_attenuation_exploration.py` 中 `qc["keep"]` 一行 |

改完后从被修改的那一步往后重跑。

## 5. MATLAB 模拟器

```
cd('D:\CAT401\05_Term_Project\07_GitHub_Upload\matlab')
rainFadeSelfTest      % 应显示 all checks passed
RainFadeSimulator     % 打开界面
```

- 左侧“Preset”可直接选 NOAA Atlas 14 纽约 1/2/10/100 年一遇雨强；也可以手动拖动雨强、时长。
- “Parameter set”切换理想物理或 OpenMesh 标定参数（湿天线 1.35 dB、路径系数 1.06 / 0.68、干燥 29 分钟）。
- 右侧三个页面：时间响应（雨→衰减→容量→服务水平）、雨强扫描、模型说明。
- “Export CSV + PNG”导出当前情景。

## 6. 常见问题

- **找不到 `link_rain_5min.pkl`**：先跑第 2 步。
- **第 8 步找不到 `fragility_parameters.csv`**：先跑第 5 步。
- **`ModuleNotFoundError: netCDF4`**：`pip install netCDF4`（conda 安装可能卡在求解环境）。
- **MATLAB 提示函数未定义**：当前文件夹不是 `matlab/`，或版本低于 R2020b。
