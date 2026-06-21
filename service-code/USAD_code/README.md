# USAD 算法复现与多论文对比实验

本目录包含软件测试与维护大作业中 **AIOps 异常检测** 部分的源码，以 KDD 2020 论文 **USAD**（*UnSupervised Anomaly Detection on Multivariate Time Series*）为主复现算法，并在同一套 Chaos Mesh 故障注入数据上，分别与 KDD 2019 **OmniAnomaly**、AAAI 2021 **RAMED** 进行横向对比实验。

实验数据来源：在 Kubernetes 集群中对 `paymentservice` 注入故障期间，由 Prometheus + Grafana 采集并导出的 CPU、内存时序指标（CSV 格式）。

---

## 项目结构

```
USAD_code/
├── README.md              # 本说明文件
├── requirements.txt       # Python 依赖列表
├── run_usad.py            # 【主脚本】USAD 算法复现与评估
├── compare1.py            # USAD 与 OmniAnomaly 对比实验
├── ramed1.py              # USAD 与 RAMED 对比实验
├── run1.py                # USAD 探索版（训练分阶段 + 故障演进分析）
├── cpu.csv                # 实验用 CPU 利用率时序数据
├── memory.csv             # 实验用内存占用时序数据
└── results/               # （可选）实验结果截图，用于报告与汇报
    └── *.png
```

### 文件说明

| 文件 | 用途 |
|:---|:---|
| `run_usad.py` | **主复现脚本**。完成数据预处理 → USAD 双通道对抗自编码器训练 → 异常得分计算 → 分位数阈值网格搜索 → 输出 Precision / Recall / F1 及故障事件明细。 |
| `compare1.py` | **OmniAnomaly 对比脚本**。在相同预处理与相同 Ground Truth 标签下，并行训练 USAD 与 OmniAnomaly（GRU + VAE），输出两者在不同分位数阈值下的指标对比表及训练耗时。 |
| `ramed1.py` | **RAMED 对比脚本**。在相同数据上训练 USAD 与 RAMED（多分辨率 Fine/Coarse 双解码器），对比 F1、召回率与计算效率。 |
| `run1.py` | **探索版脚本**。将 USAD 训练拆为「基础重构 + 对抗博弈」两阶段，并输出故障突变期 / 扩散期 / 环境误报的三态划分日志，供算法机制验证使用。 |
| `cpu.csv` | Prometheus 导出的 `paymentservice` **CPU 利用率**时序，与 `memory.csv` 按时间戳对齐后作为模型输入。 |
| `memory.csv` | Prometheus 导出的 `paymentservice` **内存占用**时序。 |
| `requirements.txt` | 运行上述脚本所需的 Python 第三方库。 |
| `results/` | 存放实验运行截图（可选），便于报告引用。 |

> **说明：** 三个主脚本（`run_usad.py`、`compare1.py`、`ramed1.py`）均从**当前目录**读取 `cpu.csv` 与 `memory.csv`，运行前请确保数据文件与脚本位于同一目录。

---

## 环境准备

请先安装 Python 3.8 及以上版本，然后安装依赖：

```bash
pip install -r requirements.txt
```

依赖包括：`pandas`、`numpy`、`torch`、`scikit-learn`。

---

## 运行方式

进入本目录后，按需执行：

```bash
# 1. USAD 主算法复现（KDD'20）
python run_usad.py

# 2. USAD vs OmniAnomaly 对比（KDD'19）
python compare1.py

# 3. USAD vs RAMED 对比（AAAI'21）
python ramed1.py

# 4. （可选）USAD 探索版，含故障演进三态分析
python run1.py
```

---

## 数据与评估说明

- **输入特征：** CPU + 内存 2 维 KPI，滑动窗口长度 `K=5`，展平为 10 维向量。
- **预处理：** `MinMaxScaler` 归一化至 `[0, 1]`。
- **Ground Truth：** 2026-06-14 **12:56:00 ~ 12:58:30** 为 Chaos Mesh 故障注入核心时段，共 6 条异常窗口；标签**仅用于评估 F1**，不参与模型训练。
- **预期输出：** 控制台打印训练日志、最优阈值、Precision / Recall / F1-Score，以及检出的故障时间点明细。

---

## 参考论文

| 算法 | 论文 |
|:---|:---|
| USAD | Audibert et al., *USAD: UnSupervised Anomaly Detection on Multivariate Time Series*, KDD 2020 |
| OmniAnomaly | Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*, KDD 2019 |
| RAMED | Shen et al., *Time Series Anomaly Detection with Multiresolution Ensemble Decoding*, AAAI 2021 |
