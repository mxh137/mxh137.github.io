**项目分析与元数据提取：**

*   **核心主题：** PICU 患者数据分析、数据可视化与机器学习模型评估。
*   **项目英文名：** `picu-data-analysis`
*   **标题：** PICU 患者数据分析与预测模型评估
*   **摘要：** 本项目旨在对重症监护室 (PICU) 患者数据进行深度分析，包括数据清洗、特征工程、关键指标可视化，并构建机器学习模型以评估患者的预后。项目内容涵盖数据探索、可视化图表生成（如年龄分布、性别比例、实验室指标箱线图），以及模型性能评估（混淆矩阵、ROC 曲线）。
*   **标签：** `数据分析`, `Python`, `Jupyter Notebook`, `机器学习`, `可视化`, `PICU`, `医疗数据`

**图片资源规划：**

根据代码输出和指南中的示例，我为您规划以下图片资源。请您回到您的 Jupyter Notebook，在运行完相关代码后，手动保存以下图片，并上传到指定文件夹。

**图片资源清单**

请在您的网站项目中创建文件夹：`images/portfolio/picu-data-analysis/`

请回到您的 Jupyter Notebook，手动保存以下位置的图片，并重命名为指定文件名，上传到该文件夹：

| 原图位置 (Notebook 章节/代码块) | 建议文件名 | 用途 |
| :--- | :--- | :--- |
| 性别比例甜甜圈图 | `gender_is_male_donut.png` | 正文插图 |
| 年龄分布直方图 | `age_month_hist.png` | 正文插图 |
| 住院死亡标志甜甜圈图 | `hospital_expire_flag_donut.png` | 正文插图 |
| 混淆矩阵 | `confusion_matrix.png` | 模型评估 |
| ROC 曲线 | `roc_curve.png` | 模型评估 |
| （如果代码中有箱线图） | `lab_indicators_boxplot.png` | 正文插图 |

**Markdown 内容生成：**

以下是根据您的代码和指南要求生成的 Markdown 文件内容。

```markdown
---
title: "PICU 患者数据分析与预测模型评估"
collection: portfolio
excerpt: "本项目旨在对重症监护室 (PICU) 患者数据进行深度分析，包括数据清洗、特征工程、关键指标可视化，并构建机器学习模型以评估患者的预后。项目内容涵盖数据探索、可视化图表生成（如年龄分布、性别比例、实验室指标箱线图），以及模型性能评估（混淆矩阵、ROC 曲线）。"
date: 2026-01-12
tags:
  - 数据分析
  - Python
  - Jupyter Notebook
  - 机器学习
  - 可视化
  - PICU
  - 医疗数据
---

## 项目概述

本项目专注于对儿科重症监护室 (PICU) 患者的数据进行深入探索性数据分析 (EDA)、数据清洗，并构建和评估预测模型。通过对患者的生理指标、实验室数据以及临床结果进行分析，旨在提高对 PICU 患者预后的理解和预测能力。

## 数据导入与初步探索

项目首先导入必要的Python库，并加载原始数据集。对数据集的初步检查包括查看数据维度、前几行数据以及整体信息摘要。

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# 加载数据（假设数据路径已定义）
# path = "your_data_path.csv"
# df = pd.read_csv(path)
# print(df.shape)
# print(df.head())
```

## 数据清洗与预处理

数据清洗是任何数据分析项目的关键步骤。本项目着重处理缺失值和异常值，并进行特征工程。

```python
# 示例：缺失值和无穷值检查
# assert na_total == 0, f"Still has NaN: {na_total}"
# assert inf_total == 0, f"Still has inf: {inf_total}"
# print("✅ No missing values and no infinities.")

# 示例：保存清洗后的数据
# out_path = path.replace(".csv", f"_clean_drop{col_missing_thresh}.csv")
# df_imputed.to_csv(out_path, index=False)
# print("Saved to:", out_path)
```

## 数据可视化

为了更好地理解数据分布和特征之间的关系，我们进行了多项数据可视化。

### 性别比例分布

![性别比例甜甜圈图](/images/portfolio/picu-data-analysis/gender_is_male_donut.png){: .align-center}
上图展示了数据集中患者的性别比例，有助于了解样本的构成。

### 年龄分布直方图

![年龄分布直方图](/images/portfolio/picu-data-analysis/age_month_hist.png){: .align-center}
直方图展示了 PICU 患者的年龄（以月为单位）分布，揭示了患者群体的年龄特征。

### 住院死亡标志分布

![住院死亡标志甜甜圈图](/images/portfolio/picu-data-analysis/hospital_expire_flag_donut.png){: .align-center}
该图显示了患者住院期间的死亡比例，是评估模型预测目标的关键指标。

### 关键实验室指标箱线图（如果代码中包含）

![关键实验室指标箱线图](/images/portfolio/picu-data-analysis/lab_indicators_boxplot.png){: .align-center}
箱线图对比了存活与死亡患者在各项实验室指标上的分布差异，有助于识别潜在的预测因子。

## 机器学习模型评估

在构建预测模型后，我们使用混淆矩阵和 ROC 曲线来评估模型的性能。

### 混淆矩阵

![混淆矩阵](/images/portfolio/picu-data-analysis/confusion_matrix.png){: .align-center}
混淆矩阵直观地展示了模型在分类任务中的表现，包括真阳性、真阴性、假阳性和假阴性的数量。

### ROC 曲线

![ROC 曲线](/images/portfolio/picu-data-analysis/roc_curve.png){: .align-center}
ROC 曲线及其曲线下面积 (AUC) 是评估二分类模型判别能力的重要指标。更高的 AUC 值通常表示更好的模型性能。

## 结论与展望

本项目通过对 PICU 患者数据的深入分析和可视化，为理解患者特征和疾病模式提供了见解。构建的机器学习模型在预测患者预后方面也展现出一定的潜力。未来的工作可以进一步探索更复杂的模型架构、集成更多临床数据，并进行更全面的特征工程，以提高预测的准确性和临床实用性。
