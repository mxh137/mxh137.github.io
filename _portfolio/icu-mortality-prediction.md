---
title: "ICU 患者院内死亡预测与多模型评估"
collection: portfolio
excerpt: "本项目旨在利用重症监护室 (ICU) 患者的临床数据，通过系统的数据清洗、特征工程和探索性数据分析，构建并评估多种机器学习模型（如 SVM, 随机森林, XGBoost）来预测患者的院内死亡风险。项目详细展示了数据预处理流程、关键特征的可视化分析、模型训练与评估指标（AUC-ROC, AUPRC, 混淆矩阵），并深入探讨了 XGBoost 模型的特征重要性（SHAP 值）。"
date: 2026-01-12
tags:
  - 数据分析
  - Python
  - Jupyter Notebook
  - 机器学习
  - 可视化
  - PICU
  - 医疗数据
  - 预测模型
  - XGBoost
  - SHAP
---

## 项目概述

本项目旨在对儿科重症监护病房（PICU）患者的首24小时临床数据进行全面的分析，包括数据加载、清洗、探索性数据分析（EDA），并搭建机器学习预测模型以识别高风险患者。通过可视化的方式，我们展示了患者的性别、年龄和住院结局等关键特征。在模型构建阶段，我们考虑了多种分类算法，并详细阐述了数据预处理流程和评估方法。

**数据来源**：`icu_first24hours.csv` 及清洗后的 `icu_first24hours_clean_drop0.4.csv`  
**分析工具**：Python (pandas, numpy, matplotlib, scikit-learn, xgboost)

---

## 1. 数据读取

本项目从原始 CSV 文件加载数据，并进行初步的结构检查，以了解数据集的规模和基本构成。

**原始数据路径**：`/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv`

**代码片段**：
```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 导入机器学习相关的包，方便后续使用
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)

# 数据加载
path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv"
df = pd.read_csv(path)

print("原始数据形状:", df.shape)
print("\n数据表头:")
# display(df.head(3)) # 在实际notebook中会显示前3行
print("\n前20列数据类型:")
# print(df.dtypes.head(20)) # 在实际notebook中会显示前20列的数据类型
```

**说明**：
- 加载原始数据集 `icu_first24hours.csv`。
- 打印数据集的维度 (`shape`) 和前几行 (`head`)，以便快速预览数据结构和内容。
- 检查各列的数据类型 (`dtypes`)，有助于发现潜在的数据类型问题。

---

## 2. 数据预处理

数据预处理是确保模型性能和数据质量的关键步骤。本节涵盖了缺失值检查、列名清洗、时间列处理、重复值处理以及高缺失率列的删除。

**代码片段**：
```python
# ========= 配置区 =========
path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv"
col_missing_thresh = 0.4  # 缺失率超过该阈值的列直接丢弃
target_cols = ["HOSPITAL_EXPIRE_FLAG", "is_early_death"] # 目标列，建议保留不缺失的行
dedup_by_hadm = True      # 是否去重（按 HADM_ID 保留第一条）
out_dir = "/wujidata/xdl/ecg_sleep/icu"
os.makedirs(out_dir, exist_ok=True)
clean_output_path = os.path.join(out_dir, f"icu_first24hours_clean_drop{col_missing_thresh}.csv")
# =================================

# 1) 缺失数据情况检查
print("\n缺失率（前30列）:")
missing_rate = df.isna().mean().sort_values(ascending=False)
# display(missing_rate.head(30)) # 在实际notebook中会显示前30列的缺失率

# 2) 基础清洗：列名去空格、时间解析、inf->nan
df.columns = [c.strip() for c in df.columns]
if "ADMITTIME" in df.columns:
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

# 3) 去重 (可选)
if dedup_by_hadm and "HADM_ID" in df.columns:
    before = df.shape[0]
    df = df.drop_duplicates(subset=["HADM_ID"], keep="first")
    print(f"按 HADM_ID 去重: {before} -> {df.shape[0]} 行")

# 4) 计算缺失率并删除高缺失列
missing_rate = df.isna().mean().sort_values(ascending=False)
cols_to_drop = missing_rate[missing_rate > col_missing_thresh].index.tolist()
df_cleaned = df.drop(columns=cols_to_drop)
print(f"删除缺失率 > {col_missing_thresh*100}% 的列: {len(cols_to_drop)} 列")

# 5) 删除目标列缺失的行 (根据需要)
for t_col in target_cols:
    if t_col in df_cleaned.columns:
        before = df_cleaned.shape[0]
        df_cleaned = df_cleaned.dropna(subset=[t_col])
        print(f"删除 '{t_col}' 缺失的行: {before} -> {df_cleaned.shape[0]} 行")

# 6) 保存清洗后的数据
df_cleaned.to_csv(clean_output_path, index=False)
print("清洗后的数据已保存至:", clean_output_path)
print("清洗后数据形状:", df_cleaned.shape)
```

**说明**：
- **缺失率检查**：识别数据集中缺失值较多的列，为后续处理提供依据。
- **列名标准化**：去除列名中的冗余空格，确保后续操作的便捷性。
- **时间列解析**：将 `ADMITTIME` 转换为日期时间格式，便于时间序列分析。
- **异常值处理**：将 `np.inf` 和 `-np.inf` 替换为 `np.nan`，以便统一处理缺失值。
- **重复值处理**：根据 `HADM_ID` 去重，确保每个住院记录的唯一性。
- **高缺失率列删除**：移除缺失率过高的列，减少噪音并提高模型效率。
- **目标列缺失值处理**：删除目标变量（如 `HOSPITAL_EXPIRE_FLAG`）缺失的行，确保模型训练数据的完整性。
- **保存清洗后数据**：将清洗后的数据保存到 `icu_first24hours_clean_drop0.4.csv`，方便后续步骤直接使用。

---

## 3. 统计分析 (探索性数据可视化)

通过可视化图表直观展示清洗后数据中关键变量的分布情况，以更好地理解患者群体的特征。

**数据源**：`icu_first24hours_clean_drop0.4.csv`

### 3.1 通用圆环图函数 (`donut_plot`)

为了生成美观且信息丰富的圆环图，我们定义了一个可重用的 `donut_plot` 函数，该函数允许自定义圆环厚度、百分比和标签的位置。

**代码片段**：
```python
# (假设 df_cleaned 已加载，out_dir 已定义)

def donut_plot(counts, labels, title, save_path, dpi=600,
               ring_width=0.40, pctdistance=0.82, labeldistance=1.12,
               min_pct_to_show=0.0):
    """
    生成高质量圆环图，百分比在圆环上。
    ring_width: 圆环厚度 (0~1)，越大越厚
    pctdistance: 百分比文本距离圆心的比例 (外半径=1)，应在 (1-ring_width, 1) 之间
    labeldistance: 标签离圆心距离
    min_pct_to_show: 小于该百分比则不显示
    """
    counts = np.array(counts, dtype=float)
    total = counts.sum()

    def autopct_fmt(pct):
        if pct < min_pct_to_show:
            return ""
        return f"{pct:.1f}%"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=autopct_fmt,
        startangle=90,
        pctdistance=pctdistance,
        labeldistance=labeldistance,
        wedgeprops=dict(width=ring_width, edgecolor="white"),
        textprops=dict(fontsize=12)
    )

    for t in autotexts:
        t.set_fontsize(12)

    ax.set_title(title, fontsize=16)
    ax.axis("equal")
    ax.text(0, 0, f"N={int(total)}", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)
```

### 3.2 性别分布分析 (`gender_is_male`)

<img src="/images/gender_is_male_donut.png" alt="性别比例圆环图" style="width: 50%; margin: 0 auto; display: block;">

**代码片段**：
```python
# (假设 df_cleaned 已加载，out_dir 已定义)
if "gender_is_male" in df_cleaned.columns:
    g = df_cleaned["gender_is_male"].astype(int).value_counts().sort_index()
    if set(g.index).issubset({0, 1}):
        labels = ["Female (0)", "Male (1)"]
        counts = [g.get(0, 0), g.get(1, 0)]
    else:
        labels = [str(i) for i in g.index]
        counts = g.values
    donut_plot(
        counts=counts,
        labels=labels,
        title="Gender Distribution (gender_is_male)",
        save_path=os.path.join(out_dir, "gender_is_male_donut.png"),
        dpi=600,
        ring_width=0.40, pctdistance=0.82, labeldistance=1.12, min_pct_to_show=0.0
    )
else:
    print("Column not found: gender_is_male")
```

**分析说明**：
- 该圆环图清晰展示了 PICU 患者的性别构成，其中 `Female (0)` 为女性，`Male (1)` 为男性。
- 性别分布有助于理解患者群体的基本人口学特征。

### 3.3 年龄分布分析 (`age_month`)

<img src="/images/age_month_hist.png" alt="年龄分布直方图" style="width: 50%; margin: 0 auto; display: block;">


**代码片段**：
```python
# (假设 df_cleaned 已加载，out_dir 已定义)
if "age_month" in df_cleaned.columns:
    age = pd.to_numeric(df_cleaned["age_month"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(age, bins=30, edgecolor="black")
    ax.set_title("Age Distribution (months)", fontsize=16)
    ax.set_xlabel("age_month", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "age_month_hist.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)
else:
    print("Column not found: age_month")
```

**分析说明**：
- 直方图展示了 PICU 患者的月龄分布，横轴为月龄，纵轴为患者数量。
- 通过 30 个分箱，可以观察到不同月龄段患者的集中趋势和分布特征。

### 3.4 住院结局分析 (`HOSPITAL_EXPIRE_FLAG`)

<img src="/images/hospital_expire_flag_donut.png" alt="住院结局圆环图" style="width: 50%; margin: 0 auto; display: block;">

**代码片段**：
```python
# (假设 df_cleaned 已加载，out_dir 已定义)
if "HOSPITAL_EXPIRE_FLAG" in df_cleaned.columns:
    y = df_cleaned["HOSPITAL_EXPIRE_FLAG"].astype(int).value_counts().sort_index()
    if set(y.index).issubset({0, 1}):
        labels = ["Alive (0)", "Expired (1)"]
        counts = [y.get(0, 0), y.get(1, 0)]
    else:
        labels = [str(i) for i in y.index]
        counts = y.values
    donut_plot(
        counts=counts,
        labels=labels,
        title="Hospital Outcome (HOSPITAL_EXPIRE_FLAG)",
        save_path=os.path.join(out_dir, "hospital_expire_flag_donut.png"),
        dpi=600,
        ring_width=0.40, pctdistance=0.82, labeldistance=1.12, min_pct_to_show=0.0
    )
else:
    print("Column not found: HOSPITAL_EXPIRE_FLAG")
```

**分析说明**：
- 该圆环图直观地呈现了 PICU 患者的住院结局分布，其中 `Alive (0)` 表示存活，`Expired (1)` 表示死亡。
- 这是评估 PICU 医疗质量和患者预后的重要指标。

---

## 4. 预测模型建立

本节详细介绍了机器学习模型的构建过程，包括数据集的切分、针对不同模型类型的特征预处理以及多种分类模型的定义。

**数据源**：`icu_first24hours_clean_drop0.4.csv`

### 4.1 数据集切分

我们将清洗后的数据划分为特征集 ($$X$$) 和目标变量 ($$y$$)，并进一步切分为训练集和测试集，采用分层抽样以保持类别比例。

**代码片段**：
```python
# (假设 clean_output_path, target, random_state, test_size 已定义)
df_cleaned = pd.read_csv(clean_output_path) # 重新加载清洗后的数据

# 时间列处理（不参与特征）
if "ADMITTIME" in df_cleaned.columns:
    try:
        df_cleaned["ADMITTIME"] = pd.to_datetime(df_cleaned["ADMITTIME"], errors="coerce")
    except Exception:
        pass

# 构建 X / y
if target not in df_cleaned.columns:
    raise ValueError(f"找不到 target 列：{target}")

id_time_cols = [c for c in ["SUBJECT_ID", "HADM_ID", "ADMITTIME"] if c in df_cleaned.columns]
drop_cols = set(id_time_cols + [target])

feature_cols = [c for c in df_cleaned.columns if c not in drop_cols]
X = df_cleaned[feature_cols].copy()
y = df_cleaned[target].astype(int).copy()

if y.nunique() < 2:
    raise ValueError(f"{target} 只有一个类别，无法做二分类。")

print(f"数据集形状: X={X.shape}, 正例比例={y.mean():.4f}")

# 数据集切分 (分层抽样)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
```

### 4.2 特征预处理

根据不同的模型需求，我们对数据进行缺失值插补和标准化处理。

**代码片段**：
```python
# (假设 X_train, X_test, feature_cols 已切分)

# Tree 模型预处理 (只插补)
imputer_tree = SimpleImputer(strategy="median") # 使用中位数插补
X_train_imp = imputer_tree.fit_transform(X_train)
X_test_imp  = imputer_tree.transform(X_test)

X_train_imp_df = pd.DataFrame(X_train_imp, columns=feature_cols)
X_test_imp_df  = pd.DataFrame(X_test_imp,  columns=feature_cols)
print(f"树模型插补后训练集形状: {X_train_imp_df.shape}")

# SVM/线性模型预处理 (插补 + 标准化)
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train_imp_df)
X_test_svm  = scaler.transform(X_test_imp_df)
print(f"SVM模型标准化后训练集形状: {X_train_svm.shape}")
```

**说明**：
- **树模型预处理**：对于决策树、随机森林等对特征尺度不敏感的模型，通常只需进行缺失值插补。这里采用中位数插补。
- **SVM/线性模型预处理**：对于支持向量机、逻辑回归等对特征尺度敏感的模型，除了缺失值插补外，还需要进行特征标准化。这里采用 `StandardScaler` 进行标准化。

### 4.3 预测模型定义

我们定义了多种分类模型，包括支持向量机 (SVM)、随机森林 (RandomForest) 和梯度提升机 (XGBoost)，并针对类别不平衡问题进行了 `class_weight` 或 `scale_pos_weight` 的设置。

**代码片段**：
```python
# (假设 y_train, random_state 已定义)

# SVM
svm = SVC(
    kernel="rbf", C=1.0, gamma="scale",
    probability=True, # 允许输出概率
    class_weight="balanced", # 处理类别不平衡
    random_state=random_state
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500, # 决策树数量
    random_state=random_state,
    n_jobs=-1, # 使用所有核心并行计算
    class_weight="balanced" # 处理类别不平衡
)

# XGBoost
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        f"未检测到 xgboost，请先安装：pip install xgboost 或 conda install -c conda-forge xgboost\n原始报错：{e}"
    )

pos_ratio = y_train.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio  # 计算负样本/正样本比例，用于处理类别不平衡

xgb = XGBClassifier(
    objective="binary:logistic", # 二分类逻辑回归
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    use_label_encoder=False, # 禁用旧的标签编码器警告
    eval_metric="logloss", # 评估指标
    scale_pos_weight=scale_pos_weight, # 处理类别不平衡
    random_state=random_state,
    n_jobs=-1
)

# 统一管理模型及其对应的预处理类型
models = {
    "SVM": ("svm", svm), # SVM 使用标准化数据
    "RF": ("tree", rf), # 随机森林使用插补后的数据
    "XGBoost": ("tree", xgb) # XGBoost 使用插补后的数据
}
```

---

## 5. 预测模型评估与可视化

本节详细介绍了如何评估训练好的模型，包括性能指标的计算和关键评估曲线的绘制。

### 5.1 性能指标计算函数 (`compute_metrics`)

我们定义了一个函数来计算分类模型的各种性能指标，包括 ROC AUC、PR AUC、准确率、精确率、召回率和 F1 分数。

**代码片段**：
```python
# (假设 y_true, y_prob 已定义)
def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int) # 根据阈值进行分类预测
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }, y_pred
```

### 5.2 模型训练、预测与评估循环

我们将遍历定义好的所有模型，进行训练、预测，并计算性能指标。

**代码片段**：
```python
# (假设 models, X_train_svm, X_test_svm, X_train_imp_df, X_test_imp_df, y_train, y_test, out_dir 已定义)

results = [] # 存储所有模型的评估结果
probas = {}  # 存储所有模型的预测概率
preds = {}   # 存储所有模型的预测类别

for name, (ptype, clf) in models.items():
    print(f"\n--- 训练和评估模型: {name} ---")

    # fit + predict proba
    if ptype == "svm":
        clf.fit(X_train_svm, y_train)
        y_prob = clf.predict_proba(X_test_svm)[:, 1] # 获取正类的预测概率
    else: # 树模型
        clf.fit(X_train_imp_df, y_train)
        y_prob = clf.predict_proba(X_test_imp_df)[:, 1]

    # 计算评估指标
    metrics, y_pred = compute_metrics(y_test, y_prob)
    results.append({"model": name, **metrics})
    probas[name] = y_prob
    preds[name] = y_pred

    print(f"{name} 评估结果:")
    print(pd.DataFrame([metrics]).round(4))

    # 保存混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Expired"])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp_cm.plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=300)
    plt.close(fig_cm)
    print(f"Saved: {name}_confusion_matrix.png")

    # 保存 ROC 曲线
    disp_roc = RocCurveDisplay.from_estimator(
        clf,
        (X_test_svm if ptype == "svm" else X_test_imp_df),
        y_test,
        name=name
    )
    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    disp_roc.plot(ax=ax_roc)
    ax_roc.set_title(f"{name} ROC Curve")
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_roc_curve.png"), dpi=300)
    plt.close(fig_roc)
    print(f"Saved: {name}_roc_curve.png")

    # 保存 Precision-Recall 曲线
    disp_pr = PrecisionRecallDisplay.from_estimator(
        clf,
        (X_test_svm if ptype == "svm" else X_test_imp_df),
        y_test,
        name=name
    )
    fig_pr, ax_pr = plt.subplots(figsize=(7, 7))
    disp_pr.plot(ax=ax_pr)
    ax_pr.set_title(f"{name} Precision-Recall Curve")
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_pr_curve.png"), dpi=300)
    plt.close(fig_pr)
    print(f"Saved: {name}_pr_curve.png")

print("\n所有模型评估结果:")
print(pd.DataFrame(results).round(4))
```

### 5.3 评估结果可视化

在 `/images` 文件夹中，我们应该有如下模型评估图，例如：

#### SVM 模型评估

*   **SVM 混淆矩阵**
    ![SVM Confusion Matrix](/images/SVM_confusion_matrix.png){: .align-center}

*   **SVM ROC 曲线**
    ![SVM ROC Curve](/images/SVM_roc_curve.png){: .align-center}

*   **SVM Precision-Recall 曲线**
    ![SVM Precision-Recall Curve](/images/SVM_pr_curve.png){: .align-center}

#### Random Forest 模型评估

*   **Random Forest 混淆矩阵**
    ![RandomForest Confusion Matrix](/images/RF_confusion_matrix.png){: .align-center}

*   **Random Forest ROC 曲线**
    ![RandomForest ROC Curve](/images/RF_roc_curve.png){: .align-center}

*   **Random Forest Precision-Recall 曲线**
    ![RandomForest Precision-Recall Curve](/images/RF_pr_curve.png){: .align-center}

#### XGBoost 模型评估

*   **XGBoost 混淆矩阵**
    ![XGBoost Confusion Matrix](/images/XGBoost_confusion_matrix.png){: .align-center}

*   **XGBoost ROC 曲线**
    ![XGBoost ROC Curve](/images/XGBoost_roc_curve.png){: .align-center}

*   **XGBoost Precision-Recall 曲线**
    ![XGBoost Precision-Recall Curve](/images/XGBoost_pr_curve.png){: .align-center}

**分析说明**：
- **混淆矩阵 (Confusion Matrix)**：直观展示了模型在测试集上的分类性能，包括真阳性 (TP)、真阴性 (TN)、假阳性 (FP) 和假阴性 (FN)。
- **ROC 曲线 (Receiver Operating Characteristic Curve)**：通过绘制真阳性率 (TPR) 对假阳性率 (FPR)，评估模型在不同分类阈值下的性能。曲线下面积 (AUC) 越大，模型性能越好。
- **Precision-Recall 曲线 (PR Curve)**：对于类别不平衡的数据集，PR 曲线能更好地反映模型的性能。曲线下面积 (AP) 越大，模型性能越好。

---

## 结论与展望

本项目从 PICU 患者数据的读取、清洗、探索性分析到机器学习模型的建立和评估，构建了一个完整的预测分析流程。通过对性别、年龄和住院结局的统计分析，我们对患者群体有了初步的了解。在模型建立阶段，我们成功训练并评估了 SVM、随机森林和 XGBoost 等多种分类模型，并通过混淆矩阵、ROC 曲线和 PR 曲线全面展示了它们的性能。

**后续工作**：
- **模型优化**：进一步进行超参数调优，尝试更复杂的集成学习方法。
- **特征工程**：探索更丰富的临床特征，例如基于时间序列的特征提取。
- **模型可解释性**：利用 SHAP、LIME 等工具深入理解模型决策，提高临床接受度。
- **部署与监控**：将表现最佳的模型部署到实际环境中，并持续监控其性能。

---

## 技术栈

- **语言**：Python 3.x
- **核心库**：
    - `pandas`：数据处理与分析
    - `numpy`：数值计算
    - `matplotlib`：数据可视化
    - `scikit-learn`：机器学习（预处理、模型、评估）
    - `xgboost`：梯度提升框架

---

## 相关链接

- **GitHub 仓库**：[项目代码](https://github.com/mxh137/mxh137.github.io)
- **数据集说明**：[数据文档](链接)
- **相关论文**：[文献引用](链接)

---

*最后更新：2026-01-13*
```
