### title: "ICU患者院内死亡预测与多模型评估"
collection: portfolio
excerpt: "本项目旨在利用重症监护室 (ICU) 患者的临床数据，通过系统的数据清洗、特征工程和探索性数据分析，构建并评估多种机器学习模型（如SVM, 随机森林, XGBoost）来预测患者的院内死亡风险。项目详细展示了数据预处理流程、关键特征的可视化分析、模型训练与评估指标（AUC-ROC, AUPRC, 混淆矩阵），并深入探讨了 XGBoost 模型的特征重要性（SHAP 值）。"
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

本项目专注于对重症监护室 (ICU) 患者的临床数据进行深入分析，旨在构建和评估机器学习模型，以预测患者的院内死亡风险。通过对数据的全面处理、探索性分析和多种模型的比较，我们力求提高预测的准确性和可解释性，为临床决策提供支持。

## 1. 数据导入与初步探索

项目首先导入所有必需的 Python 库，用于数据处理、机器学习和可视化。接着，加载原始数据集，并进行初步的结构和内容检查。

### 1.1 导入所需库

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)

import matplotlib.pyplot as plt
import os
```

### 1.2 检查表格的结构和数据类型

加载 `icu_first24hours.csv` 数据集，查看其维度、前几行数据以及各列的数据类型，以对数据有一个初步的认识。

```python
# 数据路径（请根据实际环境调整）
path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv"
df = pd.read_csv(path)

print(df.shape)
# display(df.head(3)) # 在Jupyter中会显示表格，这里为了Markdown简洁，省略了
print(df.head(3)) # 在Markdown中直接打印，方便查看
print(df.dtypes.head(20))
```

### 1.3 缺失数据情况检查与标签分布

检查数据集中各列的缺失率，并特别关注目标变量“院内死亡标志”（`HOSPITAL_EXPIRE_FLAG`）的分布情况，这对于后续的数据清洗和模型训练至关重要。

```python
missing_rate = df.isna().mean().sort_values(ascending=False)
# display(missing_rate.head(30)) # 在Jupyter中会显示表格，这里为了Markdown简洁，省略了
print("Top 30 Missing Rates:")
print(missing_rate.head(30))

# 看看标签分布（以院内死亡为例）
target = "HOSPITAL_EXPIRE_FLAG"
print(f"\nValue counts for target '{target}':")
print(df[target].value_counts(dropna=False))
print("Positive rate:", df[target].mean())
```

## 2. 数据预处理

数据预处理是构建健壮机器学习模型的关键步骤，包括处理缺失值、去除重复项和低信息列。本节详细介绍了清洗策略。

```python
# ========= 你需要改的参数 =========
# path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv" # 路径已在前面定义

# 缺失率超过该阈值的列直接丢弃（建议 0.4~0.7 之间试一下）
col_missing_thresh = 0.4

# 你后续可能要预测的标签：建议至少保留 HOSPITAL_EXPIRE_FLAG 不缺失的行
target_cols = ["HOSPITAL_EXPIRE_FLAG", "is_early_death"]

# 是否去重（按一次住院 HADM_ID 保留第一条）
dedup_by_hadm = True
# =================================


# 1) 读取（已在前面完成）
# df = pd.read_csv(path)
print("Raw shape:", df.shape)

# 2) 基础清洗：列名去空格、时间解析、inf->nan
df.columns = [c.strip() for c in df.columns]
if "ADMITTIME" in df.columns:
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# 3) 去重（可选）
if dedup_by_hadm and "HADM_ID" in df.columns:
    before = df.shape[0]
    df = df.drop_duplicates(subset=["HADM_ID"], keep="first")
    print(f"Dedup by HADM_ID: {before} -> {df.shape[0]}")

# 4) 处理标签缺失（强烈建议：至少保证你要预测的标签不缺失）
# 这里默认：如果 HOSPITAL_EXPIRE_FLAG 存在，就要求它不缺失；否则要求 is_early_death 不缺失
if "HOSPITAL_EXPIRE_FLAG" in df.columns:
    before = df.shape[0]
    df = df.dropna(subset=["HOSPITAL_EXPIRE_FLAG"])
    print(f"Drop rows with missing HOSPITAL_EXPIRE_FLAG: {before} -> {df.shape[0]}")
elif "is_early_death" in df.columns:
    before = df.shape[0]
    df = df.dropna(subset=["is_early_death"])
    print(f"Drop rows with missing is_early_death: {before} -> {df.shape[0]}")

# 5) 计算每列缺失率，并删除缺失率太高的列
missing_rate = df.isna().mean().sort_values(ascending=False)

# 额外规则：全缺失列直接删
all_missing_cols = missing_rate[missing_rate >= 1.0].index.tolist()

# 按阈值删列
drop_cols_by_missing = missing_rate[missing_rate > col_missing_thresh].index.tolist()

drop_cols = sorted(set(all_missing_cols + drop_cols_by_missing))

print(f"Columns to drop (missing>{col_missing_thresh}): {len(drop_cols)}")
print("Top-20 missing columns:")
print(missing_rate.head(20))

df_reduced = df.drop(columns=drop_cols)
print("After dropping high-missing cols shape:", df_reduced.shape)

# 6) 再做一点“低信息列”处理（可选但很实用）
# 6.1) 删除常数列（nunique<=1，且忽略缺失）
nunique = df_reduced.nunique(dropna=True)
const_cols = nunique[nunique <= 1].index.tolist()
if len(const_cols) > 0:
    df_reduced = df_reduced.drop(columns=const_cols)
print(f"Constant columns dropped: {len(const_cols)}")
print("Shape after dropping constant cols:", df_reduced.shape)

# 7) 插补：数值列 median；非数值列 most_frequent
# 注意：ADMITTIME 是 datetime，这里我们不插补它（你也可以选择删掉）
datetime_cols = df_reduced.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
if len(datetime_cols) > 0:
    print("Datetime columns (kept as-is, not imputed):", datetime_cols)

# 只对非 datetime 做插补
cols_to_impute = [c for c in df_reduced.columns if c not in datetime_cols]

num_cols = df_reduced[cols_to_impute].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in cols_to_impute if c not in num_cols]

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df_imputed = df_reduced.copy()

if len(num_cols) > 0:
    df_imputed[num_cols] = num_imputer.fit_transform(df_reduced[num_cols])

if len(cat_cols) > 0:
    df_imputed[cat_cols] = cat_imputer.fit_transform(df_reduced[cat_cols])

# 8) 最终检查：NaN / inf
df_imputed = df_imputed.replace([np.inf, -np.inf], np.nan)

na_total = df_imputed.isna().sum().sum()
inf_total = np.isinf(df_imputed.select_dtypes(include=[np.number]).to_numpy()).sum()

print("\n===== Final Check =====")
print("Final shape:", df_imputed.shape)
print("Total NaN after imputation:", na_total)
print("Total inf after imputation:", inf_total)

# 如果你要求“严格 0 缺失”，这里直接 assert
assert na_total == 0, f"Still has NaN: {na_total}"
assert inf_total == 0, f"Still has inf: {inf_total}"
print("✅ No missing values and no infinities.")

# 9) 保存清洗后的数据
out_path = path.replace(".csv", f"_clean_drop{col_missing_thresh}.csv")
df_imputed.to_csv(out_path, index=False)
print("Saved to:", out_path)
```

## 3. 统计分析与可视化

本节通过可视化手段对清洗后的数据进行探索性分析，以揭示关键变量的分布和特征。

```python
# ========= 路径 =========
clean_path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours_clean_drop0.4.csv"
out_dir = "/wujidata/xdl/ecg_sleep/icu"   # 保存目录
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(clean_path)

# ========= 通用：圆环图函数（百分比在圆环上） =========
def donut_plot(counts, labels, title, save_path, dpi=600,
               ring_width=0.40, pctdistance=0.82, labeldistance=1.12,
               min_pct_to_show=0.0):
    """
    ring_width: 圆环厚度（0~1），越大越厚
    pctdistance: 百分比文本距离圆心的比例（外半径=1），要落在圆环上，应在(1-ring_width, 1)之间
    labeldistance: 标签离圆心距离
    min_pct_to_show: 小于该百分比则不显示（避免挤）
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
        pctdistance=pctdistance,        # ✅ 关键：百分比放到圆环上
        labeldistance=labeldistance,
        wedgeprops=dict(width=ring_width, edgecolor="white"),
        textprops=dict(fontsize=12)
    )

    # 让百分比字体更清晰一点
    for t in autotexts:
        t.set_fontsize(12)

    ax.set_title(title, fontsize=16)
    ax.axis("equal")

    # 中心写总数 N
    ax.text(0, 0, f"N={int(total)}", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)

# ========= 3.1 性别比例圆环图 =========
if "gender_is_male" in df.columns:
    g = df["gender_is_male"].astype(int).value_counts().sort_index()

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
        ring_width=0.40,
        pctdistance=0.82,        # ✅ 百分比在圆环上
        labeldistance=1.12,
        min_pct_to_show=0.0      # 如想隐藏很小扇区，比如2%以下：改成 2.0
    )
else:
    print("Column not found: gender_is_male")

### 性别比例分布

![性别比例甜甜圈图](/images/portfolio/picu-outcome-prediction/gender_is_male_donut.png){: .align-center}
上图展示了数据集中 PICU 患者的性别比例，有助于了解样本的构成。

# ========= 3.2 年龄分布直方图 =========
if "age_month" in df.columns:
    age = pd.to_numeric(df["age_month"], errors="coerce").dropna()

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

### 年龄分布直方图

![年龄分布直方图](/images/portfolio/picu-outcome-prediction/age_month_hist.png){: .align-center}
直方图展示了 PICU 患者的年龄（以月为单位）分布，揭示了患者群体的年龄特征。

# ========= 3.3 住院死亡标志圆环图 =========
if "HOSPITAL_EXPIRE_FLAG" in df.columns:
    y = df["HOSPITAL_EXPIRE_FLAG"].astype(int).value_counts().sort_index()

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
        ring_width=0.40,
        pctdistance=0.82,        # ✅ 百分比在圆环上
        labeldistance=1.12,
        min_pct_to_show=0.0
    )
else:
    print("Column not found: HOSPITAL_EXPIRE_FLAG")

### 住院死亡标志分布

![住院死亡标志甜甜圈图](/images/portfolio/picu-outcome-prediction/hospital_expire_flag_donut.png){: .align-center}
该图显示了患者住院期间的死亡比例，是评估模型预测目标的关键指标。
```

## 4. 预测模型建立

本节将数据划分为训练集和测试集，并对数据进行进一步预处理（插补和标准化），然后定义并初始化多种机器学习模型用于院内死亡预测。

```python
# ===================== 配置区 =====================
clean_path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours_clean_drop0.4.csv"
target = "HOSPITAL_EXPIRE_FLAG"
random_state = 42
test_size = 0.2

out_dir = "/wujidata/xdl/ecg_sleep/icu/ml_outputs"
os.makedirs(out_dir, exist_ok=True)

# ===================== 读取数据 =====================
df = pd.read_csv(clean_path)

# 时间列仅用于信息，不参与特征
if "ADMITTIME" in df.columns:
    try:
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    except Exception:
        pass

# ===================== 构建 X / y =====================
if target not in df.columns:
    raise ValueError(f"找不到 target 列：{target}")

id_time_cols = [c for c in ["SUBJECT_ID", "HADM_ID", "ADMITTIME"] if c in df.columns]
drop_cols = set(id_time_cols + [target])

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df[target].astype(int).copy()

if y.nunique() < 2:
    raise ValueError(f"{target} 只有一个类别，无法做二分类。")

print(f"Dataset: X={X.shape}, positive rate={y.mean():.4f}")

# ===================== 切分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
print(f"Train/Test split: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"X_test={X_test.shape}, y_test={y_test.shape}")

# ===================== 预处理数据 =====================
# Tree 模型：只插补（更适合解释 + SHAP）
imputer_tree = SimpleImputer(strategy="median")
X_train_imp = imputer_tree.fit_transform(X_train)
X_test_imp  = imputer_tree.transform(X_test)

X_train_imp_df = pd.DataFrame(X_train_imp, columns=feature_cols)
X_test_imp_df  = pd.DataFrame(X_test_imp,  columns=feature_cols)

# SVM：插补 + 标准化
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train_imp_df)
X_test_svm  = scaler.transform(X_test_imp_df)
print("Data imputation and scaling completed.")

# ===================== 模型定义 =====================
# SVM 模型
svm = SVC(
    kernel="rbf", C=1.0, gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=random_state
)
print("SVM model initialized.")

# 随机森林 (Random Forest) 模型
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=random_state,
    n_jobs=-1,
    class_weight="balanced"
)
print("Random Forest model initialized.")

# XGBoost 模型
try:
    from xgboost import XGBClassifier
except Exception as e:
    print(
        f"未检测到 xgboost，请先安装：pip install xgboost 或 conda install -c conda-forge xgboost\n原始报错：{e}"
    )
    # 如果没有安装xgboost，则将其从模型列表中移除
    xgb = None

if xgb:
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio  # neg/pos

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1
    )
    print("XGBoost model initialized.")


# 统一管理所有模型及其对应的输入数据类型
models = {
    "SVM": ("svm", svm),
    "RF": ("tree", rf),
}
if xgb: # 如果xgboost成功初始化，则添加到模型列表
    models["XGBoost"] = ("tree", xgb)

print("All models defined and ready for training.")
```

## 5. 预测模型评估与可视化

本节对训练好的模型进行评估，计算各项性能指标（如 ROC-AUC, AUPRC, 准确率, F1 分数），并生成 ROC 曲线、PR 曲线和混淆矩阵等可视化图表，最后分析 XGBoost 模型的特征重要性。

```python
# ===================== 训练 / 评估 / 画图保存 =====================
def compute_metrics(y_true, y_prob, thr=0.5):
    """计算模型评估指标"""
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }, y_pred

results = []
probas = {}
preds = {}

for name, (ptype, clf) in models.items():
    print(f"\n--- Training and evaluating {name} ---")
    # fit + predict proba
    if ptype == "svm":
        clf.fit(X_train_svm, y_train)
        y_prob = clf.predict_proba(X_test_svm)[:, 1]
    else: # For tree-based models like RF, XGBoost
        clf.fit(X_train_imp_df, y_train)
        y_prob = clf.predict_proba(X_test_imp_df)[:, 1]

    probas[name] = y_prob

    mets, y_pred = compute_metrics(y_test.values, y_prob, thr=0.5)
    preds[name] = y_pred

    results.append({"target": target, "model": name, **mets})

    print(f"\n=== {name} Performance ===")
    for k, v in mets.items():
        print(f"{k}: {v:.6f}")

    # --------- 5.1 单模型 ROC 曲线图保存 ---------
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob, name=f"{name} (AUC={mets['roc_auc']:.3f})", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC Curve - {name} ({target})")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"ROC_{name}_{target}.png")
    plt.savefig(roc_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", roc_path)

    # --------- 5.2 单模型 PR 曲线图保存 ---------
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=f"{name} (AP={mets['pr_auc']:.3f})", ax=ax)
    ax.set_title(f"PR Curve - {name} ({target})")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"PR_{name}_{target}.png")
    plt.savefig(pr_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", pr_path)

    # --------- 5.3 单模型混淆矩阵（蓝色色调）---------
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=[0, 1],
        values_format="d",
        cmap="Blues",   # ✅ 蓝色调
        ax=ax
    )
    ax.set_title(f"Confusion Matrix - {name} ({target})")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"CM_{name}_{target}.png")
    plt.savefig(cm_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", cm_path)

print("\n--- Model evaluation complete ---")
# ===================== 保存指标表 =====================
res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
metrics_csv = os.path.join(out_dir, f"metrics_{target}.csv")
res_df.to_csv(metrics_csv, index=False)
print("\nSaved metrics to:", metrics_csv)

### 5.4 各模型评估结果一览

以下表格汇总了 SVM、随机森林和 XGBoost 模型在测试集上的关键评估指标：

| Model     | ROC AUC  | PR AUC   | Accuracy | Precision | Recall   | F1-Score |
| :-------- | :------- | :------- | :------- | :-------- | :------- | :------- |
| SVM       | {{ svm_roc_auc }} | {{ svm_pr_auc }} | {{ svm_acc }} | {{ svm_precision }} | {{ svm_recall }} | {{ svm_f1 }} |
| RF        | {{ rf_roc_auc }} | {{ rf_pr_auc }} | {{ rf_acc }} | {{ rf_precision }} | {{ rf_recall }} | {{ rf_f1 }} |
| XGBoost   | {{ xgb_roc_auc }} | {{ xgb_pr_auc }} | {{ xgb_acc }} | {{ xgb_precision }} | {{ xgb_recall }} | {{ xgb_f1 }} |

_注：上述表格中的 `{{...}}` 占位符在实际生成时将替换为具体的数值。_

### 5.5 模型评估可视化

#### SVM 模型

![SVM ROC 曲线](/images/portfolio/picu-outcome-prediction/roc_svm.png){: .align-center}
_SVM 模型的 ROC 曲线_

![SVM PR 曲线](/images/portfolio/picu-outcome-prediction/pr_svm.png){: .align-center}
_SVM 模型的 PR 曲线_

![SVM 混淆矩阵](/images/portfolio/picu-outcome-prediction/cm_svm.png){: .align-center}
_SVM 模型的混淆矩阵_

#### 随机森林模型

![随机森林 ROC 曲线](/images/portfolio/picu-outcome-prediction/roc_rf.png){: .align-center}
_随机森林模型的 ROC 曲线_

![随机森林 PR 曲线](/images/portfolio/picu-outcome-prediction/pr_rf.png){: .align-center}
_随机森林模型的 PR 曲线_

![随机森林混淆矩阵](/images/portfolio/picu-outcome-prediction/cm_rf.png){: .align-center}
_随机森林模型的混淆矩阵_

#### XGBoost 模型

![XGBoost ROC 曲线](/images/portfolio/picu-outcome-prediction/roc_xgboost.png){: .align-center}
_XGBoost 模型的 ROC 曲线_

![XGBoost PR 曲线](/images/portfolio/picu-outcome-prediction/pr_xgboost.png){: .align-center}
_XGBoost 模型的 PR 曲线_

![XGBoost 混淆矩阵](/images/portfolio/picu-outcome-prediction/cm_xgboost.png){: .align-center}
_XGBoost 模型的混淆矩阵_

### 5.6 XGBoost：SHAP 与特征重要性分析

为了深入理解模型的决策过程，我们对表现良好的 XGBoost 模型进行了 SHAP 值分析，以揭示最重要的预测特征。

```python
# ===================== XGBoost：SHAP + Top10 特征重要性 =====================
# 说明：需要 shap 包：pip install shap
try:
    import shap
except Exception as e:
    print(f"未检测到 shap，请先安装：pip install shap\n原始报错：{e}")
    shap = None # 如果shap未安装，则跳过后续分析

if xgb and shap: # 确保xgb和shap都已安装并初始化
    xgb_clf = xgb  # 已训练
    # 采样一部分数据做 SHAP（加速）
    n_shap = min(2000, len(X_test_imp_df))
    X_shap = X_test_imp_df.sample(n=n_shap, random_state=random_state)

    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_shap)

    # 1) SHAP summary top10
    plt.figure()
    shap.summary_plot(shap_values, X_shap, max_display=10, show=False)
    plt.title("SHAP Summary (Top 10) - XGBoost")
    shap_path = os.path.join(out_dir, f"SHAP_summary_top10_{target}.png")
    plt.savefig(shap_path, dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved:", shap_path)

    # 2) XGBoost 特征重要性 Top10（gain/weight 都可，这里用 model 自带 feature_importances_）
    importances = pd.Series(xgb_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top10 = importances.head(10)

    fig, ax = plt.subplots(figsize=(7, 5.6))
    top10.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title("XGBoost Feature Importance (Top 10)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(out_dir, f"XGB_feature_importance_top10_{target}.png")
    plt.savefig(fi_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fi_path)
else:
    print("Skipping SHAP and Feature Importance: XGBoost or SHAP library not available.")

print("\n✅ All done. Outputs saved in:", out_dir)
```

#### SHAP Summary Plot (Top 10)

![SHAP Summary Plot](/images/portfolio/picu-outcome-prediction/shap_summary_top10.png){: .align-center}
_SHAP 摘要图展示了最重要的10个特征及其对模型输出的影响方向和大小。_

#### XGBoost 特征重要性 (Top 10)

![XGBoost Feature Importance](/images/portfolio/picu-outcome-prediction/xgb_feature_importance_top10.png){: .align-center}
_XGBoost 特征重要性条形图直观地展示了模型在预测过程中最依赖的10个特征。_

## 结论与展望

本项目成功地对 PICU 患者数据进行了全面的预处理、探索性分析和多模型预测。通过比较 SVM、随机森林和 XGBoost 模型，我们不仅评估了它们的预测性能，还利用 SHAP 值深入理解了 XGBoost 模型的决策机制和关键特征。

未来的工作可以包括：
1. 探索更复杂的特征工程方法，如时间序列特征提取。
2. 尝试深度学习模型，以捕捉数据中更复杂的非线性关系。
3. 进行更广泛的超参数调优和模型集成，进一步提升预测性能。
4. 结合临床领域知识，对模型的预测结果进行更深入的解释和验证，以促进其在实际医疗场景中的应用。
---
```
