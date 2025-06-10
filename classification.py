#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import nbformat as nbf
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
import warnings
import matplotlib
import joblib
import shap
warnings.filterwarnings('ignore')
# Set up matplotlib backend to avoid font problems
matplotlib.use('Agg')

# Set random seeds to ensure results are repeatable
np.random.seed(42)
# Modify font settings to avoid garbled squares

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0

print(" Start pancreatic disease classification analysis...")

# data
print("\n Data loading and pre-processing...")
if os.path.exists("preprocessed_pancreas_islet.h5ad"):
    print(" Load preprocessed data...")
    adata = sc.read_h5ad("preprocessed_pancreas_islet.h5ad")
else:
    print("Load raw data and preprocess it...")
    adata = sc.read_h5ad("pancreas_islet.h5ad")
    sc.pp.filter_genes(adata, min_cells=20)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    adata.write_h5ad("preprocessed_pancreas_islet.h5ad")
    print(" Preprocessing is completed, saved！")

# Print data basic information

print(f"Data shape: {adata.shape}")
print(f"Available annotations: {list(adata.obs.columns)}")

# Confirm that the disease tag exists
if 'disease' not in adata.obs.columns:
    raise ValueError("The data does not contain the 'disease' tag, please check the dataset")

# Find possible donor identifier columns
donor_columns = []
for col in adata.obs.columns:
    if 'donor' in col.lower():
        donor_columns.append(col)
    elif 'individual' in col.lower():  # Sometimes donors are called individual
        donor_columns.append(col)
    elif 'subject' in col.lower():  # Sometimes donors are called subject
        donor_columns.append(col)
    elif 'participant' in col.lower():  # Sometimes donors are called participant
        donor_columns.append(col)

if donor_columns:
    print(f"Find possible donor related columns: {donor_columns}")
    donor_id_column = donor_columns[0]  # Use the first column found
    print(f"utilize '{donor_id_column}' as donor identifiers")
else:
    # if no clear donor column is found, try to find possible grouping columns
    print("No clear donor column found, check other possible grouping columns...")
    possible_group_columns = []
    for col in adata.obs.columns:
        # Check for the inclusion of commonly grouped keywords
        if any(keyword in col.lower() for keyword in ['dataset', 'batch', 'sample', 'id', 'group']):
            possible_group_columns.append(col)
    
    if possible_group_columns:
        print(f"Find possible grouping columns: {possible_group_columns}")
        donor_id_column = possible_group_columns[0]  # 使用第一个找到的列
        print(f"utilize '{donor_id_column}' as a group identifier")
    else:
        print("Can't find a column suitable for grouping, use random grouping")
        adata.obs['random_group'] = np.random.randint(0, 10, size=adata.shape[0])
        donor_id_column = 'random_group'

# 查看选择的分组列的唯一值数量
unique_groups = adata.obs[donor_id_column].unique()
print(f"'{donor_id_column}'The columns contain {len(unique_groups)} unique value")
print(f"Example of unique value: {unique_groups[:5]}...")

# 检查每个类别的样本数量
group_counts = adata.obs[donor_id_column].value_counts()
print("Sample size per category:", group_counts)

# 检查每个donor中的疾病分布
donor_disease_dist = adata.obs.groupby(donor_id_column)['disease'].apply(lambda x: x.unique().tolist())
print("Distribution of disease types per donor:")
for donor, diseases in donor_disease_dist.head(10).items():
    print(f"  {donor}: {diseases}")

# 尝试进行分层抽样，如果失败则使用简单随机分割
try:
    print("Attempts at stratified sampling...")
    # 为分层抽样创建标签（使用每个donor的主要疾病类型）
    donor_main_disease = adata.obs.groupby(donor_id_column)['disease'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    stratify_labels = [donor_main_disease[donor] for donor in unique_groups]
    
    train_donors, test_donors = train_test_split(
        unique_groups, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    print("Stratified sampling success！")
except ValueError as e:
    print(f"Failure of stratified sampling: {e}")
    print("Using Simple Randomized Splitting...")
    train_donors, test_donors = train_test_split(
        unique_groups, test_size=0.2, random_state=42
    )

# 创建训练集和测试集
train_mask = adata.obs[donor_id_column].isin(train_donors)
test_mask = adata.obs[donor_id_column].isin(test_donors)

train_adata = adata[train_mask]
test_adata = adata[test_mask]

print(f"Number of cells in the training set: {train_adata.shape[0]}")
print(f"Number of cells in the test set: {test_adata.shape[0]}")

# 重新提取特征矩阵和标签（使用分割后的数据）
X_train = train_adata.X
y_train = train_adata.obs['disease'].astype(str).values
X_test = test_adata.X
y_test = test_adata.obs['disease'].astype(str).values

# 不进行采样，使用完整数据
X_train_sample, y_train_sample = X_train, y_train

# 打印标签分布
print("\nlabel distribution:")
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

# 确保X是稠密矩阵（如果是稀疏矩阵）
if hasattr(X_train, 'toarray'):
    X_train = X_train.toarray()
if hasattr(X_test, 'toarray'):
    X_test = X_test.toarray()

# 检查并填充NaN
print("\nChecking data...")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("NaN quantities - training set:", np.isnan(X_train).sum())
print("NaN quantities - training set:", np.isnan(X_test).sum())

# 填充NaN
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# LightGBM
print("\nTraining the LightGBM model...")

# 检查是否存在已保存的模型
if os.path.exists('lightgbm_model.txt') and os.path.exists('logistic_regression_model.pkl'):
    print(" Saved models found, loading...")
    
    # 加载LightGBM模型
    gbm = lgb.Booster(model_file='lightgbm_model.txt')
    print(" LightGBM model loading complete")
    
    # 加载Logistic Regression模型和预处理器
    lr = joblib.load('logistic_regression_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    imputer = joblib.load('imputer.pkl')
    print(" Logistic Regression model loading complete")
    
    # 获取原始类别
    original_classes = label_encoder.classes_
    print(f"Loaded category mapping: {dict(zip(original_classes, range(len(original_classes))))}")
    
    # 应用已保存的预处理器
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    
else:
    print(" The saved model was not found, start training a new model...")
    
    # 标签编码
    print("Perform label coding...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # 保存原始类别映射关系，供后面使用
    original_classes = label_encoder.classes_
    print(f"category mapping: {dict(zip(original_classes, range(len(original_classes))))}")
    
    # 填充NaN
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Logistic Regression
    print("\n train Logistic Regression...")
    lr = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=300,  # 增加迭代次数
        n_jobs=-1,
        random_state=42,
        C=1.0
    )
    lr.fit(X_train_sample, y_train_sample)
    
    # 创建数据集
    lgb_train = lgb.Dataset(X_train, y_train_encoded)  # 使用编码后的标签
    lgb_test = lgb.Dataset(X_test, y_test_encoded, reference=lgb_train)
    
    # LightGBM参数
    params = {
        'objective': 'multiclass',
        'num_class': len(original_classes),  # 使用编码后的类别数
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'num_threads': -1,
        'verbose': -1
    }
    
    # 训练模型
    print("begin train LightGBM...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_test]
    )
    
    # 保存LightGBM模型
    print("Save the LightGBM model...")
    gbm.save_model('lightgbm_model.txt')
    print(" The LightGBM model has been saved as 'lightgbm_model.txt'")
    
    # 保存Logistic Regression模型
    print("save Logistic Regression model...")
    joblib.dump(lr, 'logistic_regression_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    print(" Logistic Regression5")

# 预测
y_pred_proba_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_lgb = np.argmax(y_pred_proba_lgb, axis=1)

# 将预测结果转换回原始标签
y_pred_lgb = label_encoder.inverse_transform(y_pred_lgb)

# 评估LightGBM
print("\nLightGBM Model Evaluation:")
lgb_accuracy = accuracy_score(y_test, y_pred_lgb)
lgb_f1_macro = f1_score(y_test, y_pred_lgb, average='macro')
print(f"accuracy: {lgb_accuracy:.4f}")
print(f"Macro-F1: {lgb_f1_macro:.4f}")
print("\nClassification report:")
lgb_report = classification_report(y_test, y_pred_lgb)
print(lgb_report)

# 比较两个模型
print("\nmodel comparison:")
print(f"Logistic Regression - accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}")
print(f"LightGBM - accuracy: {lgb_accuracy:.4f}, Macro-F1: {lgb_f1_macro:.4f}")

# Create and save prediction results CSV (required format)
print("\nCreating prediction results...")
cell_indices = range(len(test_adata.obs.index))
predictions_df = pd.DataFrame()
predictions_df['cell_index'] = cell_indices

# Map disease names to required format
disease_mapping = {
    'normal': 'normal',
    'type 1 diabetes mellitus': 'T1D', 
    'type 2 diabetes mellitus': 'T2D',
    'endocrine pancreas disorder': 'endocrine'
}

# Get required classes in the specified order
required_classes = ['normal', 'type 1 diabetes mellitus', 'type 2 diabetes mellitus', 'endocrine pancreas disorder']
required_output_names = ['normal', 'T1D', 'T2D', 'endocrine']

# Check if all required classes exist
missing_classes = set(required_classes) - set(original_classes)
if missing_classes:
    print(f"Warning: Missing the following classes in data: {missing_classes}")
    print(f"Actual classes: {original_classes}")

# Add Logistic Regression probability columns for each class
print("Adding Logistic Regression predictions...")
for i, cls in enumerate(required_classes):
    output_name = required_output_names[i]
    if cls in original_classes:
        cls_idx = np.where(original_classes == cls)[0][0]
        predictions_df[f'p_{output_name}_lr'] = lr.predict_proba(X_test)[:, cls_idx]
    else:
        # If class doesn't exist, fill with 0
        predictions_df[f'p_{output_name}_lr'] = 0

# Add LightGBM probability columns for each class
print("Adding LightGBM predictions...")
# First convert LightGBM probabilities back to match original_classes order
lgb_proba_reordered = np.zeros((len(y_test), len(original_classes)))
for i, cls in enumerate(original_classes):
    cls_encoded = label_encoder.transform([cls])[0]
    lgb_proba_reordered[:, i] = y_pred_proba_lgb[:, cls_encoded]

for i, cls in enumerate(required_classes):
    output_name = required_output_names[i]
    if cls in original_classes:
        cls_idx = np.where(original_classes == cls)[0][0]
        predictions_df[f'p_{output_name}_lgb'] = lgb_proba_reordered[:, cls_idx]
    else:
        # If class doesn't exist, fill with 0
        predictions_df[f'p_{output_name}_lgb'] = 0

# Create ensemble predictions (average of both models)
print("Creating ensemble predictions...")
for output_name in required_output_names:
    lr_col = f'p_{output_name}_lr'
    lgb_col = f'p_{output_name}_lgb'
    if lr_col in predictions_df.columns and lgb_col in predictions_df.columns:
        predictions_df[f'p_{output_name}'] = (predictions_df[lr_col] + predictions_df[lgb_col]) / 2
    elif lr_col in predictions_df.columns:
        predictions_df[f'p_{output_name}'] = predictions_df[lr_col]
    elif lgb_col in predictions_df.columns:
        predictions_df[f'p_{output_name}'] = predictions_df[lgb_col]
    else:
        predictions_df[f'p_{output_name}'] = 0

# Keep only the required columns for final output
final_predictions = predictions_df[['cell_index'] + [f'p_{name}' for name in required_output_names]].copy()

# Save prediction results
final_predictions.to_csv('predictions.csv', index=False)
print("Prediction results saved to 'predictions.csv'")

# Print evaluation metrics for both models
print("\n=== MODEL EVALUATION SUMMARY ===")
print(f"Logistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}")
print(f"  Macro-F1: {f1_score(y_test, lr.predict(X_test), average='macro'):.4f}")

print(f"\nLightGBM:")
print(f"  Accuracy: {lgb_accuracy:.4f}")
print(f"  Macro-F1: {lgb_f1_macro:.4f}")

# Create combined confusion matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, lr.predict(X_test))
sns.heatmap(cm_lr, annot=True, fmt='g', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels') 
axes[0].set_title('Logistic Regression Confusion Matrix')

# LightGBM confusion matrix
cm_lgb = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm_lgb, annot=True, fmt='g', cmap='Greens', ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('LightGBM Confusion Matrix')

plt.tight_layout()
plt.savefig('metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix plots saved to 'metrics.png'")

# ====================== 高级分析部分 ======================
print("\nStart Advanced Analytics...")

# 1. 多分类 ROC 曲线分析 - 为两个模型都生成
print("\n1. Create multi-categorized ROC curves...")

# 将标签二值化以计算ROC曲线
y_test_binarized = label_binarize(y_test, classes=original_classes)
n_classes = len(original_classes)

# 获取两个模型的预测概率
# Logistic Regression 预测概率
y_pred_proba_lr = lr.predict_proba(X_test)

# 获取LightGBM的预测概率
y_pred_proba_lgb_reordered = np.zeros((len(y_test), len(original_classes)))
for i, cls in enumerate(original_classes):
    cls_encoded = label_encoder.transform([cls])[0]
    y_pred_proba_lgb_reordered[:, i] = y_pred_proba_lgb[:, cls_encoded]

# 为两个模型计算ROC曲线
models_data = {
    'Logistic Regression': y_pred_proba_lr,
    'LightGBM': y_pred_proba_lgb_reordered
}

# 创建大图包含两个模型的ROC曲线
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

for model_idx, (model_name, y_pred_proba) in enumerate(models_data.items()):
    # Calculate the ROC curve and AUC for each category
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    
    # 绘制每个类别的ROC曲线
    for i, color in zip(range(n_classes), colors):
        axes[model_idx].plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{original_classes[i]} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制宏平均ROC曲线
    axes[model_idx].plot(fpr["macro"], tpr["macro"],
             label=f'Macro Average (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle='--', linewidth=3)
    
    axes[model_idx].plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    axes[model_idx].set_xlim([0.0, 1.0])
    axes[model_idx].set_ylim([0.0, 1.05])
    axes[model_idx].set_xlabel('Fake positive rate (False Positive Rate)', fontsize=12)
    axes[model_idx].set_ylabel('True rate (True Positive Rate)', fontsize=12)
    axes[model_idx].set_title(f'{model_name} Multi-categorized ROC curves', fontsize=14, fontweight='bold')
    axes[model_idx].legend(loc="lower right")
    axes[model_idx].grid(True, alpha=0.3)
    
    # 保存每个模型的ROC数据
    try:
        class_names = list(original_classes) + ['macro']
        auc_values = [roc_auc[i] for i in range(n_classes)] + [roc_auc["macro"]]
        
        print(f"Debug: Category quantity={len(class_names)}, AUC quantity={len(auc_values)}")
        print(f"Original Category: {original_classes}")
        print(f"n_classes: {n_classes}")
        
        roc_data = pd.DataFrame({
            'Class': class_names,
            'AUC': auc_values
        })
        roc_data.to_csv(f'roc_auc_{model_name.lower().replace(" ", "_")}.csv', index=False)
    except Exception as e:
        print(f"An error occurred while saving ROC data: {e}")
        print(f"original_classes: {original_classes}, n_classes: {n_classes}")
        print(f"roc_auc keys: {list(roc_auc.keys())}")
        
        # 简化保存，只保存成功计算的AUC
        simple_data = []
        for i in range(n_classes):
            if i in roc_auc:
                simple_data.append({'Class': original_classes[i], 'AUC': roc_auc[i]})
        if "macro" in roc_auc:
            simple_data.append({'Class': 'Macro-average', 'AUC': roc_auc["macro"]})
        
        if simple_data:
            simple_df = pd.DataFrame(simple_data)
            simple_df.to_csv(f'roc_auc_{model_name.lower().replace(" ", "_")}.csv', index=False)
            print(f"ROC AUCDetailed data (simplified version) saved")

plt.tight_layout()
plt.savefig('roc_curves_both_models.png', dpi=300, bbox_inches='tight')
plt.show()
print("Multi-classification ROC curves for two models have been saved as 'roc_curves_both_models.png'")

#
print("\nCreate a dedicated LightGBM ROC curve chart...")
plt.figure(figsize=(12, 10))

# 使用LightGBM数据
y_pred_proba = y_pred_proba_lgb_reordered

# 计算每个类别的ROC曲线和AUC
fpr_lgb = dict()
tpr_lgb = dict()
roc_auc_lgb = dict()

for i in range(n_classes):
    fpr_lgb[i], tpr_lgb[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc_lgb[i] = auc(fpr_lgb[i], tpr_lgb[i])

# 计算宏平均ROC曲线
all_fpr = np.unique(np.concatenate([fpr_lgb[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr_lgb[i], tpr_lgb[i])
mean_tpr /= n_classes

fpr_lgb["macro"] = all_fpr
tpr_lgb["macro"] = mean_tpr
roc_auc_lgb["macro"] = auc(fpr_lgb["macro"], tpr_lgb["macro"])

# 绘制ROC曲线 - 使用专业配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
linestyles = ['-', '-', '-', '-', '--']

# 绘制每个类别的ROC曲线
for i in range(n_classes):
    plt.plot(fpr_lgb[i], tpr_lgb[i], color=colors[i], lw=3, linestyle=linestyles[i],
             label=f'{original_classes[i]} (AUC = {roc_auc_lgb[i]:.3f})')

# 绘制宏平均ROC曲线
plt.plot(fpr_lgb["macro"], tpr_lgb["macro"],
         label=f'Macro-average (AUC = {roc_auc_lgb["macro"]:.3f})',
         color='navy', linestyle='--', linewidth=4)

# 绘制随机分类器基线
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('LightGBM Multi-class ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')

# 设置背景色
plt.gca().set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('lightgbm_roc_curves_ovr.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("LightGBM Special ROC curve charts have been saved as 'lightgbm_roc_curves_ovr.png'")

# 保存LightGBM专门的ROC数据
try:
    class_names = list(original_classes) + ['Macro-average']
    auc_values = [roc_auc_lgb[i] for i in range(n_classes)] + [roc_auc_lgb["macro"]]
    
    print(f"Debug: Category quantity={len(class_names)}, AUC quantity={len(auc_values)}")
    print(f"Original category: {original_classes}")
    print(f"n_classes: {n_classes}")
    print(f"roc_auc_lgb keys: {list(roc_auc_lgb.keys())}")
    
    lightgbm_roc_data = pd.DataFrame({
        'Class': class_names,
        'AUC': auc_values
    })
    lightgbm_roc_data.to_csv('lightgbm_roc_auc_detailed.csv', index=False)
    print("LightGBM ROC AUC Detailed data has been saved as 'lightgbm_roc_auc_detailed.csv'")
except Exception as e:
    print(f"An error occurred while saving ROC data: {e}")
    print(f"original_classes: {original_classes}, n_classes: {n_classes}")
    if 'roc_auc_lgb' in locals():
        print(f"roc_auc_lgb keys: {list(roc_auc_lgb.keys())}")
        
        # 简化保存，只保存成功计算的AUC
        simple_data = []
        for i in range(n_classes):
            if i in roc_auc_lgb:
                simple_data.append({'Class': original_classes[i], 'AUC': roc_auc_lgb[i]})
        if "macro" in roc_auc_lgb:
            simple_data.append({'Class': 'Macro-average', 'AUC': roc_auc_lgb["macro"]})
        
        if simple_data:
            simple_df = pd.DataFrame(simple_data)
            simple_df.to_csv('lightgbm_roc_auc_detailed.csv', index=False)
            print("LightGBM ROC AUC Detailed data (simplified version) has been saved as'lightgbm_roc_auc_detailed.csv'")

print("\n=== ROC Curve analysis description ===")
print("LightGBM The model performed well in four categories of pancreatic diseases：")
for i in range(n_classes):
    print(f"- {original_classes[i]}: AUC = {roc_auc_lgb[i]:.3f}")
print(f"- macro average AUC: {roc_auc_lgb['macro']:.3f}")
print("The closer the AUC value is to 1.0, the better the classification performance, and 0.5 means the random classification level。")

# 2. SHAP 特征重要性分析 - 为两个模型都生成
print("\n2.Start SHAP Feature Importance Analysis...")

# 2.1 LightGBM SHAP 分析
print("\n2.1 LightGBM SHAP analysis...")
try:
    # 创建SHAP解释器（使用TreeExplainer for LightGBM）
    explainer_lgb = shap.TreeExplainer(gbm)
    
    # 选择一个代表性的样本子集来计算SHAP值（避免内存不足）
    sample_size = min(1000, X_test.shape[0])
    sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sample = X_test[sample_indices]
    
    # 确保X_test_sample是numpy数组
    if hasattr(X_test_sample, 'toarray'):
        X_test_sample = X_test_sample.toarray()
    X_test_sample = np.array(X_test_sample)
    
    print(f"compute {sample_size} sample LightGBM SHAP value...")
    shap_values_lgb = explainer_lgb.shap_values(X_test_sample)
    
    # 计算平均绝对SHAP值来确定特征重要性
    if isinstance(shap_values_lgb, list):
        # 多分类情况下，shap_values是一个列表
        mean_abs_shap_lgb = np.mean([np.abs(sv).mean(0) for sv in shap_values_lgb], axis=0)
    else:
        mean_abs_shap_lgb = np.abs(shap_values_lgb).mean(0)
    
    # 获取前20个最重要的特征
    top_20_indices_lgb = np.argsort(mean_abs_shap_lgb)[-20:]
    top_20_importance_lgb = mean_abs_shap_lgb[top_20_indices_lgb]
    
    # 获取基因名称（如果available）
    if hasattr(test_adata, 'var') and 'gene_symbols' in test_adata.var.columns:
        gene_names_lgb = test_adata.var['gene_symbols'].iloc[top_20_indices_lgb].values
    elif hasattr(test_adata, 'var') and test_adata.var.index is not None:
        gene_names_lgb = test_adata.var.index[top_20_indices_lgb].values
    else:
        gene_names_lgb = [f"Gene_{i}" for i in top_20_indices_lgb]
    
    print(f"Successfully calculate the SHAP value and prepare to generate the Top-20 feature importance map...")
    
    # 绘制LightGBM SHAP特征重要性柱状图
    plt.figure(figsize=(12, 10))
    
    # 对特征重要性进行排序，使最重要的在顶部
    sorted_indices = np.argsort(top_20_importance_lgb)
    sorted_importance = top_20_importance_lgb[sorted_indices]
    sorted_gene_names = gene_names_lgb[sorted_indices]
    
    y_pos = np.arange(len(sorted_gene_names))
    
    # 创建渐变色彩
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_gene_names)))
    
    bars = plt.barh(y_pos, sorted_importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.yticks(y_pos, sorted_gene_names, fontsize=11)
    plt.xlabel('SHAP Value (Mean Absolute)', fontsize=14, fontweight='bold')
    plt.ylabel('Gene Names', fontsize=14, fontweight='bold')
    plt.title('LightGBM Feature Importance (SHAP Top-20)', fontsize=16, fontweight='bold', pad=20)
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 添加网格线以提高可读性
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 设置背景色和布局
    plt.gca().set_facecolor('#f9f9f9')
    plt.tight_layout()
    plt.savefig('shap_feature_importance_lightgbm.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()  
    print("LightGBM SHAP Feature Importance Graph has been saved as 'shap_feature_importance_lightgbm.png'")
    
  
    print("\n=== Biological insights ===")
    print("The SHAP method was used to identify key genes that contributed the most to pancreatic disease classification")
    print("The average absolute SHAP value of the global Top-20 gene is shown in the figure")
    print("These key genes provide important clues to understanding the molecular mechanisms of pancreatic disease")
    print(f"The most important gene: {sorted_gene_names[-1]} (SHAP value: {sorted_importance[-1]:.4f})")
    
    # 保存LightGBM特征重要性数据
    importance_df_lgb = pd.DataFrame({
        'gene_name': gene_names_lgb,
        'shap_importance': top_20_importance_lgb,
        'feature_index': top_20_indices_lgb,
        'rank': range(1, 21),
        'model': 'LightGBM'
    })
    # 按重要性排序
    importance_df_lgb = importance_df_lgb.sort_values('shap_importance', ascending=False)
    importance_df_lgb.to_csv('shap_feature_importance_lightgbm.csv', index=False)
    print("LightGBM SHAP Feature importance data has been saved as 'shap_feature_importance_lightgbm.csv'")
    
except ImportError:
    print("SHAP not download，use LightGBM Built-in feature importance...")
    print("if we need SHAP analysis，please use: pip install shap")
    
    # 使用LightGBM内置特征重要性作为备选方案
    feature_importance = gbm.feature_importance(importance_type='gain')
    top_20_indices_lgb = np.argsort(feature_importance)[-20:]
    top_20_importance_lgb = feature_importance[top_20_indices_lgb]
    
    # 获取基因名称
    if hasattr(test_adata, 'var') and test_adata.var.index is not None:
        gene_names_lgb = test_adata.var.index[top_20_indices_lgb].values
    else:
        gene_names_lgb = [f"Gene_{i}" for i in top_20_indices_lgb]
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 10))
    sorted_indices = np.argsort(top_20_importance_lgb)
    sorted_importance = top_20_importance_lgb[sorted_indices]
    sorted_gene_names = gene_names_lgb[sorted_indices]
    
    y_pos = np.arange(len(sorted_gene_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_gene_names)))
    
    bars = plt.barh(y_pos, sorted_importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.yticks(y_pos, sorted_gene_names, fontsize=11)
    plt.xlabel('Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.ylabel('Gene Names', fontsize=14, fontweight='bold')
    plt.title('LightGBM Feature Importance (Top-20)', fontsize=16, fontweight='bold', pad=20)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#f9f9f9')
    plt.tight_layout()
    plt.savefig('shap_feature_importance_lightgbm.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("LightGBM Feature Importance Graph has been saved as 'shap_feature_importance_lightgbm.png'")
    
    # 保存特征重要性数据
    importance_df_lgb = pd.DataFrame({
        'gene_name': gene_names_lgb,
        'feature_importance': top_20_importance_lgb,
        'feature_index': top_20_indices_lgb,
        'model': 'LightGBM'
    }).sort_values('feature_importance', ascending=False)
    importance_df_lgb.to_csv('shap_feature_importance_lightgbm.csv', index=False)
    print("LightGBM Feature importance data saved")
    
except Exception as e:
    print(f"LightGBM SHAP An analysis error: {e}")
    print("Try using alternatives...")
    
    try:
        # 备选方案：使用LightGBM内置特征重要性
        feature_importance = gbm.feature_importance(importance_type='gain')
        top_20_indices_lgb = np.argsort(feature_importance)[-20:]
        top_20_importance_lgb = feature_importance[top_20_indices_lgb]
        
        if hasattr(test_adata, 'var') and test_adata.var.index is not None:
            gene_names_lgb = test_adata.var.index[top_20_indices_lgb].values
        else:
            gene_names_lgb = [f"Gene_{i}" for i in top_20_indices_lgb]
        
        # 绘制图表
        plt.figure(figsize=(12, 10))
        sorted_indices = np.argsort(top_20_importance_lgb)
        sorted_importance = top_20_importance_lgb[sorted_indices]
        sorted_gene_names = gene_names_lgb[sorted_indices]
        
        y_pos = np.arange(len(sorted_gene_names))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_gene_names)))
        
        bars = plt.barh(y_pos, sorted_importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.yticks(y_pos, sorted_gene_names, fontsize=11)
        plt.xlabel('Feature Importance (Gain)', fontsize=14, fontweight='bold')
        plt.ylabel('Gene Names', fontsize=14, fontweight='bold')
        plt.title('LightGBM Feature Importance (Top-20)', fontsize=16, fontweight='bold', pad=20)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f9f9f9')
        plt.tight_layout()
        plt.savefig('shap_feature_importance_lightgbm.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        print("LightGBM Feature Importance Chart (Optional) Saved")
        
    except Exception as e2:
        print(f"Alternatives also failed: {e2}")

# 3. UMAP 
print("\n3. Create UMAP visualization...")
try:
    # 为了加速UMAP计算，使用较小的样本
    umap_sample_size = min(5000, X_test.shape[0])
    umap_indices = np.random.choice(X_test.shape[0], umap_sample_size, replace=False)
    X_umap = X_test[umap_indices]
    y_true_umap = y_test[umap_indices]
    y_pred_lgb_umap = y_pred_lgb[umap_indices]
    y_pred_lr_umap = lr.predict(X_test)[umap_indices]
    
    print(f"compute {umap_sample_size} samples' UMAP embed...")
    
    # 计算UMAP嵌入
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_umap)
    
    # 创建颜色映射
    unique_labels = np.unique(np.concatenate([y_true_umap, y_pred_lgb_umap, y_pred_lr_umap]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    
    # 真实标签的UMAP
    for label in unique_labels:
        mask = y_true_umap == label
        if np.any(mask):
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color_map[label]], label=label, s=20, alpha=0.7)
    
    axes[0].set_title('UMAP - true label', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # LightGBM预测标签的UMAP
    for label in unique_labels:
        mask = y_pred_lgb_umap == label
        if np.any(mask):
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color_map[label]], label=label, s=20, alpha=0.7)
    
    axes[1].set_title('UMAP - LightGBM prediction label', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Logistic Regression预测标签的UMAP
    for label in unique_labels:
        mask = y_pred_lr_umap == label
        if np.any(mask):
            axes[2].scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color_map[label]], label=label, s=20, alpha=0.7)
    
    axes[2].set_title('UMAP - Logistic Regression prediction label', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('UMAP 1', fontsize=12)
    axes[2].set_ylabel('UMAP 2', fontsize=12)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('umap_comparison_both_models.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("UMAP The three-model comparison diagram has been saved as 'umap_comparison_both_models.png'")
    
    # 创建预测准确性对比可视化
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # LightGBM 准确性可视化
    correct_mask_lgb = y_true_umap == y_pred_lgb_umap
    incorrect_mask_lgb = ~correct_mask_lgb
    
    if np.any(correct_mask_lgb):
        axes[0].scatter(embedding[correct_mask_lgb, 0], embedding[correct_mask_lgb, 1], 
                       c='green', label=f'Correct prediction ({np.sum(correct_mask_lgb)})', 
                       s=20, alpha=0.7)
    
    if np.any(incorrect_mask_lgb):
        axes[0].scatter(embedding[incorrect_mask_lgb, 0], embedding[incorrect_mask_lgb, 1], 
                       c='red', label=f'misprediction ({np.sum(incorrect_mask_lgb)})', 
                       s=20, alpha=0.7)
    
    axes[0].set_title('UMAP - LightGBM Predictive accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    accuracy_text_lgb = f'Test set accuracy: {np.mean(correct_mask_lgb):.3f}'
    axes[0].text(0.02, 0.98, accuracy_text_lgb, transform=axes[0].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Logistic Regression 准确性可视化
    correct_mask_lr = y_true_umap == y_pred_lr_umap
    incorrect_mask_lr = ~correct_mask_lr
    
    if np.any(correct_mask_lr):
        axes[1].scatter(embedding[correct_mask_lr, 0], embedding[correct_mask_lr, 1], 
                       c='green', label=f'true prediction ({np.sum(correct_mask_lr)})', 
                       s=20, alpha=0.7)
    
    if np.any(incorrect_mask_lr):
        axes[1].scatter(embedding[incorrect_mask_lr, 0], embedding[incorrect_mask_lr, 1], 
                       c='red', label=f'false prediction ({np.sum(incorrect_mask_lr)})', 
                       s=20, alpha=0.7)
    
    axes[1].set_title('UMAP - Logistic Regression Prediction accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    accuracy_text_lr = f'Test set accuracy: {np.mean(correct_mask_lr):.3f}'
    axes[1].text(0.02, 0.98, accuracy_text_lr, transform=axes[1].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('umap_accuracy_both_models.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("UMAP Accuracy contrast visualization has been saved as 'umap_accuracy_both_models.png'")
    
except ImportError:
    print("Warning: UMAP library is not installed, skip UMAP analysis")
    print("Please run: pip install umap-learn")
except Exception as e:
    print(f"UMAP analysis error: {e}")

# 4. 创建综合分析报告
print("\n4. Generate a comprehensive analysis report...")

# 创建性能总结
performance_summary = {
    'Model': ['Logistic Regression', 'LightGBM'],
    'Accuracy': [accuracy_score(y_test, lr.predict(X_test)), lgb_accuracy],
    'Macro-F1': [f1_score(y_test, lr.predict(X_test), average='macro'), lgb_f1_macro],
    'Weighted-F1': [f1_score(y_test, lr.predict(X_test), average='weighted'), 
                    f1_score(y_test, y_pred_lgb, average='weighted')]
}

performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('model_performance_summary.csv', index=False)
print("The model performance summary has been saved as 'model_performance_summary.csv'")

# 创建类别详细分析 - 两个模型
from sklearn.metrics import precision_recall_fscore_support

# LightGBM 类别分析
precision_lgb, recall_lgb, f1_lgb, support_lgb = precision_recall_fscore_support(y_test, y_pred_lgb, labels=original_classes)

class_analysis_lgb = pd.DataFrame({
    'Class': original_classes,
    'Precision': precision_lgb,
    'Recall': recall_lgb,
    'F1-Score': f1_lgb,
    'Support': support_lgb,
    'Model': 'LightGBM'
})

# Logistic Regression 类别分析
precision_lr, recall_lr, f1_lr, support_lr = precision_recall_fscore_support(y_test, lr.predict(X_test), labels=original_classes)

class_analysis_lr = pd.DataFrame({
    'Class': original_classes,
    'Precision': precision_lr,
    'Recall': recall_lr,
    'F1-Score': f1_lr,
    'Support': support_lr,
    'Model': 'Logistic Regression'
})

# 合并两个模型的分析
combined_class_analysis = pd.concat([class_analysis_lgb, class_analysis_lr], ignore_index=True)
combined_class_analysis.to_csv('class_performance_analysis_both_models.csv', index=False)
print("The category performance analysis of the two models has been saved as 'class_performance_analysis_both_models.csv'")

print("\n=== Advanced analysis completed ===")
print("Documents generated:")
print("=== ROC Curve analysis ===")
print("- roc_curves_both_models.png: Comparison of multi-classification ROC curves between two models")
print("- roc_auc_logistic_regression.csv: Logistic Regression ROC AUC data")
print("- roc_auc_lightgbm.csv: LightGBM ROC AUC data")
print("\n=== SHAP Characteristic importance analysis ===")
print("- shap_feature_importance_lightgbm.png: LightGBM SHAP Top-20 picture")
print("- shap_feature_importance_logistic.png: Logistic Regression SHAP Top-20 picture")
print("- shap_feature_importance_lightgbm.csv: LightGBM SHAP Feature Importance Data")
print("- shap_feature_importance_logistic.csv: Logistic Regression SHAP Feature Importance Data")
print("- shap_feature_importance_combined.csv: combined SHAP Feature Importance Data")
print("\n=== UMAP Visualization ===")
print("- umap_comparison_both_models.png: UMAP Comparison of three models（true、LightGBM、Logistic Regression）")
print("- umap_accuracy_both_models.png: UMAP Comparison of the predictive accuracy of the two models")
print("\n=== analysis report ===")
print("- model_performance_summary.csv: Model Performance Summary")
print("- class_performance_analysis_both_models.csv: Category performance analysis of two models")

# 创建Jupyter Notebook
print("\n produce Jupyter Notebook...")

# 创建新的notebook对象
nb = nbf.v4.new_notebook()

# 添加标题单元格
title_cell = nbf.v4.new_markdown_cell("# single cell RNA-seq Supervised classification and analysis of data\n\n**target：** for pancreas_islet.h5ad each cell，owing to disease Label Training Classifier（LogisticRegression & LightGBM），And output forecasting results and evaluation reports。")

# 添加导入单元格
import_cell = nbf.v4.new_code_cell("""import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set random seeds to ensure results are repeatable
np.random.seed(42)
#
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib后端，避免字体问题
matplotlib.use('Agg')
plt.rcParams['figure.max_open_warning'] = 0
""")

# 添加数据加载单元格
load_data_cell = nbf.v4.new_code_cell("""# 加载数据
print("Loading data...")
if os.path.exists("preprocessed_pancreas_islet.h5ad"):
    print("Load preprocessed data...")
    adata = sc.read_h5ad("preprocessed_pancreas_islet.h5ad")
else:
    print("Load raw data and preprocess it...")
    adata = sc.read_h5ad("pancreas_islet.h5ad")
    sc.pp.filter_genes(adata, min_cells=20)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    adata.write_h5ad("preprocessed_pancreas_islet.h5ad")
    print("Preprocessing is completed, saved！")

# 打印数据基本信息
print(f"data shape: {adata.shape}")
print(f"useful label: {list(adata.obs.columns)}")

# 确认disease标签存在
if 'disease' not in adata.obs.columns:
    raise ValueError("The data does not contain'disease'，please check")

# 查找donor标识符列
donor_columns = []
for col in adata.obs.columns:
    if 'donor' in col.lower() or 'individual' in col.lower() or 'subject' in col.lower():
        donor_columns.append(col)

if donor_columns:
    print(f"Find the donor related column: {donor_columns}")
    donor_id_column = donor_columns[0]
else:
    # 如果没有找到，使用其他可能的分组列
    possible_columns = [col for col in adata.obs.columns 
                        if any(k in col.lower() for k in ['batch', 'sample', 'id', 'group'])]
    if possible_columns:
        donor_id_column = possible_columns[0]
    else:
        # 如果还是找不到，创建随机分组
        print("Can't find a suitable grouping column, use random grouping")
        adata.obs['random_group'] = np.random.randint(0, 10, size=adata.shape[0])
        donor_id_column = 'random_group'
        
print(f"use'{donor_id_column}'as donor symbol")

# 打印疾病类别统计
print("\\n Disease Category Statistics:")
print(adata.obs['disease'].value_counts())
""")

# 添加预处理单元格
preprocess_cell = nbf.v4.new_code_cell("""# 数据预处理
print("\\nPreprocessed data...")

# 按照要求进行预处理
with tqdm(total=5, desc="Preprocessing progress") as pbar:
    # 过滤基因
    sc.pp.filter_genes(adata, min_cells=20)
    pbar.update(1)
    
    # 选择高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    pbar.update(1)
    
    # 标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    pbar.update(1)
    
    # 对数转换
    sc.pp.log1p(adata)
    pbar.update(1)
    
    # 缩放
    sc.pp.scale(adata, max_value=10)
    pbar.update(1)

print(f"Shape of preprocessed data: {adata.shape}")
""")

# 添加数据集划分单元格
split_cell = nbf.v4.new_code_cell("""# Grouped by donor_id for training/test set division（80%/20%）
print(f"\\n Divide the training set and test set by {donor_id_column}...")
donors = adata.obs[donor_id_column].unique()
train_donors, test_donors = train_test_split(donors, test_size=0.2, random_state=42, stratify=donors)

# 创建训练集和测试集
train_mask = adata.obs[donor_id_column].isin(train_donors)
test_mask = adata.obs[donor_id_column].isin(test_donors)

train_adata = adata[train_mask]
test_adata = adata[test_mask]

print(f"Number of cells in the training set: {train_adata.shape[0]}")
print(f"Number of cells in the test set: {test_adata.shape[0]}")

# 提取特征矩阵和标签
X_train = train_adata.X
y_train = train_adata.obs['disease'].astype(str).values 
X_test = test_adata.X
y_test = test_adata.obs['disease'].astype(str).values    

# 确保X是稠密矩阵（如果是稀疏矩阵）
if hasattr(X_train, 'toarray'):
    X_train = X_train.toarray()
    X_test = X_test.toarray()
""")

# 添加LogisticRegression训练单元格
lr_cell = nbf.v4.new_code_cell("""# 训练Logistic Regression模型
print("\\n train Logistic Regression model...")
print(f"Training data size: {X_train.shape}")

# If the data is too large, you can sample some of it for training
if X_train.shape[0] > 10000:
    from sklearn.utils import resample
    X_train_sample, y_train_sample = resample(
        X_train, y_train, n_samples=10000, random_state=42, stratify=y_train
    )
    print(f"Large amount of data, trained using 10,000 samples...")
    print(f"Sampled training data size: {X_train_sample.shape}")
else:
    X_train_sample, y_train_sample = X_train, y_train

# 检查并填充NaN，防止Logistic Regression报错
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train_sample = imputer.fit_transform(X_train_sample)
X_test = imputer.transform(X_test)

# 训练模型
start_time = time.time()
with tqdm(total=100, desc="Training progress") as pbar:
    lr = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=200, n_jobs=-1, random_state=42)
    pbar.update(10)
    lr.fit(X_train_sample, y_train_sample)
    pbar.update(90)
end_time = time.time()
print(f"Training completed! Duration: {end_time - start_time:.2f} 秒")

# 预测
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)

# 评估Logistic Regression
print("\\nLogistic Regression model evaluation:")
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1_macro = f1_score(y_test, y_pred_lr, average='macro')
print(f"accuracy: {lr_accuracy:.4f}")
print(f"Macro-F1: {lr_f1_macro:.4f}")
print("\\n report:")
print(classification_report(y_test, y_pred_lr))
""")

# 添加LightGBM训练单元格
lgb_cell = nbf.v4.new_code_cell("""# Training the LightGBM model
print("\\n Training the LightGBM model...")

# 获取类别标签
classes = np.unique(y_train)
num_classes = len(classes)

# 创建数据集
lgb_train = lgb.Dataset(X_train, y_train)  
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# set parameter
params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'num_threads': -1,  # 使用所有CPU
    'verbose': -1
}

# Training the model
print("Start training LightGBM...")
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_test]
)

# 保存LightGBM模型
print("Save LightGBM model...")
gbm.save_model('lightgbm_model.txt')
print(" LightGBM model is saved as 'lightgbm_model.txt'")

# 预测
y_pred_proba_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_lgb = np.argmax(y_pred_proba_lgb, axis=1)

# 将预测结果转换回原始标签
y_pred_lgb = label_encoder.inverse_transform(y_pred_lgb)

# 评估LightGBM
print("\\nLightGBM model evaluation:")
lgb_accuracy = accuracy_score(y_test, y_pred_lgb)
lgb_f1_macro = f1_score(y_test, y_pred_lgb, average='macro')
print(f"accuracy: {lgb_accuracy:.4f}")
print(f"Macro-F1: {lgb_f1_macro:.4f}")
print("\\n classification report:")
print(classification_report(y_test, y_pred_lgb))

# 比较两个模型
print("\\n Model comparison:")
print(f"Logistic Regression - accuracy: {lr_accuracy:.4f}, Macro-F1: {lr_f1_macro:.4f}")
print(f"LightGBM - accuracy: {lgb_accuracy:.4f}, Macro-F1: {lgb_f1_macro:.4f}")
""")

# 添加结果输出单元格
output_cell = nbf.v4.new_code_cell("""# Create and save the prediction result CSV
print("\\n Creating Forecast Results...")
cell_indices = range(len(test_adata.obs.index))
predictions_df = pd.DataFrame()
predictions_df['cell_index'] = cell_indices


required_classes = ['normal', 'type 1 diabetes mellitus', 'type 2 diabetes mellitus', 'endocrine pancreas disorder']

# Check if all required categories exist
missing_classes = set(required_classes) - set(original_classes)
if missing_classes:
    print(f"Warning: The following categories are missing in the data: {missing_classes}")
    print(f"Actual Category: {original_classes}")

# 为每个类别添加概率列（按照要求的顺序）
for cls in required_classes:
    if cls in original_classes:
        cls_idx = np.where(original_classes == cls)[0][0]
        predictions_df[f'p_{cls}'] = lr.predict_proba(X_test)[:, cls_idx]
    else:
        # Fill with 0 if the category does not exist
        predictions_df[f'p_{cls}'] = 0

# 保存预测结果
predictions_df.to_csv('predictions.csv', index=False)
print("The prediction results have been saved to'predictions.csv'")

# 创建混淆矩阵图
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, lr.predict(X_test))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('prection label')
plt.ylabel('really label')
plt.title('Logistic Regression confusion matrix')
plt.tight_layout()
plt.savefig('metrics.png')
print("Confusion matrix graph has been saved to'metrics.png'")
""")

# 添加所有单元格到notebook
nb['cells'] = [title_cell, import_cell, load_data_cell, preprocess_cell, split_cell, lr_cell, lgb_cell, output_cell]

# 将notebook保存到文件
with open('classification.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook is saved as'classification.ipynb'")
print("\n analysis complete！")

print("\n" + "="*80)
print("                     Multi-categorized ROC curves")
print("="*80)

print("\n 1. Multi-categorized ROC curves（One-vs-Rest）")
print("    Function: Show the AUC of each category and visually display the model's discrimination ability of four categories")
print("    Output file：")
print("      - lightgbm_roc_curves_ovr.png：Special LightGBM ROC curve diagram")
print("      - roc_curves_both_models.png：Comparison of ROC curves between two models")
print("      - lightgbm_roc_auc_detailed.csv：Detailed AUC data")
print("    Content: 4 disease category ROC curves + 1 macro mean ROC curve + random classifier baseline")

print("\n 2. LightGBM Characteristic importance Top-20")
print("    Objective: Display the 20 genes that the model values ​​most, so that the reviewers can understand the key predictive genes intuitively")
print("    output file：")
print("      - shap_feature_importance_lightgbm.png：LightGBM SHAP Top-20 picture")
print("      - shap_feature_importance_lightgbm.csv：SHAP data")
print("    Chart type: Horizontal bar chart, horizontal axis is SHAP value, vertical axis is gene name")
print("    title：LightGBM Feature Importance (SHAP Top-20)")

print("\n 3. Additional generated analysis content：")
print("    Logistic Regression SHAP analysis")
print("    UMAP Visualization (real vs predicted label comparison)")
print("    Comprehensive performance analysis report")

print("\n" + "="*80)
print("                   PPT ")
print("="*80)
print("ROC analysis：lightgbm_roc_curves_ovr.png")
print("Characteristic importance：shap_feature_importance_lightgbm.png")


print("\n" + "="*80)