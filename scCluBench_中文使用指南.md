# scCluBench 中文使用指南

> **scCluBench — 单细胞RNA测序聚类方法标准化评测框架**
>
> 论文来源：AAAI 2026
>
> 本文档为计算机专业博士研究生复现本 benchmark 以及开发新算法而编写。

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码结构](#2-代码结构)
3. [核心模块详解](#3-核心模块详解)
4. [模型算法原理](#4-模型算法原理)
5. [评估指标说明](#5-评估指标说明)
6. [快速开始](#6-快速开始)
7. [添加新模型](#7-添加新模型)
8. [关键代码解读](#8-关键代码解读)
9. [复现注意事项](#9-复现注意事项)
10. [为你的算法写一个超越基准的版本](#10-为你的算法写一个超越基准的版本)

---

## 1. 项目概述

### 1.1 什么是 scCluBench？

scCluBench 是一个标准化的单细胞 RNA 测序（scRNA-seq）聚类方法评测框架。它解决了以下问题：

| 问题 | scCluBench 的解决方案 |
|------|----------------------|
| 数据格式不统一 | 统一使用 `.h5ad` 格式 |
| 预处理流程各异 | 标准化 6 步预处理流程 |
| 评估指标不一致 | 统一使用 8 个评估指标 |
| 模型接口差异大 | 定义标准 `run.py` 接口 |
| 难以公平比较 | 相同数据集、相同预处理、相同评估 |

### 1.2 支持的数据集

| 数据集名称 | 细胞数 | 基因数 | 细胞类型数 | 来源 |
|-----------|--------|--------|-----------|------|
| Arabidopsis_scRNA_synthetic | 1,500 | 3,000 | 8 | 植物 |
| Arabidopsis_Stereo-seq_leaf | 721 | 18,257 | 6 | 植物 |
| HumanPancreas_1 | 2,544 | 61,497 | 7 | 人类胰腺 |
| HumanPancreas_2 | 2,126 | 61,497 | 10 | 人类胰腺 |
| MousePancreas_Aging | 6,201 | 53,384 | 9 | 小鼠胰腺 |
| TabulaSapiens_Pancreas | 14,140 | 61,497 | 23 | 人类多组织 |
| Blood_BoneMarrow | 15,502 | 61,497 | 35 | 血液/骨髓 |

### 1.3 支持的模型类别

```
scCluBench/
├── DeepLearning/        # 深度学习方法
│   ├── scDCC/         # 深度约束聚类（变分自编码器 + ZINB 损失）
│   ├── scMAE/         # 掩码自编码器
│   ├── scNAME/        # 邻域聚合掩码嵌入
│   ├── scziDesk/      # 零膨胀深度嵌入
│   ├── desc/          # 深度嵌入单细胞聚类
│   └── dec/           # 深度嵌入聚类
├── GNN/               # 图神经网络方法
│   ├── scCDCG/        # 基于图注意力的聚类
│   ├── scDSC/         # 结构化深度聚类网络
│   ├── scGNN/         # 图神经网络
│   └── AttentionAE-sc/  # 注意力自编码器
├── Foundation/         # 基座模型
│   ├── scGPT/         # 单细胞 GPT
│   ├── Geneformer/    # 基因 Transformer
│   └── GeneCompass/   # 基因指南针
└── Traditional/       # 传统方法
    └── Leiden/        # Leiden 社区检测
```

---

## 2. 代码结构

### 2.1 目录树

```
scCluBench-main/
│
├── 📄 核心数据处理
├── preprocess.py         ← 数据预处理的统一入口
├── evaluation.py        ← 评估指标计算
├── utils.py             ← 工具函数和保存接口
│
├── 📄 运行脚本
├── run_all_selected.sh  ← 主运行脚本（循环所有模型×数据集）
├── run_one_model.sh     ← 单模型单数据集运行
├── run_leiden.py        ← Leiden 基线方法
├── run_template.py      ← 新模型模板
│
├── 📄 scCDCG（图神经网络方法）
└── GNN/scCDCG/
    ├── run.py           ← 入口脚本
    ├── model.py         ← 神经网络模型定义
    ├── scCDCG_layer.py  ← 图注意力层
    ├── scCDCG_utils.py  ← 工具函数
    └── train_scCDCG.py  ← 训练逻辑
│
├── 📄 scMAE（掩码自编码器方法）
└── DeepLearning/scMAE/
    ├── run.py           ← 入口脚本
    ├── model.py         ← MAE 模型定义
    ├── main.py          ← 原始训练脚本
    └── ...
│
├── 📄 数据与配置
├── DATA_INVENTORY.tsv   ← 数据集目录
├── RUNBOOK.md          ← 使用说明
├── ENV_SETUP.md        ← 环境配置
└── REPRODUCE_NOTES.md ← 复现笔记
```

### 2.2 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流图                                  │
│                                                                 │
│  原始 .h5ad 文件                                               │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │ preprocess.py::prepare_data_for_model()      │               │
│  │  1. 读取 h5ad                              │               │
│  │  2. Per-cell 归一化                        │               │
│  │  3. Log1p 变换                            │               │
│  │  4. HVG 筛选（top 1000）                  │               │
│  │  5. 计算 size factor                      │               │
│  │  6. Z-score 标准化                        │               │
│  └──────────────┬────────────────────────────┘               │
│                 │ X (n_cells, 1000)                          │
│                 │ Y (细胞类型标签)                              │
│                 ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │              模型训练                          │               │
│  │  scCDCG / scMAE / scDCC / DEC / Leiden     │               │
│  └──────────────┬────────────────────────────┘               │
│                 │ embedding (n_cells, 16/32/128)             │
│                 ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │           KMeans / Leiden 聚类                │               │
│  └──────────────┬────────────────────────────┘               │
│                 │ y_pred (预测标签)                           │
│                 ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │ utils.py::save()                           │               │
│  │  evaluation.py::evaluation()               │               │
│  └──────────────┬────────────────────────────┘               │
│                 │                                              │
│                 ▼                                              │
│  输出：metrics.json / embedding.npy / types_pred.csv          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 `preprocess.py` — 数据预处理

这是所有模型的数据入口。**所有模型都应该调用 `prepare_data_for_model()` 函数获取预处理后的数据。**

```python
from preprocess import prepare_data_for_model

# 获取预处理后的数据
X, Y, sf, adata = prepare_data_for_model(
    'data.h5ad',
    size_factors=True,       # 计算 size factor（ZINB 模型需要）
    filter_min_counts=True,   # 执行 HVG 筛选
    logtrans_input=True,      # 执行 log1p 变换
    normalize_input=True      # 执行 Z-score 标准化
)

# 返回值
# X: DataFrame, shape=(n_cells, 1000) — 预处理后的表达矩阵
# Y: Series — 细胞类型标签
# sf: Series — size factor（用于 ZINB 损失）
# adata: AnnData — 完整的 AnnData 对象
```

**预处理流程详解：**

| 步骤 | 操作 | 目的 | 关键参数 |
|------|------|------|---------|
| 1 | Per-cell 归一化 | 消除测序深度差异 | target_sum=1e4 |
| 2 | Log1p 变换 | 稳定方差、压缩大值 | log(1+x) |
| 3 | HVG 筛选 | 去除噪声、保留生物信号 | top 1000 genes |
| 4 | Z-score 标准化 | 同一尺度 | mean=0, std=1 |

**为什么需要 HVG 筛选？**

高度可变基因（Highly Variable Genes）是表达量在不同细胞间变化最大的基因。它们：
- 排除了 housekeeping genes（所有细胞都高表达，无区分度）
- 排除了技术噪声基因（所有细胞都低表达）
- 保留了真正反映细胞类型差异的基因

**HVG 筛选的参数含义：**

```python
sc.pp.highly_variable_genes(
    adata,
    min_mean=0.0125,    # 过滤低表达基因（平均表达 > 0.0125）
    max_mean=3,        # 过滤高表达管家基因（平均表达 < 3）
    min_disp=0.5,      # 保留离散度高的基因
    n_top_genes=1000,  # 只保留变化最大的 1000 个基因
    subset=True        # 直接修改 adata
)
```

### 3.2 `evaluation.py` — 评估指标

本模块计算 8 个聚类质量指标：

```python
from evaluation import evaluation

acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, y_pred_ = evaluation(y_true, y_pred)
```

**指标详解：**

| 指标 | 全称 | 含义 | 最优值 |
|------|------|------|--------|
| ACC | Accuracy | 聚类准确率（标签匹配后） | 1.0 |
| NMI | Normalized Mutual Information | 标准化互信息 | 1.0 |
| ARI | Adjusted Rand Index | 调整兰德指数 | 1.0 |
| F1-macro | F1 Score (Macro) | 宏平均 F1 | 1.0 |
| FMI | Fowlkes-Mallows Index | 福克斯-马洛斯指数 | 1.0 |
| V-measure | V Measure | 同质性+完整性的调和平均 | 1.0 |
| Homogeneity | 同质性 | 每簇只含单一类别 | 1.0 |
| Completeness | 完整性 | 同类成员全在同一簇 | 1.0 |

**标签匹配问题（Hungarian 算法）：**

聚类产生的标签是"无名"的，直接比较会出错：

```
真实标签:   [0, 0, 1, 1]  → Alpha 细胞
预测标签:   [1, 1, 0, 0]  → 如果直接算 ACC = 0！

因为 0≠1, 0≠1, 1≠0, 1≠0 → 准确率 0%

但实际上：预测的 [1,1,0,0] 完美对应真实 [0,0,1,1]
只是标签命名不同而已！
```

解决方案：使用 Hungarian 算法找到最优映射：

```
混淆矩阵 G[i,j] = |真实类别 i ∩ 预测类别 j|
    预测:   0     1
真实 0:  [ 2     0  ]    ← 类别 0 全部分到预测类别 1
真实 1:  [ 0     2  ]    ← 类别 1 全部分到预测类别 0

最优映射：真实 0 → 预测 1，真实 1 → 预测 0
重排后：[1,1,0,0] → [0,0,1,1]
ACC = 1.0 ✓
```

### 3.3 `utils.py` — 工具函数

**统一保存接口（所有模型都应使用）：**

```python
from utils import save

save(
    embedding_path='./results/scCDCG/HumanPancreas_1',
    y=y_true,           # 真实标签
    y_pred=y_pred,      # 预测标签
    epoch=200,          # 当前轮次
    embedding=embedding  # 嵌入向量 (n_cells, embedding_dim)
)
```

**保存的文件：**

```
results/scCDCG/HumanPancreas_1/
├── metrics_200.json       # 8 个评估指标
├── embedding_200.npy     # 嵌入向量
├── embedding.h5          # HDF5 格式的嵌入+标签
└── types_200_pred.csv    # 标签对照表
```

---

## 4. 模型算法原理

### 4.1 scCDCG — 图神经网络聚类

**核心思想：** 将单细胞数据建模为图结构，利用图神经网络学习细胞间的相似性。

**网络架构：**

```
┌─────────────────────────────────────────────────────────────────┐
│                     scCDCG 训练流程                               │
│                                                                 │
│  Step 1: 图构建                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 原始特征 ─L2归一化→ 自环邻接矩阵                          │   │
│  │                  └─余弦相似度─传递闭包→ KNN 邻接矩阵        │   │
│  │                                                         │   │
│  │ L₁ = D⁻¹/² · A₁ · D⁻¹/²   （基于自环）                  │   │
│  │ L₂ = D⁻¹/² · A₂ · D⁻¹/²   （基于 KNN）                 │   │
│  │ L = λ·L₁ + (1-λ)·L₂        （双拉普拉斯混合）            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│  Step 2: 预训练（无聚类损失）                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Encoder: Linear(n_genes, 256) → Linear(256, 16)        │   │
│  │ Decoder: Linear(16, 256) → Linear(256, n_genes)         │   │
│  │                                                         │   │
│  │ L_pretrain = α·L_recon + β·L_ort + γ·L_cov            │   │
│  │                                                         │   │
│  │ • L_recon: 重构损失 MSE(x̂, x)                          │   │
│  │ • L_ort:   正交损失 MSE(zᵀz, I) — 各维度独立            │   │
│  │ • L_cov:   图结构损失 -Tr(zᵀLz)/n — 相邻节点靠近        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│  Step 3: KMeans 初始化聚类中心                                 │
│                           │                                     │
│  Step 4: 微调（加入聚类损失）                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ClusterAssignment: 软聚类分配 q_{ij}                      │   │
│  │                                                         │   │
│  │ Sinkhorn 算法计算目标分布 P                               │   │
│  │                                                         │   │
│  │ L_finetune = α·L_recon + β·L_ort + γ·L_cov + δ·L_KL   │   │
│  │                                                         │   │
│  │ L_KL = KL(q || p) — DEC 风格的聚类损失                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                      嵌入向量 z                                 │
│                           │                                     │
│                      KMeans 聚类 → 预测标签                     │
└─────────────────────────────────────────────────────────────────┘
```

**Sinkhorn 算法详解：**

Sinkhorn 用于将软聚类分配转换为最优目标分布：

```python
def sinkhorn(pred, lambdas, row, col):
    """
    pred: 软分配矩阵 (n_cells, n_clusters)
    lambdas: 指数（控制分布尖锐程度，λ越大越集中）
    row: 行边际（每个细胞权重，通常=1/n）
    col: 列边际（每个类别权重，通常=类别频率）

    迭代：
        u[i] = row[i] / Σ_j P[i,j]·v[j]
        v[j] = col[j] / Σ_i P[i,j]·u[i]
    """
    # 迭代1000次确保收敛
    for _ in range(1000):
        u = row * (p @ v)⁻¹
        v = col * (u.T @ p)⁻¹
    target = diag(u) @ p @ diag(v)
    return target
```

### 4.2 scMAE — 掩码自编码器

**灵感来源：** BERT 的掩码语言模型 + Vision Transformer 的掩码图像模型

**核心思想：** 随机掩码部分基因，模型学习预测被掩码的值，从而学习基因间的潜在关系。

**网络架构：**

```
┌─────────────────────────────────────────────────────────────────┐
│                     scMAE 流程                                   │
│                                                                 │
│  输入: X = [x₁, x₂, x₃, ..., x₁₀₀₀]                         │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 1: 随机掩码（掩码概率 p=0.4）                        │   │
│  │                                                         │   │
│  │ X_masked:                                               │   │
│  │   被掩码位置: 替换为从其他细胞随机采样的值                 │   │
│  │   未掩码位置: 保持原样                                    │   │
│  │                                                         │   │
│  │ mask: [0, 1, 0, 0, 1, ..., 0]  (1=被掩码)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 2: 编码器                                          │   │
│  │                                                         │   │
│  │ Dropout(0)                                              │   │
│  │ Linear(1000, 256) → LayerNorm → Mish                   │   │
│  │ Linear(256, 128)   → LayerNorm → Mish                   │   │
│  │ Linear(128, 128)   → 输出 latent (n_cells, 128)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│  ┌─────────────────────────┐  ┌─────────────────────────┐     │
│  │ Step 3: 掩码预测器       │  │ Step 4: 解码器          │     │
│  │                         │  │                         │     │
│  │ Linear(128, 1000)       │  │ concat([latent,         │     │
│  │ → predicted_mask        │  │        predicted_mask])  │     │
│  │                         │  │ → Linear(1128, 1000)    │     │
│  │ BCE 损失                │  │ → reconstruction        │     │
│  └─────────────────────────┘  │                         │     │
│                               │ 加权 MSE 损失             │     │
│                               └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**损失函数：**

```
L_total = L_reconstruction + L_mask

1. 重构损失（加权 MSE）：
   L_recon = (1 - mask_loss_weight) × mean(w_nums × MSE(x̂, x))

   其中 w_nums = mask × masked_data_weight + (1-mask) × (1-masked_data_weight)
   即被掩码位置权重更高（默认 0.75），未掩码位置权重更低（默认 0.25）

2. 掩码损失（BCE）：
   L_mask = mask_loss_weight × BCE(predicted_mask, mask)
```

### 4.3 Leiden — 传统基线

Leiden 是 Louvain 的改进版本，用于社区检测：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Leiden 聚类流程                              │
│                                                                 │
│  原始数据 → 归一化 → log1p → HVG → PCA → KNN 图 → Leiden     │
│                                                                 │
│  优化目标：最大化模块度 Q                                      │
│                                                                 │
│  Q = Σ_{ij} (A_ij - k_i×k_j/2m) × δ(c_i, c_j)               │
│                                                                 │
│  其中：                                                         │
│    A_ij: 边权重（相似度）                                      │
│    k_i: 节点 i 的度                                            │
│    m: 总边权重                                                 │
│    δ(c_i, c_j): 节点 i 和 j 是否在同一簇                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 评估指标说明

### 5.1 指标详解

**ACC（准确率）：**
- 经过 Hungarian 算法标签匹配后的准确率
- 衡量"有多少细胞被正确分类"

**NMI（标准化互信息）：**
```
NMI = 2 × I(Y_true; Y_pred) / (H(Y_true) + H(Y_pred))
```
- I: 互信息，H: 熵
- 衡量两个聚类之间的信息共享程度
- 对标签命名不敏感，只关注划分结构

**ARI（调整兰德指数）：**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```
- RI: 兰德指数
- 调整了随机划分的期望
- 对类别不均衡更鲁棒

**F1-macro（宏平均 F1）：**
```
F1_i = 2 × precision_i × recall_i / (precision_i + recall_i)
F1_macro = mean(F1_i) for all classes i
```
- 每个类别的 F1 分数算术平均
- 对所有类别一视同仁

**V-measure（同质性-完整性调和平均）：**
```
V = (1 + β) × h × c / (β × h + c)
```
- h: 同质性（每个簇只含单一类）
- c: 完整性（同类成员全在同一簇）
- β=1 时两者等权重

### 5.2 指标选择建议

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 快速比较 | ACC, NMI | 直观易懂 |
| 不均衡数据 | ARI | 对类别不均衡鲁棒 |
| 重视少数类 | F1-macro | 对所有类一视同仁 |
| 结构质量 | V-measure | 同时考虑同质性和完整性 |

---

## 6. 快速开始

### 6.1 环境配置

```bash
# 创建 conda 环境
conda create -n scclubench-main python=3.10
conda activate scclubench-main

# 安装核心依赖
pip install scanpy scikit-learn torch numpy pandas scipy
pip install leidenalg igraph  # Leiden 聚类需要

# 可选：GPU 支持
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install h5py munkres
```

### 6.2 运行单个模型

```bash
# 运行 scCDCG
bash run_one_model.sh scCDCG Arabidopsis_scRNA_synthetic 200

# 运行 scMAE
bash run_one_model.sh scMAE HumanPancreas_1 100

# 运行 Leiden 基线
bash run_one_model.sh Leiden MousePancreas_Aging 0
```

### 6.3 运行所有选定模型

```bash
# 在项目根目录执行
bash run_all_selected.sh
```

### 6.4 查看结果

```bash
# 结果保存在 results_summary.csv
cat results_summary.csv

# 各模型的具体结果
cat results/scCDCG/HumanPancreas_1/metrics.json
```

---

## 7. 添加新模型

### 7.1 模型模板

参考 `run_template.py` 创建你的模型：

```python
#!/usr/bin/env python3
"""
YourModel — 你的新模型名称
"""

import os
import sys
import argparse
import numpy as np
import torch

# 导入 benchmark 核心模块
from preprocess import prepare_data_for_model
from utils import save
from evaluation import evaluation

def parse_args():
    parser = argparse.ArgumentParser(description='Your Model')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--n_clusters', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ========== 数据加载（必须使用统一接口）==========
    X, Y, sf, adata = prepare_data_for_model(
        args.data_path,
        size_factors=True,
        filter_min_counts=True,
        logtrans_input=True,
        normalize_input=True
    )
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    # ========== 你的模型代码 ==========
    # 在这里实现你的算法

    # ========== 保存结果（必须使用统一接口）==========
    save(args.save_dir, Y, y_pred, args.epochs, embedding)

if __name__ == '__main__':
    main()
```

### 7.2 注册新模型

在 `run_one_model.sh` 中添加：

```bash
YourModel)
    cd ${REPO_ROOT}/YourModelFolder
    python run.py \
        --data_path $DATA_PATH \
        --n_clusters $N_CLUSTERS \
        --save_dir ${RESULTS_DIR}/${MODEL}/${DATASET} \
        --epochs $EPOCHS \
        2>&1 | tee -a $LOG_FILE
    ;;
```

在 `run_all_selected.sh` 的 `MODELS` 数组中添加模型名。

---

## 8. 关键代码解读

### 8.1 数据预处理的关键代码

```python
# preprocess.py 中的核心逻辑

# Step 1: Per-cell 归一化
sc.pp.normalize_per_cell(adata)
# 等价于：每个细胞的总 counts 缩放到 target_sum
# X_new = X × (target_sum / X.sum())

# Step 2: Log1p 变换
sc.pp.log1p(adata)
# 等价于：X_new = log(1 + X)

# Step 3: HVG 筛选
sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
# 只保留变化最大的 1000 个基因

# Step 4: Z-score 标准化
sc.pp.scale(adata)
# 等价于：X_new = (X - X.mean()) / X.std()
```

### 8.2 评估模块的核心逻辑

```python
# evaluation.py 中的标签匹配

def best_map(y_true, y_pred):
    """
    使用 Hungarian 算法找最优标签映射
    """
    # 1. 构建混淆矩阵
    G[i,j] = |真实标签 i ∩ 预测标签 j|
    G = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            G[i,j] = np.sum((y_true == i) & (y_pred == j))

    # 2. Hungarian 算法求解
    # 目标：最大化 Σ G[i, j(i)]，约束：每个真实标签恰好匹配一个预测标签
    A = linear_assignment(-G)  # 负号：因为要最大化

    # 3. 重排预测标签
    new_y_pred = np.zeros_like(y_pred)
    for i in range(len(A[0])):
        # 将预测标签 A[1][i] 映射到真实标签 A[0][i]
        new_y_pred[y_pred == A[1][i]] = A[0][i]

    return new_y_pred.astype(int)
```

### 8.3 scCDCG 的关键损失函数

```python
# scCDCG 的四项损失

# 1. 重构损失
loss_recon = F.mse_loss(x_hat, x)

# 2. 正交损失
# 促使嵌入向量各维度正交（独立）
z_norm = F.normalize(z, p=2, dim=0)
orth_loss = F.mse_loss(z_norm.T @ z_norm, torch.eye(n_dim))

# 3. 协方差损失
# 保持图结构：相邻节点在嵌入空间靠近
L = balancer * L_1 + (1-balancer) * L_2  # 双拉普拉斯
cov_loss = -trace(z_norm.T @ L @ z_norm) / n

# 4. KL 散度损失（DEC 聚类）
p_target = sinkhorn(q)  # 计算目标分布
kl_loss = F.kl_div(q.log(), p_target, reduction='sum')
```

---

## 9. 复现注意事项

### 9.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | 批量太大 | 减小 batch_size 或数据量 |
| 数据找不到 | 路径错误 | 检查 DATA_DIR 配置 |
| 导入错误 | 包未安装 | `pip install -r requirements.txt` |
| 结果不一致 | 随机种子不同 | 设置相同的 seed |

### 9.2 GPU 内存优化

```python
# 如果 CUDA OOM，尝试以下方法：

# 1. 减小批次大小
batch_size = 128  # 从 256 减小

# 2. 使用梯度累积
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 使用混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
```

### 9.3 复现性保证

```python
# 在所有模型开头添加：
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 10. 为你的算法写一个超越基准的版本

### 10.1 超越基准的策略

基于对代码的深入理解，以下是超越现有基准的可能方向：

#### 策略 1：改进图结构

**现有方法的问题：**
- scCDCG 使用简单的余弦相似度构建图
- 没有利用细胞类型的层级信息

**改进方向：**
```python
# 改进的图构建：使用 UMAP 后的距离作为边权重
from umap import UMAP

# 1. 先用 UMAP 降维
X_umap = UMAP(n_components=30, random_state=42).fit_transform(X)

# 2. 使用 k-近邻距离构建图
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=15).fit(X_umap)
distances, indices = nn.kneighbors(X_umap)

# 3. 基于距离的高斯核权重
sigma = np.median(distances)
adj = np.exp(-distances**2 / (2 * sigma**2))
```

#### 策略 2：多尺度聚合

**现有方法的问题：**
- scCDCG 只使用单一尺度的 KNN 图
- 不同细胞类型可能需要在不同尺度下才能区分

**改进方向：**
```python
# 多尺度图融合
scales = [10, 20, 40]  # 不同的邻居数
graphs = []

for k in scales:
    sc.pp.neighbors(adata, n_neighbors=k)
    L_k = compute_laplacian(adata.obsp['connectivities'])
    graphs.append(L_k)

# 加权融合
L_fused = sum(w_k * L_k for w_k, L_k in zip(weights, graphs))
```

#### 策略 3：时序/对比学习

**改进方向：**
```python
# 使用对比学习增强嵌入
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        # z1, z2: 同一数据的两个增强视图
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # InfoNCE 损失
        sim = (z1 @ z2.T) / self.temp
        labels = torch.arange(len(z1))
        loss = F.cross_entropy(sim, labels)
        return loss
```

#### 策略 4：伪标签自训练

**改进方向：**
```python
# 迭代式自训练
for round in range(3):
    # 1. 用当前嵌入聚类
    y_pred = KMeans(n_clusters=k).fit_predict(embedding)

    # 2. 计算伪标签置信度
    confidence = compute_cluster_confidence(embedding, y_pred)

    # 3. 筛选高置信度样本作为伪标签
    high_conf_mask = confidence > threshold
    pseudo_labels[high_conf_mask] = y_pred[high_conf_mask]

    # 4. 用伪标签监督训练
    loss = main_loss + alpha * supervised_loss(pseudo_labels)
```

### 10.2 消融实验设计

为了验证你的改进，需要做消融实验：

| 实验 | 描述 | 预期结果 |
|------|------|---------|
| 基准 | scCDCG 默认设置 | 基线分数 |
| A | 只改图构建 | +X% |
| B | 只改损失函数 | +Y% |
| C | A + B | > A + B |

### 10.3 论文写作建议

如果你的工作超越了基准，可以考虑：

1. **分析失败案例**：哪些数据集/细胞类型上表现差？为什么？
2. **可视化**：t-SNE/UMAP 可视化嵌入，观察簇的分离程度
3. **生物学解释**：识别差异表达基因，是否与已知 marker 一致？
4. **时间复杂度**：虽然聚类精度重要，但计算效率也不可忽视

---

## 附录：快捷命令参考

```bash
# 运行特定模型
bash run_one_model.sh scCDCG Arabidopsis_scRNA_synthetic 200

# 查看结果
cat results/scCDCG/Arabidopsis_scRNA_synthetic/metrics.json

# 生成汇总表格
python -c "import pandas as pd; print(pd.read_csv('results_summary.csv'))"

# 检查 GPU
python check_gpu.py

# 检查数据
python check_data.py
```

---

**祝你的研究顺利！** 🎓

如果你有任何问题或需要进一步的帮助，请随时提问。
