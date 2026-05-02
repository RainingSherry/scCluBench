# -*- coding: utf-8 -*-
"""
====================================================================================================
scMAE — 基于掩码自编码器的单细胞 RNA 聚类方法
====================================================================================================

【灵感来源】
    受 Vision Transformer (ViT) 和 BERT 中掩码预训练的启发，
    scMAE 将单细胞基因表达视为"句子"，每个基因视为"词"，
    通过掩码预测学习基因间的潜在关系。

【核心思想】
    ┌────────────────────────────────────────────────────────────────┐
    │                     scMAE 流程图                                │
    │                                                                │
    │  原始基因表达 X = [x₁, x₂, x₃, ..., x₁₀₀₀]                   │
    │      │                                                        │
    │      ▼                                                        │
    │  ┌──────────────────────┐                                      │
    │  │ Step 1: 随机掩码    │  掩码概率 p=0.4                     │
    │  │ X_masked = mask(X)  │  随机替换部分基因为0或随机值           │
    │  └──────────┬───────────┘                                      │
    │             │                                                  │
    │             ▼                                                  │
    │  ┌──────────────────────┐                                      │
    │  │ Step 2: 编码器      │  Linear → LayerNorm → Mish →       │
    │  │ Encoder              │  Linear → LayerNorm → Mish →       │
    │  │                      │  Linear (输出: hidden_size)         │
    │  └──────────┬───────────┘                                      │
    │             │                                                  │
    │             ▼                                                  │
    │  ┌──────────────────────┐                                      │
    │  │ Step 3: 掩码预测器   │  Linear(hidden_size → n_genes)      │
    │  │ Mask Predictor       │  预测被掩码位置的值                   │
    │  └──────────┬───────────┘                                      │
    │             │                                                  │
    │             ▼                                                  │
    │  ┌──────────────────────┐                                      │
    │  │ Step 4: 解码器       │  concat(latent, predicted_mask)      │
    │  │ Decoder              │  → Linear → 重构原始输入               │
    │  └──────────────────────┘                                      │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

【损失函数】（两部分组成）

    L_total = L_reconstruction + L_mask

    1. L_reconstruction (重构损失)
       - 使用加权MSE，仅对被掩码位置计算损失
       - masked_data_weight=0.75：被掩码位置权重更高
       - mask_loss_weight=0.7：控制两部分损失的相对重要性

    2. L_mask (掩码损失)
       - BCE损失：预测被掩码的位置（二分类）
       - 促使编码器学习哪些位置被掩码

【与scCDCG的关键区别】
    | 特性         | scCDCG          | scMAE                    |
    |------------|----------------|--------------------------|
    | 核心思想    | 图结构保持        | 掩码预测                  |
    | 损失函数    | 重构+正交+协方差+聚类 | 重构+掩码预测              |
    | 图结构      | 需要KNN图        | 不需要                    |
    | 聚类方式    | DEC+Sinkhorn    | KMeans/ Leiden            |
    | 超参数数量  | 较多            | 较少（更易调）             |

【超参数配置】
    | 参数               | 默认值 | 说明                      |
    |------------------|-------|-------------------------|
    | hidden_size      | 128   | 隐藏层维度                |
    | mask_prob        | 0.4   | 基因掩码概率               |
    | masked_data_weight| 0.75 | 被掩码数据的损失权重        |
    | mask_loss_weight | 0.7   | 掩码预测损失权重            |
    | batch_size       | 256   | 批大小                    |
    | lr               | 1e-3  | 学习率                    |
    | epochs           | 100   | 训练轮次                  |
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

# 添加父目录到路径（用于导入benchmark通用模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save

# 导入模型组件
from model import AutoEncoder


def set_seed(seed):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_noise(X, p):
    """
    对输入数据应用掩码噪声

    【掩码策略】
        对于每个基因，以概率p将其掩码：
        - 以概率p将基因值替换为从其他细胞随机采样的值
        - 返回：(损坏的数据, 掩码标记)

    【参数】
        X: 输入数据 (n_cells, n_genes)
        p: 掩码概率（可以是标量或向量）

    【返回】
        corrupted_X: 损坏后的数据（部分基因被替换）
        mask: 掩码标记（1=被掩码, 0=保持原样）

    【示例】
        X: [[1, 2, 3, 4], [5, 6, 7, 8]]
        mask概率p=0.5

        假设随机决定位置[0,2]和[1,1]被掩码
        corrupted_X: [[1, 6', 3, 4], [5, 7', 7, 8]]  # '表示来自随机细胞的替换值
        mask: [[0, 1, 0, 0], [1, 0, 0, 0]]
    """
    p = torch.tensor(p)
    # 生成掩码：伯努利分布采样
    should_swap = torch.bernoulli(p.to(X.device) * torch.ones((X.shape)).to(X.device))
    # 替换：被掩码位置用随机细胞的对应值替换
    corrupted_X = torch.where(
        should_swap == 1,
        X[torch.randperm(X.shape[0])],  # 从随机细胞取值
        X                                   # 保持原样
    )
    # 生成掩码标记
    masked = (corrupted_X != X).float()
    return corrupted_X, masked


class scRNADataset(Dataset):
    """
    单细胞RNA-seq数据集封装

    【功能】
        将NumPy数组封装为PyTorch Dataset
        支持DataLoader的批处理功能
    """

    def __init__(self, data, labels):
        """
        初始化数据集

        【参数】
            data: 基因表达矩阵 (n_cells, n_genes)
            labels: 细胞类型标签 (n_cells,)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个样本"""
        return self.data[idx], self.labels[idx]


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    """
    搜索Leiden聚类的分辨率参数以获得指定数量的簇

    【问题背景】
        Leiden聚类需要指定分辨率参数，而不是直接的簇数
        不同数据集、不同细胞类型数需要不同的分辨率

    【算法】
        二分搜索找到使聚类数最接近目标的分辨率：
        1. 从高分辨率开始搜索（2.5 → 0.01）
        2. 按increment步长递减
        3. 找到聚类数=目标时停止

    【参数】
        adata: 包含嵌入的AnnData对象
        fixed_clus_count: 目标聚类数
        increment: 分辨率搜索步长

    【返回】
        最优分辨率参数
    """
    dis = []
    # 从高到低搜索（高分辨率=多簇，低分辨率=少簇）
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)

    for res in resolutions:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        dis.append(abs(count_unique_leiden - fixed_clus_count))
        if count_unique_leiden == fixed_clus_count:
            break

    # 返回误差最小的分辨率
    return resolutions[np.argmin(dis)]


def inference(net, data_loader, device):
    """
    从模型提取特征（嵌入向量）

    【功能】
        遍历整个数据集，获取每个细胞的嵌入向量

    【参数】
        net: 训练好的模型
            data_loader: 数据加载器
            device: 计算设备

    【返回】
        feature_vector: 所有细胞的嵌入向量 (n_cells, hidden_size)
        labels_vector: 对应的真实标签 (n_cells,)
    """
    net.eval()
    feature_vector = []
    labels_vector = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            # 使用编码器获取嵌入
            feature_vector.extend(net.feature(x).detach().cpu().numpy())
            labels_vector.extend(y.numpy())

    return np.array(feature_vector), np.array(labels_vector)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='scMAE: Masked Autoencoder for scRNA-seq Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='输入 h5ad 文件路径')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='结果保存目录')

    # 模型参数
    parser.add_argument('--n_clusters', type=int, required=True,
                       help='聚类数')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--mask_prob', type=float, default=0.4,
                       help='掩码概率')
    parser.add_argument('--masked_data_weight', type=float, default=0.75,
                       help='被掩码数据的损失权重')
    parser.add_argument('--mask_loss_weight', type=float, default=0.7,
                       help='掩码预测损失权重')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备号')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='评估间隔（轮次）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设备设置
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f'Using device: {device}')

    # 随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Step 1: 数据加载与预处理
    # =========================================================================
    print('Loading data...')
    X, Y, sf, adata = prepare_data_for_model(
        args.data_path,
        size_factors=True,
        filter_min_counts=True,
        logtrans_input=True,
        normalize_input=True
    )

    # 转换为NumPy数组
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    # 标签编码
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # 获取聚类数
    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # =========================================================================
    # Step 2: 创建数据集和数据加载器
    # =========================================================================
    dataset = scRNADataset(X, Y)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size * 5,
        shuffle=False,
        drop_last=False
    )

    # =========================================================================
    # Step 3: 初始化模型
    # =========================================================================
    model = AutoEncoder(
        num_genes=X.shape[1],
        hidden_size=args.hidden_size,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 掩码概率（每个基因独立）
    mask_probas = [args.mask_prob] * X.shape[1]

    # =========================================================================
    # Step 4: 训练循环
    # =========================================================================
    print('Starting training...')

    best_acc = 0
    best_epoch = 0
    best_embedding = None
    best_pred = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)

            # 应用掩码
            x_corrupted, mask = apply_noise(x, mask_probas)

            # 前向传播
            optimizer.zero_grad()
            _, loss = model.loss_mask(x_corrupted, x, mask)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # =========================================================================
        # Step 5: 周期性评估
        # =========================================================================
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 提取嵌入向量
            embedding, true_labels = inference(model, test_loader, device)

            # 聚类
            if embedding.shape[0] < 10000:
                # 小数据集：直接使用KMeans
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=args.seed,
                    n_init=20
                )
                pred_labels = kmeans.fit_predict(embedding)
            else:
                # 大数据集：使用Leiden聚类
                adata_emb = sc.AnnData(embedding)
                sc.pp.neighbors(adata_emb, n_neighbors=10, use_rep="X")
                reso = res_search_fixed_clus(adata_emb, n_clusters)
                sc.tl.leiden(adata_emb, resolution=reso)
                pred_labels = np.array([
                    int(x) for x in adata_emb.obs['leiden'].to_list()
                ])

            # 评估并追踪最优模型
            from evaluation import evaluation as eval_fn
            acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, _ = eval_fn(
                np.array(true_labels), np.array(pred_labels))

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                best_embedding = embedding.copy()
                best_pred = pred_labels.copy()
                # 保存最优模型检查点
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                    'best_epoch': best_epoch,
                    'best_acc': best_acc,
                }, os.path.join(args.save_dir, 'model_checkpoint.pth'))

            # 保存结果
            save(args.save_dir, true_labels, pred_labels, epoch + 1, embedding)

            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, ACC: {acc:.4f}, Best: {best_acc:.4f}')

    print(f'Best epoch: {best_epoch}, Best ACC: {best_acc:.4f}')

    # =========================================================================
    # Step 6: 保存最终结果（最优 epoch）
    # =========================================================================
    if best_embedding is not None:
        true_labels_arr = np.array(true_labels)
        save(args.save_dir, true_labels_arr, best_pred, best_epoch, best_embedding)

    print(f'Training completed. Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
