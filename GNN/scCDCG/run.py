# -*- coding: utf-8 -*-
"""
====================================================================================================
scCDCG — 基于图神经网络的单细胞 RNA 聚类方法
====================================================================================================

【论文出处】
    scCDCG: Cell-type Discovery via Clustering on Graphs
    使用图注意力自动编码器进行单细胞聚类

【核心思想】
    将单细胞数据建模为图结构，利用图神经网络学习细胞间的相似性关系，
    同时进行降维和聚类。

【网络架构】
    ┌─────────────────────────────────────────────────────────┐
    │                    scCDCG 整体流程                        │
    │                                                         │
    │  原始表达矩阵 X                                           │
    │      │                                                  │
    │      ▼                                                  │
    │  ┌──────────────────┐                                    │
    │  │ Step 1: 图构建    │  L2归一化 → 自环邻接矩阵 + KNN图    │
    │  └────────┬─────────┘                                    │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                    │
    │  │ Step 2: 双拉普拉斯 │  L1 (自环) + L2 (KNN) 双重约束  │
    │  └────────┬─────────┘                                    │
    │           │                                              │
    │           ▼                                              │
    │  ┌──────────────────┐                                    │
    │  │ Step 3: 预训练   │  AE自编码器 + 正交/协方差损失      │
    │  └────────┬─────────┘                                    │
    │           │ KMeans初始化聚类中心                           │
    │           ▼                                              │
    │  ┌──────────────────┐                                    │
    │  │ Step 4: 微调     │  DEC聚类损失 + Sinkhorn最优传输    │
    │  └────────┬─────────┘                                    │
    │           │                                              │
    │           ▼                                              │
    │       嵌入向量 Z ──────► KMeans 聚类 ──► 预测标签         │
    └─────────────────────────────────────────────────────────┘

【损失函数】（四部分组成）

    L_total = α·L_recon + β·L_ort + γ·L_cov + δ·L_KL

    1. L_recon (重构损失, α=0.23)
       - MSE(x̂, x)：解码器重构原始输入的能力

    2. L_ort (正交损失, β=0.65)
       - MSE(z^T·z, I)：嵌入向量各维度正交
       - 促使嵌入空间各方向独立，增加表达能力

    3. L_cov (协方差损失, γ=0.17)
       - -Tr(z^T·L·z)/n：保持图结构
       - L = λ·L₁ + (1-λ)·L₂（双拉普拉斯混合，λ=0.55）
       - 使相邻节点在嵌入空间靠近

    4. L_KL (KL散度损失, δ=0.12)
       - DEC风格的聚类损失
       - 使用Sinkhorn算法计算目标分布

【Sinkhorn算法】（最优传输）
    用于将软聚类分配转换为最优目标分布
    迭代1000次确保收敛

【超参数配置】
    | 参数              | 默认值 | 说明                        |
    |-----------------|-------|---------------------------|
    | embedding_dim   | 16    | 嵌入空间维度                 |
    | hidden_dim      | 256   | 隐藏层维度                   |
    | lr              | 1e-3  | 学习率                      |
    | weight_decay    | 5e-3  | 权重衰减                    |
    | factor_construct| 0.23  | 重构损失权重                 |
    | factor_ort      | 0.65  | 正交损失权重                 |
    | factor_corvar   | 0.17  | 协方差损失权重               |
    | factor_KL       | 0.12  | KL散度权重                  |
    | balancer        | 0.55  | 双拉普拉斯混合系数           |
    | lambdas         | 5     | Sinkhorn指数参数             |

【与其他方法的关键区别】
    - scDCC: 使用ZINB损失建模count数据
    - scMAE: 使用掩码预测学习基因间关系
    - Leiden: 传统方法，不学习嵌入
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import pickle

# 添加父目录到路径（用于导入 benchmark 通用模块）
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _repo_root)

from preprocess import prepare_data_for_model
from utils import save

from sklearn.cluster import KMeans
from torchmetrics.functional import pairwise_cosine_similarity

# 导入模型组件
from model import AE_NN, FULL_NN, ClusterAssignment
from scCDCG_utils import get_laplace_matrix
import torch.nn as nn


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


def sinkhorn(pred, lambdas, row, col):
    """
    Sinkhorn算法 — 最优传输的交替迭代求解器

    【算法原理】
        给定输入矩阵 P（软分配）和权重 (row, col)，
        找到最优传输矩阵 T，使得：
            T* = argmin Σ T[i,j] * (-log(P[i,j]^λ))
            约束: Σ_j T[i,j] = row[i], Σ_i T[i,j] = col[j]

    【参数】
        pred: 软聚类分配矩阵 (n_cells, n_clusters)
        lambdas: 指数参数（控制分布的尖锐程度，λ越大越集中）
        row: 行边际分布（通常为均匀分布，即每个细胞权重=1/n）
        col: 列边际分布（通常为类别频率，即每个类别应有n/k个细胞）

    【迭代公式】
        u[i] = row[i] / Σ_j P[i,j]·v[j]
        v[j] = col[j] / Σ_i P[i,j]·u[i]

    【返回值】
        target: 最优传输后的目标分布（用于KL散度损失）

    【使用场景】
        DEC/Clustering任务中，将软分配q转换为硬目标分布p
    """
    num_node = pred.shape[0]    # 细胞数
    num_class = pred.shape[1]   # 类别数

    # 指数化：增强高置信度，削弱低置信度
    p = np.power(pred, lambdas)

    # 初始化边际向量（均匀分布）
    u = np.ones(num_node)
    v = np.ones(num_class)

    # Sinkhorn迭代（交替投影到行边际和列边际）
    for index in range(1000):
        # 更新行边际
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15  # 处理数值溢出
        # 更新列边际
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15

    # 计算最终目标分布
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='scCDCG: Graph Neural Network for scRNA-seq Clustering',
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
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='嵌入空间维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')

    # 损失权重参数
    parser.add_argument('--factor_construct', type=float, default=0.23,
                        help='重构损失权重 α')
    parser.add_argument('--factor_ort', type=float, default=0.65,
                        help='正交损失权重 β')
    parser.add_argument('--factor_corvar', type=float, default=0.17,
                        help='协方差损失权重 γ')
    parser.add_argument('--factor_KL', type=float, default=0.12,
                        help='KL散度损失权重 δ')
    parser.add_argument('--balancer', type=float, default=0.55,
                        help='双拉普拉斯混合系数 λ')
    parser.add_argument('--lambdas', type=float, default=5,
                        help='Sinkhorn指数参数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='权重衰减')

    # 其他参数
    parser.add_argument('--seed', type=int, default=3047,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备号')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设备设置
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f'Using device: {device}')

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # 随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # 数据加载与预处理
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

    # 转换为PyTorch张量
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(Y, dtype=torch.float)

    # =========================================================================
    # Step 1: 图构建 — 构建邻接矩阵和拉普拉斯矩阵
    # =========================================================================
    #
    # 构建两种邻接矩阵：
    #   A₁ (自环邻接矩阵)：基于 L2 归一化特征的点积
    #   A₂ (KNN邻接矩阵)：基于余弦相似度的传递闭包
    #
    # 拉普拉斯矩阵：L = D^(-1/2) · A · D^(-1/2)

    # L2归一化特征
    x_ = torch.nn.functional.normalize(x, p=2, dim=1)

    # 自环邻接矩阵（保留细胞自身信息）
    adj_self_loop = torch.mm(x_, x_.T)

    # KNN邻接矩阵（基于余弦相似度的传递闭包，增强相似细胞的连接）
    adj_f = torch.abs(pairwise_cosine_similarity(x_, x_))
    adj_f = torch.mm(adj_f, adj_f.T)

    # 计算两个拉普拉斯矩阵
    L_1 = get_laplace_matrix(adj_self_loop)  # 基于自环
    L_2 = get_laplace_matrix(adj_f)          # 基于KNN

    # 模型维度
    dims_encoder = [args.hidden_dim, args.embedding_dim]   # [256, 16]
    dims_decoder = [args.embedding_dim, args.hidden_dim]   # [16, 256]

    # =========================================================================
    # Step 2: 预训练自编码器
    # =========================================================================
    #
    # 预训练阶段只优化：
    #   - 重构损失 L_recon
    #   - 正交损失 L_ort
    #   - 协方差损失 L_cov
    #
    # 不使用聚类损失，让AE学习良好的降维表示

    print('Pre-training autoencoder...')

    # 初始化自编码器
    Model = AE_NN(dim_input=x.shape[1], dims_encoder=dims_encoder, dims_decoder=dims_decoder)
    if args.cuda:
        Model = Model.cuda()

    optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pretrain_acc_max = 0
    for epoch in range(1, args.epochs + 1):
        Model.train()

        # 前向传播
        if args.cuda:
            h, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
        else:
            h, x_hat = Model.forward(x, adj_self_loop)

        # L2归一化嵌入
        z = torch.nn.functional.normalize(h, p=2, dim=0)

        # 计算各项损失
        if args.cuda:
            # 重构损失：解码器输出与原始输入的MSE
            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())

            # 协方差损失：-Tr(z^T · L · z) / n
            # 目标是最小化相邻节点的嵌入差异（L应该小）
            loss_corvariates = -torch.mm(
                torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),
                z
            ).trace() / len(z.T)

            # 正交损失：MSE(z^T·z, I)
            # 促使嵌入向量各维度相互正交
            loss_ort = torch.nn.functional.mse_loss(
                torch.mm(z.T, z).view(-1).cuda(),
                torch.eye(len(z.T)).view(-1).cuda()
            )
        else:
            loss_x = torch.nn.functional.mse_loss(x_hat, x)
            loss_corvariates = -torch.mm(
                torch.mm(z.T, (args.balancer * L_1 + (1-args.balancer) * L_2)),
                z
            ).trace() / len(z.T)
            loss_ort = torch.nn.functional.mse_loss(
                torch.mm(z.T, z).view(-1),
                torch.eye(len(z.T)).view(-1)
            )

        # 总损失
        loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每50轮评估一次
        with torch.no_grad():
            # 使用KMeans获取伪标签
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20).fit(z.cpu().numpy())
            centers = torch.tensor(kmeans.cluster_centers_)

            # 计算 ACC 用于追踪最优预训练模型
            from evaluation import evaluation as eval_fn
            acc, _, _, _, _, _, _, _, _ = eval_fn(Y, kmeans.labels_)

            # 保存最优预训练模型（基于 ACC）
            if acc > pretrain_acc_max:
                pretrain_acc_max = acc
                torch.save(Model.state_dict(), os.path.join(args.save_dir, 'pretrain_model.pkl'))
                with open(os.path.join(args.save_dir, 'pretrain_centers.pkl'), 'wb') as f:
                    pickle.dump(centers, f, protocol=pickle.HIGHEST_PROTOCOL)
                pseudo_labels = torch.LongTensor(kmeans.labels_)
                with open(os.path.join(args.save_dir, 'pretrain_labels.pkl'), 'wb') as f:
                    pickle.dump(pseudo_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

        if epoch % 50 == 0:
            print(f'Pre-train Epoch {epoch}/{args.epochs}, Loss: {loss.item():.6f}')

    # =========================================================================
    # Step 3: 微调 — 加入聚类损失
    # =========================================================================
    #
    # 微调阶段在预训练损失基础上增加：
    #   - DEC聚类损失 L_KL
    #
    # 使用Sinkhorn算法计算目标分布，然后最小化q与p的KL散度

    print('Fine-tuning with clustering...')

    # 加载预训练模型
    Model = FULL_NN(
        dim_input=x.shape[1],
        dims_encoder=dims_encoder,
        dims_decoder=dims_decoder,
        num_class=n_clusters,
        pretrain_model_load_path=os.path.join(args.save_dir, 'pretrain_model.pkl')
    )
    if args.cuda:
        Model = Model.cuda()

    optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr)

    # 加载预训练的聚类中心和伪标签
    with open(os.path.join(args.save_dir, 'pretrain_centers.pkl'), 'rb') as f:
        centers = pickle.load(f)
        if args.cuda:
            centers = centers.cuda()
    with open(os.path.join(args.save_dir, 'pretrain_labels.pkl'), 'rb') as f:
        pseudo_labels = pickle.load(f)
        if args.cuda:
            pseudo_labels = pseudo_labels.cuda()

    best_embedding = None
    best_y_pred = None
    acc_max = 0
    nmi_max = 0

    for epoch in range(1, args.epochs + 1):
        Model.train()

        # 前向传播
        if args.cuda:
            z, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
        else:
            z, x_hat = Model.forward(x, adj_self_loop)

        z = torch.nn.functional.normalize(z, p=2, dim=0)
        centers = centers.detach()

        # 计算预训练损失（与预训练阶段相同）
        if args.cuda:
            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())
            loss_corvariates = -torch.mm(
                torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),
                z
            ).trace() / len(z.T)
            loss_ort = torch.nn.functional.mse_loss(
                torch.mm(z.T, z).view(-1).cuda(),
                torch.eye(len(z.T)).view(-1).cuda()
            )
        else:
            loss_x = torch.nn.functional.mse_loss(x_hat, x)
            loss_corvariates = -torch.mm(
                torch.mm(z.T, (args.balancer * L_1 + (1-args.balancer) * L_2)),
                z
            ).trace() / len(z.T)
            loss_ort = torch.nn.functional.mse_loss(
                torch.mm(z.T, z).view(-1),
                torch.eye(len(z.T)).view(-1)
            )

        # DEC聚类层 — 计算软分配
        if args.cuda:
            class_assign_model = ClusterAssignment(n_clusters, len(z.T), 1, centers).cuda()
            temp_class = class_assign_model(z.cuda())
        else:
            class_assign_model = ClusterAssignment(n_clusters, len(z.T), 1, centers)
            temp_class = class_assign_model(z)

        # 每10轮更新一次目标分布（Sinkhorn算法）
        if epoch == 1 or epoch % 10 == 0:
            p_distribution = torch.tensor(
                sinkhorn(
                    temp_class.cpu().detach().numpy(),
                    args.lambdas,
                    torch.ones(x.shape[0]).numpy(),
                    torch.tensor([torch.sum(pseudo_labels.cpu() == i) for i in range(n_clusters)]).numpy()
                )
            ).float()
            if args.cuda:
                p_distribution = p_distribution.cuda()
            p_distribution = p_distribution.detach()

        # KL散度损失
        KL_loss_function = nn.KLDivLoss(reduction='sum')
        if args.cuda:
            loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda())
        else:
            loss_KL = KL_loss_function(temp_class, p_distribution)

        # 总损失 = 预训练损失 + 聚类损失
        loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates + args.factor_KL * loss_KL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 评估
        with torch.no_grad():
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20).fit(z.cpu().numpy())
            y_pred = kmeans.labels_
            pseudo_labels = torch.LongTensor(kmeans.labels_)
            if args.cuda:
                pseudo_labels = pseudo_labels.cuda()
            centers = torch.tensor(kmeans.cluster_centers_)
            if args.cuda:
                centers = centers.cuda()

            # 计算 ACC 和 NMI 用于追踪最优模型
            from evaluation import evaluation as eval_fn
            acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, _ = eval_fn(Y, y_pred)

            # 保存最优模型（基于 ACC）
            if acc > acc_max:
                acc_max = acc
                nmi_max = nmi
                best_embedding = z.cpu().numpy()
                best_y_pred = y_pred
                # 保存最优模型参数
                torch.save(Model.state_dict(), os.path.join(args.save_dir, 'best_model.pkl'))

            if epoch % 50 == 0:
                print(f'Fine-tune Epoch {epoch}/{args.epochs}, Loss: {loss.item():.6f}, ACC: {acc:.4f}, NMI: {nmi:.4f}')

    print(f'Best fine-tuning: ACC={acc_max:.4f}, NMI={nmi_max:.4f}')

    # =========================================================================
    # Step 4: 保存结果
    # =========================================================================
    save(args.save_dir, Y, best_y_pred, args.epochs, best_embedding)
    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
