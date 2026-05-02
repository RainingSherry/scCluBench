# -*- encoding: utf-8 -*-
"""
====================================================================================================
scCDCG 层定义和损失函数模块
====================================================================================================

本模块定义了 scCDCG 的核心层组件：

【包含内容】
    1. GAT_Layer — 图注意力层
    2. ZINBLoss — 零膨胀负二项分布损失
    3. GaussianNoise — 高斯噪声层
    4. MeanAct — 均值激活函数（指数激活）
    5. DispAct — 离散度激活函数（softplus激活）

【ZINB分布背景】
    单细胞RNA-seq数据具有两个特点：
    1. 过度离散（overdispersion）：方差远大于均值
    2. 零过多（zero-inflation）：零值比例远超泊松分布预测

    ZINB（零膨胀负二项）分布能很好地建模这两个特点：
    P(X=0) = π + (1-π) * NB(0; r, p)
    P(X=k) = (1-π) * NB(k; r, p)  for k >= 1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# GAT_Layer — 图注意力层
# =============================================================================

class GAT_Layer(torch.nn.Module):
    """
    图注意力层（Graph Attention Layer）

    【核心思想】
        与标准图卷积（GCN）不同，GAT使用注意力机制为不同邻居分配不同权重

    【数学公式】
        注意力系数：α_{ij} = softmax(LeakyReLU(a^T · [W·h_i || W·h_j]))
        节点更新：h_i' = σ(Σ_j α_{ij} · W·h_j)

    【参数】
        dim_in: 输入特征维度
        dim_out: 输出特征维度
        negative_slope: LeakyReLU的负斜率（默认0.2）
    """

    def __init__(self, dim_in, dim_out, negative_slope=0.2):
        super(GAT_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 可学习的线性变换矩阵 W
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))

        # 注意力机制的参数：目标节点和邻居节点的变换
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))

        # Xavier初始化（Gain=1.414）
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x, adj):
        """
        图注意力前向传播

        【参数】
            x: 节点特征 (num_nodes, dim_in)
            adj: 邻接矩阵 (num_nodes, num_nodes)

        【返回】
            更新后的节点特征 (num_nodes, dim_out)

        【计算流程】
            1. 线性变换：x_ = W · x
            2. 计算注意力系数：α_{ij} = LeakyReLU(a^T·[x_i || x_j])
            3. 掩码非连接边：将无关节点的注意力设为极小值
            4. Softmax归一化：α_{ij} = exp(α_{ij}) / Σ_k exp(α_{ik})
            5. 加权聚合：h_i' = Σ_j α_{ij} · x_j
            6. ELU激活
        """
        # 线性变换
        x_ = torch.mm(x, self.w)

        # 计算注意力分数
        scores_target = torch.mm(x_, self.a_target)  # [n, 1]
        scores_neighbor = torch.mm(x_, self.a_neighbor)  # [n, 1]
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1)  # [n, n]

        # 掩码：非邻居节点的注意力设为极小值
        scores = torch.mul(adj, scores)
        scores = self.leakyrelu(scores)
        scores = torch.where(
            adj > 0,
            scores,
            -9e15 * torch.ones_like(scores)
        )

        # Softmax归一化
        coefficients = torch.nn.functional.softmax(scores, dim=1)

        # 加权聚合 + ELU激活
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        return x_


# =============================================================================
# ZINBLoss — 零膨胀负二项分布损失
# =============================================================================

class ZINBLoss(nn.Module):
    """
    零膨胀负二项分布（ZINB）损失

    【ZINB分布的三个参数】
        1. π (pi): 零点的概率质量（由神经网络预测）
        2. r (dispersion): 离散度参数（由神经网络预测）
        3. μ (mean): 均值参数（由神经网络预测）

    【损失函数】
        L = -log P(x | π, r, μ)
          = -log(π * I[x=0] + (1-π) * NB(x; r, μ))

    【优势】
        - 能建模scRNA-seq的过度离散
        - 能建模scRNA-seq的零点过多
        - 保留了原始计数信息
    """

    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        """
        计算ZINB负对数似然

        【参数】
            x: 原始计数向量 (n_cells, n_genes)
            mean: 预测的均值 μ (n_cells, n_genes)
            disp: 预测的离散度 r (n_cells, n_genes)
            pi: 预测的零点概率 π (n_cells, n_genes)
            scale_factor: size factor校正 (n_cells, 1)
            ridge_lambda: L2正则化系数

        【返回】
            负对数似然（标量）
        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        # ============ 负二项分布部分 ============
        # lgamma(x+1) - lgamma(x+r) + lgamma(r)
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)

        # (r+x) * log(1 + μ/r) + x * log(r/μ)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean + eps)))

        nb_final = t1 + t2  # log NB(x; r, μ)

        # ============ ZINB组合 ============
        # 零点部分的NB概率
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)

        # 非零点部分的负对数似然
        nb_case = nb_final - torch.log(1.0 - pi + eps)
        # 零点部分的负对数似然
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # 根据实际值选择使用哪种情况
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        # L2正则化
        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


# =============================================================================
# GaussianNoise — 高斯噪声层
# =============================================================================

class GaussianNoise(nn.Module):
    """
    高斯噪声层 — 用于变分自编码器（VAE）的重参数化技巧

    【用途】
        在训练时向输入添加噪声，增强模型的鲁棒性
        常用于VAE中实现随机性

    【参数】
        sigma: 噪声的标准差
    """

    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        """仅在训练模式下添加噪声"""
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


# =============================================================================
# MeanAct — 均值激活函数
# =============================================================================

class MeanAct(nn.Module):
    """
    均值激活函数：exp(x)

    【用途】
        ZINB中，均值参数需要为正数
        使用指数激活确保均值始终为正

    【数学】
        y = exp(x), 范围 (0, +∞)
    """

    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        # 限制范围避免数值溢出
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


# =============================================================================
# DispAct — 离散度激活函数
# =============================================================================

class DispAct(nn.Module):
    """
    离散度激活函数：softplus(x)

    【用途】
        ZINB中，离散度参数需要为正数
        softplus是ReLU的平滑版本，确保正值

    【数学】
        y = log(1 + exp(x)), 范围 (0, +∞)

    【优势】
        - 处处可导
        - 不会像ReLU那样产生"死神经元"
    """

    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        # 限制范围避免数值问题
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
