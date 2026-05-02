# -*- encoding: utf-8 -*-
"""
====================================================================================================
scCDCG 模型定义模块
====================================================================================================

本模块定义了 scCDCG 的核心神经网络组件：

【模型架构】
    1. AE_GAT   — 基于图注意力层的自编码器（原始版本）
    2. AE_NN    — 基于全连接层的自编码器（推荐版本，效率更高）
    3. FULL      — 完整的GAT版本（预训练 + 微调）
    4. FULL_NN   — 完整的NN版本（推荐）
    5. ClusterAssignment — DEC风格的软聚类分配层

【核心设计理念】
    - 编码器：将高维基因表达降维到低维嵌入空间
    - 解码器：从低维嵌入重建原始表达
    - 损失函数：重构 + 正交 + 协方差 + 聚类
"""

from scCDCG_layer import GAT_Layer
import torch
from scCDCG_utils import pdf_norm
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
from scCDCG_layer import ZINBLoss, MeanAct, DispAct


# =============================================================================
# AE_GAT — 基于图注意力层的自编码器（原始版本）
# =============================================================================

class AE_GAT(torch.nn.Module):
    """
    使用图注意力层（GAT）的自编码器

    【网络结构】
        编码器：Linear → GAT_Layer → GAT_Layer → ... → 嵌入向量
        解码器：GAT_Layer → GAT_Layer → ... → 重建向量

    【特点】
        - 使用注意力机制加权聚合邻居信息
        - 能够学习细胞间的相对重要性
        - 计算开销较大

    【参数】
        dim_input: 输入基因数
        dims_encoder: 编码器各层维度 [256, embedding_dim]
        dims_decoder: 解码器各层维度 [embedding_dim, 256]
    """

    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_GAT, self).__init__()
        self.dims_en = [dim_input] + dims_encoder  # [输入, 256, 嵌入维度]
        self.dims_de = dims_decoder + [dim_input]  # [嵌入维度, 256, 输入]

        self.num_layer = len(self.dims_en) - 1

        # 构建编码器和解码器层
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        for index in range(self.num_layer):
            self.Encoder.append(GAT_Layer(self.dims_en[index], self.dims_en[index + 1]))
            self.Decoder.append(GAT_Layer(self.dims_de[index], self.dims_de[index + 1]))

    def forward(self, x, adj):
        """
        前向传播

        【参数】
            x: 输入特征 (n_cells, n_genes)
            adj: 邻接矩阵 (n_cells, n_cells)

        【返回】
            h: 编码器输出的嵌入向量
            x_hat: 解码器重建的向量
        """
        # 编码器
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x, adj)
        h = x

        # 解码器
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x, adj)
        x_hat = x

        return h, x_hat


# =============================================================================
# AE_NN — 基于全连接层的自编码器（推荐版本）
# =============================================================================

class AE_NN(torch.nn.Module):
    """
    使用全连接层的自编码器（scCDCG推荐版本）

    【与GAT版本的区别】
        - 不使用图注意力，而是简单的全连接层
        - 计算效率更高，适合中等规模数据集
        - 仍能学习良好的降维表示

    【网络结构】
        编码器：Linear(in=基因数, out=256) → Linear(256, 嵌入维度)
        解码器：Linear(嵌入维度, 256) → Linear(256, 基因数)

    【特点】
        - 轻量级，计算效率高
        - 无需显式图结构信息
        - 通过损失函数隐式学习图结构
    """

    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder  # [基因数, 256, 嵌入维度]
        self.dims_de = dims_decoder + [dim_input]  # [嵌入维度, 256, 基因数]

        self.num_layer = len(self.dims_en) - 1

        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        # 构建编码器和解码器层
        for index in range(self.num_layer):
            self.Encoder.append(torch.nn.Linear(self.dims_en[index], self.dims_en[index + 1]))
            self.Decoder.append(torch.nn.Linear(self.dims_de[index], self.dims_de[index + 1]))

    def forward(self, x, adj):
        """
        前向传播

        【注意】
            adj 参数在AE_NN中未被使用（保留接口一致性）
            图结构信息通过损失函数（L_cov）隐式融入训练
        """
        # 编码器
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x)
        h = x

        # 解码器
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x)
        x_hat = x

        return h, x_hat


# =============================================================================
# FULL — 完整的GAT版本
# =============================================================================

class FULL(torch.nn.Module):
    """
    完整的scCDCG模型（GAT版本）

    【组成】
        - 预训练的AE_GAT作为编码器/解码器
        - DEC风格的聚类预测

    【用途】
        完整的预训练+微调流程
    """

    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        # 加载预训练的AE
        self.AE = AE_GAT(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(
            torch.load(pretrain_model_load_path, map_location='cpu')
        )

    def forward(self, x, adj):
        """返回归一化的嵌入向量和重建向量"""
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        """
        基于vMF分布的聚类预测

        【参数】
            kappas: 浓度参数（控制分布的集中程度）
            centers: 聚类中心
            normalize_constants: 归一化常数
            mixture_cofficences: 混合系数

        【返回】
            p: 每个样本属于各聚类的概率
        """
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(
            torch.mul(mixture_cofficences, pdf_component), p=1, dim=1
        )
        return p


# =============================================================================
# FULL_NN — 完整的NN版本（推荐）
# =============================================================================

class FULL_NN(torch.nn.Module):
    """
    完整的scCDCG模型（NN版本，推荐使用）

    【组成】
        - 预训练的AE_NN作为编码器/解码器
        - 支持DEC风格的聚类预测

    【优势】
        - 比GAT版本计算效率更高
        - 适合大多数单细胞数据集
        - 保持相同的聚类性能
    """

    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL_NN, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        # 加载预训练的AE
        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(
            torch.load(pretrain_model_load_path, map_location='cpu')
        )

    def forward(self, x, adj):
        """
        前向传播

        【参数】
            x: 输入特征 (n_cells, n_genes)
            adj: 邻接矩阵（未使用，保留接口一致性）

        【返回】
            z: L2归一化的嵌入向量 (n_cells, embedding_dim)
            x_hat: 重建向量 (n_cells, n_genes)
        """
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        """基于vMF分布的聚类预测（与FULL相同）"""
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(
            torch.mul(mixture_cofficences, pdf_component), p=1, dim=1
        )
        return p


# =============================================================================
# ClusterAssignment — DEC风格的软聚类分配层
# =============================================================================

class ClusterAssignment(nn.Module):
    """
    DEC风格的软聚类分配层

    【论文出处】
        Xie, Girshick, Farhadi. "Unsupervised Deep Clustering..." (ICML 2016)

    【核心思想】
        使用Student's t分布作为核函数，计算样本到各聚类中心的相似度

    【数学公式】
        q_{ij} = (1 + ||z_i - μ_j||²/α)^(-(α+1)/2) / Σ_k (1 + ||z_i - μ_k||²/α)^(-(α+1)/2)

        其中：
            - z_i: 样本i的嵌入向量
            - μ_j: 聚类中心j
            - α: 自由度参数（默认=1）
            - q_{ij}: 样本i属于聚类j的软分配概率

    【特点】
        - 与k-means相比，提供软分配（概率而非硬标签）
        - α=1时退化为Student's t分布
        - 输出的概率分布用于KL散度损失
    """

    def __init__(
        self,
        cluster_number: int,              # 聚类数
        embedding_dimension: int,          # 嵌入维度
        alpha: float = 1.0,               # Student's t分布的自由度参数
        cluster_centers: Optional[torch.Tensor] = None,  # 初始聚类中心
    ) -> None:
        """
        初始化聚类分配层

        【参数】
            cluster_number: 聚类数量
            embedding_dimension: 嵌入向量维度
            alpha: Student's t分布的自由度参数（越大分布越尖锐）
            cluster_centers: 初始聚类中心，若为None则使用Xavier初始化
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha

        if cluster_centers is None:
            # Xavier均匀初始化
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        # 可学习的聚类中心参数
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        计算软聚类分配

        【参数】
            batch: 嵌入向量 (batch_size, embedding_dimension)

        【返回】
            FloatTensor (batch_size, cluster_number): 每个样本属于各聚类的概率

        【计算过程】
            1. 计算每个样本到各聚类中心的欧氏距离 ||z_i - μ_j||²
            2. 代入Student's t核函数
            3. 归一化为概率分布
        """
        # 计算 ||z_i - μ_j||² for all i,j
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)

        # 核函数：(1 + ||z-μ||²/α)^(-(α+1)/2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power

        # 归一化为概率分布
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
