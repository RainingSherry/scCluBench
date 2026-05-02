import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse


class AutoEncoder(torch.nn.Module):
    """
    scMAE 掩码自编码器模型

    【网络架构】
        ┌─────────────────────────────────────────────────────────┐
        │  输入 X (n_cells, n_genes)                              │
        │      │                                                  │
        │      ▼                                                  │
        │  ┌──────────────────────────────────────────┐           │
        │  │ Encoder                                  │           │
        │  │   Dropout                               │           │
        │  │   Linear(n_genes, 256)                  │           │
        │  │   LayerNorm(256)                        │           │
        │  │   Mish()                                │           │
        │  │   Linear(256, hidden_size)              │           │
        │  │   LayerNorm(hidden_size)                │           │
        │  │   Mish()                                │           │
        │  │   Linear(hidden_size, hidden_size)     │           │
        │  └──────────┬───────────────────────────┘           │
        │             │ latent (n_cells, hidden_size)           │
        │             │                                        │
        │             ▼                                        │
        │  ┌──────────────────────────────────────────┐           │
        │  │ Mask Predictor                           │           │
        │  │   Linear(hidden_size, n_genes)          │           │
        │  │   → predicted_mask (n_cells, n_genes)   │           │
        │  └──────────────────────────────────────────┘           │
        │             │                                        │
        │             ▼                                        │
        │  ┌──────────────────────────────────────────┐           │
        │  │ Decoder                                 │           │
        │  │   concat([latent, predicted_mask])      │           │
        │  │   → Linear(hidden_size+n_genes, n_genes)│           │
        │  │   → reconstruction (n_cells, n_genes)  │           │
        │  └──────────────────────────────────────────┘           │
        └─────────────────────────────────────────────────────────┘

    【特点】
        - 使用LayerNorm而非BatchNorm（适合单细胞数据的小批次特性）
        - 使用Mish激活函数（比ReLU更平滑）
        - 掩码预测器帮助学习基因间关系
        - 联合重构和掩码预测

    【参数】
        num_genes: 基因数量（输入维度）
        hidden_size: 隐藏层维度（默认128）
        dropout: Dropout率（默认0，即不使用）
        masked_data_weight: 被掩码数据的损失权重（默认0.75）
        mask_loss_weight: 掩码预测损失的权重（默认0.7）
    """

    def __init__(
        self,
        num_genes,
        hidden_size=128,
        dropout=0,
        masked_data_weight=.75,
        mask_loss_weight=0.7,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight

        # 编码器：多层全连接网络
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        # 掩码预测器：从嵌入预测被掩码的位置
        self.mask_predictor = nn.Linear(hidden_size, num_genes)

        # 解码器：拼接嵌入和掩码预测，重构原始输入
        # 输入维度 = hidden_size（嵌入） + num_genes（掩码预测）
        self.decoder = nn.Linear(
            in_features=hidden_size + num_genes,
            out_features=num_genes
        )

    def forward_mask(self, x):
        """
        带掩码的前向传播

        【参数】
            x: 输入数据 (n_cells, n_genes)

        【返回】
            latent: 嵌入向量 (n_cells, hidden_size)
            predicted_mask: 掩码预测 (n_cells, n_genes)
            reconstruction: 重建向量 (n_cells, n_genes)
        """
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        # 拼接嵌入和掩码预测作为解码器输入
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1)
        )
        return latent, predicted_mask, reconstruction

    def loss_mask(self, x, y, mask):
        """
        计算掩码损失

        【损失函数】
            L = L_reconstruction + L_mask

            1. 重构损失（加权MSE）：
               L_reconstruction = (1-mask_loss_weight) * mean(w_nums * MSE(reconstruction, y))

               其中 w_nums = mask * masked_data_weight + (1-mask) * (1-masked_data_weight)
               即被掩码位置的权重为masked_data_weight，未被掩码的为(1-masked_data_weight)

            2. 掩码损失（BCE）：
               L_mask = mask_loss_weight * BCE(predicted_mask, mask)
               促使模型学习识别被掩码的位置

        【参数】
            x: 输入数据（可以是损坏后的）
            y: 目标数据（原始未损坏的数据）
            mask: 掩码标记（1=被掩码，0=保持原样）

        【返回】
            latent: 嵌入向量
            loss: 总损失
        """
        latent, predicted_mask, reconstruction = self.forward_mask(x)

        # 计算权重：被掩码位置权重更高
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)

        # 加权MSE重构损失
        reconstruction_loss = (1 - self.mask_loss_weight) * torch.mul(
            w_nums, mse(reconstruction, y, reduction='none')
        )

        # BCE掩码损失
        mask_loss = self.mask_loss_weight * bce_logits(predicted_mask, mask, reduction="mean")

        reconstruction_loss = reconstruction_loss.mean()
        loss = reconstruction_loss + mask_loss

        return latent, loss

    def feature(self, x):
        """
        提取特征（用于聚类）

        【用途】
            训练完成后，使用此方法获取细胞的嵌入向量，
            然后用KMeans或Leiden进行聚类

        【参数】
            x: 输入数据 (n_cells, n_genes)

        【返回】
            latent: 嵌入向量 (n_cells, hidden_size)
        """
        latent = self.encoder(x)
        return latent
