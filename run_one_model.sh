#!/bin/bash
#====================================================================================================
# run_one_model.sh — 单模型单数据集运行脚本
#====================================================================================================
#
# 【脚本功能】
#     运行一个模型在一个数据集上的完整流程，包括：
#     - 环境设置（Conda、GPU、缓存目录）
#     - 数据集参数识别（细胞类型数）
#     - 模型执行（调用对应的 Python 脚本）
#     - 日志记录
#
# 【使用方法】
#     ./run_one_model.sh <model> <dataset> <epochs>
#     示例：
#         ./run_one_model.sh scCDCG Arabidopsis_scRNA_synthetic 200
#         ./run_one_model.sh scMAE HumanPancreas_1 100
#         ./run_one_model.sh Leiden MousePancreas_Aging 0
#
# 【支持的模型】
#     scCDCG — 图神经网络聚类（深度学习）
#     scMAE  — 掩码自编码器（深度学习）
#     scDCC  — 深度约束聚类（深度学习）
#     DEC    — 深度嵌入聚类（深度学习）
#     Leiden — 社区检测（传统方法）
#
# 【数据集配置】
#     每个数据集需要指定真实的细胞类型数（n_clusters），
#     用于模型训练时的聚类层初始化。

set -e

#====================================================================================================
# 命令行参数解析
#====================================================================================================
MODEL=$1       # 模型名称
DATASET=$2     # 数据集名称
EPOCHS=${3:-200}  # 训练轮次（默认 200）
GPU=${GPU:-0}  # GPU 设备号（默认 0，可通过环境变量覆盖）

#====================================================================================================
# 路径配置
#====================================================================================================
REPO_ROOT=~/biopipeline/dimension-reduction/scCluBench-main  # 代码仓库根目录
DATA_DIR=/data/luolie/biopipeline/scCluBench/data          # 数据文件目录
RESULTS_DIR=/data/luolie/biopipeline/dimension-reduction/scCluBench/results  # 结果保存目录
LOGS_DIR=/data/luolie/biopipeline/dimension-reduction/scCluBench/logs        # 日志目录
DATA_PATH=${DATA_DIR}/${DATASET}.h5ad  # 完整数据文件路径

#====================================================================================================
# 数据集 → 聚类数映射表
#====================================================================================================
# 每个数据集的真实细胞类型数（ground truth）
# 这是聚类任务的真实标签数，用于：
#   - Leiden: 分辨率参数搜索的目标
#   - 深度学习: 聚类层的初始化
#
# | 数据集名称                    | 聚类数 |
# |------------------------------|--------|
# | Arabidopsis_scRNA_synthetic   | 8      |
# | Arabidopsis_Stereo-seq_leaf  | 6      |
# | Arabidopsis_Stereo-seq_leaf_S1-2 | 6  |
# | Plant_scRNA_synthetic         | 8      |
# | HumanPancreas_1              | 7      |
# | HumanPancreas_2              | 10     |
# | MousePancreas_Aging          | 9      |
# | TabulaSapiens_Pancreas       | 23     |
# | Blood_BoneMarrow            | 35     |

case $DATASET in
    Arabidopsis_scRNA_synthetic) N_CLUSTERS=8 ;;
    Arabidopsis_Stereo-seq_leaf) N_CLUSTERS=6 ;;
    Arabidopsis_Stereo-seq_leaf_S1-2) N_CLUSTERS=6 ;;
    Plant_scRNA_synthetic) N_CLUSTERS=8 ;;
    HumanPancreas_1) N_CLUSTERS=7 ;;
    HumanPancreas_2) N_CLUSTERS=10 ;;
    MousePancreas_Aging) N_CLUSTERS=9 ;;
    TabulaSapiens_Pancreas) N_CLUSTERS=23 ;;
    Blood_BoneMarrow) N_CLUSTERS=35 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

#====================================================================================================
# 环境检查
#====================================================================================================
# 验证数据文件存在
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# 创建结果和日志目录
mkdir -p ${RESULTS_DIR}/${MODEL}/${DATASET}
mkdir -p ${LOGS_DIR}

#====================================================================================================
# 环境变量设置
#====================================================================================================
# GPU 配置
export CUDA_VISIBLE_DEVICES=$GPU

# 缓存目录配置（避免占用默认 tmp 目录）
export TMPDIR=/data/luolie/biopipeline/dimension-reduction/scCluBench/tmp
# Hugging Face 缓存（Foundation Models 需要）
export TORCH_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/torch

#====================================================================================================
# Conda 环境激活
#====================================================================================================
# 尝试多个可能的 conda 安装位置
for conda_sh in \
    "/data/luolie/conda/base/etc/profile.d/conda.sh" \
    "/data/luolie/conda/pkgs/conda-26.1.1-py313h06a4308_0/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done
conda activate scclubench-main

#====================================================================================================
# 日志设置
#====================================================================================================
LOG_FILE=${LOGS_DIR}/${MODEL}_${DATASET}.log
echo "========================================" | tee $LOG_FILE
echo "Running $MODEL on $DATASET" | tee -a $LOG_FILE
echo "Epochs: $EPOCHS, Clusters: $N_CLUSTERS" | tee -a $LOG_FILE
echo "GPU: $GPU" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

#====================================================================================================
# 模型执行（根据 MODEL 选择对应的运行方式）
#====================================================================================================
case $MODEL in
    # -------------------------------------------------------------------------
    # scCDCG — 图神经网络聚类方法
    # -------------------------------------------------------------------------
    # 特点：
    #   - 基于图注意力自动编码器
    #   - 使用双拉普拉斯矩阵保持图结构
    #   - DEC 风格的聚类目标分布
    #
    # 超参数：
    #   - embedding_dim=16: 嵌入维度
    #   - hidden_dim=256: 隐藏层维度
    #   - lr=1e-3: 学习率
    # -------------------------------------------------------------------------
    scCDCG)
        cd ${REPO_ROOT}/GNN/scCDCG
        python run.py \
            --data_path $DATA_PATH \
            --n_clusters $N_CLUSTERS \
            --save_dir ${RESULTS_DIR}/${MODEL}/${DATASET} \
            --epochs $EPOCHS \
            --seed 42 \
            --embedding_dim 16 \
            --hidden_dim 256 \
            --lr 1e-3 \
            2>&1 | tee -a $LOG_FILE
        ;;

    # -------------------------------------------------------------------------
    # scMAE — 掩码自编码器
    # -------------------------------------------------------------------------
    # 特点：
    #   - 受 Vision Transformer 启发
    #   - 随机掩码部分基因，预测被掩码的值
    #   - 学习基因间的潜在关系
    #
    # 超参数：
    #   - hidden_size=128: 隐藏层维度
    #   - batch_size=256: 批大小
    # -------------------------------------------------------------------------
    scMAE)
        cd ${REPO_ROOT}/DeepLearning/scMAE
        python run.py \
            --data_path $DATA_PATH \
            --n_clusters $N_CLUSTERS \
            --save_dir ${RESULTS_DIR}/${MODEL}/${DATASET} \
            --epochs $EPOCHS \
            --seed 42 \
            --hidden_size 128 \
            --batch_size 256 \
            2>&1 | tee -a $LOG_FILE
        ;;

    # -------------------------------------------------------------------------
    # scDCC — 深度约束聚类
    # -------------------------------------------------------------------------
    # 特点：
    #   - 基于变分自编码器
    #   - ZINB 损失建模 scRNA-seq 的过度离散
    #   - 支持 Must-Link/Cannot-Link 约束
    # -------------------------------------------------------------------------
    scDCC)
        cd ${REPO_ROOT}/DeepLearning/scDCC
        python run.py \
            --data_path $DATA_PATH \
            --n_clusters $N_CLUSTERS \
            --save_dir ${RESULTS_DIR}/${MODEL}/${DATASET} \
            --epochs $EPOCHS \
            2>&1 | tee -a $LOG_FILE
        ;;

    # -------------------------------------------------------------------------
    # DEC — 深度嵌入聚类
    # -------------------------------------------------------------------------
    # 特点：
    #   - 经典深度聚类方法
    #   - 预训练 AE + 微调聚类
    #   - 使用 t-SNE 初始化聚类中心
    # -------------------------------------------------------------------------
    DEC)
        cd ${REPO_ROOT}/DeepLearning/dec
        python run.py \
            --data_path $DATA_PATH \
            --n_clusters $N_CLUSTERS \
            --save_dir ${RESULTS_DIR}/${MODEL}/${DATASET} \
            --epochs $EPOCHS \
            2>&1 | tee -a $LOG_FILE
        ;;

    # -------------------------------------------------------------------------
    # Leiden — 社区检测（传统方法基线）
    # -------------------------------------------------------------------------
    # 特点：
    #   - Louvain 算法的改进版
    #   - 快速、确定性
    #   - 无需训练，作为深度学习的对比基线
    #
    # 参数：
    #   - 无需 epochs（无训练过程）
    # -------------------------------------------------------------------------
    Leiden)
        cd ${REPO_ROOT}
        python run_leiden.py \
            "$DATA_PATH" \
            "${RESULTS_DIR}" \
            "${DATASET}" \
            "${N_CLUSTERS}" \
            2>&1 | tee -a $LOG_FILE
        ;;

    # -------------------------------------------------------------------------
    # 未知模型
    # -------------------------------------------------------------------------
    *)
        echo "Unknown model: $MODEL"
        echo "Available models: scCDCG, scMAE, scDCC, DEC, Leiden"
        exit 1
        ;;
esac

echo "Done. Results in ${RESULTS_DIR}/${MODEL}/${DATASET}/"
