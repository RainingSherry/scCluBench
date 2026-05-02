#!/bin/bash
#====================================================================================================
# run_all_selected.sh — scCluBench 主运行脚本
#====================================================================================================
#
# 【脚本功能】
#     这是 benchmark 的主入口脚本，按顺序运行选定的模型在选定的数据集上。
#     它循环遍历每个模型 × 数据集组合，调用 run_one_model.sh 执行实际训练，
#     最后生成汇总结果表格。
#
# 【使用方法】
#     bash run_all_selected.sh
#
#     可通过环境变量覆盖默认设置：
#         GPU=0 EPOCHS=200 bash run_all_selected.sh
#
# 【输出】
#     - 各模型在各数据集上的 metrics.json 文件
#     - results_summary.csv — 所有结果的汇总表格
#
# 【设计模式】
#     - 外层循环：按优先级顺序运行模型（scCDCG → scMAE → Leiden）
#     - 内层循环：按数据集从小到大运行
#     - 小数据集先跑：减少调试时间，快速反馈

set -e

#====================================================================================================
# 环境变量设置（可覆盖）
#====================================================================================================
# GPU 设备号，默认使用 GPU 0
GPU=${GPU:-0}
# 训练轮次，默认 200
EPOCHS=${EPOCHS:-200}

echo "========================================"
echo "  scCluBench Selected Model Reproduction"
echo "========================================"
echo "GPU: $GPU"
echo "Epochs: $EPOCHS"
echo ""

#====================================================================================================
# 数据集列表（从小到大排列）
#====================================================================================================
# 设计原则：小数据集先跑，快速验证，早发现问题
#
# | 数据集名称                      | 细胞数 | 基因数 | 细胞类型数 |
# |--------------------------------|--------|--------|-----------|
# | Arabidopsis_scRNA_synthetic     | 1,500 | 3,000 | 8         |
# | Arabidopsis_Stereo-seq_leaf    | 721   | 18,257| 6         |
# | HumanPancreas_2                | 2,126 | 61,497| 10        |
# | HumanPancreas_1                | 2,544 | 61,497| 7         |
# | MousePancreas_Aging            | 6,201 | 53,384| 9         |

DATASETS=(
    "Arabidopsis_scRNA_synthetic"
    "Arabidopsis_Stereo-seq_leaf"
    "HumanPancreas_2"
    "HumanPancreas_1"
    "MousePancreas_Aging"
)

#====================================================================================================
# 模型列表（按优先级顺序）
#====================================================================================================
# 当前选定的三个模型：
#   - scCDCG:  图神经网络方法（深度学习）
#   - scMAE:   掩码自编码器（深度学习）
#   - Leiden:  传统社区检测方法（基线）

MODELS=(
    "scCDCG"
    "scMAE"
    "Leiden"
)

SCRIPT_DIR=$(dirname "$0")

#====================================================================================================
# Conda 环境检测
#====================================================================================================
# 尝试多个可能的 conda 安装位置，兼容不同的服务器配置
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

echo "Models to run: ${MODELS[*]}"
echo "Datasets to run: ${DATASETS[*]}"
echo ""

#====================================================================================================
# 主循环：模型 × 数据集
#====================================================================================================
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "========================================"
        echo "  Running $MODEL on $DATASET"
        echo "========================================"

        # 调用单模型运行脚本
        # 传递参数：模型名、数据集名、训练轮次
        bash "$SCRIPT_DIR/run_one_model.sh" "$MODEL" "$DATASET" "$EPOCHS"

        echo ""
        echo "  Result summary:"
        # 读取并打印该组合的评估结果
        METRICS_FILE="/data/luolie/biopipeline/dimension-reduction/scCluBench/results/${MODEL}/${DATASET}/metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            cat "$METRICS_FILE"
        else
            echo "  (metrics file not found)"
        fi
        echo ""
    done
done

echo ""
echo "========================================"
echo "  All runs complete!"
echo "========================================"

#====================================================================================================
# 生成汇总结果表格
#====================================================================================================
# 使用 Python 脚本从各 metrics.json 文件收集结果，生成 CSV 表格
python - << 'PY'
import os, json, csv
import pandas as pd

# 结果目录结构：results/{model}/{dataset}/metrics.json
results_dir = "/data/luolie/biopipeline/dimension-reduction/scCluBench/results"
summary_file = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "results_summary.csv")

rows = []
# 遍历所有模型和数据集
for model in os.listdir(results_dir):
    model_path = os.path.join(results_dir, model)
    if not os.path.isdir(model_path):
        continue
    for dataset in os.listdir(model_path):
        dataset_path = os.path.join(model_path, dataset)
        if not os.path.isdir(dataset_path):
            continue
        metrics_file = os.path.join(dataset_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                m = json.load(f)
            rows.append({
                'dataset': dataset,             # 数据集名称
                'model': model,                # 模型名称
                'ACC': m.get('acc', ''),      # 准确率
                'NMI': m.get('nmi', ''),      # 标准化互信息
                'ARI': m.get('ari', ''),       # 调整兰德指数
                'F1_macro': m.get('f1_macro', ''),  # 宏平均 F1
                'FMI': m.get('fmi', ''),       # 福克斯-马洛斯指数
                'v_measure': m.get('v_measure', ''),  # V 度量
                'homogeneity': m.get('homogeneity', ''),  # 同质性
                'completeness': m.get('completeness', ''),  # 完整性
                'status': 'SUCCESS',           # 运行状态
                'notes': ''                    # 备注
            })

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    print(df.to_string())
PY

echo ""
echo "Summary saved to results_summary.csv"
