# scCluBench Environment Setup Guide

## Overview

This document describes the conda environments used for reproducing the scCluBench paper (AAAI 2026). The repository contains diverse models requiring different dependency versions.

## Hardware

- **GPU**: 3x NVIDIA H100 80GB HBM3
- **CUDA**: 12.8 (driver 570.133.20)
- **System**: Linux 5.15.0-94-generic, Python 3.13 (base)

---

## Environment A: scclubench-main

**Purpose**: Primary environment for preprocessing, evaluation, modern PyTorch models (scCDCG, scMAE, scDCC, etc.)

### Creation

```bash
conda create -n scclubench-main python=3.9 -y -c conda-forge
conda activate scclubench-main
```

### Installation

```bash
# PyTorch 2.1.2 with CUDA 11.8
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core scientific stack
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==1.24.3 scipy scikit-learn munkres tqdm h5py \
  scanpy==1.10.0 anndata==0.10.5 \
  matplotlib seaborn pandas pyyaml \
  leidenalg louvain umap-learn networkx jgraph \
  torchmetrics loguru

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Models Using This Environment

- scCDCG (priority)
- scMAE
- scDCC
- DEC
- scDSC
- scGNN
- AttentionAE-sc
- Leiden / Louvain baselines

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.2+cu118 | PyTorch with CUDA |
| scanpy | 1.10.0 | Single-cell preprocessing |
| anndata | 0.10.5 | AnnData format |
| scikit-learn | 1.6.1 | Clustering, metrics |
| torchmetrics | 1.8.2 | Pairwise cosine similarity |
| loguru | 0.7.3 | Logging |

---

## Environment B: scclubench-sccdcg

**Purpose**: Reproducing scCDCG with older PyTorch version per original requirements.

### Creation

```bash
conda create -n scclubench-sccdcg python=3.8 -y -c conda-forge
conda activate scclubench-sccdcg
```

### Installation

```bash
# PyTorch 1.12.0 with CUDA 11.3
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --index-url https://download.pytorch.org/whl/cu113

# Core packages
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==1.19.5 pandas==1.3.5 scanpy==1.8.2 anndata h5py \
  scipy scikit-learn munkres tqdm matplotlib seaborn networkx \
  keras==2.4.3 torchmetrics loguru
```

### Note

The original scCDCG paper uses Keras 2.4.3, numpy 1.19.5, pandas 1.3.5, Scanpy 1.8.2, and torch 1.12.0. Python 3.8 is required for numpy 1.19.5 compatibility.

---

## Environment C: scclubench-tf1 (Optional)

**Purpose**: TensorFlow 1 models (scDeepCluster, DESC, scziDesk)

### Creation

```bash
conda create -n scclubench-tf1 python=3.6 -y -c conda-forge
conda activate scclubench-tf1
```

### Installation

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==1.19.5 scipy pandas==1.1.5 scikit-learn h5py==2.10.0 \
  keras==2.1.4 tensorflow-gpu==1.15.5 scanpy==1.4.6 anndata==0.7.8 munkres
```

### Note

TensorFlow 1.x is difficult to install on modern CUDA. If installation fails, use CPU mode or TF 1.15.5 which has better compatibility. Do not replace TensorFlow models with PyTorch alternatives.

---

## Environment D: scclubench-r (Optional)

**Purpose**: R-based methods (SC3, Seurat)

### Creation

```bash
conda create -n scclubench-r -c conda-forge r-base=4.2 -y
conda activate scclubench-r
```

### Installation

```bash
conda install -c conda-forge r-seurat r-irkernel r-devtools r-remotes -y
Rscript -e 'install.packages("BiocManager", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")'
Rscript -e 'BiocManager::install(c("SingleCellExperiment","SC3"), ask=FALSE, update=FALSE)'
```

---

## Environment E: scclubench-foundation

**Purpose**: Foundation models (scGPT, GeneFormer, GeneCompass)

### Creation

```bash
conda create -n scclubench-foundation python=3.10 -y -c conda-forge
conda activate scclubench-foundation
```

### Installation

```bash
# PyTorch (latest compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Foundation model packages
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scgpt geneformer

# GeneCompass (if available)
pip install git+https://github.com/xCompass-AI/GeneCompass.git
```

### Note

Foundation models require model weights that may need manual download. Check the respective GitHub repositories for pre-trained weights.

---

## Environment Variables

Set these in your shell profile or before each run:

```bash
export TMPDIR=/data/luolie/biopipeline/dimension-reduction/scCluBench/tmp
export HF_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/huggingface
export TRANSFORMERS_CACHE=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/huggingface
export TORCH_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/torch
export XDG_CACHE_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache
```

---

## Directory Structure

```
~/biopipeline/dimension-reduction/scCluBench-main/   # Code (git repo)
├── data -> /data/luolie/.../scCluBench/data/      # Symlink to data
├── results -> /data/luolie/.../scCluBench/results/ # Symlink to results
├── logs -> /data/luolie/.../scCluBench/logs/       # Symlink to logs
├── weights -> /data/luolie/.../scCluBench/weights/ # Symlink to weights
└── cache -> /data/luolie/.../scCluBench/cache/     # Symlink to cache

/data/luolie/biopipeline/dimension-reduction/scCluBench/  # Large data storage
├── data/          # .h5ad datasets
├── results/       # Model results
├── logs/          # Training logs
├── weights/       # Model weights
├── cache/         # HuggingFace/Torch cache
└── tmp/           # Temporary files
```

---

## Verified Status

| Environment | Status | Notes |
|------------|--------|-------|
| scclubench-main | ✅ Verified | PyTorch 2.1.2+cu118, scCDCG, scMAE, Leiden tested |
| scclubench-sccdcg | ⚙️ Partial | PyTorch 1.12.0+cu113 installed, not yet tested |
| scclubench-tf1 | ⏳ Pending | TensorFlow 1.x compatibility uncertain |
| scclubench-r | ⏳ Pending | R environment not yet built |
| scclubench-foundation | ⏳ Pending | Foundation model packages not yet installed |
