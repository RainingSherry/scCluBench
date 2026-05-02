# scCluBench Runbook

## Quick Start

### Step 1: Activate Environment

```bash
conda activate scclubench-main
```

### Step 2: Set Environment Variables

```bash
export TMPDIR=/data/luolie/biopipeline/dimension-reduction/scCluBench/tmp
export HF_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/huggingface
export TORCH_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache/torch
export XDG_CACHE_HOME=/data/luolie/biopipeline/dimension-reduction/scCluBench/cache
export CUDA_VISIBLE_DEVICES=0
```

### Step 3: Verify Data

```bash
python - << 'PY'
import scanpy as sc, os
data_dir = "/data/luolie/biopipeline/scCluBench/data"
for f in sorted(os.listdir(data_dir)):
    if f.endswith('.h5ad'):
        ad = sc.read_h5ad(os.path.join(data_dir, f))
        print(f"{f}: {ad.X.shape[0]} cells, {ad.obs['cell_type'].nunique()} types")
PY
```

---

## Running Models

### Priority 1: scCDCG (Graph Neural Network)

**Dataset**: Arabidopsis_scRNA_synthetic (1500 cells, 8 types) - smallest for quick testing

```bash
cd ~/biopipeline/dimension-reduction/scCluBench-main/GNN/scCDCG

python run.py \
    --data_path /data/luolie/biopipeline/scCluBench/data/Arabidopsis_scRNA_synthetic.h5ad \
    --n_clusters 8 \
    --save_dir /data/luolie/biopipeline/dimension-reduction/scCluBench/results/scCDCG/Arabidopsis_scRNA_synthetic \
    --epochs 200 \
    --seed 42 \
    --embedding_dim 16 \
    --hidden_dim 256 \
    --lr 1e-3 \
    --weight_decay 5e-3 \
    --factor_construct 0.23 \
    --factor_ort 0.65 \
    --factor_corvar 0.17 \
    --factor_KL 0.12 \
    --balancer 0.55 \
    --lambdas 5 \
    2>&1 | tee /data/luolie/biopipeline/dimension-reduction/scCluBench/logs/scCDCG_Arabidopsis_scRNA_synthetic.log
```

**Larger dataset test** (Human Pancreas 1):

```bash
python run.py \
    --data_path /data/luolie/biopipeline/scCluBench/data/HumanPancreas_1.h5ad \
    --n_clusters 7 \
    --save_dir /data/luolie/biopipeline/dimension-reduction/scCluBench/results/scCDCG/HumanPancreas_1 \
    --epochs 200 \
    --seed 42 \
    --embedding_dim 16 \
    --hidden_dim 256 \
    2>&1 | tee /data/luolie/biopipeline/dimension-reduction/scCluBench/logs/scCDCG_HumanPancreas_1.log
```

---

### Priority 2: scMAE (Masked Autoencoder)

```bash
cd ~/biopipeline/dimension-reduction/scCluBench-main/DeepLearning/scMAE

python run.py \
    --data_path /data/luolie/biopipeline/scCluBench/data/Arabidopsis_scRNA_synthetic.h5ad \
    --n_clusters 8 \
    --save_dir /data/luolie/biopipeline/dimension-reduction/scCluBench/results/scMAE/Arabidopsis_scRNA_synthetic \
    --epochs 100 \
    --seed 42 \
    --hidden_size 128 \
    --mask_prob 0.4 \
    --masked_data_weight 0.75 \
    --mask_loss_weight 0.7 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee /data/luolie/biopipeline/dimension-reduction/scCluBench/logs/scMAE_Arabidopsis_scRNA_synthetic.log
```

---

### Priority 3: Leiden Clustering (Baseline)

```bash
cd ~/biopipeline/dimension-reduction/scCluBench-main

python - << 'PY'
import scanpy as sc, numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

data_path = "/data/luolie/biopipeline/scCluBench/data/Arabidopsis_scRNA_synthetic.h5ad"
adata = sc.read_h5ad(data_path)

# Simple Leiden clustering
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=1000, subset=True)
sc.pp.scale(adata)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
sc.tl.leiden(adata, resolution=0.5, random_state=42)

y_true = adata.obs['cell_type'].values
y_pred = adata.obs['leiden'].values.astype(int)

# Best mapping
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true_enc = le.fit_transform(y_true)

# Hungarian matching
from scipy.optimize import linear_sum_assignment
G = np.zeros((len(np.unique(y_true_enc)), len(np.unique(y_pred))))
for i in range(len(np.unique(y_true_enc))):
    for j in range(len(np.unique(y_pred))):
        G[i,j] = np.sum((y_true_enc == i) & (y_pred == j))
A = linear_sum_assignment(-G)
new_pred = np.zeros_like(y_pred)
for i in range(len(A[0])):
    new_pred[y_pred == A[1][i]] = A[0][i]

acc = accuracy_score(y_true_enc, new_pred)
nmi = nmi_score(y_true_enc, y_pred, average_method='arithmetic')
ari = ari_score(y_true_enc, y_pred)

print(f"Leiden: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
PY
```

---

## Dataset Reference

### Recommended Test Datasets (smallest first)

| Dataset | Cells | Genes | Cell Types | Best For |
|--------|-------|-------|-----------|---------|
| Arabidopsis_Stereo-seq_leaf_S1-2 | 618 | 18257 | 6 | Quick test |
| Arabidopsis_scRNA_synthetic | 1500 | 3000 | 8 | Synthetic, known labels |
| Arabidopsis_Stereo-seq_leaf | 721 | 18257 | 6 | Plant data |
| HumanPancreas_2 | 2126 | 61497 | 10 | Human pancreas |
| HumanPancreas_1 | 2544 | 61497 | 7 | Human pancreas |
| MousePancreas_Aging | 6201 | 53384 | 9 | Mouse pancreas |

### Large Datasets (use with caution)

| Dataset | Cells | Genes | Cell Types | Notes |
|--------|-------|-------|-----------|-------|
| TabulaSapiens_Pancreas | 14140 | 61497 | 23 | May need GPU with >40GB RAM |
| Blood_BoneMarrow | 15502 | 61497 | 35 | Large, 35 cell types |

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'layer'`

**Cause**: Import path conflict between scCDCG local `layer.py` and root modules.

**Fix**: The run.py uses a two-path approach: repo root for shared modules, scCDCG dir for local modules. This is already fixed in the current version. If you encounter this, make sure you:
1. Have `__init__.py` files in GNN/ and GNN/scCDCG/
2. The scCDCG local files are named `scCDCG_layer.py`, `scCDCG_utils.py`, `scCDCG_preprocess.py`

### Issue: `numpy.core._exceptions._UFuncOutputCastingError`

**Cause**: Integer count data cannot be modified in-place with float operations.

**Fix**: Fixed in `preprocess.py` - the `normalize_sc` function now converts integer arrays to float32 before normalization.

### Issue: `RuntimeWarning: invalid value encountered in power` in scCDCG

**Cause**: Some cells have zero variance after normalization, causing D^(-0.5) to produce NaN.

**Fix**: The `get_laplace_matrix` function uses `torch.nan_to_num()` to handle this. This warning is expected for some datasets and does not affect results.

### Issue: CUDA OOM on large datasets

**Fix**: 
1. Reduce batch size
2. Use CPU mode: add `--no_cuda` flag
3. Use smaller hidden dimensions
4. Split large datasets into chunks

---

## Output Format

All models save results in the same format:

```
save_dir/
├── metrics_{epoch}.json     # Evaluation metrics
├── embedding_{epoch}.npy    # Latent embeddings (n_cells × n_dims)
├── embedding.h5            # Embedding + predictions
└── types_{epoch}_pred.csv  # True and predicted labels
```

### metrics.json contents:

```json
{
    "acc": 0.85,
    "nmi": 0.78,
    "ari": 0.72,
    "f1_macro": 0.82,
    "fmi": 0.75,
    "v_measure": 0.78,
    "homogeneity": 0.75,
    "completeness": 0.82
}
```

---

## Checking Results

```bash
cat /data/luolie/biopipeline/dimension-reduction/scCluBench/results/scCDCG/Arabidopsis_scRNA_synthetic/metrics_200.json

python - << 'PY'
import pandas as pd, numpy as np
df = pd.read_csv("/data/luolie/biopipeline/dimension-reduction/scCluBench/results/scCDCG/Arabidopsis_scRNA_synthetic/types_200_pred.csv")
print(df.head(10))
print(f"Predicted clusters: {sorted(df['pred'].unique())}")
print(f"True cell types: {df['true'].nunique()}")
PY
```
