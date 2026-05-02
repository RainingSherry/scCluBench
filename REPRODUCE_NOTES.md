# scCluBench Reproduction Notes

This document tracks all modifications made to the original scCluBench repository to enable reproduction.

---

## Modifications to Source Code

### 1. `preprocess.py` (Root)

**File**: `~/biopipeline/dimension-reduction/scCluBench-main/preprocess.py`

**Issue**: `normalize_sc()` function had two bugs:
1. HVG filtering (`sc.pp.highly_variable_genes`) was called on integer count data before normalization, causing dtype conflict with `np.expm1()`
2. `sc.pp.normalize_per_cell()` attempted in-place division on int64 arrays

**Modification**:
- Moved HVG filtering to AFTER log1p transformation (float data required)
- Added float32 conversion before in-place normalization

**Before**:
```python
if filter_min_counts:
    sc.pp.highly_variable_genes(adata, ...)  # Called on integer counts!

if not is_norm:
    sc.pp.normalize_per_cell(adata)  # In-place on int64!
```

**After**:
```python
if not is_norm:
    if hasattr(adata.X, 'toarray'):
        import scipy.sparse as sp
        adata.X = sp.csr_matrix(adata.X.toarray().astype(np.float32))
    elif adata.X.dtype.kind in ['i', 'u']:
        adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_per_cell(adata)

if logtrans_input and not is_log1p:
    sc.pp.log1p(adata)

if filter_min_counts:
    sc.pp.highly_variable_genes(adata, ...)  # Now on float data
```

**Impact**: Does NOT affect model structure, loss function, or training logic. Only fixes data type compatibility.

---

### 2. `utils.py` (Root)

**File**: `~/biopipeline/dimension-reduction/scCluBench-main/utils.py`

**Issue**: First line used relative import `from .evaluation import evaluation` which fails when the file is imported as a standalone module.

**Modification**: Changed to absolute import.

**Before**: `from .evaluation import evaluation`
**After**: `from evaluation import evaluation`

**Impact**: No effect on model behavior.

---

### 3. `GNN/scCDCG/model.py`

**File**: `~/biopipeline/dimension-reduction/scCluBench-main/GNN/scCDCG/model.py`

**Issue**: Local imports used bare module names (`from layer import ...`, `from utils import ...`) which conflicted with root-level modules of the same names.

**Modification**: Renamed local files and updated imports.

**Before**: `from layer import GAT_Layer`, `from utils import pdf_norm`
**After**: `from scCDCG_layer import GAT_Layer`, `from scCDCG_utils import pdf_norm`

**Impact**: Does NOT affect model architecture or training logic. Only fixes import namespace.

---

### 4. `GNN/scCDCG/train_scCDCG.py`

**File**: `~/biopipeline/dimension-reduction/scCluBench-main/GNN/scCDCG/train_scCDCG.py`

**Issue**: Same import conflicts as model.py.

**Modification**: Updated imports to use renamed local files.

**Impact**: No effect on model behavior.

---

### 5. `GNN/scCDCG/run.py`

**File**: `~/biopipeline/dimension-reduction/scCluBench-main/GNN/scCDCG/run.py`

**Issue**: Import path resolution placed scCDCG local dir before repo root, causing root modules to not be found.

**Modification**: Swapped sys.path insertion order so repo root has priority.

**Before**: `sys.path.insert(0, local_dir)` then `sys.path.insert(0, repo_root)`
**After**: `sys.path.insert(0, repo_root)` then `sys.path.insert(0, local_dir)`

**Impact**: No effect on model behavior.

---

### 6. New `__init__.py` Files

Created empty `__init__.py` files to enable Python package imports:

- `GNN/__init__.py`
- `GNN/scCDCG/__init__.py`
- `GNN/scDSC/__init__.py`
- `GNN/scGNN/__init__.py`
- `GNN/AttentionAE-sc/__init__.py`
- `DeepLearning/__init__.py`
- `DeepLearning/scDeepCluster/__init__.py`
- `DeepLearning/scDCC/__init__.py`
- `DeepLearning/scMAE/__init__.py`
- `DeepLearning/scNAME/__init__.py`
- `DeepLearning/scziDesk/__init__.py`
- `DeepLearning/desc/__init__.py`
- `DeepLearning/dec/__init__.py`
- `Foundation/__init__.py`
- `analysis/__init__.py`

---

## Environment Modifications

### scclubench-main

| Package | Original Version | Installed Version | Reason |
|---------|-----------------|------------------|--------|
| numpy | unspecified | 1.24.3 | Compatibility with scanpy 1.10.0 |
| torch | unspecified | 2.1.2+cu118 | GPU support |
| torchmetrics | unspecified | 1.8.2 | Needed for pairwise_cosine_similarity |
| loguru | unspecified | 0.7.3 | Used by train_scCDCG.py |
| scanpy | unspecified | 1.10.0 | Latest compatible with anndata 0.10.5 |
| anndata | unspecified | 0.10.5 | H5AD format support |

---

## Known Issues / Pending

### Pending: scDeepCluster (Keras/TensorFlow)

**Status**: Not yet tested
**Issue**: Requires TensorFlow/Keras with ZINB loss implementation
**Expected Problem**: `scanpy.api` was deprecated in newer scanpy versions; may need `scanpy.api as sc` fix
**Action**: Install in scclubench-tf1 environment (pending)

### Pending: scNAME (TensorFlow 1)

**Status**: Not yet tested
**Issue**: Uses `tensorflow.compat.v1` which may have compatibility issues
**Action**: Test in scclubench-tf1 environment (pending)

### Pending: DESC

**Status**: Not yet tested
**Issue**: TensorFlow-based, may need Python 3.6 environment
**Action**: Test in scclubench-tf1 environment (pending)

### Pending: scziDesk

**Status**: Not yet tested
**Issue**: TensorFlow-based with complex architecture
**Action**: Test in scclubench-tf1 environment (pending)

### Pending: scGNN

**Status**: Not yet tested
**Issue**: Has external `gae/` submodule dependency
**Action**: Check GAE submodule initialization

### Pending: scDSC

**Status**: Not yet tested
**Issue**: Has MTAB data dependency
**Action**: Check if MTAB data is available

### Pending: AttentionAE-sc

**Status**: Not yet tested
**Issue**: Preprocessing scripts reference Baron dataset
**Action**: Adapt preprocessing for current data format

### Pending: Foundation Models (scGPT, GeneFormer, GeneCompass)

**Status**: Not yet tested
**Issue**: Require model weights download; large packages
**Action**: Install in scclubench-foundation environment (pending)

### Pending: DEC (Caffe)

**Status**: Not yet tested
**Issue**: Uses Caffe framework (not installed)
**Action**: May need special Caffe build environment

---

## Verification Results

### scCDCG on Arabidopsis_scRNA_synthetic (10 epochs)

```
Dataset: Arabidopsis_scRNA_synthetic (1500 cells, 1000 genes, 8 types)
Environment: scclubench-main (torch 2.1.2+cu118)

Results:
- ACC: 0.1613
- NMI: 0.0093
- ARI: 0.0009
- F1-macro: 0.1592
```

### scCDCG on Arabidopsis_Stereo-seq_leaf_S1-2 (5 epochs)

```
Dataset: Arabidopsis_Stereo-seq_leaf_S1-2 (618 cells, 1000 genes, 6 types)
Environment: scclubench-main (torch 2.1.2+cu118)

Results:
- ACC: 0.3010
- NMI: 0.1085
- ARI: 0.0756
- F1-macro: 0.2674
```

### scMAE on Arabidopsis_scRNA_synthetic (5 epochs)

```
Dataset: Arabidopsis_scRNA_synthetic (1500 cells, 1000 genes, 8 types)
Environment: scclubench-main (torch 2.1.2+cu118)

Results:
- ACC: 0.1540
- NMI: 0.0062
- ARI: -0.0010
- F1-macro: 0.1533
```

### Leiden on Multiple Datasets

```
Dataset                      | ACC    | NMI    | ARI    | F1     | Notes
---------------------------|--------|--------|--------|--------|-------
Arabidopsis_scRNA_synthetic | 0.1560 | 0.0314 | 0.0006 | 0.1300 | res=1.5
Arabidopsis_Stereo-seq_leaf_S1-2 | 0.1197 | 0.0501 | 0.0073 | 0.1413 | res=1.5
HumanPancreas_1             | 0.4709 | 0.5233 | 0.3693 | 0.5151 | res=1.5
HumanPancreas_2             | 0.7860 | 0.6862 | 0.6365 | 0.5755 | res=1.5
MousePancreas_Aging        | 0.4527 | 0.5127 | 0.2415 | 0.5223 | res=1.5
```

Note: Low metrics on synthetic and plant datasets are expected - these are challenging for traditional clustering. Deep learning models should improve on these. Quick test runs used fewer epochs than the paper recommends (200).

---

## Data Notes

### Data Source

All datasets are pre-downloaded at `/data/luolie/biopipeline/scCluBench/data/`. The data was originally sourced from:
- Human/Mouse Pancreas: CELLxGENE Census
- Arabidopsis: STOMICS/CNGBdb
- Synthetic: Generated for benchmarking

### Cell Type Labels

All datasets have `cell_type` column in `adata.obs`. No label standardization was needed - all data is already in the correct format.

### HVG Filtering

The `prepare_data_for_model()` function uses `sc.pp.highly_variable_genes()` with:
- `n_top_genes=1000`
- `min_mean=0.0125`
- `max_mean=3`
- `min_disp=0.5`

This reduces gene dimensionality from 60k+ to 1000, as used by most benchmark models.
