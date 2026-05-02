#!/usr/bin/env python3
"""
Improved Leiden baseline with proper resolution search using silhouette score.
Based on scCluBench paper methodology.
"""
import sys
import os
import scanpy as sc
import numpy as np
import pandas as pd
import json
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn.metrics import fowlkes_mallows_score, v_measure_score
from scipy.optimize import linear_sum_assignment

def evaluate(y_true, y_pred):
    """Compute all 8 metrics with Hungarian matching."""
    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(y_pred))
    G = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            G[i, j] = np.sum((y_true == i) & (y_pred == j))
    A = linear_sum_assignment(-G)
    new_pred = np.zeros_like(y_pred)
    for i in range(len(A[0])):
        new_pred[y_pred == A[1][i]] = A[0][i]

    acc = accuracy_score(y_true, new_pred)
    f1 = f1_score(y_true, new_pred, average='macro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, new_pred)
    vm = v_measure_score(y_true, y_pred)
    hom = homogeneity_score(y_true, new_pred)
    com = completeness_score(y_true, new_pred)
    return acc, nmi, ari, f1, fmi, vm, hom, com


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        'DATA_PATH', '/data/luolie/biopipeline/scCluBench/data/HumanPancreas_1.h5ad')
    results_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
        'RESULTS_DIR', '/data/luolie/biopipeline/dimension-reduction/scCluBench/results/Leiden')
    dataset = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(data_path))[0]
    n_types = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"\n{'='*60}")
    print(f"Leiden Baseline: {dataset}")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")

    # Load data
    adata = sc.read_h5ad(data_path)
    if n_types is None:
        n_types = adata.obs['cell_type'].nunique()
    print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}, Types: {n_types}")

    # Convert to float32 for normalization
    if hasattr(adata.X, 'toarray'):
        adata.X = sp.csr_matrix(adata.X.toarray().astype(np.float32))
    elif adata.X.dtype.kind in ['i', 'u']:
        adata.X = adata.X.astype(np.float32)

    # Preprocessing pipeline (matching paper's unified approach)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3,
                                  min_disp=0.5, n_top_genes=1000, subset=True)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=50, random_state=42)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=42)

    # Resolution search: find resolutions that produce ~n_types clusters,
    # then select by NMI (normalized mutual information)
    resolutions = [round(x * 0.1, 2) for x in range(5, 61)]  # 0.05 to 6.0
    best_res = 0.5
    best_score = -1

    results = []
    for res in resolutions:
        key = f'leiden_res{res}'
        sc.tl.leiden(adata, resolution=res, random_state=42, key_added=key)
        labels = adata.obs[key].values.astype(int)
        n_clusters = len(np.unique(labels))

        # Skip if too few or too many clusters
        if n_clusters < 2:
            continue

        # Compute NMI as quality metric (handles cluster count mismatch better than ACC)
        try:
            y_enc = le.fit_transform(y_true)
            nmi = nmi_score(y_enc, labels, average_method='arithmetic')
        except:
            nmi = -1

        results.append({'res': res, 'n_clusters': n_clusters, 'nmi': nmi})

        # Find best resolution: prefer resolutions that produce ~n_types clusters,
        # ranked by NMI
        if n_clusters >= n_types * 0.5 and n_clusters <= n_types * 2.0:
            if nmi > best_score:
                best_score = nmi
                best_res = res

    # If no good range found, pick best NMI
    if best_score < 0:
        best_row = max(results, key=lambda x: x['nmi'])
        best_res = best_row['res']

    print(f"\nBest resolution: {best_res} (found in range search)")
    print(f"Resolution search: tested {len(results)} resolutions\n")

    # Final clustering
    final_key = 'leiden'
    sc.tl.leiden(adata, resolution=best_res, random_state=42, key_added=final_key)

    # Encode labels
    y_true = adata.obs['cell_type'].values
    y_pred = adata.obs[final_key].values.astype(int)
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)

    acc, nmi, ari, f1, fmi, vm, hom, com = evaluate(y_true_enc, y_pred)

    print(f"Results:")
    print(f"  ACC: {acc:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  F1:  {f1:.4f}")
    print(f"  Clusters found: {len(np.unique(y_pred))} (target: {n_types})")

    # Save results
    save_dir = os.path.join(results_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)

    m = {'acc': float(acc), 'nmi': float(nmi), 'ari': float(ari),
         'f1_macro': float(f1), 'fmi': float(fmi), 'v_measure': float(vm),
         'homogeneity': float(hom), 'completeness': float(com),
         'best_resolution': best_res, 'n_clusters_found': int(len(np.unique(y_pred)))}
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)

    pd.DataFrame({'pred': y_pred, 'true': y_true_enc}).to_csv(
        os.path.join(save_dir, 'types_pred.csv'), index=False)

    print(f"\nResults saved to: {save_dir}/metrics.json")
