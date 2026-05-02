# -*- coding: utf-8 -*-
"""
Pure Python SC3 Implementation for scCluBench
============================================
SC3: Single-Cell Consensus Clustering (Kiselev et al., 2017)
Implemented in pure Python (no R dependency required).

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, fowlkes_mallows_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocess import prepare_data_for_model
from utils import save


def build_consensus_matrix(X, n_clusters, n_pcs=20, seed=42):
    """
    Build consensus matrix by running multiple clustering methods.
    SC3 approach: run k-means and hierarchical clustering with different
    parameters, then aggregate into a consensus matrix.
    """
    n_cells = X.shape[0]
    consensus = np.zeros((n_cells, n_cells))

    # PCA for dimensionality reduction (X is already HVG-filtered, ~1000 genes)
    pca = PCA(n_components=min(n_pcs, n_cells - 1), random_state=seed)
    X_pca = pca.fit_transform(X)

    n_runs = 0

    # Run k-means with different k values around the target
    for k in range(max(2, n_clusters - 1), n_clusters + 2):
        if k > n_cells or k < 2:
            continue
        for n_init in [10, 20]:
            for rs in [seed, seed + 1, seed + 2]:
                try:
                    km = KMeans(n_clusters=k, n_init=n_init, random_state=rs)
                    labels = km.fit_predict(X_pca)
                    for i in range(n_cells):
                        for j in range(i + 1, n_cells):
                            if labels[i] == labels[j]:
                                consensus[i, j] += 1
                                consensus[j, i] += 1
                    n_runs += 1
                except Exception:
                    continue

    # Run hierarchical clustering with different linkages
    for metric in ['euclidean', 'cosine']:
        for linkage in ['ward', 'average', 'complete']:
            try:
                if metric == 'cosine':
                    X_norm = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-10)
                    dists = pdist(X_norm, metric='cosine')
                else:
                    dists = pdist(X_pca, metric=metric)
                Z = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
                labels = Z.fit_predict(squareform(dists))
                for i in range(n_cells):
                    for j in range(i + 1, n_cells):
                        if labels[i] == labels[j]:
                            consensus[i, j] += 1
                            consensus[j, i] += 1
                n_runs += 1
            except Exception:
                continue

    # Normalize
    if n_runs > 0:
        consensus /= n_runs
    else:
        consensus = np.eye(n_cells)

    np.fill_diagonal(consensus, 1.0)
    return consensus, X_pca


def parse_args():
    parser = argparse.ArgumentParser(
        description='SC3: Single-Cell Consensus Clustering (Pure Python)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print('Loading data...')
    X, Y, sf, adata = prepare_data_for_model(
        args.data_path,
        size_factors=False,
        filter_min_counts=True,
        logtrans_input=True,
        normalize_input=False
    )

    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    print('Building consensus matrix...')
    consensus, X_pca = build_consensus_matrix(X, n_clusters, n_pcs=20, seed=args.seed)

    print('Performing final clustering on consensus matrix...')
    try:
        dist_matrix = 1 - consensus
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)
        final_clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        y_pred = final_clustering.fit_predict(dist_matrix)
    except Exception:
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
        y_pred = kmeans.fit_predict(X_pca)

    print(f'Number of clusters found: {len(np.unique(y_pred))}')

    save(args.save_dir, Y, y_pred, 0, X_pca)

    # Metrics
    metrics_path = os.path.join(args.save_dir, 'metrics.json')

    from munkres import Munkres
    m = Munkres()
    D = max(int(y_pred.max()), int(Y.max())) + 1
    cost = np.zeros((D, D))
    for i in range(len(y_pred)):
        cost[int(y_pred[i]), int(Y[i])] -= 1
    assignment = m.compute(cost)
    y_map = {row: col for row, col in assignment}
    y_pred_aligned = np.array([y_map.get(int(p), int(p)) for p in y_pred])

    acc = float(np.mean(y_pred_aligned == Y))
    nmi = float(normalized_mutual_info_score(Y, y_pred))
    ari = float(adjusted_rand_score(Y, y_pred))
    f1 = float(f1_score(Y, y_pred, average='macro', zero_division=0))
    fmi = float(fowlkes_mallows_score(Y, y_pred))
    vms = float(v_measure_score(Y, y_pred))
    hom = float(homogeneity_score(Y, y_pred))
    comp = float(completeness_score(Y, y_pred))

    metrics = {
        'acc': acc,
        'nmi': nmi,
        'ari': ari,
        'f1_macro': f1,
        'fmi': fmi,
        'v_measure': vms,
        'homogeneity': hom,
        'completeness': comp
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nSC3 Results (Pure Python):')
    print(f'  ACC:        {acc:.4f}')
    print(f'  NMI:        {nmi:.4f}')
    print(f'  ARI:        {ari:.4f}')
    print(f'  F1-macro:   {f1:.4f}')
    print(f'  FMI:        {fmi:.4f}')
    print(f'  V-measure:  {vms:.4f}')
    print(f'\nResults saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
