# -*- coding: utf-8 -*-
"""
Unified GeneCompass Model Interface for scCluBench
================================================

GeneCompass: Foundation model for single-cell analysis
Yang et al., 2023

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocess import prepare_data_for_model
from utils import save


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='GeneCompass: Single-cell Foundation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

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

    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Try to use GeneCompass
    try:
        from GeneCompass import GeneCompassModel, GeneCompassTokenizer

        print('Initializing GeneCompass...')
        tokenizer = GeneCompassTokenizer()
        model = GeneCompassModel(model_size='base', device=device)

        # Prepare data
        adata_gc = adata.copy()
        n_hvg = min(3000, X.shape[1])
        sc.pp.highly_variable_genes(adata_gc, n_top_genes=n_hvg)
        hvg_genes = adata_gc.var_names[adata_gc.var.highly_variable].tolist()
        adata_gc = adata_gc[:, hvg_genes].copy()
        sc.pp.normalize_total(adata_gc, target_sum=1e4)
        sc.pp.log1p(adata_gc)

        # Tokenize
        tokens = tokenizer.tokenize(adata_gc)
        embeddings = model.extract_embeddings(tokens)

        if isinstance(embeddings, torch.Tensor):
            z = embeddings.cpu().numpy()
        else:
            z = np.array(embeddings)

        print(f'GeneCompass embedding shape: {z.shape}')

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
        y_pred = kmeans.fit_predict(z)

        save(args.save_dir, Y, y_pred, 0, z)
        print('GeneCompass completed.')

    except ImportError:
        print('GeneCompass not available. Install from:')
        print('  git+https://github.com/xCompass-AI/GeneCompass.git')
        print('Using PCA embedding as fallback...')

        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        pca = PCA(n_components=min(50, X.shape[1]), random_state=args.seed)
        z = pca.fit_transform(X)
        print(f'PCA embedding shape: {z.shape}')

        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
        y_pred = kmeans.fit_predict(z)

        save(args.save_dir, Y, y_pred, 0, z)
        print('PCA baseline completed.')


if __name__ == '__main__':
    main()
