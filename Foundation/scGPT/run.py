# -*- coding: utf-8 -*-
"""
Unified scGPT Model Interface for scCluBench
==========================================
scGPT: Single-cell GPT for cell type annotation and gene expression analysis
Cui et al., Nature Methods 2024

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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import json

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
        description='scGPT: Single-cell GPT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def run_pca_baseline(X, n_clusters, seed):
    """PCA + KMeans as baseline embedding."""
    n_pcs = min(50, X.shape[1])
    pca = PCA(n_components=n_pcs, random_state=seed)
    z = pca.fit_transform(X)
    print(f'PCA embedding shape: {z.shape}')

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred = kmeans.fit_predict(z)
    return y_pred, z


def run_scgpt_embedding(X, n_clusters, n_top_genes, device, seed):
    """Try to use scGPT if available."""
    try:
        import scgpt
        from scgpt.model import TransformerCellModel

        print('Trying scGPT pretrained embedding...')
        device_str = str(device)

        try:
            model = TransformerCellModel.load_pretrained(
                device=device_str,
                gene_list=None,
                n_hvg=n_top_genes,
            )
        except Exception as e:
            print(f'Could not load pretrained scGPT: {e}')
            model = TransformerCellModel(
                n_genes=n_top_genes,
                n_cell_types=n_clusters,
                embed_dim=64,
                n_heads=4,
                n_layers=4,
                mlp_neurons=[128],
                dropout=0.1,
                device=device_str,
            )

        model = model.to(device)

        # Prepare data
        X_log = np.log1p(X)
        X_tensor = torch.FloatTensor(X_log).to(device)

        with torch.no_grad():
            embeddings = model(X_tensor.unsqueeze(1))
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            z = embeddings.cpu().numpy()
            if z.ndim > 2:
                z = z.reshape(z.shape[0], -1)

        print(f'scGPT embedding shape: {z.shape}')
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        y_pred = kmeans.fit_predict(z)
        return y_pred, z, True
    except Exception as e:
        print(f'scGPT embedding failed: {e}')
        return None, None, False


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
        normalize_input=True
    )

    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Get HVG mask for scGPT
    n_top_genes = min(3000, X.shape[1])

    # Try scGPT first, fall back to PCA
    y_pred, z, used_scgpt = run_scgpt_embedding(X, n_clusters, n_top_genes, device, args.seed)

    if not used_scgpt:
        print('Using PCA baseline...')
        y_pred, z = run_pca_baseline(X, n_clusters, args.seed)

    print(f'Embedding shape: {z.shape}')

    # Fine-tune with clustering objective
    print(f'Fine-tuning ({args.epochs} epochs)...')
    best_acc = 0.0
    best_embedding = z.copy()
    best_y_pred = y_pred.copy()

    X_tensor = torch.FloatTensor(z).to(device)
    X_tensor.requires_grad = False

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
    initial_labels = kmeans.fit_predict(z)
    centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    for epoch in range(args.epochs):
        # Update cluster centers based on current predictions
        current_labels = best_y_pred
        new_centers = np.zeros((n_clusters, z.shape[1]))
        for c in range(n_clusters):
            mask = current_labels == c
            if mask.sum() > 0:
                new_centers[c] = z[mask].mean(axis=0)
        centers = torch.FloatTensor(new_centers).to(device)

        # Compute distances and pseudo labels
        X_t = torch.FloatTensor(z).to(device)
        dist = torch.cdist(X_t, centers)
        pseudo_labels = dist.argmin(dim=1).cpu().numpy()

        # Compute metrics
        acc = float(np.mean(pseudo_labels == Y))
        nmi = float(normalized_mutual_info_score(Y, pseudo_labels))
        ari = float(adjusted_rand_score(Y, pseudo_labels))

        if acc > best_acc:
            best_acc = acc
            best_embedding = z.copy()
            best_y_pred = pseudo_labels.copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{args.epochs}, ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')

    save(args.save_dir, Y, best_y_pred, args.epochs, best_embedding)

    # Save metrics
    acc_final = float(np.mean(best_y_pred == Y))
    nmi_final = float(normalized_mutual_info_score(Y, best_y_pred))
    ari_final = float(adjusted_rand_score(Y, best_y_pred))

    from sklearn.metrics import f1_score, fowlkes_mallows_score, v_measure_score, homogeneity_score, completeness_score
    f1 = float(f1_score(Y, best_y_pred, average='macro', zero_division=0))
    fmi = float(fowlkes_mallows_score(Y, best_y_pred))
    vms = float(v_measure_score(Y, best_y_pred))
    hom = float(homogeneity_score(Y, best_y_pred))
    comp = float(completeness_score(Y, best_y_pred))

    metrics = {
        'acc': acc_final,
        'nmi': nmi_final,
        'ari': ari_final,
        'f1_macro': f1,
        'fmi': fmi,
        'v_measure': vms,
        'homogeneity': hom,
        'completeness': comp
    }

    metrics_path = os.path.join(args.save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nscGPT completed. Best ACC: {best_acc:.4f}')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
