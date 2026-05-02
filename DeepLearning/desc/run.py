# -*- coding: utf-8 -*-
"""
Unified DESC Model Interface for scBench
=========================================

DESC: Deep Embedded Single-cell Clustering

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import random
import scanpy as sc

# Add project root directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save

# Import desc from local package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc.models.desc import train as desc_train


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DESC: Deep Embedded Single-cell Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    # Model arguments
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters (used for resolution tuning)')
    parser.add_argument('--louvain_resolution', type=float, default=1.0,
                        help='Louvain resolution for clustering')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of neighbors for graph construction')
    parser.add_argument('--dims', type=str, default=None,
                        help='Encoder dimensions, e.g., "64,32"')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=300,
                        help='Number of pretraining epochs')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for clustering')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--tol', type=float, default=0.005,
                        help='Tolerance for stopping criterion')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (None for CPU)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load and preprocess data using standard interface
    print('Loading data...')
    X, Y, sf, adata = prepare_data_for_model(
        args.data_path,
        size_factors=True,
        filter_min_counts=True,
        logtrans_input=True,
        normalize_input=True
    )

    # Convert to numpy arrays
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {args.n_clusters}')

    # Prepare adata for DESC
    adata_desc = sc.AnnData(X)
    adata_desc.obs_names = adata.obs_names
    adata_desc.var_names = adata.var_names if hasattr(adata, 'var_names') else [f'gene_{i}' for i in range(X.shape[1])]

    # Scale data (DESC expects scaled data)
    sc.pp.scale(adata_desc, max_value=6)

    # Parse dims if provided
    dims = None
    if args.dims:
        dims = [X.shape[1]] + [int(d) for d in args.dims.split(',')]

    # Set GPU
    use_GPU = not args.no_cuda and args.gpu is not None
    GPU_id = args.gpu if use_GPU else None

    # Train DESC
    print('Training DESC model...')
    adata_result = desc_train(
        adata_desc,
        dims=dims,
        tol=args.tol,
        n_neighbors=args.n_neighbors,
        batch_size=args.batch_size,
        louvain_resolution=args.louvain_resolution,
        save_dir=args.save_dir,
        do_tsne=False,
        use_GPU=use_GPU,
        GPU_id=GPU_id,
        num_Cores=1,
        pretrain_epochs=args.pretrain_epochs,
        max_iter=args.max_iter,
        use_ae_weights=False,
        random_seed=args.seed,
        do_umap=False
    )

    # Get results
    resolution_key = f'desc_{args.louvain_resolution}'
    embedding_key = f'X_Embeded_z{args.louvain_resolution}'

    y_pred = np.array(adata_result.obs[resolution_key].astype(int))
    embedding = adata_result.obsm[embedding_key]

    # Save results using standard interface
    save(args.save_dir, Y, y_pred, args.max_iter, embedding)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
