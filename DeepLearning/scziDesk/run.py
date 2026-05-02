# -*- coding: utf-8 -*-
"""
Unified scziDesk Model Interface for scBench
=============================================

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save

# Import TensorFlow 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from network import autoencoder


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='scziDesk: Zero-Inflated Deep Embedding for scRNA-seq Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    # Model arguments
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters')
    parser.add_argument('--dims', type=int, nargs='+', default=[256, 64, 32],
                        help='Hidden layer dimensions (excluding input dim)')
    parser.add_argument('--distribution', type=str, default='ZINB',
                        choices=['ZINB', 'NB'],
                        help='Distribution type for reconstruction loss')
    parser.add_argument('--self_training', type=bool, default=True,
                        help='Whether to use self-training')
    parser.add_argument('--t_alpha', type=float, default=1.0,
                        help='Alpha parameter for t-distribution')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Weight for kmeans loss')
    parser.add_argument('--gamma', type=float, default=0.001,
                        help='Weight for KL loss')
    parser.add_argument('--noise_sd', type=float, default=1.5,
                        help='Standard deviation of Gaussian noise')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=1000,
                        help='Number of pretraining epochs')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--update_epoch', type=int, default=10,
                        help='Update interval for checking convergence')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='Tolerance for stopping criterion')

    # Other arguments
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')

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
    sf = np.array(sf).astype(np.float32).reshape(-1, 1)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Get raw counts for ZINB loss - use HVG-filtered normalized data
    if 'norm_log' in adata.layers:
        raw_counts = adata.layers['norm_log']
        if hasattr(raw_counts, 'toarray'):
            raw_counts = raw_counts.toarray()
        raw_counts = np.array(raw_counts).astype(np.float32)
    elif adata.raw is not None:
        raw_counts = adata.raw.X
        if hasattr(raw_counts, 'toarray'):
            raw_counts = raw_counts.toarray()
        raw_counts = np.array(raw_counts).astype(np.float32)
    else:
        raw_counts = X.copy()

    # Get number of clusters from data if not specified
    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # Build model dimensions
    dims = [X.shape[1]] + args.dims

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Initialize model
    print('Initializing scziDesk model...')
    model = autoencoder(
        dataname='scziDesk',
        distribution=args.distribution,
        self_training=args.self_training,
        dims=dims,
        cluster_num=n_clusters,
        t_alpha=args.t_alpha,
        alpha=args.alpha,
        gamma=args.gamma,
        learning_rate=args.lr,
        noise_sd=args.noise_sd
    )

    # Pretrain
    print('Pretraining...')
    model.pretrain(
        X=X,
        count_X=raw_counts,
        size_factor=sf,
        batch_size=args.batch_size,
        pretrain_epoch=args.pretrain_epochs,
        gpu_option=args.gpu
    )

    # Fine-tune
    print('Fine-tuning...')
    y_pred = model.funetrain(
        X=X,
        count_X=raw_counts,
        size_factor=sf,
        batch_size=args.batch_size,
        funetrain_epoch=args.epochs,
        update_epoch=args.update_epoch,
        error=args.tol
    )

    # Get embeddings
    embedding = model.latent_repre

    # Save results using standard interface
    save(args.save_dir, Y, y_pred, args.epochs, embedding)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
