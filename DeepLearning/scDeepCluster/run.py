# -*- coding: utf-8 -*-
"""
Unified scDeepCluster Model Interface for scBench
=================================================

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
from time import time

# Add project root directory to path for imports (add first so it takes precedence)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
# Add code/ directory for local model imports
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

from preprocess import prepare_data_for_model
from utils import save

# Set random seeds before importing TensorFlow/Keras
from numpy.random import seed
seed(2211)

import tensorflow as tf
tf.random.set_seed(2211)

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Import model components from code directory
from scDeepCluster import SCDeepCluster


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='scDeepCluster: Deep Clustering for scRNA-seq Data',
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
    parser.add_argument('--noise_sd', type=float, default=2.5,
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for clustering layer')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=400,
                        help='Number of pretraining epochs')
    parser.add_argument('--maxiter', type=int, default=20000,
                        help='Maximum number of iterations')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Coefficient of clustering loss')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='Tolerance for stopping criterion')
    parser.add_argument('--update_interval', type=int, default=0,
                        help='Update interval (0 for auto)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--ae_weights', type=str, default=None,
                        help='Path to pretrained autoencoder weights')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

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
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Get raw counts for ZINB loss - use HVG-filtered normalized data
    # adata.layers['norm_log'] contains normalized+log1p HVG data (1000 genes)
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

    # Set update interval
    if args.update_interval == 0:
        args.update_interval = int(X.shape[0] / args.batch_size)

    # Build model dimensions
    dims = [X.shape[1]] + args.dims

    # Initialize model
    print('Initializing scDeepCluster model...')
    model = SCDeepCluster(
        dims=dims,
        n_clusters=n_clusters,
        noise_sd=args.noise_sd,
        alpha=args.alpha
    )

    print('Model summary:')
    model.autoencoder.summary()

    t0 = time()

    # Pretrain autoencoder
    if args.ae_weights is None:
        print('Pretraining autoencoder...')
        from tensorflow.keras.optimizers import Adam
        optimizer1 = Adam(amsgrad=True)
        ae_weight_file = os.path.join(args.save_dir, 'ae_weights.h5')
        model.pretrain(
            x=[X, sf],
            y=raw_counts,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            optimizer=optimizer1,
            ae_file=ae_weight_file
        )

    # Train clustering model
    print('Training clustering model...')
    y_pred = model.fit(
        x_counts=X,
        sf=sf,
        y=Y,
        raw_counts=raw_counts,
        batch_size=args.batch_size,
        tol=args.tol,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        ae_weights=args.ae_weights,
        save_dir=args.save_dir,
        loss_weights=[args.gamma, 1],
        optimizer='adadelta'
    )

    # Extract embeddings
    embedding = model.extract_feature([X, sf])

    # Save results using standard interface
    save(args.save_dir, Y, y_pred, args.maxiter, embedding)

    print(f'Training completed in {int(time() - t0)} seconds.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
