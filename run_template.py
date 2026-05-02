# -*- coding: utf-8 -*-
"""
Unified Model Running Template for scBench
==========================================

This template defines the standard interface that all models should follow.
Each model should implement a `run.py` file based on this template.

Standard Interface:
- Input: h5ad file path, n_clusters, epochs, save_dir, etc.
- Output: metrics JSON, embedding h5/npy, predictions CSV

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --epochs 200 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import prepare_data_for_model
from utils import save
from evaluation import evaluation


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified scRNA-seq Clustering Model',
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
    parser.add_argument('--n_z', type=int, default=32,
                        help='Dimension of latent space')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 64],
                        help='Hidden layer dimensions')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Number of pretraining epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f'Using device: {device}')

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
    X = np.array(X)
    Y = np.array(Y)
    sf = np.array(sf)

    # Get number of clusters from data if not specified
    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # ============================================================
    # MODEL-SPECIFIC CODE STARTS HERE
    # Replace this section with your model's training code
    # ============================================================

    # Example: Import your model
    # from model import YourModel

    # Example: Initialize model
    # model = YourModel(
    #     n_input=X.shape[1],
    #     n_z=args.n_z,
    #     n_clusters=n_clusters,
    #     hidden_dims=args.hidden_dims
    # ).to(device)

    # Example: Training loop
    # for epoch in range(args.epochs):
    #     # Train model
    #     embedding, y_pred = model.fit(X, ...)
    #
    #     # Evaluate and save results using standard interface
    #     if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
    #         save(args.save_dir, Y, y_pred, epoch, embedding)

    # ============================================================
    # MODEL-SPECIFIC CODE ENDS HERE
    # ============================================================

    # Placeholder for demonstration
    print('Model training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
