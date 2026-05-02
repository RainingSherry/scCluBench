# -*- coding: utf-8 -*-
"""
Unified scGNN Model Interface for scBench
==========================================

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.cluster import KMeans

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save

# Import model components
from model import AE, VAE
from graph_function import generateAdj
from benchmark_util import generateLouvainCluster


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


class scDataset(Dataset):
    """Dataset class for scRNA-seq data."""
    def __init__(self, features):
        self.features = features
        if isinstance(features, np.ndarray):
            self.features = torch.FloatTensor(features)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], idx


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='scGNN: Graph Neural Network for scRNA-seq Clustering',
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
    parser.add_argument('--model_type', type=str, default='AE',
                        choices=['AE', 'VAE'],
                        help='Model type: AE or VAE')

    # Graph arguments
    parser.add_argument('--k', type=int, default=10,
                        help='Number of neighbors for KNN graph')
    parser.add_argument('--knn_distance', type=str, default='euclidean',
                        help='Distance metric for KNN')
    parser.add_argument('--clustering_method', type=str, default='KMeans',
                        choices=['KMeans', 'LouvainK'],
                        help='Clustering method')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--em_epochs', type=int, default=200,
                        help='Number of EM epochs')
    parser.add_argument('--em_iterations', type=int, default=10,
                        help='Number of EM iterations')
    parser.add_argument('--batch_size', type=int, default=12800,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    z_all = None

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        if isinstance(model, VAE):  # VAE
            recon_batch, mu, logvar, z = model(data)
            # VAE loss
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
        else:  # AE
            recon_batch, z = model(data)
            loss = F.mse_loss(recon_batch, data, reduction='sum')

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if z_all is None:
            z_all = z.detach()
        else:
            z_all = torch.cat([z_all, z.detach()], dim=0)

    return train_loss / len(train_loader.dataset), z_all


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
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Get number of clusters from data if not specified
    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # Create dataset and dataloader
    dataset = scDataset(X)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print('Initializing scGNN model...')
    if args.model_type == 'VAE':
        model = VAE(dim=X.shape[1]).to(device)
    else:
        model = AE(dim=X.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initial training
    print('Initial training...')
    for epoch in range(1, args.epochs + 1):
        loss, z = train_epoch(model, train_loader, optimizer, device)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}')

    zOut = z.cpu().numpy()

    # Build graph and perform clustering
    print('Building graph...')
    try:
        adj, edgeList = generateAdj(
            zOut,
            graphType='KNNgraphStatsSingleThread',
            para=f'{args.knn_distance}:{args.k}',
            adjTag=False
        )
    except:
        # Fallback to simple KNN
        from sklearn.neighbors import kneighbors_graph
        adj = kneighbors_graph(zOut, args.k, mode='connectivity', include_self=False)
        edgeList = []

    # EM iterations
    print('Starting EM iterations...')
    listResult = None

    for em_iter in range(args.em_iterations):
        print(f'EM iteration {em_iter + 1}/{args.em_iterations}')

        # Clustering
        if args.clustering_method == 'LouvainK' and len(edgeList) > 0:
            try:
                listResult, _ = generateLouvainCluster(edgeList)
                k = len(np.unique(listResult))
                k = max(int(k * 0.5), 2)
                kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
                listResult = kmeans.fit_predict(zOut)
            except:
                kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
                listResult = kmeans.fit_predict(zOut)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            listResult = kmeans.fit_predict(zOut)

        # EM training
        for epoch in range(1, args.em_epochs + 1):
            loss, z = train_epoch(model, train_loader, optimizer, device)

        zOut = z.cpu().numpy()

        # Update graph
        try:
            adj, edgeList = generateAdj(
                zOut,
                graphType='KNNgraphStatsSingleThread',
                para=f'{args.knn_distance}:{args.k}',
                adjTag=False
            )
        except:
            pass

    # Final clustering
    print('Final clustering...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20)
    y_pred = kmeans.fit_predict(zOut)

    # Save results using standard interface
    save(args.save_dir, Y, y_pred, args.epochs, zOut)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
