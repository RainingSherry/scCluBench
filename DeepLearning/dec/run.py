# -*- coding: utf-8 -*-
"""
Unified DEC Model Interface for scBench
========================================

DEC: Deep Embedded Clustering (PyTorch Implementation)

Original paper: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.cluster import KMeans

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save


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


class Autoencoder(nn.Module):
    """Stacked Autoencoder for DEC."""
    def __init__(self, dims, activation='relu', dropout=0.0):
        super(Autoencoder, self).__init__()
        self.dims = dims
        self.n_layers = len(dims) - 1

        # Encoder layers
        encoder_layers = []
        for i in range(self.n_layers):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.ReLU())
                if dropout > 0:
                    encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(self.n_layers - 1, -1, -1):
            decoder_layers.append(nn.Linear(dims[i + 1], dims[i]))
            if i > 0:
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ClusteringLayer(nn.Module):
    """Clustering layer for DEC."""
    def __init__(self, n_clusters, n_features, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, n_features))

    def forward(self, x):
        # Student's t-distribution
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


class DEC(nn.Module):
    """Deep Embedded Clustering model."""
    def __init__(self, dims, n_clusters, alpha=1.0, dropout=0.0):
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(dims, dropout=dropout)
        self.clustering_layer = ClusteringLayer(n_clusters, dims[-1], alpha)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def forward(self, x):
        z = self.autoencoder.encode(x)
        q = self.clustering_layer(z)
        return q, z

    def pretrain(self, x, epochs=200, batch_size=256, lr=1e-3, device='cpu'):
        """Pretrain autoencoder."""
        dataset = TensorDataset(torch.FloatTensor(x))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.autoencoder.parameters(), lr=lr)

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch_x = batch[0].to(device)
                optimizer.zero_grad()
                x_recon, _ = self.autoencoder(batch_x)
                loss = F.mse_loss(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

            if (epoch + 1) % 50 == 0:
                print(f'Pretrain Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(x):.6f}')


def target_distribution(q):
    """Compute target distribution P from soft assignment Q."""
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DEC: Deep Embedded Clustering',
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
    parser.add_argument('--n_z', type=int, default=10,
                        help='Dimension of latent space')
    parser.add_argument('--dims', type=str, default='500,500,2000',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Degrees of freedom for Student t-distribution')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                        help='Number of pretraining epochs')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of clustering epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='Tolerance for stopping criterion')
    parser.add_argument('--update_interval', type=int, default=1,
                        help='Update target distribution every N epochs')

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
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # Build network dimensions
    hidden_dims = [int(d) for d in args.dims.split(',')]
    dims = [X.shape[1]] + hidden_dims + [args.n_z]

    # Initialize model
    print('Initializing DEC model...')
    model = DEC(dims=dims, n_clusters=n_clusters, alpha=args.alpha).to(device)

    # Pretrain autoencoder
    print('Pretraining autoencoder...')
    model.pretrain(X, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                   lr=args.lr, device=device)

    # Initialize cluster centers with KMeans
    print('Initializing cluster centers...')
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        z = model.encode(X_tensor).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
    y_pred = kmeans.fit_predict(z)
    model.clustering_layer.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    # Training
    print('Training clustering model...')
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_embedding = None
    best_y_pred = None
    y_pred_last = y_pred.copy()

    for epoch in range(args.epochs):
        model.train()

        # Update target distribution
        if epoch % args.update_interval == 0:
            with torch.no_grad():
                q, z = model(X_tensor)
                q = q.cpu().numpy()
            p = target_distribution(q)
            y_pred = q.argmax(1)

            # Check convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(y_pred)
            y_pred_last = y_pred.copy()

            if epoch > 0 and delta_label < args.tol:
                print(f'Converged at epoch {epoch}')
                break

        # Training step
        p_tensor = torch.FloatTensor(p).to(device)
        q, z = model(X_tensor)
        loss = F.kl_div(q.log(), p_tensor, reduction='batchmean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_embedding = z.detach().cpu().numpy()
        best_y_pred = y_pred

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.6f}')

    # Save results using standard interface
    save(args.save_dir, Y, best_y_pred, args.epochs, best_embedding)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
