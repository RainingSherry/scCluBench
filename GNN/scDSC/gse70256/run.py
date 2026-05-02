# -*- coding: utf-8 -*-
"""
Unified scDSC (gse70256 version) Model Interface for scBench
=============================================================

This is a deeper version of scDSC with 3 latent layers and 7 GNN layers.

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
from torch.nn.parameter import Parameter
from torch.nn import Linear
import random
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from preprocess import prepare_data_for_model
from utils import save

# Import model components from parent scDSC folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GNN import GNNLayer
from layers import ZINBLoss, MeanAct, DispAct


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


def normalize_adj(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(X, k=10):
    """Build k-nearest neighbor graph from data."""
    # Compute KNN graph
    adj = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    # Make symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    # Normalize
    adj = normalize_adj(adj)
    # Convert to torch sparse tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


class AE(nn.Module):
    """Deep Autoencoder with 3 latent layers for scDSC gse70256."""
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3):
        super(AE, self).__init__()

        # Encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        # Latent layers
        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        # Decoder
        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z3, z2, z1, dec_h3


class SDCN_Deep(nn.Module):
    """Deep Structural Deep Clustering Network with 7 GNN layers."""
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3, n_clusters, pretrain_path=None,
                 device='cuda', v=1):
        super(SDCN_Deep, self).__init__()
        self.ae = AE(
            n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_enc_3=n_enc_3,
            n_dec_1=n_dec_1, n_dec_2=n_dec_2, n_dec_3=n_dec_3,
            n_input=n_input, n_z1=n_z1, n_z2=n_z2, n_z3=n_z3
        )

        if pretrain_path is not None and os.path.exists(pretrain_path):
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # 7 GNN layers
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z1)
        self.gnn_5 = GNNLayer(n_z1, n_z2)
        self.gnn_6 = GNNLayer(n_z2, n_z3)
        self.gnn_7 = GNNLayer(n_z3, n_clusters)

        # Cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # ZINB decoder
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())

        self.v = v
        self.zinb_loss = ZINBLoss()
        if device == 'cuda':
            self.zinb_loss = self.zinb_loss.cuda()

    def forward(self, x, adj):
        # AE forward
        x_bar, tra1, tra2, tra3, z3, z2, z1, dec_h3 = self.ae(x)
        sigma = 0.5

        # GNN forward (7 layers)
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z1, adj)
        h = self.gnn_6((1 - sigma) * h + sigma * z2, adj)
        h = self.gnn_7((1 - sigma) * h + sigma * z3, adj, active=False)

        predict = F.softmax(h, dim=1)

        # ZINB parameters
        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)

        # Soft assignment
        q = 1.0 / (1.0 + torch.sum(torch.pow(z3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z3, _mean, _disp, _pi


def target_distribution(q):
    """Compute target distribution."""
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='scDSC (gse70256): Deep Structural Deep Clustering for scRNA-seq',
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
    parser.add_argument('--n_z1', type=int, default=2000,
                        help='Dimension of latent layer 1')
    parser.add_argument('--n_z2', type=int, default=500,
                        help='Dimension of latent layer 2')
    parser.add_argument('--n_z3', type=int, default=10,
                        help='Dimension of latent layer 3 (final embedding)')
    parser.add_argument('--n_enc_1', type=int, default=1000,
                        help='Encoder layer 1 dimension')
    parser.add_argument('--n_enc_2', type=int, default=1000,
                        help='Encoder layer 2 dimension')
    parser.add_argument('--n_enc_3', type=int, default=4000,
                        help='Encoder layer 3 dimension')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Number of neighbors for graph construction')

    # Loss weights
    parser.add_argument('--w_bce', type=float, default=0.1,
                        help='Weight for binary cross entropy loss')
    parser.add_argument('--w_ce', type=float, default=0.01,
                        help='Weight for cross entropy loss')
    parser.add_argument('--w_re', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--w_zinb', type=float, default=0.1,
                        help='Weight for ZINB loss')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                        help='Number of pretraining epochs')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                        help='Pretraining learning rate')
    parser.add_argument('--n_init', type=int, default=20,
                        help='Number of KMeans initializations')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    return parser.parse_args()


def pretrain_ae(model, X, args, device):
    """Pretrain autoencoder."""
    from torch.optim import Adam
    optimizer = Adam(model.ae.parameters(), lr=args.pretrain_lr)
    X_tensor = torch.tensor(X).to(device)

    for epoch in range(args.pretrain_epochs):
        model.train()
        x_bar, _, _, _, z3, _, _, _ = model.ae(X_tensor)
        loss = F.mse_loss(x_bar, X_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Pretrain Epoch {epoch + 1}/{args.pretrain_epochs}, Loss: {loss.item():.6f}')

    return model


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
    sf = np.array(sf).astype(np.float32).reshape(-1, 1)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Get raw counts for ZINB loss
    if adata.raw is not None:
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

    # Build graph
    print('Building KNN graph...')
    adj = build_graph(X, k=args.k_neighbors)
    if args.cuda:
        adj = adj.cuda()

    # Initialize model
    print('Initializing scDSC (gse70256) model...')
    model = SDCN_Deep(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_enc_3=args.n_enc_3,
        n_dec_1=args.n_enc_3,
        n_dec_2=args.n_enc_2,
        n_dec_3=args.n_enc_1,
        n_input=X.shape[1],
        n_z1=args.n_z1,
        n_z2=args.n_z2,
        n_z3=args.n_z3,
        n_clusters=n_clusters,
        device='cuda' if args.cuda else 'cpu'
    ).to(device)

    # Pretrain autoencoder
    print('Pretraining autoencoder...')
    model = pretrain_ae(model, X, args, device)

    # Initialize cluster centers with KMeans
    X_tensor = torch.tensor(X).to(device)
    with torch.no_grad():
        _, _, _, _, z3, _, _, _ = model.ae(X_tensor)
    kmeans = KMeans(n_clusters=n_clusters, n_init=args.n_init, random_state=args.seed)
    y_pred = kmeans.fit_predict(z3.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # Training
    print('Training clustering model...')
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=args.lr)

    X_raw_tensor = torch.tensor(raw_counts).to(device)
    sf_tensor = torch.tensor(sf).to(device)

    best_embedding = None
    best_y_pred = None
    p = None

    for epoch in range(args.epochs):
        model.train()

        # Update target distribution
        if epoch % 1 == 0:
            _, tmp_q, pred, z, _, _, _ = model(X_tensor, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            y_pred = pred.data.cpu().numpy().argmax(1)

        x_bar, q, pred, z, meanbatch, dispbatch, pibatch = model(X_tensor, adj)

        # Compute losses
        bce_loss = F.binary_cross_entropy(q, p)
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, X_tensor)
        zinb_loss = model.zinb_loss(X_raw_tensor, meanbatch, dispbatch, pibatch, sf_tensor)

        loss = args.w_bce * bce_loss + args.w_ce * ce_loss + args.w_re * re_loss + args.w_zinb * zinb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save best results
        best_embedding = z.detach().cpu().numpy()
        best_y_pred = y_pred

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.6f}')

    # Save results using standard interface
    save(args.save_dir, Y, best_y_pred, args.epochs, best_embedding)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
