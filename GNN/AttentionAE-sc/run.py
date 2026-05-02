# -*- coding: utf-8 -*-
"""
Unified AttentionAE-sc Model Interface for scBench
==================================================

Usage:
    python run.py --data_path /path/to/data.h5ad --n_clusters 10 --save_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
import scanpy as sc
from sklearn import preprocessing
from sklearn.cluster import KMeans

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess import prepare_data_for_model
from utils import save

# Import model components
from model import AttentionAE
from loss import ZINBLoss


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    import scipy.sparse as sp
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(adata, n_neighbors=15, method='gauss'):
    """Build cell connectivity graph."""
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    adj = adata.obsp['connectivities']
    # Normalize adjacency matrix
    adj = adj + adj.T
    adj = adj / adj.max()
    # Add self-loops
    import scipy.sparse as sp
    adj = adj + sp.eye(adj.shape[0])
    r_adj = adj.copy()
    return adj, r_adj


def use_leiden(z, resolution=1.0):
    """Use Leiden clustering to get cluster centers."""
    adata = sc.AnnData(z)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)
    labels = np.array(adata.obs['leiden'].astype(int))
    n_clusters = len(np.unique(labels))
    centers = np.zeros((n_clusters, z.shape[1]))
    for i in range(n_clusters):
        centers[i] = z[labels == i].mean(axis=0)
    return centers, labels


def dist_2_label(p):
    """Convert soft assignment to hard labels."""
    return p.argmax(dim=1).cpu().numpy()


def loss_func(z, cluster_layer, alpha=1):
    """Compute KL divergence loss for clustering."""
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_layer) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p, reduction='batchmean')
    return loss, p


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AttentionAE-sc: Attention-based Autoencoder for scRNA-seq Clustering',
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
    parser.add_argument('--n_z', type=int, default=16,
                        help='Dimension of latent space')
    parser.add_argument('--n_enc_1', type=int, default=256,
                        help='Encoder layer 1 dimension')
    parser.add_argument('--n_enc_2', type=int, default=64,
                        help='Encoder layer 2 dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Number of neighbors for graph construction')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Resolution for Leiden clustering')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                        help='Number of pretraining epochs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of clustering epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='Tolerance for stopping criterion')

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
    sf = np.array(sf).astype(np.float32)

    # Encode labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    if Y.dtype.kind not in ['i', 'u']:
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    # Get raw counts for ZINB loss (use layers['norm_log'] which has normalized+log1p HVG data)
    if 'norm_log' in adata.layers:
        raw_counts = adata.layers['norm_log']
    else:
        raw_counts = adata.X
    if hasattr(raw_counts, 'toarray'):
        raw_counts = raw_counts.toarray()
    raw_counts = np.array(raw_counts).astype(np.float32)

    # Get number of clusters from data if not specified
    n_clusters = args.n_clusters if args.n_clusters > 0 else len(np.unique(Y))
    print(f'Number of cells: {X.shape[0]}, Number of genes: {X.shape[1]}')
    print(f'Number of clusters: {n_clusters}')

    # Z-score normalization
    Zscore_data = preprocessing.scale(X)

    # Build graph
    print('Building cell connectivity graph...')
    adata_graph = sc.AnnData(Zscore_data)
    adj, r_adj = build_graph(adata_graph, n_neighbors=args.n_neighbors)

    # Convert to tensors
    data = torch.Tensor(Zscore_data).to(device)
    sf_tensor = torch.autograd.Variable(
        torch.from_numpy(sf[:, None]).type(torch.FloatTensor).to(device),
        requires_grad=True
    )

    import scipy.sparse
    if isinstance(adj, scipy.sparse.spmatrix):
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        r_adj_tensor = torch.Tensor(r_adj.toarray()).to(device)
    else:
        adj_tensor = torch.Tensor(adj).to(device)
        r_adj_tensor = torch.Tensor(r_adj).to(device)

    # Initialize model
    print('Initializing AttentionAE-sc model...')
    model = AttentionAE(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_dec_1=args.n_enc_2,
        n_dec_2=args.n_enc_1,
        n_input=X.shape[1],
        n_z=args.n_z,
        heads=args.n_heads,
        device=device
    ).to(device)

    # ==================== PRE-TRAINING ====================
    print('Pre-training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    for epoch in range(args.pretrain_epochs):
        model.train()
        z, A_pred, pi, mean, disp = model(data, adj_tensor)

        zinb_loss_fn = ZINBLoss(theta_shape=(X.shape[1],))
        zinb_loss = zinb_loss_fn(mean * sf_tensor, pi, target=torch.tensor(raw_counts).to(device), theta=disp)
        re_graphloss = F.mse_loss(A_pred.view(-1), r_adj_tensor.view(-1))
        loss = zinb_loss + 0.1 * re_graphloss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f'Pretrain Epoch {epoch + 1}/{args.pretrain_epochs}, Loss: {loss.item():.4f}')

    # ==================== CLUSTERING ====================
    print('Clustering...')

    # Get initial cluster centers using Leiden
    with torch.no_grad():
        z, _, _, _, _ = model(data, adj_tensor)
    cluster_centers, init_label = use_leiden(z.detach().cpu().numpy(), resolution=args.resolution)
    cluster_layer = torch.autograd.Variable(
        torch.from_numpy(cluster_centers).type(torch.FloatTensor).to(device),
        requires_grad=True
    )

    # Clustering optimizer
    optimizer = torch.optim.Adam(
        list(model.enc_1.parameters()) + list(model.enc_2.parameters()) +
        list(model.attn1.parameters()) + list(model.attn2.parameters()) +
        list(model.gnn_1.parameters()) + list(model.gnn_2.parameters()) +
        list(model.z_layer.parameters()) + [cluster_layer],
        lr=0.001
    )

    best_embedding = None
    best_y_pred = None
    last_label = None

    for epoch in range(args.epochs):
        model.train()
        z, A_pred, pi, mean, disp = model(data, adj_tensor)

        kl_loss, ae_p = loss_func(z, cluster_layer)
        zinb_loss_fn = ZINBLoss(theta_shape=(X.shape[1],))
        zinb_loss = zinb_loss_fn(mean * sf_tensor, pi, target=torch.tensor(raw_counts).to(device), theta=disp)
        re_graphloss = F.mse_loss(A_pred.view(-1), r_adj_tensor.view(-1))
        loss = kl_loss + 0.1 * zinb_loss + 0.01 * re_graphloss

        label = dist_2_label(ae_p)

        # Check convergence
        if epoch == 0:
            last_label = label
        else:
            delta_label = np.sum(label != last_label).astype(np.float32) / len(label)
            last_label = label
            if epoch > 20 and delta_label < args.tol:
                print(f'Converged at epoch {epoch + 1}')
                break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        best_embedding = z.detach().cpu().numpy()
        best_y_pred = label

        if (epoch + 1) % 20 == 0:
            print(f'Clustering Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}')

    # Save results using standard interface
    save(args.save_dir, Y, best_y_pred, args.epochs, best_embedding)

    print(f'Training completed.')
    print(f'Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
