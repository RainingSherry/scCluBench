# -*- encoding: utf-8 -*-
"""
====================================================================================================
scCDCG 工具函数和辅助模块
====================================================================================================

本模块提供 scCDCG 的辅助函数，包括：

【包含内容】
    1. 图处理函数 — 归一化、稀疏矩阵转换
    2. 评估函数 — 聚类指标计算
    3. vMF分布 — von Mises-Fisher分布相关函数
    4. 最优传输 — Sinkhorn和Greenkhorn算法
    5. 可视化函数 — PCA/t-SNE可视化

【vMF分布背景】
    von Mises-Fisher (vMF) 分布是球面上方向数据的方向统计分布
    在scCDCG中用于：
    - 估计方向集中程度（kappa参数）
    - 计算聚类中心的相似度
"""

import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import recall_score, precision_score


# =============================================================================
# 图处理函数
# =============================================================================

def normalize(mx):
    """
    行归一化稀疏矩阵

    【公式】
        D^(-1) · A
        其中 D_ii = Σ_j A_ij（度矩阵）

    【用途】
        将邻接矩阵归一化后，节点特征传播更稳定
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # D^(-1)
    r_inv[np.isinf(r_inv)] = 0.             # 处理度为0的节点
    r_mat_inv = sp.diags(r_inv)             # 对角矩阵
    mx = r_mat_inv.dot(mx)                 # D^(-1) · A
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为PyTorch稀疏张量

    【用途】
        PyTorch的稀疏操作比scipy更高效
        用于图神经网络的邻接矩阵表示
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# =============================================================================
# 评估函数
# =============================================================================

def best_map(y_true, y_pred):
    """
    使用Hungarian算法找到最优标签映射

    【与evaluation.py中best_map的关系】
        本函数是旧版本，evaluation.py中的是新版本
        功能相同，只是实现略有差异
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    # 构建混淆矩阵
    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    # Hungarian算法
    A = linear_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]


def evaluation(y_true, y_pred):
    """聚类评估函数（与主模块一致）"""
    y_pred_, label_original, label_truth = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari, f1_macro


# =============================================================================
# vMF (von Mises-Fisher) 分布函数
# =============================================================================

def pdf_norm(dim, kappas):
    """
    vMF分布的归一化常数

    【数学公式】
        C_d(κ) = κ^(d/2-1) / ((2π)^(d/2) · I_(d/2-1)(κ))

        其中 I_ν(x) 是第一类修正贝塞尔函数

    【参数】
        dim: 嵌入空间维度 d
        kappas: 集中度参数 κ

    【用途】
        vMF概率密度需要除以此常数才能归一化
    """
    numerator = torch.pow(kappas, dim / 2 - 1)
    denominator = torch.pow(
        torch.mul(
            torch.pow(torch.ones_like(kappas) * 2 * math.pi, dim / 2),
            iv(dim / 2 - 1, kappas)
        ),
        -1
    )
    return torch.mul(numerator, denominator)


def A_d(dim, kappas):
    """
    vMF分布的A函数（用于估计κ）

    【数学公式】
        A_d(κ) = I_(d/2)(κ) / I_(d/2-1)(κ)

    【用途】
        用于矩估计集中度参数κ
    """
    numerator = iv(dim / 2, kappas)
    denominator = torch.pow(iv(dim / 2 - 1, kappas), -1)
    return torch.mul(numerator, denominator)


def estimate_kappa(dim, kappas):
    """
    估计vMF分布的集中度参数κ

    【数学公式】
        κ的近似估计：κ ≈ d·r - r³

        其中 r = ||Σ x_i|| / n（样本向量的平均长度）

    【参数】
        dim: 嵌入空间维度
        kappas: 当前κ值
    """
    r = A_d(dim, kappas)
    numerator = dim * r - torch.pow(r, 3)
    denominator = torch.pow(1 - torch.pow(r, 2), -1)
    return torch.mul(numerator, denominator)


# =============================================================================
# 可视化函数
# =============================================================================

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.family'] = 'Times New Roman'


def visual(num_class, h, y, c, pred_q, save_path_truth, save_path_pred_q):
    """
    可视化聚类结果

    【参数】
        num_class: 聚类数
        h: 细胞嵌入向量 (n_cells, embedding_dim)
        y: 真实标签
        c: 聚类中心
        pred_q: 预测标签
        save_path_truth: 真实标签可视化保存路径
        save_path_pred_q: 预测标签可视化保存路径

    【可视化方法】
        使用PCA降维到2维后散点图
    """
    # 合并细胞和聚类中心用于统一PCA
    h = np.vstack((h, c))

    # PCA降维到2维
    pca = PCA(n_components=2)
    h_ = pca.fit_transform(h)

    # 分离细胞和聚类中心
    h = h_[:-c.shape[0]]
    c = h_[-c.shape[0]:]

    # 颜色映射
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink',
              'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']

    # 绘制真实标签
    fig, ax = plt.subplots()
    for index in range(num_class):
        mask = (y[:] == index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=colors[index % len(colors)],
                   label=f'cluster {index}', s=10, alpha=1, edgecolors='none')
    plt.axis('off')
    plt.savefig(save_path_truth, bbox_inches='tight')

    # 绘制预测标签（带聚类中心）
    fig, ax = plt.subplots()
    for index in range(num_class):
        mask = (pred_q[:] == index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=colors[index % len(colors)],
                   label=f'cluster {index}', s=10, alpha=1, edgecolors='none')
        # 标记聚类中心
        ax.scatter(c[index, 0], c[index, 1], c=colors[index % len(colors)],
                   label=f'center {index}', s=100, alpha=1, edgecolors='black')
    plt.axis('off')
    plt.savefig(save_path_pred_q, bbox_inches='tight')


# =============================================================================
# 最优传输算法
# =============================================================================

def obj_func(target, pred):
    """最优传输的目标函数：负交叉熵"""
    target = target.reshape(pred.shape[0], pred.shape[1])
    loss = -np.mean(target * np.log(pred))
    return loss


def grad_func(target, pred):
    """最优传输的梯度"""
    gradient = -np.log(pred)
    return np.ravel(gradient)


def cons_row(i, shape0, shape1):
    """行约束：每行和为1"""
    return {'type': 'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=1)[i] - 1}


def cons_col(j, shape0, shape1):
    """列约束：每列和为n/k"""
    return {'type': 'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=0)[j] - shape0 / shape1}


def cons_positive(k):
    """非负约束"""
    return {'type': 'ineq', 'fun': lambda x: x[k]}


def cons_orthogonal(j1, j2, shape0, shape1):
    """正交约束"""
    return {'type': 'eq', 'fun': lambda x: np.dot(x.reshape(shape0, shape1).T, x.reshape(shape0, shape1))[j1][j2]}


def re_assignment(pred):
    """
    使用优化求解最优指派问题

    【方法】
        使用scipy.optimize.minimize求解约束优化问题
        目标：最小化负交叉熵
        约束：行和=1，列和=n/k，非负，正交

    【用途】
        替代Sinkhorn的精确求解版本（计算量大但精确）
    """
    num_node = pred.shape[0]
    num_class = pred.shape[1]

    # 构建约束
    cons_1 = list(map(cons_row, list(range(num_node)),
                       [num_node for i in range(num_node)],
                       [num_class for i in range(num_node)]))
    cons_2 = list(map(cons_col, list(range(num_class)),
                       [num_node for i in range(num_class)],
                       [num_class for i in range(num_class)]))
    cons_3 = list(map(cons_positive, list(range(num_node * num_class))))

    cons = cons_1 + cons_2 + cons_3

    # 均匀初始化
    init_target = np.ravel(np.ones_like(pred) / num_class)

    # 求解
    from scipy.optimize import minimize
    res = minimize(fun=obj_func, x0=init_target, args=pred,
                   jac=grad_func, constraints=cons)
    return res.success, res.x.reshape(num_node, num_class)


# =============================================================================
# Greenkhorn算法（Sinkhorn的贪心版本）
# =============================================================================

def dist_pho(a, b):
    """计算KL距离"""
    return b - a + a * np.log(a / b)


def greenkhorn(pred):
    """
    Greenkhorn算法 — Sinkhorn的贪心版本

    【与Sinkhorn的区别】
        - Sinkhorn: 交替更新所有行和所有列
        - Greenkhorn: 每次只更新变化最大的行或列
        - 更适合大规模问题

    【用途】
        近似求解最优传输问题
    """
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, 1).T

    row = np.ones(num_node)
    col = np.ones(num_class) * (num_node / num_class)

    x = np.ones_like(row)
    y = np.ones_like(col)

    for index in range(1000):
        max_i = np.argmax(dist_pho(row, np.sum(p, axis=1)))
        max_j = np.argmax(dist_pho(col, np.sum(p, axis=0)))

        if dist_pho(row[max_i], torch.sum(q, dim=1)[max_i]) > dist_pho(col[max_j], torch.sum(q, dim=0)[max_j]):
            x[max_i] = x[max_i] + row[max_i] / torch.sum(q, dim=1)[max_i]
        else:
            y[max_j] = y[max_j] + col[max_j] / torch.sum(q, dim=0)[max_j]

        q = torch.mm(
            torch.mul(p, torch.exp(x).unsqueeze(1)),
            torch.diag(torch.exp(y))
        )

    return q


# =============================================================================
# DEC目标分布
# =============================================================================

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    计算DEC的目标分布P

    【论文出处】
        Xie, Girshick, Farhadi. "Unsupervised Deep Clustering..." (ICML 2016)

    【数学公式】
        P_{ij} = q_{ij}² / f_j

        其中 f_j = Σ_i q_{ij}（列和）

    【用途】
        相比Q分布，P分布更加尖锐（高置信度信号被放大）
        用于KL散度损失，引导模型学习更确定的聚类
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


# =============================================================================
# 拉普拉斯矩阵
# =============================================================================

def get_laplace_matrix(tensor_matrix):
    """
    计算对称归一化拉普拉斯矩阵

    【数学公式】
        L = D^(-1/2) · A · D^(-1/2)

        其中 D_ii = Σ_j A_ij（度矩阵）

    【用途】
        用于图正则化损失
        L_cov = -Tr(z^T · L · z) / n
        最小化此损失使相邻节点在嵌入空间靠近

    【对称归一化的优势】
        - 特征值范围 [0, 2]，数值稳定
        - 对度差异更鲁棒
    """
    A = np.array(tensor_matrix)
    D = A.sum(axis=1)                           # 度向量
    L_matrix = np.diag(D ** (-0.5)).dot(        # D^(-1/2)
        A.dot(np.diag(D ** (-0.5)))              # D^(-1/2) · A · D^(-1/2)
    )
    L_matrix = torch.tensor(L_matrix, dtype=torch.float)
    # 处理NaN（度为0的节点）
    return torch.nan_to_num(L_matrix)


# =============================================================================
# 以下为旧版工具函数（保留兼容性）
# =============================================================================

import json
import functools
import operator
import collections
import jgraph
import tqdm


class dotdict(dict):
    """支持点号访问的字典"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():
    """检测Jupyter环境"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def smart_tqdm():
    """根据环境选择进度条"""
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    """TensorFlow图上下文装饰器"""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


def minibatch(batch_size, desc, use_last=False, progress_bar=True):
    """批处理装饰器"""
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(total_size / float(batch_size)).astype(np.int)
            else:
                n_batch = max(1, np.floor(total_size / float(batch_size)).astype(np.int))
            for batch_idx in smart_tqdm()(
                range(n_batch), desc=desc, unit="batches",
                leave=False, disable=not progress_bar
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


def encode_integer(label, sort=False):
    """整数编码"""
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


def encode_onehot(label, sort=False, ignore=None):
    """One-Hot编码"""
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


class CellTypeDAG(object):
    """细胞类型层级DAG（与utils.py一致）"""

    def __init__(self, graph=None, vdict=None):
        self.graph = jgraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file):
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file):
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file):
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = jgraph.Graph(directed=True), {}
        for item in ont:
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue
            graph.add_vertex(
                name=item.id, cell_ontology_class=item.name,
                desc=str(item.desc), synonyms=[
                    ("%s (%s)" % (syn.desc, syn.scope)) for syn in item.synonyms]
            )
            vdict[item.id] = item.id
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id
        for source in graph.vs:
            for relation in ont[source["name"]].relations:
                if relation.obo_name != "is_a":
                    continue
                for target in ont[source["name"]].relations[relation]:
                    if not target.id.startswith("CL"):
                        continue
                    graph.add_edge(
                        source["name"],
                        graph.vs.find(name=target.id.split()[0])["name"]
                    )
        return cls(graph, vdict)

    def _build_tree(self, d, parent=None):
        self.graph.add_vertex(name=d["name"])
        v = self.graph.vs.find(d["name"])
        if parent is not None:
            self.graph.add_edge(v, parent)
        self.vdict[d["name"]] = d["name"]
        if "alias" in d:
            for alias in d["alias"]:
                self.vdict[alias] = d["name"]
        if "children" in d:
            for subd in d["children"]:
                self._build_tree(subd, v)

    def get_vertex(self, name):
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1, name2):
        return self.is_descendant_of(name1, name2) or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0
        self.graph.vs["prob"] = 0
        v2_parents = list(self.graph.bfsiter(
            self.get_vertex(name2), mode=jgraph.OUT))
        v1_parents = list(self.graph.bfsiter(
            self.get_vertex(name1), mode=jgraph.OUT))
        for v in v2_parents:
            v["prob"] = 1
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:
                if v1_parent["prob"] != 0:
                    continue
                v1_parent["prob"] = np.prod([
                    v["prob"] / v.degree(mode=jgraph.IN)
                    for v in v1_parent.neighbors(mode=jgraph.OUT)
                ])
                if v1_parent["prob"] != 0:
                    changed = True
            if not changed:
                break
        return self.get_vertex(name1)["prob"]

    def similarity(self, name1, name2, method="probability"):
        if method == "probability":
            return (
                self.conditional_prob(name1, name2) +
                self.conditional_prob(name2, name1)
            ) / 2
        raise ValueError("Invalid method!")

    def count_reset(self):
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0
        self.graph.vs["count"] = 0

    def count_set(self, name, count):
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        origins = [v for v in self.graph.vs.select(raw_count_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=jgraph.OUT):
                if v != origin:
                    v["prop_count"] += origin["raw_count"]
        self.graph.vs["count"] = list(map(
            operator.add, self.graph.vs["raw_count"],
            self.graph.vs["prop_count"]
        ))

    def best_leaves(self, thresh, retrieve="name"):
        subgraph = self.graph.subgraph(self.graph.vs.select(count_ge=thresh))
        leaves, max_count = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if leaf["count"] > max_count:
                max_count = leaf["count"]
                leaves = [leaf[retrieve]]
            elif leaf["count"] == max_count:
                leaves.append(leaf[retrieve])
        return leaves


class DataDict(collections.OrderedDict):
    """支持切片和shuffle的有序字典"""

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr):
    """稀疏矩阵转密集数组"""
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    """安全向量化函数"""
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)
