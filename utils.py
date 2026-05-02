"""
====================================================================================================
scCluBench 工具函数模块
====================================================================================================

本模块提供通用工具函数，供各个模型和脚本使用。

【主要功能】
    1. save() — 保存模型结果（嵌入、指标、预测标签）
    2. dotdict — 支持点号访问的字典（简化代码）
    3. DataDict — 增强版有序字典（支持切片和 shuffle）
    4. CellTypeDAG — 细胞类型有向无环图（用于语义相似性分析）
    5. minibatch — 将函数包装为小批次版本
    6. 编码/解码工具 — 处理字符串和字节的转换

【设计理念】
    - 所有模型使用统一的 save() 接口保存结果
    - 结果格式一致，便于后续分析和比较
    - 支持 PyTorch tensor 和 numpy 数组
"""

from evaluation import evaluation
import os
import json
import h5py
import numpy as np
import pandas as pd
import torch


def save(embedding_path, y, y_pred, epoch, embedding):
    """
    保存聚类模型的训练结果（统一接口）

    【这是 benchmark 的标准保存接口】
    所有模型（scCDCG、scMAE、scDCC 等）都应调用此函数保存结果。

    【保存内容】
        1. metrics_{epoch}.json — 8 个评估指标
        2. embedding_{epoch}.npy — 降维后的嵌入向量（NumPy 数组）
        3. embedding.h5 — HDF5 格式的嵌入向量和预测标签
        4. types_{epoch}_pred.csv — 真实标签和预测标签的对照表

    【参数详解】
        embedding_path: 保存目录路径
        y: 真实细胞类型标签（一维数组）
        y_pred: 预测的聚类标签（一维数组）
        epoch: 当前训练轮次（用于文件名）
        embedding: 降维后的嵌入向量
            - PyTorch tensor: 自动转为 NumPy
            - NumPy array: 直接保存

    【嵌入向量格式】
        embedding.shape = (n_cells, embedding_dim)
        例如：(2544, 16) 表示 2544 个细胞，每个细胞 16 维嵌入向量

    【使用示例】
        embedding = model.encode(X)  # shape: (2544, 16)
        save('./results/Leiden/Arabidopsis', y_true, y_pred, epoch=200, embedding=embedding)
        # 生成文件：
        #   ./results/Leiden/Arabidopsis/metrics_200.json
        #   ./results/Leiden/Arabidopsis/embedding_200.npy
        #   ./results/Leiden/Arabidopsis/embedding.h5
        #   ./results/Leiden/Arabidopsis/types_200_pred.csv
    """
    # Step 1: 调用 evaluation 计算所有 8 个指标
    acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, y_pred_ = evaluation(y, y_pred)

    # Step 2: 组织指标字典
    metrics = {
        'acc': acc,            # 准确率
        'nmi': nmi,            # 标准化互信息
        'ari': ari,            # 调整兰德指数
        'f1_macro': f1_macro, # 宏平均 F1
        'fmi': fmi,            # 福克斯-马洛斯指数
        'v_measure': v_measure,  # V 度量
        'homogeneity': hom,    # 同质性
        'completeness': com,   # 完整性
    }

    # Step 3: 打印结果（方便查看训练过程）
    print(f"Epoch {epoch}: "
          f"acc: {acc:.4f}, "
          f"nmi: {nmi:.4f}, "
          f"ari: {ari:.4f}, "
          f"f1_macro: {f1_macro:.4f}, "
          f"fmi: {fmi:.4f}, "
          f"v_measure: {v_measure:.4f}, "
          f"homogeneity: {hom:.4f}, "
          f"completeness: {com:.4f}")

    # Step 4: 保存 JSON 格式指标
    metrics_file = os.path.join(embedding_path, f'metrics_{epoch}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Also save as metrics.json (latest/best) for easy access
    with open(os.path.join(embedding_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Step 5: 保存嵌入向量（HDF5 和 NumPy 两种格式）
    with h5py.File(os.path.join(embedding_path, 'embedding.h5'), 'w') as f:
        if isinstance(embedding, torch.Tensor):
            # PyTorch tensor → 转为 NumPy 后保存
            emb_np = embedding.cpu().detach().numpy()
            f.create_dataset('X', data=emb_np)
            np.save(os.path.join(embedding_path, f'embedding_{epoch}.npy'), emb_np)
        else:
            # NumPy array → 直接保存
            f.create_dataset('X', data=embedding)
            np.save(os.path.join(embedding_path, f'embedding_{epoch}.npy'), embedding)
        # 保存预测标签（HDF5 格式）
        f.create_dataset('Y', data=y_pred_)

    # Step 6: 保存标签对照表（CSV 格式，方便查看）
    pd.DataFrame({
        "pred": y_pred_,  # 重排后的预测标签
        "true": y,        # 真实标签
    }).to_csv(os.path.join(embedding_path, f'types_{epoch}_pred.csv'), index=False)


# =============================================================================
# 以下为通用工具函数
# =============================================================================

import json
import functools
import operator
import collections
import jgraph
import numpy as np
import scipy.sparse
import tqdm


class dotdict(dict):
    """
    支持点号访问的字典（简化代码编写）

    【功能】
        允许使用 dict.attr 而非 dict['attr'] 的方式访问字典

    【示例】
        d = dotdict({'name': 'Alice', 'age': 25})
        d.name    # 返回 'Alice'
        d['name'] # 返回 'Alice'
        d.get('name')  # 返回 'Alice'
    """
    __getattr__ = dict.get      # dict.attr → dict.get('attr')
    __setattr__ = dict.__setitem__  # dict.attr = value → dict['attr'] = value
    __delattr__ = dict.__delitem__  # del dict.attr → del dict['attr']


def in_ipynb():
    """
    检测当前是否在 Jupyter Notebook 环境中运行

    【返回值】
        True: Jupyter notebook 或 qtconsole
        False: 终端或标准 Python 解释器
    """
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
    """
    根据运行环境返回合适的进度条
    - Jupyter 环境返回 tqdm.tqdm_notebook（显示为 notebook 进度条）
    - 其他环境返回 tqdm.tqdm（标准进度条）
    """
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    """
    装饰器：在指定的 TensorFlow 图上下文中执行函数
    （用于旧版 TensorFlow 代码，现已弃用）
    """
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


def minibatch(batch_size, desc, use_last=False, progress_bar=True):
    """
    将一个逐样本函数包装为小批次版本

    【功能】
        假设你有一个函数 func(batch_X)，它每次处理一个 batch 的数据。
        使用 minibatch 装饰器后，函数可以接受完整数据，自动分批处理。

    【参数】
        batch_size: 每批的样本数
        desc: 进度条描述文字
        use_last: 是否使用最后不完整的批次
        progress_bar: 是否显示进度条

    【使用示例】
        @minibatch(batch_size=256, desc='Training')
        def train_batch(batch_X, batch_Y):
            loss = model(batch_X, batch_Y)
            loss.backward()
            optimizer.step()

        # 调用时传入完整数据，自动分批处理
        train_batch(all_X, all_Y)
    """
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
                # 切片获取当前批次数据
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)

        return wrapped_func
    return minibatch_wrapper


# =============================================================================
# 编码/解码工具
# =============================================================================

def encode_integer(label, sort=False):
    """
    将标签编码为从 0 开始的整数

    【参数】
        label: 标签数组（可以是字符串或整数）
        sort: 是否排序后再编码

    【返回】
        encoded: 整数编码后的数组
        classes: 原始唯一类别列表
    """
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


def encode_onehot(label, sort=False, ignore=None):
    """
    将标签编码为 One-Hot 格式

    【参数】
        label: 标签数组
        sort: 是否排序后再编码
        ignore: 要忽略的类别列表

    【返回】
        稀疏矩阵格式的 One-Hot 编码
    """
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


# =============================================================================
# CellTypeDAG — 细胞类型有向无环图
# =============================================================================

class CellTypeDAG(object):
    """
    细胞类型的层级结构有向无环图（Directed Acyclic Graph）

    【用途】
        - 分析细胞类型之间的层级关系
        - 计算细胞类型的语义相似性
        - 支持基于本体论（ontology）的分析

    【数据来源】
        支持两种格式：
        - .json: 自定义 JSON 格式的层级结构
        - .obo: Gene Ontology (GO) 的 OBO 格式（支持 CL 本体）

    【核心概念】
        - 节点：细胞类型（如 'beta cell', 'alpha cell'）
        - 边：is_a 关系（如 'beta cell' is_a 'endocrine cell'）
        - 根节点：最通用的类型（如 'cell'）
        - 叶节点：最具体的类型

    【使用示例】
        # 加载细胞类型层级
        dag = CellTypeDAG.load('cell_ontology.json')

        # 检查层级关系
        dag.is_descendant_of('beta cell', 'endocrine cell')  # True
        dag.is_ancestor_of('cell', 'beta cell')              # True

        # 计算语义相似性
        dag.similarity('beta cell', 'alpha cell')  # 返回相似度分数
    """

    def __init__(self, graph=None, vdict=None):
        self.graph = jgraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file):
        """
        根据文件扩展名加载细胞类型 DAG

        【支持格式】
            .json: 自定义 JSON 格式
            .obo: Gene Ontology OBO 格式
        """
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file):
        """从 JSON 文件加载细胞类型层级"""
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file):
        """
        从 OBO 文件加载（使用 pronto 库解析）
        仅构建 CL（Cell Ontology）相关的 is_a 关系
        """
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = jgraph.Graph(directed=True), {}

        for item in ont:
            # 只处理 CL（Cell Ontology）类型的条目
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue

            # 添加节点
            graph.add_vertex(
                name=item.id,
                cell_ontology_class=item.name,
                desc=str(item.desc),
                synonyms=[("%s (%s)" % (syn.desc, syn.scope)) for syn in item.synonyms]
            )
            # 建立 id 和 name 的映射
            vdict[item.id] = item.id
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id

        # 添加 is_a 边
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
        """递归构建树（用于 JSON 加载）"""
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
        """获取节点对象"""
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1, name2):
        """检查两个细胞类型是否有层级关系（祖先或后代）"""
        return self.is_descendant_of(name1, name2) or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1, name2):
        """
        检查 name1 是否为 name2 的后代（更具体的类型）

        【示例】
            dag.is_descendant_of('beta cell', 'endocrine cell')  # True
            # 即：beta cell 是 endocrine cell 的子类
        """
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1, name2):
        """
        检查 name1 是否为 name2 的祖先（更通用的类型）

        【示例】
            dag.is_ancestor_of('cell', 'beta cell')  # True
            # 即：cell 是 beta cell 的祖先类型
        """
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1, name2):
        """
        计算条件概率 P(name1 | name2)
        即在 name2 的所有后代中，有多少比例属于 name1
        """
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
        """
        计算两个细胞类型的语义相似性

        【方法】
            probability: (P(name1|name2) + P(name2|name1)) / 2
        """
        if method == "probability":
            return (
                self.conditional_prob(name1, name2) +
                self.conditional_prob(name2, name1)
            ) / 2
        raise ValueError("Invalid method!")

    def count_reset(self):
        """重置节点计数"""
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0  # 从子节点传播的计数
        self.graph.vs["count"] = 0

    def count_set(self, name, count):
        """设置节点的原始计数"""
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        """
        更新计数：从叶节点向根节点传播
        每个节点的 count = raw_count + 所有子节点的 prop_count
        """
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
        """
        找到计数最高的叶节点（最具体的细胞类型）
        用于细胞注释时选择最具体的标签
        """
        subgraph = self.graph.subgraph(self.graph.vs.select(count_ge=thresh))
        leaves, max_count = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if leaf["count"] > max_count:
                max_count = leaf["count"]
                leaves = [leaf[retrieve]]
            elif leaf["count"] == max_count:
                leaves.append(leaf[retrieve])
        return leaves


# =============================================================================
# DataDict — 增强版有序字典
# =============================================================================

class DataDict(collections.OrderedDict):
    """
    支持切片和 shuffle 的有序字典

    【用途】
        用于存储多个数组（如 X, Y, mask），并支持同时切片或打乱

    【示例】
        data = DataDict([
            ('X', X_array),
            ('Y', Y_array),
            ('mask', mask_array)
        ])

        # shuffle：同时打乱所有数组
        shuffled = data.shuffle()

        # 切片：同时切片所有数组
        batch = data[0:100]  # 每个数组取前100个元素
    """

    def shuffle(self, random_state=np.random):
        """
        随机打乱所有数组（保持对应关系）

        【参数】
            random_state: 随机状态（默认使用 numpy 的全局随机）
        """
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        """所有数组的第一维大小必须相同，返回该大小"""
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):
        """兼容性属性，返回 [size]"""
        return [self.size]

    def __getitem__(self, fetch):
        """支持切片和数组索引，同时作用于所有数组"""
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


# =============================================================================
# 其他工具函数
# =============================================================================

def densify(arr):
    """
    如果是稀疏矩阵则转为密集数组

    【用途】
        某些操作（如索引）不能在稀疏矩阵上直接进行，需要先转为密集数组
    """
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    """
    创建安全的向量化函数（处理空数组）

    【功能】
        如果输入为空数组，直接返回空数组；
        否则应用函数 fn
    """
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


# 字符串/字节转换工具
decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
"""将字节数组解码为字符串（如 [b'A', b'B'] → ['A', 'B']）"""

encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
"""将字符串编码为字节（如 ['A', 'B'] → [b'A', b'B']）"""

upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
"""将字符串转为大写"""

lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
"""将字符串转为小写"""

tostr = empty_safe(np.vectorize(str), str)
"""将任意类型转为字符串"""
