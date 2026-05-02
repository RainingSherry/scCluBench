"""
====================================================================================================
scCluBench 单细胞RNA测序数据预处理模块
====================================================================================================

本模块提供统一的 scRNA-seq 数据预处理流程，是整个 benchmark 的数据入口。
核心功能包括：读取 h5ad 格式数据、细胞/基因过滤、归一化、log1p 变换、高度可变基因筛选、Z-score 标准化。

【预处理流程】（共6个步骤，对应 scanpy 标准流程）
    Step 1: 保存原始数据到 adata.raw（备份）
    Step 2: Per-cell 归一化（size factor 校正）
    Step 3: Log1p 变换（log(1+x)，稳定方差）
    Step 4: 高度可变基因（HVG）筛选（默认 top 1000）
    Step 5: 计算 size factor（用于深度学习模型）
    Step 6: Z-score 标准化（均值=0，方差=1）

【为什么这样预处理？】
    - scRNA-seq 数据是 counts 矩阵，直接使用会受 library size 影响
    - Per-cell 归一化：将每个细胞的 counts 缩放到相同总量，消除测序深度差异
    - Log1p：压缩大值、稳定方差、使数据接近正态分布、减少极端值影响
    - HVG 筛选：去除 housekeeping genes 和技术噪声基因，保留生物学变异
    - Z-score：使不同基因在同一尺度上，便于神经网络训练

【数据格式】
    输入：.h5ad 格式（AnnData 对象）
    输出：DataFrame X（预处理后基因表达）、Series Y（细胞类型标签）、Series sf（size factors）
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle, os, numbers

import h5py
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import utils as utils


# TODO: Fix this
class AnnSequence:
    """
    用于 Keras/TensorFlow 风格训练的批数据生成器（已弃用，仅作向后兼容）
    将矩阵数据封装为批次形式，每次返回 (X, Y) 对，其中包含 count 数据和 size factors
    """
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            # 如果没有提供 size factor，默认全部设为 1.0
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        # 返回总批次数（向下取整，丢弃最后不完整批次）
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        # 获取一个批次的数据
        batch = self.matrix[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sf = self.size_factors[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 返回 {'count': 数据, 'size_factors': 校正因子}, 数据本身
        return {'count': batch, 'size_factors': batch_sf}, batch


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    """
    读取并验证 AnnData 数据集（主要用于 DCA/Autoencoder 类型方法）

    【参数】
        adata: AnnData 对象或字符串（h5ad 文件路径）
        transpose: 是否转置矩阵（基因×细胞 → 细胞×基因）
        test_split: 是否划分训练/测试集（用于有监督方法）
        copy: 是否复制数据

    【验证逻辑】
        - 检查 adata.X 是否为原始 counts 数据（整数类型）
        - 跳过大型数据集（>50M 元素）的类型检查以节省内存
        - 大型稀疏矩阵检查：验证非零元素是否为整数

    【返回】
        验证并标注后的 AnnData 对象
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    # 确保数据是原始 counts（未归一化），即整数类型
    assert 'n_count' not in adata.obs, norm_error

    # 仅对小数据集检查整数类型（大型矩阵检查耗时）
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            # 稀疏矩阵：检查转换为 float 后是否与原数据相同（即原数据为整数）
            assert (adata.X.astype(float) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(float) == adata.X), norm_error

    if transpose:
        adata = adata.transpose()

    if test_split:
        # 划分 10% 数据作为测试集（用于有监督方法的评估）
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    """
    归一化、log1p 变换和标准化 AnnData 数据（旧版接口，保留向后兼容）

    【参数详解】
        filter_min_counts: 是否过滤低表达细胞和低表达基因
            - sc.pp.filter_cells: 过滤基因表达量低于 1 的细胞
            - sc.pp.filter_genes: 过滤细胞表达量低于 1 的基因
        size_factors: 是否计算 size factor
            - sc.pp.normalize_per_cell: 每个细胞归一化到相同总量（默认 1e4）
            - size_factor = n_counts / median(n_counts)
        logtrans_input: 是否进行 log1p 变换
            - sc.pp.log1p: log(1 + x)，稳定方差
        normalize_input: 是否进行 Z-score 标准化
            - sc.pp.scale: 每个基因标准化到均值=0，方差=1

    【adata.raw 的作用】
        保存原始未处理的数据，便于后续回退或比较
    """
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        # Per-cell 归一化：使每个细胞的 total counts 相同
        sc.pp.normalize_per_cell(adata)
        # Size factor = 细胞总 counts / 中位数总 counts（反映相对测序深度）
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        # Log1p：log(1+x)，压缩大值、稳定方差
        sc.pp.log1p(adata)

    if normalize_input:
        # Z-score 标准化：每个基因减去均值除以标准差
        sc.pp.scale(adata)

    return adata


def read_genelist(filename):
    """读取基因列表文件（每行一个基因名）"""
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))
    return genelist


def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    """
    将矩阵写入 TSV 文本文件

    【参数】
        matrix: 要写入的矩阵
        filename: 输出文件路径
        rownames: 行名（基因名或细胞名）
        colnames: 列名
        transpose: 是否转置后写入
    """
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')

def read_pickle(inputfile):
    """读取 pickle 序列化文件"""
    return pickle.load(open(inputfile, "rb"))

# =============================================================================
# 以下为新版预处理函数（2024年重写版本）
# =============================================================================

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, v_measure_score, silhouette_score, accuracy_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def check_normalization(adata):
    """
    检查 AnnData 是否已经完成预处理（用于智能跳过已完成的步骤）

    【检查逻辑】
        - is_norm: 检查数据是否为浮点数（非整数=已归一化）
        - is_log1p: log1p 变换后最大值一般 <20
        - is_scaled: Z-score 后均值≈0，标准差≈1

    【返回】
        (is_norm, is_log1p, is_scaled): 三个布尔值的元组

    【稀疏矩阵处理】
        稀疏矩阵不能直接比较类型，需要先转为密集数组
    """
    X = adata.X

    # **稀疏矩阵转换为密集数组**
    if sp.issparse(X):
        X = X.toarray()

    # 如果全是整数，则未归一化；浮点数则已归一化
    is_norm = not np.all(X.astype(int) == X)
    # log1p 后最大值一般 < 20（原始 counts 可能达到数万）
    is_log1p = X.max() < 20
    # Z-score 标准化后：均值=0，标准差=1
    is_scaled = np.isclose(X.mean(), 0, atol=0.1) and np.isclose(X.std(), 1, atol=0.1)

    return is_norm, is_log1p, is_scaled


def normalize_sc(adata, size_factors=True, filter_min_counts=True, logtrans_input=True, normalize_input=True):
    """
    归一化、log1p 变换和标准化 scRNA-seq 数据（新版本）

    【完整流程】
        Step 1: 备份原始数据到 adata.raw
        Step 2: 检查是否已归一化，如未归一化则执行 Per-cell 归一化
        Step 3: 检查是否已 log1p，如未则执行 log1p 变换
        Step 4: 执行高度可变基因（HVG）筛选（**必须在 log1p 后**）
        Step 5: 计算 size factor（用于 ZINB 等需要校正的模型）
        Step 6: 检查是否已标准化，如未则执行 Z-score

    【HVG 筛选的重要性】
        - HVG 筛选必须在上一步（归一化+log1p）之后进行
        - 使用 scanpy 的 dispersion 方法筛选：
            - min_mean=0.0125: 过滤低表达基因
            - max_mean=3: 过滤高表达管家基因
            - min_disp=0.5: 保留表达差异显著的基因
            - n_top_genes=1000: 只保留变化最大的 1000 个基因
        - subset=True: 直接在 adata 中保留 HVG，其余基因被移除

    【注意】
        - 与旧版 normalize() 不同，本函数 HVG 筛选是默认开启的
        - 筛选后基因数从数万个降至 1000，大幅降低计算复杂度
    """
    # Step 1: 确保 adata.raw 存储原始未归一化数据
    if adata.raw is None:
        adata.raw = adata.copy()

    # Step 2: 检查预处理状态，智能决定需要执行哪些步骤
    is_norm, is_log1p, is_scaled = check_normalization(adata)
    print(f"是否归一化: {is_norm}, 是否 log1p 变换: {is_log1p}, 是否标准化: {is_scaled}")

    # Step 2.1: 如果未归一化，则进行 Per-cell 归一化
    if not is_norm:
        print("数据未归一化，进行 normalize_per_cell 处理")
        # 确保 X 是浮点数（归一化操作的必要条件）
        if hasattr(adata.X, 'toarray'):
            import scipy.sparse as sp
            # 稀疏矩阵转密集数组后转为 float32
            adata.X = sp.csr_matrix(adata.X.toarray().astype(np.float32))
        elif adata.X.dtype.kind in ['i', 'u']:
            # 整数类型直接转为 float32
            adata.X = adata.X.astype(np.float32)
        # Per-cell 归一化：每个细胞归一化到相同总量
        sc.pp.normalize_per_cell(adata)
        is_norm = True

    # Step 3: 如果未 log1p 变换，则进行变换
    if logtrans_input and not is_log1p:
        print("数据未 log1p 变换，进行 log1p 处理")
        # Clip BEFORE log1p to prevent overflow (especially for sparse data)
        # log(1+x) with large x can overflow even when x is a count
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            data = adata.X.data.copy()
            data = np.clip(data, 0, 1e8)  # Max 1e8 as count value
            adata.X.data[:] = data
        sc.pp.log1p(adata)
        is_log1p = True

    # Step 3.5: Handle infinity values in normalized data (newer pandas/scanpy versions)
    # After log1p, clip extreme values to prevent inf/nan in HVG
    if is_log1p:
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            data = adata.X.data.copy()
            data = np.where(np.isfinite(data), data, 0.0)
            data = np.clip(data, -10, 10)
            adata.X.data[:] = data
        else:
            X_arr = np.array(adata.X)
            X_arr = np.where(np.isfinite(X_arr), X_arr, 0.0)
            X_arr = np.clip(X_arr, -10, 10)
            adata.X = X_arr

    # Step 4: 高度可变基因筛选（**必须在 log1p 后执行**）
    # HVG 筛选会直接修改 adata，只保留 top 1000 个基因
    if filter_min_counts:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=1000, subset=True)

    # Step 4.5: Clean data after HVG (which may create sparse data)
    # Ensure no inf/nan values remain in the HVG-filtered data
    if filter_min_counts:
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            data = adata.X.data.copy()
            data = np.where(np.isfinite(data), data, 0.0)
            data = np.clip(data, -10, 10)
            adata.X.data[:] = data
        else:
            X_arr = np.array(adata.X)
            X_arr = np.where(np.isfinite(X_arr), X_arr, 0.0)
            X_arr = np.clip(X_arr, -10, 10)
            adata.X = X_arr

    # Step 5: 计算 size factor（用于深度学习模型的损失函数，如 ZINB 损失）
    if size_factors:
        count_column = 'n_counts' if 'n_counts' in adata.obs.columns else 'total_counts'
        # Size factor = 细胞总 counts / 中位数总 counts
        adata.obs['size_factors'] = adata.obs[count_column] / np.median(adata.obs[count_column])

    # Step 5.5: 保存 HVG 的 normalized+log1p 数据（用于 ZINB 损失）
    # ZINB 模型需要 normalized+log1p counts（不是 Z-score 标准化数据）
    # 在 Z-score 之前保存 normalized+log1p 数据到 layers
    if hasattr(adata.X, 'toarray'):
        norm_log_data = adata.X.toarray().astype(np.float32)
    else:
        norm_log_data = np.array(adata.X).astype(np.float32)
    adata.layers['norm_log'] = norm_log_data

    # Step 6: 如果未标准化，则进行 Z-score 标准化
    if normalize_input and not is_scaled:
        print("数据未标准化，进行 scale 处理")
        # Z-score: 每个基因减去均值除以标准差
        sc.pp.scale(adata)
        is_scaled = True

    return adata


def prepare_data_for_model(file_path, size_factors=True, filter_min_counts=True, logtrans_input=True, normalize_input=True):
    """
    主入口函数：读取 h5ad 文件并预处理，为模型训练准备数据

    【参数】
        file_path: .h5ad 文件路径
        size_factors: 是否计算 size factors
        filter_min_counts: 是否执行 HVG 筛选
        logtrans_input: 是否执行 log1p 变换
        normalize_input: 是否执行 Z-score 标准化

    【返回】
        X: DataFrame，预处理后的基因表达矩阵（细胞 × 基因）
        Y: Series，细胞类型标签（ground truth）
        sf: Series，size factors（每个细胞的校正因子）
        data: AnnData，预处理后的完整 AnnData 对象

    【使用示例】
        X, Y, sf, adata = prepare_data_for_model('data.h5ad')
        # X: shape = (n_cells, 1000) — 1000 个高度可变基因
        # Y: 细胞类型标签，如 ['Beta', 'Alpha', 'Beta', ...]
        # sf: size factors，如 [0.8, 1.2, 1.0, ...]

    【重要】
        - 本函数是所有模型（scCDCG、scMAE、scDCC 等）的统一数据接口
        - 所有模型都应调用此函数获取预处理后的数据
    """
    # 读取 h5ad 文件（scanpy 的标准数据格式）
    data = sc.read_h5ad(file_path)

    # 执行完整的预处理流程
    data = normalize_sc(data, size_factors=size_factors, filter_min_counts=filter_min_counts,
                         logtrans_input=logtrans_input, normalize_input=normalize_input)

    # 准备模型输入
    X = data.to_df()           # 获取预处理后的 DataFrame（细胞 × 基因）
    Y = data.obs['cell_type']  # 获取细胞类型标签（ground truth）
    sf = data.obs['size_factors']  # 获取 size factors

    return X, Y, sf, data


# =============================================================================
# 以下为旧版 HDF5 格式数据读取函数（保留向后兼容）
# =============================================================================

def read_clean(data):
    """清洗 HDF5 读取的字节数据（处理 bytes_ 类型）"""
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    """
    递归解析 HDF5 Group 为字典
    用于读取旧版 .h5 格式数据
    """
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify=False, skip_exprs=False):
    """
    从 HDF5 文件读取 scRNA-seq 数据（旧版格式）

    【参数】
        filename: .h5 文件路径
        sparsify: 是否转为稀疏矩阵
        skip_exprs: 是否跳过表达量数据

    【返回】
        mat: 表达量矩阵（稀疏或密集）
        obs: 细胞元数据 DataFrame
        var: 基因元数据 DataFrame
        uns: 非结构化数据字典
    """
    with h5py.File(filename, "r") as f:
        # 读取细胞元数据
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        # 读取基因元数据
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        # 读取非结构化数据
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                # 稀疏矩阵格式（data, indices, indptr）
                mat = sp.csr_matrix((
                    exprs_handle["data"][...],
                    exprs_handle["indices"][...],
                    exprs_handle["indptr"][...]
                ), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.csr_matrix(mat)
        else:
            mat = sp.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(data_path):
    """
    预处理 HDF5 格式数据（旧版接口）

    【返回】
        X: 基因表达矩阵（numpy 数组）
        cell_label: 细胞类型标签（整数编码）
        var: 基因元数据
        cell_name: 细胞类型名称
    """
    data_path = data_path

    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    # 获取唯一细胞类型并编码为整数
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label, var, cell_name


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    """
    归一化函数（新版，支持 HVG 筛选参数）

    【新增参数】
        highly_genes: 指定保留的高度可变基因数量（替代旧版的默认 1000）
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    # 小数据集检查整数类型
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    # HVG 筛选（可选，默认不启用）
    if highly_genes is not None:
        sc.pp.highly_variable_genes(
            adata,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_top_genes=highly_genes,
            subset=True
        )

    if normalize_input:
        sc.pp.scale(adata)

    return adata
