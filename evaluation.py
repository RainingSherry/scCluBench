"""
====================================================================================================
scCluBench 聚类评估指标模块
====================================================================================================

本模块实现了单细胞聚类的标准评估指标，用于衡量聚类结果与真实细胞类型标签的匹配程度。

【核心评估指标】（8个指标，全面衡量聚类质量）

    1. ACC (Accuracy) — 聚类准确率
       使用 Hungarian 算法（线性指派）找到最优标签匹配后计算准确率
       范围 [0, 1]，越接近 1 越好

    2. NMI (Normalized Mutual Information) — 标准化互信息
       衡量两个聚类之间的信息共享程度
       范围 [0, 1]，越接近 1 越好

    3. ARI (Adjusted Rand Index) — 调整兰德指数
       调整了随机划分的期望，考虑了类别不均衡
       范围 [-1, 1]，越接近 1 越好

    4. F1-macro — 宏平均 F1 分数
       每个类别的 F1 分数的算术平均
       范围 [0, 1]，越接近 1 越好

    5. FMI (Fowlkes-Mallows Index) — 福克斯-马洛斯指数
       几何平均形式的精确率和召回率
       范围 [0, 1]，越接近 1 越好

    6. V-measure — V 度量
       同质性（每个簇只含一个类）和完整性（每个类的成员全在同一簇）的调和平均
       范围 [0, 1]，越接近 1 越好

    7. Homogeneity — 同质性
       衡量每个簇是否只包含单一类别的成员
       范围 [0, 1]，越接近 1 越好

    8. Completeness — 完整性
       衡量属于同一类别的成员是否都被分到同一簇
       范围 [0, 1]，越接近 1 越好

【标签匹配问题】
    聚类算法产生的标签是未知的（如 [0,1,2]），
    真实标签也是未知的（如 ['Alpha','Beta','Alpha']），
    直接比较会导致标签不对应的问题。

    例如：聚类结果 [0,0,1,1] vs 真实标签 ['A','A','B','B'] 是完美匹配
         但直接计算准确率会很低（因为 0!=A）

    解决方案：Hungarian 算法（linear_sum_assignment）
    找到一个最优的标签映射，使得整体准确率最大

【代码结构】
    best_map()     — 使用 Hungarian 算法找到最优标签映射
    evaluation()   — 计算所有 8 个评估指标（标准接口）
    cluster_acc()  — 旧版准确率计算（使用 Munkres 算法）
    eva()          — 旧版打印函数
"""

import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    """
    计算聚类准确率（旧版，使用 Munkres 算法）

    【注意】本函数已弃用，建议使用新版 evaluation() 函数

    【算法流程】
        1. 构建混淆矩阵 G，元素 G[i,j] = 真实标签=i且预测标签=j 的细胞数
        2. 使用 Munkres 算法找最优指派（最大化匹配数）
        3. 根据最优指派重新标记预测标签
        4. 计算标记后的准确率

    【参数】
        y_true: 真实标签（整数数组）
        y_pred: 预测标签（整数数组）

    【返回】
        (acc, f1_macro): 准确率和宏平均 F1
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # 如果类别数不匹配，用缺失类别填补
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    # 构建混淆矩阵：G[i,j] = 真实标签 l1[i] 且预测标签 l2[j] 的细胞数
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # Munkres 算法（最大化匹配 → 取负号转为最小化问题）
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # 根据最优指派重新标记
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    """
    打印聚类评估结果（旧版接口）

    【使用示例】
        eva(y_true, y_pred, epoch=100)
        # 输出: 100 :acc 0.8523, nmi 0.7821, ari 0.7234, f1 0.8145
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))


# =============================================================================
# 新版评估函数（2024年重写）
# =============================================================================

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import fowlkes_mallows_score, v_measure_score, silhouette_score, accuracy_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score


def best_map(y_true, y_pred):
    """
    使用 Hungarian 算法（线性指派）找到预测标签到真实标签的最优映射

    【核心思想】
        聚类产生的标签是"无名"的——聚类0可能对应真实类别中的任何一类。
        我们需要找到一个最优映射，使得标签重新标记后的准确率最大化。

    【算法步骤】
        1. 构建匹配矩阵 G：
           G[i,j] = |{细胞: 真实标签=i 且 预测标签=j}|
           即真实类别 i 与预测类别 j 之间重叠的细胞数量

        2. 使用 scipy.optimize.linear_sum_assignment 求解最优指派：
           目标：最大化 Σ G[真实类别, 预测类别]
           约束：每个真实类别恰好匹配一个预测类别

        3. 根据指派结果重排预测标签

    【示例】
        假设真实标签 [0,0,1,1]，预测标签 [0,1,0,1]：
        混淆矩阵可能为：
            预测: 0   1
        真实 0:  1   1
        真实 1:  1   1
        最优指派可能是：真实0→预测0，真实1→预测1
        重排后：[0,0,1,1] — 完全匹配，准确率=1.0

    【参数】
        y_true: 真实标签（整数或字符串数组）
        y_pred: 预测标签（整数数组）

    【返回】
        new_y_pred: 重排后的预测标签（整数数组）
        label_original: 预测标签中的唯一值
        label_truth: 真实标签中的唯一值
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    # 获取唯一标签
    label_set = np.unique(y_true)
    num_class = len(label_set)

    # 构建匹配矩阵 G[i,j] = |真实标签=i ∩ 预测标签=j|
    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            # 计算两个集合的交集大小
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    # Hungarian 算法：最大化 G → 最小化 -G
    A = linear_assignment(-G)

    # 根据指派结果重排预测标签
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        # 将预测标签 label_set[A[1][i]] 映射到真实标签 label_set[A[0][i]]
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]

    return new_y_pred.astype(int), A[1], A[0]


def evaluation(y_true, y_pred):
    """
    主评估函数：计算所有 8 个聚类质量指标

    【这是 benchmark 的标准评估接口】
    所有模型的保存函数 utils.save() 都调用此函数计算指标。

    【参数】
        y_true: 真实细胞类型标签
        y_pred: 模型预测的聚类标签

    【返回】
        (acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, y_pred_):
        9 元组，包含所有指标和重排后的预测标签

    【指标详解】

    1. ACC（准确率）：
       重排标签后正确分类的细胞比例
       ACC = (# 正确分类的细胞) / (# 总细胞)

    2. NMI（标准化互信息）：
       衡量两个聚类之间信息共享的程度
       NMI = 2 × I(Y_true; Y_pred) / (H(Y_true) + H(Y_pred))
       其中 I 为互信息，H 为熵

    3. ARI（调整兰德指数）：
       调整了随机划分的期望
       ARI = (RI - E[RI]) / (max(RI) - E[RI])
       对类别不均衡更鲁棒

    4. F1-macro（宏平均 F1）：
       每个类别的 F1 = 2×precision×recall / (precision+recall)
       宏平均 = 所有类别 F1 的算术平均
       对所有类别一视同仁（即使某些类细胞数很少）

    5. FMI（Fowlkes-Mallows 指数）：
       FMI = TP / sqrt((TP+FP)(TP+FN))
       衡量成对精确率和召回率的几何平均

    6. V-measure（同质性-完整性调和平均）：
       V = (1+β) × 同质性 × 完整性 / (β × 同质性 + 完整性)
       β 默认=1，即两者等权重

    7. Homogeneity（同质性）：
       每个簇是否只包含单一类别的成员
       h = 1 - H(C|G) / H(C)
       其中 H(C|G) 为给定聚类下类别的条件熵

    8. Completeness（完整性）：
       属于同一类别的成员是否全在同一簇
       c = 1 - H(G|C) / H(G)
       其中 H(G|C) 为给定类别下聚类的条件熵
    """
    # Step 1: 使用 Hungarian 算法重排预测标签
    y_pred_, _, _ = best_map(y_true, y_pred)

    # Step 2: 计算各指标
    acc = accuracy_score(y_true, y_pred_)           # 准确率
    f1_macro = f1_score(y_true, y_pred_, average='macro')  # 宏平均 F1
    nmi = nmi_score(y_true, y_pred_, average_method='arithmetic')  # 标准化互信息
    ari = ari_score(y_true, y_pred_)               # 调整兰德指数
    fmi = fowlkes_mallows_score(y_true, y_pred_)    # 福克斯-马洛斯指数
    v_measure = v_measure_score(y_true, y_pred_)   # V 度量
    hom = homogeneity_score(y_true, y_pred_)        # 同质性
    com = completeness_score(y_true, y_pred_)       # 完整性

    # 返回所有指标和重排后的预测标签
    return acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, y_pred_
