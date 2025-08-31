# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np
import pandas as pd

# ============ 工具函数：熵、条件熵、信息增益 ============
def entropy(y: np.ndarray) -> float:
    """
    公式(1) H(D) = - sum_i p_i * log2(p_i)
    y: 类别标签向量 (n_samples,)
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # 避免log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _discretize_feature(f: np.ndarray, bins: int = 10, strategy: str = "quantile") -> np.ndarray:
    """
    连续特征离散化用于条件熵计算：
    - strategy="quantile" 用分位数等频切分
    - strategy="uniform"  用等宽切分
    """
    f = np.asarray(f).astype(float)
    # 全常数列直接返回单一箱
    if np.nanstd(f) == 0 or np.all(np.isnan(f)):
        return np.zeros_like(f, dtype=int)

    if strategy == "quantile":
        # 生成分位点，去重防止相同边界
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.nanquantile(f, qs))
    else:  # uniform
        fmin, fmax = np.nanmin(f), np.nanmax(f)
        if fmax == fmin:
            edges = np.array([fmin, fmax + 1e-9])
        else:
            edges = np.linspace(fmin, fmax, bins + 1)

    # pandas.cut 处理边界/缺失比较稳妥
    codes = pd.cut(f, bins=edges, include_lowest=True, labels=False)
    # 缺失值单独视作一箱
    codes = codes.astype("float")
    codes[pd.isna(codes)] = -1  # 缺失箱
    return codes.astype(int)


def conditional_entropy(y: np.ndarray, f: np.ndarray, bins: int = 10, strategy: str = "quantile") -> float:
    """
    公式(2) H(D|F) = sum_{f in values(F)} P(F=f) * H(D|F=f)
    对连续F做离散化；对离散F直接分组。
    """
    y = np.asarray(y)
    f = np.asarray(f)

    # 判断是否需要离散化
    is_numeric = np.issubdtype(f.dtype, np.number)
    fv = _discretize_feature(f, bins=bins, strategy=strategy) if is_numeric else f

    H = 0.0
    values, counts = np.unique(fv, return_counts=True)
    n = len(fv)
    for v, c in zip(values, counts):
        mask = (fv == v)
        if c == 0:
            continue
        p_v = c / n
        H_D_given_v = entropy(y[mask])
        H += p_v * H_D_given_v
    return H


def information_gain(y: np.ndarray, X: np.ndarray, bins: int = 10, strategy: str = "quantile") -> np.ndarray:
    """
    公式(3) IG(F_j) = H(D) - H(D|F_j) 逐列计算
    返回形状: (n_features,)
    """
    base_H = entropy(y)
    ig_list = []
    for j in range(X.shape[1]):
        H_cond = conditional_entropy(y, X[:, j], bins=bins, strategy=strategy)
        ig_list.append(base_H - H_cond)
    return np.asarray(ig_list)


# ============ 相似度 S：欧氏 + 皮尔逊（公式4） ============
def euclidean_similarity(Z: np.ndarray) -> np.ndarray:
    """
    将特征先标准化后的矩阵 Z (n_samples, n_features)
    先算欧氏距离，再映射到[0,1]相似度： SimE = 1 / (1 + dist)
    返回 (n_features, n_features)
    """
    # 列向量为特征：转置后按列算距离更直观
    F = Z.T  # (n_features, n_samples)
    # 距离矩阵
    # dist(i,j) = ||F_i - F_j||
    # 用向量化技巧：||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq = np.sum(F * F, axis=1, keepdims=True)  # (n_features,1)
    dist2 = sq + sq.T - 2 * F @ F.T
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)
    # 映射为相似度
    return 1.0 / (1.0 + dist)


def pearson_similarity(Z: np.ndarray) -> np.ndarray:
    """
    标准化后的矩阵 Z 的列间皮尔逊相关 |corr|，范围[0,1]
    返回 (n_features, n_features)
    """
    # np.corrcoef 期望变量为行，这里转置：
    C = np.corrcoef(Z, rowvar=False)  # (n_features, n_features)
    C = np.nan_to_num(C, nan=0.0)
    return np.abs(C)


def combined_similarity(Z: np.ndarray, w1: float = 0.5, w2: float = 0.5) -> np.ndarray:
    """
    公式(4) S = | w1 * SimEuclidean + w2 * Pearson |
    注意：输入 Z 应该是已经做过标准化的特征矩阵
    """
    if w1 < 0 or w2 < 0 or (w1 + w2) == 0:
        raise ValueError("w1, w2 必须非负且 w1 + w2 > 0")
    # 归一化权重
    s = w1 + w2
    w1, w2 = w1 / s, w2 / s
    Se = euclidean_similarity(Z)
    Sp = pearson_similarity(Z)
    S = np.abs(w1 * Se + w2 * Sp)
    # 数值稳定/对角置1
    np.fill_diagonal(S, 1.0)
    return S


# ============ 主流程封装：IG选特征 + 相似度成 token ============
@dataclass
class HOHEConfig:
    bins: int = 10                 # 条件熵离散箱数
    bin_strategy: str = "quantile" # "quantile" or "uniform"
    topk_ig: Optional[int] = None  # 以Top-K的方式选特征；与 ig_threshold 二选一
    ig_threshold: Optional[float] = None  # 以阈值方式选特征
    w1: float = 0.5                # 欧氏相似度权重
    w2: float = 0.5                # 皮尔逊相似度权重
    sim_threshold: float = 0.7     # 组 token 的相似度阈值
    standardize: bool = True       # 在相似度计算前是否标准化


class HOHESelector:
    """
    实现论文里公式(1)-(4) + “高相似度成 token”。
    fit(X, y) 后得到:
      - self.info_gain_: 每个原始特征的 IG
      - self.selected_index_: 被选特征的下标
      - self.S_: 被选特征之间的相似度矩阵
      - self.tokens_: token 列表（每个元素是一个特征下标的list）
    """
    def __init__(self, config: HOHEConfig = HOHEConfig()):
        self.cfg = config
        self.info_gain_: Optional[np.ndarray] = None
        self.selected_index_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.tokens_: Optional[List[List[int]]] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HOHESelector":
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        assert X.ndim == 2, "X 需要是二维矩阵 (n_samples, n_features)"

        # ---- (1)(2)(3) 信息增益 ----
        ig = information_gain(y, X, bins=self.cfg.bins, strategy=self.cfg.bin_strategy)
        self.info_gain_ = ig

        # 选择特征
        if self.cfg.topk_ig is not None and self.cfg.topk_ig > 0:
            idx = np.argsort(ig)[::-1][: self.cfg.topk_ig]
        elif self.cfg.ig_threshold is not None:
            idx = np.where(ig >= self.cfg.ig_threshold)[0]
        else:
            # 若两者都未设，默认保留全部（也方便只做相似度成 token）
            idx = np.arange(X.shape[1])
        self.selected_index_ = np.sort(idx)

        # ---- 特征标准化（用于相似度计算）----
        Z = X[:, self.selected_index_]
        if self.cfg.standardize:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            Z = self.scaler_.fit_transform(Z)

        # ---- (4) 计算综合相似度 S ----
        S = combined_similarity(Z, w1=self.cfg.w1, w2=self.cfg.w2)
        self.S_ = S

        # ---- 基于阈值组 token ----
        self.tokens_ = self._build_tokens(S, threshold=self.cfg.sim_threshold)
        return self

    @staticmethod
    def _build_tokens(S: np.ndarray, threshold: float) -> List[List[int]]:
        """
        规则：“每个属性与其相似度 > T 的 N-1 个属性组合成一个自然 token”
        这里实现为：对每个特征 i，取所有 j!=i 且 S[i,j] > T，形成一个包含 i 的集合。
        """
        n = S.shape[0]
        tokens = []
        for i in range(n):
            group = [i] + [j for j in range(n) if j != i and S[i, j] > threshold]
            # 去重+排序，保证稳定
            group = sorted(set(group))
            tokens.append(group)
        return tokens
def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x.cpu())
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    # print("F_list shape：",F_list.shape)
    # print("元组的大小是：", len(F_list))  2

    for f in F_list:

        # print("f.type :",type(f))
        # if f is not None:
        #     print(f.shape)   #(12311,2048)
        if f is not None and len(f) != 0:
            # deal with the dimension that more than two

            if len(f.shape) > 2:


                f = f.reshape(-1, f.shape[-1])

            # if features is None:
            #     features=f
            # else:
            #     features = np.hstack((features, f))

            # normal each column
            if normal_col:

                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max

            # facing the first feature matrix appended to fused feature matrix

            if features is None:
                features = f

            else:
                features = np.hstack((features, f))



    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and len(h) != 0:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


# 生成w矩阵

# print("Output shape:", output_data.shape)#1320,1
def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # print(f'n_edge: {n_edge}')#1320

    # the weight of the hyperedge
    W = np.ones(n_edge)


    # the degree of the node
    DV = np.sum(H * W, axis=1)  #相当于返回了本身
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    # print("Shape of DE:", DE.shape)#(1320,)
    # print("Shape of DV:", DV.shape)#(1320,)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    # print("invDE shape",invDE.shape)

    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # print("DV2 shape",DV2.shape)
    # print("np.diag(W)  shape",np.diag(W))
    W = np.mat(np.diag(W))

    # print("W shape", W.shape)  #1,1
    H = np.mat(H)
    HT = H.T
    # print("HT shape",HT.shape)

    if variable_weight:

        DV2_H = DV2 * H  #1320,1320
        invDE_HT_DV2 = invDE * HT * DV2
        # print("i am here1")
        return DV2_H, W, invDE_HT_DV2
    else:
        # print("i am here2")

        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[5], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    # print("X.shape :", X.shape)#1320，128

    dis_mat = Eu_dis(X)
    # print("dis_mat.shape:", dis_mat.shape)  # 1320,1320
    #3,3
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        # print("H_tmp.shape After construct_H_with_KNN_from_distance:", H_tmp.shape)#1320,1320

        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
