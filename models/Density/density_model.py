import numpy as np
from collections import deque
from sklearn.metrics.pairwise import pairwise_distances

class DBSCAN:
    def __init__(self, eps=0.3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

    def fit(self, X):
        """ 核心的fit方法，返回self，以便链式调用 """
        n_samples = len(X)
        self.labels_ = np.full(n_samples, -1)  # 初始化所有点的标签为-1（噪声）
        visited = np.zeros(n_samples, dtype=bool)  # 访问标记
        cluster_id = 0  # 初始化簇ID

        # 计算距离矩阵
        distance_matrix = pairwise_distances(X, metric=self.metric)

        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors = self._region_query(distance_matrix, i)
                if len(neighbors) < self.min_samples:
                    # 标记为噪声
                    self.labels_[i] = -1
                else:
                    # 发现一个新簇
                    self._expand_cluster(X, distance_matrix, i, neighbors, cluster_id, visited)
                    cluster_id += 1

        # 计算核心样本的索引和组件
        self.core_sample_indices_ = np.where(self.labels_ != -1)[0]
        self.components_ = X[self.core_sample_indices_]

        return self  # 返回self，以便链式调用

    def fit_predict(self, X):
        """ 和sklearn一样，fit_predict方法返回标签 """
        self.fit(X)
        return self.labels_

    def _region_query(self, distance_matrix, point_idx):
        """ 查找eps邻域内的点，使用距离矩阵 """
        neighbors = np.where(distance_matrix[point_idx] < self.eps)[0]
        return neighbors

    def _expand_cluster(self, X, distance_matrix, point_idx, neighbors, cluster_id, visited):
        """ 扩展簇 """
        self.labels_[point_idx] = cluster_id
        queue = deque(neighbors)
        
        while queue:
            current_point = queue.popleft()
            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = self._region_query(distance_matrix, current_point)
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            if self.labels_[current_point] == -1:
                self.labels_[current_point] = cluster_id

    def get_params(self, deep=True):
        """ 获取参数，符合sklearn的API """
        return {"eps": self.eps, "min_samples": self.min_samples, "metric": self.metric}

    def __repr__(self):
        return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples}, metric='{self.metric}')"

