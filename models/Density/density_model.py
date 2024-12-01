import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.3, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        # 样本数
        n_samples = len(X)
        # 用-1表示所有点未被访问
        self.labels_ = np.full(n_samples, -1)
        # 访问过的点
        visited = np.zeros(n_samples, dtype=bool)
        # cluster_id初始化为0
        cluster_id = 0
        
        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors = self._region_query(X, i)
                if len(neighbors) < self.min_samples:
                    # 噪声点
                    self.labels_[i] = -1
                else:
                    # 发现一个新簇
                    self._expand_cluster(X, i, neighbors, cluster_id, visited)
                    cluster_id += 1

    def _region_query(self, X, point_idx):
        """ 查找eps邻域内的点 """
        neighbors = []
        for i, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """ 扩展簇 """
        self.labels_[point_idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            current_point = queue.popleft()
            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = self._region_query(X, current_point)
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            if self.labels_[current_point] == -1:
                self.labels_[current_point] = cluster_id