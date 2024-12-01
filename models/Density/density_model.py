import sys
sys.path.append('d:/pythonProject/机器学习三/Customer-Analysis')
from models.utils.feature_engineering import load_data
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 加载数据
    data = load_data('d:/pythonProject/机器学习三/Customer-Analysis/data')
    print(f"number of features extracted: {len(data.columns)}")
    print(data.shape)
    print(data.head(5))
    
    # # 进行基于密度的聚类分析（DBSCAN）
    # # 选择一个合理的eps和min_samples值，通常需要调参
    # dbscan = DBSCAN(eps=0.5, min_samples=5)  # 这里的参数需要根据数据进行调整
    # data['cluster'] = dbscan.fit_predict(data)

    # # 聚类结果可视化
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['cluster'], palette='viridis', marker='o', s=60, edgecolor='w', alpha=0.7)
    # plt.title('DBSCAN Clustering Results')
    # plt.xlabel('Feature 1')  # 如果你有多个特征，可以选择不同的特征进行可视化
    # plt.ylabel('Feature 2')
    # plt.legend(title='Cluster', loc='best')
    # plt.show()

if __name__ == "__main__":
    main()