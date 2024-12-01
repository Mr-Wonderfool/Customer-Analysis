import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import warnings
import sys
from sklearn.metrics import silhouette_score
import seaborn as sns

# 忽略警告信息
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# 设置随机种子，确保结果可重现
np.random.seed(42)


def clean_data():
    """数据清洗"""
    # 加载数据
    data = pd.read_csv("../../data/marketing_campaign.csv", sep="\t")

    # 删除包含缺失值的行
    data = data.dropna()

    # 转换 "Dt_Customer" 列为日期格式
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")

    # 提取注册日期与最新日期的天数差
    dates = [i.date() for i in data["Dt_Customer"]]
    days = [(max(dates) - i).days for i in dates]
    data["Customer_For"] = days

    # 转换 "Customer_For" 为数值型数据
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    # 计算客户年龄并添加新特征
    data["Age"] = 2024 - data["Year_Birth"]

    # 计算总消费额
    data["Spent"] = data[
        [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
        ]
    ].sum(axis=1)

    # 根据婚姻状况推测生活状况
    data["Living_With"] = data["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone",
        }
    )

    # 创建家庭成员特征
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = (
        data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    )
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

    # 简化教育水平分组
    data["Education"] = data["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )

    # 修改消费类别列名
    data = data.rename(
        columns={
            "MntWines": "Wines",
            "MntFruits": "Fruits",
            "MntMeatProducts": "Meat",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold",
        }
    )

    # 删除冗余列
    to_drop = [
        "Marital_Status",
        "Dt_Customer",
        "Z_CostContact",
        "Z_Revenue",
        "Year_Birth",
        "ID",
    ]
    data = data.drop(to_drop, axis=1)

    # 删除异常值：年龄大于90岁和收入超过60万
    data = data[(data["Age"] < 90) & (data["Income"] < 600000)]

    return data


def preprocess_data(data):
    """数据预处理"""
    # 对所有分类特征进行标签编码
    s = data.dtypes == "object"  # 查找分类特征
    object_cols = list(s[s].index)  # 获取分类特征名
    LE = LabelEncoder()  # 初始化标签编码器
    for i in object_cols:
        data[i] = data[[i]].apply(LE.fit_transform)  # 标签编码

    # 删除与促销相关的列
    ds = data.copy()
    cols_del = [
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Complain",
        "Response",
    ]
    ds = ds.drop(cols_del, axis=1)

    # 数据标准化
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

    return scaled_ds


def apply_pca(scaled_ds):
    """使用PCA降维"""
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])

    return PCA_ds


def choose_best_k(PCA_ds):
    """使用肘部法则选择最佳聚类数"""
    Elbow_M = KElbowVisualizer(KMeans(), k=10)
    Elbow_M.fit(PCA_ds)
    Elbow_M.show()


def kmeans_custom(X, k, max_iters=300):
    """手写实现KMeans"""
    # 随机选择k个初始质心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # 计算每个点到质心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 为每个点分配簇
        labels = np.argmin(distances, axis=1)

        # 计算每个簇的平均值作为新质心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # 如果质心不变，停止迭代
        if np.all(centroids == new_centroids):
            break

        # 更新质心
        centroids = new_centroids

    return centroids, labels


def plot_3d_clusters(PCA_ds, cluster_labels, title):
    """绘制三维聚类图"""
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, projection="3d", label="bla")

    ax.scatter(
        PCA_ds["col1"],
        PCA_ds["col2"],
        PCA_ds["col3"],
        s=40,
        c=cluster_labels,
        cmap="viridis",
        marker="o",
    )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    plt.show()


def calculate_silhouette(PCA_ds, cluster_labels, method):
    """计算轮廓系数"""
    silhouette_avg = silhouette_score(PCA_ds, cluster_labels)
    print(f"Silhouette Score ({method}): {silhouette_avg:.4f}")


def plot_spending_distribution(data, cluster_labels):
    """绘制消费金额与聚类之间的分布图"""
    # 将聚类标签添加到数据中
    data["Clusters"] = cluster_labels

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 使用swarmplot显示每个聚类的散点分布，alpha控制透明度
    sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5)

    # 使用boxenplot绘制每个聚类的消费金额箱形图，显示分布情况
    sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette="viridis")

    # 设置标题和标签
    plt.title("Distribution of Spending (Spent) by Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Spent Amount")

    # 显示图表
    plt.show()


def plot_category_comparison(data, cluster_labels, categories):
    """绘制各聚类在各消费类别上的消费比较"""
    # 添加聚类标签到数据中
    data["Cluster"] = cluster_labels

    # 使用 melt() 将数据从宽格式转换为长格式，方便绘制多类别比较图
    melted_data = data[["Cluster"] + categories].melt(
        id_vars=["Cluster"], value_vars=categories
    )

    # 设置图形大小
    plt.figure(figsize=(16, 8))

    # 绘制箱形图，显示每个聚类在各消费类别上的分布情况
    sns.boxplot(
        x="Cluster", y="value", hue="variable", data=melted_data, palette="Set3"
    )

    # 设置标题和标签
    plt.title("Category Spending Comparison Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Spending Amount")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    # 显示图表
    plt.show()


def plot_user_profile(data, cluster_labels):
    """逐个分析每条个人特征，按聚类结果绘制用户画像"""
    # 定义个人特征列表
    Personal = [
        "Kidhome",
        "Teenhome",
        "Customer_For",
        "Age",
        "Children",
        "Family_Size",
        "Is_Parent",
        "Education",
        "Living_With",
    ]

    # 将聚类标签添加到数据中
    data["Clusters"] = cluster_labels

    # 自定义颜色调色板
    pal = sns.color_palette("viridis", as_cmap=True)

    # 遍历个人特征列表，对每个特征与消费金额 ("Spent") 之间的关系进行可视化
    for i in Personal:
        # 使用核密度估计 (KDE) 绘制特征与消费金额之间的关系，并按聚类结果着色
        sns.jointplot(
            x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal
        )
        # 显示绘制的图形
        plt.show()


if __name__ == "__main__":
    # 清洗数据
    data = clean_data()

    # 数据预处理
    scaled_ds = preprocess_data(data)

    # 降维
    PCA_ds = apply_pca(scaled_ds)

    # 选择最佳聚类数
    choose_best_k(PCA_ds)

    # 使用手写KMeans进行聚类
    centroids, cluster_labels = kmeans_custom(PCA_ds.values, k=4)
    plot_3d_clusters(
        PCA_ds, cluster_labels, "3D Clustering Plot(Custom Implementation)"
    )
    calculate_silhouette(PCA_ds, cluster_labels, "Custom Implementation")

    # 使用库函数KMeans进行聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(PCA_ds)
    cluster_labels_sklearn = kmeans.labels_
    plot_3d_clusters(
        PCA_ds, cluster_labels_sklearn, "3D Clustering Plot(Library Function)"
    )
    calculate_silhouette(PCA_ds, cluster_labels_sklearn, "Library Function")

    # 绘制消费金额与聚类之间的分布图
    plot_spending_distribution(data, cluster_labels)

    # 各类别消费对比
    categories = ["Wines", "Meat", "Fish", "Fruits", "Sweets", "Gold"]
    plot_category_comparison(data, cluster_labels, categories)

    # 绘制用户画像
    plot_user_profile(data, cluster_labels)
