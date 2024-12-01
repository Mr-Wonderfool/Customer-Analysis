from models.utils.feature_engineering import load_data, standard_scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from models.utils.visualization import Plotter
import seaborn as sns
plt.style.use("ggplot")
palette = Plotter._PALETTE_

def main():
    data = load_data("../../data")
    data_scaled = standard_scale(data)
    pca = PCA(n_components=3)
    pca.fit(data_scaled)
    data_pca = pca.transform(data_scaled)
    cluster = AgglomerativeClustering(n_clusters=4, )
    yhat = cluster.fit_predict(data_pca)
    data["Clusters"] = yhat
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111, projection='3d', label="bla")
    ax.scatter(data_pca[..., 0], data_pca[..., 1], data_pca[..., 2], s=40, c=yhat, marker='o', cmap='Blues')
    ax.set_title("Hierarchical Clustering Visualization")
    plt.savefig("../../images/hierarchical_clustering.png", dpi=150)
    pl = sns.scatterplot(data = data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=palette)
    pl.set_title("Cluster's Profile Based On Income And Spending")
    plt.legend()
    plt.savefig("../../images/cluster_profile_spent_income.png", dpi=150)

if __name__ == "__main__":
    main()