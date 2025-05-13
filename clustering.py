'''
1. Select at least two suitable clustering algorithm (e.g., K-means, hierarchical clustering).
2. Apply the algorithms to the preprocessed dataset.
3. Evaluate the results using multiple metrics.
4. Visualize the clusters (e.g., scatter plot with cluster labels).
5. Determine the best clustering results and justify it.
'''
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Internal clustering evaluation metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# External clustering evaluation metrics
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
import pandas as pd
import numpy as np

# 1.读取数据
data = pd.read_csv('./data/adult_train_processed.csv')
# 标签/响应变量：年收入是否大于50K
label = data['income']
# 数据/输入变量：其他特征
feature = data.drop(columns=['income'])

# 2.数据降维
# 2.1使用PCA进行降维
print("使用PCA进行降维")
pca = PCA(n_components=0.99)
feature_pca = None
feature_pca = pca.fit_transform(feature)
if feature_pca is not None:
    print("PCA降维完成")
else:
    print("PCA降维失败")
    exit()
print(f"累积方差贡献率: {sum(pca.explained_variance_ratio_):.4f}")

# 2.2使用MDS进行降维
#print("使用MDS进行降维")
#feature_mds = None
#feature_mds = MDS(n_components=2, random_state=42).fit_transform(feature)
#if feature_mds is not None:
#    print("MDS降维完成")
#else:
#    print("MDS降维失败")
#    exit()

# 2.3使用LDA进行降维
print("使用LDA进行降维")
# n_components cannot be larger than min(n_classes - 1, n_features)
# 这里n_classes = 2, n_features = 2，所以n_components = 1
lda = LinearDiscriminantAnalysis(n_components=1)
feature_lda = None
feature_lda = lda.fit_transform(feature, label)
if feature_lda is not None:
    print("LDA降维完成")
else:
    print("LDA降维失败")
    exit()

# 2.4先使用PCA进行降维，然后使用LDA进行降维
feature_pca_lda = None
feature_pca_lda = lda.fit_transform(feature_pca, label)
if feature_pca_lda is not None:
    print("PCA+LDA降维完成")
else:
    print("PCA+LDA降维失败")
    exit()
# 3.聚类算法
# 3.1 K-Means++聚类

n_clusters = label.nunique()  # 2个聚类
kmeans = KMeans(n_clusters, init='k-means++', random_state=42)
# 尝试不同降维方法和原始数据进行K-Means聚类
kmeans_labels_pca = kmeans.fit_predict(feature_pca)
#kmeans_labels_mds = kmeans.fit_predict(feature_mds)
kmeans_labels_lda = kmeans.fit_predict(feature_lda)
kmeans_labels_raw = kmeans.fit_predict(feature)
kmeans_labels_pca_lda = kmeans.fit_predict(feature_pca_lda)

# 3.2 DBSCAN聚类

dbscan = DBSCAN(eps=0.5, min_samples=3)
# 尝试不同降维方法和原始数据进行DBSCAN聚类
dbscan_labels_pca = dbscan.fit_predict(feature_pca)
#dbscan_labels_mds = dbscan.fit_predict(feature_mds)
dbscan_labels_lda = dbscan.fit_predict(feature_lda)
dbscan_labels_raw = dbscan.fit_predict(feature)
dbscan_labels_pca_lda = dbscan.fit_predict(feature_pca_lda)

# DBSCAN聚类结果中，-1表示噪声点

# 3.3 层次聚类

agglo = AgglomerativeClustering(n_clusters=n_clusters)
# 尝试不同降维方法和原始数据进行层次聚类
agglo_labels_pca = agglo.fit_predict(feature_pca)
#agglo_labels_mds = agglo.fit_predict(feature_mds)
agglo_labels_lda = agglo.fit_predict(feature_lda)
agglo_labels_raw = agglo.fit_predict(feature)
agglo_labels_pca_lda = agglo.fit_predict(feature_pca_lda)

# 4.评估聚类结果

# 4.1 内部评估指标
def evaluate_clustering_internal(data, labels):
    """
    Evaluate clustering results using internal metrics.
    
    Parameters:
    -----------
    data : array-like
        High-dimensional data used for clustering.
    labels : array-like
        Cluster labels assigned to each data point.
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics.
    """
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    
    return silhouette, calinski_harabasz, davies_bouldin

# 4.2 外部评估指标
def evaluate_clustering_external(true_labels, predicted_labels):
    """
    Evaluate clustering results using external metrics.
    
    Parameters:
    -----------
    true_labels : array-like
        True class labels for each data point.
    predicted_labels : array-like
        Cluster labels assigned to each data point.
    
    Returns:
    --------
    dict : Dictionary containing evaluation metrics.
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    
    return ari, ami, v_measure

# 4.3 对于三种聚类方法以及不同降维方法的评估

def evaluate(external_metrics:tuple, internal_metrics:tuple, method:str):
    # print the evaluation results for the clustering methods
    ari, ami, v_measure = external_metrics
    silhouette, calinski_harabasz, davies_bouldin = internal_metrics
    print(f"聚类方法: {method}")
    print(f"外部评估指标: ARI:{ari:.4f}(Range: [-1, 1]), AMI:{ami:.4f}(Range: [0, 1]), V-measure:{v_measure:.4f}(Range: [0, 1])")
    print(f"内部评估指标: Silhouette:{silhouette:.4f}(Range: [-1, 1]), Calinski-Harabasz:{calinski_harabasz:.4f}(Range: [0, +∞]), Davies-Bouldin:{davies_bouldin:.4f}(Range: [0, +∞])")
    print("-----------------------------------------------------")

# 4.3.1 K-Means++聚类评估
k_means_results = {"Raw": kmeans_labels_raw, "PCA": kmeans_labels_pca, "LDA": kmeans_labels_lda, "PCA+LDA": kmeans_labels_pca_lda}
'''
for method, kmeans_labels in k_means_results.items():
    internal_metrics = evaluate_clustering_internal(feature, kmeans_labels)
    external_metrics = evaluate_clustering_external(label, kmeans_labels)
    evaluate(external_metrics, internal_metrics, f"K-Means++({method})")
'''


# 4.3.2 DBSCAN聚类评估
dbscan_results = {"Raw": dbscan_labels_raw, "PCA": dbscan_labels_pca, "LDA": dbscan_labels_lda, "PCA+LDA": dbscan_labels_pca_lda}
'''
for method, dbscan_labels in dbscan_results.items():
    if len(np.unique(dbscan_labels)) == 1:
        print(f"DBSCAN({method})聚类结果只有一个簇，无法评估")
        continue
    else:
        internal_metrics = evaluate_clustering_internal(feature, dbscan_labels)
    external_metrics = evaluate_clustering_external(label, dbscan_labels)
    evaluate(external_metrics, internal_metrics, f"DBSCAN({method})")
'''


# 4.3.3 层次聚类评估
agglo_results = {"Raw": agglo_labels_raw, "PCA": agglo_labels_pca, "LDA": agglo_labels_lda, "PCA+LDA": agglo_labels_pca_lda}
'''
for method, agglo_labels in agglo_results.items():
    internal_metrics = evaluate_clustering_internal(feature, agglo_labels)
    external_metrics = evaluate_clustering_external(label, agglo_labels)
    evaluate(external_metrics, internal_metrics, f"AgglomerativeClustering({method})")
'''



# 5.可视化聚类结果
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d(data, labels, method: str):
    """
    Visualize clustering results in 2D or 1D.
    
    Parameters:
    -----------
    data : array-like
        Data used for clustering.
    labels : array-like
        Cluster labels assigned to each data point.
    method : str
        Clustering method used.
    """
    print(f"可视化{method}聚类结果...")
    plt.figure(figsize=(8, 6))
    if data.shape[1] == 1:  # 如果只有1个特征
        plt.scatter(data[:, 0], [0] * len(data), c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.xlabel("Feature 1")
        plt.ylabel("Cluster Label")
    else:  # 正常二维可视化
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.title(f"Clustering Results ({method})")
    plt.colorbar(label="Cluster Label")
    plt.show()
    # 保存可视化结果到./figures/下
    plt.savefig(f"./figures/{method}.png")
    plt.close()
    print(f"可视化结果已保存到./figures/{method}.png")

# 5.1 可视化K-Means++聚类结果
print("可视化K-Means++聚类结果...")
# Raw数据可视化
visualize_2d(feature.values, kmeans_labels_raw, "K-Means++(Raw)")
# PCA数据可视化
visualize_2d(feature_pca, kmeans_labels_pca, "K-Means++(PCA)")
# LDA数据可视化
visualize_2d(feature_lda, kmeans_labels_lda, "K-Means++(LDA)")
# PCA+LDA数据可视化
visualize_2d(feature_pca_lda, kmeans_labels_pca_lda, "K-Means++(PCA+LDA)")


# 5.2 可视化DBSCAN聚类结果
print("可视化DBSCAN聚类结果...")
# Raw数据可视化
visualize_2d(feature.values, dbscan_labels_raw, "DBSCAN(Raw)")
# PCA数据可视化
visualize_2d(feature_pca, dbscan_labels_pca, "DBSCAN(PCA)")
# LDA数据可视化
visualize_2d(feature_lda, dbscan_labels_lda, "DBSCAN(LDA)")
# PCA+LDA数据可视化
visualize_2d(feature_pca_lda, dbscan_labels_pca_lda, "DBSCAN(PCA+LDA)")

# 5.3 可视化层次聚类结果
print("可视化层次聚类结果...")
# Raw数据可视化
visualize_2d(feature.values, agglo_labels_raw, "AgglomerativeClustering(Raw)")
# PCA数据可视化
visualize_2d(feature_pca, agglo_labels_pca, "AgglomerativeClustering(PCA)")
# LDA数据可视化
visualize_2d(feature_lda, agglo_labels_lda, "AgglomerativeClustering(LDA)")
# PCA+LDA数据可视化
visualize_2d(feature_pca_lda, agglo_labels_pca_lda, "AgglomerativeClustering(PCA+LDA)")

