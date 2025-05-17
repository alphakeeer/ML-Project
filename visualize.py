'''
对数据进行可视化
1.Visualize high-dimensional data in a 2D and/or 3D space using t-SNE
2. Create a scatter plot of the resulting embedding, coloring points by class labels if applicable.
3. Analyze the visualization to identify patterns or clusters.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def visualize_tsne_2d(data, labels=None, title='t-SNE Visualization (2D)', 
                     save_path=None, colormap='tab20', balance_classes=False,
                     point_size=30, alpha=0.5, legend_loc='best'):
    """
    优化后的2D t-SNE可视化函数，支持类别不平衡处理
    
    Parameters新增:
    balance_classes : bool, 是否进行类别平衡采样
    point_size : int/array, 数据点大小
    alpha : float, 透明度
    legend_loc : str, 图例位置
    """
    # 类别平衡处理
    if balance_classes and labels is not None:
        data, labels = _balance_classes(data, labels)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1200)
    embedded_data = tsne.fit_transform(data)
    
    # 创建画布
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # 按类别样本量倒序排列（先画多数类）
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_labels = unique_labels[np.argsort(-counts)]
        
        # 创建离散颜色映射
        cmap = plt.get_cmap('Set1')  # 使用对比度更高的离散调色板
        colors = cmap(np.linspace(0, 1, len(unique_labels)))

        # 分层绘制各个类别
        for idx, label in enumerate(sorted_labels):
            mask = labels == label
            plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1],
                        c=[colors[idx]], label=str(label),
                        s=point_size, alpha=alpha, edgecolors='w', linewidth=0.3)
            
        # 添加图例
        plt.legend(title='Class Labels', loc=legend_loc,
                  frameon=True, framealpha=0.8)
    else:
        plt.scatter(embedded_data[:, 0], embedded_data[:, 1],
                   s=point_size, alpha=alpha, edgecolors='w', linewidth=0.3)
    
    # 添加标签和网格
    plt.title(title, pad=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    
    plt.show()
    return embedded_data

def _balance_classes(data, labels):
    """内部使用的类别平衡函数"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    
    sampled_indices = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > min_count:
            indices = np.random.choice(indices, min_count, replace=False)
        sampled_indices.append(indices)
    
    sampled_indices = np.concatenate(sampled_indices)
    return data[sampled_indices], labels[sampled_indices]


def visualize_tsne_3d(data, labels=None, title='t-SNE Visualization (3D)', save_path=None, colormap='viridis', balance_classes=False):
    """
    Visualize high-dimensional data in 3D using t-SNE.
    
    Parameters:
    -----------
    data : array-like
        High-dimensional data to visualize.
    labels : array-like, optional
        Class labels for each data point.
    title : str, optional
        Title for the plot.
    save_path : str, optional
        Path to save the visualization.
    colormap : str, optional
        Colormap to use for the scatter plot.
    """
    if balance_classes and labels is not None:
        data, labels = _balance_classes(data, labels)
    
    # Apply t-SNE for dimensionality reduction to 3D
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1200)
    embedded_data = tsne.fit_transform(data)
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        # If labels are provided, color points by labels
        scatter = ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 
                  c=labels, cmap='Set1', alpha=0.5, s=10)
        plt.colorbar(scatter, label='Class Labels')
    else:
        # If no labels, use a single color
        ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 
                   alpha=0.8, s=50)
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE Feature 1')
    ax.set_ylabel('t-SNE Feature 2')
    ax.set_zlabel('t-SNE Feature 3')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    
    return embedded_data

def analyze_clusters(embedded_data, labels=None):
    """
    Analyze the t-SNE embedding to identify patterns or clusters.
    
    Parameters:
    -----------
    embedded_data : array-like
        Low-dimensional embedding from t-SNE.
    labels : array-like, optional
        Class labels for each data point.
    """
    if labels is None:
        print("No labels provided for cluster analysis.")
        return
    
    # Calculate cluster statistics
    unique_labels = np.unique(labels)
    print(f"Number of clusters/classes: {len(unique_labels)}")
    
    # Compute basic statistics for each cluster
    for label in unique_labels:
        cluster_points = embedded_data[labels == label]
        center = np.mean(cluster_points, axis=0)
        std_dev = np.std(cluster_points, axis=0)
        count = len(cluster_points)
        
        print(f"\nCluster/Class {label}:")
        print(f"  Number of points: {count}")
        print(f"  Center: {center}")
        print(f"  Standard deviation: {std_dev}")
        
    # Visual analysis with a pairplot if using 3D embedding
    if embedded_data.shape[1] >= 2:
        df = pd.DataFrame(embedded_data[:, :3], columns=[f'Component {i+1}' for i in range(min(3, embedded_data.shape[1]))])
        if labels is not None:
            df['Label'] = labels
            sns.pairplot(df, hue='Label')
            plt.suptitle('Pairwise Relationships Between t-SNE Components', y=1.02)
            plt.show()

if __name__ == "__main__":
    # Load data from the specified CSV files
    train_data = pd.read_csv('./data/adult_train_processed.csv')
    test_data = pd.read_csv('./data/adult_test_processed.csv')
    
    print(f"Loaded training data shape: {train_data.shape}")
    print(f"Loaded test data shape: {test_data.shape}")
    
    # Assuming the last column is the target/label
    X_train = train_data.iloc[:, :-7].values
    y_train = train_data.iloc[:, 7].values
    
    X_test = test_data.iloc[:, :7].values
    y_test = test_data.iloc[:, 7].values
    
    
    print("\n--- Training Data Visualization ---")
    print("Visualizing training data in 2D...")
    embedded_train_2d = visualize_tsne_2d(
        X_train, 
        y_train, 
        title='t-SNE Visualization of Adult Income Dataset - Training Data (2D)',
        save_path='./figures/tsne_train_2d.png',
        colormap='tab10'  # Using a discrete colormap better for categorical data
    )
    
    
    print("\nVisualizing training data in 3D...")
    embedded_train_3d = visualize_tsne_3d(
        X_train, 
        y_train, 
        title='t-SNE Visualization of Adult Income Dataset - Training Data (3D)',
        save_path='./figures/tsne_train_3d.png',
        colormap='tab10'  # Using a discrete colormap better for categorical data
    )
    
    print("\n--- Analyzing Clusters in Training Data ---")
    analyze_clusters(embedded_train_2d, y_train)
    analyze_clusters(embedded_train_3d, y_train)