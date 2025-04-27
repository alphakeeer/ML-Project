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


def Visualization_2D(data):
    """
    Visualize high-dimensional data in 2D using t-SNE.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        High-dimensional data to be visualized.
    
    Returns: tsne_results : array-like, shape (n_samples, 2)
        2D t-SNE embedding of the data.
    --------
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric='euclidean', learning_rate=200)
    tsne_results = tsne.fit_transform(data)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=50, alpha=0.7)
    plt.title("t-SNE Visualization of High-Dimensional Data")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid()
    plt.show()

def Visualization_3D(data):
    """
    Visualize high-dimensional data in 3D using t-SNE.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        High-dimensional data to be visualized.
    
    Returns: tsne_results : array-like, shape (n_samples, 3)
        3D t-SNE embedding of the data.
    --------
    """
    # Perform t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000, metric='euclidean', learning_rate=200)
    tsne_results = tsne.fit_transform(data)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], s=50, alpha=0.7)
    ax.set_title("t-SNE Visualization of High-Dimensional Data")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    plt.isinteractive()
    plt.show()

if __name__ == "__main__":

    # Laod the data
    data = pd.read_csv("./data/adult_train_processed.csv")
    # Extract features and labels
    features = data.drop(columns=['income'])
    labels = data['income']
    # use features only for visualization
    Visualization_2D(features)
    Visualization_3D(features)



