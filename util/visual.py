import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_umap(df, path, sz=10):
    """
    Plot features with umap and save figure

    dataframe is expected as [img_id, class_id, *features...]
    """
    embedding = umap.UMAP().fit_transform(df.iloc[:, 2:])
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], 
                c=df.iloc[:, 1], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Paired',
                s=sz)
    ax.axis('off')
    fig.savefig(f'{path}')
    plt.close(fig)

def plot_tsne(df, path, sz=10):
    """
    Plot features with tsne and save figure

    dataframe is expected as [img_id, class_id, *features...]
    """
    tsne = TSNE(n_components=2, random_state=10).fit_transform(df.iloc[:, 2:])
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.scatter(tsne[:, 0], tsne[:, 1], 
                c=df.iloc[:, 1], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Paired',
                s=sz)
    ax.axis('off')
    fig.savefig(f'{path}')
    plt.close(fig)