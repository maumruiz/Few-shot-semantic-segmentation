import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_umap(df, path, sz=10):
    """
    Plot features with umap and save figure

    dataframe is expected as [img_id, class_id, *features...]
    """
    embedding = umap.UMAP().fit_transform(df.iloc[:, 2:])
    plt.figure(figsize=(12,12))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=df.iloc[:, 1], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Paired',
                s=sz)
    plt.axis('off')
    plt.savefig(f'{path}')
    plt.clf()

def plot_tsne(df, path, sz=10):
    """
    Plot features with tsne and save figure

    dataframe is expected as [img_id, class_id, *features...]
    """
    tsne = TSNE(n_components=2, random_state=10).fit_transform(df.iloc[:, 2:])
    plt.clf()
    plt.figure(figsize=(12,12))
    plt.scatter(tsne[:, 0], tsne[:, 1], 
                c=df.iloc[:, 1], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Paired',
                s=sz)
    plt.axis('off')
    plt.savefig(f'{path}')
    plt.clf()