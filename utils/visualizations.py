import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import math
from typing import List

def plot_losses_f1(model_names, optimal_configurations):
    # as many rows as models and 2 columns for loss and f1
    _, axes = plt.subplots(len(model_names), 2, figsize=(12, 4 * len(model_names)))

    # loop over the models
    for i, model_name in enumerate(model_names):

        # get the data for this specific model
        config = optimal_configurations[model_name]
        epochs = list(range(1, len(config['train_losses']) + 1))

        # create the loss subplot
        ax_loss = axes[i, 0]
        ax_loss.plot(epochs, config['train_losses'], marker='o', label='Train Loss')
        ax_loss.plot(epochs, config['val_losses'], marker='o', label='Validation Loss')
        ax_loss.set_title(f"{model_name} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # create the F1 subplot
        ax_f1 = axes[i, 1]
        ax_f1.plot(epochs, config['f1_scores_train'], marker='o', label='Train F1')
        ax_f1.plot(epochs, config['f1_scores_val'], marker='o', label='Validation F1')
        ax_f1.set_title(f"{model_name} - F1 Score")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("F1 Score")
        ax_f1.legend()
        ax_f1.grid(True)

    plt.tight_layout()
    plt.show()

################################### Visualizations for Clustering ###################################
    
def plot_2d_embeddings_grid(reducer, embeddings, category_list, cat2id, reducer_name, classes_per_plot=10, ncols=2):

    emb_2d = reducer.fit_transform(embeddings)

    # get axis ranges
    x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
    y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
    pad = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min

    # get all unique categories as a list and the total number of categories
    categories = list(cat2id.keys())
    num_classes = len(categories)

    dictionary_labels_np = np.array([cat2id[cat] for cat in category_list])

    # split categories into groups
    groups = [
        range(i, min(i + classes_per_plot, num_classes))
        for i in range(0, num_classes, classes_per_plot)
    ]

    # dynamic grid size
    nplots = len(groups)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    for ax, group in zip(axes, groups):
        for c in group:
            idx = np.where(dictionary_labels_np == c)[0]
            ax.scatter(
                emb_2d[idx, 0],
                emb_2d[idx, 1],
                label=categories[c],
                alpha=0.7,
                s=40
            )

        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
        ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        ax.set_xlabel(f"{reducer_name}-1")
        ax.set_ylabel(f"{reducer_name}-2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    for ax in axes[nplots:]:
        ax.axis("off")

    fig.suptitle(
        f"{reducer_name} 2D-Representation of Social Group Embeddings",
        y=1.02
    )
    plt.tight_layout()
    plt.show()

def plot_clustering_dev(
        ks: List[int],
        metrics_list: List[float],
        metric_name: str,
        min_max: str,
        preknown_best_k: int = None
) -> None:
    metrics_array = np.array(metrics_list)

    if min_max == "max":
        best_idx = metrics_array.argmax()
    elif min_max == "min":
        best_idx = metrics_array.argmin()
    
    best_k = ks[best_idx]
    best_score = metrics_list[best_idx]

    # plot development of silhouette scores
    plt.figure(figsize=(12, 8))
    plt.plot(ks, metrics_list)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(f"{metric_name} score")
    plt.title(f"{metric_name} score across k")
    plt.grid(True)
    plt.axvline(best_k, color="blue", label=f"Best k based on {metric_name}")
    if preknown_best_k:
        plt.axvline(preknown_best_k, color="green", label="Original number of categories")
        plt.legend()
    plt.annotate(
        f"k={best_k}, score={best_score:.4f}",
        xy=(best_k, best_score),
        xytext=(best_k + 1, best_score),
    )
    plt.show()

    return best_k, best_score

def static_mentions_boxplot(cluster_df, cluster_id, cluster_col="cluster", top_n=5, point_jitter=0.03, vertical_spacing=0.03):
    """
    Boxplot with mentions displayed in fixed vertical bands relative to the figure:
    - closest (green) at bottom
    - median (orange) in middle
    - farthest (red) at top
    Data points are jittered horizontally and plotted at their real distances.
    """
    cluster_data = cluster_df[cluster_df[cluster_col] == cluster_id]
    distances = cluster_data["distance_to_centroid"].values
    mentions = cluster_data["mention"].values

    # Sort indices
    sorted_idx = np.argsort(distances)
    closest_idx = sorted_idx[:top_n]
    median_idx = sorted_idx[len(sorted_idx)//2 - top_n//2 : len(sorted_idx)//2 + top_n//2 + 1]
    farthest_idx = sorted_idx[-top_n:]

    categories = {
        'closest': (closest_idx, 'green'),
        'median': (median_idx, 'orange'),
        'farthest': (farthest_idx, 'red')
    }

    fig, ax = plt.subplots(figsize=(8,6))
    x_base = 1  # base for boxplot and data points

    ax.boxplot(distances, vert=True, showfliers=False)

    jittered_x = x_base + np.random.uniform(-point_jitter, point_jitter, size=len(distances))
    ax.scatter(jittered_x, distances, color='blue', alpha=0.6)

    y_min, y_max = distances.min(), distances.max()
    band_positions = {
        'closest': 0.25,   # bottom
        'median': 0.45,    # middle above boxplot
        'farthest': 0.85   # top
    }

    for cat_name, (idxs, color) in categories.items():
        y_start = band_positions[cat_name]
        for i, idx in enumerate(idxs):
            y_pos = y_start - i*vertical_spacing
            ax.text(1.1, y_pos, mentions[idx], fontsize=8, color=color,
                    verticalalignment='top', horizontalalignment='left')

    ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cosine distance to centroid")
    ax.set_title(f"Cluster {cluster_id} centroid distances")
    plt.show()

    closest = mentions[closest_idx]
    median = mentions[median_idx]
    farthest = mentions[farthest_idx]
    return closest, median, farthest

def interactive_mentions_boxplot(cluster_df, clusters, cluster_col='cluster', distance_col='distance_to_centroid', mention_col='mention'):
    cluster_df = cluster_df[cluster_df[cluster_col].isin(clusters)]
    cluster_ids = sorted(cluster_df[cluster_col].unique())
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    
    # repeat colors if more clusters than colors
    colors = (colors * ((len(cluster_ids) // len(colors)) + 1))[:len(cluster_ids)]
    
    fig = go.Figure(layout=dict(width=800, height=600))
    
    for cid, color in zip(cluster_ids, colors):
        cluster_data = cluster_df[cluster_df[cluster_col] == cid]
        y_values = cluster_data[distance_col].values
        hover_texts = cluster_data[mention_col].values
        
        fig.add_trace(go.Box(
            y=y_values,
            name=f'Cluster {cid}',
            boxpoints='all',
            jitter=0.3,
            whiskerwidth=0.2,
            fillcolor=color,
            marker_size=3,
            line_width=1,
            text=hover_texts,
            hoverinfo='text+y'
        ))
    
    fig.update_layout(
        title=dict(
            text="Clustered Social Group Mentions with Cosine Distance to Centroid",
            x=0.5
        ),
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
            title="Cosine distance to centroid"
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        margin=dict(l=40, r=30, t=80, b=100),
        showlegend=False
    )
    
    fig.show()

