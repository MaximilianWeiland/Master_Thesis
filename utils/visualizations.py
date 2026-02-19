import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from typing import List

########################### Model Loss Development ###########################
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
        ax_f1.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.show()

########################### Entity Recognition Evaluation ###########################
def convert_dict_to_df(eval_results, focus_metrics):

    rows = []
    for model, eval_dict in eval_results.items():
        for focus_metric in focus_metrics:
            for metric, scores in eval_dict[focus_metric].items():
                rows.append(
                    {"model": model,
                     "main_metric": focus_metric,
                     "sub_metric": metric,
                     "mean": scores["mean"],
                     "lower": scores["lower"],
                     "upper": scores["upper"]}
                )
        df = pd.DataFrame(rows)
    return df

def within_metric_barplot(
        eval_results,
        focus_metrics,
        metric_title_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.tab10.colors,
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
    ):

    metrics_df = convert_dict_to_df(eval_results, focus_metrics)
    sub_metrics = metrics_df["sub_metric"].unique()
    models = metrics_df["model"].unique()

    x = np.arange(len(sub_metrics))
    width = group_width/len(models)
    
    _, ax = plt.subplots(figsize=(12, 8))
    
    for i, model in enumerate(models):
        subset = metrics_df[metrics_df["model"]==model]
        subset = subset.set_index('sub_metric').reindex(sub_metrics)
        means = subset['mean'].values
        yerr_lower = subset['mean'] - subset['lower']
        yerr_upper = subset['upper'] - subset['mean']
        yerr = np.array([yerr_lower, yerr_upper])

        ax.bar(x+i*width, means, width, yerr=yerr, capsize=5, label=model_names_remap[model], color=color_scheme[i%len(color_scheme)])
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels([m.capitalize() for m in sub_metrics])
    ax.set_ylabel('Score')
    ax.set_ylim(ylim[0], ylim[1])
    if printtitle:
        if task == "ner":
            ax.set_title(f'Model Performance on {metric_title_remap[focus_metrics[0]]} Metric')
        elif task == "stance":
            ax.set_title(f'Model Performance on {metric_title_remap[focus_metrics[0]]} Class')
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    if len(models) >= 2:
        ax.legend(title="Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()

def within_metric_dotplot(
        eval_results,
        focus_metrics,
        metric_title_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.tab10.colors,
        marker="o",
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
):

    metrics_df = convert_dict_to_df(eval_results, focus_metrics)
    sub_metrics = metrics_df["sub_metric"].unique()
    models = metrics_df["model"].unique()

    x = np.arange(len(sub_metrics))
    width = group_width/len(models)
    
    _, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        subset = metrics_df[metrics_df["model"]==model]
        subset = subset.set_index('sub_metric').reindex(sub_metrics)
        means = subset['mean'].values
        yerr_lower = subset['mean'] - subset['lower']
        yerr_upper = subset['upper'] - subset['mean']
        yerr = np.array([yerr_lower, yerr_upper])
        ax.errorbar(x+i*width, means, yerr=yerr, fmt=marker, markersize=6, capsize=5, color=color_scheme[i%len(color_scheme)], label=model_names_remap[model], linestyle="none")
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels([m.capitalize() for m in sub_metrics])
    ax.set_ylabel('Score')
    ax.set_ylim(ylim[0], ylim[1])
    if printtitle:
        if task == "ner":
            ax.set_title(f'Model Performance on {metric_title_remap[focus_metrics[0]]} Metric')
        elif task == "stance":
            ax.set_title(f'Model Performance on {metric_title_remap[focus_metrics[0]]} Class')
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    if len(models) >= 2:
        ax.legend(title="Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def multi_within_metric_barplot(
        eval_results,
        focus_metrics,
        metric_title_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.tab10.colors,
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
):

    metrics_df = convert_dict_to_df(eval_results, focus_metrics)
    sub_metrics = metrics_df["sub_metric"].unique()
    models = metrics_df["model"].unique()

    n_metrics = len(focus_metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12*ncols, 8*nrows))
    axes = axes.flatten()

    x = np.arange(len(sub_metrics))
    width = group_width / len(models)

    for idx, focus_metric in enumerate(focus_metrics):
        ax = axes[idx]
    
        for i, model in enumerate(models):
            subset = metrics_df[(metrics_df["model"]==model) & (metrics_df["main_metric"]==focus_metric)]
            subset = subset.set_index('sub_metric').reindex(sub_metrics)
            means = subset['mean'].values
            yerr_lower = subset['mean'] - subset['lower']
            yerr_upper = subset['upper'] - subset['mean']
            yerr = np.array([yerr_lower, yerr_upper])
            
            ax.bar(
                x + i*width,
                means,
                width,
                yerr=yerr,
                capsize=5,
                label=model_names_remap[model],
                color=color_scheme[i % len(color_scheme)]
            )

        ax.set_xticks(x + width*(len(models)-1)/2)
        ax.set_xticklabels([m.capitalize() for m in sub_metrics])
        ax.set_ylabel('Score')
        ax.set_ylim(ylim[0], ylim[1])
        if printtitle:
            if task == "ner":
                ax.set_title(f'Model Performance on {metric_title_remap[focus_metric]} Metric')
            elif task == "stance":
                ax.set_title(f'Model Performance on {metric_title_remap[focus_metric]} Class')
        ax.yaxis.grid(True, alpha=0.7)
        ax.xaxis.grid(False)

    # remove empty subplots if any
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # add only one legend for all subplots
    if len(models) >= 2:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model", loc='upper center', ncol=len(models))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def multi_within_metric_dotplot(
        eval_results,
        focus_metrics,
        metric_title_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.tab10.colors,
        marker="o",
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
):
    
    metrics_df = convert_dict_to_df(eval_results, focus_metrics)
    sub_metrics = metrics_df["sub_metric"].unique()
    models = metrics_df["model"].unique()

    n_metrics = len(focus_metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12*ncols, 8*nrows))
    axes = axes.flatten()

    x = np.arange(len(sub_metrics))
    width = group_width / len(models)

    for idx, focus_metric in enumerate(focus_metrics):
        ax = axes[idx]
    
        for i, model in enumerate(models):
            subset = metrics_df[(metrics_df["model"]==model) & (metrics_df["main_metric"]==focus_metric)]
            subset = subset.set_index('sub_metric').reindex(sub_metrics)
            means = subset['mean'].values
            yerr_lower = subset['mean'] - subset['lower']
            yerr_upper = subset['upper'] - subset['mean']
            yerr = np.array([yerr_lower, yerr_upper])
            ax.errorbar(x+i*width, means, yerr=yerr, fmt=marker, markersize=6, capsize=5, color=color_scheme[i%len(color_scheme)], label=model_names_remap[model], linestyle="none")

        ax.set_xticks(x + width*(len(models)-1)/2)
        ax.set_xticklabels([m.capitalize() for m in sub_metrics])
        ax.set_ylabel('Score')
        ax.set_ylim(ylim[0], ylim[1])
        if printtitle:
            if task == "ner":
                ax.set_title(f'Model Performance on {metric_title_remap[focus_metric]} Metric')
            elif task == "stance":
                ax.set_title(f'Model Performance on {metric_title_remap[focus_metric]} Class')
        ax.yaxis.grid(True, alpha=0.7)
        ax.xaxis.grid(False)

    # remove empty subplots if any
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # add only one legend for all subplots
    if len(models) >= 2:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model", loc='upper center', ncol=len(models))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def across_metric_barplot(
        eval_results,
        focus_metrics,
        comparison_metric,
        main_metrics_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.tab10.colors,
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
):

    metrics_df = convert_dict_to_df(eval_results=eval_results, focus_metrics=focus_metrics)
    metrics_df = metrics_df[metrics_df["sub_metric"]==comparison_metric]

    
    main_metrics = metrics_df["main_metric"].unique()
    models = metrics_df["model"].unique()

    x = np.arange(len(main_metrics))
    width = group_width/len(models)
    
    _, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        subset = metrics_df[metrics_df["model"]==model]
        subset = subset.set_index('main_metric').reindex(main_metrics)
        means = subset['mean'].values
        yerr_lower = subset['mean'] - subset['lower']
        yerr_upper = subset['upper'] - subset['mean']
        yerr = np.array([yerr_lower, yerr_upper])

        ax.bar(x+i*width, means, width, yerr=yerr, capsize=5, label=model_names_remap[model], color=color_scheme[i%len(color_scheme)])
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels([main_metrics_remap[m] for m in main_metrics])
    ax.set_ylabel(f'{comparison_metric.capitalize()} Score')
    ax.set_ylim(ylim[0], ylim[1])
    if printtitle:
        if task == "ner":
            ax.set_title(f'Model Performance on {comparison_metric.capitalize()} score Across Metrics')
        elif task == "stance":
            ax.set_title(f'Model Performance on {comparison_metric.capitalize()} score Across Metrics')
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    if len(models) >= 2:
        ax.legend(title="Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def across_metric_dotplot(
        eval_results,
        focus_metrics,
        comparison_metric,
        main_metrics_remap,
        model_names_remap,
        group_width=0.7,
        color_scheme=plt.cm.Set2.colors,
        marker="o",
        ylim=[0, 1],
        task="ner",
        printtitle=True,
        output_path=None
):

    metrics_df = convert_dict_to_df(eval_results=eval_results, focus_metrics=focus_metrics)
    metrics_df = metrics_df[metrics_df["sub_metric"]==comparison_metric]

    
    main_metrics = metrics_df["main_metric"].unique()
    models = metrics_df["model"].unique()

    x = np.arange(len(main_metrics))
    width = group_width/len(models)
    
    _, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        subset = metrics_df[metrics_df["model"]==model]
        subset = subset.set_index('main_metric').reindex(main_metrics)
        means = subset['mean'].values
        yerr_lower = subset['mean'] - subset['lower']
        yerr_upper = subset['upper'] - subset['mean']
        yerr = np.array([yerr_lower, yerr_upper])
        ax.errorbar(x+i*width, means, yerr=yerr, fmt=marker, markersize=6, capsize=5, color=color_scheme[i%len(color_scheme)], label=model_names_remap[model], linestyle="none")
    ax.set_xticks(x+width*(len(models)-1)/2)
    ax.set_xticklabels([main_metrics_remap[m] for m in main_metrics])
    ax.set_ylabel(f'{comparison_metric.capitalize()} Score')
    ax.set_ylim(ylim[0], ylim[1])
    if printtitle:
        if task == "ner":
            ax.set_title(f'Model Performance on {comparison_metric.capitalize()} score Across Metrics')
        elif task == "stance":
            ax.set_title(f'Model Performance on {comparison_metric.capitalize()} score Across Classes')
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    if len(models) >= 2:
        ax.legend(title="Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def few_shot_dev(
        eval_results,
        focus_metrics,
        comparison_metric,
        main_metrics_remap,
        model_names_remap,
        color_scheme=plt.cm.tab10.colors,
        marker="o",
        ylim=[0, 1],
        printtitle=True,
        output_path=None
):
    
    few_shot_keys = sorted(
        next(iter(eval_results.values())).keys(),
        key=lambda x: int(x.split("_")[0])
    )

    list_metrics_df = []
    for model, eval_dict in eval_results.items():
        for fs_key in few_shot_keys:
            sub_dict = {model: eval_dict[fs_key]}
            sub_df = convert_dict_to_df(sub_dict, focus_metrics)
            sub_df = sub_df[sub_df["sub_metric"] == comparison_metric]
            sub_df["few_shot_examples"] = int(fs_key.split("_")[0])
            list_metrics_df.append(sub_df)
    metrics_df = pd.concat(list_metrics_df, ignore_index=True)

    fs_examples = metrics_df["few_shot_examples"].unique()
    models = metrics_df["model"].unique()

    _, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        subset = metrics_df[metrics_df["model"] == model]
        means = subset['mean'].values
        yerr_lower = subset['mean'] - subset['lower']
        yerr_upper = subset['upper'] - subset['mean']
        yerr = np.array([yerr_lower, yerr_upper])
        ax.errorbar(fs_examples, means, yerr=yerr, linestyle="-", fmt=marker, markersize=6, capsize=5, color=color_scheme[i%len(color_scheme)], label=model_names_remap[model])
        
    ax.set_xticks(fs_examples)
    ax.set_xlabel("Number of Few-Shot Examples")
    ax.set_ylabel(f"{main_metrics_remap[focus_metrics[0]]} {comparison_metric.capitalize()} Score")
    ax.set_ylim(ylim[0], ylim[1])
    if printtitle:
        ax.set_title(f"{main_metrics_remap[focus_metrics[0]]} {comparison_metric.capitalize()} Score vs. Number of Few-Shot Examples")
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)
    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def multi_few_shot_dev(
        eval_results,
        focus_metrics,
        comparison_metric,
        main_metrics_remap,
        model_names_remap,
        color_scheme=plt.cm.tab10.colors,
        marker="o",
        ylim=[0, 1],
        printtitle=True,
        output_path=None
):
    few_shot_keys = sorted(
        next(iter(eval_results.values())).keys(),
        key=lambda x: int(x.split("_")[0])
    )

    list_metrics_df = []
    for model, eval_dict in eval_results.items():
        for fs_key in few_shot_keys:
            sub_dict = {model: eval_dict[fs_key]}
            sub_df = convert_dict_to_df(sub_dict, focus_metrics)
            sub_df = sub_df[sub_df["sub_metric"] == comparison_metric]
            sub_df["few_shot_examples"] = int(fs_key.split("_")[0])
            list_metrics_df.append(sub_df)
    metrics_df = pd.concat(list_metrics_df, ignore_index=True)

    fs_examples = metrics_df["few_shot_examples"].unique()
    models = metrics_df["model"].unique()

    n_metrics = len(focus_metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12*ncols, 8*nrows))
    axes = axes.flatten()
    
    for idx, focus_metric in enumerate(focus_metrics):
        ax = axes[idx]

        for i, model in enumerate(models):
            subset = metrics_df[(metrics_df["model"]==model) & (metrics_df["main_metric"]==focus_metric)]
            means = subset['mean'].values
            yerr_lower = subset['mean'] - subset['lower']
            yerr_upper = subset['upper'] - subset['mean']
            yerr = np.array([yerr_lower, yerr_upper])
            ax.errorbar(fs_examples, means, yerr=yerr, linestyle="-", fmt=marker, markersize=6, capsize=5, color=color_scheme[i%len(color_scheme)], label=model_names_remap[model])
        
        ax.set_xticks(fs_examples)
        ax.set_xlabel("Number of Few-Shot Examples")
        ax.set_ylabel(f"{main_metrics_remap[focus_metric]} {comparison_metric.capitalize()} Score")
        ax.set_ylim(ylim[0], ylim[1])
        if printtitle:
            ax.set_title(f"{main_metrics_remap[focus_metric]} {comparison_metric.capitalize()} Score vs. Number of Few-Shot Examples")
        ax.yaxis.grid(True, alpha=0.7)
        ax.xaxis.grid(False)
       
    # remove empty subplots if any
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # add only one legend for all subplots
    if len(models) >= 2:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model", loc='upper center', ncol=len(models))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

########################### Clustering Evaluation and Interpretation ###########################
    
def plot_embeddings_grid(
    reducer,
    embeddings,
    category_list,
    cat2id,
    reducer_name,
    classes_per_plot=10,
    ncols=2,
    color_scheme=plt.cm.tab10.colors,
    output_path=None
):

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
        for i, c in enumerate(group):
            idx = np.where(dictionary_labels_np == c)[0]
            ax.scatter(
                emb_2d[idx, 0],
                emb_2d[idx, 1],
                label=categories[c],
                alpha=0.7,
                s=40,
                color=color_scheme[i%len(color_scheme)]
            )

        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
        ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        ax.set_xlabel(f"{reducer_name}-1")
        ax.set_ylabel(f"{reducer_name}-2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.7)

    for ax in axes[nplots:]:
        ax.axis("off")

    # fig.suptitle(
    #     f"{reducer_name} 2D-Representation of Social Group Embeddings",
    #     y=1.02
    # )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_embeddings_two_groups(
        embeddings1,
        embeddings2,
        label1,
        label2,
        reducer,
        reducer_name,
        color_scheme=plt.cm.tab10.colors,
        output_path=None
):

    # stack to a numpy array
    X = np.vstack([
        embeddings1,
        embeddings2
    ])

    labels = np.array(
        [label1] * len(embeddings1)
        + [label2] * len(embeddings2)
    )

    # reduce dimension
    emb_2d = reducer.fit_transform(X)

    # plot all embeddings colored by cluster group
    plt.figure(figsize=(12, 8))

    for i, group in enumerate([label1, label2]):
        idx = labels == group
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            label=group,
            color=color_scheme[i%len(color_scheme)],
            alpha=0.7,
            s=40
        )

    if reducer_name == "PCA":
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
    else:
        plt.xlabel(f"{reducer_name} 1")
        plt.ylabel(f"{reducer_name} 2")
    plt.grid(True, alpha=0.7)
    plt.legend()
    # plt.title(f"{reducer_name} Reduced Embeddings")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def clustering_dev_single_metric(
        ks: List[int],
        metrics_list: List[float],
        metric_name: str,
        min_max: str,
        preknown_best_k: int = None,
        color_scheme = plt.cm.tab10.colors,
        output_path=None
):
    """
    
    """
    metrics_array = np.array(metrics_list)

    if min_max == "max":
        best_idx = metrics_array.argmax()
    elif min_max == "min":
        best_idx = metrics_array.argmin()
    
    best_k = ks[best_idx]
    best_score = metrics_list[best_idx]

    # plot development of silhouette scores
    plt.figure(figsize=(12, 8))
    plt.plot(ks, metrics_list, color=color_scheme[0])
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(f"{metric_name.capitalize()} score")
    # plt.title(f"{metric_name.capitalize()} score across k")
    plt.grid(True, alpha=0.7)
    plt.ylim(0, 1)

    if preknown_best_k:
        if preknown_best_k == best_k:
            plt.axvline(preknown_best_k, linestyle="--", color=color_scheme[1], label=f"Optimal k ({metric_name} score and original categories)")
            plt.legend()
        else:
            plt.axvline(best_k, linestyle="--", color=color_scheme[1], label=f"Optimal k ({metric_name} score)")
            plt.axvline(preknown_best_k, linestyle="--", color=color_scheme[2], label="Original number of categories")
            plt.legend()
    else:
        plt.axvline(best_k, linestyle="--", color=color_scheme[1], label=f"Optimal k ({metric_name} score)")
        plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return best_k, best_score

def clustering_dev_combined_metrics(
        ks: List[int],
        silscore_list: List[float],
        nmiscore_list: List[float],
        preknown_best_k: int = None,
        color_scheme = plt.cm.tab10.colors,
        output_path=None
):
    """
    
    """
    silscore_array = np.array(silscore_list)
    nmi_array = np.array(nmiscore_list)

    best_idx_silscore = silscore_array.argmax()
    best_idx_nmi = nmi_array.argmax()
    
    best_k_silscore = ks[best_idx_silscore]
    best_score_silscore = silscore_list[best_idx_silscore]
    best_k_nmi = ks[best_idx_nmi]
    best_score_nmi = nmiscore_list[best_idx_nmi]

    # plot development of silhouette scores
    plt.figure(figsize=(12, 8))
    plt.plot(ks, silscore_list, color=color_scheme[0], label="Silhouette scores")
    plt.plot(ks, nmiscore_list, color=color_scheme[1], label="NMI scores")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(f"Score")
    # plt.title(f"{metric_name.capitalize()} score across k")
    plt.grid(True, alpha=0.7)
    plt.ylim(0, 1)

    if preknown_best_k:
        if preknown_best_k == best_k_silscore == best_k_nmi:
            plt.axvline(preknown_best_k, linestyle="--", color=color_scheme[2], label=f"Optimal k (metrics and original categories)")
            plt.legend()
        else:
            plt.axvline(best_k_silscore, linestyle="--", color=color_scheme[2], label=f"Optimal k (Silhouette score)")
            plt.axvline(best_k_nmi, linestyle="--", color=color_scheme[3], label=f"Optimal k (NMI score)")
            plt.axvline(preknown_best_k, linestyle="--", color=color_scheme[4], label="Original number of categories")
            plt.legend()
    else:
        plt.axvline(best_k_silscore, linestyle="--", color=color_scheme[2], label=f"Optimal k (Silhouette score)")
        plt.axvline(best_k_nmi, linestyle="--", color=color_scheme[3], label=f"Optimal k (NMI score)")
        plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return best_k_silscore, best_score_silscore, best_k_nmi, best_score_nmi

def static_mentions_boxplot(
    df,
    cluster_ids,
    cluster_col="cluster",
    distance_type="Euclidean",
    jitter=0.05,
    color_scheme=plt.cm.tab10.colors,
    output_path=None
):
    
    if distance_type == "Euclidean":
        distance_col = "euclidean_distance_to_centroid"
    elif distance_type == "Cosine":
        distance_col = "cosine_distance_to_centroid"

    distances_per_cluster = [df[df[cluster_col] == cid][distance_col].values for cid in cluster_ids]

    _, ax = plt.subplots(figsize=(12, 8))

    ax.boxplot(distances_per_cluster, showfliers=False, widths=.4)

    for i, distances in enumerate(distances_per_cluster, start=1):
        x = np.random.uniform(i - jitter, i + jitter, size=len(distances))
        ax.scatter(x, distances, s=10, alpha=0.6, color=color_scheme[i-1 % len(color_scheme)])

    ax.set_xticks(range(1, len(cluster_ids) + 1))
    ax.set_xticklabels([f"Cluster {cluster_id}" for cluster_id in cluster_ids])
    ax.set_ylabel(f"{distance_type} distance to centroid")
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def interactive_mentions_boxplot(
    cluster_df,
    clusters,
    cluster_col='cluster',
    distance_type='Euclidean',
    mention_col='mention'
) -> None:
    
    if distance_type == "Euclidean":
        distance_col = "euclidean_distance_to_centroid"
    elif distance_type == "Cosine":
        distance_col = "cosine_distance_to_centroid"

    cluster_df = cluster_df[cluster_df[cluster_col].isin(clusters)]
    cluster_ids = sorted(cluster_df[cluster_col].unique())
    
    fig = go.Figure(layout=dict(width=800, height=600))
    
    for i, cid in enumerate(cluster_ids):
        cluster_data = cluster_df[cluster_df[cluster_col] == cid]
        y_values = cluster_data[distance_col].values
        hover_texts = cluster_data[mention_col].values
        
        fig.add_trace(go.Box(
            y=y_values,
            name=f'Cluster {cid}',
            boxpoints='all',
            jitter=0.3,
            whiskerwidth=0.2,
            marker=dict(size=3),
            line=dict(width=1),
            text=hover_texts,
            hovertemplate='%{text}<br>%{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Clustered Social Group Mentions with {distance_type} Distance to Centroid",
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
            title=f"{distance_type} distance to centroid"
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        margin=dict(l=40, r=30, t=80, b=100),
        showlegend=False
    )
    
    fig.show()