# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:11:06 2026

@author: bingbing & baobao
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def cluster_spatial_structures(
    adata,
    embeddings_typical_scales,
    res,
    n_clusters=15,
    random_state = 42,
    cluster_col_prefix="Spatial_Structure_cluster_scale_",
    copy=False,
    verbose=True
):
    """
    Perform KMeans clustering on cell embeddings at each typical scale and write
    the resulting cluster labels back into ``adata.obs``.

    Workflow
    --------
    1. Collect cell embeddings for each typical scale across all samples.
    2. Perform KMeans clustering independently at each scale.
    3. Save cluster labels to ``adata.obs`` with one column per scale.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to which clustering labels will be added.
    embeddings_typical_scales : dict
        Output from ``compute_typical_scales``.
        Dictionary mapping sample IDs to a list of embedding DataFrames.
    res : dict
        Output result dictionary from ``compute_typical_scales``.
        Must contain ``res['center_scales']``.
    n_clusters : int or dict, default=15
        Number of KMeans clusters.
        - If int, the same number is used for all scales.
        - If dict, keys should be scale values and values should be cluster numbers.
          Example: ``{8: 15, 19: 15, 44: 15, 125: 15}``
    random_state : int, default=42
        Random seed used for KMeans clustering to ensure reproducibility of clustering results.
    cluster_col_prefix : str, default='Spatial_Structure_cluster_scale_'
        Prefix used to create output column names in ``adata.obs``.
        Final names will look like:
        ``Spatial_Structure_cluster_scale_8``
    copy : bool, default=False
        If True, return a copy of ``adata`` with clustering results.
        If False, modify ``adata`` in place.
    verbose : bool, default=True
        Whether to print clustering progress.

    Returns
    -------
    adata_out : anndata.AnnData
        AnnData object with clustering columns added to ``obs``.
    cluster_labels_by_scale : dict
        Dictionary mapping each typical scale to a NumPy array of cluster labels.
    embeddings_by_scale : dict
        Dictionary mapping each typical scale to the stacked embedding matrix used for clustering.
    cell_ids_by_scale : dict
        Dictionary mapping each typical scale to the corresponding cell IDs.

    Examples
    --------
    >>> adata, cluster_labels_by_scale, embeddings_by_scale, cell_ids_by_scale = \
    ...     cluster_spatial_structures(
    ...         adata=adata,
    ...         embeddings_typical_scales=embeddings_typical_scales,
    ...         res=res,
    ...         n_clusters={8: 15, 19: 15, 44: 15, 125: 15},
    ...         cluster_col_prefix='Spatial_Structure_cluster_scale_'
    ...     )
    >>>
    >>> print(adata.obs.columns)
    >>> print(cluster_labels_by_scale.keys())
    """
    adata_out = adata.copy() if copy else adata

    typical_scales = res["center_scales"]
    scale_to_index = {s: i for i, s in enumerate(typical_scales)}

    embeddings_by_scale = {s: [] for s in typical_scales}
    cell_ids_by_scale = {s: [] for s in typical_scales}

    for sid, per_scale_list in embeddings_typical_scales.items():
        for s in typical_scales:
            if s not in scale_to_index:
                if verbose:
                    print(f"Skip scale {s}, not in center scales")
                continue
            i_scale = scale_to_index[s]
            emb = per_scale_list[i_scale]
            embeddings_by_scale[s].append(emb.values)
            cell_ids_by_scale[s].extend(emb.index.tolist())

    for s in typical_scales:
        embeddings_by_scale[s] = np.vstack(embeddings_by_scale[s])
        if verbose:
            print(f"Scale {s}: {embeddings_by_scale[s].shape[0]} cells × {embeddings_by_scale[s].shape[1]} dims")

    cluster_labels_by_scale = {}

    for s in typical_scales:
        X = embeddings_by_scale[s]

        if isinstance(n_clusters, dict):
            n_clusters_scale = n_clusters.get(s, 15)
        else:
            n_clusters_scale = int(n_clusters)

        if verbose:
            print(f"Perform KMeans clustering on scale {s} (n_clusters={n_clusters_scale}) ...")

        km = KMeans(n_clusters=n_clusters_scale, random_state=random_state, n_init="auto").fit(X)
        cluster_labels_by_scale[s] = km.labels_

        ids = cell_ids_by_scale[s]
        ser = pd.Series(km.labels_, index=ids, name=f"{cluster_col_prefix}{s}")
        adata_out.obs = adata_out.obs.join(ser, how="left")

    return adata_out, cluster_labels_by_scale, embeddings_by_scale, cell_ids_by_scale


def plot_structure_celltype_heatmap(
    adata,
    cluster_column='Spatial_Structure_cluster_scale_8',
    cell_type_column='detailed_anno',
    figsize=(10, 8),
    cmap='bwr',
    vmin=-3,
    vmax=3,
    output_dir=None,
    pdf_name='Spatial_Structure_cluster_scale_8_anno_heatmap.pdf',
    xlabel='Cell Types',
    ylabel='Spatial Structures',
    xtick_fontsize=20,
    ytick_fontsize=20,
    xlabel_fontsize=20,
    ylabel_fontsize=20,
    legend_fontsize=20,
    font_family='Arial',
    x_labelpad=15,
    y_labelpad=15
):
    """
    Visualize cell type composition across spatial structure clusters as a clustered heatmap.

    This function summarizes the relative abundance of cell types within each
    cluster and then standardizes the proportions across clusters for visualization.

    Workflow
    --------
    1. Build a cluster × cell type contingency table.
    2. Convert counts to within-cluster proportions.
    3. Standardize proportions using z-score transformation.
    4. Plot a clustered heatmap using seaborn.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing clustering results and cell type annotations.
    cluster_column : str, default='Spatial_Structure_cluster_scale_8'
        Column in ``adata.obs`` storing cluster labels.
    cell_type_column : str, default='detailed_anno'
        Column in ``adata.obs`` storing cell type annotations.
    figsize : tuple, default=(10, 8)
        Figure size passed to ``seaborn.clustermap``.
    cmap : str, default='bwr'
        Colormap used for the heatmap.
    vmin : float, default=-3
        Minimum value for color scaling.
    vmax : float, default=3
        Maximum value for color scaling.
    output_dir : str or None, default=None
        Directory to save the output PDF. If None, figure is not saved.
    pdf_name : str, default='Spatial_Structure_cluster_scale_8_anno_heatmap.pdf'
        Output file name used when ``output_dir`` is provided.
    xlabel : str, default='Cell Types'
        X-axis label.
    ylabel : str, default='Spatial Structures'
        Y-axis label.
    xtick_fontsize : int, default=20
        Font size of x-axis tick labels.
    ytick_fontsize : int, default=20
        Font size of y-axis tick labels.
    xlabel_fontsize : int, default=20
        Font size of x-axis label.
    ylabel_fontsize : int, default=20
        Font size of y-axis label.
    legend_fontsize : int, default=20
        Font size of colorbar ticks.
    font_family : str, default='Arial'
        Font family used in the plot.
    x_labelpad : int, default=15
        Padding between x-axis label and axis.
    y_labelpad : int, default=15
        Padding between y-axis label and axis.

    Returns
    -------
    g : seaborn.matrix.ClusterGrid
        Seaborn ClusterGrid object for further customization if needed.

    Examples
    --------
    >>> plot_structure_celltype_heatmap(
    ...     adata,
    ...     cluster_column='Spatial_Structure_cluster_scale_8',
    ...     cell_type_column='detailed_anno',
    ...     figsize=(10, 8.5),
    ...     cmap='RdBu_r',
    ...     output_dir='./scale_diagnostics_GAT_shared',
    ...     pdf_name='Spatial_Structure_cluster_scale_8_anno_heatmap.pdf'
    ... )
    """
    rcParams['font.family'] = font_family

    adata.obs[cluster_column] = adata.obs[cluster_column].astype('category')
    adata.obs[cell_type_column] = adata.obs[cell_type_column].astype('category')

    df = adata.obs[[cluster_column, cell_type_column]]

    count_table = pd.crosstab(df[cluster_column], df[cell_type_column])
    proportion_table = count_table.div(count_table.sum(axis=1), axis=0)

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(proportion_table),
        index=proportion_table.index,
        columns=proportion_table.columns
    )

    g = sns.clustermap(
        scaled,
        cmap=cmap,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        col_cluster=True,
        row_cluster=True,
        linewidths=0.5,
        annot=False,
        xticklabels=True,
        yticklabels=True
    )

    g.ax_heatmap.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontname=font_family)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontname=font_family)
    g.ax_heatmap.xaxis.labelpad = x_labelpad
    g.ax_heatmap.yaxis.labelpad = y_labelpad
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=xtick_fontsize, rotation=90, fontname=font_family)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=ytick_fontsize, rotation=0, fontname=font_family)
    g.cax.tick_params(labelsize=legend_fontsize)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        g.figure.savefig(f"{output_dir}/{pdf_name}", dpi=300, bbox_inches='tight')

    plt.show()

    if output_dir:
        print(f"Heatmap saved to {output_dir}/{pdf_name}")

    return g


def compute_cluster_shannon_diversity(
    adata,
    cluster_cols,
    celltype_col='detailed_anno',
    output_dir=None,
    csv_name='shannon_index_per_cluster_anno.csv',
    plot=True,
    figsize=(8, 5),
    box_color='skyblue',
    strip_color='black'
):
    """
    Compute the Shannon diversity index of cell type composition for each cluster
    across one or multiple spatial scales.

    For each cluster column, the function:
    1. groups cells by cluster
    2. computes cell type proportions within each cluster
    3. calculates Shannon diversity index
    4. optionally visualizes the distribution across scales
    5. optionally saves the results to CSV

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing clustering columns and cell type annotations.
    cluster_cols : dict
        Dictionary mapping scale values to clustering column names.
        Example:
        ``{8: 'Spatial_Structure_cluster_scale_8',
           19: 'Spatial_Structure_cluster_scale_19'}``
    celltype_col : str, default='detailed_anno'
        Column in ``adata.obs`` storing cell type annotations.
    output_dir : str or None, default=None
        Directory to save the CSV result file.
    csv_name : str, default='shannon_index_per_cluster_anno.csv'
        Output CSV filename.
    plot : bool, default=True
        Whether to draw a boxplot + stripplot of Shannon index by scale.
    figsize : tuple, default=(8, 5)
        Figure size for the plot.
    box_color : str, default='skyblue'
        Boxplot fill color.
    strip_color : str, default='black'
        Stripplot point color.

    Returns
    -------
    df_shannon : pandas.DataFrame
        DataFrame with columns:
        - ``scale``
        - ``cluster``
        - ``n_cells``
        - ``shannon_index``

    Examples
    --------
    >>> cluster_cols = {
    ...     8: 'Spatial_Structure_cluster_scale_8',
    ...     19: 'Spatial_Structure_cluster_scale_19',
    ...     44: 'Spatial_Structure_cluster_scale_44',
    ...     125: 'Spatial_Structure_cluster_scale_125'
    ... }
    >>>
    >>> df_shannon = compute_cluster_shannon_diversity(
    ...     adata=adata,
    ...     cluster_cols=cluster_cols,
    ...     celltype_col='detailed_anno',
    ...     output_dir='./scale_diagnostics_GAT_shared',
    ...     csv_name='shannon_index_per_cluster_anno.csv',
    ...     plot=True
    ... )
    """
    results = []

    for scale, clust_col in cluster_cols.items():
        if clust_col not in adata.obs.columns:
            print(f"Skip scale {scale}, column {clust_col} not exist")
            continue

        df = adata.obs[[clust_col, celltype_col]].dropna()
        groups = df.groupby(clust_col)

        for clust, sub in groups:
            counts = sub[celltype_col].value_counts(normalize=True)
            H = entropy(counts, base=np.e)
            results.append({
                "scale": scale,
                "cluster": clust,
                "n_cells": len(sub),
                "shannon_index": H
            })

    df_shannon = pd.DataFrame(results)
    df_shannon.sort_values(["scale", "cluster"], inplace=True)

    if plot and not df_shannon.empty:
        plt.figure(figsize=figsize)
        sns.boxplot(data=df_shannon, x="scale", y="shannon_index", color=box_color)
        sns.stripplot(
            data=df_shannon,
            x="scale",
            y="shannon_index",
            color=strip_color,
            alpha=0.5,
            jitter=True,
            size=3
        )
        plt.title("Shannon Diversity Index per Cluster (across Scales)")
        plt.ylabel("Shannon Index (H)")
        plt.xlabel("Scale (μm)")
        plt.show()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_shannon.to_csv(f"{output_dir}/{csv_name}", index=False)

    return df_shannon





