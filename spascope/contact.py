# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:58:15 2026

@author: bingbing & baobao
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import binary_fill_holes


def compute_boundaries_and_interactions(raster, background=-1, pixel_size=1.0):
    """
    Compute per-label outer boundary lengths and pairwise shared boundaries from a rasterized label map.

    This function is the geometric basis for SpaScope contact-score analysis.
    Given a 2D raster of cluster labels, it calculates:

    1. the total outer boundary length for each label
    2. the shared boundary matrix between different labels

    Parameters
    ----------
    raster : numpy.ndarray
        2D array containing rasterized cluster labels.
    background : int, default=-1
        Background pixel value to exclude from calculations.
    pixel_size : float, default=1.0
        Physical size of one pixel edge. Boundary lengths are multiplied by this value.

    Returns
    -------
    boundary_length_dict : dict
        Dictionary mapping each label to its total outer boundary length.
    shared_boundary : numpy.ndarray
        Square matrix where entry ``(i, j)`` represents the shared boundary count
        between label ``i`` and label ``j``.

    Examples
    --------
    >>> boundary_dict, shared_matrix = compute_boundaries_and_interactions(
    ...     raster=raster,
    ...     background=-1,
    ...     pixel_size=1.0
    ... )
    >>> print(boundary_dict)
    >>> print(shared_matrix.shape)
    """
    labels = np.unique(raster)
    labels = labels[labels != background]
    n_labels = labels.max() + 1
    h, w = raster.shape

    shared_boundary = np.zeros((n_labels, n_labels), dtype=float)
    boundary_length_dict = {lab: 0.0 for lab in labels}

    for lab in labels:
        mask = (raster == lab)
        filled = binary_fill_holes(mask)
        labeled = measure.label(filled, connectivity=1)

        for region_idx in np.unique(labeled):
            if region_idx == 0:
                continue
            region_mask = (labeled == region_idx)
            ys, xs = np.where(region_mask)

            edges = set()
            for y_pix, x_pix in zip(ys, xs):
                if y_pix == 0 or not region_mask[y_pix-1, x_pix]:
                    edges.add(((x_pix, y_pix), (x_pix+1, y_pix)))
                if y_pix == region_mask.shape[0]-1 or not region_mask[y_pix+1, x_pix]:
                    edges.add(((x_pix, y_pix+1), (x_pix+1, y_pix+1)))
                if x_pix == 0 or not region_mask[y_pix, x_pix-1]:
                    edges.add(((x_pix, y_pix), (x_pix, y_pix+1)))
                if x_pix == region_mask.shape[1]-1 or not region_mask[y_pix, x_pix+1]:
                    edges.add(((x_pix+1, y_pix), (x_pix+1, y_pix+1)))
            boundary_length_dict[lab] += len(edges) * pixel_size

    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for y in range(h):
        for x in range(w):
            center = raster[y, x]
            if center == background:
                continue
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor = raster[ny, nx]
                    if neighbor == background:
                        continue
                    if neighbor != center:
                        shared_boundary[center, neighbor] += 1

    return boundary_length_dict, shared_boundary


def compute_global_contact_scores(
    raster_dir,
    n_labels=15,
    background_val=-1,
    pixel_size=1.0,
    output_dir=None
):
    """
    Compute global cluster–cluster contact scores aggregated across all raster slices.

    For each raster slice:
    1. compute outer boundary length of each label
    2. compute shared boundaries between label pairs
    3. aggregate them across all slices
    4. normalize shared boundary by each label's own total boundary length

    The resulting interaction score is:

    ``interaction_score(i, j) = shared_boundary(i, j) / total_boundary_length(i)``

    Self-interactions (``i == j``) are excluded from the output.

    Parameters
    ----------
    raster_dir : str
        Directory containing rasterized TIFF files.
    n_labels : int, default=15
        Total number of cluster labels, assumed to be labeled from 0 to ``n_labels - 1``.
    background_val : int, default=-1
        Background pixel value in the raster.
    pixel_size : float, default=1.0
        Physical size of each pixel edge, used to scale boundary lengths.
    output_dir : str or None, default=None
        Output directory for CSV files and heatmap.
        If None, results are saved under ``raster_dir``.

    Returns
    -------
    global_shared_df : pandas.DataFrame
        Global shared-boundary matrix between labels.
    boundary_df : pandas.DataFrame
        Total outer boundary length of each label.
    interaction_score_df : pandas.DataFrame
        Global interaction score matrix.

    Examples
    --------
    >>> global_shared_df, boundary_df, interaction_score_df = \
    ...     compute_global_contact_scores(
    ...         raster_dir='./raster_results/scale_125',
    ...         n_labels=15,
    ...         background_val=-1,
    ...         pixel_size=1.0,
    ...         output_dir='./raster_results/scale_125/Global Contact Score'
    ...     )
    """
    tifs = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]
    if len(tifs) == 0:
        raise FileNotFoundError(f"No tif files found: {raster_dir}")

    global_boundary_dict = {lab: 0.0 for lab in range(n_labels)}
    global_shared_matrix = np.zeros((n_labels, n_labels), dtype=float)

    for tif in tqdm(tifs, desc="Calculate boundary information for each slice"):
        path = os.path.join(raster_dir, tif)
        with rasterio.open(path) as src:
            raster = src.read(1)

        raster = raster.astype(int)

        boundary_dict, shared_matrix = compute_boundaries_and_interactions(
            raster, background=background_val, pixel_size=pixel_size
        )

        for lab, val in boundary_dict.items():
            if lab < n_labels:
                global_boundary_dict[lab] += val

        if shared_matrix.shape[0] <= n_labels:
            global_shared_matrix[:shared_matrix.shape[0], :shared_matrix.shape[1]] += shared_matrix

    global_shared_df = pd.DataFrame(global_shared_matrix,
                                    index=np.arange(n_labels),
                                    columns=np.arange(n_labels))

    if output_dir is None:
        output_dir = raster_dir
    os.makedirs(output_dir, exist_ok=True)

    shared_csv_path = os.path.join(output_dir, "global_shared_boundary_matrix.csv")
    global_shared_df.to_csv(shared_csv_path, index=True)
    print(f"Global shared boundary matrix saved to: {shared_csv_path}")

    boundary_df = pd.DataFrame(list(global_boundary_dict.items()), columns=["label", "total_boundary_length"])
    boundary_csv_path = os.path.join(output_dir, "label_boundary_length.csv")
    boundary_df.to_csv(boundary_csv_path, index=False)
    print(f"Outer boundary length of each label saved to: {boundary_csv_path}")

    interaction_score_df = global_shared_df.copy()
    for lab in range(n_labels):
        total_len = global_boundary_dict.get(lab, 0)
        if total_len > 0:
            interaction_score_df.loc[lab, :] = global_shared_df.loc[lab, :] / total_len
        else:
            interaction_score_df.loc[lab, :] = 0

    inter_csv_path = os.path.join(output_dir, "label_interaction_score_matrix.csv")
    interaction_score_df.to_csv(inter_csv_path, index=True)
    print(f"Label interaction score matrix saved to: {inter_csv_path}")

    col_norm_inter = (interaction_score_df - interaction_score_df.min()) / (
        interaction_score_df.max() - interaction_score_df.min()
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(col_norm_inter, cmap="YlGnBu", square=True)
    plt.title("Label–Label Interaction Score (shared / own boundary)", fontsize=16)
    plt.xlabel("Neighbor Label (surround)")
    plt.ylabel("Center Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interaction_score_heatmap_colnorm.png"), dpi=300)
    plt.show()

    return global_shared_df, boundary_df, interaction_score_df


def compute_per_sample_contact_scores(
    raster_dir,
    n_labels=15,
    background_val=-1,
    pixel_size=1.0,
    output_dir=None
):
    """
    Compute per-slice cluster–cluster contact scores for all label pairs.

    For each raster slice and each label pair ``(i, j)``, the interaction score is:

    ``interaction_score(i, j) = shared_boundary(i, j) / total_boundary_length(i)``

    Self-interactions (``i == j``) are excluded from the output.

    Parameters
    ----------
    raster_dir : str
        Directory containing rasterized TIFF files.
    n_labels : int, default=15
        Total number of cluster labels, assumed to be labeled from 0 to ``n_labels - 1``.
    background_val : int, default=-1
        Background pixel value.
    pixel_size : float, default=1.0
        Physical size of one pixel edge.
    output_dir : str or None, default=None
        Output directory used to save the per-slice summary CSV.
        If None, results are saved under ``raster_dir``.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame containing per-slice interaction results with columns:
        - ``sample``
        - ``label_center``
        - ``label_neighbor``
        - ``shared_length``
        - ``boundary_length_center``
        - ``interaction_score``

    Examples
    --------
    >>> result_df = compute_per_sample_contact_scores(
    ...     raster_dir='./raster_results/scale_125',
    ...     n_labels=15,
    ...     background_val=-1,
    ...     pixel_size=1.0,
    ...     output_dir='./raster_results/scale_125/per Slide Contact Score'
    ... )
    """
    tifs = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]
    if len(tifs) == 0:
        raise FileNotFoundError(f"No tif files found: {raster_dir}")

    all_results = []

    for tif in tqdm(tifs, desc="Calculate the interaction matrix for each slice"):
        path = os.path.join(raster_dir, tif)
        with rasterio.open(path) as src:
            raster = src.read(1).astype(int)

        boundary_dict, shared_matrix = compute_boundaries_and_interactions(
            raster, background=background_val, pixel_size=pixel_size
        )

        for i in range(n_labels):
            total_len_i = boundary_dict.get(i, 0.0)
            if total_len_i == 0:
                continue

            for j in range(n_labels):
                if i == j:
                    continue

                if shared_matrix.shape[0] <= i or shared_matrix.shape[1] <= j:
                    shared_len = 0.0
                else:
                    shared_len = shared_matrix[i, j]

                interaction_score = shared_len / total_len_i if total_len_i > 0 else 0.0

                all_results.append({
                    "sample": os.path.splitext(tif)[0],
                    "label_center": i,
                    "label_neighbor": j,
                    "shared_length": shared_len,
                    "boundary_length_center": total_len_i,
                    "interaction_score": interaction_score
                })

    result_df = pd.DataFrame(all_results)
    print(f"Computed interaction scores for all slices and all label–label pairs, totaling {len(result_df)} records")

    if output_dir is None:
        output_dir = raster_dir
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "interaction_all_labels_per_slice.csv")
    result_df.to_csv(output_csv, index=True)
    print(f"Saved interaction score summary table to: {output_csv}")

    return result_df

