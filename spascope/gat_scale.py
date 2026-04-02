# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:11:53 2026

@author: bingbing & baobao
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import remove_self_loops
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def build_rbf_graph(coords, radius, sigma_factor=0.5, device="cpu"):
    """
    Construct a spatial graph using radius-based neighbors and assign Gaussian
    RBF edge weights based on Euclidean distance.

    This function builds the spatial graph for one candidate scale in SpaScope.
    Each cell is treated as a node, and edges are created between cells whose
    spatial distance is within the specified radius. Edge weights are computed
    using a Gaussian radial basis function (RBF):

    ``w_ij = exp(-d_ij^2 / (2 * sigma^2))``

    where ``sigma = sigma_factor * radius``.

    Parameters
    ----------
    coords : numpy.ndarray
        Cell spatial coordinates with shape ``(n_cells, 2)`` or ``(n_cells, n_dims)``.
    radius : float
        Neighborhood radius used to define graph connectivity.
    sigma_factor : float, default=0.5
        Scaling factor used to compute the Gaussian kernel width:
        ``sigma = sigma_factor * radius``.
    device : str, default='cpu'
        PyTorch device on which the graph tensors are created.

    Returns
    -------
    edge_index : torch.Tensor or None
        Edge index tensor with shape ``(2, n_edges)`` in PyTorch Geometric format.
        Returns None if no edges are found.
    edge_attr : torch.Tensor or None
        Edge weight tensor with shape ``(n_edges, 1)``.
        Returns None if no edges are found.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 1], [10, 10]])
    >>> edge_index, edge_attr = build_rbf_graph(
    ...     coords=coords,
    ...     radius=2.0,
    ...     sigma_factor=0.5,
    ...     device='cpu'
    ... )
    >>> print(edge_index.shape)
    >>> print(edge_attr.shape)
    """
    sigma = sigma_factor * radius
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    distances_list, indices_list = nbrs.radius_neighbors(coords)

    rows, cols, weights = [], [], []
    for i, (dists, inds) in enumerate(zip(distances_list, indices_list)):
        for d, j in zip(dists, inds):
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
            w = np.exp(- (d ** 2) / (2 * (sigma ** 2)))
            weights.append(w)

    if len(rows) == 0:
        return None, None

    edge_index = np.vstack([rows, cols]).astype(np.int64)
    edge_attr = np.array(weights, dtype=np.float32).reshape(-1, 1)

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    return edge_index, edge_attr


class GATv2WithNeighbors(nn.Module):
    """
    Two-layer GATv2 network for spatial neighbor-aware node feature transformation.

    This model is used in SpaScope to transform cell-type one-hot features into
    learned node embeddings under different spatial graph scales.

    Architecture
    ------------
    - First GATv2 layer:
      transforms input node features into hidden representations with multi-head attention
    - ELU activation
    - Dropout
    - Second GATv2 layer:
      outputs final node embeddings

    Parameters
    ----------
    in_dim : int
        Input feature dimension, typically the number of one-hot encoded cell types.
    hidden_dim : int, default=32
        Output dimension of each attention head in the first GATv2 layer.
    out_dim : int, default=64
        Final output embedding dimension.
    heads : int, default=4
        Number of attention heads in the first GATv2 layer.
    dropout : float, default=0.1
        Dropout rate used in GATv2 layers and intermediate representation.

    Attributes
    ----------
    gat1 : torch_geometric.nn.GATv2Conv
        First graph attention layer.
    gat2 : torch_geometric.nn.GATv2Conv
        Second graph attention layer.
    act : torch.nn.ELU
        Nonlinear activation layer.
    dropout : torch.nn.Dropout
        Dropout layer applied between the two GAT layers.

    Examples
    --------
    >>> model = GATv2WithNeighbors(
    ...     in_dim=20,
    ...     hidden_dim=32,
    ...     out_dim=64,
    ...     heads=4,
    ...     dropout=0.1
    ... )
    >>> out = model(x, edge_index, edge_attr)
    >>> print(out.shape)
    """
    def __init__(self, in_dim, hidden_dim=32, out_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim, out_channels=hidden_dim, heads=heads,
            edge_dim=1, dropout=dropout, concat=True
            ,add_self_loops=True
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads, out_channels=out_dim,
            heads=1, edge_dim=1, dropout=dropout, concat=False
            ,add_self_loops=True
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x


def compute_gat_features_at_scales(coords, node_type_onehot, candidate_scales,
                                    shared_model, sigma_factor=0.5, device="cpu"):
    """
    Compute GAT-transformed node features across multiple candidate spatial scales.

    For each candidate scale, this function:
    1. builds a radius-based spatial graph
    2. applies the shared GAT model to transform node features
    3. stores the resulting node embeddings

    The output is a 3D tensor-like NumPy array with dimensions:

    ``(n_cells, out_dim, n_scales)``

    Parameters
    ----------
    coords : numpy.ndarray
        Spatial coordinates of cells with shape ``(n_cells, 2)`` or ``(n_cells, n_dims)``.
    node_type_onehot : numpy.ndarray
        One-hot encoded cell-type matrix with shape ``(n_cells, n_features)``.
    candidate_scales : list of int or float
        List of candidate spatial radii used to construct graphs.
    shared_model : torch.nn.Module
        Shared GAT model, typically an instance of ``GATv2WithNeighbors``.
    sigma_factor : float, default=0.5
        Factor controlling the Gaussian RBF edge-weight width.
    device : str, default='cpu'
        PyTorch device used for inference.

    Returns
    -------
    transformed : numpy.ndarray
        Array of transformed node features with shape ``(n_cells, out_dim, n_scales)``.

    Examples
    --------
    >>> transformed = compute_gat_features_at_scales(
    ...     coords=coords,
    ...     node_type_onehot=feat,
    ...     candidate_scales=[8, 19, 44, 125],
    ...     shared_model=model,
    ...     sigma_factor=0.5,
    ...     device='cpu'
    ... )
    >>> print(transformed.shape)
    """
    x = torch.tensor(node_type_onehot, dtype=torch.float32, device=device)
    n, _ = x.shape
    S = len(candidate_scales)
    out_dim = shared_model.gat2.out_channels
    transformed = torch.zeros((n, out_dim, S), dtype=torch.float32, device=device)

    shared_model.eval()
    for si, r in enumerate(candidate_scales):
        edge_index, edge_attr = build_rbf_graph(coords, r, sigma_factor=sigma_factor, device=device)
        if edge_index is None:
            transformed[:, :, si] = (
                x[:, :out_dim] if x.shape[1] >= out_dim else
                torch.nn.functional.pad(x, (0, out_dim - x.shape[1]))
            )
            continue

        with torch.no_grad():
            out = shared_model(x, edge_index, edge_attr)
            transformed[:, :, si] = out

    return transformed.cpu().numpy()


def compute_scale_correlation(transformed_signals):
    """
    Compute the scale-by-scale correlation matrix from multi-scale node embeddings.

    This function summarizes the similarity between candidate scales.

    Parameters
    ----------
    transformed_signals : numpy.ndarray
        Multi-scale node embedding array with shape ``(n_cells, n_features, n_scales)``.

    Returns
    -------
    mean_corr : numpy.ndarray
        Mean scale correlation matrix with shape ``(n_scales, n_scales)``.

    Examples
    --------
    >>> mean_corr = compute_scale_correlation(transformed)
    >>> print(mean_corr.shape)
    """
    n, d, S = transformed_signals.shape
    corr_acc = np.zeros((S, S), dtype=np.float64)
    for feat in range(d):
        mat = transformed_signals[:, feat, :]
        C = np.corrcoef(mat.T)
        C = np.nan_to_num(C, nan=0.0)
        corr_acc += C
    mean_corr = corr_acc / d
    return mean_corr


def optimal_scale_clustering(corr_matrix, k):
    """
    Segment ordered candidate scales into ``k`` contiguous groups using dynamic programming.

    This function is used in SpaScope to partition the scale correlation matrix into
    contiguous scale segments so that scales within the same segment are maximally similar.

    Parameters
    ----------
    corr_matrix : numpy.ndarray
        Scale correlation matrix with shape ``(n_scales, n_scales)``.
    k : int
        Number of contiguous segments to partition the scales into.

    Returns
    -------
    parts : list of tuple
        List of ``(start_idx, end_idx)`` tuples defining the optimal segmentation.

    Examples
    --------
    >>> parts = optimal_scale_clustering(mean_corr, k=3)
    >>> print(parts)
    """
    S = corr_matrix.shape[0]
    cost = np.zeros((S, S))
    for i in range(S):
        for j in range(i, S):
            sub = corr_matrix[i:j+1, i:j+1]
            cost[i, j] = np.sum(1.0 - sub)

    dp = np.full((S+1, k+1), np.inf)
    prev = np.full((S+1, k+1), -1, dtype=int)
    dp[0, 0] = 0.0

    for i in range(1, S+1):
        for kk in range(1, min(i, k)+1):
            best = np.inf
            best_j = -1
            for j in range(kk-1, i):
                val = dp[j, kk-1] + cost[j, i-1]
                if val < best:
                    best = val
                    best_j = j
            dp[i, kk] = best
            prev[i, kk] = best_j

    parts = []
    i = S
    kk = k
    while kk > 0:
        j = prev[i, kk]
        parts.append((j, i-1))
        i = j
        kk -= 1
    return parts[::-1]


def identify_typical_scales_from_correlation(mean_corr, scales, min_k=1, max_k=None):
    """
    Select representative typical scales from a scale correlation matrix using
    dynamic-programming segmentation and a penalized cost criterion.

    This function evaluates different numbers of scale segments ``k``, computes
    the total segmentation cost for each ``k``, adds a linear penalty term, and
    selects the optimal number of segments. Within each selected segment, the
    representative typical scale is chosen as the most central scale in terms of
    within-segment correlation.

    Parameters
    ----------
    mean_corr : numpy.ndarray
        Scale correlation matrix with shape ``(n_scales, n_scales)``.
    scales : list of int or float
        Candidate scale values corresponding to the rows/columns of ``mean_corr``.
    min_k : int, default=1
        Minimum number of scale segments to evaluate.
    max_k : int or None, default=None
        Maximum number of scale segments to evaluate.
        If None, defaults to ``min(6, n_scales)``.

    Returns
    -------
    result : dict
        Dictionary containing the segmentation and representative scales, including:
        - ``chosen_k`` : optimal number of segments
        - ``parts`` : list of segment index ranges
        - ``center_idx`` : indices of representative scales
        - ``center_scales`` : representative scale values
        - ``segment_scales`` : list of scale values in each segment
        - ``costs`` : raw total costs for each k
        - ``penalized_costs`` : penalized costs for each k
        - ``ks`` : tested k values
        - ``lambda_pen`` : penalty coefficient

    Examples
    --------
    >>> res = identify_typical_scales_from_correlation(
    ...     mean_corr=mean_corr,
    ...     scales=[5, 8, 12, 19, 28, 44, 67, 125],
    ...     min_k=1,
    ...     max_k=6
    ... )
    >>> print(res['chosen_k'])
    >>> print(res['center_scales'])
    """
    S = mean_corr.shape[0]
    if max_k is None:
        max_k = min(6, S)

    costs = []
    parts_list = []
    ks = list(range(min_k, max_k+1))

    for k in ks:
        parts = optimal_scale_clustering(mean_corr, k)
        total_cost = sum([np.sum(1.0 - mean_corr[s:e+1, s:e+1]) for s,e in parts])
        parts_list.append(parts)
        costs.append(total_cost)

    costs = np.array(costs)

    diffs = np.diff(costs)
    lambda_pen = 0.2* np.mean(np.abs(diffs))

    penalized_costs = costs + lambda_pen * np.array(ks)

    idx_min = np.argmin(penalized_costs)
    chosen_k = ks[idx_min]
    
    parts = parts_list[ks.index(chosen_k)]
    centers_idx = []
    segment_scales = []
    for s,e in parts:
        sub = mean_corr[s:e+1, s:e+1]
        costs_local = np.sum(1.0 - sub, axis=1)
        center_local = int(np.argmin(costs_local))
        centers_idx.append(s + center_local)
        
        segment_scale_indices = list(range(s, e+1))
        segment_scale_values = [scales[i] for i in segment_scale_indices]
        segment_scales.append(segment_scale_values)
        
    centers = [scales[i] for i in centers_idx]

    return {
        'chosen_k': chosen_k,
        'parts': parts,
        'center_idx': centers_idx,
        'center_scales': centers,
        'segment_scales': segment_scales,  
        'costs': list(costs),                
        'penalized_costs': list(penalized_costs),  
        'ks': ks,
        'lambda_pen': float(lambda_pen)
    }


def plot_scale_correlation_heatmap(mean_corr, candidate_scales, segmentation_result, output_dir=None):
    """
    Plot the global scale correlation heatmap with segmentation annotations and
    representative typical scales.

    This visualization is used in SpaScope to display:
    - the correlation structure among candidate scales
    - the segmented scale groups
    - the representative scale selected in each segment

    Parameters
    ----------
    mean_corr : numpy.ndarray
        Scale correlation matrix with shape ``(n_scales, n_scales)``.
    candidate_scales : list of int or float
        Candidate scale values shown on the x- and y-axes.
    segmentation_result : dict
        Result dictionary returned by ``identify_typical_scales_from_correlation``.
        Must contain:
        - ``parts``
        - ``center_idx``
        - ``center_scales``
        - ``segment_scales``
    output_dir : str or None, default=None
        Directory used to save the annotated heatmap PDF.
        If None, the figure is only displayed and not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure object.

    Examples
    --------
    >>> fig = plot_scale_correlation_heatmap(
    ...     mean_corr=mean_corr,
    ...     candidate_scales=[5, 8, 12, 19, 28, 44, 67, 125],
    ...     segmentation_result=res,
    ...     output_dir='./scale_diagnostics_GAT_shared'
    ... )
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(10, 6.6))
    
    im = sns.heatmap(mean_corr, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
                    xticklabels=candidate_scales, yticklabels=candidate_scales,
                    ax=ax, cbar_kws={'label': 'Correlation'},square = True)
    
    plt.title("Global Scale Correlation Matrix with Segmentation", fontsize=15, pad=10)
    plt.xlabel("Scale (μm)", fontsize=15)
    plt.ylabel("Scale (μm)", fontsize=15)
    
    cbar = im.collections[0].colorbar
    if cbar:
        cbar.set_label('Correlation', fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        
    
    parts = segmentation_result['parts']
    centers_idx = segmentation_result['center_idx']
    center_scales = segmentation_result['center_scales']
    segment_scales = segmentation_result['segment_scales']
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, ((s, e), center_idx, color) in enumerate(zip(parts, centers_idx, colors)):
        rect = plt.Rectangle((s, s), e-s+1, e-s+1, fill=False, 
                           edgecolor=color, linewidth=3, linestyle='--', alpha=0.8)
        ax.add_patch(rect)
        
        ax.plot(center_idx, center_idx, 'o', color=color, markersize=10, 
               label=f'Segment {i+1}: {center_scales[i]}μm')
        
        segment_info = f"Seg {i+1}: {min(segment_scales[i]):.0f}-{max(segment_scales[i]):.0f}μm\nCenter: {center_scales[i]}μm"
        ax.text(e-5, s, segment_info, fontsize=12, va='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
    
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/M_global_annotated_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def run_typical_scale_analysis(
    adata, coord_key='spatial', type_key='detailed_anno', Image_id='IMAGE_ID',
    candidate_scales=None, sigma_factor=0.5, min_cells_per_sample=50,
    min_k=1, max_k=6, verbose=True, plot_results=True, output_dir=None,
    model_params=None, device="cuda"
):
    """
    Identify biologically representative spatial scales using a shared GAT model
    and export cell node embeddings at the selected typical scales.

    Workflow
    --------
    1. Encode cell types into one-hot node features.
    2. Build radius-based spatial graphs at multiple candidate scales.
    3. Use a shared GATv2 model to transform node features across scales.
    4. Compute scale–scale correlation matrices for each sample.
    5. Aggregate sample-level correlation matrices into a global correlation matrix.
    6. Segment candidate scales into typical scale groups using dynamic programming.
    7. Select representative (center) scales from each segment.
    8. Export cell embeddings under the identified typical scales for downstream clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial coordinates and cell type annotations.
    coord_key : str, default='spatial'
        Key in ``adata.obsm`` storing spatial coordinates of cells, typically shape ``(n_cells, 2)``.
    type_key : str, default='detailed_anno'
        Key in ``adata.obs`` storing cell type annotations.
    Image_id : str, default='IMAGE_ID'
        Key in ``adata.obs`` indicating sample / image / ROI identity.
    candidate_scales : list of int or float, optional
        Candidate spatial radii (e.g. in microns) used to construct graphs.
        If None, defaults to ``[10, 20, 40, 80, 160, 320]``.
    sigma_factor : float, default=0.5
        Controls the Gaussian kernel width in graph edge weighting:
        ``sigma = sigma_factor * radius``.
    min_cells_per_sample : int, default=50
        Minimum number of cells required for a sample to be analyzed.
        Samples below this threshold are skipped.
    min_k : int, default=1
        Minimum number of scale segments when selecting typical scales.
    max_k : int, default=6
        Maximum number of scale segments when selecting typical scales.
    verbose : bool, default=True
        Whether to print progress and scale segmentation details.
    plot_results : bool, default=True
        Whether to generate diagnostic plots, including:
        - global scale correlation heatmap
        - global cost curve
        - per-sample cost curves
    output_dir : str or None, default=None
        Directory used to save plots and serialized results.
        If provided, outputs such as heatmaps and pickle files are saved there.
    model_params : dict or None, default=None
        Parameters for the shared GATv2 model. Example:
        ``dict(hidden_dim=32, out_dim=64, heads=4, dropout=0.1)``.
        If None, default values are used.
    device : str, default='cuda'
        PyTorch device used for GAT inference. Common choices are ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    res_global : dict
        Global scale identification results, including:
        - ``chosen_k``: optimal number of scale segments
        - ``center_scales``: representative typical scales
        - ``center_idx``: indices of typical scales in ``candidate_scales``
        - ``segment_scales``: scale ranges of each segment
        - ``M_global``: global mean correlation matrix
        - ``per_sample``: per-sample segmentation details
    ohe : sklearn.preprocessing.OneHotEncoder
        Fitted one-hot encoder used to encode cell types.
    embeddings_typical_scales : dict
        Dictionary mapping each sample ID to a list of DataFrames,
        where each DataFrame contains cell embeddings under one typical scale.

        Structure:
        ``embeddings_typical_scales[sample_id][i]``
        corresponds to the i-th typical scale in ``res_global['center_scales']``.

    Examples
    --------
    >>> candidate_scales = np.unique(
    ...     np.round(np.logspace(np.log10(5), np.log10(200), 40))
    ... ).astype(int).tolist()
    >>>
    >>> set_seed(42)
    >>> res, ohe, embeddings_typical_scales = run_typical_scale_analysis(
    ...     adata=adata,
    ...     coord_key='spatial',
    ...     type_key='detailed_anno',
    ...     Image_id='IMAGE_ID',
    ...     candidate_scales=candidate_scales,
    ...     sigma_factor=0.5,
    ...     min_cells_per_sample=50,
    ...     min_k=1,
    ...     max_k=8,
    ...     model_params=dict(hidden_dim=32, out_dim=64, heads=4, dropout=0.1),
    ...     device='cpu',
    ...     output_dir='./scale_diagnostics_GAT_shared'
    ... )
    >>>
    >>> print(res['center_scales'])
    >>> print(list(embeddings_typical_scales.keys())[:3])
    """
    if candidate_scales is None:
        candidate_scales = [10, 20, 40, 80, 160, 320]

    types_all = adata.obs[type_key].astype(str).values.reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(types_all)
    sample_ids = adata.obs[Image_id].unique().tolist()

    sample_corrs = []
    sample_details = []
    embeddings_per_sample = {}

    in_dim = ohe.transform(types_all[:1]).shape[1]
    model_params = model_params or dict(hidden_dim=32, out_dim=64, heads=4, dropout=0.1)
    shared_model = GATv2WithNeighbors(
        in_dim=in_dim,
        hidden_dim=model_params["hidden_dim"],
        out_dim=model_params["out_dim"],
        heads=model_params["heads"],
        dropout=model_params["dropout"]
    ).to(device)
    print(f"Using a shared GATv2 model (in_dim={in_dim}, out_dim={model_params['out_dim']})")

    for i, sid in enumerate(sample_ids):
        sub = adata[adata.obs[Image_id] == sid]
        n = sub.n_obs
        if n < min_cells_per_sample:
            if verbose:
                print(f"sample {sid} skip (n={n} < {min_cells_per_sample})")
            continue

        coords = np.asarray(sub.obsm[coord_key]).astype(float)
        feat = ohe.transform(sub.obs[type_key].astype(str).values.reshape(-1, 1))

        transformed = compute_gat_features_at_scales(
            coords, feat, candidate_scales,
            shared_model=shared_model,
            sigma_factor=sigma_factor,
            device=device
        )

        embeddings_per_sample[sid] = transformed
        mean_corr = compute_scale_correlation(transformed)
        sample_corrs.append(mean_corr)

        local_res = identify_typical_scales_from_correlation(mean_corr, candidate_scales, min_k=min_k, max_k=max_k)
        local_res["sample_id"] = sid
        sample_details.append(local_res)

        if verbose:
            print(f" [{i+1}/{len(sample_ids)}] sample {sid} processed "
                  f"(n={n}, chosen_k={local_res['chosen_k']}, scales={local_res['center_scales']})")
            print("\nDetailed segment information:")
            for seg_i, (segment, center_scale) in enumerate(zip(local_res['segment_scales'], local_res['center_scales'])):
                print(f"   Segment {seg_i+1}:")
                print(f"     - Scale range: {min(segment):.1f}μm ~ {max(segment):.1f}μm")
                print(f"     - Scale inclusion: {[f'{s:.1f}' for s in segment]}")
                print(f"     - Center scale: {center_scale}μm")
                print(f"     - Scale numbers: {len(segment)}")

    M_global = np.mean(np.stack(sample_corrs, axis=0), axis=0)
    res_global = identify_typical_scales_from_correlation(M_global, candidate_scales, min_k=min_k, max_k=max_k)
    res_global["M_global"] = M_global
    res_global["candidate_scales"] = candidate_scales
    res_global["sample_corrs"] = sample_corrs
    res_global["per_sample"] = sample_details
    res_global["embeddings_per_sample"] = embeddings_per_sample

    if verbose:
        print("\n" + "="*60)
        print("Global typical scale identification results：")
        print(f"   Optimal number of segments: {res_global['chosen_k']}")
        print(f"   Center scale: {res_global['center_scales']}")
        print("\nDetailed segment information:")
        for seg_i, (segment, center_scale) in enumerate(zip(res_global['segment_scales'], res_global['center_scales'])):
            print(f"   Segment {seg_i+1}:")
            print(f"     - Scale range: {min(segment):.1f}μm ~ {max(segment):.1f}μm")
            print(f"     - Scale inclusion: {[f'{s:.1f}' for s in segment]}")
            print(f"     - Center scale: {center_scale}μm")
            print(f"     - Scale numbers: {len(segment)}")
        print("="*60)

    if plot_results:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plot_scale_correlation_heatmap(res_global['M_global'], candidate_scales, res_global, output_dir)

        ks = np.array(res_global['ks'])
        costs = np.array(res_global['penalized_costs'])
        knee_k = res_global['chosen_k']

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.figure(figsize=(5, 4))
        plt.plot(ks, costs, '-o', label="Global cost curve", color='royalblue')
        plt.axvline(knee_k, color='red', linestyle='--', label=f"Knee k={knee_k}")
        plt.xlabel("k (number of scale segments)", fontsize=20)
        plt.ylabel("Total cost (∑(1 - corr))", fontsize=20)
        plt.title("Global cost vs k (lower=better)", fontsize=20)
        plt.legend(fontsize=20)
        if output_dir:
            plt.savefig(f"{output_dir}/global_cost_curve_knee.pdf", dpi=300)
        plt.show()

        plt.figure(figsize=(7, 6))
        for sd_i in sample_details:
            plt.plot(sd_i['ks'], sd_i['penalized_costs'], alpha=0.3)
            plt.scatter(sd_i['chosen_k'], sd_i['penalized_costs'][sd_i['ks'].index(sd_i['chosen_k'])],
                        color='red', s=15)
        plt.plot(ks, costs, '-o', color='black', lw=2, label="Global")
        plt.xlabel("k", fontsize=20)
        plt.ylabel("Cost (∑(1 - corr))", fontsize=20)
        plt.title("Per-sample cost curves with knees", fontsize=20)
        plt.legend(fontsize=20)
        if output_dir:
            plt.savefig(f"{output_dir}/per_sample_cost_knees.pdf", dpi=300)
        plt.show()

    embeddings_typical_scales = {}

    print("\nOrganizing cell embedding features under typical scales")
    center_scales = res_global["center_scales"]
    center_idx = res_global["center_idx"]

    for sid, transformed in embeddings_per_sample.items():
        sub = adata[adata.obs[Image_id] == sid]
        cell_ids = np.array(sub.obs_names)

        embeddings_per_scale = []
        for scale_value, scale_idx in zip(center_scales, center_idx):
            emb = transformed[:, :, scale_idx]
            df = pd.DataFrame(
                emb,
                index=cell_ids,
                columns=[f"GAT64_s{scale_value}_{j}" for j in range(emb.shape[1])]
            )
            embeddings_per_scale.append(df)

        embeddings_typical_scales[sid] = embeddings_per_scale

    print("Completed")

    if output_dir:
        with open(f"{output_dir}/gat_scale_results.pkl", "wb") as f:
            pickle.dump({
                "res": res_global,
                "ohe": ohe,
                "embeddings_typical_scales": embeddings_typical_scales
            }, f)
        print("Saved res_global, ohe, embeddings_typical_scales to:",
              f"{output_dir}/gat_scale_results.pkl")

    return res_global, ohe, embeddings_typical_scales





