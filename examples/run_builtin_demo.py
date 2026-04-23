# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:54:06 2026

@author: bingbing & baobao

Run SpaScope on the built-in demo AnnData dataset.

Usage
-----
Basic demo:
    python examples/run_builtin_demo.py

Run the full workflow including raster / patch / contact score:
    python examples/run_builtin_demo.py --full

Custom output directory:
    python examples/run_builtin_demo.py --output-dir demo_output

Use CPU explicitly:
    python examples/run_builtin_demo.py --device cpu
"""

from pathlib import Path
import argparse
import json
import numpy as np

from spascope import (
    set_seed,
    load_demo_adata,
    get_demo_adata_path,
    run_typical_scale_analysis,
    cluster_spatial_structures,
    plot_structure_celltype_heatmap,
    compute_cluster_shannon_diversity,
    run_landscape_metric_analysis,
    plot_cluster_patches,
    compute_global_contact_scores,
    compute_per_sample_contact_scores,
)


def parse_candidate_scales(candidate_scales_str):
    if candidate_scales_str is None:
        return None
    values = [x.strip() for x in candidate_scales_str.split(",") if x.strip()]
    return [int(float(x)) for x in values]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SpaScope using the built-in demo AnnData dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_output",
        help="Directory for saving all demo outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for GAT inference. Recommended: cpu for demo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full workflow including raster, patch, and contact score analysis.",
    )
    parser.add_argument(
        "--candidate-scales",
        type=str,
        default=None,
        help="Comma-separated candidate scales, e.g. '5,8,12,19,30,44,60'. "
             "If omitted, a compact default demo scale list will be used.",
    )
    parser.add_argument(
        "--min-cells-per-sample",
        type=int,
        default=50,
        help="Minimum number of cells required per sample for typical scale analysis.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Number of KMeans clusters used at each typical scale in the demo.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Pixel size used for rasterization in the full workflow.",
    )
    parser.add_argument(
        "--coord-key",
        type=str,
        default="spatial",
        help="Key in adata.obsm for spatial coordinates.",
    )
    parser.add_argument(
        "--type-key",
        type=str,
        default="detailed_anno",
        help="Key in adata.obs for cell type annotation.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="IMAGE_ID",
        help="Key in adata.obs for sample / ROI / image ID.",
    )
    parser.add_argument(
        "--cluster-prefix",
        type=str,
        default="Spatial_Structure_cluster_scale_",
        help="Prefix for clustering result columns written into adata.obs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting to make the demo run faster in headless environments.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    adata = load_demo_adata()
    print("Loaded built-in demo AnnData.")
    print(f"Demo data resource: {get_demo_adata_path()}")
    print(f"Cell numbers: n_obs={adata.n_obs}")
    print(f"Available obs columns: {list(adata.obs.columns)}")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")

    candidate_scales = parse_candidate_scales(args.candidate_scales)
    if candidate_scales is None:
        candidate_scales = np.unique(
            np.round(np.logspace(np.log10(5), np.log10(200), 40))
        ).astype(int).tolist()

    print(f"Candidate scales: {candidate_scales}")

    typical_scale_dir = output_dir / "01_typical_scale_analysis"
    typical_scale_dir.mkdir(parents=True, exist_ok=True)

    res, ohe, embeddings_typical_scales = run_typical_scale_analysis(
        adata=adata,
        coord_key=args.coord_key,
        type_key=args.type_key,
        Image_id=args.image_key,
        candidate_scales=candidate_scales,
        sigma_factor=0.5,
        min_cells_per_sample=args.min_cells_per_sample,
        min_k=1,
        max_k=8,
        verbose=True,
        plot_results=not args.no_plots,
        output_dir=typical_scale_dir,
        model_params=dict(hidden_dim=32, out_dim=64, heads=4, dropout=0.1),
        device=args.device,
    )

    print(f"Identified typical scales: {res['center_scales']}")

    clustering_dir = output_dir / "02_clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    adata, cluster_labels_by_scale, embeddings_by_scale, cell_ids_by_scale = cluster_spatial_structures(
        adata=adata,
        embeddings_typical_scales=embeddings_typical_scales,
        res=res,
        n_clusters=args.n_clusters,
        random_state=args.seed,
        cluster_col_prefix=args.cluster_prefix,
        copy=False,
        verbose=True,
    )

    cluster_cols = {
        scale: f"{args.cluster_prefix}{scale}"
        for scale in res["center_scales"]
    }

    adata_out_path = output_dir / "demo_with_spascope_results.h5ad"
    adata.write_h5ad(adata_out_path)
    print(f"Saved AnnData with clustering results to: {adata_out_path}")

    third_scale = res["center_scales"][2]
    third_cluster_col = cluster_cols[third_scale]

    if not args.no_plots:
        plot_structure_celltype_heatmap(
            adata=adata,
            cluster_column=third_cluster_col,
            cell_type_column=args.type_key,
            figsize=(15, 8),
            cmap="RdBu_r",
            output_dir=str(clustering_dir),
            pdf_name=f"{third_cluster_col}_celltype_heatmap.pdf",
        )

    shannon_df = compute_cluster_shannon_diversity(
        adata=adata,
        cluster_cols=cluster_cols,
        celltype_col=args.type_key,
        output_dir=str(clustering_dir),
        csv_name="shannon_index_per_cluster.csv",
        plot=not args.no_plots,
    )
    print("Computed Shannon diversity per cluster.")
    print(shannon_df.head())

    summary = {
        "demo_data_path": get_demo_adata_path(),
        "n_obs": int(adata.n_obs),
        "candidate_scales": candidate_scales,
        "center_scales": list(map(int, res["center_scales"])),
        "cluster_columns": cluster_cols,
        "adata_output": str(adata_out_path),
    }

    if args.full:
        raster_dir = output_dir / "03_raster_metrics"
        raster_dir.mkdir(parents=True, exist_ok=True)

        all_landscape_df, all_class_df = run_landscape_metric_analysis(
            adata=adata,
            cluster_cols=list(cluster_cols.values()),
            Image_col=args.image_key,
            pixel_size=args.pixel_size,
            output_dir=str(raster_dir),
            coord_key=args.coord_key,
            palette=None,
            landscape_metrics=None,
            class_metrics=None,
            plot=not args.no_plots,
        )
        print("Completed landscape metric analysis.")
        print(all_landscape_df.head())
        print(all_class_df.head())

        patch_dir = output_dir / "04_patch_plot"
        patch_dir.mkdir(parents=True, exist_ok=True)

        first_sample = str(adata.obs[args.image_key].astype(str).iloc[0])
        patch_tif = patch_dir / f"{first_sample}_{third_cluster_col}_patch.tif"

        if not args.no_plots:
            plot_cluster_patches(
                adata=adata,
                cluster_col=third_cluster_col,
                coord_key=args.coord_key,
                pixel_size=args.pixel_size,
                Image_col=args.image_key,
                sample_id=first_sample,
                output_tif=str(patch_tif),
                plot=True,
                random_seed=args.seed,
                palette=None,
            )

        raster_scale_dir = raster_dir / f"scale_{third_scale}"
        n_labels = int(adata.obs[third_cluster_col].dropna().astype(int).max()) + 1

        global_contact_dir = raster_scale_dir / "Global_Contact_Score"
        global_contact_dir.mkdir(parents=True, exist_ok=True)

        compute_global_contact_scores(
            raster_dir=str(raster_scale_dir),
            n_labels=n_labels,
            background_val=-1,
            pixel_size=1.0,
            output_dir=str(global_contact_dir),
        )

        per_sample_contact_dir = raster_scale_dir / "Per_Sample_Contact_Score"
        per_sample_contact_dir.mkdir(parents=True, exist_ok=True)

        compute_per_sample_contact_scores(
            raster_dir=str(raster_scale_dir),
            n_labels=n_labels,
            background_val=-1,
            pixel_size=1.0,
            output_dir=str(per_sample_contact_dir),
        )

        summary["full_workflow"] = True
        summary["landscape_output_dir"] = str(raster_dir)
        summary["contact_scale"] = int(third_scale)
        summary["contact_n_labels"] = int(n_labels)
    else:
        summary["full_workflow"] = False

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved run summary to: {summary_path}")
    print("SpaScope built-in demo finished successfully.")


if __name__ == "__main__":
    main()