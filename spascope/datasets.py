# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:22:22 2026

@author: bingbing & baobao

Built-in demo dataset utilities for SpaScope.
"""

from importlib.resources import as_file, files
import anndata as ad


def get_demo_adata_path():
    """
    Return the path-like resource handle of the built-in demo AnnData file.

    Returns
    -------
    str
        Path-like string pointing to ``spascope/data/demo/demo_adata.h5ad``.

    Notes
    -----
    This is mainly intended for users who want to know where the bundled
    demo file is located after installation.
    """
    resource = files("spascope").joinpath("data", "demo", "demo_adata.h5ad")
    return str(resource)


def load_demo_adata():
    """
    Load the built-in demo AnnData object shipped with SpaScope.

    Returns
    -------
    anndata.AnnData
        The bundled demo dataset.

    Examples
    --------
    >>> from spascope import load_demo_adata
    >>> adata = load_demo_adata()
    >>> print(adata)
    """
    resource = files("spascope").joinpath("data", "demo", "demo_adata.h5ad")
    with as_file(resource) as demo_path:
        adata = ad.read_h5ad(demo_path)
    return adata
