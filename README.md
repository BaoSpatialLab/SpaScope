<div align="center">

<img src="logo/spascope_logo.png" alt="SpaScope logo">

**A Python package for multi-scale spatial structure analysis from spatial single-cell data**

</div>

SpaScope provides a unified computational framework for identifying and characterizing spatial structures in spatial single-cell data through graph-based feature extraction, representative scale identification, spatial structure clustering, diversity analysis, raster-based landscape analysis, patch visualization, and contact score quantification.

## Installation

<<<<<<< HEAD
=======
### Install from GitHub

>>>>>>> 6d51740 (add built-in AnnData demo, example script, packaging updates, and README)
```bash
git clone https://github.com/BaoSpatialLab/SpaScope.git
cd SpaScope
pip install -e .
```

## Quick start: run the built-in demo

Run the basic demo:

```bash
python examples/run_builtin_demo.py
```

Run the full workflow:

```bash
python examples/run_builtin_demo.py --full
```

By default, outputs are written to:

```text
./demo_outputs
```

You can change the output directory and device:

```bash
python examples/run_builtin_demo.py --output-dir ./demo_outputs_cpu --device cpu
```

---


