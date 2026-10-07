# Installation

Broom requires Python 3.10 or newer. Clone the repository and install it:

```bash
git clone https://github.com/AvaCapo/broom.git
cd broom
python -m pip install -e .
```

This installs NumPy, SciPy, and Matplotlib. The editable install uses your local checkout; omit `-e` for a regular installation.

## Optional features

Install only the dependencies needed for your workflow:

| Feature | Command |
|---|---|
| Browser visualization with Viser | `python -m pip install viser` |
| MeshCat visualization in Jupyter | `python -m pip install meshcat ipywidgets ipython` |
| Torch-based IK refinement | `python -m pip install -e ".[ik]"` |

NumPy FABRIK and SciPy space-time optimization work with the core installation and do not require Torch. The `[all]` extra currently adds Torch, not the viewers. See [Visualization](../visualization.md) for viewer setup and compatibility notes.

## Check the installation

```python
from broom import Joint, Hierarchy, Motion
from broom.io import load_bvh, write_bvh
```

Continue with [First example](quickstart.md).
