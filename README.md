# BROOM: BVH Read, Optimize, Operate and Manipulate

<p align="center">
  <img src="cover.png" alt="broom - BVH motion processing toolkit" width="100%">
</p>

`broom` is a Python toolkit for reading, editing, interpolating, and retargeting skeletal animation stored in [BVH](https://en.wikipedia.org/wiki/Biovision_Hierarchy) files. It provides numerical motion-processing functions and optional interactive viewers for inspecting the results.

## Main features

- Read and write BVH hierarchies, animation channels, and frame timing.
- Inspect skeletons and compute world-space joint positions and rotations with FK.
- Trim, reverse, scale, resample, and blend motion clips.
- Transfer animation with explicit or suggested joint mappings; refine motion along chains.
- Use space-time optimization with position, stationary, floor, joint-limit, and pair-distance constraints.
- Solve joint chains with FABRIK, or remove locomotion with in-place conversion.
- Compare animations in Viser or MeshCat.

## Data model

`Joint` and `Hierarchy` describe the immutable skeleton and channel layout. `Motion` stores that hierarchy, frame timing, and an editable NumPy sample table. This keeps structural changes separate from animation edits. Most processing operations return a new motion; direct writes to `motion.values` change it in place.

The API is under active development. See the [data model](docs/concepts/data-model.md) for array ownership and the [changelog](CHANGELOG.md) for recent changes.

## Installation

Python 3.10 or newer. Clone the repository and install from its root:

```bash
git clone https://github.com/AvaCapo/broom.git
cd broom
python -m pip install -e .
```

NumPy, SciPy, and Matplotlib are installed automatically. Interactive viewers are optional; see [Installation](docs/getting-started/installation.md).

## Load, inspect, edit, and save

```python
from broom.io import load_bvh, write_bvh
from broom.kinematics import compute_global_positions
from broom.ops.motion_editing import trim_frames
from broom.ops.resample import resample_fps

motion = load_bvh("walk.bvh")
print(motion.hierarchy.format_tree())
print(f"{motion.frame_count} frames, {1 / motion.frame_time:.2f} FPS")

positions = compute_global_positions(motion)  # (frames, joints, 3)
print("Root start:", positions[0, motion.hierarchy.root])

# Keep the first two seconds, including the sample at t=2 if available.
stop = min(motion.frame_count, int(2.0 / motion.frame_time) + 1)
clip = trim_frames(motion, stop=stop)
clip = resample_fps(clip, target_fps=30.0)
write_bvh(clip, "walk_short_30fps.bvh")
```

BVH does not specify length units. Broom does not infer units, floor placement, or exporter-specific root conventions. Check these before using world-space targets or default solver tolerances; see [Coordinates and timing](docs/concepts/coordinates.md).

## Documentation

- [First example](docs/getting-started/quickstart.md): work with a BVH file, then create a motion programmatically.
- [Data model](docs/concepts/data-model.md): skeleton structure, channel layout, and mutable samples.
- [Editing](docs/guides/editing.md) and [interpolation](docs/guides/interpolation.md).
- [Forward and inverse kinematics](docs/guides/kinematics.md).
- [Retargeting](docs/guides/retargeting.md) and [space-time usage](docs/spacetime_usage.md).
- [Root motion](docs/guides/root-motion.md) and [interactive visualization](docs/visualization.md).
- [Documentation home](docs/index.md) and [Motion API](docs/api/motion.md).

## License

`broom` is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

You may use, modify, and distribute the software for purposes permitted by that license. Commercial use is not granted by the public license and requires a separate license from the copyright holder.

Because it restricts commercial use, `broom` is source-available software rather than Open Source Initiative (OSI)-approved open-source software. See the [`LICENSE`](LICENSE) file for the complete terms.
