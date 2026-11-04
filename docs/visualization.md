# Visualization BVH motion

`broom.visualization` provides two optional interactive viewers. Both accept one or more `Motion` instances and render their world-space joint positions. They can also accept raw NumPy arrays with shape `(frames, joints, 3)` when `parents` or `edges` are provided.

## Compatibility and installation

The viewers are optional and are not declared in `pyproject.toml` yet. The earlier viewer implementation was manually tested on macOS with the versions below. These are recorded compatibility results, not a new cross-platform test:

| Viewer | Tested version |
| --- | --- |
| Viser | `1.0.26` |
| MeshCat | `0.3.2` |

Other versions and operating systems may work, but they have not been verified by this project. In particular, follow the upstream installation guidance instead of treating a generic `pip install` command as a compatibility guarantee:

- [Viser upstream repository and installation documentation](https://github.com/nerfstudio-project/viser)
- [MeshCat Python upstream repository and installation documentation](https://github.com/meshcat-dev/meshcat-python)

`MeshcatSkeletonViewer` also requires `ipywidgets` and IPython in its Jupyter mode.

## Choosing a viewer

Use `MeshcatSkeletonViewer` for iterative work inside Jupyter. Its default `display_mode="jupyter"` renders the scene and playback controls in the output of the current cell. Set `display_mode="browser"` when the same Meshcat scene should open in a separate browser tab.

Use `ViserSkeletonViewer` for a browser-based scene. It starts a local HTTP server and is better suited to inspecting a scene outside the notebook. Keep a reference to the viewer in a variable so its server remains available while the kernel runs.

## `ViserSkeletonViewer`

```python
from broom.io import load_bvh
from broom.visualization import ViserSkeletonViewer

walk_motion = load_bvh("walk.bvh")
run_motion = load_bvh("run.bvh")
viewer = ViserSkeletonViewer(
    [walk_motion, run_motion],
    labels=["walk", "run"],
    offsets=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    port=8081,
    reuse_port=True,
)
```

After construction, open `http://127.0.0.1:8081` in a browser. In a notebook, `reuse_port=True` closes the previous viewer on that port and reuses its server instead of opening a new connection after every cell execution.

Important parameters:

- `clips`: one `Motion`, one world-position array, or a sequence of them;
- `labels`: a label per clip;
- `offsets`: one `(x, y, z)` translation per clip, useful for side-by-side comparison;
- `edges`, `parents`, `names`, `fps`: hierarchy metadata for raw arrays;
- `spheres`: static or animated debug points;
- `port`, `reuse_port`: Viser server configuration;
- `dark_mode`: use a dark Viser scene.

Call `viewer.set_frame(index)` to select a frame and `viewer.close()` when the server is no longer needed.

## `MeshcatSkeletonViewer`

```python
from broom.io import load_bvh
from broom.visualization import MeshcatSkeletonViewer

walk_motion = load_bvh("walk.bvh")
run_motion = load_bvh("run.bvh")
viewer = MeshcatSkeletonViewer(
    [walk_motion, run_motion],
    labels=["walk", "run"],
    offsets=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    display_mode="jupyter",
    reuse_connection=True,
)
```

`MeshcatSkeletonViewer` is notebook-first: it displays the scene together with playback controls in the cell output. To open the same scene in a browser, use `display_mode="browser"`. It accepts the same clip, hierarchy, label, offset, frame-rate, and `spheres` inputs as the Viser viewer. `reuse_connection=True` clears and reuses the current Meshcat connection when a notebook cell is executed again.

Call `viewer.set_frame(index)` to update the scene and `viewer.close()` to remove the rendered objects and stop its playback task.

## Debug spheres

Pass `spheres` as one dictionary or a sequence of dictionaries. A sphere is either static through `position` or animated through `positions`.

```python
from broom.kinematics import compute_global_positions

marker_positions = compute_global_positions(walk_motion)[:, walk_motion.hierarchy.root]
spheres = [
    {
        "label": "animated_marker",
        "positions": marker_positions,  # (frames, 3)
        "radius": 0.03,
        "color": 0xE63946,
    },
    {
        "label": "fixed_marker",
        "position": [0.0, 1.0, 0.0],
        "radius": 0.02,
    },
]
```

Pass the dictionaries with `ViserSkeletonViewer(walk_motion, spheres=spheres)` or `MeshcatSkeletonViewer(walk_motion, spheres=spheres)`.

For Viser, `color` can also be an `(R, G, B)` tuple. Offsets affect skeletons only; debug-sphere coordinates are interpreted in world space.


## Raw positions, frame rates, and display offsets

```python
import numpy as np
from broom.kinematics import compute_global_positions

positions = compute_global_positions(walk_motion)  # (F, J, 3)
parents = np.array([joint.parent for joint in walk_motion.hierarchy.joints])
viewer = ViserSkeletonViewer(
    positions,
    parents=parents,
    names=walk_motion.hierarchy.joint_names,
    fps=30,
    port=8081,
    reuse_port=True,
)
```

F is the frame count; J is the joint count. Raw positions need `parents` or an edge list and a suitable playback FPS. A `Motion` supplies hierarchy and timing; its frame rate is currently rounded to an integer for playback. An explicit `fps` overrides it. Multiple clips share a frame cursor, so resample them to a common FPS when comparing their timing.

Viewer offsets only move the rendered skeletons. They do not modify a `Motion` or affect BVH export. The MeshCat floor (`plane_axis`, `plane_offset`, `plane_size`) is also display geometry, not a collision constraint.

[Visualization API](api/visualization.md)
