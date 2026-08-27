# Rendering BVH motion

`broom.bvh.render` provides two optional interactive viewers. Both accept one or more `BVHDocument` instances and render their world-space joint positions. They can also accept raw NumPy arrays with shape `(frames, joints, 3)` when `parents` or `edges` are provided.

## Compatibility and installation

The viewers are optional and are not declared in `pyproject.toml` yet. The current implementation was manually tested on macOS with:

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
from broom.bvh.render import ViserSkeletonViewer

viewer = ViserSkeletonViewer(
    [walk_document, run_document],
    labels=["walk", "run"],
    offsets=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    port=8081,
    reuse_port=True,
)
```

After construction, open `http://127.0.0.1:8081` in a browser. In a notebook, `reuse_port=True` closes the previous viewer on that port and reuses its server instead of opening a new connection after every cell execution.

Important parameters:

- `clips`: one `BVHDocument`, one world-position array, or a sequence of them;
- `labels`: a label per clip;
- `offsets`: one `(x, y, z)` translation per clip, useful for side-by-side comparison;
- `edges`, `parents`, `names`, `fps`: hierarchy metadata for raw arrays;
- `spheres`: static or animated debug points;
- `port`, `reuse_port`: Viser server configuration;
- `dark_mode`: use a dark Viser scene.

Call `viewer.set_frame(index)` to select a frame and `viewer.close()` when the server is no longer needed.

## `MeshcatSkeletonViewer`

```python
from broom.bvh.render import MeshcatSkeletonViewer

viewer = MeshcatSkeletonViewer(
    [walk_document, run_document],
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

For Viser, `color` can also be an `(R, G, B)` tuple. Offsets affect skeletons only; debug-sphere coordinates are interpreted in world space.
