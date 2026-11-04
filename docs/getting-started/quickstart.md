# First example

Start with a BVH animation such as `walk.bvh`. This example reads the file, inspects its skeleton, trims the clip, and saves a new BVH.

## Load and inspect an animation

```python
from broom.io import load_bvh, write_bvh
from broom.kinematics import compute_global_positions

motion = load_bvh("walk.bvh")
hierarchy = motion.hierarchy
print(hierarchy.format_tree(show_channels=True))
print(f"Joints: {hierarchy.joint_count}")
print(f"Frames: {motion.frame_count}; FPS: {1 / motion.frame_time:.3f}")
print(f"Duration: {motion.duration:.3f} seconds")
print(f"Sample table: {motion.values.shape}")

positions = compute_global_positions(motion)
print("Root position at frame 0:", positions[0, hierarchy.root])
```

`load_bvh` returns a `Motion`. Its hierarchy stores joints and channel layout; `values` stores animation samples. FK returns world positions with shape `(F, J, 3)`, where F is the frame count and J is the joint count.

The loader does not infer units or convert exporter conventions. See [Coordinates and timing](../concepts/coordinates.md) if the motion appears at an unexpected scale or height.

## Trim the animation and save it

```python
from broom.ops.motion_editing import trim_frames

# Retain up to the first two seconds. stop is exclusive, like a Python slice.
stop = min(motion.frame_count, int(2.0 / motion.frame_time) + 1)
short_clip = trim_frames(motion, start=0, stop=stop)
write_bvh(short_clip, "walk_short.bvh")

print(f"Saved {short_clip.frame_count} frames")
print(f"Original still has {motion.frame_count} frames")
```

`trim_frames` returns an independent motion; the source is unchanged. For a specific interval, use `trim_frames(motion, start=30, stop=90)` on a clip with enough frames. An empty slice is rejected.

To view the saved result, follow [Visualization](../visualization.md). Direct array edits have different ownership rules; see [Motion values](../concepts/data-model.md#motion-values-and-array-ownership).

## Create a motion programmatically

You can also construct data without a BVH file, for generated poses or small numerical experiments. This is optional for file-based workflows.

```python
import numpy as np
from broom import Joint, Hierarchy, Motion

hierarchy = Hierarchy((
    Joint("Root", -1, (0.0, 1.0, 0.0), (
        "Xposition", "Yposition", "Zposition",
        "Zrotation", "Xrotation", "Yrotation",
    )),
    Joint("Foot", 0, (0.0, -1.0, 0.0)),
))
values = np.zeros((31, hierarchy.total_channels))
root_x = hierarchy.channel_index("Root", "Xposition")
values[:, root_x] = np.linspace(0, 1, 31)
generated = Motion(hierarchy, values, frame_time=1 / 30)
write_bvh(generated, "generated.bvh")
```

`Foot` has no channels and contributes no columns to the sample table. Its offset still participates in FK. The constructor copies `values`; editing that original array later does not change `generated`.
