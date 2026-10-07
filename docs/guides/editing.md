# Edit skeletons and motion

The examples use `motion` loaded with `broom.io.load_bvh`.

## Edit samples

```python
from broom.ops.motion_editing import trim_frames, reverse, repeat_pose

clip = trim_frames(motion, start=0, stop=30)  # Python slice semantics; must be nonempty.
backward = reverse(clip)
held = repeat_pose(motion.hierarchy, motion.frame(0), frame_count=60, frame_time=1 / 30)

values = motion.values.copy()
root = motion.hierarchy.root_name
column = motion.hierarchy.channel_index(root, "Xposition")
values[:, column] -= values[0, column]
centered = motion.with_values(values)
```

Query channel indices through `Hierarchy`; do not assume that the first three columns are root XYZ. Missing channels raise `KeyError`.

## Scale units or change proportions

```python
from broom import Motion
from broom.ops.motion_editing import scale_skeleton
from broom.ops.skeleton_editing import scale_offsets, scale_joint_offset

smaller = scale_skeleton(motion, 0.5)  # Offsets AND all position channels.
target_hierarchy = scale_offsets(motion.hierarchy, 0.5)  # Offsets only.
# Replace the name with a joint from your hierarchy:
# target_hierarchy = scale_joint_offset(target_hierarchy, "RightFoot", 0.8)
fk_baseline = Motion(target_hierarchy, motion.values, motion.frame_time)
```

`scale_joint_offset` changes the link **entering** the named non-root joint. It does not scale the whole subtree. `scale_offsets` also scales End Site offsets. Neither hierarchy operation modifies motion samples. If a joint has translation channels, its effective displacement remains `offset + d`; changing the offset alone does not scale `d`.

A direct copy of channels onto another hierarchy requires matching layouts. Use [retargeting](retargeting.md) when joints or channels differ.
