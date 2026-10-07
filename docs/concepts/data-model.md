# Data model

Broom follows the main BVH structure: a hierarchy of joints with rest offsets, an ordered list of channels for each joint, and a table of channel samples over time. It separates **immutable structure** from **mutable animation data**.

Changing a joint's parent or channel list underneath an existing sample table could change the meaning of its columns. Keeping `Joint` and `Hierarchy` immutable prevents that kind of accidental inconsistency. Motion samples remain editable NumPy arrays so numerical operations do not require rebuilding the skeleton.

| Object | Stores | Mutability |
|---|---|---|
| `Joint` | Name, parent index, offset, channels, rest orientation, optional End Site | Immutable |
| `Hierarchy` | Ordered joints and derived name/channel indices | Immutable |
| `Motion` | Hierarchy, scalar samples, frame time | Attributes are fixed; sample array is writable |

## Joint and hierarchy

A hierarchy has one root (`parent=-1`). Every parent precedes its children, and names are unique. A joint's channel order determines its columns in `Motion.values`. `local_orientation` is a unit rest-orientation quaternion in **wxyz** order.

A BVH End Site is stored as `Joint.end_site_offset`; it is not another joint and has no animation channels. A named terminal joint is still a normal joint.

```python
from broom.io import load_bvh

motion = load_bvh("walk.bvh")
hierarchy = motion.hierarchy
root = hierarchy.root
print(hierarchy.root_name)
print(hierarchy.children_names(root))
print(hierarchy.rotation_channel_indices(hierarchy.root_name))
```

To change geometry, create a new hierarchy and construct a new `Motion` with a compatible sample layout. Changing offsets does not automatically change translations.

## Motion values and array ownership

`Motion.values` has shape `(F, C)`: F is the number of frames (`F >= 1`), and C is `hierarchy.total_channels`. Columns follow joint order, then channel order. Rotations are scalar Euler angles in degrees, not quaternion rows. Joints without channels occupy no columns.

The constructor copies its input into a finite `float64` table. Once constructed, accessing `motion.values` exposes the owned array directly.

!!! warning "Arrays and basic slices are writable views"
    Assigning `values = motion.values` does not copy data. Basic NumPy slices such as `motion.values[:10]` also share storage. Writes through either change the motion immediately. `motion.frame(i)` is writable too. Use `.copy()` when you want independent data. NumPy advanced indexing has different copy rules; do not rely on an indexing expression to express ownership.

```python
root_x = hierarchy.channel_index(hierarchy.root_name, "Xposition")
values = motion.values
values[:, root_x] += 1.0  # Changes every root X sample in motion.

first_frames = motion.values[:10]
first_frames[:, root_x] -= 0.5  # Also changes motion, in these frames only.
assert first_frames[0, root_x] == motion.values[0, root_x]
```

For edits that must leave the original untouched:

```python
before = motion.values.copy()
working_values = motion.values.copy()
working_values[:, root_x] += 2.0

import numpy as np
np.testing.assert_array_equal(motion.values, before)
edited_motion = motion.with_values(working_values)
```

`with_values` copies the replacement samples and shares the immutable hierarchy. The returned motion is independent. Direct writes bypass constructor validation: keep values finite and do not change the array shape or its channel interpretation.

`frame_time` must be finite and positive; a one-frame motion represents a pose. See [Coordinates and timing](coordinates.md) for units and duration.

[Joint API](../api/joint.md) · [Hierarchy API](../api/hierarchy.md) · [Motion API](../api/motion.md)
