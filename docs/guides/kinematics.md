# Forward and inverse kinematics

## FK from motion

```python
from broom.io import load_bvh
from broom.kinematics import compute_global_transforms, compute_rest_joint_positions

motion = load_bvh("walk.bvh")
positions, rotations = compute_global_transforms(motion)
rest = compute_rest_joint_positions(motion.hierarchy)
```

Here **F is the number of frames** and **J is the number of joints**. Positions have shape `(F, J, 3)`, rotations `(F, J, 3, 3)`, and rest positions `(J, 3)`. Joint order follows `Hierarchy.joints`. End Sites are not extra rows. Use `compute_global_positions(motion)` when only positions are needed.

## FK from local animation deltas

```python
import numpy as np
from broom.kinematics import compute_global_transforms_from_local

hierarchy = motion.hierarchy
q = np.broadcast_to(np.eye(3), (1, hierarchy.joint_count, 3, 3)).copy()
d = np.zeros((1, hierarchy.joint_count, 3))
positions, rotations = compute_global_transforms_from_local(
    hierarchy, local_rotations=q, local_translations=d,
)
```

The function applies hierarchy offsets and rest orientations itself. Do not add them to the inputs. Supplied arrays cover every joint; use identity/zero for unanimated joints. Either array may be omitted; omitting both returns one rest frame. Rotations must be proper rotation matrices; no implicit repair is done.

## FABRIK

The NumPy point solver works without a `Motion`:

```python
from broom.ik.fabrik import solve_fabrik

chain = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
solved = solve_fabrik(chain, np.array([1., 1., 0.]), threshold=1e-3)
```

For animation, `solve_fabrik_bvh_frame` returns one updated channel row; `solve_fabrik_bvh_clip` returns a `Motion`. Supply a contiguous ancestor-to-child chain and world-space targets `(F, 3)` for a clip. Despite their historical names, these functions accept the current `Motion` class.

Use the Motion-writing FABRIK path with identity rest orientations: conversion back to channels does not yet account for nonidentity `local_orientation`. FABRIK here does not automatically apply joint limits.

[Kinematics API](../api/kinematics.md)
