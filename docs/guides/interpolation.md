# Interpolation and resampling

## Change FPS

```python
from broom.io import load_bvh
from broom.ops.resample import resample_fps

motion = load_bvh("walk.bvh")
resampled = resample_fps(motion, target_fps=60.0, endpoint_policy="preserve")
```

| Policy | Behavior |
|---|---|
| `preserve` | Keeps first/last poses; picks the nearest frame count. Duration may change slightly. |
| `truncate` | Samples on the exact target-FPS time grid; drops a final partial interval. |

Both record exactly `frame_time = 1 / target_fps`. Translations use linear interpolation. Three-axis rotations use declared-order SLERP; one-axis rotations are unwrapped along the shortest adjacent path; two-axis groups use linear channel interpolation.

## Blend two compatible clips

```python
from broom.interpolation.utils import interpolate_motion

# first and second are Motion objects with identical hierarchies and frame_time.
# Each must contain at least 10 frames.
blended = interpolate_motion(first, second, transition_frames=10)
```

The transition overlaps the end of the first clip with the start of the second. Output length is `F1 + F2 - transition_frames`. The second clip's root translation is aligned to the first clip's final root position by default; pass `align_root_translation=False` to preserve its position values. A zero-length transition concatenates clips, still applying the selected root alignment.

This does not match different skeletons or infer a transition duration. [Operations API](../api/operations.md)
