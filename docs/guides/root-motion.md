# Root motion and floor handling

These are different operations:

| Operation | Effect |
|---|---|
| Root representation conversion | Add/subtract the rest root offset from position samples. |
| Floor alignment | Shift the whole clip by one constant vertical translation. |
| In-place conversion | Remove a smoothed locomotion trend from root XZ. |
| Space-time constraints | Optimize motion over selected frames. |

## Inspect an unfamiliar clip's floor level

When a BVH's vertical placement is uncertain, estimate the lowest level reached by its foot joints before deciding whether to move the animation:

```python
from broom.io import load_bvh
from broom.retargeting.root_motion import estimate_floor_level, align_root_to_floor

motion = load_bvh("walk.bvh")  # Assume meter-scale, Y-up coordinates here.
floor_y = estimate_floor_level(motion, up_axis="Y", use_rest_pose=False)
print(f"Lowest selected foot level: {floor_y:.4f} m")
print(f"Constant shift needed to place that level at Y=0: {-floor_y:.4f} m")
```

The estimate uses the minimum Y over the selected frames and joints whose names contain foot/toe/ankle/heel. If no names match, it uses all joints. A positive value means those samples stay above Y=0; a negative value means some lie below it. This is a geometric diagnostic, not proof of where the physical floor is: airborne clips and incorrect joint names can give a misleading estimate.

If the data should be grounded at Y=0, apply the separate alignment operation:

```python
aligned = align_root_to_floor(motion, floor_height=0.0, up_axis="Y", use_rest_pose=False)
print("After alignment:", estimate_floor_level(aligned, use_rest_pose=False))
```

`align_root_to_floor` computes the estimate itself and shifts the entire clip by one constant vertical translation. It does not remove foot sliding or lock contacts. Keep the source unchanged when its world placement is meaningful.

`use_rest_pose=True` also considers a zero-rotation pose using the first frame's translations. `first_frame_only=True` limits animation sampling to the first frame. Use the same settings when comparing the estimate before and after alignment.

## In-place conversion

```python
from broom.in_place.config import InPlacePCAConfig
from broom.in_place.inplace_transform import InPlaceConverter

converter = InPlaceConverter.load(InPlacePCAConfig(root_name=None, source="root"))
root_xz, diagnostics = converter.transform_motion(motion, foot_lock=False)
values = motion.values.copy()
root_name = motion.hierarchy.root_name
columns = [motion.hierarchy.channel_index(root_name, f"{axis}position") for axis in "XZ"]
values[:, columns] = root_xz
in_place = motion.with_values(values)
```

`transform_motion` returns root coordinates and diagnostics, not a `Motion`. `transform_file(input_path, output_path, foot_lock=False)` also writes BVH. Foot locking is opt-in and uses heuristic contacts. It differs from the [space-time stationary constraint](../spacetime_usage.md).
