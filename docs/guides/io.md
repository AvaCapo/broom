# BVH input and output

```python
from broom.io import load_bvh, load_bvh_from_text, load_bvh_from_bytes, write_bvh

motion = load_bvh("walk.bvh")
print(motion.hierarchy.format_tree(show_channels=True))
write_bvh(motion, "copy.bvh", precision=8, frame_time_precision=8)
```

For in-memory data, use `load_bvh_from_text(text)` or `load_bvh_from_bytes(data, encoding="utf-8-sig")`. The optional `root_name` argument overrides the parsed root name.

## Behavior

- The loader returns a `Motion`; it does not normalize units, axes, root semantics, or floor placement.
- The writer emits canonical depth-first BVH and reorders channel blocks for export without modifying the input motion. Original whitespace is not preserved.
- End Sites are stored on their owning joints.
- BVH export currently rejects nonidentity `Joint.local_orientation`.
- Loading does not infer or attach rotation limits.

See [Coordinates and timing](../concepts/coordinates.md) for explicit unit and root-position conversion, and [I/O API](../api/io.md) for signatures.
