# Rotation representations

Motion channels use Euler degrees. The low-level `broom.rotations.quat` angle functions use radians and **wxyz** quaternions. Ortho6D stores the first two matrix columns as `(..., 3, 2)`; dual quaternions use `(..., 8)`, real part first.

```python
import numpy as np
from broom.rotations import quat, ortho6d

q = np.array([[1.0, 0.0, 0.0, 0.0]])
matrices = quat.to_matrix(q)
encoded = ortho6d.from_matrix(matrices)
restored = ortho6d.to_matrix(encoded)
np.testing.assert_allclose(restored, matrices)
```

::: broom.rotations.quat.from_matrix

::: broom.rotations.quat.to_matrix

::: broom.rotations.quat.from_euler

::: broom.rotations.quat.to_euler

::: broom.rotations.quat.slerp

::: broom.rotations.quat.unroll

::: broom.rotations.ortho6d.from_matrix

::: broom.rotations.ortho6d.to_matrix

::: broom.rotations.dual_quat.from_rotation_translation

::: broom.rotations.dual_quat.to_rotation_translation

