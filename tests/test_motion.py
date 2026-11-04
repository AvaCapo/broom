import numpy as np
import pytest
from broom import Hierarchy, Joint, Motion


@pytest.fixture
def hierarchy():
    return Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), ("Xposition", "Zrotation")),
            Joint("Child", 0, (0, 1, 0), "Yrotation"),
        )
    )


def test_motion_owns_input_but_supports_numeric_assignment(hierarchy):
    source = np.zeros((2, 3))
    motion = Motion(hierarchy, source, 1 / 30)
    source[0, 0] = 4
    assert motion.values[0, 0] == 0
    assert motion.values.flags.writeable
    motion.values[0, 0] = 2
    assert motion.frame(0)[0] == 2


def test_with_values_is_independent(hierarchy):
    source = Motion(hierarchy, np.zeros((1, 3)), 0.1)
    result = source.with_values(np.ones((2, 3)))
    result.values[0, 0] = 9
    assert source.values[0, 0] == 0 and result.frame_count == 2


def test_shape_timing_and_finite_policy(hierarchy):
    with pytest.raises(ValueError):
        Motion(hierarchy, np.zeros((1, 4)), 0.1)
    with pytest.raises(ValueError):
        Motion(hierarchy, np.zeros((1, 3)), 0)
    with pytest.raises(ValueError):
        Motion(hierarchy, np.zeros((1, 3)), None)
    with pytest.raises(ValueError, match="finite"):
        Motion(hierarchy, np.array([[np.nan, 0.0, 0.0]]), 0.1)
    with pytest.raises(ValueError, match="finite"):
        Motion(hierarchy, np.array([[np.inf, 0.0, 0.0]]), 0.1)
    with pytest.raises(ValueError):
        Motion(hierarchy, np.empty((0, 3)), 0.1)
    with pytest.raises(ValueError):
        Motion(Hierarchy((Joint("Root", -1, (0, 0, 0)),)), np.empty((1, 0)), 0.1)
