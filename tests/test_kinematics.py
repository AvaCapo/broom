import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Joint, Motion
from broom.kinematics import (
    compute_global_transforms,
    compute_global_transforms_from_local,
)


def _identity_rotations(frames: int, joints: int) -> np.ndarray:
    return np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy()


def test_low_level_fk_applies_offsets_without_mutating_inputs():
    hierarchy = Hierarchy(
        (
            Joint("Root", -1, (1, 0, 0), "Xposition"),
            Joint("Child", 0, (1, 0, 0)),
        )
    )
    rotations = _identity_rotations(2, 2)
    rotations[1, 0] = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    translations = np.array(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    rotations_before = rotations.copy()
    translations_before = translations.copy()

    positions, world_rotations = compute_global_transforms_from_local(
        hierarchy, rotations, translations
    )

    np.testing.assert_allclose(positions, [[[2, 0, 0], [5, 0, 0]], [[4, 0, 0], [4, 3, 0]]])
    np.testing.assert_allclose(world_rotations[1, 1], rotations[1, 0])
    np.testing.assert_allclose(rotations, rotations_before)
    np.testing.assert_allclose(translations, translations_before)


def test_motion_fk_handles_partial_channels_and_non_root_translation():
    hierarchy = Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), "Zrotation"),
            Joint("Child", 0, (1, 0, 0), "Xposition"),
        )
    )
    motion = Motion(hierarchy, np.array([[90.0, 2.0]]), 1 / 30)

    positions, rotations = compute_global_transforms(motion)

    np.testing.assert_allclose(positions, [[[0, 0, 0], [0, 3, 0]]], atol=1e-12)
    np.testing.assert_allclose(
        rotations[0, 1], Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    )


def test_motion_fk_composes_rest_orientation_before_channel_rotation():
    rest_orientation = Rotation.from_euler("X", 90, degrees=True).as_quat()
    hierarchy = Hierarchy(
        (
            Joint(
                "Root",
                -1,
                (0, 0, 0),
                "Zrotation",
                local_orientation=(
                    rest_orientation[3],
                    rest_orientation[0],
                    rest_orientation[1],
                    rest_orientation[2],
                ),
            ),
            Joint("Child", 0, (1, 0, 0)),
        )
    )
    motion = Motion(hierarchy, np.array([[90.0]]), 1 / 30)

    positions, rotations = compute_global_transforms(motion)
    expected = Rotation.from_euler("X", 90, degrees=True).as_matrix() @ Rotation.from_euler(
        "Z", 90, degrees=True
    ).as_matrix()

    np.testing.assert_allclose(rotations[0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(positions[0, 1], expected @ [1, 0, 0], atol=1e-12)


def test_child_rest_orientation_does_not_rotate_its_own_offset_but_rotates_descendant():
    child_orientation = Rotation.from_euler("Z", 90, degrees=True).as_quat()
    hierarchy = Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), "Xposition"),
            Joint(
                "Child",
                0,
                (1, 0, 0),
                local_orientation=(
                    child_orientation[3],
                    child_orientation[0],
                    child_orientation[1],
                    child_orientation[2],
                ),
            ),
            Joint("Grandchild", 1, (1, 0, 0)),
        )
    )
    motion = Motion(hierarchy, np.zeros((1, 1)), 1 / 30)

    positions, _ = compute_global_transforms(motion)

    np.testing.assert_allclose(positions[0], [[0, 0, 0], [1, 0, 0], [1, 1, 0]])


@pytest.mark.parametrize("order", ("X", "XY", "ZXY", "ZYX"))
def test_motion_fk_uses_declared_euler_order(order):
    channels = tuple(f"{axis}rotation" for axis in order)
    hierarchy = Hierarchy((Joint("Root", -1, (0, 0, 0), channels),))
    angles = np.arange(1, len(order) + 1, dtype=float) * 15
    motion = Motion(hierarchy, angles[None, :], 1 / 30)

    _, rotations = compute_global_transforms(motion)

    np.testing.assert_allclose(
        rotations[0, 0],
        Rotation.from_euler(
            order, angles[0] if len(order) == 1 else angles, degrees=True
        ).as_matrix(),
    )


@pytest.mark.parametrize(
    "rotations, translations, message",
    (
        (np.empty((0, 1, 3, 3)), np.empty((0, 1, 3)), "non-empty"),
        (_identity_rotations(1, 1), np.zeros((2, 1, 3)), "same non-empty"),
        (np.full((1, 1, 3, 3), np.nan), np.zeros((1, 1, 3)), "finite"),
        (np.ones((1, 1, 3, 3)), np.zeros((1, 1, 3)), "orthonormal"),
        (
            np.array([[[[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]]]]),
            np.zeros((1, 1, 3)),
            "determinant",
        ),
    ),
)
def test_low_level_fk_rejects_invalid_local_transforms(rotations, translations, message):
    hierarchy = Hierarchy((Joint("Root", -1, (0, 0, 0), "Xposition"),))

    with pytest.raises(ValueError, match=message):
        compute_global_transforms_from_local(hierarchy, rotations, translations)


def test_motion_fk_rejects_nonfinite_values_after_writable_mutation():
    hierarchy = Hierarchy((Joint("Root", -1, (0, 0, 0), "Xposition"),))
    motion = Motion(hierarchy, np.zeros((1, 1)), 1 / 30)
    motion.values[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_global_transforms(motion)


def _rest_pose_hierarchy():
    half = np.sqrt(0.5)
    return Hierarchy(
        (
            Joint("Root", -1, (1, 2, 3), ("Xposition", "Zrotation"),
                  local_orientation=(half, half, 0, 0)),
            Joint("Child", 0, (2, 0, 0),
                  ("Yposition", "Xposition", "Xrotation"),
                  local_orientation=(half, 0, 0, half)),
            Joint("Tip", 1, (0, 1, 0)),
        )
    )


def test_local_deltas_apply_rest_pose_in_order_and_translations_in_parent_axes():
    hierarchy = _rest_pose_hierarchy()
    rx = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    rz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    inputs = np.array([[rz, rx, np.eye(3)]])
    deltas = np.array([[[1., 0, 0], [1, 2, 0], [0, 0, 0]]])
    inputs_before, deltas_before = inputs.copy(), deltas.copy()

    positions, rotations = compute_global_transforms_from_local(hierarchy, inputs, deltas)

    # Root O @ R is Rx @ Rz, not Rz @ Rx. Its own delta stays on world X.
    expected_root = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])
    expected_child = np.diag([-1., -1, 1])
    np.testing.assert_allclose(positions, [[[2, 2, 3], [0, 2, 6], [0, 1, 6]]], atol=1e-12)
    np.testing.assert_allclose(rotations[0], [expected_root, expected_child, expected_child], atol=1e-12)
    np.testing.assert_array_equal(inputs, inputs_before)
    np.testing.assert_array_equal(deltas, deltas_before)
    assert not np.shares_memory(positions, deltas)
    assert not np.shares_memory(rotations, inputs)


@pytest.mark.parametrize("explicit", (False, True))
def test_identity_animation_recovers_rest_pose(explicit):
    hierarchy = _rest_pose_hierarchy()
    args = (_identity_rotations(1, 3), np.zeros((1, 3, 3))) if explicit else ()
    positions, rotations = compute_global_transforms_from_local(hierarchy, *args)
    rx = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    child_rest = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])
    np.testing.assert_allclose(positions, [[[1, 2, 3], [3, 2, 3], [2, 2, 3]]], atol=1e-12)
    np.testing.assert_allclose(rotations[0], [rx, child_rest, child_rest], atol=1e-12)


@pytest.mark.parametrize("omit", ("rotations", "translations"))
def test_missing_local_component_uses_other_components_frame_count(omit):
    hierarchy = _rest_pose_hierarchy()
    rotations = _identity_rotations(2, 3)
    rotations[1, 0] = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    translations = np.ones((2, 3, 3))
    if omit == "rotations":
        actual = compute_global_transforms_from_local(hierarchy, local_translations=translations)
        expected = compute_global_transforms_from_local(hierarchy, _identity_rotations(2, 3), translations)
    else:
        actual = compute_global_transforms_from_local(hierarchy, rotations)
        expected = compute_global_transforms_from_local(hierarchy, rotations, np.zeros((2, 3, 3)))
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference)
        assert result.shape[0] == 2


def test_motion_and_local_deltas_apply_rest_pose_exactly_once():
    hierarchy = _rest_pose_hierarchy()
    motion = Motion(hierarchy, [[1, 90, 2, 1, 90], [0, 0, 0, 0, 0]], 1 / 30)
    rotations = _identity_rotations(2, 3)
    rotations[0, 0] = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    rotations[0, 1] = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    translations = np.zeros((2, 3, 3))
    translations[0, 0] = (1, 0, 0)
    translations[0, 1] = (1, 2, 0)
    values_before = motion.values.copy()

    positions, world_rotations = compute_global_transforms(motion)
    reference = compute_global_transforms_from_local(hierarchy, rotations, translations)

    np.testing.assert_allclose(positions, reference[0], atol=1e-12)
    np.testing.assert_allclose(world_rotations, reference[1], atol=1e-12)
    np.testing.assert_allclose(positions, [
        [[2, 2, 3], [0, 2, 6], [0, 1, 6]],
        [[1, 2, 3], [3, 2, 3], [2, 2, 3]],
    ], atol=1e-12)
    np.testing.assert_array_equal(motion.values, values_before)


@pytest.mark.parametrize("rotations, translations, message", (
    (np.empty((0, 3, 3, 3)), None, "non-empty"),
    (None, np.empty((0, 3, 3)), "non-empty"),
    (np.eye(3), None, "shape"),
    (None, np.zeros((1, 2, 3)), "shape"),
    (None, np.full((1, 3, 3), np.inf), "finite"),
))
def test_optional_local_inputs_still_validate_supplied_arrays(rotations, translations, message):
    with pytest.raises(ValueError, match=message):
        compute_global_transforms_from_local(_rest_pose_hierarchy(), rotations, translations)
