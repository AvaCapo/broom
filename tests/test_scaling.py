import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Joint, Motion
from broom.io import load_bvh_from_text, write_bvh
from broom.kinematics import compute_global_positions, compute_global_transforms
from broom.ops.motion_editing import scale_skeleton as scale_motion_skeleton
from broom.ops.skeleton_editing import scale_offsets


def _hierarchy() -> Hierarchy:
    orientation = Rotation.from_euler("Z", 90, degrees=True).as_quat()
    return Hierarchy(
        (
            Joint(
                "Root",
                -1,
                (1.0, 0.0, 0.0),
                ("Xposition", "Zrotation"),
            ),
            Joint(
                "Child",
                0,
                (0.0, 2.0, 0.0),
                "Yposition",
                local_orientation=(
                    orientation[3],
                    orientation[0],
                    orientation[1],
                    orientation[2],
                ),
                end_site_offset=(0.0, 0.5, 0.0),
            ),
        )
    )


def test_scale_offsets_scales_rest_offsets_and_end_sites_only():
    hierarchy = _hierarchy()

    scaled = scale_offsets(hierarchy, 2.5)

    assert scaled is not hierarchy
    assert scaled.joints[0].offset == (2.5, 0.0, 0.0)
    assert scaled.joints[1].offset == (0.0, 5.0, 0.0)
    assert scaled.joints[1].end_site_offset == (0.0, 1.25, 0.0)
    assert scaled.joints[1].channels == hierarchy.joints[1].channels
    assert scaled.joints[1].local_orientation == hierarchy.joints[1].local_orientation


@pytest.mark.parametrize("factor", (0.0, -1.0, np.nan, np.inf, True))
def test_scale_offsets_rejects_non_positive_or_non_finite_factor(factor):
    with pytest.raises(ValueError, match="finite positive"):
        scale_offsets(_hierarchy(), factor)


def test_scale_motion_scales_geometry_and_all_translation_channels():
    motion = Motion(
        _hierarchy(),
        np.array([[2.0, 45.0, 3.0], [-1.0, 90.0, 4.0]]),
        frame_time=1 / 24,
    )
    positions, rotations = compute_global_transforms(motion)

    scaled = scale_motion_skeleton(motion, 3.0)
    scaled_positions, scaled_rotations = compute_global_transforms(scaled)

    np.testing.assert_allclose(scaled.values, [[6.0, 45.0, 9.0], [-3.0, 90.0, 12.0]])
    np.testing.assert_allclose(scaled_positions, positions * 3.0)
    np.testing.assert_allclose(scaled_rotations, rotations)
    assert scaled.frame_time == motion.frame_time
    assert not np.shares_memory(scaled.values, motion.values)

    np.testing.assert_allclose(compute_global_positions(scaled), scaled_positions)


def test_scaled_motion_round_trips_through_bvh_with_offsets_and_end_sites(tmp_path):
    motion = Motion(
        Hierarchy(
            (
                Joint("Root", -1, (1.0, 0.0, 0.0), "Xposition"),
                Joint(
                    "Child",
                    0,
                    (0.0, 2.0, 0.0),
                    "Yposition",
                    end_site_offset=(0.0, 0.5, 0.0),
                ),
            )
        ),
        np.array([[2.0, 3.0], [-1.0, 4.0]]),
        frame_time=1 / 30,
    )

    scaled = scale_motion_skeleton(motion, 2.0)
    output_path = tmp_path / "scaled.bvh"
    write_bvh(scaled, output_path, precision=10)
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert reloaded.hierarchy == scaled.hierarchy
    assert reloaded.frame_time == pytest.approx(scaled.frame_time, abs=1e-8)
    np.testing.assert_allclose(reloaded.values, scaled.values)
