import numpy as np

from broom import Hierarchy, Joint, Motion
from broom.io import load_bvh_from_text, write_bvh


_BVH = """HIERARCHY
ROOT Root
{
    OFFSET 0 0 0
    CHANNELS 3 Xposition Zrotation Yrotation
    JOINT Foot
    {
        OFFSET 0 -1 0
        CHANNELS 1 Xrotation
        End Site
        {
            OFFSET 0 -0.2 0.1
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.0333333333
1 20 30 40
2 50 60 70
"""


def test_read_and_write_bvh_round_trip_preserves_motion_and_end_site(tmp_path):
    motion = load_bvh_from_text(_BVH)

    assert motion.hierarchy.joint_names == ("Root", "Foot")
    assert motion.hierarchy.joints[0].channels == (
        "Xposition",
        "Zrotation",
        "Yrotation",
    )
    assert motion.hierarchy.joints[1].end_site_offset == (0.0, -0.2, 0.1)
    assert motion.frame_time == 0.0333333333

    output_path = tmp_path / "round_trip.bvh"
    write_bvh(motion, output_path, precision=10)
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert reloaded.hierarchy == motion.hierarchy
    assert reloaded.frame_time == motion.frame_time
    np.testing.assert_allclose(reloaded.values, motion.values)


def test_writer_adds_zero_end_site_for_a_leaf_without_one(tmp_path, capsys):
    motion = Motion(
        Hierarchy((Joint("Root", -1, (0, 0, 0), "Xposition"),)),
        np.zeros((1, 1)),
        1 / 30,
    )

    output_path = tmp_path / "zero_end_site.bvh"
    write_bvh(motion, output_path)
    assert "Added zero-offset End Site for leaf joint 'Root'" in capsys.readouterr().out
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert reloaded.hierarchy.root_joint.end_site_offset == (0.0, 0.0, 0.0)


def test_round_trip_preserves_partial_channel_order(tmp_path):
    motion = Motion(
        Hierarchy(
            (
                Joint("Root", -1, (0, 0, 0), ("Zrotation", "Xposition")),
                Joint("Child", 0, (0, 1, 0), "Yrotation"),
            )
        ),
        np.array([[10.0, 1.0, 20.0]]),
        1 / 30,
    )

    output_path = tmp_path / "partial_channels.bvh"
    write_bvh(motion, output_path)
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert tuple(joint.channels for joint in reloaded.hierarchy.joints) == (
        ("Zrotation", "Xposition"),
        ("Yrotation",),
    )
    np.testing.assert_allclose(reloaded.values, motion.values)


def test_rejects_offset_with_more_than_three_components():
    malformed = _BVH.replace("OFFSET 0 -1 0", "OFFSET 0 -1 0 0")

    with np.testing.assert_raises_regex(ValueError, "exactly three"):
        load_bvh_from_text(malformed)
