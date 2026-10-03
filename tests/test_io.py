import numpy as np
import pytest

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
    assert reloaded.frame_time == pytest.approx(motion.frame_time, abs=1e-8)
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


def test_writer_preserves_joint_values_when_canonicalising_to_dfs(tmp_path):
    hierarchy = Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), "Xposition"),
            Joint("A", 0, (0, 1, 0), ("Xrotation", "Yrotation")),
            Joint("B", 0, (1, 0, 0), "Zrotation"),
            Joint("C", 1, (0, 1, 0), ("Xposition", "Yposition", "Zposition")),
        )
    )
    motion = Motion(hierarchy, np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float), 1 / 30)

    output_path = tmp_path / "dfs_order.bvh"
    write_bvh(motion, output_path)
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert reloaded.hierarchy.joint_names == ("Root", "A", "C", "B")
    expected_by_name = {
        joint.name: motion.values[
            :, hierarchy.channel_start(index) : hierarchy.channel_start(index)
            + hierarchy.channel_count(index)
        ]
        for index, joint in enumerate(hierarchy.joints)
    }
    for index, joint in enumerate(reloaded.hierarchy.joints):
        start = reloaded.hierarchy.channel_start(index)
        stop = start + reloaded.hierarchy.channel_count(index)
        np.testing.assert_allclose(reloaded.values[:, start:stop], expected_by_name[joint.name])
    np.testing.assert_allclose(motion.values, [[1, 2, 3, 4, 5, 6, 7]])


@pytest.mark.parametrize("precision", (0, 1))
@pytest.mark.parametrize("frame_time", (1 / 24, 1 / 30, 1 / 60, 1e-9))
def test_frame_time_uses_separate_safe_precision(tmp_path, precision, frame_time):
    motion = Motion(
        Hierarchy((Joint("Root", -1, (0, 0, 0), "Xposition"),)),
        np.zeros((1, 1)),
        frame_time,
    )

    output_path = tmp_path / "frame_time.bvh"
    write_bvh(motion, output_path, precision=precision)
    reloaded = load_bvh_from_text(output_path.read_text(encoding="utf-8"))

    assert reloaded.frame_time > 0
    assert reloaded.frame_time == pytest.approx(frame_time, rel=0, abs=1e-8)


def test_parser_accepts_signed_and_exponent_numbers():
    text = _BVH.replace("OFFSET 0 0 0", "OFFSET +1 1e+00 1e-03").replace(
        "Frame Time: 0.0333333333", "Frame Time: 1e-03"
    )

    motion = load_bvh_from_text(text)

    assert motion.hierarchy.root_joint.offset == (1.0, 1.0, 0.001)
    assert motion.frame_time == 0.001


@pytest.mark.parametrize(
    ("text", "message"),
    (
        (_BVH.replace("CHANNELS 1 Xrotation", "CHANNELS banana"), "CHANNELS"),
        (_BVH.replace("Frames: 2\n", ""), "Frames"),
        (_BVH.replace("        End Site\n", "        UNKNOWN thing\n"), "Unknown"),
        (_BVH.replace("        OFFSET 0 -0.2 0.1\n", "        JOINT Illegal\n"), "JOINT"),
        (_BVH.replace("        CHANNELS 1 Xrotation\n", ""), "without CHANNELS"),
    ),
)
def test_parser_rejects_malformed_structure(text, message):
    with pytest.raises(ValueError, match=message):
        load_bvh_from_text(text)


def test_channel_less_joint_uses_explicit_channels_zero(tmp_path):
    motion = Motion(
        Hierarchy(
            (
                Joint("Root", -1, (0, 0, 0), "Xposition"),
                Joint("Fixed", 0, (0, 1, 0)),
                Joint("Tip", 1, (0, 1, 0), "Yrotation"),
            )
        ),
        np.array([[1.0, 2.0]]),
        1 / 30,
    )

    output_path = tmp_path / "channel_less.bvh"
    write_bvh(motion, output_path)
    text = output_path.read_text(encoding="utf-8")
    reloaded = load_bvh_from_text(text)

    assert "CHANNELS 0" in text
    assert reloaded.hierarchy.joints[1].channels == ()
    np.testing.assert_allclose(reloaded.values, motion.values)


def test_writer_rejects_nonfinite_values_after_mutation(tmp_path):
    motion = load_bvh_from_text(_BVH)
    motion.values[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        write_bvh(motion, tmp_path / "nonfinite.bvh")
