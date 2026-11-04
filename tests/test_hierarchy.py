import pytest
from broom import Hierarchy, Joint


def test_scalar_channels_preserve_order_and_derived_starts():
    h = Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), ("Zrotation", "Xposition")),
            Joint("Child", 0, (0, 1, 0), "Yrotation"),
            Joint("Tip", 1, (0, 1, 0)),
        )
    )
    assert h.total_channels == 3
    assert [h.channel_start(i) for i in range(3)] == [0, 2, 3]
    assert [h.channel_count(i) for i in range(3)] == [2, 1, 0]
    assert h.joints[0].channels == ("Zrotation", "Xposition")
    assert h.joint_names == ("Root", "Child", "Tip")
    assert h.root == 0
    assert h.root_name == "Root"
    assert h.root_joint is h.joints[0]
    assert h.joint_name(1) == "Child"
    assert h.parent(1) == 0
    assert h.parent_index("Child") == 0
    assert h.parent_name(1) == "Root"
    assert h.parent_joint(0) is None
    assert h.children(0) == (1,)
    assert h.children_indices("Root") == (1,)
    assert h.children_names(0) == ("Child",)
    assert h.children_joints(0) == (h.joints[1],)


def test_end_site_offset_is_part_of_its_parent_joint_only():
    hierarchy = Hierarchy(
        (
            Joint("Root", -1, (0, 0, 0), "Xposition"),
            Joint("Foot", 0, (0, -1, 0), end_site_offset=(0, -0.2, 0)),
        )
    )

    foot = hierarchy.joints[1]
    assert foot.end_site_offset == (0.0, -0.2, 0.0)
    assert hierarchy.joint_count == 2
    assert hierarchy.total_channels == 1
    assert hierarchy.children(1) == ()


@pytest.mark.parametrize("offset", [(), (0, 1), (0, 1, float("nan"))])
def test_rejects_invalid_end_site_offset(offset):
    with pytest.raises(ValueError):
        Joint("Root", -1, (0, 0, 0), end_site_offset=offset)


@pytest.mark.parametrize("index", [-1, 3, True])
def test_index_helpers_reject_invalid_indices(index):
    h = Hierarchy((Joint("Root", -1, (0, 0, 0)),))
    with pytest.raises(IndexError):
        h.joint_name(index)


@pytest.mark.parametrize(
    "channels", [("Xrotation", "Xrotation"), ("scale",), {"Xrotation"}]
)
def test_rejects_invalid_scalar_channels(channels):
    with pytest.raises(ValueError):
        Joint("Root", -1, (0, 0, 0), channels)


@pytest.mark.parametrize(
    "joints",
    [
        (),
        (Joint("Root", -1, (0, 0, 0)), Joint("Root", 0, (0, 0, 0))),
        (Joint("Root", -1, (0, 0, 0)), Joint("Child", 2, (0, 0, 0))),
    ],
)
def test_rejects_invalid_topology(joints):
    with pytest.raises(ValueError):
        Hierarchy(joints)
