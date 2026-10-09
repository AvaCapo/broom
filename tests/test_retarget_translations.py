import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Joint, Motion
from broom.kinematics import compute_global_positions
from broom.retargeting.retarget import retarget_mapped_motion

ROT = ('Xrotation', 'Yrotation', 'Zrotation')
POS = ('Xposition', 'Yposition', 'Zposition')


def hierarchy(angle=0, child_channels=POS, child_orientation=(1, 0, 0, 0)):
    orientation = Rotation.from_euler('Z', angle, degrees=True).as_quat()[[3, 0, 1, 2]]
    return Hierarchy((
        Joint('root', -1, (0, 0, 0), POS + ROT, local_orientation=orientation),
        Joint('child', 0, (0, 0, 1), child_channels + ROT,
              local_orientation=child_orientation),
    ))


def source_motion(h):
    values = np.zeros((3, h.total_channels))
    values[:, h.channel_index('child', 'Xposition')] = [0, 1, 2]
    values[:, h.channel_index('root', 'Xrotation')] = [0, 20, 40]
    return Motion(h, values, 1 / 30)


def transfer(motion, target, **kwargs):
    return retarget_mapped_motion(motion, target, {'root': 'root', 'child': 'child'}, **kwargs)


def test_parent_basis_conversion_preserves_fk_with_animated_parent():
    source = source_motion(hierarchy(child_channels=('Xposition',)))
    # Child's own orientation must not be used to convert its displacement.
    child_q = Rotation.from_euler('Y', 35, degrees=True).as_quat()[[3, 0, 1, 2]]
    target = hierarchy(90, ('Yposition',), tuple(child_q))
    before = source.values.copy()
    result = transfer(source, target, rotation_correction='local_orientation')
    np.testing.assert_allclose(compute_global_positions(source), compute_global_positions(result), atol=1e-12)
    np.testing.assert_allclose(result.values[:, target.channel_index('child', 'Yposition')], [0, -1, -2], atol=1e-12)
    np.testing.assert_array_equal(source.values, before)
    assert result.hierarchy is target


def test_scale_applies_to_transformed_displacement():
    source = source_motion(hierarchy(child_channels=('Xposition',)))
    target = hierarchy(90)
    result = transfer(source, target, rotation_correction='local_orientation', scale=2)
    np.testing.assert_allclose(result.values[:, target.channel_index('child', 'Yposition')], [0, -2, -4], atol=1e-12)


@pytest.mark.parametrize('channels', [(), ('Xposition',)])
def test_missing_target_component_raises(channels):
    with pytest.raises(ValueError, match="child.*Yposition"):
        transfer(source_motion(hierarchy()), hierarchy(90, channels), rotation_correction='local_orientation')


def test_zero_displacement_does_not_require_target_channels():
    source = source_motion(hierarchy())
    source.values[:, source.hierarchy.channel_index('child', 'Xposition')] = 0
    transfer(source, hierarchy(90, ()), rotation_correction='local_orientation')


@pytest.mark.parametrize('mode', ['none', 'rest_pose'])
def test_other_modes_keep_existing_translation_behavior(mode):
    source = source_motion(hierarchy())
    target = hierarchy(90)
    result = transfer(source, target, rotation_correction=mode)
    for axis in 'XYZ':
        np.testing.assert_array_equal(result.values[:, target.channel_index('child', axis + 'position')],
                                      source.values[:, source.hierarchy.channel_index('child', axis + 'position')])


def test_root_translation_keeps_first_frame_rebasing():
    source = source_motion(hierarchy())
    source.values[:, source.hierarchy.channel_index('root', 'Xposition')] = [4, 5, 7]
    target = hierarchy(90)
    result = transfer(source, target, rotation_correction='local_orientation', scale=2)
    np.testing.assert_allclose(result.values[:, target.channel_index('root', 'Xposition')], [0, 2, 6])


def test_root_to_nonroot_translation_is_rejected():
    source = source_motion(hierarchy())
    with pytest.raises(ValueError, match='root and non-root'):
        retarget_mapped_motion(source, hierarchy(), {'root': 'child'}, rotation_correction='local_orientation')


def test_parent_basis_includes_all_rest_ancestors():
    q45 = tuple(Rotation.from_euler('Z', 45, degrees=True).as_quat()[[3, 0, 1, 2]])
    def nested(q):
        return Hierarchy((
            Joint('root', -1, (0, 0, 0), ROT, local_orientation=q),
            Joint('parent', 0, (0, 0, 1), ROT, local_orientation=q),
            Joint('child', 1, (0, 0, 1), POS + ROT),
        ))
    source_h = nested((1, 0, 0, 0))
    target_h = nested(q45)
    values = np.zeros((1, source_h.total_channels))
    values[:, source_h.channel_index('child', 'Xposition')] = 1
    source = Motion(source_h, values, 1 / 30)
    result = retarget_mapped_motion(source, target_h,
                                   {name: name for name in source_h.joint_names},
                                   rotation_correction='local_orientation')
    np.testing.assert_allclose(compute_global_positions(source), compute_global_positions(result), atol=1e-12)
    np.testing.assert_allclose(result.values[:, target_h.channel_index('child', 'Yposition')], [-1], atol=1e-12)
