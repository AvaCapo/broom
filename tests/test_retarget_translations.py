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
def test_missing_target_component_is_discarded(channels):
    result = transfer(source_motion(hierarchy()), hierarchy(90, channels), rotation_correction='local_orientation')
    for axis in channels:
        np.testing.assert_allclose(result.values[:, result.hierarchy.channel_index('child', axis)], 0, atol=1e-12)


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


@pytest.mark.parametrize("mode", ["none", "rest_pose", "local_orientation"])
def test_root_translation_preserves_first_sample(mode):
    source = source_motion(hierarchy())
    source.values[:, source.hierarchy.channel_index('root', 'Xposition')] = [4, 5, 7]
    target = hierarchy(90)
    result = transfer(source, target, rotation_correction=mode, scale=2)
    np.testing.assert_allclose(result.values[:, target.channel_index('root', 'Xposition')], [8, 10, 14])


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


@pytest.mark.parametrize('source_has_parent', [False, True])
@pytest.mark.parametrize('target_has_parent', [False, True])
def test_all_parent_combinations_preserve_rest_space_displacement(source_has_parent, target_has_parent):
    def skeleton(has_parent, angle):
        # Different joint orientations must not affect its own displacement.
        joint_q = tuple(Rotation.from_euler('Y', angle / 2, degrees=True).as_quat()[[3, 0, 1, 2]])
        joints = []
        if has_parent:
            parent_q = tuple(Rotation.from_euler('Z', angle, degrees=True).as_quat()[[3, 0, 1, 2]])
            joints.append(Joint('parent', -1, (0, 0, 0), POS + ROT, local_orientation=parent_q))
        joints.append(Joint('hips', 0 if has_parent else -1, (0, 0, 0), POS + ROT,
                            local_orientation=joint_q))
        return Hierarchy(tuple(joints))
    sh = skeleton(source_has_parent, 30)
    th = skeleton(target_has_parent, 70)
    values = np.zeros((3, sh.total_channels))
    displacement = np.array([[1, 2, 3], [2, 3, 4], [-1, 4, 2]])
    for axis, samples in zip('XYZ', displacement.T):
        values[:, sh.channel_index('hips', axis + 'position')] = samples
    source = Motion(sh, values, 1 / 30)
    output = retarget_mapped_motion(source, th, {'hips': 'hips'}, scale=2)
    np.testing.assert_allclose(
        compute_global_positions(output)[:, th.joint_index('hips')],
        2 * compute_global_positions(source)[:, sh.joint_index('hips')], atol=1e-12,
    )
    if target_has_parent:
        np.testing.assert_array_equal(output.values[:, :6], 0)
    np.testing.assert_array_equal(source.values, values)


def test_jump_moves_hips_under_stationary_root():
    sh = Hierarchy((Joint('hips', -1, (0, 0, 0), POS + ROT),))
    th = Hierarchy((Joint('root', -1, (0, 0, 0), POS + ROT),
                    Joint('hips', 0, (0, 0, 0), POS + ROT)))
    source = Motion(sh, np.zeros((3, 6)), 1 / 30)
    source.values[:, sh.channel_index('hips', 'Yposition')] = [1, 1.5, 1]
    output = retarget_mapped_motion(source, th, {'hips': 'hips'})
    positions = compute_global_positions(output)
    np.testing.assert_array_equal(positions[:, 0], 0)
    np.testing.assert_allclose(positions[:, 1, 1], [1, 1.5, 1])


def test_identity_transfer_preserves_full_motion_and_first_frame():
    h = hierarchy(37)
    values = np.random.default_rng(15).uniform(-40, 40, (8, h.total_channels))
    source = Motion(h, values, 1 / 30)
    output = transfer(source, h)
    np.testing.assert_allclose(output.values, values, atol=1e-12)
    np.testing.assert_allclose(compute_global_positions(output), compute_global_positions(source), atol=1e-12)
    assert not np.shares_memory(output.values, source.values)


@pytest.mark.parametrize('mode', ['none', 'rest_pose', 'local_orientation'])
def test_missing_root_channels_do_not_raise(mode):
    sh = Hierarchy((Joint('root', -1, (0, 0, 0), POS + ROT),))
    th = Hierarchy((Joint('root', -1, (0, 0, 0), ('Yposition',) + ROT),))
    source = Motion(sh, [[4, 5, 6, 0, 0, 0]], 1 / 30)
    result = retarget_mapped_motion(source, th, {'root': 'root'}, rotation_correction=mode)
    np.testing.assert_array_equal(result.values, [[5, 0, 0, 0]])


def test_unsupported_rotation_channels_remain_allowed():
    sh = Hierarchy((Joint('root', -1, (0, 0, 0), ('Xrotation',)),))
    th = Hierarchy((Joint('root', -1, (0, 0, 0), ROT),))
    result = retarget_mapped_motion(Motion(sh, [[30]], 1 / 30), th, {'root': 'root'})
    np.testing.assert_array_equal(result.values, [[0, 0, 0]])
