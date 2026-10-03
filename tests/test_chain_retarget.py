import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from broom import Hierarchy, Joint, Motion
from broom.kinematics import compute_global_transforms
from broom.retargeting.chain_retarget import (
    refine_mapped_chains,
    retarget_mapped_chains,
)


XYZ = ('Xrotation', 'Yrotation', 'Zrotation')


def chain(names, lengths, orientations=None):
    orientations = orientations or [(1, 0, 0, 0)] * len(names)
    return Hierarchy(tuple(
        Joint(name, i - 1, (0, 0, 0) if i == 0 else (lengths[i - 1], 0, 0),
              XYZ, local_orientation=orientations[i])
        for i, name in enumerate(names)
    ))


def motion(hierarchy, frames=3):
    return Motion(hierarchy, np.zeros((frames, hierarchy.total_channels)), 1 / 30)


def test_removed_elbow_turns_shoulder_towards_source_wrist():
    source = motion(chain(('shoulder', 'elbow', 'wrist'), (1, 1)))
    source.values[:, 5] = 90
    target = chain(('arm', 'hand'), (2,))
    before = source.values.copy()
    result = retarget_mapped_chains(
        source, target, {'shoulder': 'arm', 'wrist': 'hand'},
        chains=(('shoulder', 'wrist'),),
    )
    positions, rotations = compute_global_transforms(result)
    np.testing.assert_allclose(positions[:, 1], np.tile([np.sqrt(2), np.sqrt(2), 0], (3, 1)), atol=1e-8)
    np.testing.assert_allclose(rotations[:, 1], compute_global_transforms(source)[1][:, 2], atol=1e-8)
    np.testing.assert_array_equal(source.values, before)
    assert not np.shares_memory(source.values, result.values)


def test_added_link_follows_sampled_polyline_directions():
    source = motion(chain(('a', 'b', 'c'), (1, 1)))
    source.values[:, 5] = 90
    target = chain(('u', 'v', 'w', 'x'), (2/3, 2/3, 2/3))
    result = retarget_mapped_chains(source, target, {'a': 'u', 'c': 'x'}, chains=(('a', 'c'),))
    points, _ = compute_global_transforms(result)
    links = np.diff(points, axis=1)
    expected = np.array([[1, 0, 0], [2**-.5, 2**-.5, 0], [0, 1, 0]])
    np.testing.assert_allclose(links, np.tile(expected[None] * (2/3), (3, 1, 1)), atol=1e-8)
    assert np.all(np.abs(result.values[:, 5]) > 1)
    assert np.all(np.abs(result.values[:, 8]) > 1)


def test_equal_paths_with_only_endpoint_mapping_preserve_source_fk():
    h = chain(('a', 'b', 'c'), (1, 1))
    source = motion(h)
    source.values[:] = np.random.default_rng(11).uniform(-50, 50, source.values.shape)
    output = retarget_mapped_chains(source, h, {'a': 'a', 'c': 'c'}, chains=(('a', 'c'),))
    for expected, actual in zip(compute_global_transforms(source), compute_global_transforms(output)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_full_mapping_does_not_skip_refinement_of_arbitrary_seed():
    h = chain(('a', 'b'), (1,))
    source = motion(h)
    source.values[:, 2] = 65
    seed = motion(h)
    result = refine_mapped_chains(source, seed, {'a': 'a', 'b': 'b'}, chains=(('a', 'b'),))
    np.testing.assert_allclose(compute_global_transforms(result)[0], compute_global_transforms(source)[0], atol=1e-8)
    np.testing.assert_array_equal(seed.values, 0)


def test_rest_orientation_is_removed_once_and_twist_is_preserved():
    q = Rotation.from_euler('Z', 37, degrees=True).as_quat()[[3, 0, 1, 2]]
    h = chain(('a', 'b', 'c'), (1, 1), [tuple(q)] * 3)
    source = motion(h)
    source.values[:, 0] = [10, 30, 60]
    source.values[:, 3] = [25, 35, 45]
    result = refine_mapped_chains(source, motion(h), {'a': 'a', 'c': 'c'}, chains=(('a', 'c'),))
    for actual, expected in zip(compute_global_transforms(result), compute_global_transforms(source)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_refinement_preserves_translations_outside_channels_and_seed():
    h = Hierarchy((
        Joint('a', -1, (0, 0, 0), ('Xposition',) + XYZ),
        Joint('b', 0, (1, 0, 0), ('Yposition',) + XYZ),
        Joint('side', 0, (0, 1, 0), XYZ),
    ))
    source, seed = motion(h), motion(h)
    source.values[:, 3] = 60
    seed.values[:, 0] = [2, 3, 4]
    seed.values[:, 4] = .3
    seed.values[:, 8:] = [12, 23, 34]
    before = seed.values.copy()
    result = refine_mapped_chains(source, seed, {'a': 'a', 'b': 'b'}, chains=(('a', 'b'),))
    np.testing.assert_array_equal(seed.values, before)
    np.testing.assert_array_equal(result.values[:, [0, 4, 8, 9, 10]], before[:, [0, 4, 8, 9, 10]])
    p, _ = compute_global_transforms(result)
    direction = np.array([.5, np.sqrt(3)/2, 0])
    np.testing.assert_allclose(p[:, 1] - p[:, 0], np.tile(direction * np.sqrt(1.09), (3, 1)), atol=1e-8)


def test_shared_start_is_fitted_jointly_and_branch_order_is_irrelevant():
    h = Hierarchy((Joint('a', -1, (0, 0, 0), XYZ),
                   Joint('b', 0, (1, 0, 0), XYZ),
                   Joint('c', 0, (0, 1, 0), XYZ)))
    source = motion(h)
    source.values[:, :3] = [20, 30, 40]
    mapping = {n: n for n in h.joint_names}
    chains = (('a', 'b'), ('a', 'c'))
    one = refine_mapped_chains(source, motion(h), mapping, chains=chains)
    two = refine_mapped_chains(source, motion(h), mapping, chains=chains[::-1])
    np.testing.assert_array_equal(one.values, two.values)
    np.testing.assert_allclose(compute_global_transforms(one)[0], compute_global_transforms(source)[0], atol=1e-8)


def test_sequential_segments_and_duplicates_are_order_independent():
    h = chain(('a', 'b', 'c'), (1, 1))
    source = motion(h)
    source.values[:, :6] = [0, 0, 30, 0, 0, 70]
    mapping = {n: n for n in h.joint_names}
    a = refine_mapped_chains(source, motion(h), mapping, chains=(('a', 'c'),))
    b = refine_mapped_chains(source, motion(h), mapping, chains=(('b', 'c'), ('a', 'b'), ('a', 'b')))
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_allclose(compute_global_transforms(a)[0], compute_global_transforms(source)[0], atol=1e-8)


def test_antiparallel_link_is_finite_and_points_in_required_direction():
    source = motion(chain(('a', 'b'), (1,)))
    target = Hierarchy((Joint('x', -1, (0, 0, 0), XYZ), Joint('y', 0, (-1, 0, 0), XYZ)))
    result = refine_mapped_chains(source, motion(target), {'a': 'x', 'b': 'y'}, chains=(('a', 'b'),))
    assert np.isfinite(result.values).all()
    np.testing.assert_allclose(compute_global_transforms(result)[0][:, 1], np.tile([1, 0, 0], (3, 1)), atol=1e-8)


def test_channel_less_terminal_keeps_its_channels_and_accepts_projection():
    source = motion(chain(('a', 'b'), (1,)))
    source.values[:, 2] = 45
    target = Hierarchy((Joint('x', -1, (0, 0, 0), XYZ), Joint('y', 0, (1, 0, 0))))
    output = refine_mapped_chains(source, motion(target), {'a': 'x', 'b': 'y'}, chains=(('a', 'b'),))
    np.testing.assert_allclose(compute_global_transforms(output)[0][:, 1], np.tile([2**-.5, 2**-.5, 0], (3, 1)), atol=1e-8)


@pytest.mark.parametrize('kind', ['frames', 'time', 'rest_zero', 'chord_zero', 'nan', 'hinge'])
def test_invalid_inputs_fail_without_mutating_seed(kind):
    h = chain(('a', 'b', 'c'), (1, 1))
    source, seed = motion(h), motion(h)
    chains = (('a', 'c'),)
    mapping = {n: n for n in h.joint_names}
    if kind == 'frames':
        seed = motion(h, 2)
    elif kind == 'time':
        seed = Motion(h, seed.values, 1/60)
    elif kind == 'rest_zero':
        source = motion(chain(('a', 'b', 'c'), (0, 1)))
    elif kind == 'chord_zero':
        source.values[:, 5] = 180
        seed = motion(chain(('a', 'c'), (2,)))
        mapping = {'a': 'a', 'c': 'c'}
    elif kind == 'nan':
        source.values[0, 0] = np.nan
    elif kind == 'hinge':
        seed = motion(Hierarchy((Joint('a', -1, (0, 0, 0), 'Zrotation'), Joint('c', 0, (2, 0, 0), XYZ))))
        mapping = {'a': 'a', 'c': 'c'}
    before = seed.values.copy()
    with pytest.raises(ValueError):
        refine_mapped_chains(source, seed, mapping, chains=chains)
    np.testing.assert_array_equal(seed.values, before)


def test_chain_reference_can_be_used_by_spacetime_with_different_source_topology():
    from broom.retargeting.spacetime import solve_motion_spacetime
    source = motion(chain(('a', 'b', 'c'), (1, 1)))
    source.values[:, 5] = 90
    target = chain(('u', 'v'), (2,))
    reference = retarget_mapped_chains(source, target, {'a': 'u', 'c': 'v'}, chains=(('a', 'c'),))
    before = reference.values.copy()
    target_point = np.tile([1., np.sqrt(3), 0], (3, 1))
    constraints = [{'type': 'position', 'joint': 'v', 'frames': np.arange(3), 'positions': target_point, 'weight': 100.}]
    result = solve_motion_spacetime(reference, constraints, max_nfev=30)
    prior_error = np.linalg.norm(compute_global_transforms(reference)[0][:, 1] - target_point)
    error = np.linalg.norm(compute_global_transforms(result.motion)[0][:, 1] - target_point)
    assert error < prior_error * .01
    np.testing.assert_array_equal(reference.values, before)


def test_added_link_retains_interpolated_twist_without_changing_straight_geometry():
    source = motion(chain(('a', 'b'), (2,)))
    source.values[:, 3] = [30, 60, 90]
    target = chain(('u', 'v', 'w'), (1, 1))
    result = refine_mapped_chains(source, motion(target), {'a': 'u', 'b': 'w'}, chains=(('a', 'b'),))
    positions, rotations = compute_global_transforms(result)
    expected = Rotation.from_euler('X', [15, 30, 45], degrees=True).as_matrix()
    np.testing.assert_allclose(rotations[:, 1], expected, atol=1e-8)
    np.testing.assert_allclose(rotations[:, 2], compute_global_transforms(source)[1][:, 1], atol=1e-8)
    np.testing.assert_allclose(positions[:, 2], np.tile([2, 0, 0], (3, 1)), atol=1e-8)


def test_smooth_source_does_not_introduce_an_euler_wrap_at_180_degrees():
    h = chain(('a', 'b'), (1,))
    source = motion(h, 5)
    source.values[:, 2] = [170, 175, 180, 185, 190]
    result = refine_mapped_chains(source, motion(h, 5), {'a': 'a', 'b': 'b'}, chains=(('a', 'b'),))
    assert np.max(np.abs(np.diff(result.values, axis=0))) < 10
    np.testing.assert_allclose(compute_global_transforms(result)[1], compute_global_transforms(source)[1], atol=1e-8)


def test_intermediate_mapping_splits_chain_and_removing_it_changes_projection():
    source = motion(chain(('a', 'b', 'c'), (1, 1)))
    source.values[:, 5] = 90
    target = chain(('u', 'v', 'w'), (.5, 1.5))
    seed = motion(target)
    mapping = {'a': 'u', 'b': 'v', 'c': 'w'}
    split = refine_mapped_chains(source, seed, mapping, chains=(('a', 'c'),))
    np.testing.assert_allclose(compute_global_transforms(split)[0][:, -1], np.tile([.5, 1.5, 0], (3, 1)), atol=1e-8)
    del mapping['b']
    merged = refine_mapped_chains(source, seed, mapping, chains=(('a', 'c'),))
    # The unmapped elbow still bends the source polyline; its motion is not lost.
    end = compute_global_transforms(merged)[0][:, -1]
    assert np.all(end[:, 1] > 1)
    assert not np.allclose(merged.values, split.values)
    assert mapping == {'a': 'u', 'c': 'w'}
    np.testing.assert_array_equal(seed.values, 0)


def test_mapped_intermediate_outside_target_path_is_rejected():
    source = motion(chain(('a', 'b', 'c'), (1, 1)))
    h = Hierarchy((Joint('u', -1, (0, 0, 0), XYZ), Joint('v', 0, (1, 0, 0), XYZ),
                   Joint('w', 1, (1, 0, 0), XYZ), Joint('side', 0, (0, 1, 0), XYZ)))
    with pytest.raises(ValueError, match='outside target chain'):
        refine_mapped_chains(source, motion(h), {'a': 'u', 'b': 'side', 'c': 'w'}, chains=(('a', 'c'),))


def test_intermediate_mapping_order_must_match_target_path():
    h = chain(('a', 'b', 'c', 'd'), (1, 1, 1))
    with pytest.raises(ValueError, match='out of order'):
        refine_mapped_chains(motion(h), motion(h), {'a': 'a', 'b': 'c', 'c': 'b', 'd': 'd'}, chains=(('a', 'd'),))


@pytest.mark.parametrize('endpoints, match', [
    (('a', 'b', 'c'), 'exactly two'), (('a', 'a'), 'distinct'),
    (('a', 'b'), 'absent'), (('c', 'a'), 'descendant'),
])
def test_invalid_chain_endpoint_description(endpoints, match):
    h = chain(('a', 'b', 'c'), (1, 1))
    with pytest.raises(ValueError, match=match):
        refine_mapped_chains(motion(h), motion(h), {'a': 'a', 'c': 'c'}, chains=(endpoints,))


def test_overlapping_chains_with_unmapped_branchpoint_are_rejected():
    source = Hierarchy((Joint('a', -1, (0, 0, 0), XYZ), Joint('b', 0, (1, 0, 0), XYZ),
                        Joint('c', 1, (1, 0, 0), XYZ), Joint('d', 1, (0, 1, 0), XYZ)))
    target = Hierarchy((Joint('u', -1, (0, 0, 0), XYZ), Joint('v', 0, (2, 0, 0), XYZ),
                        Joint('w', 0, (0, 2, 0), XYZ)))
    with pytest.raises(ValueError, match='overlapping paths'):
        refine_mapped_chains(motion(source), motion(target), {'a': 'u', 'c': 'v', 'd': 'w'}, chains=(('a', 'c'), ('a', 'd')))


def test_endpoint_default_preset_does_not_require_optional_joint_mappings():
    from broom.retargeting import DEFAULT_SOURCE_CHAINS, refine_mapped_chains
    joints = [Joint('Hips', -1, (0, 0, 0), XYZ)]
    paths = (
        ('Hips', 'Spine', 'Spine1', 'Spine2'),
        ('Spine2', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'),
        ('Spine2', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'),
        ('Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'),
        ('Hips', 'RightUpLeg', 'RightLeg', 'RightFoot'),
        ('Spine2', 'Neck', 'Head'),
    )
    for path in paths:
        parent = next(i for i, j in enumerate(joints) if j.name == path[0])
        for name in path[1:]:
            joints.append(Joint(name, parent, (1, 0, 0), XYZ))
            parent = len(joints) - 1
    h = Hierarchy(tuple(joints))
    mapping = {name: name for endpoints in DEFAULT_SOURCE_CHAINS for name in endpoints}
    source = motion(h)
    result = refine_mapped_chains(source, motion(h), mapping, chains=DEFAULT_SOURCE_CHAINS)
    np.testing.assert_allclose(compute_global_transforms(result)[0], compute_global_transforms(source)[0], atol=1e-8)


def test_one_link_and_hundred_link_target_chains_can_coexist():
    source = Hierarchy((Joint('root', -1, (0, 0, 0), XYZ),
                        Joint('l0', 0, (0, 1, 0), XYZ), Joint('l1', 1, (1, 0, 0), XYZ), Joint('l2', 2, (1, 0, 0), XYZ),
                        Joint('r0', 0, (0, -1, 0), XYZ), Joint('r1', 4, (1, 0, 0), XYZ), Joint('r2', 5, (1, 0, 0), XYZ)))
    joints = [Joint('root', -1, (0, 0, 0), XYZ), Joint('left_start', 0, (0, 1, 0), XYZ),
              Joint('left_end', 1, (2, 0, 0), XYZ), Joint('right_start', 0, (0, -1, 0), XYZ)]
    for i in range(100):
        joints.append(Joint(f'right_{i}', len(joints)-1, (.02, 0, 0), XYZ))
    target = Hierarchy(tuple(joints))
    src = motion(source)
    src.values[:, 8] = 60
    src.values[:, 17] = -45
    mapping = {'l0': 'left_start', 'l2': 'left_end', 'r0': 'right_start', 'r2': 'right_99'}
    result = refine_mapped_chains(src, motion(target), mapping, chains=(('l0', 'l2'), ('r0', 'r2')))
    points, _ = compute_global_transforms(result)
    assert np.isfinite(result.values).all()
    np.testing.assert_allclose(np.linalg.norm(points[:, 4:] - points[:, 3:-1], axis=-1), .02, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(points[:, 2] - points[:, 1], axis=-1), 2, atol=1e-8)
