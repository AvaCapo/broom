import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bvh.schemas import BVHDocument, BVHJoint
from bvh.interpolation.schemas import InterpolationResult
from bvh.io import validate_motion_values



def count_trailing_static_frames(
    motion_values: np.ndarray,
    threshold: float,
    window: int,
) -> int:
    """Count redundant static frames at the end of motion data.

    The first frame of the static segment is kept as an anchor; only frames
    after it are counted as removable.
    """

    values = np.asarray(motion_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Motion values must be a 2D array.")
    if values.shape[0] < 2:
        return 0

    threshold = float(threshold)
    window = max(1, int(window))
    frame_deltas = np.max(np.abs(np.diff(values, axis=0)), axis=1)

    static_diffs = 0
    for delta in frame_deltas[::-1]:
        if float(delta) <= threshold:
            static_diffs += 1
            continue
        break

    if static_diffs < window:
        return 0
    return static_diffs


def trim_trailing_static_frames(
    motion_values: np.ndarray,
    min_frames: int,
    threshold: float,
    window: int,
) -> np.ndarray:
    """Remove redundant trailing static frames while keeping min_frames."""

    values = np.asarray(motion_values, dtype=np.float64)
    removable = count_trailing_static_frames(
        motion_values=values,
        threshold=threshold,
        window=window,
    )
    if removable == 0:
        return values.copy()

    keep_count = max(int(min_frames), values.shape[0] - removable)
    return values[:keep_count].copy()


def interpolate_documents(
    first_document: BVHDocument,
    second_document: BVHDocument,
    transition_frames: int,
    align_root_translation: bool,
) -> InterpolationResult:
    """Interpolate two compatible BVH documents."""

    motion_values = interpolate_motion_values(
        first_document=first_document,
        first_values=first_document.motion_values,
        second_document=second_document,
        second_values=second_document.motion_values,
        transition_frames=transition_frames,
        align_root_translation=align_root_translation,
    )
    return InterpolationResult(
        document=first_document,
        motion_values=motion_values,
    )


def interpolate_motion_values(
    first_document: BVHDocument,
    first_values: np.ndarray,
    second_document: BVHDocument,
    second_values: np.ndarray,
    transition_frames: int,
    align_root_translation: bool = True,
) -> np.ndarray:
    """Blend two motion arrays that share the same BVH hierarchy."""

    validate_compatible_documents(first_document, second_document)
    first_values = validate_motion_shape(first_document, first_values)
    second_values = validate_motion_shape(second_document, second_values)
    transition_frames = int(transition_frames)

    if transition_frames < 0:
        raise ValueError("Transition frame count cannot be negative.")
    if first_values.shape[0] == 0 or second_values.shape[0] == 0:
        raise ValueError("Cannot interpolate empty motion data.")
    if transition_frames == 0:
        return np.vstack(
            [
                first_values,
                align_second_motion(
                    first_document,
                    first_values,
                    second_values,
                    align_root_translation=align_root_translation,
                ),
            ]
        )
    if transition_frames > first_values.shape[0]:
        raise ValueError("Transition is longer than the first animation.")
    if transition_frames > second_values.shape[0]:
        raise ValueError("Transition is longer than the second animation.")

    second_aligned = align_second_motion(
        first_document,
        first_values,
        second_values,
        align_root_translation=align_root_translation,
    )
    cut_index = first_values.shape[0] - transition_frames
    prefix = first_values[:cut_index]
    transition = blend_frame_ranges(
        document=first_document,
        first_values=first_values[cut_index:],
        second_values=second_aligned[:transition_frames],
    )
    suffix = second_aligned[transition_frames:]
    return np.vstack([prefix, transition, suffix])


def validate_motion_shape(
    document: BVHDocument,
    motion_values: np.ndarray,
) -> np.ndarray:
    """Validate motion array shape for a document and return float64 copy."""

    return validate_motion_values(
        document=document,
        motion_values=motion_values,
    )


def validate_compatible_documents(
    first_document: BVHDocument,
    second_document: BVHDocument,
) -> None:
    """Ensure two BVH documents can be blended channel-by-channel."""

    if len(first_document.joints) != len(second_document.joints):
        raise ValueError("Skeletons have different joint counts.")
    if first_document.total_channels != second_document.total_channels:
        raise ValueError("Skeletons have different channel counts.")

    for first_joint, second_joint in zip(
        first_document.joints,
        second_document.joints,
    ):
        validate_compatible_joints(first_joint, second_joint)


def validate_compatible_joints(
    first_joint: BVHJoint,
    second_joint: BVHJoint,
) -> None:
    """Validate joint metadata required for interpolation."""

    if first_joint.name != second_joint.name:
        raise ValueError(
            "Skeleton joint names differ: "
            f"{first_joint.name!r} != {second_joint.name!r}."
        )
    if first_joint.parent != second_joint.parent:
        raise ValueError(f"Joint {first_joint.name!r} has different parent.")
    if first_joint.channels != second_joint.channels:
        raise ValueError(
            f"Joint {first_joint.name!r} has different channels."
        )


def align_second_motion(
    document: BVHDocument,
    first_values: np.ndarray,
    second_values: np.ndarray,
    align_root_translation: bool,
) -> np.ndarray:
    """Shift second motion so its first root position matches first end."""

    aligned = second_values.copy()
    if not align_root_translation:
        return aligned

    indices = root_position_indices(document)
    if not indices:
        return aligned

    offset = first_values[-1, indices] - aligned[0, indices]
    aligned[:, indices] += offset
    return aligned


def root_position_indices(document: BVHDocument) -> list[int]:
    """Return absolute channel indices for root position channels."""

    root_joint = next(
        (joint for joint in document.joints if joint.parent == -1),
        None,
    )
    if root_joint is None:
        return []

    return [
        root_joint.channel_start + offset
        for offset, channel in enumerate(root_joint.channels)
        if channel.endswith("position")
    ]


def blend_frame_ranges(
    document: BVHDocument,
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> np.ndarray:
    """Blend matching frame ranges from two motions."""

    if first_values.shape != second_values.shape:
        raise ValueError(
            "Blend ranges must have the same shape, got "
            f"{first_values.shape} and {second_values.shape}."
        )
    frame_count = first_values.shape[0]
    if frame_count == 0:
        return first_values.copy()
    if frame_count == 1:
        alphas = np.asarray([1.0], dtype=np.float64)
    else:
        alphas = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)

    blended = (
        first_values * (1.0 - alphas[:, None])
        + second_values * alphas[:, None]
    )
    for indices, order in rotation_channel_groups(document):
        for frame_index, alpha in enumerate(alphas):
            blended[frame_index, indices] = blend_euler_degrees(
                first_values[frame_index, indices],
                second_values[frame_index, indices],
                order=order,
                alpha=float(alpha),
            )
    return blended


def rotation_channel_groups(
    document: BVHDocument,
) -> list[tuple[list[int], np.ndarray]]:
    """Return per-joint Euler rotation channel groups."""

    groups: list[tuple[list[int], np.ndarray]] = []
    for joint in document.joints:
        rotation_offsets = [
            offset
            for offset, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
        ]
        if len(rotation_offsets) != 3:
            continue

        indices = [
            joint.channel_start + offset
            for offset in rotation_offsets
        ]
        order = np.asarray(
            [
                joint.channels[offset][0].lower()
                for offset in rotation_offsets
            ]
        )
        groups.append((indices, order))
    return groups


def blend_euler_degrees(
    first_angles: np.ndarray,
    second_angles: np.ndarray,
    order: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend Euler angles in degrees with SciPy quaternion SLERP."""

    order_string = "".join(str(axis) for axis in order)
    key_rotations = Rotation.from_euler(
        order_string,
        np.vstack([first_angles, second_angles]),
        degrees=True,
    )
    blended = Slerp([0.0, 1.0], key_rotations)([float(alpha)])
    return wrap_degrees(blended.as_euler(order_string, degrees=True)[0])


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap angles to the [-180, 180) interval."""

    return (values + 180.0) % 360.0 - 180.0
