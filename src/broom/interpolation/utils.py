import numpy as np

from broom import Hierarchy, Motion
from broom.rotations.euler import blend_euler_degrees


def interpolate_motion(
    first_motion: Motion,
    second_motion: Motion,
    transition_frames: int,
    *,
    align_root_translation: bool = True,
) -> Motion:
    """Blend two motion arrays that share the same BVH hierarchy."""

    # TODO: Add numerical checks for overlap endpoints and root alignment.

    validate_compatible_motions(first_motion, second_motion)
    transition_frames = int(transition_frames)

    if transition_frames < 0:
        raise ValueError("Transition frame count cannot be negative.")
    if transition_frames == 0:
        return first_motion.with_values(
            np.vstack(
                (
                    first_motion.values,
                    align_second_motion(
                        first_motion,
                        second_motion,
                        align_root_translation=align_root_translation,
                    ),
                )
            )
        )
    if transition_frames > first_motion.frame_count:
        raise ValueError("Transition is longer than the first animation.")
    if transition_frames > second_motion.frame_count:
        raise ValueError("Transition is longer than the second animation.")

    second_aligned = align_second_motion(
        first_motion,
        second_motion,
        align_root_translation=align_root_translation,
    )
    cut_index = first_motion.frame_count - transition_frames
    prefix = first_motion.values[:cut_index]
    transition = blend_frame_ranges(
        hierarchy=first_motion.hierarchy,
        first_values=first_motion.values[cut_index:],
        second_values=second_aligned[:transition_frames],
    )
    suffix = second_aligned[transition_frames:]
    return first_motion.with_values(np.vstack((prefix, transition, suffix)))


# TODO: Consider whether compatibility validation belongs in ops or another module.
def validate_compatible_motions(
    first_motion: Motion,
    second_motion: Motion,
) -> None:
    """Ensure two motions have the same hierarchy and frame time."""

    if not isinstance(first_motion, Motion) or not isinstance(second_motion, Motion):
        raise TypeError("Both inputs must be Motion instances.")
    if first_motion.hierarchy != second_motion.hierarchy:
        raise ValueError("Motions must have identical hierarchies.")
    if first_motion.frame_time != second_motion.frame_time:
        raise ValueError("Motions must have the same frame_time.")


def align_second_motion(
    first_motion: Motion,
    second_motion: Motion,
    align_root_translation: bool,
) -> np.ndarray:
    """Shift second motion so its first root position matches first end."""

    aligned = second_motion.values.copy()
    if not align_root_translation:
        return aligned

    indices = first_motion.hierarchy.position_channel_indices(
        first_motion.hierarchy.root_name
    )
    if not indices:
        return aligned

    offset = first_motion.values[-1, indices] - aligned[0, indices]
    aligned[:, indices] += offset
    return aligned


def blend_frame_ranges(
    hierarchy: Hierarchy,
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> np.ndarray:
    """Blend matching frame ranges from two motions."""

    # TODO: Add numerical checks for partial axes and winding.

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
    for joint in hierarchy.joints:
        indices = list(hierarchy.rotation_channel_indices(joint.name))
        if len(indices) != 3:
            continue
        order = np.asarray(
            [
                channel[0]
                for channel in joint.channels
                if channel.endswith("rotation")
            ]
        )
        for frame_index, alpha in enumerate(alphas):
            blended[frame_index, indices] = blend_euler_degrees(
                first_values[frame_index, indices],
                second_values[frame_index, indices],
                order=order,
                alpha=float(alpha),
            )
    return blended
