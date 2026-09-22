"""Space-time retargeting for BVH skeletons with identical topology.

This module implements the initial algorithm from Gleicher (1998): the source
motion is re-applied to a target with different segment lengths, then a cubic
B-spline displacement is optimized over the complete clip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.spatial.transform import Rotation

from broom.bvh.kinematics import compute_global_positions, compute_global_transforms, axis_rotation_matrices
from broom.bvh.rotations.rotvec import right_jacobian
from broom.bvh.ops.motion import with_motion_values
from broom.bvh.retargeting.root_motion import skeleton_scale
from broom.bvh.retargeting.rotation_transfer import matrices_to_euler_near_reference
from broom.bvh.schemas import BVHDocument


# 5 mm is a practical default positional error for human animation in meters,
# not a claim that every motion or target has the same acceptable tolerance.
DEFAULT_CONSTRAINT_TOLERANCE = 5.0e-3


def _coordinate_indices(axes: str) -> list[int]:
    """Validate a nonempty selection of world axes in XYZ order."""
    if not isinstance(axes, str) or not axes or axes not in ("X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"):
        raise ValueError("axes must be a nonempty subset of 'XYZ' in XYZ order.")
    return ["XYZ".index(axis) for axis in axes]


def _position_targets(positions, frame_count: int, coordinates: list[int]) -> np.ndarray:
    """Accept full XYZ or selected coordinates, constant or per frame."""
    values = np.asarray(positions, dtype=np.float64)
    count = len(coordinates)
    if values.shape in ((3,), (frame_count, 3)):
        values = values[..., coordinates]
    elif values.shape not in ((count,), (frame_count, count)):
        raise ValueError(
            f"positions must have shape (3,), ({frame_count}, 3), "
            f"({count},) or ({frame_count}, {count}); got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Selected position coordinates must be finite.")
    return np.broadcast_to(values, (frame_count, count))


def position_constraint(
    global_positions: np.ndarray,
    joint_index: int,
    frames: np.ndarray,
    positions: np.ndarray,
    weight: float = 1.0,
    *,
    axes: str = "XYZ",
) -> np.ndarray:
    """Match selected world coordinates to constant or per-frame targets.

    axes is an XYZ-ordered subset, e.g. 'Y' or 'XZ'. positions contains either
    full XYZ vectors or only selected coordinates, shape (K,) or (N, K).
    """

    coordinates = _coordinate_indices(axes)
    targets = _position_targets(positions, len(frames), coordinates)
    return np.sqrt(float(weight)) * (
        global_positions[frames, joint_index][:, coordinates] - targets
    ).reshape(-1)


def floor_constraint(
    global_positions: np.ndarray,
    joint_index: int,
    frames: np.ndarray,
    normal: np.ndarray,
    offset: float = 0.0,
    weight: float = 1.0,
) -> np.ndarray:
    """Return residuals for a joint staying on the permitted side of a plane."""

    signed_distance = global_positions[frames, joint_index] @ normal - float(offset)
    return np.sqrt(float(weight)) * np.minimum(signed_distance, 0.0)


def stationary_constraint(
    global_positions: np.ndarray,
    joint_index: int,
    frames: np.ndarray,
    weight: float = 1.0,
    *,
    axes: str = "XYZ",
) -> np.ndarray:
    """Keep selected world coordinates stationary across ordered frames."""

    coordinates = _coordinate_indices(axes)
    return np.sqrt(float(weight)) * np.diff(
        global_positions[frames, joint_index][:, coordinates], axis=0
    ).reshape(-1)


def relational_constraint(
    global_positions: np.ndarray,
    joint_a_index: int,
    joint_b_index: int,
    frames: np.ndarray,
    source_normalized_distance: np.ndarray,
    target_path_length: float,
    weight: float = 1.0,
) -> np.ndarray:
    """Match source-normalized distance between two target joint origins."""

    displacement = global_positions[frames, joint_a_index] - global_positions[frames, joint_b_index]
    distance = np.linalg.norm(displacement, axis=1)
    return np.sqrt(float(weight)) * (
        distance / float(target_path_length) - source_normalized_distance
    )


def joint_limit_constraint(
    motion_values: np.ndarray,
    channel_indices: np.ndarray,
    frames: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Return soft residuals for channel values outside their allowed range."""

    values = motion_values[np.ix_(frames, channel_indices)]
    below = np.minimum(values - minimum, 0.0)
    above = np.maximum(values - maximum, 0.0)
    return np.sqrt(float(weight)) * np.concatenate(
        [below.reshape(-1), above.reshape(-1)]
    )


def spline_jacobian(derivatives: np.ndarray, basis: np.ndarray) -> csr_matrix:
    """Lift derivatives[F, residuals, channels] to a sparse spline Jacobian.

    Rows are frame-major; columns are coefficient-major (k * channels + a).
    """
    frames, residuals, channels = derivatives.shape
    f, r, a = np.nonzero(derivatives)
    values = derivatives[f, r, a, None] * basis[f]
    rows = np.broadcast_to((f * residuals + r)[:, None], values.shape)
    cols = a[:, None] + channels * np.arange(basis.shape[1])[None, :]
    nonzero = values != 0
    return coo_matrix(
        (values[nonzero], (rows[nonzero], cols[nonzero])),
        shape=(frames * residuals, basis.shape[1] * channels),
    ).tocsr()


def position_jacobian(
    positions: np.ndarray,
    world_axes: np.ndarray,
    joint_index: int,
    variable_joints: np.ndarray,
    translation_mask: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Differentiate a joint origin wrt frame parameters, returning [F, 3, D].

    world_axes[F, D, 3] includes the SO(3) right Jacobian for rotvec parameters.
    Only strict ancestor rotations affect a joint origin; root translations
    affect every joint in the (single-root) skeleton.
    """
    result = np.zeros((positions.shape[0], 3, variable_joints.size))
    result[:, :, translation_mask] = world_axes[:, translation_mask].swapaxes(1, 2)
    ancestors = []
    parent = parents[joint_index]
    while parent >= 0:
        ancestors.append(parent)
        parent = parents[parent]
    active = np.isin(variable_joints, ancestors) & ~translation_mask
    lever = positions[:, joint_index, None] - positions[:, variable_joints[active]]
    result[:, :, active] = np.cross(world_axes[:, active], lever).swapaxes(1, 2)
    return result


def euler_rotvec_jacobian(
    angles: np.ndarray, order: str, local_rotations: np.ndarray, corrections: np.ndarray
) -> np.ndarray:
    """Derivative of decoded Euler radians wrt right-composed rotvec radians.

    Euler limits remain singular at gimbal lock; never substitute an arbitrary
    pseudo-inverse for an undefined derivative.
    """
    prefix = np.broadcast_to(np.eye(3), local_rotations.shape).copy()
    axes = np.empty_like(prefix)
    for column, axis in enumerate(order):
        axes[:, :, column] = prefix[:, :, "XYZ".index(axis)]
        prefix = prefix @ axis_rotation_matrices(axis, angles[:, column])
    body_axes = local_rotations.swapaxes(-1, -2) @ axes
    if np.any(np.abs(np.linalg.det(body_axes)) < 1e-7):
        raise ValueError("Euler joint_limits are singular at gimbal lock; change the limits' parameterization.")
    return np.linalg.solve(body_axes, right_jacobian(corrections))


def retarget_motion_spacetime(
    source_document: BVHDocument,
    target_document: BVHDocument,
    constraints: Sequence[Mapping[str, object]],
    *,
    scale: float | None = None,
    control_point_spacing: int = 4,
    control_weight: float = 1.0e-3,
    parameter_weights: np.ndarray | Sequence[float] | None = None,
    constraint_tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE,
    smoothing_sigma: float = 1.0,
    max_nfev: int | None = None,
    ftol: float = 1.0e-6,
    xtol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    verbose: int = 0,
) -> OptimizeResult:
    """Retarget equal-topology BVH motion with exponential B-spline correction.

    Three-axis joints use R_initial @ Exp(phi), with XYZ rotvec coefficients
    in radians. One-axis joints retain their hinge axis; two-axis joints are
    rejected rather than silently gaining a degree of freedom. Root translation
    coefficients use document length units. The three slots formerly occupied
    by Euler channels now mean rotvec X/Y/Z, independent of BVH channel order.
    parameter_weights and control_weight penalize these new units; old degree
    weights are NOT automatically reinterpreted as an angular distance metric.
    Output motion_values remain BVH Euler degrees. Euler joint limits also remain
    in degrees, on the branch nearest the initial motion.

    ``constraints`` is a sequence of plain dictionaries. Each dictionary uses
    one of these forms:

    ``{"type": "position", "joint": str, "frames": array_like,
    "positions": (N, K) or (K,) array_like, "axes": "XYZ", "weight": float}``

    ``{"type": "floor", "joint": str, "frames": array_like,
    "normal": (3,) array_like, "offset": float, "weight": float}``

    ``{"type": "stationary", "joint": str, "frames": array_like,
    "axes": "XYZ", "weight": float}``

    ``{"type": "joint_limit", "joint": str, "frames": array_like,
    "minimum": array_like, "maximum": array_like, "weight": float}``

    ``{"type": "relational", "joint_a": str, "joint_b": str,
    "frames": array_like, "source_normalized_distance": array_like,
    "target_path_length": float, "activation": array_like, "weight": float}``

    position and stationary accept axes as a nonempty XYZ-ordered subset.
    Position targets may contain full XYZ or only selected coordinates. Only
    selected coordinates contribute to root centering, residuals and Jacobians.

    For position, stationary, and floor constraints without an explicit
    ``weight``, the solver uses ``1 / (constraint_tolerance * scale)**2``.
    The default tolerance is 5 mm in source length units and is scaled into
    target units with the resolved root scale. An explicit ``weight`` always
    takes precedence. Joint-limit residuals remain in degrees and retain their
    existing default weight of 1.0.

    A relational residual is the distance between the two target joint origins,
    normalized by their target rest-pose kinematic path length, minus the
    supplied source-normalized distance. ``activation`` defaults to 1.0 and
    scales both this residual and its analytic Jacobian.

    The result is SciPy's :class:`~scipy.optimize.OptimizeResult` with extra
    attributes: ``document``, ``motion_values``, ``initial_motion_values``,
    ``control_points``, ``constraint_residuals``, and ``scale``.
    """

    if source_document.frame_count < 2:
        raise ValueError("Space-time retargeting requires at least two source frames.")
    if control_point_spacing <= 0:
        raise ValueError("control_point_spacing must be positive.")
    if not np.isfinite(control_weight) or control_weight < 0.0:
        raise ValueError("control_weight cannot be negative.")
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0.0:
        raise ValueError("constraint_tolerance must be finite and positive.")
    if len(source_document.joints) != len(target_document.joints):
        raise ValueError("Source and target skeletons have different joint counts.")
    if source_document.total_channels != target_document.total_channels:
        raise ValueError("Source and target skeletons have different channel counts.")

    for source_joint, target_joint in zip(source_document.joints, target_document.joints):
        if source_joint.channel_start != target_joint.channel_start:
            raise ValueError("Source and target channel starts differ.")
        if source_joint.parent != target_joint.parent:
            raise ValueError(
                f"Joint {source_joint.name!r} has a different parent in the target."
            )
        if source_joint.channels != target_joint.channels:
            raise ValueError(
                f"Joint {source_joint.name!r} has different channels in the target."
            )

    frame_count = source_document.frame_count
    root_joint = next(joint for joint in target_document.joints if joint.parent == -1)
    root_position_channels = {
        channel[0]: root_joint.channel_start + channel_offset
        for channel_offset, channel in enumerate(root_joint.channels)
        if channel.endswith("position")
    }
    variable_indices = np.asarray(
        [
            joint.channel_start + channel_offset
            for joint in target_document.joints
            for channel_offset, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
            or (joint.parent == -1 and channel.endswith("position"))
        ],
        dtype=np.int64,
    )
    if variable_indices.size == 0:
        raise ValueError("Target skeleton has no root translation or rotation channels.")

    if parameter_weights is None:
        control_weights = np.ones(variable_indices.size, dtype=np.float64)
    else:
        control_weights = np.asarray(parameter_weights, dtype=np.float64)
        if control_weights.shape != (variable_indices.size,):
            raise ValueError(
                "parameter_weights must have one value per optimized channel, got "
                f"{control_weights.shape} and {variable_indices.size}."
            )
        if np.any(~np.isfinite(control_weights)) or np.any(control_weights < 0.0):
            raise ValueError("parameter_weights cannot contain negative values.")

    resolved_scale = (
        skeleton_scale(source_document, target_document)
        if scale is None
        else float(scale)
    )
    initial_motion = source_document.motion_values.copy()
    if not np.all(np.isfinite(source_document.motion_values)):
        raise ValueError("Source motion must contain finite values.")
    if not np.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("scale must be finite and positive.")
    scaled_constraint_tolerance = float(constraint_tolerance) * resolved_scale
    default_spatial_weight = 1.0 / scaled_constraint_tolerance**2
    constraints = tuple(
        {
            **constraint,
            **(
                {"weight": default_spatial_weight}
                if "weight" not in constraint
                and constraint.get("type") in ("position", "stationary", "floor")
                else {}
            ),
        }
        for constraint in constraints
    )
    for channel_index in root_position_channels.values():
        initial_motion[:, channel_index] *= resolved_scale

    # Estimate the translation center from explicitly positioned features.
    translation = np.zeros((frame_count, 3), dtype=np.float64)
    translation_weights = np.zeros((frame_count, 3), dtype=np.float64)
    scaled_positions = compute_global_positions(
        replace(target_document, motion_values=initial_motion)
    )
    # Validate before centering; position targets and axes share the residual API.
    _constraint_residuals(target_document, initial_motion, scaled_positions, constraints)
    for constraint in constraints:
        if constraint.get("type") != "position":
            continue
        frames = np.asarray(constraint["frames"], dtype=np.int64).reshape(-1)
        coordinates = _coordinate_indices(constraint.get("axes", "XYZ"))
        positions = _position_targets(constraint["positions"], frames.size, coordinates)
        weight = float(constraint.get("weight", 1.0))
        joint_index = target_document.joint_index[str(constraint["joint"])]
        for column, axis in enumerate(coordinates):
            np.add.at(
                translation[:, axis], frames,
                weight * (positions[:, column] - scaled_positions[frames, joint_index, axis]),
            )
            np.add.at(translation_weights[:, axis], frames, weight)

    # Each coordinate has its own observations; Y targets must not anchor X/Z.
    for axis in range(3):
        known_frames = np.flatnonzero(translation_weights[:, axis] > 0.0)
        if not known_frames.size:
            continue
        translation[known_frames, axis] /= translation_weights[known_frames, axis]
        translation[:, axis] = np.interp(
            np.arange(frame_count), known_frames, translation[known_frames, axis],
        )
        if smoothing_sigma > 0.0:
            translation[:, axis] = gaussian_filter1d(
                translation[:, axis], sigma=float(smoothing_sigma), mode="nearest",
            )
    for axis, channel_index in root_position_channels.items():
        initial_motion[:, channel_index] += translation[:, "XYZ".index(axis)]

    degree = 3
    breakpoints = np.arange(0, frame_count - 1, control_point_spacing, dtype=np.float64)
    if breakpoints.size == 0 or breakpoints[-1] != frame_count - 1:
        breakpoints = np.append(breakpoints, frame_count - 1.0)
    knots = np.concatenate(
        [
            np.repeat(breakpoints[0], degree),
            breakpoints,
            np.repeat(breakpoints[-1], degree),
        ]
    )
    basis = BSpline.design_matrix(
        np.arange(frame_count, dtype=np.float64),
        knots,
        degree,
    ).toarray()
    control_count = basis.shape[1]

    parents = np.array([joint.parent for joint in target_document.joints], dtype=int)
    if np.count_nonzero(parents == -1) != 1 or any(p >= j or p < -1 for j, p in enumerate(parents)):
        raise ValueError("Expected one root and parent-before-child hierarchy.")
    parameter_count = variable_indices.size
    slot_for_channel = {int(channel): slot for slot, channel in enumerate(variable_indices)}
    variable_joints = np.empty(parameter_count, dtype=int)
    translation_mask = np.zeros(parameter_count, dtype=bool)
    translation_axes = np.zeros((parameter_count, 3))
    initial_local = np.broadcast_to(
        np.eye(3), (frame_count, len(parents), 3, 3)
    ).copy()
    rotation_data = {}
    for j, joint in enumerate(target_document.joints):
        channels = []
        order = ""
        for offset, name in enumerate(joint.channels):
            channel = joint.channel_start + offset
            if channel in slot_for_channel:
                slot = slot_for_channel[channel]
                variable_joints[slot] = j
                if name.endswith("position"):
                    translation_mask[slot] = True
                    translation_axes[slot, "XYZ".index(name[0])] = 1
            if name.endswith("rotation"):
                channels.append(channel)
                order += name[0]
        if not channels:
            continue
        if len(channels) not in (1, 3) or len(set(order)) != len(order):
            raise ValueError(f"{joint.name}: expected one hinge axis or three distinct rotation axes.")
        channels = np.asarray(channels)
        slots = np.array([slot_for_channel[int(ch)] for ch in channels])
        reference = np.deg2rad(initial_motion[:, channels])
        initial_local[:, j] = Rotation.from_euler(
            order, reference if len(channels) == 3 else reference[:, 0]
        ).as_matrix()
        rotation_data[j] = (channels, slots, order, reference)

    prepared = []
    names = set()
    for index, spec in enumerate(constraints):
        spec = dict(spec)
        name = str(spec.get("name", f"{spec['type']}_{index}"))
        if name in names:
            raise ValueError(f"Duplicate constraint name: {name!r}.")
        names.add(name)
        spec["name"] = name
        spec["frames"] = np.asarray(spec["frames"], dtype=int).reshape(-1)
        spec["weight"] = float(spec.get("weight", 1))
        if not np.isfinite(spec["weight"]):
            raise ValueError("Constraint weights must be finite.")
        if spec["type"] == "relational":
            for field in ("joint_a", "joint_b"):
                joint_name = str(spec[field])
                if joint_name not in target_document.joint_index:
                    raise ValueError(f"{name}: unknown {field} {joint_name!r}.")
                spec[field + "_index"] = target_document.joint_index[joint_name]
            values = np.asarray(spec["source_normalized_distance"], dtype=float)
            spec["source_normalized_distance"] = np.broadcast_to(values, spec["frames"].shape)
            spec["activation"] = np.broadcast_to(
                np.asarray(spec.get("activation", 1.0), dtype=float), spec["frames"].shape
            )
            spec["target_path_length"] = float(spec["target_path_length"])
            if (not np.all(np.isfinite(spec["source_normalized_distance"]))
                    or not np.all(np.isfinite(spec["activation"]))
                    or np.any(spec["activation"] < 0)
                    or not np.isfinite(spec["target_path_length"])
                    or spec["target_path_length"] <= 0):
                raise ValueError(f"{name}: relational inputs must be finite and valid.")
        else:
            spec["joint_index"] = target_document.joint_index[str(spec["joint"])]
        if spec["type"] in ("position", "stationary"):
            spec["axes"] = spec.get("axes", "XYZ")
            spec["coordinates"] = _coordinate_indices(spec["axes"])
        if spec["type"] == "floor":
            normal = np.array(spec.get("normal", (0, 1, 0)), dtype=float)
            spec["normal"] = normal / np.linalg.norm(normal)
        for field in ("minimum", "maximum", "normal", "offset"):
            if field in spec and not np.all(np.isfinite(spec[field])):
                raise ValueError(f"{name}: {field} must be finite.")
        if spec["type"] == "joint_limit" and spec["joint_index"] not in rotation_data:
            raise ValueError(f"{name}: joint has no rotation channels.")
        prepared.append(spec)

    diagonal = np.tile(np.sqrt(control_weight * control_weights), control_count)
    regularization_jac = diags(diagonal, format="csr")
    cached_x = None
    cached = None

    def evaluate(flat_controls):
        nonlocal cached_x, cached
        if cached_x is not None and np.array_equal(flat_controls, cached_x):
            return cached
        controls = np.asarray(flat_controls).reshape(control_count, parameter_count)
        displacement = basis @ controls
        motion_values = initial_motion.copy()
        motion_values[:, variable_indices[translation_mask]] += displacement[:, translation_mask]
        local = initial_local.copy()
        corrections = {}
        for j, (channels, slots, order, reference) in rotation_data.items():
            phi = displacement[:, slots]
            if len(slots) == 1:
                phi = phi * np.eye(3)["XYZ".index(order)]
            corrections[j] = phi
            local[:, j] = initial_local[:, j] @ Rotation.from_rotvec(phi).as_matrix()
        positions, global_rotations = compute_global_transforms(
            replace(target_document, motion_values=motion_values), local_rotations=local
        )
        world_axes = np.broadcast_to(
            translation_axes, (frame_count, parameter_count, 3)
        ).copy()
        for j, (channels, slots, order, reference) in rotation_data.items():
            if len(slots) == 3:
                world_axes[:, slots] = (
                    global_rotations[:, j] @ right_jacobian(corrections[j])
                ).swapaxes(1, 2)
            else:
                world_axes[:, slots[0]] = global_rotations[:, j, :, "XYZ".index(order)]

        blocks = {}
        jacobians = [regularization_jac]
        position_derivatives = {}
        for spec in prepared:
            kind, frames = spec["type"], spec["frames"]
            weight = spec["weight"]
            if weight == 0:
                # Retain declared residual rows without evaluating singular Euler charts.
                count = len(frames) * (len(spec["coordinates"]) if kind == "position" else 1)
                if kind == "stationary":
                    count = len(spec["coordinates"]) * (len(frames) - 1)
                elif kind == "joint_limit":
                    count = 2 * len(frames) * len(rotation_data[spec["joint_index"]][0])
                blocks[spec["name"]] = np.zeros(count)
                jacobians.append(csr_matrix((count, diagonal.size)))
                continue
            if kind == "relational":
                joint_a, joint_b = spec["joint_a_index"], spec["joint_b_index"]
                for joint in (joint_a, joint_b):
                    if joint not in position_derivatives:
                        position_derivatives[joint] = position_jacobian(
                            positions, world_axes, joint, variable_joints, translation_mask, parents
                        )
                difference = positions[frames, joint_a] - positions[frames, joint_b]
                distance = np.linalg.norm(difference, axis=1)
                direction = np.divide(
                    difference, distance[:, None], out=np.zeros_like(difference), where=distance[:, None] > 1e-12
                )
                derivative = np.einsum(
                    "fi,fid->fd",
                    direction,
                    position_derivatives[joint_a][frames] - position_derivatives[joint_b][frames],
                )
                scale = np.sqrt(weight * spec["activation"]) / spec["target_path_length"]
                residual = relational_constraint(
                    positions, joint_a, joint_b, frames,
                    spec["source_normalized_distance"], spec["target_path_length"], weight,
                ) * np.sqrt(spec["activation"])
                jac = spline_jacobian((scale[:, None] * derivative)[:, None, :], basis[frames])
                blocks[spec["name"]] = residual
                jacobians.append(jac)
                continue
            j = spec["joint_index"]
            if kind != "joint_limit":
                if j not in position_derivatives:
                    position_derivatives[j] = position_jacobian(
                        positions, world_axes, j, variable_joints, translation_mask, parents
                    )
                derivatives = position_derivatives[j][frames]
                if kind in ("position", "stationary"):
                    derivatives = derivatives[:, spec["coordinates"], :]
                if kind == "position":
                    residual = position_constraint(positions, j, frames, spec["positions"], weight, axes=spec["axes"])
                    jac = spline_jacobian(np.sqrt(weight) * derivatives, basis[frames])
                elif kind == "stationary":
                    residual = stationary_constraint(positions, j, frames, weight, axes=spec["axes"])
                    jac = np.sqrt(weight) * (
                        spline_jacobian(derivatives[1:], basis[frames[1:]])
                        - spline_jacobian(derivatives[:-1], basis[frames[:-1]])
                    )
                else:
                    normal = spec["normal"]
                    distance = positions[frames, j] @ normal - float(spec.get("offset", 0))
                    residual = floor_constraint(
                        positions, j, frames, normal, spec.get("offset", 0), weight
                    )
                    projected = np.einsum("i,fid->fd", normal, derivatives)
                    projected *= (distance < 0)[:, None] * np.sqrt(weight)
                    jac = spline_jacobian(projected[:, None, :], basis[frames])
            else:
                channels, slots, order, reference = rotation_data[j]
                derivative = np.zeros((len(frames), len(slots), parameter_count))
                if len(slots) == 1:
                    angles = reference[frames] + displacement[frames][:, slots]
                    derivative[:, 0, slots[0]] = 180 / np.pi
                else:
                    angles = matrices_to_euler_near_reference(
                        local[frames, j], order, reference[frames]
                    )
                    angle_derivative = euler_rotvec_jacobian(
                        angles, order, local[frames, j], corrections[j][frames]
                    )
                    derivative[:, :, slots] = np.rad2deg(angle_derivative)
                values = np.rad2deg(angles)
                below = values - np.asarray(spec["minimum"])
                above = values - np.asarray(spec["maximum"])
                residual = np.sqrt(weight) * np.r_[
                    np.minimum(below, 0).ravel(), np.maximum(above, 0).ravel()
                ]
                jac = vstack([
                    spline_jacobian(
                        np.sqrt(weight) * derivative * (below < 0)[..., None], basis[frames]
                    ),
                    spline_jacobian(
                        np.sqrt(weight) * derivative * (above > 0)[..., None], basis[frames]
                    ),
                ], format="csr")
            blocks[spec["name"]] = residual
            jacobians.append(jac)
        residual = np.concatenate([diagonal * flat_controls, *blocks.values()])
        jac = vstack(jacobians, format="csr")
        cached_x = np.array(flat_controls, copy=True)
        cached = residual, jac, motion_values, local, blocks
        return cached

    x0 = np.zeros(control_count * parameter_count)
    if any(spec["weight"] > 0 for spec in prepared):
        solver_result = least_squares(
            lambda x: evaluate(x)[0],
            x0=x0,
            jac=lambda x: evaluate(x)[1],
            method="trf",
            tr_solver="lsmr",
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=verbose,
        )
    else:
        residual, jac, *_ = evaluate(x0)
        solver_result = OptimizeResult(
            x=x0, fun=residual, jac=jac, cost=0.0, grad=np.zeros_like(x0),
            optimality=0.0, active_mask=np.zeros_like(x0), nfev=0, njev=0,
            success=True, status=1, message="No positive-weight constraints; returned initial motion.",
        )
    _, _, motion_values, local, blocks = evaluate(solver_result.x)
    control_points = solver_result.x.reshape(control_count, parameter_count)
    displacement = basis @ control_points
    # Convert to BVH only at the boundary, not in positional residual evaluation.
    for j, (channels, slots, order, reference) in rotation_data.items():
        if len(slots) == 1:
            angles = reference + displacement[:, slots]
        else:
            angles = matrices_to_euler_near_reference(local[:, j], order, reference)
        unchanged = np.all(displacement[:, slots] == 0, axis=1)
        angles[unchanged] = reference[unchanged]
        motion_values[:, channels] = np.where(
            unchanged[:, None], initial_motion[:, channels], np.rad2deg(angles)
        )
    solver_result.document = with_motion_values(
        target_document, motion_values, frame_time=source_document.frame_time
    )
    solver_result.motion_values = solver_result.document.motion_values
    solver_result.initial_motion_values = initial_motion
    solver_result.control_points = control_points
    solver_result.constraint_residuals = blocks
    solver_result.scale = resolved_scale
    solver_result.control_point_spacing = control_point_spacing
    solver_result.constraint_tolerance = scaled_constraint_tolerance
    solver_result.default_spatial_weight = default_spatial_weight
    solver_result.rotation_parameterization = "right_composed_rotvec_radians"
    parameter_names = [None] * parameter_count
    for slot in np.flatnonzero(translation_mask):
        parameter_names[slot] = (
            target_document.joints[variable_joints[slot]].name,
            "translation_" + "XYZ"[np.argmax(translation_axes[slot])],
        )
    for j, (_, slots, order, _) in rotation_data.items():
        for slot, axis in zip(slots, "XYZ" if len(slots) == 3 else order):
            parameter_names[slot] = (target_document.joints[j].name, "rotvec_" + axis)
    solver_result.parameter_names = tuple(parameter_names)
    return solver_result


def _constraint_residuals(
    document: BVHDocument,
    motion_values: np.ndarray,
    global_positions: np.ndarray,
    constraints: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    """Evaluate the declared constraints once for the current full motion."""

    residuals: dict[str, np.ndarray] = {}
    for constraint_index, constraint in enumerate(constraints):
        constraint_type = str(constraint.get("type", ""))
        name = str(constraint.get("name", f"{constraint_type}_{constraint_index}"))
        frames = np.asarray(constraint["frames"], dtype=np.int64).reshape(-1)
        if frames.size == 0 or np.any((frames < 0) | (frames >= motion_values.shape[0])):
            raise ValueError(f"Constraint {name!r} has invalid frames.")
        weight = float(constraint.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Constraint {name!r} must have a finite nonnegative weight.")
        if constraint_type == "relational":
            joint_a = document.joint_index.get(str(constraint.get("joint_a", "")))
            joint_b = document.joint_index.get(str(constraint.get("joint_b", "")))
            if joint_a is None or joint_b is None:
                raise ValueError(f"Relational constraint {name!r} references an unknown endpoint.")
            source_distance = np.broadcast_to(
                np.asarray(constraint["source_normalized_distance"], dtype=float), frames.shape
            )
            activation = np.broadcast_to(np.asarray(constraint.get("activation", 1.0), dtype=float), frames.shape)
            length = float(constraint["target_path_length"])
            if (not np.all(np.isfinite(source_distance)) or not np.all(np.isfinite(activation))
                    or np.any(activation < 0) or not np.isfinite(length) or length <= 0):
                raise ValueError(f"Relational constraint {name!r} has invalid inputs.")
            residuals[name] = relational_constraint(
                global_positions, joint_a, joint_b, frames, source_distance, length, weight
            ) * np.sqrt(activation)
            continue
        joint_name = str(constraint.get("joint", ""))
        joint_index = document.joint_index.get(joint_name)
        if joint_index is None:
            raise ValueError(
                f"Constraint {name!r} references unknown joint {joint_name!r}."
            )

        if constraint_type == "position":
            residuals[name] = position_constraint(
                global_positions, joint_index, frames, constraint["positions"],
                weight, axes=constraint.get("axes", "XYZ"),
            )
        elif constraint_type == "floor":
            normal = np.array(constraint.get("normal", (0.0, 1.0, 0.0)), dtype=np.float64, copy=True)
            if normal.shape != (3,) or np.linalg.norm(normal) <= 1.0e-12:
                raise ValueError(f"Floor constraint {name!r} must have a nonzero 3D normal.")
            normal /= np.linalg.norm(normal)
            residuals[name] = floor_constraint(
                global_positions,
                joint_index,
                frames,
                normal,
                float(constraint.get("offset", 0.0)),
                weight,
            )
        elif constraint_type == "stationary":
            if frames.size < 2:
                raise ValueError(f"Stationary constraint {name!r} needs at least two frames.")
            if np.any(np.diff(frames) <= 0):
                raise ValueError(f"Stationary constraint {name!r} frames must be increasing.")
            residuals[name] = stationary_constraint(
                global_positions,
                joint_index,
                frames,
                weight,
                axes=constraint.get("axes", "XYZ"),
            )
        elif constraint_type == "joint_limit":
            channel_indices = np.asarray(
                [
                    document.joints[joint_index].channel_start + channel_offset
                    for channel_offset, channel in enumerate(document.joints[joint_index].channels)
                    if channel.endswith("rotation")
                ],
                dtype=np.int64,
            )
            minimum = np.asarray(constraint["minimum"], dtype=np.float64).reshape(-1)
            maximum = np.asarray(constraint["maximum"], dtype=np.float64).reshape(-1)
            if minimum.shape != channel_indices.shape or maximum.shape != channel_indices.shape:
                raise ValueError(
                    f"Joint-limit constraint {name!r} needs one minimum and maximum "
                    f"per rotation channel of {joint_name!r}."
                )
            if np.any(minimum > maximum):
                raise ValueError(f"Joint-limit constraint {name!r} has minimum > maximum.")
            residuals[name] = joint_limit_constraint(
                motion_values,
                channel_indices,
                frames,
                minimum,
                maximum,
                weight,
            )
        else:
            raise ValueError(
                f"Unsupported constraint type {constraint_type!r}. Expected one of: "
                "'position', 'floor', 'stationary', 'joint_limit', 'relational'."
            )
    return residuals


def solve_motion_spacetime(
    reference_document: BVHDocument,
    constraints: Sequence[Mapping[str, object]],
    **solver_options,
) -> OptimizeResult:
    """Optimize an already projected target reference motion.

    ``reference_document`` is both the initial motion and target skeleton, so
    this entry point has no equal-topology source requirement. It deliberately
    keeps all residual evaluation and sparse optimization in
    :func:`retarget_motion_spacetime`.
    """

    if "scale" in solver_options:
        raise ValueError("solve_motion_spacetime uses target-space reference motion; omit scale.")
    return retarget_motion_spacetime(
        reference_document,
        reference_document,
        constraints,
        scale=1.0,
        **solver_options,
    )
