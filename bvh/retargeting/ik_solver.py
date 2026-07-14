"""Torch-based inverse kinematics helpers for BVH retargeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from bvh.kinematics import compute_global_positions
from bvh.schemas import BVHDocument



@dataclass(frozen=True)
class IKPositionObjective:
    """Match target-skeleton global joint positions to source-derived targets."""

    joint_indices: tuple[int, ...]
    target_positions: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class IKJointLimitObjective:
    """Penalize BVH rotation channels that exceed joint limits."""

    channel_indices: tuple[int, ...]
    min_values: tuple[float, ...]
    max_values: tuple[float, ...]
    weight: float = 10.0


@dataclass(frozen=True)
class IKRegularizationObjective:
    """Keep optimized channels close to the FK retarget seed."""

    channel_indices: tuple[int, ...]
    seed_values: np.ndarray
    weight: float = 1.0e-3


class TorchBVHIKSolver:
    """Batched BVH IK solver using :class:`torch.optim.LBFGS`.

    The solver mirrors Newton's high-level idea: a set of objectives produces
    residuals, then an optimizer updates joint coordinates to minimize their
    squared norm. Here the coordinates are BVH Euler rotation channels in
    degrees, and PyTorch autograd supplies the Jacobian implicitly.
    """

    def __init__(
        self,
        document: BVHDocument,
        objectives: tuple[
            IKPositionObjective | IKJointLimitObjective | IKRegularizationObjective,
            ...,
        ],
        max_iterations: int = 20,
        history_size: int = 10,
        line_search_fn: str | None = "strong_wolfe",
        tolerance_grad: float = 1.0e-7,
        tolerance_change: float = 1.0e-9,
        device: str | None = None,
        dtype: str = "float64",
        clamp_to_limits: bool = True,
    ) -> None:
        self.document = document
        self.objectives = tuple(objectives)
        self.max_iterations = int(max_iterations)
        self.history_size = int(history_size)
        self.line_search_fn = line_search_fn
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.device = torch.device(device or "cpu")
        self.dtype = getattr(torch, dtype)
        self.clamp_to_limits = bool(clamp_to_limits)
        self.rotation_channels = _rotation_channel_indices(document)
        self._hierarchy = _TorchHierarchy.from_document(
            document=document,
            device=self.device,
            dtype=self.dtype,
        )

    def step(self, motion_values: np.ndarray) -> np.ndarray:
        """Optimize all frames in ``motion_values`` and return a copy."""

        if not self.rotation_channels:
            return np.asarray(motion_values, dtype=np.float64).copy()

        seed_np = np.nan_to_num(np.asarray(motion_values, dtype=np.float64), nan=0.0)
        seed = torch.as_tensor(seed_np, dtype=self.dtype, device=self.device)
        variable = seed[:, self.rotation_channels].clone().detach()
        variable.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [variable],
            max_iter=self.max_iterations,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            full_motion = seed.clone()
            full_motion[:, self.rotation_channels] = variable
            positions = self._hierarchy.forward_positions(full_motion)
            loss = self._loss(
                full_motion=full_motion,
                positions=positions,
                seed_motion=seed,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        optimized = seed.clone()
        optimized[:, self.rotation_channels] = variable.detach()
        if self.clamp_to_limits:
            optimized = self._clamp_limits(optimized)
        return optimized.detach().cpu().numpy()

    def _loss(
        self,
        full_motion: torch.Tensor,
        positions: torch.Tensor,
        seed_motion: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros((), dtype=self.dtype, device=self.device)
        for objective in self.objectives:
            if isinstance(objective, IKPositionObjective):
                target = torch.as_tensor(
                    objective.target_positions,
                    dtype=self.dtype,
                    device=self.device,
                )
                actual = positions[:, list(objective.joint_indices), :]
                residual = (actual - target) * float(objective.weight)
                loss = loss + torch.sum(residual * residual)
            elif isinstance(objective, IKJointLimitObjective):
                channels = list(objective.channel_indices)
                values = full_motion[:, channels]
                lower = torch.as_tensor(
                    objective.min_values,
                    dtype=self.dtype,
                    device=self.device,
                )
                upper = torch.as_tensor(
                    objective.max_values,
                    dtype=self.dtype,
                    device=self.device,
                )
                violation = torch.relu(values - upper) + torch.relu(lower - values)
                residual = violation * float(objective.weight)
                loss = loss + torch.sum(residual * residual)
            elif isinstance(objective, IKRegularizationObjective):
                channels = list(objective.channel_indices)
                seed = torch.as_tensor(
                    objective.seed_values,
                    dtype=self.dtype,
                    device=self.device,
                )
                residual = (
                    full_motion[:, channels] - seed
                ) * float(objective.weight)
                loss = loss + torch.sum(residual * residual)
            else:
                raise TypeError(f"Unsupported IK objective: {type(objective)}")
        return loss

    def _clamp_limits(self, motion: "torch.Tensor") -> "torch.Tensor":
        result = motion.clone()
        for objective in self.objectives:
            if not isinstance(objective, IKJointLimitObjective):
                continue
            channels = list(objective.channel_indices)
            lower = torch.as_tensor(
                objective.min_values,
                dtype=self.dtype,
                device=self.device,
            )
            upper = torch.as_tensor(
                objective.max_values,
                dtype=self.dtype,
                device=self.device,
            )
            result[:, channels] = torch.maximum(
                torch.minimum(result[:, channels], upper),
                lower,
            )
        return result


def optimize_retargeted_motion_with_ik(
    source_document: BVHDocument,
    target_document: BVHDocument,
    target_motion: np.ndarray,
    source_to_target: dict[str, str],
    root_scale: float,
    iterations: int = 20,
    position_weight: float = 1.0,
    limit_weight: float = 10.0,
    regularization_weight: float = 1.0e-3,
    history_size: int = 10,
    device: str | None = None,
    clamp_to_limits: bool = True,
) -> np.ndarray:
    """Run LBFGS IK after FK-style retargeting."""

    if target_motion.shape[0] != source_document.frame_count:
        raise ValueError(
            "IK target motion must have the same frame count as the source "
            f"document: got {target_motion.shape[0]} and "
            f"{source_document.frame_count}."
        )

    position_objective = build_position_objective(
        source_document=source_document,
        target_document=target_document,
        target_motion=target_motion,
        source_to_target=source_to_target,
        root_scale=root_scale,
        weight=position_weight,
    )
    objectives: list[
        IKPositionObjective | IKJointLimitObjective | IKRegularizationObjective
    ] = []
    if position_objective is not None:
        objectives.append(position_objective)
    objectives.extend(
        build_joint_limit_objectives(
            target_document=target_document,
            weight=limit_weight,
        )
    )

    rotation_channels = _rotation_channel_indices(target_document)
    if rotation_channels and regularization_weight > 0.0:
        objectives.append(
            IKRegularizationObjective(
                channel_indices=tuple(rotation_channels),
                seed_values=np.asarray(target_motion[:, rotation_channels]),
                weight=regularization_weight,
            )
        )

    if not objectives:
        return np.asarray(target_motion, dtype=np.float64).copy()

    solver = TorchBVHIKSolver(
        document=target_document,
        objectives=tuple(objectives),
        max_iterations=iterations,
        history_size=history_size,
        device=device,
        clamp_to_limits=clamp_to_limits,
    )
    return solver.step(target_motion)


def build_position_objective(
    source_document: BVHDocument,
    target_document: BVHDocument,
    target_motion: np.ndarray,
    source_to_target: dict[str, str],
    root_scale: float,
    weight: float,
) -> IKPositionObjective | None:
    """Create a source-to-target global-position matching objective."""

    source_positions = compute_global_positions(source_document)
    target_seed_document = _document_with_motion(
        document=target_document,
        motion_values=target_motion,
    )
    target_seed_positions = compute_global_positions(target_seed_document)
    source_root_index = source_document.joint_index[source_document.root_name]
    target_root_index = target_document.joint_index[target_document.root_name]
    source_root = source_positions[:, source_root_index : source_root_index + 1, :]
    target_root = target_seed_positions[:, target_root_index : target_root_index + 1, :]

    target_indices: list[int] = []
    targets: list[np.ndarray] = []
    for source_name, target_name in source_to_target.items():
        source_index = source_document.joint_index.get(source_name)
        target_index = target_document.joint_index.get(target_name)
        if source_index is None or target_index is None:
            continue
        target_indices.append(target_index)
        target_positions = target_root[:, 0, :] + (
            source_positions[:, source_index, :] - source_root[:, 0, :]
        ) * float(root_scale)
        targets.append(target_positions)

    if not target_indices:
        return None
    return IKPositionObjective(
        joint_indices=tuple(target_indices),
        target_positions=np.stack(targets, axis=1),
        weight=weight,
    )


def build_joint_limit_objectives(
    target_document: BVHDocument,
    weight: float,
) -> tuple[IKJointLimitObjective, ...]:
    """Return one joint-limit objective per constrained BVH joint."""

    objectives = []
    for joint in target_document.joints:
        rotation_offsets = [
            offset
            for offset, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
        ]
        if not rotation_offsets:
            continue
        if not joint.min_values or not joint.max_values:
            continue
        if len(joint.min_values) != len(rotation_offsets):
            continue
        if len(joint.max_values) != len(rotation_offsets):
            continue
        channels = tuple(joint.channel_start + offset for offset in rotation_offsets)
        objectives.append(
            IKJointLimitObjective(
                channel_indices=channels,
                min_values=tuple(float(value) for value in joint.min_values),
                max_values=tuple(float(value) for value in joint.max_values),
                weight=weight,
            )
        )
    return tuple(objectives)


@dataclass(frozen=True)
class _TorchHierarchy:
    parent_indices: tuple[int, ...]
    offsets: "torch.Tensor"
    channel_starts: tuple[int, ...]
    channels: tuple[tuple[str, ...], ...]
    device: object
    dtype: object

    @classmethod
    def from_document(
        cls,
        *,
        document: BVHDocument,
        device: object,
        dtype: object,
    ) -> "_TorchHierarchy":
        offsets = torch.as_tensor(
            np.stack([joint.offset for joint in document.joints], axis=0),
            dtype=dtype,
            device=device,
        )
        return cls(
            parent_indices=tuple(joint.parent for joint in document.joints),
            offsets=offsets,
            channel_starts=tuple(joint.channel_start for joint in document.joints),
            channels=tuple(tuple(joint.channels) for joint in document.joints),
            device=device,
            dtype=dtype,
        )

    def forward_positions(self, motion_values: "torch.Tensor") -> "torch.Tensor":
        frames = motion_values.shape[0]
        identity = torch.eye(3, dtype=self.dtype, device=self.device).expand(
            frames,
            3,
            3,
        )
        positions = []
        rotations = []

        for joint_index, channels in enumerate(self.channels):
            local_position = self.offsets[joint_index].expand(frames, 3).clone()
            local_rotation = identity.clone()
            channel_start = self.channel_starts[joint_index]
            for channel_offset, channel in enumerate(channels):
                values = motion_values[:, channel_start + channel_offset]
                if channel.endswith("position"):
                    local_position[:, _axis_dimension(channel[0])] += values
                elif channel.endswith("rotation"):
                    angle = torch.deg2rad(values)
                    local_rotation = torch.matmul(
                        local_rotation,
                        _axis_rotation_matrix(
                            axis=channel[0],
                            radians=angle,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    )

            parent = self.parent_indices[joint_index]
            if parent == -1:
                world_rotation = local_rotation
                world_position = local_position
            else:
                parent_rotation = rotations[parent]
                parent_position = positions[parent]
                world_rotation = torch.matmul(parent_rotation, local_rotation)
                world_position = parent_position + torch.einsum(
                    "fij,fj->fi",
                    parent_rotation,
                    local_position,
                )
            rotations.append(world_rotation)
            positions.append(world_position)

        return torch.stack(positions, dim=1)


def _axis_rotation_matrix(
    axis: str,
    radians: "torch.Tensor",
    dtype: object,
    device: object,
) -> "torch.Tensor":
    cos_v = torch.cos(radians)
    sin_v = torch.sin(radians)
    matrices = torch.zeros(
        (radians.shape[0], 3, 3),
        dtype=dtype,
        device=device,
    )
    if axis == "X":
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = cos_v
        matrices[:, 1, 2] = -sin_v
        matrices[:, 2, 1] = sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Y":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 2] = sin_v
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 0] = -sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Z":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 1] = -sin_v
        matrices[:, 1, 0] = sin_v
        matrices[:, 1, 1] = cos_v
        matrices[:, 2, 2] = 1.0
    else:
        raise ValueError(f"Unsupported rotation axis '{axis}'.")
    return matrices


def _axis_dimension(axis: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def _rotation_channel_indices(document: BVHDocument) -> list[int]:
    indices = []
    for joint in document.joints:
        for offset, channel in enumerate(joint.channels):
            if channel.endswith("rotation"):
                indices.append(joint.channel_start + offset)
    return indices


def _document_with_motion(
    document: BVHDocument,
    motion_values: np.ndarray,
) -> BVHDocument:
    return BVHDocument(
        path=document.path,
        prefix_lines=document.prefix_lines,
        motion_rows=tuple(),
        motion_values=np.asarray(motion_values, dtype=np.float64),
        joints=document.joints,
        total_channels=document.total_channels,
        root_name=document.root_name,
        root_channels=document.root_channels,
        frame_time=document.frame_time,
        declared_frames=int(motion_values.shape[0]),
    )

