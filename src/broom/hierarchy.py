"""Immutable skeleton topology and scalar channel layout."""

from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np

_CHANNELS = frozenset(
    ("Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation")
)
_QUATERNION_TOLERANCE = 1e-6


def _finite_tuple(
    values: Sequence[float], *, name: str, size: int
) -> tuple[float, ...]:
    """Convert finite numeric values to a tuple of the required size."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} numeric values.") from error
    if len(result) != size or not all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values.")
    return result


def _normalize_channels(channels: Sequence[str] | str) -> tuple[str, ...]:
    """Validate scalar BVH channels and preserve their order."""
    if isinstance(channels, str):
        declared = (channels,)
    elif isinstance(channels, (list, tuple)):
        declared = tuple(channels)
    else:
        raise ValueError("channels must be an ordered list or tuple of channel names.")
    if len(set(declared)) != len(declared) or not all(
        channel in _CHANNELS for channel in declared
    ):
        raise ValueError("channels must be unique BVH position/rotation channel names.")
    return declared


@dataclass(frozen=True, slots=True)
class Joint:
    """Describe one joint in a skeleton hierarchy.

    Parameters
    ----------
    name:
        Non-empty joint name, unique within its :class:`Hierarchy`.
    parent:
        Index of the parent joint. Use ``-1`` for the hierarchy root.
    offset:
        Three-dimensional rest offset. For a non-root joint, the offset is
        measured from its parent joint in the parent's local coordinate system.
        For the root, it is expressed in skeleton coordinates.
    channels:
        Ordered scalar BVH channel names. A single string or an ordered list
        or tuple is accepted and normalized to a tuple. Each name must be one
        of ``Xposition``, ``Yposition``, ``Zposition``, ``Xrotation``,
        ``Yrotation``, or ``Zrotation``; duplicate names are not allowed.
    local_orientation:
        Unit rest-orientation quaternion in ``wxyz`` order. Forward kinematics
        applies it before the joint's channel rotation. BVH export currently
        supports only the identity orientation.
    end_site_offset:
        Optional three-dimensional BVH End Site offset in this joint's local
        coordinate system. It does not create a joint or motion channels.

    Raises
    ------
    ValueError
        If a field has an invalid type, shape, value, or quaternion norm.

    Notes
    -----
    Instances are immutable. The declared channel order defines this
    joint's columns in a :class:`Motion` value table.
    """

    name: str
    parent: int
    offset: tuple[float, float, float]
    channels: Sequence[str] | str = ()
    local_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    end_site_offset: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Joint name must be a non-empty string.")
        if isinstance(self.parent, bool) or not isinstance(self.parent, int):
            raise ValueError("Joint parent must be an integer index.")
        orientation = _finite_tuple(
            self.local_orientation, name="local_orientation", size=4
        )
        if abs(float(np.linalg.norm(orientation)) - 1.0) > _QUATERNION_TOLERANCE:
            raise ValueError(
                "local_orientation must be a unit quaternion in wxyz order."
            )
        object.__setattr__(
            self, "offset", _finite_tuple(self.offset, name="offset", size=3)
        )
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        object.__setattr__(self, "local_orientation", orientation)
        if self.end_site_offset is not None:
            object.__setattr__(
                self,
                "end_site_offset",
                _finite_tuple(self.end_site_offset, name="end_site_offset", size=3),
            )


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """Store an immutable skeleton topology and its scalar channel layout.

    Parameters
    ----------
    joints:
        Joint declarations in hierarchy order. The hierarchy must have one
        root, identified by ``parent=-1``. Every other joint must reference a
        preceding joint as its parent, and joint names must be unique.

    Raises
    ------
    ValueError
        If ``joints`` is empty, contains values other than :class:`Joint`, or
        does not define a valid rooted hierarchy.

    Notes
    -----
    ``total_channels``, ``channel_start(index)``, and
    ``channel_count(index)`` are derived from the ordered ``Joint.channels``
    declarations. They describe the column layout of :class:`Motion.values`.
    """

    joints: tuple[Joint, ...]
    _names: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _children: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _starts: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _root_index: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        joints = tuple(self.joints)
        if not joints or not all(isinstance(joint, Joint) for joint in joints):
            raise ValueError("Hierarchy must contain Joint instances.")
        names = tuple(j.name for j in joints)
        if len(set(names)) != len(names):
            raise ValueError("Joint names must be unique within a Hierarchy.")
        children = [[] for _ in joints]
        roots = []
        starts = []
        cursor = 0
        for index, joint in enumerate(joints):
            starts.append(cursor)
            cursor += len(joint.channels)
            if joint.parent == -1:
                roots.append(index)
            elif 0 <= joint.parent < index:
                children[joint.parent].append(index)
            else:
                raise ValueError("Each non-root parent must precede its child.")
        if len(roots) != 1:
            raise ValueError("Hierarchy must have exactly one root joint.")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(
            self, "_names", MappingProxyType({name: i for i, name in enumerate(names)})
        )
        object.__setattr__(self, "_children", tuple(tuple(item) for item in children))
        object.__setattr__(self, "_starts", tuple(starts))
        object.__setattr__(self, "_root_index", roots[0])

    @property
    def joint_count(self) -> int:
        """Return the number of joints in hierarchy order."""
        return len(self.joints)

    @property
    def root(self) -> int:
        """Return the index of the unique root joint."""
        return self._root_index

    @property
    def root_joint(self) -> Joint:
        """Return the unique root Joint value."""
        return self.joints[self.root]

    @property
    def root_name(self) -> str:
        """Return the name of the unique root joint."""
        return self.joint_name(self.root)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return joint names in hierarchy order."""
        return tuple(self.joint_name(index) for index in range(self.joint_count))

    @property
    def total_channels(self) -> int:
        """Return the scalar-table width derived from all joint channels."""
        return sum(len(joint.channels) for joint in self.joints)

    @property
    def parameter_width(self) -> int:
        """Return ``total_channels`` as a compatibility alias."""
        return self.total_channels

    def joint_index(self, name: str) -> int:
        """Return the index of a joint name."""
        try:
            return self._names[name]
        except KeyError as error:
            raise KeyError(f"Unknown joint {name!r}.") from error

    def joint_name(self, index: int) -> str:
        """Return the name at a joint index."""
        self._validate_index(index)
        return self.joints[index].name

    def parent(self, index: int) -> int:
        """Return a parent index, or ``-1`` for the root."""
        self._validate_index(index)
        return self.joints[index].parent

    def parent_index(self, name: str) -> int:
        """Return a parent index for a joint name."""
        return self.parent(self.joint_index(name))

    def parent_joint(self, index: int) -> Joint | None:
        """Return a parent Joint, or ``None`` for the root."""
        parent_index = self.parent(index)
        return None if parent_index == -1 else self.joints[parent_index]

    def parent_name(self, index: int) -> str | None:
        """Return a parent name, or ``None`` for the root."""
        parent_joint = self.parent_joint(index)
        return None if parent_joint is None else parent_joint.name

    def children(self, index: int) -> tuple[int, ...]:
        """Return child indices in hierarchy order."""
        self._validate_index(index)
        return self._children[index]

    def children_indices(self, name: str) -> tuple[int, ...]:
        """Return child indices for a joint name."""
        return self.children(self.joint_index(name))

    def children_joints(self, index: int) -> tuple[Joint, ...]:
        """Return child Joint values in hierarchy order."""
        return tuple(self.joints[child_index] for child_index in self.children(index))

    def children_names(self, index: int) -> tuple[str, ...]:
        """Return child names in hierarchy order."""
        return tuple(
            self.joint_name(child_index) for child_index in self.children(index)
        )

    def channel_start(self, index: int) -> int:
        """Return the first scalar-table column of a joint."""
        self._validate_index(index)
        return self._starts[index]

    def channel_count(self, index: int) -> int:
        """Return the number of scalar channels declared by a joint."""
        self._validate_index(index)
        return len(self.joints[index].channels)

    def format_tree(
            self,
            *,
            show_channels: bool = False,
            show_offsets: bool = False,
            max_depth: int | None = None,
        ) -> str:
            """Return the hierarchy formatted as an ASCII depth-first tree."""
            if max_depth is not None and (
                isinstance(max_depth, bool)
                or not isinstance(max_depth, int)
                or max_depth < 0
            ):
                raise ValueError("max_depth must be a non-negative integer or None.")
    
            lines: list[str] = []
            stack: list[tuple[int, str, bool, int]] = [(self.root, "", True, 0)]
            while stack:
                joint_index, prefix, is_last, depth = stack.pop()
                joint = self.joints[joint_index]
                line = prefix + ("`-- " if is_last else "|-- ") + joint.name
    
                details: list[str] = []
                if show_channels and joint.channels:
                    details.append(f"channels={list(joint.channels)}")
                if show_offsets:
                    details.append(
                        "offset=["
                        f"{joint.offset[0]:.3f}, {joint.offset[1]:.3f}, "
                        f"{joint.offset[2]:.3f}]"
                    )
                if details:
                    line += "  (" + ", ".join(details) + ")"
                lines.append(line)
    
                if max_depth is not None and depth >= max_depth:
                    continue
    
                child_prefix = prefix + ("    " if is_last else "|   ")
                children = self.children(joint_index)
                for child_offset, child_index in reversed(list(enumerate(children))):
                    stack.append(
                        (
                            child_index,
                            child_prefix,
                            child_offset == len(children) - 1,
                            depth + 1,
                        )
                    )
            return "\n".join(lines)

    def _validate_index(self, index: int) -> None:
        """Raise IndexError unless ``index`` denotes a hierarchy joint."""
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.joint_count
        ):
            raise IndexError(f"Joint index {index!r} is outside this Hierarchy.")
