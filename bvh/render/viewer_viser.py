"""Minimal Viser BVH viewer for one or more skeletons."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from broom.bvh.kinematics import compute_global_positions
from broom.bvh.schemas import BVHDocument

try:
    import viser
except Exception:  # pragma: no cover - optional runtime dependency
    viser = None


CLIP_COLORS = [
    (91, 102, 117),
    (217, 95, 2),
    (27, 158, 119),
    (117, 112, 179),
    (231, 41, 138),
    (102, 166, 30),
]
SELECTED_COLOR = (0, 255, 106)
_SERVER_REGISTRY: dict[int, object] = {}
_ACTIVE_VIEWERS_BY_PORT: dict[int, "SkeletonViewer"] = {}


@dataclass(frozen=True)
class _ClipData:
    joints: np.ndarray
    names: list[str]
    edges: list[tuple[int, int]]
    fps: int
    label: str
    offset: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joints.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.joints.shape[1])


@dataclass(frozen=True)
class _SphereData:
    positions: np.ndarray
    radius: float
    color: tuple[int, int, int]
    label: str

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])


def _require_viser() -> None:
    """Raise install instructions when the optional Viser dependency is absent."""

    if viser is None:
        raise ImportError(
            "ViserSkeletonViewer requires viser. See docs/rendering.md for "
            "the tested version and upstream installation instructions."
        )


def _as_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_offsets(offsets: Sequence[Sequence[float]] | Sequence[float] | None, count: int) -> list[np.ndarray]:
    if offsets is None:
        return [np.zeros(3, dtype=np.float32) for _ in range(count)]
    if count == 1 and len(offsets) == 3 and not isinstance(offsets[0], (list, tuple, np.ndarray)):
        return [np.asarray(offsets, dtype=np.float32)]
    resolved = [np.asarray(offset, dtype=np.float32) for offset in offsets]
    if len(resolved) != count:
        raise ValueError(f"Expected {count} offsets, got {len(resolved)}.")
    return resolved


def _fps_from_document(document: BVHDocument) -> int:
    if document.frame_time is None or document.frame_time <= 0.0:
        return 30
    return max(1, int(round(1.0 / float(document.frame_time))))


def _resolve_spheres(spheres) -> list[_SphereData]:
    if spheres is None:
        return []
    if not isinstance(spheres, (list, tuple)):
        spheres = [spheres]

    resolved: list[_SphereData] = []
    for index, sphere in enumerate(spheres):
        if not isinstance(sphere, dict):
            raise ValueError("Each sphere must be a dict.")
        if "positions" in sphere:
            positions = np.asarray(sphere["positions"], dtype=np.float32)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("sphere['positions'] must have shape (T, 3).")
        elif "position" in sphere:
            position = np.asarray(sphere["position"], dtype=np.float32)
            if position.shape != (3,):
                raise ValueError("sphere['position'] must have shape (3,).")
            positions = position[None, :]
        else:
            raise ValueError("Each sphere must define 'position' or 'positions'.")

        color = sphere.get("color", (230, 57, 70))
        if isinstance(color, int):
            color = ((color >> 16) & 255, (color >> 8) & 255, color & 255)

        resolved.append(
            _SphereData(
                positions=positions,
                radius=float(sphere.get("radius", 0.04)),
                color=tuple(int(component) for component in color),
                label=str(sphere.get("label", f"sphere_{index}")),
            )
        )
    return resolved


def _edges_from_parents(parents: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(parent), int(index))
        for index, parent in enumerate(parents)
        if int(parent) >= 0
    ]


def _parents_from_edges(edges: Sequence[tuple[int, int]], joint_count: int) -> np.ndarray:
    parents = np.full(joint_count, -1, dtype=np.int64)
    for parent, child in edges:
        parents[int(child)] = int(parent)
    return parents


def _resolve_clip(
    clip: BVHDocument | np.ndarray,
    *,
    offset: np.ndarray,
    label: str,
    edges: Sequence[tuple[int, int]] | None,
    names: Sequence[str] | None,
    parents: Sequence[int] | np.ndarray | None,
    fps: int | None,
) -> _ClipData:
    if isinstance(clip, BVHDocument):
        document = clip
        joints = compute_global_positions(document).astype(np.float32, copy=False)
        parents_resolved = np.asarray([joint.parent for joint in document.joints], dtype=np.int64)
        return _ClipData(
            joints=joints,
            names=list(document.joint_names),
            edges=_edges_from_parents(parents_resolved),
            fps=fps or _fps_from_document(document),
            label=label,
            offset=offset,
        )

    joints = np.asarray(clip, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("Each clip must be a BVHDocument or array [T, J, 3].")
    if parents is None and edges is None:
        raise ValueError("Pass either parents or edges for raw joint arrays.")
    parents_resolved = (
        np.asarray(parents, dtype=np.int64)
        if parents is not None
        else _parents_from_edges(edges or [], joints.shape[1])
    )
    return _ClipData(
        joints=joints,
        names=list(names or [str(i) for i in range(joints.shape[1])]),
        edges=list(edges or _edges_from_parents(parents_resolved)),
        fps=int(fps or 30),
        label=label,
        offset=offset,
    )


class SkeletonViewer:
    """Minimal Viser viewer for one or more clips."""

    def __init__(
        self,
        clips,
        *,
        edges: Sequence[Sequence[tuple[int, int]]] | Sequence[tuple[int, int]] | None = None,
        names: Sequence[Sequence[str]] | Sequence[str] | None = None,
        parents: Sequence[Sequence[int]] | Sequence[int] | None = None,
        offsets: Sequence[Sequence[float]] | Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        joint_radius: float = 0.02,
        fps: int | None = None,
        port: int = 8080,
        spheres=None,
        dark_mode=False,
        reuse_port: bool = True,
    ) -> None:
        _require_viser()

        inputs = _as_sequence(clips)
        count = len(inputs)
        offsets_resolved = _resolve_offsets(offsets, count)
        labels_resolved = list(labels) if labels is not None else [f"clip_{index}" for index in range(count)]
        if len(labels_resolved) != count:
            raise ValueError(f"Expected {count} labels, got {len(labels_resolved)}.")

        edges_seq = _normalize_optional_sequence(edges, count)
        names_seq = _normalize_optional_sequence(names, count)
        parents_seq = _normalize_optional_sequence(parents, count)

        self.clips = [
            _resolve_clip(
                clip,
                offset=offsets_resolved[index],
                label=labels_resolved[index],
                edges=edges_seq[index],
                names=names_seq[index],
                parents=parents_seq[index],
                fps=fps,
            )
            for index, clip in enumerate(inputs)
        ]
        self.spheres = _resolve_spheres(spheres)

        counts = [clip.frame_count for clip in self.clips]
        counts.extend(sphere.frame_count for sphere in self.spheres)
        self.frame_count = max(counts)
        self.fps = fps or self.clips[0].fps
        self.port = int(port)
        self.reuse_port = bool(reuse_port)
        self._t = 0
        self._playing = False
        self._closed = False
        self._selected = (-1, -1)
        self.dark_mode = dark_mode
        previous_viewer = None
        if self.reuse_port:
            previous_viewer = _ACTIVE_VIEWERS_BY_PORT.get(self.port)
            if previous_viewer is not None:
                previous_viewer.close()
            self.server = _SERVER_REGISTRY.get(self.port)
        else:
            self.server = None
        if self.server is None:
            self.server = viser.ViserServer(port=self.port)
            if self.reuse_port:
                _SERVER_REGISTRY[self.port] = self.server

        self._build_gui()
        self._create_scene(float(joint_radius))
        self.set_frame(0)
        if self.reuse_port:
            _ACTIVE_VIEWERS_BY_PORT[self.port] = self

        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _build_gui(self) -> None:
        joint_options = ["None"]
        self._joint_value_map = {"None": (-1, -1)}
        for clip_index, clip in enumerate(self.clips):
            for joint_index, name in enumerate(clip.names):
                key = f"{clip.label}:{name}"
                joint_options.append(key)
                self._joint_value_map[key] = (clip_index, joint_index)

        self.play_btn = self.server.gui.add_button("Play / Pause")
        self.frame_slider = self.server.gui.add_slider("Frame", min=0, max=self.frame_count - 1, step=1, initial_value=0)
        self.joint_dropdown = self.server.gui.add_dropdown("Joint", options=joint_options, initial_value="None")
        self.command_space = self.server.gui.add_command("Play / Pause", hotkey="space")
        self.info_text = self.server.gui.add_markdown("Ready.")

        self.play_btn.on_click(self._toggle_play)
        self.command_space.on_trigger(self._toggle_play)
        self.frame_slider.on_update(self._on_frame_change)
        self.joint_dropdown.on_update(self._on_joint_change)

    def _create_scene(self, joint_radius: float) -> None:
        self.server.scene.set_up_direction(direction="+y")
        self.server.scene.add_grid(name="floor", plane="xz", cell_size=0.1)
        self.joint_nodes: list[list[object]] = []
        self.bone_handles: list[object] = []
        self._bone_points: list[np.ndarray] = []
        self.sphere_nodes: list[object] = []

        for clip_index, clip in enumerate(self.clips):
            clip_nodes = []
            color = CLIP_COLORS[clip_index % len(CLIP_COLORS)]
            for joint_index in range(clip.joint_count):
                node = self.server.scene.add_icosphere(
                    name=f"/clips/{clip_index}/joints/{joint_index}",
                    radius=joint_radius,
                    color=color,
                )
                clip_nodes.append(node)
            self.joint_nodes.append(clip_nodes)

            points = np.zeros((len(clip.edges), 2, 3), dtype=np.float32)
            for edge_index, (parent, child) in enumerate(clip.edges):
                points[edge_index, 0] = clip.joints[0, parent] + clip.offset
                points[edge_index, 1] = clip.joints[0, child] + clip.offset
            self._bone_points.append(points)
            handle = self.server.scene.add_line_segments(
                name=f"/clips/{clip_index}/bones",
                points=points,
                colors=np.array(color, dtype=np.uint8),
                line_width=2,
            )
            self.bone_handles.append(handle)

        for sphere_index, sphere in enumerate(self.spheres):
            node = self.server.scene.add_icosphere(
                name=f"/spheres/{sphere_index}",
                radius=sphere.radius,
                color=sphere.color,
            )
            self.sphere_nodes.append(node)

    def set_frame(self, frame_index: int) -> None:
        self._t = int(np.clip(frame_index, 0, self.frame_count - 1))

        for clip_index, clip in enumerate(self.clips):
            clip_frame = min(self._t, clip.frame_count - 1)
            points = clip.joints[clip_frame] + clip.offset[None, :]
            for joint_index in range(clip.joint_count):
                self.joint_nodes[clip_index][joint_index].position = points[joint_index]
            for edge_index, (parent, child) in enumerate(clip.edges):
                self._bone_points[clip_index][edge_index, 0] = points[parent]
                self._bone_points[clip_index][edge_index, 1] = points[child]
            self.bone_handles[clip_index].points = self._bone_points[clip_index]

        for sphere_index, sphere in enumerate(self.spheres):
            sphere_frame = min(self._t, sphere.frame_count - 1)
            self.sphere_nodes[sphere_index].position = sphere.positions[sphere_frame]

        self._apply_selection()
        self._update_info()

    def _apply_selection(self) -> None:
        for clip_index, clip in enumerate(self.clips):
            base_color = CLIP_COLORS[clip_index % len(CLIP_COLORS)]
            for joint_index in range(clip.joint_count):
                self.joint_nodes[clip_index][joint_index].color = (
                    SELECTED_COLOR if (clip_index, joint_index) == self._selected else base_color
                )

    def _update_info(self) -> None:
        clip_index, joint_index = self._selected
        if clip_index < 0 or joint_index < 0:
            self.info_text.value = f"Frame {self._t}/{self.frame_count - 1}"
            return

        clip = self.clips[clip_index]
        clip_frame = min(self._t, clip.frame_count - 1)
        position = clip.joints[clip_frame, joint_index] + clip.offset
        self.info_text.value = (
            f"### Clip: {clip.label}\n\n"
            f"**Frame:** {self._t}/{self.frame_count - 1}  \n"
            f"**Joint:** {clip.names[joint_index]}  \n"
            f"**Position:** [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
        )

    def _toggle_play(self, _event) -> None:
        self._playing = not self._playing

    def _play_loop(self) -> None:
        while not self._closed:
            if not self._playing:
                time.sleep(0.03)
                continue
            self.frame_slider.value = int((self._t + 1) % self.frame_count)
            time.sleep(1.0 / float(max(self.fps, 1)))

    def _on_frame_change(self, event) -> None:
        self.set_frame(int(event.target.value))

    def _on_joint_change(self, event) -> None:
        self._selected = self._joint_value_map[event.target.value]
        self.set_frame(self._t)

    def show(self) -> None:
        self.server.scene.show(dark_mode=self.dark_mode)

    def close(self) -> None:
        self._closed = True
        self._playing = False
        for handle in (
            getattr(self, "play_btn", None),
            getattr(self, "frame_slider", None),
            getattr(self, "joint_dropdown", None),
            getattr(self, "command_space", None),
            getattr(self, "info_text", None),
        ):
            if handle is not None and hasattr(handle, "remove"):
                handle.remove()
        for clip_nodes in getattr(self, "joint_nodes", []):
            for node in clip_nodes:
                if hasattr(node, "remove"):
                    node.remove()
        for handle in getattr(self, "bone_handles", []):
            if hasattr(handle, "remove"):
                handle.remove()
        for node in getattr(self, "sphere_nodes", []):
            if hasattr(node, "remove"):
                node.remove()
        active_viewer = _ACTIVE_VIEWERS_BY_PORT.get(getattr(self, "port", -1))
        if active_viewer is self:
            _ACTIVE_VIEWERS_BY_PORT.pop(self.port, None)


def _normalize_optional_sequence(value, count: int):
    if value is None:
        return [None] * count
    if count == 1:
        return [value]
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"Expected a sequence with {count} items.")
    return list(value)
