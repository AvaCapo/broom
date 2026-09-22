"""Meshcat BVH viewer for one or more skeletons."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from broom.bvh.kinematics import compute_global_positions
from broom.bvh.schemas import BVHDocument

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:  # pragma: no cover - notebook-only dependency
    widgets = None
    display = None

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
except Exception:  # pragma: no cover - optional runtime dependency
    meshcat = None
    g = None
    tf = None


THEMES = {
    "light": {
        "bg_top": [1.00, 1.00, 1.00],
        "bg_bottom": [0.96, 0.97, 0.98],
        "floor_hex": 0xCBD3DD,
    },
    "dark": {
        "bg_top": [0.22, 0.24, 0.27],
        "bg_bottom": [0.16, 0.17, 0.20],
        "floor_hex": 0x4A5563,
    },
}

CLIP_COLORS = [
    0x5B6675,
    0xD95F02,
    0x1B9E77,
    0x7570B3,
    0xE7298A,
    0x66A61E,
]
SELECTED_COLOR = 0x00C2FF
_VISUALIZER_SINGLETON = None
_ACTIVE_VIEWER: "SkeletonViewer | None" = None


@dataclass(frozen=True)
class _ClipData:
    joints: np.ndarray
    names: list[str]
    parents: np.ndarray
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
    color: int
    label: str

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])


def _require_dependencies() -> None:
    """Raise install instructions when Meshcat viewer dependencies are absent."""

    if meshcat is None or g is None or tf is None:
        raise ImportError(
            "MeshcatSkeletonViewer requires meshcat. See docs/rendering.md "
            "for the tested version and upstream installation instructions."
        )
    if widgets is None or display is None:
        raise ImportError(
            "MeshcatSkeletonViewer requires ipywidgets and IPython. See "
            "docs/rendering.md for the supported setup."
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

        resolved.append(
            _SphereData(
                positions=positions,
                radius=float(sphere.get("radius", 0.04)),
                color=int(sphere.get("color", 0xE63946)),
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
            parents=parents_resolved,
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
        parents=parents_resolved,
        edges=list(edges or _edges_from_parents(parents_resolved)),
        fps=int(fps or 30),
        label=label,
        offset=offset,
    )


def _make_transform(translation: np.ndarray) -> np.ndarray:
    transform = tf.identity_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _cylinder_transform_between(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta)) + 1.0e-9
    center = (p0 + p1) / 2.0
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    direction = delta / length
    axis = np.cross(y_axis, direction)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1.0e-9:
        rotation = tf.identity_matrix() if float(np.dot(y_axis, direction)) > 0.0 else tf.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm
        angle = float(np.arccos(np.clip(np.dot(y_axis, direction), -1.0, 1.0)))
        rotation = tf.rotation_matrix(angle, axis)

    return tf.concatenate_matrices(_make_transform(center), rotation), length


class SkeletonViewer:
    """Interactive Meshcat viewer for one or more clips."""

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
        bone_radius: float = 0.005,
        fps: int | None = None,
        display_mode: str = "jupyter",
        theme: str = "light",
        plane_size: float = 10.0,
        plane_axis: str = "xz",
        plane_offset: float = 0.0,
        spheres=None,
        reuse_connection: bool = True,
    ) -> None:
        _require_dependencies()

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
        self.joint_radius = float(joint_radius)
        self.bone_radius = float(bone_radius)
        self.theme = THEMES[theme]
        self.reuse_connection = bool(reuse_connection)
        self._t = 0
        self._playing = False
        self._closed = False
        self._internal_slider_update = False
        self._selected = (-1, -1)
        global _VISUALIZER_SINGLETON, _ACTIVE_VIEWER
        if self.reuse_connection and _ACTIVE_VIEWER is not None:
            _ACTIVE_VIEWER.close()
        if self.reuse_connection and _VISUALIZER_SINGLETON is not None:
            self.vis = _VISUALIZER_SINGLETON
        else:
            self.vis = meshcat.Visualizer()
            if self.reuse_connection:
                _VISUALIZER_SINGLETON = self.vis
        self._configure_scene(display_mode, plane_size, plane_axis.lower(), float(plane_offset))
        self._build_widgets()
        self._init_scene()
        self.set_frame(0)
        if self.reuse_connection:
            _ACTIVE_VIEWER = self

        try:
            self._task = asyncio.create_task(self._play_loop())
        except RuntimeError:
            self._task = None

    def _configure_scene(self, display_mode: str, plane_size: float, plane_axis: str, plane_offset: float) -> None:
        # Match the previous Meshcat viewer orientation: Y up, Z forward.
        self.vis.set_transform(
            matrix=np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ).T
        )
        self.vis["/Background"].set_property("top_color", self.theme["bg_top"])
        self.vis["/Background"].set_property("bottom_color", self.theme["bg_bottom"])
        self.vis["/Grid"].set_property("visible", False)

        floor_thickness = 0.001
        if plane_axis == "xz":
            floor_dims = [plane_size, floor_thickness, plane_size]
            floor_transform = tf.identity_matrix()
            floor_transform[1, 3] = plane_offset - 0.5 * floor_thickness
        elif plane_axis == "xy":
            floor_dims = [plane_size, plane_size, floor_thickness]
            floor_transform = tf.identity_matrix()
            floor_transform[2, 3] = plane_offset - 0.5 * floor_thickness
        elif plane_axis == "yz":
            floor_dims = [floor_thickness, plane_size, plane_size]
            floor_transform = tf.identity_matrix()
            floor_transform[0, 3] = plane_offset - 0.5 * floor_thickness
        else:
            raise ValueError("plane_axis must be one of: 'xy', 'xz', 'yz'.")

        self.vis["floor"].set_object(
            g.Box(floor_dims),
            g.MeshLambertMaterial(color=self.theme["floor_hex"], opacity=0.25, transparent=True),
        )
        self.vis["floor"].set_transform(floor_transform)

        if display_mode == "browser":
            self.vis.open()
        else:
            display(self.vis.jupyter_cell())

    def _build_widgets(self) -> None:
        joint_options = [("None", "-1:-1")]
        for clip_index, clip in enumerate(self.clips):
            for joint_index, name in enumerate(clip.names):
                joint_options.append((f"{clip.label}:{name}", f"{clip_index}:{joint_index}"))

        self.play_btn = widgets.ToggleButton(value=False, description="Play", icon="play")
        self.frame_slider = widgets.IntSlider(value=0, min=0, max=self.frame_count - 1, step=1, description="Frame")
        self.fps_slider = widgets.IntSlider(value=int(self.fps), min=1, max=120, step=1, description="FPS")
        self.joint_dropdown = widgets.Dropdown(options=joint_options, value="-1:-1", description="Joint")
        self.info_out = widgets.HTML(value="")

        display(widgets.VBox([
            widgets.HBox([self.play_btn, self.fps_slider]),
            self.frame_slider,
            self.joint_dropdown,
            self.info_out,
        ]))

        self.play_btn.observe(self._on_play_toggle, names="value")
        self.fps_slider.observe(self._on_fps_change, names="value")
        self.frame_slider.observe(self._on_slider_change, names="value")
        self.joint_dropdown.observe(self._on_joint_change, names="value")

    def _init_scene(self) -> None:
        self._sphere = g.Sphere(self.joint_radius)
        self._materials = [g.MeshLambertMaterial(color=color) for color in CLIP_COLORS]
        self._selected_material = g.MeshLambertMaterial(color=SELECTED_COLOR)
        self._bone_materials = [g.MeshLambertMaterial(color=color) for color in CLIP_COLORS]

        for clip_index, clip in enumerate(self.clips):
            material = self._materials[clip_index % len(self._materials)]
            bone_material = self._bone_materials[clip_index % len(self._bone_materials)]
            cylinder = g.Cylinder(1.0, self.bone_radius)
            for joint_index in range(clip.joint_count):
                self.vis[f"clips/{clip_index}/joints/{joint_index}"].set_object(self._sphere, material)
            for edge_index, _ in enumerate(clip.edges):
                self.vis[f"clips/{clip_index}/bones/{edge_index}"].set_object(cylinder, bone_material)

        for sphere_index, sphere in enumerate(self.spheres):
            self.vis[f"spheres/{sphere_index}"].set_object(
                g.Sphere(sphere.radius),
                g.MeshLambertMaterial(color=sphere.color),
            )

    def set_frame(self, frame_index: int) -> None:
        self._t = int(np.clip(frame_index, 0, self.frame_count - 1))

        for clip_index, clip in enumerate(self.clips):
            clip_frame = min(self._t, clip.frame_count - 1)
            points = clip.joints[clip_frame] + clip.offset[None, :]
            for joint_index in range(clip.joint_count):
                self.vis[f"clips/{clip_index}/joints/{joint_index}"].set_transform(_make_transform(points[joint_index]))
            for edge_index, (parent, child) in enumerate(clip.edges):
                transform, length = _cylinder_transform_between(points[parent], points[child])
                scale = tf.identity_matrix()
                scale[1, 1] = length
                self.vis[f"clips/{clip_index}/bones/{edge_index}"].set_transform(tf.concatenate_matrices(transform, scale))

        for sphere_index, sphere in enumerate(self.spheres):
            sphere_frame = min(self._t, sphere.frame_count - 1)
            self.vis[f"spheres/{sphere_index}"].set_transform(
                _make_transform(sphere.positions[sphere_frame])
            )

        self._apply_selection()
        self._update_info()

    def _apply_selection(self) -> None:
        for clip_index, clip in enumerate(self.clips):
            material = self._materials[clip_index % len(self._materials)]
            for joint_index in range(clip.joint_count):
                selected = (clip_index, joint_index) == self._selected
                self.vis[f"clips/{clip_index}/joints/{joint_index}"].set_object(
                    self._sphere,
                    self._selected_material if selected else material,
                )

    def _update_info(self) -> None:
        clip_index, joint_index = self._selected
        if clip_index < 0 or joint_index < 0:
            self.info_out.value = f"<b>Frame:</b> {self._t}/{self.frame_count - 1}"
            return

        clip = self.clips[clip_index]
        clip_frame = min(self._t, clip.frame_count - 1)
        position = clip.joints[clip_frame, joint_index] + clip.offset
        self.info_out.value = (
            f"<b>Frame:</b> {self._t}/{self.frame_count - 1} &nbsp; "
            f"<b>Clip:</b> {clip.label} &nbsp; "
            f"<b>Joint:</b> {clip.names[joint_index]}<br>"
            f"<b>Position:</b> [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
        )

    def _on_play_toggle(self, change: dict[str, object]) -> None:
        self._playing = bool(change["new"])
        self.play_btn.description = "Pause" if self._playing else "Play"
        self.play_btn.icon = "pause" if self._playing else "play"

    def _on_fps_change(self, change: dict[str, object]) -> None:
        self.fps = max(1, int(change["new"]))

    def _on_slider_change(self, change: dict[str, object]) -> None:
        if self._internal_slider_update:
            self.set_frame(int(change["new"]))
            return
        self._playing = False
        self.play_btn.value = False
        self.set_frame(int(change["new"]))

    def _on_joint_change(self, change: dict[str, object]) -> None:
        value = str(change["new"])
        clip_index, joint_index = (int(part) for part in value.split(":"))
        self._selected = (clip_index, joint_index)
        self.set_frame(self._t)

    async def _play_loop(self) -> None:
        while not self._closed:
            if self._playing:
                next_frame = (self._t + 1) % self.frame_count
                self._internal_slider_update = True
                self.frame_slider.value = next_frame
                self._internal_slider_update = False
                await asyncio.sleep(1.0 / float(max(self.fps, 1)))
            else:
                await asyncio.sleep(0.05)

    def close(self) -> None:
        self._closed = True
        self._playing = False
        task = getattr(self, "_task", None)
        if task is not None:
            task.cancel()
            self._task = None
        for path in ("clips", "spheres", "floor"):
            try:
                self.vis[path].delete()
            except Exception:
                pass
        global _ACTIVE_VIEWER
        if _ACTIVE_VIEWER is self:
            _ACTIVE_VIEWER = None


def _normalize_optional_sequence(value, count: int):
    if value is None:
        return [None] * count
    if count == 1:
        return [value]
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"Expected a sequence with {count} items.")
    return list(value)
