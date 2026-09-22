"""Notebook UI for BVH analysis plots."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import matplotlib.pyplot as plt

from broom.bvh.analysis.kinematics import (
    compute_world_kinematics,
)
from broom.bvh.analysis.plotting import (
    PANEL_PRESETS,
    SERIES,
    PlotConfig,
    plot_joint_debug,
    plot_joint_debug_multi,
    plot_joints_panel,
    plot_joints_panel_multi,
    plot_trajectory_planes,
    plot_trajectory_planes_multi,
)
from broom.bvh.analysis.schemas import WorldKinematicsResult
from broom.bvh.schemas import BVHDocument

try:
    import ipywidgets as widgets
    from IPython.display import Image, display
except Exception:  # pragma: no cover - notebook-only dependency
    widgets = None
    display = None
    Image = None


@dataclass
class MotionPlotUI:
    document: BVHDocument
    joint_labels: Optional[list[str] | tuple[str, ...]] = None
    method: str = "savgol"
    sg_window: int = 9
    sg_polyorder: int = 3
    sg_mode: str = "interp"
    prefilter: Optional[dict[str, float]] = None

    def show(self) -> None:
        _require_widgets()
        joint_labels = self._joint_labels()
        joint_opts = [(label, index) for index, label in enumerate(joint_labels)]
        frame_count = self.document.frame_count

        range_slider = widgets.IntRangeSlider(
            value=[0, frame_count - 1],
            min=0,
            max=frame_count - 1,
            step=1,
            description="Frames",
            continuous_update=False,
            layout=widgets.Layout(width="90%"),
        )
        method_dd = widgets.Dropdown(
            options=[("gradient", "gradient"), ("five_point", "five_point"), ("savgol", "savgol")],
            value=self.method,
            description="Diff",
        )
        sg_window = widgets.IntSlider(
            value=self.sg_window,
            min=5,
            max=51,
            step=2,
            description="SG win",
            continuous_update=False,
        )
        sg_poly = widgets.IntSlider(
            value=self.sg_polyorder,
            min=2,
            max=9,
            step=1,
            description="SG poly",
            continuous_update=False,
        )
        sg_mode = widgets.Dropdown(
            options=[("interp", "interp"), ("nearest", "nearest"), ("mirror", "mirror")],
            value=self.sg_mode,
            description="SG mode",
        )

        cache: dict[str, Any] = {"key": None, "kin": None}

        def t_range() -> tuple[int, int]:
            start, end = range_slider.value
            return int(start), int(end) + 1

        def get_kin() -> WorldKinematicsResult:
            key = (
                method_dd.value,
                int(sg_window.value),
                int(sg_poly.value),
                sg_mode.value,
                None if self.prefilter is None else tuple(sorted(self.prefilter.items())),
            )
            if cache["key"] == key and cache["kin"] is not None:
                return cache["kin"]
            kin = compute_world_kinematics(
                self.document,
                method=method_dd.value,
                sg_window=int(sg_window.value),
                sg_polyorder=int(sg_poly.value),
                sg_mode=sg_mode.value,
                prefilter=self.prefilter,
            )
            cache["key"] = key
            cache["kin"] = kin
            return kin

        def sync_sg_visibility(*_) -> None:
            is_savgol = method_dd.value == "savgol"
            sg_window.layout.display = "" if is_savgol else "none"
            sg_poly.layout.display = "" if is_savgol else "none"
            sg_mode.layout.display = "" if is_savgol else "none"

        joints_ms0 = widgets.SelectMultiple(
            options=joint_opts,
            value=(0,),
            description="Joints",
            rows=10,
        )
        preset0 = widgets.Dropdown(
            options=_preset_options(include_custom=True),
            value="mag" if "mag" in PANEL_PRESETS else "__custom__",
            description="Preset",
        )
        panels_ms0 = widgets.SelectMultiple(
            options=list(SERIES.keys()),
            value=("pos.mag", "vel.mag", "acc.mag"),
            description="Panels",
            rows=10,
        )
        out0 = widgets.Output()

        def sync_tab0_controls(*_) -> None:
            panels_ms0.disabled = preset0.value != "__custom__"

        def render_tab0(*_) -> None:
            kwargs = {
                "document": self.document,
                "joint_indices": list(joints_ms0.value),
                "kinematics": get_kin(),
                "t_range": t_range(),
                "joint_labels": joint_labels,
            }
            if preset0.value != "__custom__":
                fig = plot_joints_panel(
                    preset=preset0.value,
                    cfg=PlotConfig(title=f"joints_panel | preset={preset0.value}"),
                    **kwargs,
                )
            else:
                fig = plot_joints_panel(
                    panels=tuple(panels_ms0.value),
                    cfg=PlotConfig(title="joints_panel | custom"),
                    **kwargs,
                )
            _display_figure(out0, fig)

        joints_ms1 = widgets.SelectMultiple(
            options=joint_opts,
            value=(0,),
            description="Joints",
            rows=10,
        )
        color_by_time = widgets.Checkbox(value=False, description="Color by time")
        equal_aspect = widgets.Checkbox(value=True, description="Equal aspect")
        out1 = widgets.Output()

        def render_tab1(*_) -> None:
            fig = plot_trajectory_planes(
                self.document,
                list(joints_ms1.value),
                kinematics=get_kin(),
                signals=("pos", "vel", "acc"),
                t_range=t_range(),
                color_by_time=bool(color_by_time.value),
                equal_aspect=bool(equal_aspect.value),
                joint_labels=joint_labels,
                cfg=PlotConfig(figsize=(12, 12), title="trajectory_planes"),
            )
            _display_figure(out1, fig)

        joint_dd2 = widgets.Dropdown(options=joint_opts, value=0, description="Joint")
        preset2 = widgets.Dropdown(
            options=_preset_options(include_custom=True),
            value="pos" if "pos" in PANEL_PRESETS else "__custom__",
            description="Preset",
        )
        series_ms2 = widgets.SelectMultiple(
            options=list(SERIES.keys()),
            value=("pos.x", "pos.y", "pos.z"),
            description="Series",
            rows=10,
        )
        out2 = widgets.Output()

        def sync_tab2_controls(*_) -> None:
            series_ms2.disabled = preset2.value != "__custom__"

        def render_tab2(*_) -> None:
            kwargs = {
                "document": self.document,
                "joint_idx": int(joint_dd2.value),
                "kinematics": get_kin(),
                "t_range": t_range(),
            }
            if preset2.value != "__custom__":
                fig = plot_joint_debug(
                    preset=preset2.value,
                    cfg=PlotConfig(title=f"joint_debug | preset={preset2.value}"),
                    **kwargs,
                )
            else:
                fig = plot_joint_debug(
                    series=tuple(series_ms2.value),
                    cfg=PlotConfig(title="joint_debug | custom"),
                    **kwargs,
                )
            _display_figure(out2, fig)

        for control in (range_slider, method_dd, sg_window, sg_poly, sg_mode):
            control.observe(render_tab0, names="value")
            control.observe(render_tab1, names="value")
            control.observe(render_tab2, names="value")
        method_dd.observe(sync_sg_visibility, names="value")
        preset0.observe(sync_tab0_controls, names="value")
        preset0.observe(render_tab0, names="value")
        panels_ms0.observe(lambda change: _on_custom_select(change, preset0, render_tab0), names="value")
        joints_ms0.observe(render_tab0, names="value")
        joints_ms1.observe(render_tab1, names="value")
        color_by_time.observe(render_tab1, names="value")
        equal_aspect.observe(render_tab1, names="value")
        preset2.observe(sync_tab2_controls, names="value")
        preset2.observe(render_tab2, names="value")
        series_ms2.observe(lambda change: _on_custom_select(change, preset2, render_tab2), names="value")
        joint_dd2.observe(render_tab2, names="value")

        sync_sg_visibility()
        sync_tab0_controls()
        sync_tab2_controls()

        shared = widgets.VBox([
            range_slider,
            widgets.HBox([method_dd, sg_window, sg_poly, sg_mode]),
        ])
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox([widgets.HBox([joints_ms0, widgets.VBox([preset0, panels_ms0])]), out0]),
            widgets.VBox([widgets.HBox([joints_ms1, widgets.VBox([color_by_time, equal_aspect])]), out1]),
            widgets.VBox([widgets.HBox([joint_dd2, widgets.VBox([preset2, series_ms2])]), out2]),
        ]
        tab.set_title(0, "Joints panel")
        tab.set_title(1, "Trajectories")
        tab.set_title(2, "Joint debug")
        tab.observe(lambda change: _render_selected_tab(change, render_tab0, render_tab1, render_tab2), names="selected_index")

        display(widgets.VBox([shared, tab]))
        _render_selected_tab(None, render_tab0, render_tab1, render_tab2)

    def _joint_labels(self) -> list[str]:
        if self.joint_labels is not None:
            return [str(label) for label in self.joint_labels]
        return list(self.document.joint_names)


@dataclass
class MotionCompareUI:
    documents: list[BVHDocument] | tuple[BVHDocument, ...]
    clip_labels: Optional[list[str] | tuple[str, ...]] = None
    joint_labels: Optional[list[str] | tuple[str, ...]] = None
    method: str = "savgol"
    sg_window: int = 9
    sg_polyorder: int = 3
    sg_mode: str = "interp"
    prefilter: Optional[dict[str, float]] = None

    def show(self) -> None:
        _require_widgets()
        if len(self.documents) < 2:
            raise ValueError("MotionCompareUI requires at least 2 documents.")

        joint_count = len(self.documents[0].joints)
        if any(len(document.joints) != joint_count for document in self.documents):
            raise ValueError("All documents must have the same number of joints.")

        frame_count = min(document.frame_count for document in self.documents)
        labels = _normalize_labels(len(self.documents), self.clip_labels)
        joint_labels = (
            [str(label) for label in self.joint_labels]
            if self.joint_labels is not None
            else list(self.documents[0].joint_names)
        )
        joint_opts = [(label, index) for index, label in enumerate(joint_labels)]

        range_slider = widgets.IntRangeSlider(
            value=[0, frame_count - 1],
            min=0,
            max=frame_count - 1,
            step=1,
            description="Frames",
            continuous_update=False,
            layout=widgets.Layout(width="90%"),
        )
        method_dd = widgets.Dropdown(
            options=[("gradient", "gradient"), ("five_point", "five_point"), ("savgol", "savgol")],
            value=self.method,
            description="Diff",
        )
        sg_window = widgets.IntSlider(value=self.sg_window, min=5, max=51, step=2, description="SG win", continuous_update=False)
        sg_poly = widgets.IntSlider(value=self.sg_polyorder, min=2, max=9, step=1, description="SG poly", continuous_update=False)
        sg_mode = widgets.Dropdown(
            options=[("interp", "interp"), ("nearest", "nearest"), ("mirror", "mirror")],
            value=self.sg_mode,
            description="SG mode",
        )

        cache: dict[str, Any] = {"key": None, "kins": None}

        def t_range() -> tuple[int, int]:
            start, end = range_slider.value
            return int(start), int(end) + 1

        def get_kins() -> list[WorldKinematicsResult]:
            key = (
                method_dd.value,
                int(sg_window.value),
                int(sg_poly.value),
                sg_mode.value,
                None if self.prefilter is None else tuple(sorted(self.prefilter.items())),
            )
            if cache["key"] == key and cache["kins"] is not None:
                return cache["kins"]
            kins = [
                compute_world_kinematics(
                    document,
                    method=method_dd.value,
                    sg_window=int(sg_window.value),
                    sg_polyorder=int(sg_poly.value),
                    sg_mode=sg_mode.value,
                    prefilter=self.prefilter,
                )
                for document in self.documents
            ]
            cache["key"] = key
            cache["kins"] = kins
            return kins

        def sync_sg_visibility(*_) -> None:
            is_savgol = method_dd.value == "savgol"
            sg_window.layout.display = "" if is_savgol else "none"
            sg_poly.layout.display = "" if is_savgol else "none"
            sg_mode.layout.display = "" if is_savgol else "none"

        joints_ms0 = widgets.SelectMultiple(options=joint_opts, value=(0,), description="Joints", rows=10)
        preset0 = widgets.Dropdown(options=_preset_options(include_custom=True), value="mag", description="Preset")
        panels_ms0 = widgets.SelectMultiple(options=list(SERIES.keys()), value=("pos.mag", "vel.mag", "acc.mag"), description="Panels", rows=12)
        out0 = widgets.Output()

        def sync_tab0_controls(*_) -> None:
            panels_ms0.disabled = preset0.value != "__custom__"

        def render_tab0(*_) -> None:
            kwargs = {
                "documents": list(self.documents),
                "joint_indices": list(joints_ms0.value),
                "kinematics_results": get_kins(),
                "clip_labels": labels,
                "t_range": t_range(),
                "joint_labels": joint_labels,
            }
            if preset0.value != "__custom__":
                fig = plot_joints_panel_multi(
                    preset=preset0.value,
                    cfg=PlotConfig(title=f"compare | joints_panel | preset={preset0.value}"),
                    **kwargs,
                )
            else:
                fig = plot_joints_panel_multi(
                    panels=tuple(panels_ms0.value),
                    cfg=PlotConfig(title="compare | joints_panel | custom"),
                    **kwargs,
                )
            _display_figure(out0, fig)

        joints_ms1 = widgets.SelectMultiple(options=joint_opts, value=(0,), description="Joints", rows=10)
        color_by_time = widgets.Checkbox(value=False, description="Color by time")
        equal_aspect = widgets.Checkbox(value=True, description="Equal aspect")
        out1 = widgets.Output()

        def render_tab1(*_) -> None:
            fig = plot_trajectory_planes_multi(
                list(self.documents),
                list(joints_ms1.value),
                kinematics_results=get_kins(),
                clip_labels=labels,
                signals=("pos", "vel", "acc"),
                t_range=t_range(),
                color_by_time=bool(color_by_time.value),
                equal_aspect=bool(equal_aspect.value),
                joint_labels=joint_labels,
                cfg=PlotConfig(figsize=(12, 12), title="compare | trajectory_planes"),
            )
            _display_figure(out1, fig)

        joint_dd2 = widgets.Dropdown(options=joint_opts, value=0, description="Joint")
        preset2 = widgets.Dropdown(options=_preset_options(include_custom=True), value="pos", description="Preset")
        series_ms2 = widgets.SelectMultiple(options=list(SERIES.keys()), value=("pos.x", "pos.y", "pos.z"), description="Series", rows=12)
        out2 = widgets.Output()

        def sync_tab2_controls(*_) -> None:
            series_ms2.disabled = preset2.value != "__custom__"

        def render_tab2(*_) -> None:
            kwargs = {
                "documents": list(self.documents),
                "joint_idx": int(joint_dd2.value),
                "kinematics_results": get_kins(),
                "clip_labels": labels,
                "t_range": t_range(),
            }
            if preset2.value != "__custom__":
                fig = plot_joint_debug_multi(
                    preset=preset2.value,
                    cfg=PlotConfig(title=f"compare | joint_debug | preset={preset2.value}"),
                    **kwargs,
                )
            else:
                fig = plot_joint_debug_multi(
                    series=tuple(series_ms2.value),
                    cfg=PlotConfig(title="compare | joint_debug | custom"),
                    **kwargs,
                )
            _display_figure(out2, fig)

        for control in (range_slider, method_dd, sg_window, sg_poly, sg_mode):
            control.observe(render_tab0, names="value")
            control.observe(render_tab1, names="value")
            control.observe(render_tab2, names="value")
        method_dd.observe(sync_sg_visibility, names="value")
        preset0.observe(sync_tab0_controls, names="value")
        preset0.observe(render_tab0, names="value")
        panels_ms0.observe(lambda change: _on_custom_select(change, preset0, render_tab0), names="value")
        joints_ms0.observe(render_tab0, names="value")
        joints_ms1.observe(render_tab1, names="value")
        color_by_time.observe(render_tab1, names="value")
        equal_aspect.observe(render_tab1, names="value")
        preset2.observe(sync_tab2_controls, names="value")
        preset2.observe(render_tab2, names="value")
        series_ms2.observe(lambda change: _on_custom_select(change, preset2, render_tab2), names="value")
        joint_dd2.observe(render_tab2, names="value")

        sync_sg_visibility()
        sync_tab0_controls()
        sync_tab2_controls()

        shared = widgets.VBox([
            range_slider,
            widgets.HBox([method_dd, sg_window, sg_poly, sg_mode]),
        ])
        clips_summary = ", ".join(labels)
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox([widgets.HTML(value=f"<b>Clips:</b> {clips_summary}"), widgets.HBox([joints_ms0, widgets.VBox([preset0, panels_ms0])]), out0]),
            widgets.VBox([widgets.HTML(value=f"<b>Clips:</b> {clips_summary}"), widgets.HBox([joints_ms1, widgets.VBox([color_by_time, equal_aspect])]), out1]),
            widgets.VBox([widgets.HTML(value=f"<b>Clips:</b> {clips_summary}"), widgets.HBox([joint_dd2, widgets.VBox([preset2, series_ms2])]), out2]),
        ]
        tab.set_title(0, "Joints panel")
        tab.set_title(1, "Trajectories")
        tab.set_title(2, "Joint debug")
        tab.observe(lambda change: _render_selected_tab(change, render_tab0, render_tab1, render_tab2), names="selected_index")

        display(widgets.VBox([shared, tab]))
        _render_selected_tab(None, render_tab0, render_tab1, render_tab2)


def _require_widgets() -> None:
    if widgets is None or display is None or Image is None:
        raise RuntimeError("ipywidgets is not available in this environment.")


def _preset_options(include_custom: bool = True) -> list[tuple[str, str]]:
    options = [("Custom", "__custom__")] if include_custom else []
    options.extend((name, name) for name in PANEL_PRESETS)
    return options


def _display_figure(out: "widgets.Output", fig: plt.Figure) -> None:
    out.clear_output(wait=True)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    with out:
        display(Image(data=buffer.getvalue()))
    buffer.close()
    plt.close(fig)


def _on_custom_select(
    change: Any,
    preset_widget: "widgets.Dropdown",
    render: Any,
) -> None:
    if preset_widget.value != "__custom__":
        preset_widget.value = "__custom__"
    else:
        render()


def _render_selected_tab(
    change: Any,
    render_tab0: Any,
    render_tab1: Any,
    render_tab2: Any,
) -> None:
    selected = 0 if change is None else change["owner"].selected_index
    if selected == 0:
        render_tab0()
    elif selected == 1:
        render_tab1()
    else:
        render_tab2()


def _normalize_labels(
    count: int,
    labels: Optional[list[str] | tuple[str, ...]],
) -> list[str]:
    if labels is None:
        return [f"clip_{index}" for index in range(count)]
    if len(labels) != count:
        raise ValueError("clip_labels length must match number of documents.")
    return [str(label) for label in labels]
