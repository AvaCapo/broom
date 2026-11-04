# BROOM: BVH Read, Optimize, Operate and Manipulate

BROOM is a Python library for working with skeletons and motion clips. Load BVH, edit animation, compute joint transforms, transfer motion between skeletons, and inspect the result in an interactive viewer.

## Start here

1. [Install Broom](getting-started/installation.md).
2. [Create, save, and inspect a motion](getting-started/quickstart.md).
3. Understand the [data model](concepts/data-model.md) and [coordinate conventions](concepts/coordinates.md).

## Choose a task

| Task | Guide |
|---|---|
| Read or write BVH | [I/O](guides/io.md) |
| Edit samples or skeleton dimensions | [Editing](guides/editing.md) |
| Compute world transforms or solve a chain | [Kinematics](guides/kinematics.md) |
| Change FPS or blend clips | [Interpolation and resampling](guides/interpolation.md) |
| Remove locomotion or align a clip to the floor | [Root motion](guides/root-motion.md) |
| Map joints and transfer animation | [Retargeting](guides/retargeting.md) |
| Correct motion with temporal constraints | [Space-time usage](spacetime_usage.md) |
| Compare clips interactively | [Visualization](visualization.md) |

Use the **API reference** for signatures and parameter details. The package is under active development; examples describe the current API, not legacy imports.

Broom uses the [PolyForm Noncommercial 1.0.0 license](https://github.com/AvaCapo/broom/blob/main/LICENSE). Commercial use requires a separate license.
