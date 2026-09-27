"""Joint mapping helpers for skeleton retargeting."""

from __future__ import annotations

import json
from pathlib import Path
import re

from broom.channels import joint_name_matches
from broom import Hierarchy
from broom.retargeting.schemas import JointMatch, MappingResult


def build_joint_mapping(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    joint_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a source-to-target mapping for two hierarchies.

    Explicit entries win. Remaining joints are matched by normalized names and
    the namespace-tolerant matcher already used elsewhere in the BVH package.
    """

    return map_joints(
        source_hierarchy=source_hierarchy,
        target_hierarchy=target_hierarchy,
        joint_map=joint_map,
    ).mapping


def map_joints(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    joint_map: dict[str, str] | None = None,
    min_score: float = 0.65,
) -> MappingResult:
    """Map source joints to target joints with match confidence metadata.

    The mapper prefers explicit matches, then exact normalized names,
    namespace-tolerant matching, and finally substring matches. Each target can
    be used only once.
    """

    mapping, explicit_matches = _validate_explicit_mapping(
        source_hierarchy=source_hierarchy,
        target_hierarchy=target_hierarchy,
        joint_map=joint_map or {},
    )

    used_targets = set(mapping.values())
    source_names = source_hierarchy.joint_names
    target_names = target_hierarchy.joint_names
    source_order = {name: index for index, name in enumerate(source_names)}
    matches = list(explicit_matches)

    candidates = []
    for source_name in source_names:
        if source_name in mapping:
            continue
        for target_name in target_names:
            if target_name in used_targets:
                continue
            match = score_joint_match(source_name, target_name)
            if match.score >= min_score:
                candidates.append(match)

    candidates.sort(key=lambda match: match.score, reverse=True)
    used_sources = set(mapping)
    for match in candidates:
        if match.source in used_sources or match.target in used_targets:
            continue
        mapping[match.source] = match.target
        matches.append(match)
        used_sources.add(match.source)
        used_targets.add(match.target)

    return MappingResult(
        mapping=mapping,
        matches=tuple(sorted(matches, key=lambda match: source_order[match.source])),
        unmapped_sources=unmapped_sources(source_hierarchy, mapping),
        unmapped_targets=unmapped_targets(target_hierarchy, mapping),
    )


def score_joint_match(source_name: str, target_name: str) -> JointMatch:
    """Score a possible source-target joint pair."""

    source_key = _joint_key(source_name)
    target_key = _joint_key(target_name)

    if source_key == target_key:
        return JointMatch(source_name, target_name, 1.0, "exact")
    if joint_name_matches(target_name, source_name):
        return JointMatch(source_name, target_name, 0.90, "namespace")

    if _side(source_key) != _side(target_key):
        return JointMatch(source_name, target_name, 0.0, "side_mismatch")

    substring_score = _substring_score(source_key, target_key)
    if substring_score > 0.0:
        return JointMatch(
            source_name,
            target_name,
            substring_score,
            "substring",
        )

    token_score = _token_overlap_score(source_key, target_key)
    if token_score > 0.0:
        return JointMatch(source_name, target_name, token_score, "token_overlap")

    return JointMatch(source_name, target_name, 0.0, "none")


def invert_mapping(source_to_target: dict[str, str]) -> dict[str, str]:
    """Return ``target -> source`` and reject duplicate target mappings."""

    target_to_source = {}
    for source, target in source_to_target.items():
        if target in target_to_source:
            raise ValueError(
                "Multiple source joints map to target joint "
                f"{target!r}: {target_to_source[target]!r} and {source!r}."
            )
        target_to_source[target] = source
    return target_to_source


def _joint_key(name: str) -> str:
    """Normalize a joint name for fuzzy skeleton matching."""

    short_name = name.split(":")[-1]
    normalized = re.sub(r"[^a-z0-9]", "", short_name.lower())
    return normalized.replace("mixamorig", "")


def load_joint_mapping(path: str | Path) -> dict[str, str]:
    """Load a source-to-target joint mapping from a JSON object."""

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Joint mapping file must contain a JSON object.")
    return {str(source): str(target) for source, target in data.items()}


def unmapped_sources(
    source_hierarchy: Hierarchy,
    source_to_target: dict[str, str],
) -> tuple[str, ...]:
    """Return source joints with no target mapping."""

    return tuple(
        name for name in source_hierarchy.joint_names if name not in source_to_target
    )


def unmapped_targets(
    target_hierarchy: Hierarchy,
    source_to_target: dict[str, str],
) -> tuple[str, ...]:
    """Return target joints with no source mapping."""

    mapped_targets = set(source_to_target.values())
    return tuple(
        name for name in target_hierarchy.joint_names if name not in mapped_targets
    )


def _validate_explicit_mapping(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    joint_map: dict[str, str],
) -> tuple[dict[str, str], tuple[JointMatch, ...]]:
    source_names = set(source_hierarchy.joint_names)
    target_names = set(target_hierarchy.joint_names)
    mapping = {}
    matches = []

    for source, target in joint_map.items():
        source_name = _resolve_joint_name(source, source_names, "source")
        target_name = _resolve_joint_name(target, target_names, "target")
        mapping[source_name] = target_name
        matches.append(JointMatch(source_name, target_name, 1.0, "explicit"))

    invert_mapping(mapping)
    return mapping, tuple(matches)


def _resolve_joint_name(name: str, names: set[str], label: str) -> str:
    if name in names:
        return name

    matches = [candidate for candidate in names if joint_name_matches(candidate, name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {label} joint {name!r}; matched {', '.join(matches)}."
        )
    raise ValueError(f"Unknown {label} joint {name!r}.")


def _side(name: str) -> str | None:
    if name.startswith("left"):
        return "left"
    if name.startswith("right"):
        return "right"
    return None


def _substring_score(source_name: str, target_name: str) -> float:
    if source_name == target_name:
        return 1.0
    shorter, longer = sorted((source_name, target_name), key=len)
    if len(shorter) == len(longer):
        return 0.0
    if len(shorter) < 4 or shorter not in longer:
        return 0.0
    return max(0.65, min(0.86, len(shorter) / len(longer)))


def _token_overlap_score(source_name: str, target_name: str) -> float:
    source_tokens = set(_split_name_tokens(source_name))
    target_tokens = set(_split_name_tokens(target_name))
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = source_tokens & target_tokens
    if not overlap:
        return 0.0
    return max(0.0, min(0.78, len(overlap) / len(source_tokens | target_tokens)))


def _split_name_tokens(name: str) -> tuple[str, ...]:
    pattern = r"(left|right|upper|lower|spine|arm|leg|foot|hand|neck|head|toe)"
    return tuple(token for token in re.split(pattern, name) if token)
