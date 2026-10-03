"""Joint mapping helpers for skeleton retargeting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import re

from broom.channels import joint_name_matches
from broom import Hierarchy
from broom.retargeting.schemas import JointMatch, MappingResult


def map_joints(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    known_joint_map: Mapping[str, str] | None = None,
    min_score: float = 0.65,
) -> MappingResult:
    """Map source joints to target joints with match confidence metadata.

    Known pairs are validated and retained. The mapper then prefers exact normalized names,
    namespace-tolerant matching, and finally substring matches. Each target can
    be used only once.
    """

    if known_joint_map is not None:
        validate_joint_mapping(
            source_hierarchy, target_hierarchy, known_joint_map
        )
    mapping = dict(known_joint_map or {})
    explicit_matches = tuple(
        JointMatch(source, target, 1.0, "explicit")
        for source, target in mapping.items()
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
        unmapped_sources=tuple(
            name for name in source_names if name not in mapping
        ),
        unmapped_targets=tuple(
            name for name in target_names if name not in used_targets
        ),
    )


def validate_joint_mapping(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    mapping: Mapping[str, str],
) -> None:
    """Raise ValueError when a joint mapping is not a literal one-to-one map."""
    if not isinstance(mapping, Mapping):
        raise ValueError("mapping must be a mapping of source joint names to target joint names.")

    source_names = set(source_hierarchy.joint_names)
    target_names = set(target_hierarchy.joint_names)
    target_to_source: dict[str, str] = {}
    for source_name, target_name in mapping.items():
        if not isinstance(source_name, str) or not isinstance(target_name, str):
            raise ValueError("mapping joint names must be strings.")
        if source_name not in source_names:
            raise ValueError(f"Unknown source joint {source_name!r}.")
        if target_name not in target_names:
            raise ValueError(f"Unknown target joint {target_name!r}.")
        if target_name in target_to_source:
            raise ValueError(
                "Multiple source joints map to target joint "
                f"{target_name!r}: {target_to_source[target_name]!r} and "
                f"{source_name!r}."
            )
        target_to_source[target_name] = source_name


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
    if not all(isinstance(source, str) and isinstance(target, str) for source, target in data.items()):
        raise ValueError("Joint mapping file must contain string joint names.")
    return dict(data)


def _side(name: str) -> str | None:
    """Return the side marker found anywhere in a normalized joint name."""

    has_left = "left" in name
    has_right = "right" in name
    if has_left == has_right:
        return None
    if has_left:
        return "left"
    if has_right:
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
