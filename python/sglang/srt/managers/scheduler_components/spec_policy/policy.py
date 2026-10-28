from __future__ import annotations

from typing import Protocol

from sglang.srt.managers.scheduler_components.spec_policy.types import (
    Observation,
    SpecPolicyPlan,
)


class SpecPolicy(Protocol):
    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan: ...


class ObservePolicy:
    """Never changes the plan; records observations for offline replay."""

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        return current


def build_spec_policy(name: str) -> SpecPolicy:
    if name == "observe":
        return ObservePolicy()
    raise ValueError(f"Unknown speculative policy {name!r}.")
