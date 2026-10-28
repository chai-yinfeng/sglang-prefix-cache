from __future__ import annotations

from sglang.srt.managers.scheduler_components.spec_policy.types import SpecPolicyPlan


class SpecPolicyActuator:
    """Applies plan changes through the engine's existing runtime knobs."""

    def __init__(self, *, draft_worker) -> None:
        self._draft_worker = draft_worker

    def apply(self, *, plan: SpecPolicyPlan, current: SpecPolicyPlan) -> None:
        if plan.dspark_verify_budget_frac != current.dspark_verify_budget_frac:
            self._draft_worker.set_dspark_forced_budget_frac(
                plan.dspark_verify_budget_frac
            )
