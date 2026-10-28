from __future__ import annotations

from typing import Callable, Optional, Sequence

from sglang.srt.managers.scheduler_components.spec_policy.types import SpecPolicyPlan


def request_weight(*, priority: Optional[int], low_priority_weight: float) -> float:
    return 1.0 if priority is not None and priority > 0 else low_priority_weight


class SpecPolicyActuator:
    """Applies plan changes through the engine's existing runtime knobs."""

    def __init__(
        self, *, draft_worker, grow_kv_pool: Optional[Callable[[int], int]] = None
    ) -> None:
        self._draft_worker = draft_worker
        self._grow_kv_pool = grow_kv_pool
        # req_pool_idx -> (rid, weight) last written; a reused slot has a new rid.
        self._written: dict[int, tuple[str, float]] = {}

    def apply(
        self, *, plan: SpecPolicyPlan, current: SpecPolicyPlan, reqs: Sequence
    ) -> None:
        if plan.spec_paused != current.spec_paused:
            self._draft_worker.set_dspark_spec_paused(bool(plan.spec_paused))
        if plan.verify_width is not None and plan.verify_width != current.verify_width:
            self._draft_worker.set_dflash_verify_width(plan.verify_width)
        if plan.dspark_verify_budget_frac != current.dspark_verify_budget_frac:
            self._draft_worker.set_dspark_forced_budget_frac(
                plan.dspark_verify_budget_frac
            )
        if plan.low_priority_weight is not None:
            self._write_request_weights(
                reqs=reqs, low_priority_weight=plan.low_priority_weight
            )
        elif current.low_priority_weight is not None:
            self._reset_request_weights()
        if (
            plan.target_kv_tokens is not None
            and plan.target_kv_tokens != current.target_kv_tokens
            and self._grow_kv_pool is not None
        ):
            self._grow_kv_pool(plan.target_kv_tokens)

    def _write_request_weights(
        self, *, reqs: Sequence, low_priority_weight: float
    ) -> None:
        slots, weights = [], []
        for req in reqs:
            slot = req.kv.req_pool_idx
            if slot is None:
                continue
            weight = request_weight(
                priority=req.priority, low_priority_weight=low_priority_weight
            )
            if self._written.get(slot) != (req.rid, weight):
                self._written[slot] = (req.rid, weight)
                slots.append(slot)
                weights.append(weight)
        if slots:
            self._draft_worker.set_dspark_request_weights(
                req_pool_indices=slots, weights=weights
            )

    def _reset_request_weights(self) -> None:
        slots = list(self._written)
        self._written.clear()
        if slots:
            self._draft_worker.set_dspark_request_weights(
                req_pool_indices=slots, weights=[1.0] * len(slots)
            )
