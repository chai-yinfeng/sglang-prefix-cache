from __future__ import annotations

from typing import Optional, Protocol, Sequence

import msgspec

from sglang.srt.managers.scheduler_components.spec_policy.ledger import MemoryLedger
from sglang.srt.managers.scheduler_components.spec_policy.types import (
    Observation,
    SpecPolicyPlan,
)

# Measured, not derived: on Qwen3-8B / RTX 5090 / FA4, compact verify overtook full
# verify between c16 and c32 (about 200 verify tokens), where the SPS table runs at
# about 60% of its peak rate. Re-measure for another model or GPU.
_KNEE_RATE_FRACTION = 0.6
# Arbitrary; how far past the knee the load must go before handing over.
_HANDOVER_FACTOR = 1.5
# Arbitrary; about one second of decode at 8B c64 cadence.
_MIN_DWELL_STEPS = 16
# Arbitrary; low enough that a priority request's 2nd draft outranks a low one's 1st.
_LOW_PRIORITY_WEIGHT = 0.25
# Arbitrary; about half a minute of serving at 8B c64, long enough to see prefill and
# decode activation peaks before trusting the free-memory floor.
_HEADROOM_WINDOW_STEPS = 600
# Kept free on top of the observed floor. Measured: serving at a 0.74 GB floor ran
# without OOM (8B DSpark, post-capture sizing); 1 GiB adds slack for new shapes.
_HEADROOM_MARGIN_BYTES = 1 << 30
# Arbitrary; below this a growth is not worth a synchronize.
_MIN_GROWTH_TOKENS = 2048


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


def knee_tokens_from_sps(*, batch_tokens: list[int], steps_per_sec: list[float]) -> int:
    """Largest probed verify-token count whose step rate is within the knee fraction of peak."""
    peak = max(steps_per_sec)
    within = [
        tokens
        for tokens, rate in zip(batch_tokens, steps_per_sec)
        if rate >= _KNEE_RATE_FRACTION * peak
    ]
    return max(within)


class LoadAwareVerifyPolicy:
    """Verify every drafted position while the step is weight-bound; past the knee,
    let DSpark's confidence planner trim the verify budget.

    Below the knee a full verify costs about the same step time as a trimmed one,
    so trimming only loses accepted tokens. Hysteresis plus a minimum dwell keeps
    a batch hovering at the knee from flapping between the two modes.
    """

    FULL = SpecPolicyPlan(dspark_verify_budget_frac=1.0)
    PLANNER = SpecPolicyPlan()

    def __init__(self, *, knee_tokens: int) -> None:
        self._enter_full_below = knee_tokens
        self._leave_full_above = int(knee_tokens * _HANDOVER_FACTOR)
        self._steps_since_change = _MIN_DWELL_STEPS

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        if (
            observation.forward_mode != "DECODE"
            or observation.num_verify_tokens_per_req is None
        ):
            return current
        self._steps_since_change += 1
        if self._steps_since_change < _MIN_DWELL_STEPS:
            return current
        verify_tokens = observation.batch_size * observation.num_verify_tokens_per_req
        target = current
        if current != self.FULL and verify_tokens < self._enter_full_below:
            target = self.FULL
        elif current == self.FULL and verify_tokens > self._leave_full_above:
            target = self.PLANNER
        if target != current:
            self._steps_since_change = 0
        return target


class PriorityWeightPolicy:
    """Rank DSpark verify slots by accept probability x request weight.

    Requests with a positive priority keep weight 1; the rest are discounted, so when
    the confidence planner trims the verify budget under load, priority requests keep
    their deeper verify windows. Below the planner's trimming point the weights
    change nothing.
    """

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        return msgspec.structs.replace(
            current, low_priority_weight=_LOW_PRIORITY_WEIGHT
        )


class MemoryHeadroomPolicy:
    """Grow the KV pool into device memory that stays free under the real load.

    The startup split reserves activation headroom from a static fraction. This
    policy watches the free memory the load actually leaves (the caching allocator
    keeps its high-water reservations, so sampled free memory is a running floor)
    and, after a warmup window, hands what is left above a safety margin to the KV
    pool. Each growth restarts the window.
    """

    def __init__(self, *, ledger: Optional[MemoryLedger]) -> None:
        self._ledger = ledger
        self._steps_since_growth = 0
        self._min_free_bytes: Optional[int] = None

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        self._steps_since_growth += 1
        if observation.free_gpu_bytes is not None:
            self._min_free_bytes = (
                observation.free_gpu_bytes
                if self._min_free_bytes is None
                else min(self._min_free_bytes, observation.free_gpu_bytes)
            )
        if (
            self._ledger is None
            or observation.kv_capacity_tokens is None
            or self._min_free_bytes is None
            or self._steps_since_growth < _HEADROOM_WINDOW_STEPS
        ):
            return current
        extra_tokens = self._ledger.tokens_for_bytes(
            max(self._min_free_bytes - _HEADROOM_MARGIN_BYTES, 0)
        )
        if extra_tokens < _MIN_GROWTH_TOKENS:
            return current
        self._steps_since_growth = 0
        self._min_free_bytes = None
        return msgspec.structs.replace(
            current,
            target_kv_tokens=observation.kv_capacity_tokens + extra_tokens,
        )


class CompositePolicy:
    """Each member owns its own plan fields; they are applied in order."""

    def __init__(self, policies: Sequence[SpecPolicy]) -> None:
        self._policies = tuple(policies)

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        plan = current
        for policy in self._policies:
            plan = policy.decide(observation=observation, current=plan)
        return plan


def build_spec_policy(
    *, name: str, sps_table_path: Optional[str], ledger: Optional[MemoryLedger] = None
) -> SpecPolicy:
    if name == "observe":
        return ObservePolicy()
    if name == "priority":
        return PriorityWeightPolicy()
    if name == "memory":
        return MemoryHeadroomPolicy(ledger=ledger)
    if name == "conditional":
        return CompositePolicy(
            [PriorityWeightPolicy(), MemoryHeadroomPolicy(ledger=ledger)]
        )
    if name == "load-aware":
        return LoadAwareVerifyPolicy(knee_tokens=_knee_tokens(sps_table_path))
    raise ValueError(f"Unknown speculative policy {name!r}.")


def _knee_tokens(sps_table_path: Optional[str]) -> int:
    from sglang.srt.speculative.dspark_components.dspark_sps import (
        SpsCostTable,
        load_sps_table_from_path,
    )

    if sps_table_path is None:
        raise ValueError(
            "--speculative-policy load-aware needs a DSpark SPS table "
            "(--speculative-dspark-sps-table-path) to locate the step-time knee."
        )
    table = load_sps_table_from_path(sps_table_path)
    if not isinstance(table, SpsCostTable):
        raise ValueError(
            "--speculative-policy load-aware reads a diagonal SPS table "
            "(sample_batch_tokens / sample_steps_per_sec)."
        )
    return knee_tokens_from_sps(
        batch_tokens=table.sample_batch_tokens,
        steps_per_sec=table.sample_steps_per_sec,
    )
