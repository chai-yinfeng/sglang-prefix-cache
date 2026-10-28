from __future__ import annotations

import time
from typing import BinaryIO, Optional, Sequence

import msgspec

from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.managers.scheduler_components.spec_policy.actuator import (
    SpecPolicyActuator,
)
from sglang.srt.managers.scheduler_components.spec_policy.policy import (
    build_spec_policy,
)
from sglang.srt.managers.scheduler_components.spec_policy.sensor import (
    SpecPolicySensor,
)
from sglang.srt.managers.scheduler_components.spec_policy.types import (
    Observation,
    SpecPolicyPlan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

# Arbitrary; roughly a 10-step memory at decode cadence.
_EMA_ALPHA = 0.1


class _TraceRecord(msgspec.Struct, frozen=True):
    observation: Observation
    plan: SpecPolicyPlan


class SpecPolicyController:
    """Sense -> decide -> act, once per processed batch result.

    Decisions are local to this rank; a multi-rank deployment needs them agreed
    (e.g. rank_consensus) before the actuator runs.
    """

    def __init__(
        self,
        *,
        policy_name: str,
        sps_table_path: Optional[str],
        draft_worker,
        trace_path: Optional[str],
        is_trace_writer: bool,
    ) -> None:
        self.sensor = SpecPolicySensor(ema_alpha=_EMA_ALPHA)
        self.policy = build_spec_policy(name=policy_name, sps_table_path=sps_table_path)
        self.actuator = SpecPolicyActuator(draft_worker=draft_worker)
        self.plan = SpecPolicyPlan()
        self._trace: Optional[BinaryIO] = (
            open(trace_path, "ab") if trace_path and is_trace_writer else None
        )
        self._encoder = msgspec.json.Encoder()

    def on_batch_result(
        self,
        *,
        forward_mode: ForwardMode,
        batch_size: int,
        num_correct_drafts: int,
        num_verify_tokens_per_req: Optional[int],
        num_running: int,
        num_queued: int,
        pool_stats: PoolStats,
        reqs: Sequence,
    ) -> None:
        observation = self.sensor.observe(
            forward_mode=forward_mode,
            batch_size=batch_size,
            num_correct_drafts=num_correct_drafts,
            num_verify_tokens_per_req=num_verify_tokens_per_req,
            num_running=num_running,
            num_queued=num_queued,
            pool_stats=pool_stats,
            now=time.monotonic(),
        )
        plan = self.policy.decide(observation=observation, current=self.plan)
        self.actuator.apply(plan=plan, current=self.plan, reqs=reqs)
        self.plan = plan
        self._write_trace(observation=observation, plan=plan)

    def close(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def _write_trace(self, *, observation: Observation, plan: SpecPolicyPlan) -> None:
        if self._trace is None:
            return
        self._trace.write(
            self._encoder.encode(_TraceRecord(observation=observation, plan=plan))
            + b"\n"
        )
        self._trace.flush()
