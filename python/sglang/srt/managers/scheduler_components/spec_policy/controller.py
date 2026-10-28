from __future__ import annotations

import time
from typing import BinaryIO, Callable, Optional, Sequence

import msgspec
import torch

from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.managers.scheduler_components.spec_policy.actuator import (
    SpecPolicyActuator,
)
from sglang.srt.managers.scheduler_components.spec_policy.ledger import MemoryLedger
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
# Arbitrary; cudaMemGetInfo is cheap but not free, and free memory moves slowly.
_FREE_MEMORY_SAMPLE_STEPS = 32
# Policies whose plans may pause drafting.
_PAUSING_POLICIES = ("regime", "conditional")


class _TraceRecord(msgspec.Struct, frozen=True):
    observation: Observation
    plan: SpecPolicyPlan


class _TraceHeader(msgspec.Struct, frozen=True):
    ledger: Optional[MemoryLedger]


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
        ledger: Optional[MemoryLedger] = None,
        grow_kv_pool: Optional[Callable[[int], int]] = None,
    ) -> None:
        if policy_name in _PAUSING_POLICIES and not _supports_spec_pause(draft_worker):
            raise ValueError(
                f"--speculative-policy {policy_name} pauses drafting, which needs DSpark "
                "with compact ragged verify and a verify budget planner."
            )
        max_verify_width = None
        if policy_name == "width":
            max_verify_width = _max_verify_width(draft_worker)
        self.sensor = SpecPolicySensor(ema_alpha=_EMA_ALPHA)
        self.policy = build_spec_policy(
            name=policy_name,
            sps_table_path=sps_table_path,
            ledger=ledger,
            max_verify_width=max_verify_width,
        )
        self.actuator = SpecPolicyActuator(
            draft_worker=draft_worker, grow_kv_pool=grow_kv_pool
        )
        self.plan = SpecPolicyPlan()
        self._trace: Optional[BinaryIO] = (
            open(trace_path, "ab") if trace_path and is_trace_writer else None
        )
        self._encoder = msgspec.json.Encoder()
        self.ledger = ledger
        self._steps = 0
        self._free_gpu_bytes: Optional[int] = None
        self._write(_TraceHeader(ledger=ledger))

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
        self._steps += 1
        if self._steps % _FREE_MEMORY_SAMPLE_STEPS == 1:
            self._free_gpu_bytes = _free_gpu_bytes()
        observation = self.sensor.observe(
            free_gpu_bytes=self._free_gpu_bytes,
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
        self._write(_TraceRecord(observation=observation, plan=plan))

    def _write(self, record: msgspec.Struct) -> None:
        if self._trace is None:
            return
        self._trace.write(self._encoder.encode(record) + b"\n")
        self._trace.flush()


def _supports_spec_pause(draft_worker) -> bool:
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )

    return isinstance(draft_worker, DSparkWorkerV2) and draft_worker.supports_spec_pause


def _max_verify_width(draft_worker) -> int:
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    if not isinstance(draft_worker, DFlashWorkerV2) or not (
        draft_worker.supports_verify_width
    ):
        raise ValueError(
            "--speculative-policy width needs DFlash with the greedy draft head and "
            "no compact draft cache or Mamba verify state."
        )
    return int(draft_worker.block_size)


def _free_gpu_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes)
