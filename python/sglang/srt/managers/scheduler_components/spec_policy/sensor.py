from __future__ import annotations

from typing import Optional

from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.managers.scheduler_components.spec_policy.types import (
    Observation,
    PoolUsage,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _ema(previous: Optional[float], value: float, alpha: float) -> float:
    return value if previous is None else previous + alpha * (value - previous)


def _mamba_usage(pool_stats: PoolStats) -> Optional[PoolUsage]:
    if not pool_stats.is_hybrid_ssm:
        return None
    return PoolUsage(
        used=pool_stats.mamba_num_used,
        evictable=pool_stats.mamba_evictable_size,
        available=pool_stats.mamba_available_size,
    )


class SpecPolicySensor:
    """Turns per-step batch results into a smoothed load and acceptance picture."""

    def __init__(self, *, ema_alpha: float) -> None:
        self._ema_alpha = ema_alpha
        self._step = 0
        self._last_decode_ts: Optional[float] = None
        self._step_ms_ema: Optional[float] = None
        self._accept_length_ema: Optional[float] = None
        self._num_verify_tokens_per_req: Optional[int] = None

    def observe(
        self,
        *,
        forward_mode: ForwardMode,
        batch_size: int,
        num_correct_drafts: int,
        num_verify_tokens_per_req: Optional[int],
        num_running: int,
        num_queued: int,
        pool_stats: PoolStats,
        now: float,
        free_gpu_bytes: Optional[int] = None,
    ) -> Observation:
        self._step += 1
        step_ms = num_accept_tokens = None
        if forward_mode.is_decode() and batch_size > 0:
            step_ms = self._observe_decode(
                batch_size=batch_size,
                num_correct_drafts=num_correct_drafts,
                num_verify_tokens_per_req=num_verify_tokens_per_req,
                now=now,
            )
            num_accept_tokens = batch_size + num_correct_drafts
        else:
            # A prefill between two decodes is not part of the decode cadence.
            self._last_decode_ts = None
        return Observation(
            step=self._step,
            forward_mode=forward_mode.name,
            batch_size=batch_size,
            num_running=num_running,
            num_queued=num_queued,
            kv=PoolUsage(
                used=pool_stats.full_num_used,
                evictable=pool_stats.full_evictable_size,
                available=pool_stats.full_available_size,
            ),
            mamba=_mamba_usage(pool_stats),
            step_ms_ema=self._step_ms_ema,
            step_ms=step_ms,
            num_accept_tokens=num_accept_tokens,
            accept_length_ema=self._accept_length_ema,
            num_verify_tokens_per_req=self._num_verify_tokens_per_req,
            free_gpu_bytes=free_gpu_bytes,
            kv_capacity_tokens=(
                pool_stats.full_num_used
                + pool_stats.full_evictable_size
                + pool_stats.full_available_size
            ),
        )

    def _observe_decode(
        self,
        *,
        batch_size: int,
        num_correct_drafts: int,
        num_verify_tokens_per_req: Optional[int],
        now: float,
    ) -> Optional[float]:
        step_ms = None
        if self._last_decode_ts is not None:
            step_ms = (now - self._last_decode_ts) * 1000
            self._step_ms_ema = _ema(self._step_ms_ema, step_ms, self._ema_alpha)
        self._last_decode_ts = now
        if num_verify_tokens_per_req is not None:
            self._num_verify_tokens_per_req = num_verify_tokens_per_req
            self._accept_length_ema = _ema(
                self._accept_length_ema,
                1 + num_correct_drafts / batch_size,
                self._ema_alpha,
            )
        return step_ms
