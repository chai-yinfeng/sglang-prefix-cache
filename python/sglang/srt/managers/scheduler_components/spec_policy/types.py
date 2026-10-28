from __future__ import annotations

from typing import Optional

import msgspec


class PoolUsage(msgspec.Struct, frozen=True):
    """Occupancy of one device pool, in the pool's own unit (KV tokens or Mamba slots)."""

    used: int
    evictable: int
    available: int


class Observation(msgspec.Struct, frozen=True, omit_defaults=True):
    """What the policy sees after one scheduler step."""

    step: int
    forward_mode: str
    batch_size: int
    num_running: int
    num_queued: int
    kv: PoolUsage
    mamba: Optional[PoolUsage] = None
    # Wall time between consecutive decode results; None until two decode steps run back to back.
    step_ms_ema: Optional[float] = None
    # This decode step's wall time; None unless the previous result was a decode too.
    step_ms: Optional[float] = None
    # Tokens this decode step committed across the batch, bonus included.
    num_accept_tokens: Optional[int] = None
    # Tokens per request per verify step, bonus included (paper tau).
    accept_length_ema: Optional[float] = None
    num_verify_tokens_per_req: Optional[int] = None
    # Device-wide free bytes, sampled every few steps (a driver call).
    free_gpu_bytes: Optional[int] = None
    # Current token KV capacity: used + evictable + available.
    kv_capacity_tokens: Optional[int] = None


class SpecPolicyPlan(msgspec.Struct, frozen=True, omit_defaults=True):
    """Knobs the actuator applies; None leaves the engine's own behavior in place."""

    # Fraction of the full verify window the DSpark confidence planner may schedule.
    dspark_verify_budget_frac: Optional[float] = None
    # Verify-slot ranking weight for requests without a positive priority; None = unweighted.
    low_priority_weight: Optional[float] = None
    # Token KV capacity to grow to while serving; None = keep the startup size.
    target_kv_tokens: Optional[int] = None
    # Skip the draft and verify only each request's root; None = never touched.
    spec_paused: Optional[bool] = None
