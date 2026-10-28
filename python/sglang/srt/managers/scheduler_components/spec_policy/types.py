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
    # Tokens per request per verify step, bonus included (paper tau).
    accept_length_ema: Optional[float] = None
    num_verify_tokens_per_req: Optional[int] = None


class SpecPolicyPlan(msgspec.Struct, frozen=True, omit_defaults=True):
    """Knobs the actuator applies; None leaves the engine's own behavior in place."""

    # Fraction of the full verify window the DSpark confidence planner may schedule.
    dspark_verify_budget_frac: Optional[float] = None
    # Verify-slot ranking weight for requests without a positive priority; None = unweighted.
    low_priority_weight: Optional[float] = None
