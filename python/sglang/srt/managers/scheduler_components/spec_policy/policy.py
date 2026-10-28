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
# Arbitrary; the overlap scheduler applies a switch one step late, so the first
# steps after it still run the old mode.
_REGIME_SETTLE_STEPS = 4
# Arbitrary; about one second of decode at 8B c64, enough to average out the
# prefills interleaved with it.
_REGIME_WINDOW_STEPS = 64
# Arbitrary; how long a decision stands before the rival mode is re-measured. Each
# probe that loses doubles it up to the cap, since a losing probe costs goodput.
_REGIME_EXPLOIT_STEPS = 1024
_REGIME_MAX_EXPLOIT_STEPS = 16384
# Arbitrary; even a load shift does not re-probe sooner than this after a decision.
_REGIME_MIN_DECISION_STEPS = 4 * 64
# Arbitrary; a rival mode must beat the current one by this much to take over.
_REGIME_SWITCH_MARGIN = 0.03
# Measured: at c16 DSpark beat plain on every 8B workload (1.27-1.66x), so a batch
# this small never probes or stays paused.
_REGIME_MIN_PAUSE_BATCH = 24
# Arbitrary; a mean batch size that moved this much makes the last comparison stale.
_REGIME_LOAD_SHIFT = 0.25
# Measured: on Qwen3-8B / RTX 5090 at c64 a speculative step cost 1.3-1.7x a
# root-only one (W2, w1-natural), so pausing cannot win while each request commits
# more tokens per step than this; above it the pause is not even probed.
_REGIME_MAX_PAUSE_ACCEPT_LENGTH = 1.8
# Arbitrary; the verify widths the width policy moves between, widest first. Width 1
# verifies only the bonus token (drafting paused).
_WIDTH_LADDER = (16, 8, 6, 4, 3, 2, 1)
# Measured: DFlash decode steps on Qwen3-8B / RTX 5090 cost within 10% of each other
# from 8 to 64 verify tokens (c1-c16), so a window narrower than this many tokens per
# step saves little time and only loses acceptance; it is never chosen.
_WIDTH_MIN_VERIFY_TOKENS = 64
# Measured on the same setup: past this many verify tokens a step is compute-bound
# (c16: 128 -> 256 tokens cost +47%), so every wasted verify position costs time.
_WIDTH_MAX_VERIFY_TOKENS = 256
# A window that commits less than this fraction of its positions past the
# compute-bound point jumps to the widest rung under it before climbing. Measured:
# natural chat and proofreading in one c64 batch commit 0.43 of a 16-wide window
# while width 3 runs 1.8x faster; climbing back up from the ceiling costs a few
# short windows, staying wide costs seconds per window.
_WIDTH_MIN_UTILIZATION = 0.5
# Arbitrary; a window closes once its steps verified this many tokens, so a wide
# window under load (16 x 64 tokens per step) measures in 16 steps, not 64.
_WIDTH_WINDOW_VERIFY_TOKENS = 16384


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


class _GoodputWindow:
    """Decode tokens per second over a run of consecutive measured decode steps."""

    def __init__(self, *, settle_steps: int) -> None:
        self._settle = settle_steps
        self.steps = 0
        self._tokens = 0
        self._ms = 0.0
        self._batch_sum = 0

    def add(self, *, num_accept_tokens: int, step_ms: float, batch_size: int) -> None:
        if self._settle > 0:
            self._settle -= 1
            return
        self.steps += 1
        self._tokens += num_accept_tokens
        self._ms += step_ms
        self._batch_sum += batch_size

    @property
    def tokens_per_sec(self) -> float:
        return self._tokens / self._ms * 1000 if self._ms > 0 else 0.0

    @property
    def mean_batch_size(self) -> float:
        return self._batch_sum / self.steps if self.steps else 0.0

    @property
    def tokens_per_request_sec(self) -> float:
        """Decode speed one request sees: the inverse of time per output token."""
        return self._tokens / self._batch_sum / self._ms * 1000 if self._ms > 0 else 0.0

    @property
    def accept_length(self) -> float:
        return self._tokens / self._batch_sum if self._batch_sum else 0.0


class SpecRegimePolicy:
    """Pause speculation when measured decode goodput says plain decode is faster.

    Whether drafting pays depends on the load, the workload's acceptance and the
    memory it costs, none of which the engine can predict well, so the policy
    measures instead: it keeps a rolling goodput window of the current mode and,
    when the decision is old or the batch size has moved, runs one window of the
    other mode and keeps whichever was faster by a margin. Small batches and high
    acceptance never probe a pause, since there drafting reliably wins.
    """

    def __init__(self) -> None:
        self._paused = False
        self._probing = False
        self._reference_tokens_per_sec = 0.0
        self._decision_batch_size: Optional[float] = None
        self._steps_since_decision = 0
        self._exploit_steps = _REGIME_EXPLOIT_STEPS
        self._window = _GoodputWindow(settle_steps=0)

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        if observation.step_ms is None or observation.num_accept_tokens is None:
            return current
        self._window.add(
            num_accept_tokens=observation.num_accept_tokens,
            step_ms=observation.step_ms,
            batch_size=observation.batch_size,
        )
        if self._window.steps < _REGIME_WINDOW_STEPS:
            return current
        window = self._window
        self._window = _GoodputWindow(settle_steps=0)
        if self._probing:
            self._finish_probe(window)
        else:
            self._steps_since_decision += window.steps
            self._maybe_start_probe(window)
        return msgspec.structs.replace(current, spec_paused=self._paused)

    def _maybe_start_probe(self, window: _GoodputWindow) -> None:
        batch_size = window.mean_batch_size
        if self._paused and batch_size < _REGIME_MIN_PAUSE_BATCH:
            self._switch(paused=False)
            self._decide_at(batch_size)
            return
        load_shifted = self._decision_batch_size is None or (
            self._steps_since_decision >= _REGIME_MIN_DECISION_STEPS
            and abs(batch_size - self._decision_batch_size)
            > _REGIME_LOAD_SHIFT * max(self._decision_batch_size, 1.0)
        )
        if load_shifted:
            self._exploit_steps = _REGIME_EXPLOIT_STEPS
        elif self._steps_since_decision < self._exploit_steps:
            return
        if not self._paused and (
            batch_size < _REGIME_MIN_PAUSE_BATCH
            or window.accept_length >= _REGIME_MAX_PAUSE_ACCEPT_LENGTH
        ):
            self._decide_at(batch_size)
            return
        self._reference_tokens_per_sec = window.tokens_per_sec
        self._probing = True
        self._switch(paused=not self._paused)

    def _finish_probe(self, window: _GoodputWindow) -> None:
        self._probing = False
        if window.tokens_per_sec <= self._reference_tokens_per_sec * (
            1 + _REGIME_SWITCH_MARGIN
        ):
            self._switch(paused=not self._paused)
            self._exploit_steps = min(
                2 * self._exploit_steps, _REGIME_MAX_EXPLOIT_STEPS
            )
        else:
            self._exploit_steps = _REGIME_EXPLOIT_STEPS
        self._decide_at(window.mean_batch_size)

    def _switch(self, *, paused: bool) -> None:
        self._paused = paused
        self._window = _GoodputWindow(settle_steps=_REGIME_SETTLE_STEPS)

    def _decide_at(self, batch_size: float) -> None:
        self._decision_batch_size = batch_size
        self._steps_since_decision = 0


class _ClimbRound(msgspec.Struct):
    """One search from the standing width: step one rung at a time while it pays."""

    start_index: int
    direction: int
    # None until a window at the standing width is measured; the window that
    # started the round may straddle the load change that triggered it.
    reference_rate: Optional[float] = None
    flipped: bool = False


class SpecWidthPolicy:
    """Pick the verify width by measured decode goodput.

    A wider window commits more tokens per step but costs more verify compute, and
    which side wins moves with the batch size and the workload's acceptance. The
    policy hill-climbs a ladder of widths: when the standing choice is old or the
    batch size has moved, it re-measures the standing width, then the next rung
    (narrower when the load grew, wider when it fell), keeps climbing while each
    rung beats the last by a margin,
    tries the other direction once if the first rung loses, and otherwise stays.
    Rungs are compared by the decode speed one request sees (tokens per request
    per second): at a fixed concurrency that ranks widths as total goodput does,
    while under open-loop arrivals total goodput would reward a slow width for the
    backlog it builds.
    Windows only count steps that actually ran at the width being measured. Two
    verify-token bounds shortcut the climb: a width whose step would carry fewer
    than _WIDTH_MIN_VERIFY_TOKENS costs no less than a wider one, so it is never
    tried (at small batches that pins the full block, where noisy windows would
    otherwise wander narrower); and a compute-bound step that commits less than
    half of its window jumps straight to the widest rung under
    _WIDTH_MAX_VERIFY_TOKENS -- on any window that overshoots it, between rounds or
    while climbing narrower, and on the first step of a load jump that overshoots it
    twice, rather than after a window of slow steps. High-acceptance workloads keep
    wide windows, since the climb itself may go back up. A window also closes early
    once it verified _WIDTH_WINDOW_VERIFY_TOKENS, so wide steps under load are not
    measured for seconds. When narrow widths have verify graphs only up to some
    batch size, a batch that small only considers widths with graphs: eager narrow
    verify costs 9-15% there, while past it eager runs as fast as a graph.
    """

    def __init__(
        self,
        *,
        max_width: int,
        graph_widths: frozenset[int] = frozenset(),
        max_graph_batch_size: int = 0,
    ) -> None:
        self._ladder = (max_width, *(w for w in _WIDTH_LADDER if w < max_width))
        # Narrow widths with verify graphs up to max_graph_batch_size requests; empty
        # when none were captured, and then every width stays a candidate.
        self._graph_widths = graph_widths
        self._max_graph_batch_size = max_graph_batch_size
        self._batch_size = 0.0
        self._index = 0
        self._round: Optional[_ClimbRound] = None
        # Narrowest ladder index the current load allows.
        self._floor_index = 0
        self._decision_batch_size: Optional[float] = None
        self._steps_since_decision = 0
        self._exploit_steps = _REGIME_EXPLOIT_STEPS
        self._window = _GoodputWindow(settle_steps=0)
        # (width, accept length) of the last full window.
        self._last_window: Optional[tuple[int, float]] = None

    @property
    def width(self) -> int:
        return self._ladder[self._index]

    def decide(
        self, *, observation: Observation, current: SpecPolicyPlan
    ) -> SpecPolicyPlan:
        if observation.forward_mode == "DECODE" and self._wastes_a_load_jump(
            observation.batch_size
        ):
            self._jump_and_climb(observation.batch_size)
        if (
            observation.step_ms is not None
            and observation.num_accept_tokens is not None
            and observation.num_verify_tokens_per_req == self.width
        ):
            self._window.add(
                num_accept_tokens=observation.num_accept_tokens,
                step_ms=observation.step_ms,
                batch_size=observation.batch_size,
            )
            if self._window_is_full():
                window = self._window
                self._window = _GoodputWindow(settle_steps=0)
                self._last_window = (self.width, window.accept_length)
                self._batch_size = window.mean_batch_size
                self._floor_index = self._narrowest_index(window.mean_batch_size)
                if self._index > self._floor_index or not self._allowed(
                    self.width, window.mean_batch_size
                ):
                    # The load fell under the floor: the floor dominates, even mid-round.
                    self._round = None
                    self._move(self._floor_index)
                    self._decide_at(window.mean_batch_size)
                elif self._overshoots_the_ceiling(window):
                    # A load that crept past the ceiling narrows now, not a round later.
                    self._jump_and_climb(window.mean_batch_size)
                elif self._round is not None:
                    self._continue_round(window)
                else:
                    self._steps_since_decision += window.steps
                    self._maybe_start_round(window)
        return msgspec.structs.replace(current, verify_width=self.width)

    def _window_is_full(self) -> bool:
        window = self._window
        return (
            window.steps >= _REGIME_WINDOW_STEPS
            or self.width * window.mean_batch_size * window.steps
            >= _WIDTH_WINDOW_VERIFY_TOKENS
        )

    def _maybe_start_round(self, window: _GoodputWindow) -> None:
        batch_size = window.mean_batch_size
        previous = self._decision_batch_size
        load_shifted = previous is None or (
            self._steps_since_decision >= _REGIME_MIN_DECISION_STEPS
            and abs(batch_size - previous) > _REGIME_LOAD_SHIFT * max(previous, 1.0)
        )
        if load_shifted:
            self._exploit_steps = _REGIME_EXPLOIT_STEPS
        elif self._steps_since_decision < self._exploit_steps:
            return
        jump = min(self._jump_index(window), self._floor_index)
        if jump != self._index:
            self._move(jump)
        direction = -1 if previous is not None and batch_size < previous else 1
        if not self._in_ladder(self._index + direction):
            direction = -direction
        if not self._in_ladder(self._index + direction):
            self._decide_at(batch_size)
            return
        self._round = _ClimbRound(start_index=self._index, direction=direction)

    def _continue_round(self, window: _GoodputWindow) -> None:
        climb = self._round
        if climb.reference_rate is None:
            climb.reference_rate = window.tokens_per_request_sec
            if not self._in_ladder(self._index + climb.direction):
                climb.direction = -climb.direction
                climb.flipped = True
            if self._in_ladder(self._index + climb.direction):
                self._move(self._index + climb.direction)
            else:
                self._end_round(window)
            return
        if window.tokens_per_request_sec > climb.reference_rate * (
            1 + _REGIME_SWITCH_MARGIN
        ):
            climb.reference_rate = window.tokens_per_request_sec
            if self._in_ladder(self._index + climb.direction):
                self._move(self._index + climb.direction)
                return
            self._end_round(window)
            return
        self._move(self._index - climb.direction)
        if self._index == climb.start_index and not climb.flipped:
            climb.direction = -climb.direction
            climb.flipped = True
            if self._in_ladder(self._index + climb.direction):
                self._move(self._index + climb.direction)
                return
        self._end_round(window)

    def _end_round(self, window: _GoodputWindow) -> None:
        moved = self._index != self._round.start_index
        self._round = None
        self._exploit_steps = (
            _REGIME_EXPLOIT_STEPS
            if moved
            else min(2 * self._exploit_steps, _REGIME_MAX_EXPLOIT_STEPS)
        )
        self._decide_at(window.mean_batch_size)

    def _in_ladder(self, index: int) -> bool:
        return 0 <= index <= self._floor_index and self._allowed(
            self._ladder[index], self._batch_size
        )

    def _allowed(self, width: int, batch_size: float) -> bool:
        """Whether a width replays a graph at this batch size, or eager costs nothing."""
        return (
            not self._graph_widths
            or width == self._ladder[0]
            or width in self._graph_widths
            or batch_size > self._max_graph_batch_size
        )

    def _narrowest_index(self, batch_size: float) -> int:
        return max(
            (
                i
                for i, width in enumerate(self._ladder)
                if width * batch_size >= _WIDTH_MIN_VERIFY_TOKENS
                and self._allowed(width, batch_size)
            ),
            default=0,
        )

    def _jump_index(self, window: _GoodputWindow) -> int:
        """Where a wasteful compute-bound window should restart its climb; the
        standing index when it is neither."""
        if (
            self.width * window.mean_batch_size <= _WIDTH_MAX_VERIFY_TOKENS
            or window.accept_length >= _WIDTH_MIN_UTILIZATION * self.width
        ):
            return self._index
        return self._ceiling_index(window.mean_batch_size)

    def _overshoots_the_ceiling(self, window: _GoodputWindow) -> bool:
        """A wasteful compute-bound window outside a climb toward wider rungs."""
        climbing_wider = self._round is not None and self._round.direction < 0
        return not climbing_wider and self._jump_index(window) > self._index

    def _wastes_a_load_jump(self, batch_size: int) -> bool:
        last = self._last_window
        return (
            self._round is None
            and last is not None
            and last[0] == self.width
            and self.width * batch_size > 2 * _WIDTH_MAX_VERIFY_TOKENS
            and last[1] < _WIDTH_MIN_UTILIZATION * self.width
        )

    def _jump_and_climb(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._floor_index = self._narrowest_index(batch_size)
        index = self._ceiling_index(batch_size)
        self._move(index)
        self._decide_at(batch_size)
        self._round = _ClimbRound(start_index=index, direction=1)

    def _ceiling_index(self, batch_size: float) -> int:
        return next(
            (
                i
                for i, width in enumerate(self._ladder)
                if width * batch_size <= _WIDTH_MAX_VERIFY_TOKENS
                and self._allowed(width, batch_size)
            ),
            len(self._ladder) - 1,
        )

    def _move(self, index: int) -> None:
        self._index = index
        self._window = _GoodputWindow(settle_steps=_REGIME_SETTLE_STEPS)

    def _decide_at(self, batch_size: float) -> None:
        self._decision_batch_size = batch_size
        self._steps_since_decision = 0


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
    *,
    name: str,
    sps_table_path: Optional[str],
    ledger: Optional[MemoryLedger] = None,
    max_verify_width: Optional[int] = None,
    graph_widths: frozenset[int] = frozenset(),
    max_graph_batch_size: int = 0,
) -> SpecPolicy:
    if name == "observe":
        return ObservePolicy()
    if name == "priority":
        return PriorityWeightPolicy()
    if name == "memory":
        return MemoryHeadroomPolicy(ledger=ledger)
    if name == "regime":
        return SpecRegimePolicy()
    if name == "width":
        return SpecWidthPolicy(
            max_width=max_verify_width,
            graph_widths=graph_widths,
            max_graph_batch_size=max_graph_batch_size,
        )
    if name == "conditional":
        return CompositePolicy(
            [
                SpecRegimePolicy(),
                PriorityWeightPolicy(),
                MemoryHeadroomPolicy(ledger=ledger),
            ]
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
