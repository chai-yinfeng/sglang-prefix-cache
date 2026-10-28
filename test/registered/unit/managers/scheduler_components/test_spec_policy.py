import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.pool_stats_observer import (  # noqa: E402
    PoolStats,
)
from sglang.srt.managers.scheduler_components.spec_policy.actuator import (  # noqa: E402
    SpecPolicyActuator,
)
from sglang.srt.managers.scheduler_components.spec_policy.controller import (  # noqa: E402
    SpecPolicyController,
)
from sglang.srt.managers.scheduler_components.spec_policy.ledger import (  # noqa: E402
    MemoryLedger,
    _unique_param_bytes,
)
from sglang.srt.managers.scheduler_components.spec_policy.memory_split import (  # noqa: E402
    spec_state_memory_ratio,
)
from sglang.srt.managers.scheduler_components.spec_policy.policy import (  # noqa: E402
    CompositePolicy,
    LoadAwareVerifyPolicy,
    MemoryHeadroomPolicy,
    PriorityWeightPolicy,
    SpecRegimePolicy,
    SpecWidthPolicy,
    knee_tokens_from_sps,
)
from sglang.srt.managers.scheduler_components.spec_policy.sensor import (  # noqa: E402
    SpecPolicySensor,
)
from sglang.srt.managers.scheduler_components.spec_policy.types import (  # noqa: E402
    Observation,
    PoolUsage,
    SpecPolicyPlan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_planner import (  # noqa: E402
    weight_first_position,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool_stats(**mamba) -> PoolStats:
    stats = PoolStats(
        full_num_used=100,
        full_token_usage=0.1,
        full_available_size=800,
        full_evictable_size=100,
    )
    if mamba:
        stats.is_hybrid_ssm = True
        stats.mamba_num_used = mamba["used"]
        stats.mamba_available_size = mamba["available"]
        stats.mamba_evictable_size = mamba["evictable"]
    return stats


def _observe(sensor, *, mode, now, correct=0, verify=None, bs=4):
    return sensor.observe(
        forward_mode=mode,
        batch_size=bs,
        num_correct_drafts=correct,
        num_verify_tokens_per_req=verify,
        num_running=bs,
        num_queued=2,
        pool_stats=_pool_stats(),
        now=now,
    )


class TestSpecPolicySensor(CustomTestCase):
    def test_step_time_needs_back_to_back_decodes(self):
        sensor = SpecPolicySensor(ema_alpha=0.5)
        first = _observe(sensor, mode=ForwardMode.DECODE, now=1.0)
        self.assertIsNone(first.step_ms_ema)
        second = _observe(sensor, mode=ForwardMode.DECODE, now=1.02)
        self.assertAlmostEqual(second.step_ms_ema, 20.0)
        self.assertAlmostEqual(second.step_ms, 20.0)
        _observe(sensor, mode=ForwardMode.EXTEND, now=1.5)
        after_prefill = _observe(sensor, mode=ForwardMode.DECODE, now=1.6)
        self.assertAlmostEqual(after_prefill.step_ms_ema, 20.0)
        # The goodput windows read the raw step, which a prefill also invalidates.
        self.assertIsNone(after_prefill.step_ms)

    def test_accept_length_includes_bonus_token(self):
        sensor = SpecPolicySensor(ema_alpha=0.5)
        obs = _observe(sensor, mode=ForwardMode.DECODE, now=1.0, correct=6, verify=8)
        self.assertAlmostEqual(obs.accept_length_ema, 2.5)
        self.assertEqual(obs.num_accept_tokens, 4 + 6)
        self.assertEqual(obs.num_verify_tokens_per_req, 8)
        obs = _observe(sensor, mode=ForwardMode.DECODE, now=1.1, correct=2, verify=8)
        self.assertAlmostEqual(obs.accept_length_ema, 2.0)

    def test_plain_decode_has_no_accept_length(self):
        sensor = SpecPolicySensor(ema_alpha=0.5)
        obs = _observe(sensor, mode=ForwardMode.DECODE, now=1.0)
        self.assertIsNone(obs.accept_length_ema)

    def test_pool_usage(self):
        sensor = SpecPolicySensor(ema_alpha=0.5)
        obs = sensor.observe(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            num_correct_drafts=0,
            num_verify_tokens_per_req=None,
            num_running=1,
            num_queued=0,
            pool_stats=_pool_stats(used=3, available=9, evictable=1),
            now=0.0,
        )
        self.assertEqual(
            (obs.kv.used, obs.kv.available, obs.kv.evictable), (100, 800, 100)
        )
        self.assertEqual((obs.mamba.used, obs.mamba.available), (3, 9))


def _req(rid, slot, priority=None):
    # Mirrors Req: the pool row lives on req.kv.
    return SimpleNamespace(
        rid=rid, kv=SimpleNamespace(req_pool_idx=slot), priority=priority
    )


class TestSpecPolicyActuator(CustomTestCase):
    def test_only_changed_budget_reaches_worker(self):
        worker = Mock()
        actuator = SpecPolicyActuator(draft_worker=worker)
        same = SpecPolicyPlan(dspark_verify_budget_frac=0.5)
        actuator.apply(plan=same, current=same, reqs=[])
        worker.set_dspark_forced_budget_frac.assert_not_called()
        actuator.apply(plan=SpecPolicyPlan(), current=same, reqs=[])
        worker.set_dspark_forced_budget_frac.assert_called_once_with(None)

    def test_kv_growth_fires_once_per_new_target(self):
        grow = Mock(return_value=70000)
        actuator = SpecPolicyActuator(draft_worker=Mock(), grow_kv_pool=grow)
        grown = SpecPolicyPlan(target_kv_tokens=70000)
        actuator.apply(plan=grown, current=SpecPolicyPlan(), reqs=[])
        actuator.apply(plan=grown, current=grown, reqs=[])
        grow.assert_called_once_with(70000)

    def test_request_weights_follow_slot_reuse_and_reset(self):
        worker = Mock()
        actuator = SpecPolicyActuator(draft_worker=worker)
        on = SpecPolicyPlan(low_priority_weight=0.25)
        reqs = [_req("a", 3, priority=1), _req("b", 5), _req("gone", None)]
        actuator.apply(plan=on, current=on, reqs=reqs)
        worker.set_dspark_request_weights.assert_called_once_with(
            req_pool_indices=[3, 5], weights=[1.0, 0.25]
        )
        # Unchanged slots are not rewritten; a reused slot (new rid) is.
        worker.reset_mock()
        actuator.apply(
            plan=on,
            current=on,
            reqs=[_req("a", 3, priority=1), _req("c", 5, priority=2)],
        )
        worker.set_dspark_request_weights.assert_called_once_with(
            req_pool_indices=[5], weights=[1.0]
        )
        # Turning weighting off restores every slot it ever touched.
        worker.reset_mock()
        actuator.apply(plan=SpecPolicyPlan(), current=on, reqs=[])
        worker.set_dspark_request_weights.assert_called_once_with(
            req_pool_indices=[3, 5], weights=[1.0, 1.0]
        )


class TestWeightFirstPosition(CustomTestCase):
    def test_every_survival_product_scales_by_the_row_weight(self):
        confidence = torch.tensor([[0.9, 0.8, 0.5], [0.95, 0.6, 0.9]])
        original = confidence.clone()
        weights = torch.tensor([0.25, 1.0])
        weighted = weight_first_position(confidence=confidence, weights=weights)
        torch.testing.assert_close(
            torch.cumprod(weighted, dim=1),
            torch.cumprod(confidence, dim=1) * weights[:, None],
        )
        # The caller's tensor is left untouched (it is the draft's own output).
        torch.testing.assert_close(confidence, original)


def _decode(batch_size: int, *, mode: str = "DECODE") -> Observation:
    return Observation(
        step=0,
        forward_mode=mode,
        batch_size=batch_size,
        num_running=batch_size,
        num_queued=0,
        kv=PoolUsage(used=0, evictable=0, available=0),
        num_verify_tokens_per_req=8,
    )


class TestLoadAwareVerifyPolicy(CustomTestCase):
    def test_knee_is_last_probe_within_peak_fraction(self):
        # Qwen3-8B / RTX 5090 / FA4 table: 192 tokens runs at 62% of peak, 256 at 48%.
        knee = knee_tokens_from_sps(
            batch_tokens=[8, 16, 32, 64, 128, 192, 256],
            steps_per_sec=[71.9, 72.4, 71.4, 69.8, 58.9, 44.7, 34.8],
        )
        self.assertEqual(knee, 192)

    def test_hysteresis_and_dwell(self):
        policy = LoadAwareVerifyPolicy(knee_tokens=64)
        plan = policy.decide(observation=_decode(4), current=SpecPolicyPlan())
        self.assertEqual(plan, LoadAwareVerifyPolicy.FULL)
        # Inside the band (64 < 80 tokens <= 96) nothing moves, however long it stays.
        for _ in range(40):
            plan = policy.decide(observation=_decode(10), current=plan)
        self.assertEqual(plan, LoadAwareVerifyPolicy.FULL)
        plan = policy.decide(observation=_decode(13), current=plan)
        self.assertEqual(plan, LoadAwareVerifyPolicy.PLANNER)
        # A dip right after a switch waits out the dwell; prefill steps do not count toward it.
        for _ in range(30):
            plan = policy.decide(observation=_decode(1, mode="EXTEND"), current=plan)
        for _ in range(14):
            plan = policy.decide(observation=_decode(2), current=plan)
        self.assertEqual(plan, LoadAwareVerifyPolicy.PLANNER)
        plan = policy.decide(observation=_decode(2), current=plan)
        plan = policy.decide(observation=_decode(2), current=plan)
        self.assertEqual(plan, LoadAwareVerifyPolicy.FULL)


def _regime_steps(
    policy, plan, *, steps, batch_size, on_ms, paused_ms, accept_length=1.5
):
    """Drive the policy against an engine whose step time follows the plan's mode;
    a paused step commits one token per request."""
    for _ in range(steps):
        paused = bool(plan.spec_paused)
        observation = Observation(
            step=0,
            forward_mode="DECODE",
            batch_size=batch_size,
            num_running=batch_size,
            num_queued=0,
            kv=PoolUsage(used=0, evictable=0, available=0),
            step_ms=paused_ms if paused else on_ms,
            num_accept_tokens=batch_size if paused else int(batch_size * accept_length),
        )
        plan = policy.decide(observation=observation, current=plan)
    return plan


class TestSpecRegimePolicy(CustomTestCase):
    def test_pause_sticks_only_when_its_probe_beats_the_margin(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        enough = 4 * (mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS)
        # Spec: 96 tokens / 20 ms. Paused at 12 ms: 64 / 12 = 5.3 > 4.8 * 1.03 tok/ms.
        plan = _regime_steps(
            SpecRegimePolicy(),
            SpecPolicyPlan(),
            steps=enough,
            batch_size=64,
            on_ms=20.0,
            paused_ms=12.0,
        )
        self.assertTrue(plan.spec_paused)
        # Paused at 13 ms: 4.9 tok/ms is within the 3% margin, so speculation stays.
        plan = _regime_steps(
            SpecRegimePolicy(),
            SpecPolicyPlan(),
            steps=enough,
            batch_size=64,
            on_ms=20.0,
            paused_ms=13.0,
        )
        self.assertFalse(plan.spec_paused)

    def test_high_acceptance_never_probes_a_pause(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        # Pausing would be faster here, but at 2.5 tokens per step it is not tried.
        plan = _regime_steps(
            SpecRegimePolicy(),
            SpecPolicyPlan(),
            steps=4 * mod._REGIME_MAX_EXPLOIT_STEPS,
            batch_size=64,
            on_ms=20.0,
            paused_ms=1.0,
            accept_length=2.5,
        )
        self.assertFalse(plan.spec_paused)

    def test_losing_probes_back_off(self):
        # Pausing always loses here; without the doubling interval, 8192 steps would
        # hold about eight probes.
        policy, plan, probes = SpecRegimePolicy(), SpecPolicyPlan(), 0
        for _ in range(8192 // 64):
            before = bool(plan.spec_paused)
            plan = _regime_steps(
                policy, plan, steps=64, batch_size=64, on_ms=20.0, paused_ms=16.0
            )
            probes += int(plan.spec_paused and not before)
        self.assertLessEqual(probes, 4)
        self.assertFalse(plan.spec_paused)

    def test_small_batches_never_pause_and_end_a_pause(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        small = mod._REGIME_MIN_PAUSE_BATCH - 1
        steps = mod._REGIME_EXPLOIT_STEPS + 4 * mod._REGIME_WINDOW_STEPS
        policy = SpecRegimePolicy()
        plan = _regime_steps(
            policy,
            SpecPolicyPlan(),
            steps=steps,
            batch_size=small,
            on_ms=20.0,
            paused_ms=1.0,
        )
        self.assertFalse(plan.spec_paused)
        plan = _regime_steps(
            policy, plan, steps=steps, batch_size=64, on_ms=20.0, paused_ms=1.0
        )
        self.assertTrue(plan.spec_paused)
        # Resumed without a probe once a window's mean batch is small: at most two
        # windows, since the one in flight when the load drops still averages high.
        plan = _regime_steps(
            policy,
            plan,
            steps=2 * mod._REGIME_WINDOW_STEPS,
            batch_size=small,
            on_ms=20.0,
            paused_ms=1.0,
        )
        self.assertFalse(plan.spec_paused)


def _width_accept_length(width, survival=0.6):
    # Each further position survives with the same probability.
    return sum(survival**i for i in range(width))


def _width_step_ms(width, batch_size):
    # Verify and draft are each memory-bound until their token count makes them
    # compute-bound; width 1 skips the draft.
    tokens = batch_size * width
    return max(10.0, 0.09 * tokens) + (max(1.0, 0.02 * tokens) if width > 1 else 0.0)


def _width_steps(policy, plan, *, steps, batch_size, survival=0.6):
    """Drive the width policy against an engine that runs each step at the width the
    previous decision set, as the overlap scheduler does."""
    width = plan.verify_width or policy.width
    for _ in range(steps):
        observation = Observation(
            step=0,
            forward_mode="DECODE",
            batch_size=batch_size,
            num_running=batch_size,
            num_queued=0,
            kv=PoolUsage(used=0, evictable=0, available=0),
            step_ms=_width_step_ms(width, batch_size),
            num_accept_tokens=round(batch_size * _width_accept_length(width, survival)),
            num_verify_tokens_per_req=width,
        )
        plan = policy.decide(observation=observation, current=plan)
        width = plan.verify_width
    return plan


class TestSpecWidthPolicy(CustomTestCase):
    def test_climbs_to_the_best_width_for_each_load(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS
        policy = SpecWidthPolicy(max_width=16)
        # Four requests: any narrower window stays below the verify-token floor.
        plan = _width_steps(policy, SpecPolicyPlan(), steps=8 * window, batch_size=4)
        self.assertEqual(plan.verify_width, 16)
        # Compute-bound at 64: goodput peaks at width 2 (7.3 tokens/ms vs 6.4 at 1).
        plan = _width_steps(
            policy,
            plan,
            steps=mod._REGIME_MIN_DECISION_STEPS + 10 * window,
            batch_size=64,
        )
        self.assertEqual(plan.verify_width, 2)
        # The load falls back: width 2 is below the floor again, so the full block
        # returns without a probe.
        plan = _width_steps(
            policy,
            plan,
            steps=mod._REGIME_MIN_DECISION_STEPS + 2 * window,
            batch_size=4,
        )
        self.assertEqual(plan.verify_width, 16)
        # Sixteen requests: the climb stops at width 6 (3.2 tokens/ms vs 3.1 at 4).
        plan = _width_steps(
            policy,
            plan,
            steps=mod._REGIME_MIN_DECISION_STEPS + 6 * window,
            batch_size=16,
        )
        self.assertEqual(plan.verify_width, 6)

    def test_a_wasteful_load_jump_restarts_under_the_token_ceiling(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS
        policy = SpecWidthPolicy(max_width=16)
        # Whole windows at 4 requests (the width never moves, so none settles).
        plan = _width_steps(policy, SpecPolicyPlan(), steps=8 * window, batch_size=4)
        # One window at 64 requests commits 2.5 of 16 positions: 64 x 4 verify tokens
        # is the widest rung under the ceiling.
        plan = _width_steps(policy, plan, steps=window, batch_size=64)
        self.assertEqual(plan.verify_width, 4)

    def test_a_wasteful_load_spike_jumps_on_its_first_step(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(
            policy, SpecPolicyPlan(), steps=8 * mod._REGIME_WINDOW_STEPS, batch_size=4
        )
        # 64 x 16 verify tokens overshoot the ceiling fourfold while the last window
        # committed 2.5 of 16 positions: no window of slow steps first.
        plan = _width_steps(policy, plan, steps=1, batch_size=64)
        self.assertEqual(plan.verify_width, 4)
        # The well-used edit-like window does not.
        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(
            policy,
            SpecPolicyPlan(),
            steps=8 * mod._REGIME_WINDOW_STEPS,
            batch_size=4,
            survival=0.97,
        )
        plan = _width_steps(policy, plan, steps=1, batch_size=64, survival=0.97)
        self.assertEqual(plan.verify_width, 16)

    def test_a_well_used_window_climbs_instead_of_jumping(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS
        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(
            policy, SpecPolicyPlan(), steps=8 * window, batch_size=4, survival=0.97
        )
        # 13 of 16 positions commit, so after one window (64 x 16 verify tokens per
        # step close it in 16 steps) the round starts by re-measuring width 16.
        short_window = mod._WIDTH_WINDOW_VERIFY_TOKENS // (64 * 16)
        plan = _width_steps(
            policy, plan, steps=short_window, batch_size=64, survival=0.97
        )
        self.assertEqual(plan.verify_width, 16)

    def test_a_half_wasted_window_jumps(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(
            policy,
            SpecPolicyPlan(),
            steps=8 * mod._REGIME_WINDOW_STEPS,
            batch_size=4,
            survival=0.85,
        )
        # 5.9 of 16 positions commit, as in a batch mixing chat and proofreading:
        # the load spike still jumps under the ceiling on its first step.
        plan = _width_steps(policy, plan, steps=1, batch_size=64, survival=0.85)
        self.assertEqual(plan.verify_width, 4)

    def test_a_load_creeping_past_the_ceiling_narrows_within_a_window(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS
        policy = SpecWidthPolicy(max_width=16)
        # Twelve requests: the climb ends at width 8 one window before the ramp.
        plan = _width_steps(
            policy, SpecPolicyPlan(), steps=5 * window, batch_size=12, survival=0.75
        )
        self.assertEqual(plan.verify_width, 8)
        # Open-loop arrivals grow the batch to 40 before the next round may start;
        # width 8 now verifies 320 tokens a step and commits 3.6 of each 8.
        for batch_size in range(14, 41, 2):
            plan = _width_steps(
                policy, plan, steps=4, batch_size=batch_size, survival=0.75
            )
        plan = _width_steps(policy, plan, steps=window, batch_size=40, survival=0.75)
        self.assertLessEqual(plan.verify_width * 40, mod._WIDTH_MAX_VERIFY_TOKENS)

    def test_open_loop_backlog_does_not_reward_a_slow_width(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        # Open-loop arrivals: a width that serves each request slower keeps more
        # requests running. Width 6 then shows the most total goodput (24 requests,
        # 3.6 tokens/ms) though width 4 decodes each request faster (16 requests,
        # 0.19 vs 0.15 tokens/ms per request) and is what keeps the queue short.
        batch_of = {16: 40, 8: 28, 6: 24, 4: 16, 3: 16, 2: 16, 1: 16}
        policy, plan = SpecWidthPolicy(max_width=16), SpecPolicyPlan()
        window = mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS
        for _ in range(12 * window):
            plan = _width_steps(
                policy, plan, steps=1, batch_size=batch_of[policy.width]
            )
        self.assertEqual(plan.verify_width, 4)

    def test_small_batches_only_choose_widths_with_graphs(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS
        policy = SpecWidthPolicy(
            max_width=16, graph_widths=frozenset({8, 6, 4}), max_graph_batch_size=32
        )
        # 24 requests: width 3 would run eager (no graph), so it is never probed,
        # though without the graphs the climb visits it.
        plan, seen = SpecPolicyPlan(), set()
        for _ in range(20 * window):
            plan = _width_steps(policy, plan, steps=1, batch_size=24)
            seen.add(plan.verify_width)
        self.assertLessEqual(seen, {16, 8, 6, 4})
        # Past the graphs' batch sizes eager costs nothing: 64 requests climb to
        # width 2, which has no graph.
        policy = SpecWidthPolicy(
            max_width=16, graph_widths=frozenset({8, 6, 4}), max_graph_batch_size=32
        )
        plan = _width_steps(policy, SpecPolicyPlan(), steps=8 * window, batch_size=64)
        self.assertEqual(plan.verify_width, 2)

    def test_a_wide_window_under_load_closes_early(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(
            policy,
            SpecPolicyPlan(),
            steps=8 * mod._REGIME_WINDOW_STEPS,
            batch_size=4,
            survival=0.97,
        )
        # 32 steps at 64 requests: two windows of 16 x 64 verify tokens per step (one
        # starts the round, one re-measures the standing width) and the climb has
        # moved, half a 64-step window in.
        plan = _width_steps(policy, plan, steps=32, batch_size=64, survival=0.97)
        self.assertEqual(plan.verify_width, 8)

    def test_a_load_drop_mid_round_returns_to_the_floor(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        window = mod._REGIME_WINDOW_STEPS + mod._REGIME_SETTLE_STEPS
        policy = SpecWidthPolicy(max_width=16)
        plan = _width_steps(policy, SpecPolicyPlan(), steps=8 * window, batch_size=4)
        # Two windows at 64 requests: jumped to width 4 and climbing.
        plan = _width_steps(policy, plan, steps=2 * window, batch_size=64)
        self.assertIsNotNone(policy._round)
        # Within two windows at four requests the full block is back.
        plan = _width_steps(policy, plan, steps=2 * window, batch_size=4)
        self.assertEqual(plan.verify_width, 16)
        self.assertIsNone(policy._round)

    def test_steps_at_another_width_do_not_count(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = SpecWidthPolicy(max_width=16)
        observation = Observation(
            step=0,
            forward_mode="DECODE",
            batch_size=64,
            num_running=64,
            num_queued=0,
            kv=PoolUsage(used=0, evictable=0, available=0),
            step_ms=1.0,
            num_accept_tokens=64,
            num_verify_tokens_per_req=4,
        )
        plan = SpecPolicyPlan()
        for _ in range(4 * mod._REGIME_WINDOW_STEPS):
            plan = policy.decide(observation=observation, current=plan)
        # No window ever filled, so no round started from the standing width.
        self.assertEqual(plan.verify_width, 16)

    def test_losing_rounds_back_off(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = SpecWidthPolicy(max_width=16)
        widths, plan = [], SpecPolicyPlan()
        for _ in range(16 * mod._REGIME_EXPLOIT_STEPS // mod._REGIME_WINDOW_STEPS):
            plan = _width_steps(
                policy, plan, steps=mod._REGIME_WINDOW_STEPS, batch_size=16
            )
            widths.append(plan.verify_width)
        # Width 6 stands between rounds (the run may end inside one).
        self.assertEqual(max(set(widths), key=widths.count), 6)
        probes = sum(1 for a, b in zip(widths, widths[1:]) if a == 6 and b != 6)
        # Every round from width 6 loses; without the backoff one would start every
        # 1024 steps (about 15 of them).
        self.assertLessEqual(probes, 5)

    def test_width_reaches_the_worker_once_per_change(self):
        worker = Mock()
        actuator = SpecPolicyActuator(draft_worker=worker)
        narrow = SpecPolicyPlan(verify_width=4)
        actuator.apply(plan=narrow, current=SpecPolicyPlan(), reqs=[])
        actuator.apply(plan=narrow, current=narrow, reqs=[])
        worker.set_dflash_verify_width.assert_called_once_with(4)


class TestMemoryLedger(CustomTestCase):
    def test_shared_draft_storage_is_counted_once(self):
        target = torch.nn.Linear(4, 4, bias=False)
        draft = torch.nn.Module()
        draft.head = target.weight  # a draft sharing the target's lm_head storage
        draft.own = torch.nn.Parameter(torch.zeros(2, 2))
        seen: set[int] = set()
        self.assertEqual(_unique_param_bytes([target], seen), 16 * 4)
        self.assertEqual(_unique_param_bytes([draft], seen), 4 * 4)

    def test_freed_bytes_convert_at_target_plus_draft_lane(self):
        # Qwen3-8B BF16 with DSpark: 144 KiB target + 20 KiB draft per token.
        ledger = MemoryLedger(
            kv_capacity_tokens=60000,
            target_kv_bytes_per_token=144 * 1024,
            draft_kv_bytes_per_token=20 * 1024,
            target_weights_bytes=0,
            draft_weights_bytes=2 << 30,
        )
        self.assertEqual(ledger.tokens_for_bytes(ledger.draft_weights_bytes), 12787)


class TestSpecStateMemoryRatio(CustomTestCase):
    def test_memory_policies_count_each_verify_state(self):
        # Qwen3.5-4B DFlash b16 (extra_buffer + overlap: 5 slots per request): the
        # 0.9 * 21 / 5 split measured as the rule rows of policy round 4.
        for policy_name in ("memory", "conditional"):
            ratio = spec_state_memory_ratio(
                policy_name=policy_name,
                base_ratio=0.9,
                slots_per_req=5,
                num_draft_tokens=16,
            )
            self.assertAlmostEqual(ratio, 3.78)

    def test_other_policies_keep_the_configured_ratio(self):
        for policy_name in ("none", "observe", "width", "priority", "regime"):
            ratio = spec_state_memory_ratio(
                policy_name=policy_name,
                base_ratio=0.9,
                slots_per_req=5,
                num_draft_tokens=16,
            )
            self.assertEqual(ratio, 0.9)


def _headroom_obs(free_bytes, capacity=60000):
    return Observation(
        step=0,
        forward_mode="DECODE",
        batch_size=8,
        num_running=8,
        num_queued=0,
        kv=PoolUsage(used=0, evictable=0, available=0),
        free_gpu_bytes=free_bytes,
        kv_capacity_tokens=capacity,
    )


_LEDGER = MemoryLedger(
    kv_capacity_tokens=60000,
    target_kv_bytes_per_token=144 * 1024,
    draft_kv_bytes_per_token=20 * 1024,
    target_weights_bytes=0,
    draft_weights_bytes=0,
)


class TestMemoryHeadroomPolicy(CustomTestCase):
    def test_grows_by_the_window_floor_minus_margin(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = MemoryHeadroomPolicy(ledger=_LEDGER)
        plan = SpecPolicyPlan()
        # A brief dip inside the window sets the floor, not the latest sample.
        frees = [5 << 30] * (mod._HEADROOM_WINDOW_STEPS - 1) + [3 << 30]
        for i, free in enumerate(frees):
            plan = policy.decide(observation=_headroom_obs(free), current=plan)
            if i < len(frees) - 1:
                self.assertIsNone(plan.target_kv_tokens)
        expected = 60000 + _LEDGER.tokens_for_bytes(
            (3 << 30) - mod._HEADROOM_MARGIN_BYTES
        )
        self.assertEqual(plan.target_kv_tokens, expected)
        # The window restarts after a growth.
        plan = policy.decide(
            observation=_headroom_obs(5 << 30, capacity=expected), current=plan
        )
        self.assertEqual(plan.target_kv_tokens, expected)

    def test_no_growth_when_spare_is_below_the_minimum(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = MemoryHeadroomPolicy(ledger=_LEDGER)
        plan = SpecPolicyPlan()
        for _ in range(mod._HEADROOM_WINDOW_STEPS + 5):
            plan = policy.decide(
                observation=_headroom_obs(mod._HEADROOM_MARGIN_BYTES + (100 << 20)),
                current=plan,
            )
        self.assertIsNone(plan.target_kv_tokens)

    def test_composite_members_keep_each_others_fields(self):
        from sglang.srt.managers.scheduler_components.spec_policy import policy as mod

        policy = CompositePolicy(
            [PriorityWeightPolicy(), MemoryHeadroomPolicy(ledger=_LEDGER)]
        )
        plan = SpecPolicyPlan()
        for _ in range(mod._HEADROOM_WINDOW_STEPS):
            plan = policy.decide(observation=_headroom_obs(5 << 30), current=plan)
        self.assertIsNotNone(plan.target_kv_tokens)
        self.assertEqual(plan.low_priority_weight, 0.25)


class TestSpecPolicyController(CustomTestCase):
    def test_observe_policy_traces_without_acting(self):
        worker = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            controller = SpecPolicyController(
                policy_name="observe",
                sps_table_path=None,
                draft_worker=worker,
                trace_path=str(trace),
                is_trace_writer=True,
            )
            self.addCleanup(controller.close)
            for _ in range(3):
                controller.on_batch_result(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=2,
                    num_correct_drafts=3,
                    num_verify_tokens_per_req=8,
                    num_running=2,
                    num_queued=0,
                    pool_stats=_pool_stats(),
                    reqs=[],
                )
            header, *records = [
                json.loads(line) for line in trace.read_text().splitlines()
            ]
        # Offline replay reads the ledger header first, then one record per step.
        self.assertEqual(header, {"ledger": None})
        self.assertEqual([r["observation"]["step"] for r in records], [1, 2, 3])
        self.assertEqual(records[0]["observation"]["forward_mode"], "DECODE")
        self.assertEqual(records[0]["plan"], {})
        worker.set_dspark_forced_budget_frac.assert_not_called()

    def test_pausing_policy_rejects_a_worker_that_cannot_pause(self):
        # Caught at startup instead of as an AttributeError on the first pause.
        for name in ("regime", "conditional"):
            with self.assertRaises(ValueError):
                SpecPolicyController(
                    policy_name=name,
                    sps_table_path=None,
                    draft_worker=Mock(),
                    trace_path=None,
                    is_trace_writer=False,
                )

    def test_width_policy_rejects_a_worker_without_a_width_knob(self):
        with self.assertRaises(ValueError):
            SpecPolicyController(
                policy_name="width",
                sps_table_path=None,
                draft_worker=Mock(),
                trace_path=None,
                is_trace_writer=False,
            )

    def test_non_writer_rank_skips_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            SpecPolicyController(
                policy_name="observe",
                sps_table_path=None,
                draft_worker=Mock(),
                trace_path=str(trace),
                is_trace_writer=False,
            )
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
