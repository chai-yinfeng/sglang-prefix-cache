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
from sglang.srt.managers.scheduler_components.spec_policy.policy import (  # noqa: E402
    LoadAwareVerifyPolicy,
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
        _observe(sensor, mode=ForwardMode.EXTEND, now=1.5)
        after_prefill = _observe(sensor, mode=ForwardMode.DECODE, now=1.6)
        self.assertAlmostEqual(after_prefill.step_ms_ema, 20.0)

    def test_accept_length_includes_bonus_token(self):
        sensor = SpecPolicySensor(ema_alpha=0.5)
        obs = _observe(sensor, mode=ForwardMode.DECODE, now=1.0, correct=6, verify=8)
        self.assertAlmostEqual(obs.accept_length_ema, 2.5)
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
            records = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual([r["observation"]["step"] for r in records], [1, 2, 3])
        self.assertEqual(records[0]["observation"]["forward_mode"], "DECODE")
        self.assertEqual(records[0]["plan"], {})
        worker.set_dspark_forced_budget_frac.assert_not_called()

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
