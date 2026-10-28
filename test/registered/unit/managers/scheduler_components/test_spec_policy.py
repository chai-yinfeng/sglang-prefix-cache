import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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
from sglang.srt.managers.scheduler_components.spec_policy.sensor import (  # noqa: E402
    SpecPolicySensor,
)
from sglang.srt.managers.scheduler_components.spec_policy.types import (  # noqa: E402
    SpecPolicyPlan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

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


class TestSpecPolicyActuator(CustomTestCase):
    def test_only_changed_budget_reaches_worker(self):
        worker = Mock()
        actuator = SpecPolicyActuator(draft_worker=worker)
        same = SpecPolicyPlan(dspark_verify_budget_frac=0.5)
        actuator.apply(plan=same, current=same)
        worker.set_dspark_forced_budget_frac.assert_not_called()
        actuator.apply(plan=SpecPolicyPlan(), current=same)
        worker.set_dspark_forced_budget_frac.assert_called_once_with(None)


class TestSpecPolicyController(CustomTestCase):
    def test_observe_policy_traces_without_acting(self):
        worker = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            controller = SpecPolicyController(
                policy_name="observe",
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
                draft_worker=Mock(),
                trace_path=str(trace),
                is_trace_writer=False,
            )
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
