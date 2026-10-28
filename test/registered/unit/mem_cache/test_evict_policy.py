"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

import unittest
import unittest.mock
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
    ValueDensityStrategy,
)


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    return node


class TestLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LRUStrategy()

    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LFUStrategy()

    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MRUStrategy()

    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FILOStrategy()

    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))


class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


class _Node:
    """Shape the value-density strategy reads: key, parent, hits, access time, evicted."""

    def __init__(
        self, tokens, parent=None, hit_count=0, last_access_time=0.0, evicted=False
    ):
        self.key = None if tokens is None else [0] * tokens
        self.parent = parent
        self.hit_count = hit_count
        self.last_access_time = last_access_time
        self.evicted = evicted


class TestValueDensityStrategy(unittest.TestCase):
    def setUp(self):
        self.root = _Node(None)
        patcher = unittest.mock.patch(
            "sglang.srt.mem_cache.unified_cache.components.base.peek_time_counter",
            return_value=100.0,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_host_tier_keeps_short_hot_prefixes_on_device(self):
        # A host restore pays a fixed cost per fetch, so per resident token a short
        # node is worth more and the long one is evicted first.
        strategy = ValueDensityStrategy(host_tier=True)
        short = _Node(1024, self.root, last_access_time=100.0)
        long = _Node(12288, self.root, last_access_time=100.0)
        self.assertLess(strategy.get_priority(long), strategy.get_priority(short))

    def test_without_host_tier_deeper_tokens_cost_more(self):
        # Prefill time is superlinear in depth: the same 2K tokens under a 12K prefix
        # cost more to recompute than at the root.
        strategy = ValueDensityStrategy(host_tier=False)
        shallow = _Node(2048, self.root, last_access_time=100.0)
        deep = _Node(2048, _Node(12288, self.root), last_access_time=100.0)
        self.assertLess(strategy.get_priority(shallow), strategy.get_priority(deep))

    def test_host_eviction_prices_recompute(self):
        # A node already off device has no lower tier, even when HiCache is on.
        strategy = ValueDensityStrategy(host_tier=True)
        on_host = _Node(4096, self.root, last_access_time=100.0, evicted=True)
        self.assertAlmostEqual(
            strategy.refetch_ms(node=on_host, tokens=4096), strategy.prefill_ms(4096)
        )

    def test_idle_time_halves_reuse_per_half_life(self):
        strategy = ValueDensityStrategy(host_tier=True, idle_half_life_ticks=50.0)
        fresh = _Node(1024, self.root, hit_count=3, last_access_time=100.0)
        idle = _Node(1024, self.root, hit_count=3, last_access_time=50.0)
        self.assertAlmostEqual(
            strategy.get_priority(idle), strategy.get_priority(fresh) / 2
        )

    def test_registered_as_a_policy_choice(self):
        from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES
        from sglang.srt.mem_cache.utils import get_eviction_strategy

        self.assertIn("value-density", RADIX_EVICTION_POLICY_CHOICES)
        strategy = get_eviction_strategy("value-density", {"host_tier": True})
        self.assertIsInstance(strategy, ValueDensityStrategy)
        self.assertTrue(strategy.host_tier)


if __name__ == "__main__":
    unittest.main()
