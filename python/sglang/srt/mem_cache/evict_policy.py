from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU (Zhang et al., arXiv:2510.15152).

    A conversation with history length L whose next prompt is expected to add
    Q_hat tokens only has to keep L + Q_hat - threshold tokens cached to hold its
    next prefill under the TTFT budget; tokens past that budget cannot improve
    tail latency and are "TEL-safe", i.e. free to evict. Such nodes are reported
    as infinitely old, which is the implementation the paper suggests: the
    existing eviction driver then drains them before anything else (the paper's
    phase 1) and continues in plain recency order once they run out (phase 2),
    so neither eviction loop needs to know about T-LRU.

    threshold and next_prompt_estimate are token counts, whereas the paper states
    both in blocks; multiply the paper's values by page_size to convert.
    """

    def __init__(self, threshold: int = 0, next_prompt_estimate: int = 0):
        self.threshold = threshold
        self.next_prompt_estimate = next_prompt_estimate

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # node._tlru_history_len is the branch's high-water depth, i.e. the
        # paper's L, and deliberately does not shrink when the tail is trimmed.
        # Deriving L from what is still resident instead would leave the
        # shortened conversation over budget on the next pass too, and T-LRU
        # would walk it down to nothing rather than stopping after
        # (threshold - Q_hat) tokens.
        budget = max(
            node._tlru_history_len + self.next_prompt_estimate - self.threshold, 0
        )
        cached_without_this_node = node._tlru_cached_prefix_len - len(node.key)
        tel_safe = cached_without_this_node >= budget
        return (-1 if tel_safe else 0, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


# Measured on Qwen3-8B BF16 / RTX 5090 / FA4 (TTFT of a cold prefill of that many
# tokens, ms); recalibrate per model and GPU through the policy config.
_DEFAULT_PREFILL_MS_AT_TOKENS = (
    (1024, 93.0),
    (2048, 164.0),
    (4096, 319.0),
    (8192, 686.0),
    (16384, 1567.0),
)
# Same setup, HiCache direct + layer_first host restore: fixed cost + per token.
_DEFAULT_RESTORE_FIXED_MS = 15.0
_DEFAULT_RESTORE_MS_PER_TOKEN = 0.0027
# Arbitrary; logical-clock ticks (two per tree operation) after which an idle node's
# hit count counts half.
_DEFAULT_IDLE_HALF_LIFE_TICKS = 4096.0


class ValueDensityStrategy(EvictionStrategy):
    """Evict the node whose expected re-fetch cost per resident token is lowest.

    value = reuse x refetch_ms / tokens, where reuse is the hit count decayed by
    idle time (LRFU style). With a host tier (HiCache write-through) an evicted
    device node survives on host, so its re-fetch is a host restore, whose fixed
    cost favors keeping short hot prefixes on device; without one it is the
    prefill of the node's tokens at its depth, which grows with depth.
    """

    def __init__(
        self,
        host_tier: bool = False,
        idle_half_life_ticks: float = _DEFAULT_IDLE_HALF_LIFE_TICKS,
        prefill_ms_at_tokens: Sequence[Sequence[float]] = _DEFAULT_PREFILL_MS_AT_TOKENS,
        restore_fixed_ms: float = _DEFAULT_RESTORE_FIXED_MS,
        restore_ms_per_token: float = _DEFAULT_RESTORE_MS_PER_TOKEN,
    ):
        self.host_tier = host_tier
        self.idle_half_life_ticks = float(idle_half_life_ticks)
        self._prefill_tokens = [float(t) for t, _ in prefill_ms_at_tokens]
        self._prefill_ms = [float(ms) for _, ms in prefill_ms_at_tokens]
        self.restore_fixed_ms = restore_fixed_ms
        self.restore_ms_per_token = restore_ms_per_token

    def get_priority(self, node: TreeNode) -> float:
        # Imported here: components.base reaches mem_cache.utils, which imports this module.
        from sglang.srt.mem_cache.unified_cache.components.base import (
            peek_time_counter,
        )

        tokens = max(len(node.key), 1) if node.key is not None else 1
        idle = max(peek_time_counter() - node.last_access_time, 0.0)
        reuse = (node.hit_count + 1) * 0.5 ** (idle / self.idle_half_life_ticks)
        return reuse * self.refetch_ms(node=node, tokens=tokens) / tokens

    def refetch_ms(self, *, node: TreeNode, tokens: int) -> float:
        # A device node evicted with a host tier survives on host; a node already
        # off device (host eviction) has no lower tier and must be recomputed.
        if self.host_tier and not node.evicted:
            return self.restore_fixed_ms + self.restore_ms_per_token * tokens
        end = _prefix_depth(node)
        return self.prefill_ms(end) - self.prefill_ms(end - tokens)

    def prefill_ms(self, tokens: float) -> float:
        """Piecewise-linear prefill time; linear through the origin below the first probe."""
        xs, ys = self._prefill_tokens, self._prefill_ms
        if tokens <= 0:
            return 0.0
        if tokens <= xs[0]:
            return ys[0] * tokens / xs[0]
        i = min(bisect.bisect_left(xs, tokens), len(xs) - 1)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)


def _prefix_depth(node: TreeNode) -> int:
    """Tokens from the root through the end of this node."""
    depth = 0
    cur: Optional[TreeNode] = node
    while cur is not None and cur.key is not None:
        depth += len(cur.key)
        cur = cur.parent
    return depth
