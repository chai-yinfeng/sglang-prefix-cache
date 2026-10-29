from __future__ import annotations

from typing import Sequence

import msgspec
import torch


class MemoryLedger(msgspec.Struct, frozen=True):
    """Static GPU byte accounting of what speculation costs, recorded once at startup."""

    kv_capacity_tokens: int
    target_kv_bytes_per_token: float
    draft_kv_bytes_per_token: float
    target_weights_bytes: int
    draft_weights_bytes: int

    def tokens_for_bytes(self, num_bytes: int) -> int:
        """KV tokens (target + draft lanes) that num_bytes of device memory would hold."""
        per_token = self.target_kv_bytes_per_token + self.draft_kv_bytes_per_token
        return int(num_bytes // per_token) if per_token > 0 else 0


def kv_pool_bytes(pool) -> int:
    # A VMM-backed pool's tensors span the whole reservation; only the backed prefix is real.
    if pool.post_capture_active:
        return pool.post_capture_backed_bytes
    k_bytes, v_bytes = pool.get_kv_size_bytes()
    # get_kv_size_bytes sums numpy ints, which the msgspec trace encoder rejects.
    return int(k_bytes + v_bytes)


def _unique_param_bytes(models: Sequence[torch.nn.Module], seen: set[int]) -> int:
    total = 0
    for model in models:
        for param in model.parameters():
            # Drafts that share the target's embedding or lm_head point at the same storage.
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += param.numel() * param.element_size()
    return total


def build_memory_ledger(*, target_runner, draft_runners: Sequence) -> MemoryLedger:
    capacity = int(target_runner.token_to_kv_pool.size)
    target_kv = kv_pool_bytes(target_runner.token_to_kv_pool)
    draft_pools = [
        runner.token_to_kv_pool
        for runner in draft_runners
        if runner.token_to_kv_pool is not target_runner.token_to_kv_pool
    ]
    draft_kv = sum(kv_pool_bytes(pool) for pool in draft_pools)
    seen: set[int] = set()
    target_weights = _unique_param_bytes([target_runner.model], seen)
    draft_weights = _unique_param_bytes([r.model for r in draft_runners], seen)
    return MemoryLedger(
        kv_capacity_tokens=capacity,
        target_kv_bytes_per_token=target_kv / capacity if capacity else 0.0,
        draft_kv_bytes_per_token=draft_kv / capacity if capacity else 0.0,
        target_weights_bytes=target_weights,
        draft_weights_bytes=draft_weights,
    )
