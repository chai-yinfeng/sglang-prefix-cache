"""Growing a post-capture MHA KV pool while serving.

A post-capture pool reserves its whole virtual span up front and backs only what
it serves. Growing it must keep every live slot's contents and every tensor
address (CUDA graphs captured them), back the new span so writes land, and hand
the allocator only pages above the ones already given out.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_PAGE_SIZE = 16
_PRE_CAPTURE_TOKENS = 4096
_STARTUP_TOKENS = 2048
# 8 heads x 128 dims x fp16 = 2 KiB per token: growth crosses several 2 MiB VMM handles.
_HEADS, _DIM, _LAYERS = 8, 128, 2


def _pool():
    return MHATokenToKVPool(
        size=_PRE_CAPTURE_TOKENS,
        page_size=_PAGE_SIZE,
        dtype=torch.float16,
        head_num=_HEADS,
        head_dim=_DIM,
        layer_num=_LAYERS,
        device="cuda",
        enable_memory_saver=False,
        post_capture_active=True,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA VMM")
class TestPostCaptureKVGrowth(unittest.TestCase):
    def test_default_factor_reserves_only_the_pre_capture_size(self):
        pool = _pool()
        self.assertEqual(pool.post_capture_reserved_num_tokens, _PRE_CAPTURE_TOKENS)

    def test_grow_keeps_live_slots_and_addresses(self):
        with envs.SGLANG_POST_CAPTURE_KV_RESERVE_FACTOR.override(2.0):
            pool = _pool()
        grown = 2 * _PRE_CAPTURE_TOKENS
        self.assertEqual(pool.post_capture_reserved_num_tokens, grown)
        pool._finalize_backing_tokens(_STARTUP_TOKENS)
        allocator = PagedTokenToKVPoolAllocator(
            size=_STARTUP_TOKENS,
            page_size=_PAGE_SIZE,
            dtype=torch.float16,
            device="cuda",
            kvcache=pool,
            need_sort=False,
        )
        live = allocator.alloc(_STARTUP_TOKENS // 2)
        marker = torch.randn(
            live.numel(), _HEADS, _DIM, device="cuda", dtype=torch.float16
        )
        for layer in range(_LAYERS):
            pool.k_buffer[layer][live] = marker
        addresses = [t.data_ptr() for t in pool.k_buffer + pool.v_buffer]

        pool._finalize_backing_tokens(grown)
        allocator.grow(grown)

        self.assertEqual(pool.size, grown)
        self.assertEqual(
            [t.data_ptr() for t in pool.k_buffer + pool.v_buffer], addresses
        )
        for layer in range(_LAYERS):
            torch.testing.assert_close(pool.k_buffer[layer][live], marker)
        # Together with the live pages, the rest of the capacity covers every page
        # exactly once (page 0 stays the padding sink).
        rest = allocator.alloc(allocator.available_size())
        pages = sorted(set((torch.cat((live, rest)) // _PAGE_SIZE).tolist()))
        self.assertEqual(pages, list(range(1, grown // _PAGE_SIZE + 1)))
        # The newly backed span is writable end to end.
        new_slots = rest[rest >= _STARTUP_TOKENS + _PAGE_SIZE]
        self.assertEqual(new_slots.numel(), grown - _STARTUP_TOKENS)
        tail = torch.full(
            (new_slots.numel(), _HEADS, _DIM), 3.0, device="cuda", dtype=torch.float16
        )
        pool.v_buffer[_LAYERS - 1][new_slots] = tail
        torch.testing.assert_close(pool.v_buffer[_LAYERS - 1][new_slots], tail)

    def test_graph_captured_writes_reach_slots_past_the_startup_size(self):
        """A CUDA graph captured before growth keeps writing into the grown span."""
        with envs.SGLANG_POST_CAPTURE_KV_RESERVE_FACTOR.override(2.0):
            pool = _pool()
        pool._finalize_backing_tokens(_STARTUP_TOKENS)
        loc = torch.zeros(4, dtype=torch.int64, device="cuda")
        k = torch.randn(4, _HEADS, _DIM, device="cuda", dtype=torch.float16)
        v = torch.randn(4, _HEADS, _DIM, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            pool.set_kv_buffer(None, loc, k, v, layer_id_override=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_kv_buffer(None, loc, k, v, layer_id_override=0)

        grown = 2 * _PRE_CAPTURE_TOKENS
        pool._finalize_backing_tokens(grown)
        loc.copy_(torch.arange(grown - 3, grown + 1, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(pool.k_buffer[0][loc], k)
        torch.testing.assert_close(pool.v_buffer[0][loc], v)

    def test_growth_stops_at_the_reservation(self):
        pool = _pool()
        pool._finalize_backing_tokens(_STARTUP_TOKENS)
        with self.assertRaises(ValueError):
            pool._finalize_backing_tokens(_PRE_CAPTURE_TOKENS + _PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
