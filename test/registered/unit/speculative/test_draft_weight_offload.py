"""Draft weights released from the device come back at the same addresses.

CUDA graphs captured before the release keep pointing at the draft weights, so a
graph replayed after release + restore must produce exactly the pre-release output;
the release must actually return device memory; and weights shared with the target
must stay where they are.
"""

import unittest

import torch

from sglang.srt.speculative.draft_weight_offload import DraftWeightOffloader
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA VMM")
class TestDraftWeightOffload(unittest.TestCase):
    def test_graph_replays_identically_after_release_and_restore(self):
        torch.manual_seed(0)
        shared = torch.nn.Linear(1024, 1024, bias=False).cuda().half()
        draft = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096).cuda().half(),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024).cuda().half(),
        )
        draft.add_module("head", shared)
        shared_ptr = shared.weight.untyped_storage().data_ptr()
        offloader = DraftWeightOffloader(
            models=[draft],
            device_id=torch.cuda.current_device(),
            shared_ptrs={shared_ptr},
        )
        self.assertEqual(shared.weight.untyped_storage().data_ptr(), shared_ptr)

        static_in = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            draft(static_in)  # warm up outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = draft(static_in)
        graph.replay()
        torch.cuda.synchronize()
        expected = static_out.clone()

        free_before, _ = torch.cuda.mem_get_info()
        offloader.release()
        free_released, _ = torch.cuda.mem_get_info()
        self.assertTrue(offloader.released)
        # Two 1024x4096 fp16 layers plus biases: about 16 MiB returned to the driver.
        self.assertGreaterEqual(
            free_released - free_before, offloader.num_bytes - (2 << 20)
        )

        offloader.restore()
        static_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
