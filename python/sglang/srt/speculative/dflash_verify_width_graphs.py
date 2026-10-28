"""Target-verify CUDA graphs at DFlash verify widths narrower than the block.

The block-width decode runner only replays batches of its own width, so a runtime
verify width (`DFlashWorkerV2.set_dflash_verify_width`) below the block runs eager.
Each extra width gets its own attention backend: flashinfer keeps its graph metadata
per batch size, so two widths sharing one backend would overwrite each other's wrappers.
"""

import logging
import time
from typing import Dict, Optional, Sequence

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    get_attention_backend,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)


class VerifyWidthGraphRunner:
    """Routes each target forward to the decode runner captured at its verify width.

    Widths without a runner of their own go to the block-width runner, which rejects
    them, so they run eager. DFlash plans a verify batch through the runner before the
    forward replays it; both calls see the same batch width, so they reach one child.
    """

    def __init__(
        self,
        *,
        base: DecodeCudaGraphRunner,
        by_width: Dict[int, DecodeCudaGraphRunner],
    ) -> None:
        self._base = base
        self._by_width = by_width
        self._active = base
        self.capture_bs = base.capture_bs

    @property
    def bs(self) -> int:
        return self._active.bs

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        return self._route(forward_batch).can_run_graph(forward_batch)

    def load_batch(self, forward_batch: ForwardBatch, **kwargs):
        self._active = self._route(forward_batch)
        return self._active.load_batch(forward_batch, **kwargs)

    def execute(self, forward_batch: ForwardBatch, **kwargs):
        self._active = self._route(forward_batch)
        return self._active.execute(forward_batch, **kwargs)

    def _route(self, forward_batch: ForwardBatch) -> DecodeCudaGraphRunner:
        spec_info = forward_batch.spec_info
        if spec_info is None:
            return self._base
        return self._by_width.get(spec_info.num_tokens_per_req, self._base)


def build_verify_width_graph_runner(
    *,
    model_runner,
    widths: Sequence[int],
    device: str,
    gpu_id: int,
) -> Optional[VerifyWidthGraphRunner]:
    """Capture one target-verify runner per width; None when the target has no decode graphs."""
    base = model_runner.decode_cuda_graph_runner
    if not isinstance(base, DecodeCudaGraphRunner):
        return None
    by_width = {}
    for width in sorted(set(widths), reverse=True):
        tic = time.perf_counter()
        before_gb = get_available_gpu_memory(device, gpu_id)
        # The backend factory records its workspace choice on the runner; keep the
        # runner's own. The float workspace is scratch, shared like the draft's.
        init_new_workspace = model_runner.init_new_workspace
        try:
            attn_backend = get_attention_backend(
                model_runner=model_runner, init_new_workspace=False
            )
        finally:
            model_runner.init_new_workspace = init_new_workspace
        by_width[width] = DecodeCudaGraphRunner(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_draft_tokens=width,
        )
        logger.info(
            "Captured DFLASH verify graphs at width %d: %.2f s, %.2f GB.",
            width,
            time.perf_counter() - tic,
            before_gb - get_available_gpu_memory(device, gpu_id),
        )
    return VerifyWidthGraphRunner(base=base, by_width=by_width)
