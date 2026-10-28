"""Release a draft model's weights from the device and bring them back at the same addresses.

The draft's parameter storages are copied once into a CUDA VMM reservation (before CUDA
graph capture), and every parameter is re-pointed into it with its original offset and
strides. Releasing unmaps the physical memory but keeps the virtual addresses, so the
captured graphs and any tensor views stay valid; restoring maps fresh memory at the same
addresses and copies the weights back from a pinned host copy.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena

logger = logging.getLogger(__name__)

# Keeps every storage start aligned for any dtype and vectorized loads.
_STORAGE_ALIGN = 256


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class DraftWeightOffloader:
    def __init__(
        self,
        *,
        models: Sequence[torch.nn.Module],
        device_id: int,
        shared_ptrs: set[int],
    ) -> None:
        """shared_ptrs: storage data pointers also owned by the target (left in place)."""
        params = [
            p
            for model in models
            for p in model.parameters()
            if p.is_cuda and p.untyped_storage().data_ptr() not in shared_ptrs
        ]
        storages: dict[int, torch.UntypedStorage] = {}
        for p in params:
            storage = p.untyped_storage()
            storages.setdefault(storage.data_ptr(), storage)
        offsets, total = {}, 0
        for ptr, storage in storages.items():
            offsets[ptr] = total
            total = _align(total + storage.nbytes(), _STORAGE_ALIGN)
        self.num_bytes = total
        self._released = False
        if total == 0:
            self._arena = None
            return

        self._arena = KvVmmArena(device_id, reserve_bytes=total + (64 << 20))
        with torch.cuda.device(device_id), torch.cuda.use_mem_pool(self._arena.pool):
            self._flat = torch.empty(
                total, dtype=torch.uint8, device=f"cuda:{device_id}"
            )
        self._offset = self._flat.data_ptr() - self._arena.base
        self._arena.commit_range(self._offset, total)
        flat_storage = self._flat.untyped_storage()
        for ptr, storage in storages.items():
            start = offsets[ptr]
            self._flat[start : start + storage.nbytes()].copy_(
                torch.empty(0, dtype=torch.uint8, device=self._flat.device).set_(
                    storage
                )
            )
        for p in params:
            start = offsets[p.untyped_storage().data_ptr()]
            itemsize = p.element_size()
            p.data = torch.empty(0, dtype=p.dtype, device=p.device).set_(
                flat_storage,
                start // itemsize + p.storage_offset(),
                p.size(),
                p.stride(),
            )
        del storages
        torch.cuda.synchronize()
        self._host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        self._host.copy_(self._flat)
        torch.cuda.empty_cache()
        logger.info(
            "Draft weights moved into a releasable VMM region: %.2f GB in %d tensors",
            total / (1 << 30),
            len(params),
        )

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Unmap the draft weights; nothing may read them until restore()."""
        if self._arena is None or self._released:
            return
        self._arena.decommit_all()
        self._released = True

    def restore(self) -> None:
        if self._arena is None or not self._released:
            return
        self._arena.commit_range(self._offset, self.num_bytes)
        self._flat.copy_(self._host)
        torch.cuda.synchronize()
        self._released = False
