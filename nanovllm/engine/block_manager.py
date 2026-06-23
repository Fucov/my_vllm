from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_late_merge: bool = False):
        self.block_size = block_size
        self.enable_prefix_late_merge = enable_prefix_late_merge
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "allocations": 0,
            "deallocations": 0,
            "peak_used_blocks": len(self.used_block_ids),
            "prefix_probes": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
            "late_merge_attempts": 0,
            "late_merge_successes": 0,
            "late_merge_reclaimed_blocks": 0,
        }

    def _update_peak_used_blocks(self):
        self.metrics["peak_used_blocks"] = max(self.metrics["peak_used_blocks"], len(self.used_block_ids))

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.metrics["allocations"] += 1
        self._update_peak_used_blocks()
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        self.metrics["deallocations"] += 1

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            self.metrics["prefix_probes"] += 1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                self.metrics["prefix_misses"] += 1
                break
            self.metrics["prefix_hits"] += 1
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            if self.enable_prefix_late_merge:
                self._late_merge_block(seq, i, h, token_ids)
            else:
                self.hash_to_block_id[h] = block.block_id
    """
    行为：
    1. 查询 canonical block；
    2. 如果 canonical 与当前 block 不是同一个，且 token_ids 完全相同，则允许合并；
    3. 将 seq.block_table[block_index] 重定向到 canonical_id；
    4. canonical.ref_count += 1；
    5. duplicate.ref_count -= 1；
    6. duplicate.ref_count == 0 时回收 duplicate physical block；
    7. 统计 late_merge_successes 和 reclaimed_blocks；
    8. 调用 _assert_consistent() 做 ref_count 非负检查。
    """
    def _late_merge_block(self, seq: Sequence, block_index: int, h: int, token_ids: list[int]):
        block_id = seq.block_table[block_index]
        canonical_id = self.hash_to_block_id.get(h, -1)
        if canonical_id == -1:
            self.hash_to_block_id[h] = block_id
            return
        self.metrics["late_merge_attempts"] += 1
        if canonical_id == block_id:
            return
        canonical = self.blocks[canonical_id]
        duplicate = self.blocks[block_id]
        if canonical.token_ids != token_ids:
            self.hash_to_block_id[h] = block_id
            return
        if canonical_id not in self.used_block_ids:
            self.free_block_ids.remove(canonical_id)
            self.used_block_ids.add(canonical_id)
            canonical.ref_count = 0
        seq.block_table[block_index] = canonical_id
        canonical.ref_count += 1
        duplicate.ref_count -= 1
        self.metrics["late_merge_successes"] += 1
        if duplicate.ref_count == 0:
            self._deallocate_block(block_id)
            self.metrics["late_merge_reclaimed_blocks"] += 1
        self._assert_consistent()

    def _assert_consistent(self):
        used = {block.block_id for block in self.blocks if block.ref_count > 0}
        assert used == self.used_block_ids
        assert len(self.used_block_ids) + len(self.free_block_ids) == len(self.blocks)
