from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0 # 引用计数, 表示有多少 seq 的 KV 块表中包含这个块, 由 BlockManager 调度时更新, 当 ref_count 从 0 变为 1 时分配这个块, 当 ref_count 从 1 变为 0 时 deallocate 这个块
        self.hash = -1 # KV 块的哈希值, 由 BlockManager 调度时更新, 用于判断不同 seq 的 KV 块是否相同以实现块重用
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 # reset 时 ref_count 设为 1 是为了方便 BlockManager 在 allocate 时直接把它分配给 seq 而不需要再增加一次 ref_count
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 所有块实体list, 由 BlockManager 调度时更新, 存储所有 KV 块的信息，包括块号、引用计数、哈希值和块内的 token ID 列表
        self.hash_to_block_id: dict[int, int] = dict() # KV 块哈希值到块号的映射表dict, 由 BlockManager 调度时更新, 当一个块被分配给 seq 时把它的哈希值和块号添加到表中, 当一个块被 deallocate 时把它的哈希值从表中移除
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 可用块号的队列deque, 由 BlockManager 调度时更新, 当分配一个块时把它从队列中弹出, 当 deallocate 一个块时把它添加到队列尾部
        self.used_block_ids: set[int] = set() # 已用块号集合set, 由 BlockManager 调度时更新, 当分配一个块时把它添加到集合中, 当 deallocate 一个块时把它从集合中移除

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 先把前缀哈希值转换成字节串并更新哈希对象, 这样可以在计算当前块的哈希值时把前面块的哈希值也考虑进去, 从而实现块重用的跨块连续性, 避免因为块边界切分导致的哈希碰撞和错误重用
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft() # 从队列头部弹出一个可用块号
        block = self.blocks[block_id]
        assert block.ref_count == 0 #为什么不是1？因为在 reset 时 ref_count 设为 1 是为了方便 BlockManager 在 allocate 时直接把它分配给 seq 而不需要再增加一次 ref_count, 所以当一个块被 deallocate 时它的 ref_count 已经是 0 了
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id: 
            # 如果这个块之前被分配过且它的哈希值在映射表中正确地映射到它自己,说明它之前的内容还在映射表中,需要把它从映射表中移除以避免后续错误地重用这个块
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0 # deallocate 时块的引用计数应该已经是 0 了, 因为 BlockManager 在 deallocate seq 时会先把块的 ref_count 减去 1, 如果 ref_count 变为 0 就调用 _deallocate_block 来真正回收这个块
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks # 需要分配的新块数, 初始值是 seq 的 KV 块数, 后续会根据块重用的情况进行调整
        for i in range(seq.num_blocks - 1):  # 最后一个块通常不太可能被重用, 因为它可能只包含少量 token 且经常是变化的, 所以在 can_allocate 时我们只考虑前面 num_blocks - 1 个块的重用情况, 这样可以减少计算哈希值和比较 token ID 列表的开销, 同时也能覆盖大部分块重用的场景
            token_ids = seq.block(i) # 取第 i 个 KV 块的 Token ID 列表, 用于计算哈希值和判断块重用
            h = self.compute_hash(token_ids, h) # 计算第 i 个 KV 块的哈希值, 前缀哈希值是前面块的哈希值, 这样可以在计算当前块的哈希值时把前面块的哈希值也考虑进去, 从而实现块重用的跨块连续性, 避免因为块边界切分导致的哈希碰撞和错误重用
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1 # 如果这个块当前正在被使用,说明它虽然之前被分配过但现在不能重用, 需要算作一个新的块来分配, 所以需要把 num_new_blocks 减去 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks # 返回可以重用的块数, 由 LLMEngine 调度时调用, 用于判断一个 seq 是否可以被调度以及它需要分配多少新的块

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
        for block_id in reversed(seq.block_table):  # 反向迭代 seq 的 KV 块表, 从后往前处理块的 deallocate, 这样可以在遇到第一个不能重用的块时就停止迭代, 因为后面的块更不可能被重用了
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0 
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:# 作用是判断 seq 是否可以 append 一个 token 了, 也就是判断 seq 的最后一个 KV 块是否还有剩余空间可以放下一个 token, 如果没有就需要先 allocate 一个新的块才能 append
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) 

    def may_append(self, seq: Sequence):
        # 作用是当 seq 可以 append 一个 token 时, 如果 seq 的最后一个 KV 块已经满了(也就是 seq 的 token 数对 block_size 取模等于 1, 因为 append 之前还没有把这个 token 加入 seq 中), 
        # 就先 allocate 一个新的块给 seq, 这样就可以保证 seq 的最后一个 KV 块在 append 之后不会超过 block_size 的限制
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
            self.hash_to_block_id[h] = block.block_id
