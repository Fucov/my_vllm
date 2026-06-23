from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.prefill_policy = config.prefill_policy
        self.prefill_chunk_size = config.prefill_chunk_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_late_merge,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "prefill_batches": 0,
            "decode_batches": 0,
            "scheduled_prefill_tokens": 0,
            "scheduled_decode_tokens": 0,
            "preemptions": 0,
        }
        self.block_manager.reset_metrics()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    """
    修改后行为：
    1. waiting 非空且 prefill_policy == "fair" 时调用 _schedule_fair_prefill()
    2. waiting 非空且 policy 为 fcfs 时调用 _schedule_fcfs()
    3. waiting 为空时调用 _schedule_decode()
    """
    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.prefill_policy == "fair":
            scheduled = self._schedule_fair_prefill()
            if scheduled:
                self.metrics["prefill_batches"] += 1
                self.metrics["scheduled_prefill_tokens"] += sum(seq.num_scheduled_tokens for seq in scheduled)
                return scheduled, True
            return self._schedule_decode()

        return self._schedule_fcfs()

    def _schedule_fcfs(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.metrics["prefill_batches"] += 1
            self.metrics["scheduled_prefill_tokens"] += num_batched_tokens
            return scheduled_seqs, True

        return self._schedule_decode()
    """
    关键逻辑：
    1. 扫描 waiting 队列一轮；
    2. 如 sequence 尚未分配 block，则先调用 block_manager.can_allocate() 和 allocate()；
    3. remaining_budget = max_num_batched_tokens - scheduled_tokens；
    4. chunk_tokens = min(seq.num_tokens - seq.num_scheduled_tokens, prefill_chunk_size, remaining_budget)；
    5. chunk 未完成则重新放回 waiting；
    6. prompt prefill 完成后将 seq 状态改为 RUNNING 并进入 running 队列。
    """
    def _schedule_fair_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        num_waiting = len(self.waiting)
        visited = 0

        while self.waiting and visited < num_waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget == 0:
                break
            seq = self.waiting.popleft()
            visited += 1

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.append(seq)
                    continue
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if num_tokens <= 0:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                continue

            chunk = min(num_tokens, self.prefill_chunk_size, remaining_budget)
            seq.num_scheduled_tokens = chunk
            num_batched_tokens += chunk
            scheduled_seqs.append(seq)

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)

        return scheduled_seqs

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self.metrics["decode_batches"] += 1
        self.metrics["scheduled_decode_tokens"] += len(scheduled_seqs)
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.metrics["preemptions"] += 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
