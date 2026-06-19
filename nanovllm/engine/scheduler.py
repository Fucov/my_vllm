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
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 为什么prefill优先？因为prefill阶段的 seq 还没有写入 KV Cache, 所以它们的内存占用更大, 更可能成为 decode 阶段的瓶颈; 
        # 而且 prefill 阶段的 seq 还没有生成任何 token, 所以它们的 ITL/TPOT 更长, 优先调度它们可以更早地开始 prefill, 从而更早地进入 decode 阶段。
        # 数量限制上，prefill基本只受token上限限制；decode阶段的seq已经写入KV Cache了，内存占用更小，所以只受seq数量上限限制。
        scheduled_seqs = [] # scheduled_seqs 的意思是本步被调度的 Sequence 列表。每一 step 初始为空
        num_batched_tokens = 0

        # prefill 阶段
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # 为什么是waiting而不是running？因为 prefill 阶段的 seq 都还在 waiting 队列里, 只有 decode 阶段的 seq 才会被移到 running 队列里。
            """
            prefill逻辑的核心是一个 while 循环, 每次循环都从 waiting 队列里取出第一个 seq 来考虑是否调度它。这个循环会一直进行, 直到 waiting 队列空了, 或者已经调度的 seq 数量达到了 max_num_seqs 的上限了。
            """
            seq = self.waiting[0] # 取出 waiting 队列的第一个 Sequence, 但不从队列里移除它, 因为还要判断它是否能被调度。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            """
            为什么除了seq还要限制tokens? 因为即使 seq 的数量没有超过 max_num_seqs, 但如果它们的 token 数量加起来超过了 max_num_batched_tokens, 就可能导致本步的输入过大而无法在 GPU 上运行, 从而引发 OOM。
            - 这个限制是针对 prefill 阶段的 seq 的, 因为 prefill 阶段的 seq 还没有写入 KV Cache, 所以它们的内存占用更大, 更可能成为 decode 阶段的瓶颈; 
            - decode 阶段的 seq 已经写入 KV Cache, 所以它们的内存占用更小, 只要 seq 的数量没有超过 max_num_seqs 就行了。
            """
            if remaining == 0:
                # 如果本步已经没有剩余的 token 预算了, 就直接 break, 不再考虑后面的 seq 了。
                break
            if not seq.block_table: 
                # 如果这个 seq 还没有分配过 KV Cache 块, 就先调用 block_manager.can_allocate(seq) 来看看它还能分配多少块 KV Cache 块, 从而计算出它还能写入多少个 token 到 KV Cache 里。
                # 这个数量是一个 upper bound（先给整个seq分配空间，可能一次无法填充完）, 实际能写入的 token 数还要看 seq.num_tokens - seq.num_cached_tokens 的值。
                num_cached_blocks = self.block_manager.can_allocate(seq)

                if num_cached_blocks == -1: 
                    # 如果这个 seq 连最小的 KV Cache 块都分配不了, 就说明它当前无法进入 prefill 阶段, 可能是因为它的 prompt 太长了, 超过了模型的最大输入长度,
                    # 或者是因为当前 GPU 上的 KV Cache 已经被其他 seq 占满了。无论是哪种情况, 这个 seq 都暂时无法被调度, 直接 break, 不再考虑后面的 seq 了。
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size 
                # 这个 num_tokens 是在 block_manager.can_allocate 的基础上算出来的, 表示这个 seq 还剩多少 token 没有被 KV Cache 覆盖到, 也就是还需要写入 KV Cache 的 token 数量。
            else: 
                # 如果这个 seq 已经分配过 KV Cache 块了, 就直接用 seq.num_tokens - seq.num_cached_tokens 来计算它还需要写入 KV Cache 的 token 数量。
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  
                # 如果这个 seq 还需要写入 KV Cache 的 token 数量超过了本步剩余的 token 预算了,同时不是队首（不是前几个Token序列）, 就直接 break, 不再考虑后面的 seq 了。
                break
            # 下面是 prefill 真正分配，上面只是预估这个 seq 还需要写入 KV Cache 的 token 数量来判断它是否能被调度。
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 这个 seq 本步计划写入 KV Cache 的 token 数量, 不能超过它还需要写入的 token 数量, 也不能超过本步剩余的 token 预算。
            num_batched_tokens += seq.num_scheduled_tokens # 更新本步已经计划写入 KV Cache 的 token 数量, 用于后续的预算判断。

            """
            num_cached_tokens: 进入本step之前，已经写入KV Cache的Token数
            num_scheduled_tokens: 本step计划写入KV Cache的Token数
            num_tokens: 进入本step之前，尚未写入KV Cache的Token数
                - 如果 num_cached_tokens + num_scheduled_tokens == seq.num_tokens，说明本step计划写入KV Cache的Token数足以覆盖剩余的所有Token，
                    可以把status 设为 SequenceStatus.RUNNING，seq就可以切到 decode 队列了
                - prefill阶段, nums_tokens就是prompt长度（还没append过任何decode token）
            """
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:  
                seq.status = SequenceStatus.RUNNING # 切到 decode 队列
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq) # 把这个 seq 加入本步的调度列表里, 但还没有真正从 waiting 队列里移除它, 因为它可能还没有完全覆盖到 KV Cache 里, 还需要在后续的 step 里继续 prefill。

        # 早停机制：如果 prefill 阶段有序列被调度，就直接返回 scheduled_seqs 和 True，表示当前阶段是 prefill。
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 阶段
        while self.running and len(scheduled_seqs) < self.max_num_seqs: # 为什么是running而不是waiting？因为 decode 阶段的 seq 都在 running 队列里, 只有 prefill 阶段的 seq 才会在 waiting 队列里。
            seq = self.running.popleft() # 取出 running 队列的第一个 Sequence, 并从队列里移除它, 因为它要么被调度了, 要么被抢占了。
            while not self.block_manager.can_append(seq): 
                # 显存不足以为当前seq再加一个token
                if self.running:
                    self.preempt(self.running.pop()) # 如果还存在其他的 RUNNING seq，把队尾移除running队列，释放对应的KV cache块
                else:
                    self.preempt(seq) # 如果只剩自己这个seq了, 把自己设置为waiting，释放自己的KV cache块
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq) # 执行追加
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) # 还原顺序？为什么要这一步？为什么要reversed?
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # preempt 的意思是当 decode 阶段的 seq 因为 KV Cache 不够用而被抢占时, 把它切回 waiting 队列,并且如果它之前已经分配过 KV Cache 块了就先 deallocate 回收掉,
        # 等到下次被 schedule 的时候再重新 allocate。prefill 阶段的 seq 不会被 preempt, 因为 prefill 阶段的 seq 都还在 waiting 队列里, 只有 decode 阶段的 seq 才会被 preempt。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # decode阶段每生成一个token就调用一次 postprocess 来更新 seq 的状态和 KV Cache 的分配情况。
        # prefill阶段在每个step结束时调用一次 postprocess 来更新 seq 的状态和 KV Cache 的分配情况,
        #  prefill阶段的 token_ids 是本step计划写入KV Cache的Token ID列表, decode阶段的 token_ids 是本step实际生成的Token ID列表。
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue # prefill阶段还没覆盖所有token，继续prefill
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
