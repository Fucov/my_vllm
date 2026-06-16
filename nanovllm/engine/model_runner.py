import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager #eager 模式下每次 forward 都单独调用 kernel, 不使用 CUDA Graph, 适用于模型较小或输入较短的情况, 因为这时 CUDA Graph 的重放开销可能超过它带来的加速。
        self.world_size = config.tensor_parallel_size # world_size 是分布式训练的总进程数,由外部启动脚本根据 torch.distributed.launch 或 torch.multiprocessing.spawn 启动时传入。它决定了模型权重和 KV Cache 在多少个进程间分布式存储和计算。
        self.rank = rank #rank是并行进程的唯一标识,范围是 [0, world_size-1],由外部启动脚本根据 torch.distributed.launch 或 torch.multiprocessing.spawn 启动时传入。rank 0 的进程通常负责接收用户请求、做采样决策和生成 token_id,其他 rank 的进程只负责计算 logits 并等待同步采样结果。
        self.event = event # event 是一个 multiprocessing Event 对象或 Event 对象的列表,用于 rank 0 的进程与其他 rank 的进程之间的同步通信。当 world_size > 1 时, rank 0 的进程通过 event 向其他 rank 的进程发送指令和数据,其他 rank 的进程通过 event 接收指令和数据并执行相应的操作。

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink() # 只有创建 shared memory 的进程才能 unlink, unlink 的作用是标记这个 shared memory 可以被系统回收, 但实际的回收时机由操作系统决定, unlink 后其他进程仍然可以访问这个 shared memory 直到它被回收。
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # call 的意思是当 rank 0 的进程调用 model_runner.call("method_name", *args) 时, 它会先把 method_name 和 args 写入 shared memory, 
        # 然后通过 event 通知其他 rank 的进程来执行这个方法; 当其他 rank 的进程接收到通知时, 它们会从 shared memory 里读出 method_name 和 args, 
        # 然后调用相应的方法来执行。这个 call 函数在 LLMEngine 里被用来让 rank 0 的进程调用 ModelRunner 里定义的各种方法, 比如 run_model、prepare_prefill、prepare_decode 等等。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):# 预先执行一次 forward,以 warmup 模型并测量激活峰值,为后续的 KV Cache 分配和 CUDA Graph 捕获做准备。
        torch.cuda.empty_cache() # 把 PyTorch reserved 但未分配出去的缓存归还给 driver,让 mem_get_info() 返回的 GPU 总占用更准。
        torch.cuda.reset_peak_memory_stats() # 把 PyTorch 视角下的激活峰值重置为当前已分配的字节数,为后续测量激活峰值做准备。warmup 完成时这值等于"权重 + 激活峰值"。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len) # 意思是单条 seq 的长度不能超过模型支持的最大长度,也不能超过单步 forward 一次能处理的 token 上限
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs) # 其中 max_num_seqs 是单步 forward 能并发的最多 seq 条数(默认 512),这一行的意思是在不超过最大并发条数的前提下,把 seq 数填到正好让 num_seqs × seq_len ≈ max_num_batched_tokens,合计输入恰好占满 max_num_batched_tokens
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True) # 第二个实参对应 is_prefill=True,表示执行 prefill 分支(用 prefill 而非 decode,是因为 prefill 一次性处理整段输入,激活张量比 decode 大得多)。
        torch.cuda.empty_cache() #  forward 完成后再 empty_cache(),把临时激活归还给 driver,但 peak 已被记录,不会随这次清理而下降。

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        """
        total:GPU 总显存,由 driver 给出。一张卡的硬件物理上限,与本进程无关。
        free:当前仍未被任何进程占用的字节数,由 driver 给出。
        used = total - free:GPU 视角下当前已被占用的总字节数。
            等式的物理含义:driver 报告的 free 是"仍未被任何进程申请的字节",total - free 就是被申请出去的总量,
            既包含本进程的权重和 PyTorch reserved 缓存,也包含其他用户进程、CUDA context 自身、以及 GPU 上任何其他占用项。
        peak:PyTorch 视角下,从 reset_peak_memory_stats 以来本进程 PyTorch allocator 持有过的最大已分配字节数。
            经过 warmup 后,这值等于"权重 + 激活峰值"。
        current:PyTorch 视角下,本进程 PyTorch allocator 当前持有的已分配字节数。
            warmup 完成时 empty_cache() 已把临时激活归还,current 中剩下的主要就是权重张量,因此 current ≈ 权重大小。
        """
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        """
        求解一个block的大小(字节数):
        一块覆盖 block_size 个 token;
        每个 token 在 attention 的每一层都要单独存 K 和 V 两份(attention 把每个 token 投影成 K、V 两个向量,各按 head 拆分,缓存它们供后续 token 查询);
        每份 K 或 V 的形状是 [每 rank 的 KV head 数, head_dim](每 head 算出一个 head_dim 维的向量);
        每个元素占 dtype.itemsize 字节。把这四个"每"层级相乘,就得到一块的字节数:
        """
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        """
        total * gpu_memory_utilization:可用上限。gpu_memory_utilization 默认 0.9,即只动用 90% 的显存,
            留 10% 安全边界,防止 driver/CUDA context 在 runtime 增长触发 OOM。
        - used:扣除当前已被占用的全部字节(包含本进程权重 + 其他进程 + driver)。
        - peak + current:扣除激活峰值预留。承 4.1:warmup 后 current ≈ 权重大小、peak ≈ 权重 + 激活峰值,两者之差就是激活峰值。
            这部分内存当前虽然空闲,但下一次真实推理 forward 又会用到,必须为它预留空间。
            
        used = current + 其他进程占用
        peak = current + 激活峰值
        current : 权重大小 
        合起来 - used - peak + current = - (used - current) - peak,这是一个更直观的等价改写
        """
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 这个kv_cache是全进程共享的，每个attention的k/v_cache都是这个池的视图，没有数据拷贝
        # 池的大小在启动时根据显存反推
        # per_block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * (num_kv_heads // tp) * head_dim * hf_config.dtype.itemsize
        self.kv_cache = torch.empty(
            2, # K 和 V 两份
            hf_config.num_hidden_layers, # 整个Transformer层数,Qwen3-7B 是 28 层,Qwen3-14B 是 40 层
            config.num_kvcache_blocks, # KV 块数
            self.block_size, # 每块覆盖的 token 数, 默认 256
            num_kv_heads, #
            head_dim # 每个 head 的维度, Qwen3-7B 是 128, Qwen3-14B 是 160,如果 hf_config 中没有 head_dim 属性就用 hidden_size // num_attention_heads 计算得到
        )
        """
        num_attention_heads 是对每个embedding特征维度(hidden_size)进行的拆分数量 ,head_dim是每个拆分后的小块的维度,两者相乘等于 hidden_size
        num_key_value_heads 在标准MHA中等于 num_attention_heads, 也就是Q,K,V都按同样的头数拆分。
        在GQA(Grouped Query Attention)中, num_key_value_heads 是 K 和 V 的头数,通常是 num_attention_heads 的一个约数,比如 Qwen3-7B 的 num_attention_heads 是 56, num_key_value_heads 是 28,说明 K 和 V 的头数是 Q 的一半,也就是每两个 Q 共享一份 K/V。
        在MQA(Multi Query Attention)中, num_key_value_heads 是 1,说明所有 Q 共享同一份 K/V。
        """
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens 
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end # prefill阶段每条 seq 的 KV 长度 = 已缓存的 + 本步计划写入的, 因为 KV Cache 是在 prefill 阶段一起写入的
            input_ids.extend(seq[start:end]) 
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # cu_seqlens_q 存储每条 seq 的 scheduled token 数的前缀和, 用于构建 RaggedTensor 以支持 batch 内 seq 长度不一的情况
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # cu_seqlens_k 存储每条 seq 的 (cached token 数 + scheduled token 数) 的前缀和, 用于构建 RaggedTensor 以支持 batch 内 seq 长度不一的情况
            max_seqlen_q = max(seqlen_q, max_seqlen_q) 
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            """
            如果 seq 还没有 KV 块,说明这是它第一次被调度,本步计划写入的 token 数 seqlen_q 就是它前面所有 token 的数量(因为 prefill 是一次性把所有剩余 token 写入 KV Cache),
            而 start = seq.num_cached_tokens = 0, end = seqlen_q 就是它的 num_tokens。反之如果 seq 已经有 KV 块了,说明之前已经被调度过至少一次,
            之前计划写入的 token 数 seqlen_q 就是它前面所有剩余 token 的数量减去上次计划写入的数量(因为 prefill 是一次性把所有剩余 token 写入 KV Cache),
            而 start = seq.num_cached_tokens 就是它前面所有剩余 token 的数量减去本次计划写入的数量, end = start + seqlen_q 就是它前面所有剩余 token 的数量.            
            """
            if not seq.block_table: 
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size # slot 的起始位置, 要跳过已经cached, 等于 seq.block_table[i] 块号乘以每块覆盖的 token 数
                if i == start_block:
                    slot_start += start % self.block_size # 如果是起始块, slot 的起始位置还要加上本块内要跳过的 token 数, 等于 start 对 block_size 取模
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size # 如果是结束块, slot 的结束位置要加上本块内的 token 数, 等于 seq.block_table[i] 块号乘以每块覆盖的 token 数再加上 end 减去前面块数乘以每块覆盖的 token 数(即本块内的 token 数)
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    """
    decode 阶段每条 seq 只输入最后一个 token,位置是 len(seq) - 1, slot_mapping 是最后一个 token 在 KV Cache 中的槽位, 
    context_lens 是 seq 的长度(因为 decode 阶段 KV Cache 中的 token 数 = seq 的长度), block_tables 是每条 seq 的 KV 块号列表(如果有的话)。
    decode 阶段不区分 prefill 和 decode, 因为它们的输入格式完全一样。      
    """
    def prepare_decode(self, seqs: list[Sequence]): 
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) # 最后一个 token 在 KV Cache 中的槽位 = seq 的最后一个 KV 块号乘以每块覆盖的 token 数再加上最后一个 KV 块内的 token 数减去 1(因为槽位从 0 开始编号)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 只有 rank 0 的 sampler 会用到 temperatures, 因为只有它会做采样决策并生成 token_id, 其他 rank 只负责计算 logits 并等待同步采样结果
        logits = self.run_model(input_ids, positions, is_prefill) # logits 的形状是 [batch_size, vocab_size], 每行是对应输入 token 的下一个 token 的 logits 分布
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self): # 预先捕获不同 batch size 的 CUDA Graph,以便在 decode 阶段重放。prefill 阶段不使用 CUDA Graph,因为 prefill 的输入格式和 decode 不一样,而且 prefill 的激活更大,重放开销可能超过加速收益。
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
