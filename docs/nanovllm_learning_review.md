# nano-vLLM 项目复盘学习笔记

本文不是面试背诵稿，而是用来重新掌握项目机制的学习文档。目标是把一次 `prompt -> reply` 的完整链路、late merge、chunked prefill、CUDA Graph、benchmark 参数和 nano-vLLM/vLLM 的边界都串起来。

文中的代码依据来自当前仓库：

- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`
- `nanovllm/models/qwen3.py`
- `bench.py`
- `scripts/run_full_bench_matrix.sh`
- `logs/full_bench_20260619_235153/env.log`

外部模型配置参考 Hugging Face `Qwen/Qwen3-0.6B` 的 `config.json`。

## 0. 先建立全局图景

这个项目可以先理解成一个小型 vLLM-style 离线推理引擎。它做的事不是训练模型，而是把多个请求组织成一轮轮 GPU forward，让 GPU 尽量忙，同时用 KV Cache 避免重复计算历史 token。

最重要的四个对象：

| 对象 | 文件 | 作用 |
| --- | --- | --- |
| `LLMEngine` | `nanovllm/engine/llm_engine.py` | 对外接收请求，驱动 step 循环，统计 metrics |
| `Scheduler` | `nanovllm/engine/scheduler.py` | 管理 waiting/running 队列，决定本轮跑 prefill 还是 decode |
| `BlockManager` | `nanovllm/engine/block_manager.py` | 管理 physical KV blocks、prefix cache、ref_count、late merge |
| `ModelRunner` | `nanovllm/engine/model_runner.py` | 准备 GPU 输入，执行模型 forward、CUDA Graph replay/eager fallback、采样 |

一次完整生成可以先背成这个链路：

```text
用户调用 LLM.generate()
→ LLMEngine.add_request()
→ tokenizer 把 prompt 转成 token ids
→ 创建 Sequence
→ Sequence 进入 Scheduler.waiting
→ LLMEngine.step() 循环
→ Scheduler.schedule()
→ 选择 prefill batch 或 decode batch
→ ModelRunner.prepare_prefill() / prepare_decode()
→ 构造 input_ids / positions / slot_mapping / block_tables / context
→ ModelRunner.run_model()
→ Qwen3ForCausalLM.forward()
→ Attention.forward()
→ store_kvcache() 写 KV
→ flash_attn_varlen_func 或 flash_attn_with_kvcache 做 attention
→ lm_head 得到 logits
→ Sampler 采样下一个 token
→ Scheduler.postprocess()
→ hash_blocks() / late_merge / append_token / finish / deallocate
→ 如果请求没结束，下一轮继续 decode
```

这里最容易混淆的是 prefill 和 decode：

- Prefill：处理 prompt 中还没有进 KV Cache 的一段 token。一次 prefill 可以处理很多 token。
- Decode：每个请求每轮只生成 1 个新 token，但 batch 中可以有很多请求同时 decode。

## 1. Late merge 到底发生在什么环节

### 1.1 它不是 prefill 之前的优化

当前 late merge 发生在 GPU 已经完成 forward 和 KV 写入之后。入口是：

```text
LLMEngine.step()
→ ModelRunner.run()
→ GPU forward + KV 写入 + 采样
→ Scheduler.postprocess()
→ BlockManager.hash_blocks()
→ BlockManager._late_merge_block()
```

关键代码：

- `Scheduler.postprocess()`：对本轮跑过的每个 seq 调 `self.block_manager.hash_blocks(seq)`。
- `BlockManager.hash_blocks()`：只对本轮新完成的 full block 建 hash。
- `BlockManager._late_merge_block()`：如果 hash 命中并且 token_ids 完全一致，就做 block table 重定向和 ref_count 调整。

所以它不是“让 shared prefix 不计算”。它主要是“已经写出来了，但是发现 physical KV block 内容重复，于是把重复 physical block 合并掉”。

### 1.2 Allocate-time prefix cache 和 late merge 的区别

`BlockManager.can_allocate()` 是请求被调度 prefill 前的 prefix cache 查询。它做的是：

```text
准备给一个 seq 分配 KV block
→ 从第 0 个 full block 开始算链式 hash
→ 查 hash_to_block_id
→ 如果 hash 命中，并且 token_ids 完全一致
→ 说明这个 prefix block 之前已经完成并注册过
→ 当前 seq 可以直接复用这个 physical block
→ 少分配一些新 block
```

但是 allocate-time 查询有一个前提：目标 block 必须已经完成并注册到 `hash_to_block_id`。

如果多个请求共享 prefix，并且这些请求在同一轮或相近轮次进入系统，可能发生：

```text
请求 A、B、C 都有同一个 shared prefix
→ A/B/C 被 scheduler 同一轮或相近轮次选中做 prefill
→ 它们查询 prefix cache 时，目标 full block 还没有被任何请求写完并注册
→ can_allocate() 全都 miss
→ A/B/C 各自拿到不同 physical KV blocks
→ GPU 写完后，这些 physical blocks 的 token 内容其实完全相同
```

late merge 补的就是这个空窗。

### 1.3 Late merge 的运行逻辑

用你给的风格展开：

```text
请求加入 Waiting Queue
→ Scheduler 在 Token Budget 内选择多个 Prefill 请求
→ Allocate-time Prefix Cache 查询
→ 由于目标 Full Block 尚未完成/注册，多个请求同时 Miss
→ 为每个请求分配不同 Physical KV Blocks
→ ModelRunner.prepare_prefill() 构造 slot_mapping
→ GPU 完成 Prefill
→ Attention.forward() 内 store_kvcache() 把 K/V 写入各自 physical blocks
→ Sampler 采样本轮输出 token
→ Scheduler.postprocess()
→ BlockManager.hash_blocks()
→ 只处理本轮新完成的 Full Blocks
→ 使用父 Block Hash + 当前 Block Token IDs 构造链式 Hash
→ 查询 hash_to_block_id
→ Hash 未命中：当前 block 注册为 canonical
→ Hash 命中：比较 canonical.token_ids 与当前 token_ids
→ token_ids 不同：说明 hash collision 或同 hash 异内容，不合并
→ token_ids 相同：确认重复
→ seq.block_table[block_index] 重定向到 Canonical Physical Block
→ Canonical ref_count += 1
→ Duplicate ref_count -= 1
→ Duplicate ref_count == 0 时回收到 free_block_ids
→ _assert_consistent() 检查 ref_count / used / free 集合一致
```

这里有几个关键点：

1. 只合并 full block，不合并 partial block。
   full block 写完后内容稳定；partial block 后续还会继续写，如果提前共享会破坏另一个请求的 KV。

2. hash 只是索引，不是正确性依据。
   正确性还依赖 `canonical.token_ids == token_ids` 的完整比较。

3. 合并的是 block table 指针，不复制 KV。
   每个 seq 的 `block_table` 类似页表。late merge 本质上把页表条目从 duplicate physical page 改指向 canonical physical page。

4. ref_count 是生命周期保护。
   多个 seq 共享同一个 physical block 后，只有最后一个引用释放时才能真正回收。

### 1.4 Prefill 是并发的吗

要区分两种并发：

- 不是 Python 多线程并发。scheduler 和 block manager 的元数据更新是串行的。
- 是 GPU batch 并发。scheduler 一轮可以选择多个 sequence，把多个请求的 token 拼成一个 batch 送进一次 GPU forward。

在代码里，prefill 并发由两个预算控制：

```text
max_num_seqs
max_num_batched_tokens
```

`max_num_seqs` 控制一轮最多多少条 sequence。

`max_num_batched_tokens` 控制一轮 prefill 最多处理多少 token。

fair prefill 还多一个：

```text
prefill_chunk_size
```

它控制单个 sequence 在一轮中最多拿多少 prefill token。

所以“现在 prefill 是并发的吗”的准确回答是：是 batch-level 并发。多个请求可以在同一个 prefill step 中一起跑 GPU forward。

### 1.5 如果 prefill 不是并发，late merge 是否没用

如果系统严格串行，且一个请求完整 prefill 完、注册 prefix cache 后，另一个请求才开始 allocate，那么 allocate-time prefix cache 已经能复用前一个请求的 full blocks。late merge 的价值会明显降低。

但即使严格串行，late merge 仍可能作为一种安全兜底或去重机制存在，只是收益不大。

它真正有价值的场景是：

```text
多个请求共享 prefix
→ 请求在相近调度轮次进入
→ allocate-time 还看不到已完成 canonical block
→ 多个 physical block 被重复分配和写入
→ late merge 在写完后发现重复并回收
```

### 1.6 Decode 是并发的吗

decode 也是 batch-level 并发。`Scheduler._schedule_decode()` 会从 `running` 队列中拿多个 seq，每个 seq 本轮调度 1 个 token：

```text
running 队列中有多个已完成 prefill 的请求
→ Scheduler._schedule_decode()
→ 每个 seq 设置 num_scheduled_tokens = 1
→ may_append() 必要时分配新 KV block
→ scheduled_seqs 中可以有多个 seq
→ ModelRunner.prepare_decode() 构造一个 decode batch
→ GPU 一次 forward 给每个 seq 生成一个 token
```

decode 的并发数量也受 `max_num_seqs` 限制。

### 1.7 Late merge 会有并发冲突吗

当前实现没有真正的多线程共享 block manager，所以不会有传统意义上的 race condition。

原因：

```text
Scheduler.schedule() 串行选 batch
→ ModelRunner.run() 执行 GPU forward
→ forward 返回后
→ Scheduler.postprocess() 串行更新 block manager 元数据
```

虽然 GPU 内部并行计算，但 block table、hash table、ref_count 的更新发生在 Python 侧，且是一轮 forward 结束后的串行逻辑。

如果未来改成多 worker 共享同一个 KV cache manager，late merge 就需要锁、原子 ref_count 或中心化 cache manager；但当前项目没有这个问题。

### 1.8 “只是去重 KV block，技术含量低吗”

从算法名字看，它像 dedup；从系统实现看，它涉及几个推理系统核心概念：

- Paged KV Cache：logical sequence block table 到 physical KV block 的映射。
- Prefix cache identity：如何定义两个 KV block “相同”。
- Full block vs partial block：哪些内容可共享，哪些内容仍可变。
- Ref count：共享 physical block 的生命周期。
- Scheduler 时序：allocate-time miss 与 post-write merge 的空窗。
- Correctness guard：hash collision 必须 token_ids 校验。

真正的技术点不是“hash 去重”四个字，而是把去重放在正确的生命周期阶段，并确保共享后不会破坏 decode 对历史 KV 的读取。

### 1.9 为什么减少 KV block 会让 tokens/s 增加

不能简单说“KV block 少，所以算得少”。late merge 不减少已经发生的 prefill attention 计算。它可能提升 throughput 的原因主要在 decode 和系统容量：

1. Decode 每步都读历史 KV。
   decode attention 需要访问当前 seq 的历史 K/V。共享 prefix 后，多个 seq 的 block table 指向同一批 physical prefix blocks，resident KV footprint 下降。

2. 显存压力下降。
   physical KV blocks 更少，free blocks 更多，preemption/OOM 风险更低，batch 更容易维持。

3. block table 和 cache residency 更友好。
   更少的重复 KV pages 意味着访问工作集变小，可能改善 L2/cache/TLB-like 行为。不过这一点需要 Nsight profiling 才能严格证明。

4. shared-prefix workload 正好命中机制。
   本轮实验里 shared_prefix/late_merge 的 peak used blocks 从约 200.6 降到 150.2，吞吐从约 3178.20 tok/s 升到 3359.72 tok/s。这个结论应该限定在该 workload、该模型、该硬件、该实验设置下。

## 2. Prompt 到回复：CPU 和 GPU 分工

### 2.1 CPU 负责什么

CPU 侧主要负责控制流、元数据、调度和少量张量准备：

```text
LLM.generate()
→ tokenizer.encode()
→ 创建 Sequence
→ 加入 Scheduler.waiting
→ Scheduler.schedule() 选择 prefill/decode
→ BlockManager.can_allocate() / allocate() / can_append()
→ 构造 input_ids / positions / block_tables / slot_mapping 的 Python 列表
→ 转成 torch tensor，并拷到 GPU
→ Scheduler.postprocess()
→ hash_blocks() / late_merge / append_token / deallocate
→ metrics 统计
```

注意：构造 tensor 的数据来自 CPU，但 `.cuda(non_blocking=True)` 之后模型计算发生在 GPU。

### 2.2 GPU 负责什么

GPU 侧主要负责模型数学计算和 KV 写入：

```text
Embedding lookup
→ QKV projection
→ Q/K RMSNorm
→ RoPE
→ store_kvcache Triton kernel 写 K/V
→ FlashAttention prefill 或 decode
→ O projection
→ MLP gate/up/down projection
→ final RMSNorm
→ lm_head
→ softmax / sampling
```

在 `Attention.forward()` 中：

- `store_kvcache()` 用 Triton kernel 把新 K/V 写入 paged KV cache。
- prefill 使用 `flash_attn_varlen_func()`。
- decode 使用 `flash_attn_with_kvcache()`。

### 2.3 Prefill 的 GPU 输入形态

prefill 的 `prepare_prefill()` 会把多个 seq 的待处理 token 展平成一维：

```text
seq A 本轮处理 tokens: A[start:end]
seq B 本轮处理 tokens: B[start:end]
seq C 本轮处理 tokens: C[start:end]
→ input_ids = A tokens + B tokens + C tokens
→ positions = 每个 token 的绝对位置
→ cu_seqlens_q = 每个 seq query 边界
→ cu_seqlens_k = 每个 seq key/value 边界
→ slot_mapping = 每个新 token 应写入哪个 physical KV slot
```

如果有 prefix cache，`cu_seqlens_k[-1] > cu_seqlens_q[-1]`，说明 K/V 长度大于本轮 query 长度，需要把历史 cache 也作为 attention key/value 的一部分，此时会准备 `block_tables`。

### 2.4 Decode 的 GPU 输入形态

decode 的 `prepare_decode()` 每个 seq 只取最后一个 token：

```text
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(当前新 token 写入的 physical KV slot)
block_tables = 每个 seq 的 KV block table
```

所以 decode batch 的 shape 更接近：

```text
batch_size = 本轮 decode 的 seq 数
每个 seq query_len = 1
key/value length = 该 seq 当前上下文长度
```

这也是 decode 很适合 CUDA Graph 的原因：每步 query_len 固定为 1，kernel launch overhead 在小 batch 下明显。

## 3. Chunked prefill scheduler 与 fair 策略

### 3.1 原始 FCFS prefill 逻辑

FCFS 版本核心逻辑：

```text
while waiting 非空 and scheduled_seqs 数量 < max_num_seqs:
    seq = waiting[0]
    remaining = max_num_batched_tokens - num_batched_tokens
    如果 seq 没有 block_table:
        can_allocate(seq)
        allocate(seq)
    num_tokens = seq 还没 prefill 的 token 数
    如果 remaining < num_tokens 且 scheduled_seqs 已经非空:
        break
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    如果本轮后 seq prefill 完成:
        waiting.popleft()
        running.append(seq)
    scheduled_seqs.append(seq)
```

关键点是这句：

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

含义：

- 如果当前 batch 已经有别的 seq 了，而下一个 seq 放不完整，就直接停。
- 只有 batch 里第一个 seq 可以在预算不足时被 chunk。

这会导致长 prompt 对后面的短 prompt 形成 head-of-line blocking。

### 3.2 Fair chunked prefill 逻辑

fair 模式核心逻辑：

```text
Scheduler.schedule()
→ prefill_policy == "fair"
→ _schedule_fair_prefill()
→ 记录本轮 waiting 初始长度 num_waiting
→ 最多扫描 waiting 一圈
→ 每次从 waiting 左侧 pop 一个 seq
→ 如果没分配 block，先 can_allocate + allocate
→ 计算该 seq 剩余 prefill tokens
→ chunk = min(num_tokens, prefill_chunk_size, remaining_budget)
→ 本轮调度该 chunk
→ 如果 seq prefill 完成：放入 running
→ 如果 seq 没完成：append 回 waiting 队尾
→ 如果本轮 fair prefill 没调度到任何 seq：进入 decode
```

展开成箭头：

```text
Waiting Queue 中有多个请求
→ Scheduler.schedule()
→ prefill_policy == fair
→ 进入 _schedule_fair_prefill()
→ 读取本轮 token budget: max_num_batched_tokens
→ 从队首取一个 seq
→ 查询/分配 KV blocks
→ 计算该 seq 未完成 prefill tokens
→ chunk = min(剩余 prefill tokens, prefill_chunk_size, 剩余 token budget)
→ seq.num_scheduled_tokens = chunk
→ 如果 prompt 完成，seq 进入 Running Queue
→ 如果 prompt 未完成，seq 回到 Waiting Queue 队尾
→ 继续扫描下一个 waiting seq
→ 直到 token budget 用完、seq 数达到上限、或扫描完本轮 waiting
→ 统一送入 ModelRunner 做 prefill
```

### 3.3 这是不是“只是队首换队尾”

表面上是未完成 seq 回队尾，但本质是 token budget 分配策略变化。

原始 FCFS：

```text
一个长请求可能拿走一大段 token budget
后面短请求如果放不完整就等下一轮
```

fair chunked prefill：

```text
每个请求单轮最多拿 prefill_chunk_size
长请求被拆成多轮
短/中请求有机会更早获得 prefill 机会
```

它优化的是调度公平性和 TTFT，不是 attention kernel 本身。当前实验也说明这一点：mixed_lengths 下 fair 的 prefill step mean 下降，但总吞吐没有显著提升，甚至略降。

### 3.4 Fair prefill 和 decode 的关系

当前 fair 策略仍然是 prefill-first：

```text
如果 fair prefill 能调度到任何 waiting seq
→ 本轮跑 prefill
否则
→ 跑 decode
```

所以这不是 decode-first，也不是 vLLM 生产级那种更复杂的 prefill/decode 混合策略。

它不会在同一个 step 同时跑 prefill 和 decode。一个 step 的 `is_prefill` 要么是 True，要么是 False。

## 4. CUDA Graph 逻辑

### 4.1 CUDA Graph 想解决什么

decode 阶段每轮只给每个 seq 生成 1 个 token。query_len 很小，很多 kernel 很短，CPU launch overhead 可能占比明显。

CUDA Graph 的思路：

```text
提前用固定 shape 捕获一段 GPU 执行图
→ 后续相同/兼容 shape 的 decode
→ 不再逐个 launch kernel
→ 直接 replay 已捕获的 graph
```

它不是改变数学计算，而是减少调度开销。

### 4.2 当前项目捕获了什么

当前只捕获 decode，不捕获 prefill。

原因：

- Prefill 的 token 数、seq 长度、cu_seqlens、slot_mapping 变化大。
- Decode 每个 seq query_len 固定为 1，更适合 graph capture。

`capture_cudagraph()` 做的事：

```text
max_bs = min(max_num_seqs, 512)
max_num_blocks = ceil(max_seq_len_to_capture / block_size)
准备固定大小 GPU buffer:
    input_ids[max_bs]
    positions[max_bs]
    slot_mapping[max_bs]
    context_lens[max_bs]
    block_tables[max_bs, max_num_blocks]
    outputs[max_bs, hidden_size]
graph_bs = [1, 2, 4, 8] + [16, 32, 48, ... max_bs]
对每个 bs:
    设置 decode context
    warmup 一次
    capture 一次 model forward
    保存 graph
```

注意：捕获的是 `self.model(input_ids, positions)`，不是 sampling。

### 4.3 Decode 时如何 replay

运行 decode 时：

```text
ModelRunner.prepare_decode()
→ input_ids/positions/slot_mapping/context_lens/block_tables 拷到 GPU
→ run_model(is_prefill=False)
→ 找到第一个 graph_bs >= 当前 bs 的 bucket
→ 把 runtime 数据拷进 graph_vars 固定 buffer
→ graph.replay()
→ graph_vars["outputs"][:bs] 是 hidden states
→ compute_logits()
→ sampler()
```

这说明 graph bucket 可以比真实 batch size 大。比如真实 bs=10，可以用 graph_bs=16；前 10 个位置填真实数据，其余位置清零或填 -1。

### 4.4 修改后的 fallback 机制

当前 `run_model()` 中有 runtime eligibility 检查：

```text
如果 enforce_eager=True:
    fallback_reason = "enforce_eager"
elif 找不到 graph_bs:
    fallback_reason = "batch_too_large"
elif runtime block_tables 宽度 > graph buffer block_tables 宽度:
    fallback_reason = "block_table_too_wide"
elif context_lens 最大值 > max_seq_len_to_capture:
    fallback_reason = "context_too_long"
else:
    replay CUDA Graph
```

触发 fallback 后：

```text
记录 graph_fallbacks 和 graph_fallback_reasons
→ 直接 eager forward
→ compute_logits
```

### 4.5 原始不 fallback 为什么看起来也没问题

因为本轮 benchmark 的配置没有触发不兼容情况：

- `max_seq_len_to_capture=4096`
- `max_model_len=4096`
- `block_size=256`
- `max_num_seqs=512`
- workload 没超过 capture 范围

所以 12 组实验里 graph fallback 都是 0。

但是“不触发”不等于“不需要”。fallback 的价值在于：

```text
当后续换成长上下文
→ context_lens 超过 capture 上限
或 block table 比 capture buffer 更宽
或 batch size 超过 bucket
→ 不应该错误 replay 一个 shape 不匹配的 graph
→ 应该退回 eager，保证正确性和可观测性
```

所以 safe CUDA Graph 是稳定性优化，不是本轮性能收益的主要来源。

## 5. nano-vLLM 保留了什么，删减了什么

### 5.1 保留的核心能力

当前 nano-vLLM 保留了一个推理引擎最核心的路径：

- 离线 batch generate API。
- Hugging Face tokenizer 和 config 加载。
- Qwen3 CausalLM 模型结构。
- Tensor Parallel 线性层切分。
- Paged KV Cache。
- Prefix cache。
- FlashAttention prefill/decode。
- Triton KV 写入 kernel。
- CUDA Graph decode replay。
- torch.compile sampler。
- 简单 benchmark 和 metrics。

### 5.2 相比 vLLM 大量删减的部分

删减或未实现的部分包括：

- OpenAI-compatible server。
- 在线服务请求队列、流式输出、取消请求。
- 复杂 scheduler policy、priority、SLO、多租户。
- PD 分离。
- speculative decoding。
- LoRA adapter 动态加载和请求级 adapter batching。
- KV swap/offload。
- 多模型、多模态、工具调用等广泛生态。
- 复杂 prefix cache eviction、salt、多租户隔离。
- 量化、FP8 KV、更多 attention backend。
- 生产级 observability 和 fault tolerance。

所以学习时可以把它看成“把 vLLM 最核心的执行链路压缩到较少代码里”，适合理解原理，但不能等同于完整 vLLM。

### 5.3 FlashAttention 在哪里

在 `nanovllm/layers/attention.py`：

```text
prefill:
    flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, block_table=...)

decode:
    flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens, block_table=...)
```

prefill 用 varlen，是因为一个 batch 内多个 seq 的 prompt 长度不同。

decode 用 kvcache，是因为 query 是当前 token，key/value 来自 paged KV cache。

### 5.4 CUDA Graph 在哪里

在 `nanovllm/engine/model_runner.py`：

- `capture_cudagraph()`：初始化时捕获 decode graph。
- `run_model()`：decode 时决定 replay 还是 fallback eager。

## 6. Prefill-first、PD 分离、投机解码、SWAP、LoRA

### 6.1 当前 prefill-first 不是 PD 分离

当前 prefill-first 的意思只是调度优先级：

```text
如果 waiting 中有可以调度的 prefill
→ 本轮优先跑 prefill
否则
→ 跑 running 中的 decode
```

prefill 和 decode 都在同一个 engine、同一个 scheduler、同一个 model runner、同一套 KV cache 中。

PD 分离是另一回事。

### 6.2 PD 分离是什么

PD 分离是 Prefill/Decode disaggregation：

```text
Prefill worker/GPU 负责处理长 prompt
→ 生成 prompt 对应 KV Cache
→ KV Cache 通过网络/共享内存/显存拷贝交给 Decode worker/GPU
→ Decode worker/GPU 专注小步 decode
```

为什么要分离：

- Prefill 是大矩阵、大 token 数，更偏 compute-heavy。
- Decode 是小 query、长 KV 读取，更偏 memory/latency-sensitive。
- 两者资源特征不同，混在一起会互相干扰。

如果要加到本项目，位置大概是：

```text
LLMEngine
→ 拆出 PrefillEngine 和 DecodeEngine
→ Scheduler 不再只返回本地 batch，而是决定送到哪个 worker
→ ModelRunner 需要支持导出/导入 KV blocks
→ BlockManager 需要跨 worker 的 block identity 和生命周期
→ Attention decode 读取远端/本地迁移后的 KV
```

增加前：

```text
prefill 和 decode 共享同一 GPU/同一 KV block pool
```

增加后：

```text
prefill 产 KV
decode 消费 KV
中间多了 KV transfer、ownership、backpressure、cache consistency
```

### 6.3 投机解码是什么

投机解码 speculative decoding：

```text
小 draft model 先快速生成多个候选 token
→ 大 target model 一次性验证这些 token
→ 验证通过的多个 token 一次提交
→ 验证失败的位置回退并重新采样
```

目标是减少 target model decode step 次数。

如果要加到本项目，位置大概是：

```text
Scheduler._schedule_decode()
→ 不再固定每个 seq 调度 1 个 token
→ 为每个 seq 规划 draft length
→ 新增 DraftModelRunner
→ Target ModelRunner 批量 verify draft tokens
→ Scheduler.postprocess() 一次 append 0 到多个 token
→ KV block append/may_append/hash 逻辑需要支持多 token decode
```

增加前：

```text
每个 decode step 每个 seq 只生成 1 token
```

增加后：

```text
每个大模型 step 可能确认多个 token
```

### 6.4 SWAP / KV offload 是什么

SWAP/offload 是把不活跃 KV blocks 从 GPU 挪到 CPU pinned memory、NVMe 或远端 cache。

类似操作系统分页：

```text
GPU KV cache = 主存
CPU pinned memory = swap/page cache
NVMe/remote = 更慢的后备存储
```

如果要加到本项目：

```text
BlockManager
→ 区分 GPU blocks / CPU blocks / evicted blocks
→ 增加 block state: GPU_RESIDENT / CPU_RESIDENT / LOADING / EVICTED
→ Scheduler.schedule() 前做 prefetch plan
→ Attention.forward() 前确保本轮需要的 block 在 GPU
→ decode 后根据热度/引用做 demote
```

增加前：

```text
所有可用 KV 都必须在 GPU physical blocks 中
```

增加后：

```text
GPU 只放热 KV，冷 KV 可换出；调度器必须考虑换入延迟
```

### 6.5 LoRA 是什么

LoRA 是请求级轻量 adapter。基础权重不变，每个请求可以选择不同 LoRA adapter。

如果要加到本项目：

```text
Sequence 增加 lora_id
→ Scheduler batching 时按兼容 adapter 分组或混合处理
→ Linear 层支持 base output + LoRA delta
→ ModelRunner 加载/切换 LoRA weights
→ prefix cache hash 需要包含 lora_id 或影响模型输出的 extra hash
```

重点：如果 LoRA 改变模型输出，那么同样 token_ids 的 KV 不一定相同。prefix cache/late merge 的 identity 必须包含 LoRA 信息，否则会错误共享 KV。

## 7. Batch size 怎么理解

项目里没有一个单独叫 `batch_size` 的固定参数。每一轮实际 batch size 是 scheduler 动态选出来的。

### 7.1 Prefill batch size

prefill 更应该看：

```text
本轮 scheduled seq 数
本轮 scheduled prefill token 总数
```

约束是：

```text
len(scheduled_seqs) <= max_num_seqs
sum(seq.num_scheduled_tokens) <= max_num_batched_tokens
```

例如：

```text
max_num_batched_tokens = 16384
prefill_chunk_size = 1024
max_num_seqs = 512
```

fair prefill 下，一轮理论上可以调度多个 seq，每个 seq 最多 1024 prefill tokens，直到总 token 数接近 16384 或 seq 数达到上限。

### 7.2 Decode batch size

decode 的 batch size 更接近日常说的 batch size：

```text
decode batch size = 本轮 scheduled_seqs 的数量
```

因为每个 seq 只 decode 1 个 token。

约束是：

```text
decode batch size <= max_num_seqs
```

但也会受 KV block 可用性影响。如果追加 token 需要新 block 且没有 free block，scheduler 会 preempt 一些 seq。

### 7.3 “batchsize 是 schedule1 并发数量吗”

如果你说的 `schedule1` 是一次 `LLMEngine.step()`，那可以这样理解：

- Prefill 的“并发数量”是这一 step 中 scheduled seq 数，但真正决定 GPU 工作量的是 scheduled token 总数。
- Decode 的“并发数量”基本就是这一 step 中 scheduled seq 数，因为每个 seq 只生成 1 token。

项目默认：

```text
max_num_seqs = 512
max_num_batched_tokens = 16384
```

实验脚本中请求总数：

```text
num_seqs = 128
```

所以 benchmark 总共发 128 个请求，但每一轮 schedule 能拿多少请求取决于 waiting/running 队列状态和 token budget。

## 8. Qwen3-0.6B 在本项目中的配置

### 8.1 模型结构配置

本项目用 `AutoConfig.from_pretrained(config.model)` 读取 Hugging Face config，然后实例化 `Qwen3ForCausalLM(hf_config)`。

`Qwen/Qwen3-0.6B` 的关键 config：

| 参数 | 值 |
| --- | --- |
| `architectures` | `Qwen3ForCausalLM` |
| `model_type` | `qwen3` |
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 40960 |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1000000 |
| `attention_bias` | false |
| `tie_word_embeddings` | true |
| `torch_dtype` | bfloat16 |

### 8.2 项目运行配置

虽然模型 config 支持更长位置，但 benchmark 把项目运行上下文限制为：

```text
max_model_len = 4096
max_seq_len_to_capture = 4096
```

`Config.__post_init__()` 会做：

```text
max_model_len = min(用户设置, hf_config.max_position_embeddings)
max_seq_len_to_capture = min(用户设置, max_model_len)
```

所以本轮实验中上下文上限按 4096 走。

### 8.3 KV cache block 大小

项目配置：

```text
kvcache_block_size = 256
```

每个 physical KV block 存 256 个 token 的 K/V。

单个 block 的字节数计算在 `ModelRunner.allocate_kv_cache()`：

```text
block_bytes =
    2
    * num_hidden_layers
    * block_size
    * num_kv_heads_per_rank
    * head_dim
    * dtype.itemsize
```

其中 `2` 是 K 和 V 两份 cache。

对 Qwen3-0.6B、TP=1、bfloat16 来说：

```text
2 * 28 * 256 * 8 * 128 * 2 bytes
= 29,360,128 bytes
约 28 MiB / block
```

注意这里的 block 是“所有层的一段 token KV”，不是单层 block。

## 9. Late merge 和 chunked prefill 增加前后的执行逻辑

### 9.1 Late merge 增加前

```text
请求加入 waiting
→ Scheduler 选择 prefill
→ BlockManager.can_allocate()
→ 已经注册过的 full prefix block 可以复用
→ 未注册的 block miss
→ BlockManager.allocate() 分配新的 physical blocks
→ GPU prefill 写 KV
→ Scheduler.postprocess()
→ BlockManager.hash_blocks()
→ 计算 full block hash
→ block.update(hash, token_ids)
→ hash_to_block_id[hash] = 当前 block_id
→ 不检查当前 block 是否和已有 block 重复
→ 不释放 duplicate block
```

### 9.2 Late merge 增加后

```text
请求加入 waiting
→ Scheduler 选择 prefill
→ BlockManager.can_allocate()
→ 未完成/未注册 shared prefix 仍可能 miss
→ BlockManager.allocate() 分配新的 physical blocks
→ GPU prefill 写 KV
→ Scheduler.postprocess()
→ BlockManager.hash_blocks()
→ 计算 full block 链式 hash
→ block.update(hash, token_ids)
→ enable_prefix_late_merge=True
→ _late_merge_block(seq, block_index, h, token_ids)
→ hash_to_block_id 未命中：当前 block 成为 canonical
→ hash_to_block_id 命中：取 canonical block
→ 比较 canonical.token_ids 和当前 token_ids
→ 相同则 seq.block_table[block_index] = canonical_id
→ canonical.ref_count += 1
→ duplicate.ref_count -= 1
→ duplicate.ref_count == 0 时释放
```

### 9.3 Chunked prefill 增加前：FCFS

```text
waiting: [长请求 A, 短请求 B, 短请求 C, ...]
→ schedule()
→ _schedule_fcfs()
→ 看队首 A
→ 如果 budget 够 A，就调度完整 A
→ 如果 budget 不够 A 且当前 batch 为空，就给 A 一个 chunk
→ 如果 batch 已有其他 seq 且下一个 seq 放不完整，直接 break
→ B/C 可能因为 A 或 budget 碎片等待
```

### 9.4 Chunked prefill 增加后：fair

```text
waiting: [长请求 A, 短请求 B, 短请求 C, ...]
→ schedule()
→ prefill_policy == fair
→ _schedule_fair_prefill()
→ A 最多拿 prefill_chunk_size
→ A 未完成则回队尾
→ B 最多拿 prefill_chunk_size
→ B 如果完成则进入 running
→ C 同理
→ 一轮内让更多请求获得 prefill 机会
```

### 9.5 两者叠加时的关系

fair chunked prefill 改变“哪些请求何时写出 KV block”。

late merge 改变“写出的重复 KV block 如何合并”。

它们基本正交：

```text
fair prefill:
    作用在 Scheduler 选择 token 的阶段

late merge:
    作用在 GPU 写完 KV 后的 BlockManager 元数据阶段
```

叠加后 `optimized` variant 做的是：

```text
prefill_policy = fair
enable_prefix_late_merge = True
```

## 10. Benchmark workload 与参数

### 10.1 三类 workload

#### random

生成方式：

```text
每个 prompt 长度在 [min_input_len, max_input_len] 随机
每个 token id 在 [0, vocab_size - 1] 随机
每个输出长度在 [min_output_len, max_output_len] 随机
```

用途：

```text
低共享场景
检查 late merge / fair prefill 是否引入明显 overhead
```

#### mixed_lengths

生成方式：

```text
60% short: 64 到 min(512, max_model_len - 1)
30% medium: 768 到 min(1536, max_model_len - 1)
10% long: 2048 到 min(max_input_len, max_model_len - 1)
```

用途：

```text
制造长短 prompt 混合
观察 FCFS 下长请求对短请求的阻塞
验证 fair chunked prefill 是否改变 prefill step/TTFT
```

注意：当前脚本默认 `max_input_len=1024`，所以 long bucket 的 high 会变成 1024，实际 long 是 2048 到 1024 这个区间不合理。代码里有 `low = min(low, high)`，所以 long bucket 会退化成固定 1024 左右。这一点复盘时要记住：mixed_lengths 的设计意图是 2048+ long，但当前默认参数让 long 上限被 `max_input_len` 截断了。

#### shared_prefix

生成方式：

```text
shared_prefix_len = min(args.shared_prefix_len, args.max_model_len - 2)
suffix_max = max(1, min(args.max_input_len - prefix_len, args.max_model_len - prefix_len - 1))
每个请求 = shared_prefix + 随机 suffix
suffix_len 在 [1, suffix_max] 随机
```

实验默认：

```text
shared_prefix_len = 768
max_input_len = 1024
max_model_len = 4096
suffix_max = min(1024 - 768, 4096 - 768 - 1) = 256
suffix_len = 1 到 256
```

用途：

```text
高 prefix 共享场景
验证 prefix cache 和 late merge
```

### 10.2 四类实验 variant

| Variant | prefill policy | late merge | 含义 |
| --- | --- | --- | --- |
| `baseline` | `fcfs` | false | 原始调度 + 无 late merge |
| `fair` | `fair` | false | 只开 fair chunked prefill |
| `late_merge` | `fcfs` | true | 只开 late merge |
| `optimized` | `fair` | true | 两个优化都开 |

如果 `RUN_EAGER=1`，脚本还会加 `baseline_eager` 和 `optimized_eager`，用于禁用 CUDA Graph。

### 10.3 实验脚本默认参数

`scripts/run_full_bench_matrix.sh` 默认：

| 参数 | 值 |
| --- | --- |
| `REPEAT` | 5 |
| `NUM_SEQS` | 128 |
| `SEED` | 0 |
| `VOCAB_SIZE` | 10000 |
| `MIN_INPUT_LEN` | 100 |
| `MAX_INPUT_LEN` | 1024 |
| `MIN_OUTPUT_LEN` | 32 |
| `MAX_OUTPUT_LEN` | 128 |
| `SHARED_PREFIX_LEN` | 768 |
| `TEMPERATURE` | 0.6 |
| `MAX_MODEL_LEN` | 4096 |
| `MAX_NUM_BATCHED_TOKENS` | 16384 |
| `MAX_NUM_SEQS` | 512 |
| `GPU_MEMORY_UTILIZATION` | 0.9 |
| `KVCACHE_BLOCK_SIZE` | 256 |
| `PREFILL_CHUNK_SIZE` | 1024 |
| `MAX_SEQ_LEN_TO_CAPTURE` | 4096 |
| `RUN_EAGER` | 0 |

### 10.4 128 个请求够不够

够做原型验证，不够做强统计结论。

够的原因：

```text
128 seq × repeat=5
→ 可以初步观察平均吞吐、KV block 水位、prefill/decode step latency 的方向
→ shared_prefix 这类机制性强的 workload 能看出明显差异
```

不够的原因：

```text
P95/P99 对样本量敏感
真实服务流量不是一次性加入 128 个离线请求
不同输出长度、到达过程、真实 tokenizer 分布都会影响结论
缺少 Nsight 级 profiling，无法严格证明 kernel 级因果
```

所以学习时要把 128 请求理解成“可复现实验最小闭环”，不是“生产级性能证明”。

## 11. 硬件配置与 Blackwell

实验日志中硬件：

```text
GPU: NVIDIA RTX PRO 4000 Blackwell SFF Edition
显存: 24467 MiB
GPU 数量: 1
Torch: 2.7.0+cu128
torch.version.cuda: 12.8
NVIDIA Driver: 595.71.05
nvidia-smi CUDA Version: 13.2
```

如何理解：

- 这是单卡工作站/专业卡环境，不是多卡 H100/B200 服务端集群。
- Blackwell 是 NVIDIA 新一代 GPU 架构名。
- `RTX PRO 4000 Blackwell SFF Edition` 是 Blackwell 架构下的专业显卡型号，SFF 表示 Small Form Factor。
- 对这个项目来说，重要点是：单卡、约 24GB 显存、足够跑 Qwen3-0.6B 和 KV cache 实验。

复盘时不要把它说成 B200，也不要把单卡实验泛化成多机服务能力。

## 12. 参数汇总：你问的那组具体是多少

按本轮完整实验脚本和日志：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| Shared Prefix 长度 | 768 | `SHARED_PREFIX_LEN=768` |
| Suffix 长度 | 1 到 256 | shared_prefix workload 下由 `max_input_len=1024` 和 prefix 768 推出 |
| Output Length | 32 到 128 | 每个请求随机 `max_tokens` |
| `max_num_seqs` | 512 | 单步最多调度 sequence 数 |
| `max_num_batched_tokens` | 16384 | 单步 prefill token budget |
| Block Size | 256 | 每个 KV block 覆盖 256 token |
| Sampling temperature | 0.6 | benchmark 参数 |
| Sampling `ignore_eos` | true | benchmark 中固定设置 |
| Sampling `max_tokens` | 32 到 128 | 每个请求随机 |
| `prefill_chunk_size` | 1024 | fair prefill 单 seq 单轮最多 prefill token |
| `num_seqs` | 128 | benchmark 总请求数 |
| `repeat` | 5 | 每个 workload/variant 重复次数 |

## 13. 把全部问题串成一张学习地图

可以把本项目按三条主线记：

### 13.1 调度主线

```text
请求进入 waiting
→ schedule()
→ 如果 fair 且 waiting 能调度，跑 fair prefill
→ 如果 fcfs，先尝试 FCFS prefill
→ 如果没有 prefill 可跑，跑 decode
→ prefill 完成的 seq 进入 running
→ running seq 每轮 decode 1 token
→ decode 完成 max_tokens/eos 后释放 block
```

调度相关关键词：

```text
waiting queue
running queue
max_num_seqs
max_num_batched_tokens
prefill_chunk_size
prefill-first
batch-level concurrency
```

### 13.2 KV Cache 主线

```text
Sequence 只有 logical token_ids
→ block_table 把 logical block 映射到 physical block id
→ physical block 存所有层的 K/V
→ can_allocate() 做 allocate-time prefix cache
→ allocate() 分配或复用 blocks
→ attention 写 KV 到 slot_mapping 指定位置
→ hash_blocks() 注册 full blocks
→ late_merge 可重定向重复 blocks
→ deallocate() 按 ref_count 回收
```

KV 相关关键词：

```text
physical block
logical block
block table
slot mapping
full block
partial block
hash_to_block_id
ref_count
prefix cache
late merge
```

### 13.3 GPU 执行主线

```text
prepare_prefill / prepare_decode
→ set_context()
→ Qwen3 forward
→ Attention.forward()
→ store_kvcache()
→ flash attention
→ lm_head
→ sampler
→ postprocess
```

GPU 相关关键词：

```text
FlashAttention varlen
flash_attn_with_kvcache
Triton store_kvcache
CUDA Graph decode replay
eager fallback
block_tables
context_lens
cu_seqlens
```

## 14. 容易误解的点

### 14.1 late merge 不减少 prefill 计算

它发生在 KV 写入之后。它减少的是重复 resident KV blocks，主要影响后续 decode、容量和显存水位。

### 14.2 fair prefill 不等于吞吐一定提升

它改善的是 token budget 分配和等待公平性。更碎的 prefill batch 也可能带来 overhead。

### 14.3 CUDA Graph fallback 为 0 不代表 fallback 没价值

只说明当前 workload 没触发。fallback 是为了未来 shape 超界时保持正确性。

### 14.4 batch size 不是固定配置

`max_num_seqs` 是上限；实际每步 batch 由 scheduler 动态决定。

### 14.5 Prefix cache 的 identity 不能只看 token

当前项目只支持单模型、无 LoRA、无多租户，所以 token hash 足够用于原型。若加入 LoRA、多模型、多租户、cache salt，identity 必须加入这些 extra factors。

### 14.6 当前是 prefill-first，不是 PD 分离

prefill-first 是一个 scheduler policy；PD 分离是架构拆分，涉及 worker、KV transfer 和资源池。

## 15. 推荐复盘顺序

如果要重新掌握这部分知识，建议按这个顺序读代码：

1. `nanovllm/engine/sequence.py`
   先理解 `Sequence`、`num_tokens`、`num_cached_tokens`、`num_scheduled_tokens`、`block_table`。

2. `nanovllm/engine/scheduler.py`
   理解 waiting/running、FCFS、fair、decode、postprocess。

3. `nanovllm/engine/block_manager.py`
   理解 physical block、prefix cache、hash、ref_count、late merge。

4. `nanovllm/engine/model_runner.py`
   理解 prepare_prefill/prepare_decode、KV cache allocation、CUDA Graph。

5. `nanovllm/layers/attention.py`
   理解 KV 写入和 prefill/decode attention 分支。

6. `bench.py`
   理解 workload、variant、metrics。

7. `logs/full_bench_20260619_235153/*.jsonl`
   对照机制看指标变化。

最后可以用一句话总结本项目：

```text
这是一个以 Qwen3-0.6B 为模型、以 paged KV cache 为核心的 nano-vLLM 推理原型；
它通过 scheduler 控制 prefill/decode batch，通过 block table 管理 KV cache，
通过 FlashAttention 执行 attention，通过 CUDA Graph 优化 decode launch，
并新增 fair chunked prefill 与 full-block late merge 来分别改善调度公平性和 shared-prefix KV footprint。
```
