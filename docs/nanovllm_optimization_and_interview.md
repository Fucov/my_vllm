# nano-vLLM 推理优化代码审计与面试准备文档

## 1. 文档摘要

本文基于当前仓库真实代码、Git 历史和可见文件，对 nano-vLLM 推理优化项目做一次面向“大模型推理框架开发岗位”的代码审计。审计范围覆盖从 baseline commit `ffa7349` 到当前 commit `07962aa` 的全部有效改动。

结论先行：

| 项目 | 审计结论 |
| --- | --- |
| 当前分支 | `feature` |
| baseline commit | `ffa7349 init repo` |
| current commit | `07962aa trick1_2...` |
| 实质性优化数量 | 4 项：benchmark/metrics、token-budget fair prefill、shared-prefix full-block KV late-merge、CUDA Graph eligibility/fallback |
| 工程性补丁 | 1 项：matrix benchmark 使用子进程隔离，规避重复创建/销毁 LLM 后 CUDA allocator / NCCL 状态污染 |
| 当前仓库可验证性能数据 | 当前工作区没有 `logs/`、`*.jsonl`、`*.csv` 原始实验文件，无法从仓库本身复核性能数字 |
| 可写入简历的稳妥说法 | “实现原型并建立 benchmark 矩阵；在历史 shared-prefix 实验中观察到 KV block 水位下降和吞吐改善，但当前仓库缺少原始日志，需谨慎表述为初步结果” |

重要边界：

1. `Prefix Late Merge` 是 full block 级的 KV Cache 去重原型，发生在 KV 已经写入之后。它主要减少后续 resident physical KV blocks，不减少已经发生的 prefill 计算。
2. `Fair Chunked Prefill` 是 token budget 内的多请求 prefill 扫描和切分原型，不是生产级 QoS scheduler。当前代码没有实现 request class、短请求优先、decode-first 或自适应 chunk size。
3. `Safe CUDA Graph` 增加 runtime eligibility 检查和 fallback 统计，能避免部分 shape 不匹配路径直接 replay graph，但当前仓库没有失败注入实验或 fallback 非零日志。
4. 当前 benchmark 指标来自 Python wall-clock 计时和 engine 内部统计，缺少 Nsight Systems / Nsight Compute 证据，不能声称已经证明 kernel 级瓶颈。

## 2. 仓库与基线信息

### 2.1 审计命令与结果

本次审计先检查了：

```bash
git status --short --branch
git remote -v
git branch -a
git log --oneline --decorate --graph --all --max-count=80
git tag
```

关键信息：

| 项 | 结果 |
| --- | --- |
| 当前分支 | `feature`，跟踪 `origin/feature` |
| remote | `origin https://github.com/Fucov/my_vllm.git` |
| upstream remote | 当前仓库未发现 `upstream` remote |
| tag | 当前未发现 tag |
| main 分支 | `17e632c`、`a7b74fe`、`f37daea` 是从 `ffa7349` 之后分出的注释类提交 |
| feature 分支 | `9f06488`、`ea73880`、`e7a3d91`、`07962aa` 是本轮优化相关提交 |

由于不存在 upstream remote，也不存在 tag，且 `ffa7349 init repo` 是 feature 与 main 的共同起点，本文选择：

```text
baseline commit: ffa7349 init repo
current commit : 07962aa trick1_2...
diff range     : ffa7349..07962aa
```

基线存在的轻微歧义：

| 候选基线 | 是否采用 | 理由 |
| --- | --- | --- |
| upstream/tag | 否 | 当前仓库不存在 upstream remote 或 tag |
| `origin/main` / `main` | 否 | main 在 `ffa7349` 后包含注释类提交，不是优化分支的直接代码基线 |
| `ffa7349 init repo` | 是 | 当前优化分支和 main 的共同导入提交，最接近“开始优化前”的 nano-VLLM 代码 |

### 2.2 Git 差异概览

`git diff --stat ffa7349..HEAD` 显示：

```text
.gitignore                                         |   3 +-
bench.py                                           | 448 ++++++++++++++++++++-
docs/OSforInfra.md                                 | 294 ++++++++++++++
docs/OtherTrick.md                                 | 307 ++++++++++++++
docs/superpowers/plans/2026-06-11-nanovllm-inference-optimization.md | 76 ++++
docs/superpowers/specs/2026-06-11-nanovllm-inference-optimization-design.md | 301 ++++++++++++++
nanovllm/config.py                                 | 10 +
nanovllm/engine/block_manager.py                   | 63 ++-
nanovllm/engine/llm_engine.py                      | 47 +++
nanovllm/engine/model_runner.py                    | 63 ++-
nanovllm/engine/scheduler.py                       | 80 +++-
tests/test_bench_cli.py                            | 49 +++
12 files changed, 1699 insertions(+), 42 deletions(-)
```

`git diff --name-status ffa7349..HEAD` 显示实质变更文件：

| 状态 | 文件 |
| --- | --- |
| M | `.gitignore` |
| M | `bench.py` |
| A | `docs/OSforInfra.md` |
| A | `docs/OtherTrick.md` |
| A | `docs/superpowers/plans/2026-06-11-nanovllm-inference-optimization.md` |
| A | `docs/superpowers/specs/2026-06-11-nanovllm-inference-optimization-design.md` |
| M | `nanovllm/config.py` |
| M | `nanovllm/engine/block_manager.py` |
| M | `nanovllm/engine/llm_engine.py` |
| M | `nanovllm/engine/model_runner.py` |
| M | `nanovllm/engine/scheduler.py` |
| A | `tests/test_bench_cli.py` |

### 2.3 Commit 与功能对应

| Commit | 类型 | 主要内容 |
| --- | --- | --- |
| `9f06488` | 设计文档 | 新增 nano-VLLM inference optimization design spec |
| `ea73880` | 核心实现 | benchmark/metrics、Fair Chunked Prefill、Prefix Late Merge、Safe CUDA Graph、配置项和 metrics 聚合 |
| `e7a3d91` | benchmark 增强 | 增加 repeat、JSONL 输出、LLMEngine 退出逻辑、bench CLI 单测 |
| `07962aa` | benchmark 稳定性修复 | matrix/repeat 改为子进程隔离，避免同一 Python 进程反复创建/销毁 LLM 后显存估算异常 |

## 3. 原始 nano-VLLM 执行链路

当前 `nanovllm/llm.py` 中 `LLM` 只是继承 `LLMEngine`：

```text
nanovllm/llm.py
L4: class LLM(LLMEngine): pass
```

请求执行链路由 `LLMEngine` 驱动：

```text
LLM.generate()
→ LLMEngine.add_request()
→ Scheduler.add()
→ LLMEngine.step()
→ Scheduler.schedule()
→ ModelRunner.run()
→ ModelRunner.prepare_prefill() / prepare_decode()
→ ModelRunner.run_model()
→ Attention.forward()
→ store_kvcache() / flash_attn_varlen_func() / flash_attn_with_kvcache()
→ Scheduler.postprocess()
→ BlockManager.hash_blocks() / may_append() / deallocate()
```

核心代码位置：

| 步骤 | 文件 | 当前行号 | 作用 |
| --- | --- | ---: | --- |
| 请求进入 | `nanovllm/engine/llm_engine.py` | L49-L55 | token 化或接收 token ids，创建 `Sequence`，记录 request start time |
| 单步执行 | `nanovllm/engine/llm_engine.py` | L86-L102 | schedule、调用 model runner、统计 prefill/decode/TTFT |
| 调度 | `nanovllm/engine/scheduler.py` | L42-L51 | 根据 `prefill_policy` 选择 fair prefill、FCFS prefill 或 decode |
| prefill 张量准备 | `nanovllm/engine/model_runner.py` | L165-L206 | 构造 input ids、positions、cu_seqlens、slot_mapping、block_tables |
| decode 张量准备 | `nanovllm/engine/model_runner.py` | L208-L224 | 构造单 token decode 的 block table 和 context length |
| 模型执行 | `nanovllm/engine/model_runner.py` | L231-L265 | prefill eager，decode 视 CUDA Graph eligibility 选择 replay 或 eager |
| KV 写入和 attention | `nanovllm/layers/attention.py` | L59-L75 | `store_kvcache` 写 physical KV cache，prefill/decode 使用 flash attention |
| 输出处理 | `nanovllm/engine/scheduler.py` | L159-L170 | hash 已完成 block，更新 cached tokens，追加采样 token，完成则释放 block |

原始版本主要特征：

1. `Scheduler` 使用等待队列 `waiting` 和运行队列 `running`，prefill 与 decode 基于 `max_num_batched_tokens` 和 `max_num_seqs` 控制 batch。
2. `BlockManager` 使用 fixed-size physical blocks 和 per-sequence `block_table` 管理 KV Cache。
3. `Attention.forward()` 中，KV 写入由 Triton `store_kvcache_kernel` 完成，prefill 使用 `flash_attn_varlen_func`，decode 使用 `flash_attn_with_kvcache`。
4. Prefix cache 已存在 allocate-time 复用路径，但只对调度前能识别的 cached block 生效；本轮新增的是写入后 full block late merge。

## 4. 优化总览表

| 优化编号 | 优化名称 | 修改文件 | 核心函数/类 | 修改类型 | 解决的问题 | 预期收益 | 是否有实验验证 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Benchmark & Metrics | `bench.py`、`llm_engine.py`、`scheduler.py`、`block_manager.py`、`model_runner.py` | `run_once()`、`LLMEngine.metrics()`、各 `reset_metrics()` | benchmark 增强、统计增强 | 原始仓库缺少可复现实验尺子 | 量化吞吐、TTFT、prefill/decode latency、KV block、graph fallback | 当前代码可验证实现；当前仓库缺少原始 logs/jsonl |
| 2 | Token-budget Fair Chunked Prefill | `config.py`、`scheduler.py`、`bench.py` | `Scheduler._schedule_fair_prefill()` | 调度策略变化 | 长 prompt prefill 可能独占 token budget，短请求等待 | 降低单次 prefill step 长尾，改善混合长度 workload 的调度形态 | 当前代码可验证实现；性能数字当前仓库无法复核 |
| 3 | Shared-prefix Full-block KV Late Merge | `config.py`、`block_manager.py`、`scheduler.py`、`bench.py` | `BlockManager.hash_blocks()`、`_late_merge_block()` | KV Cache 管理变化、数据结构变化 | 多请求共享 prefix 但未在 allocate 阶段命中时，会产生重复 physical KV blocks | 降低 resident used KV blocks，提升 shared-prefix 场景容量和 decode 局部性 | 当前代码可验证实现；性能数字当前仓库无法复核 |
| 4 | Safe CUDA Graph Eligibility / Fallback | `config.py`、`model_runner.py`、`llm_engine.py`、`bench.py` | `ModelRunner.run_model()`、`capture_cudagraph()` | correctness guardrail、CUDA Graph 相关变化 | decode graph replay 对 batch size、context length、block table shape 敏感 | 避免 ineligible shape 直接 replay，统计 fallback reason | 当前代码可验证实现；fallback 非零实验当前仓库缺失 |
| 5 | Matrix Benchmark Subprocess Isolation | `bench.py`、`tests/test_bench_cli.py` | `run_repeated_isolated()`、`build_child_command()` | 生命周期管理、benchmark 稳定性 | 同一进程 repeat 创建/销毁 LLM 后 CUDA/NCCL 状态未完全冷启动 | 提高矩阵实验可运行性和隔离性 | CLI 单测可验证参数解析；真实 GPU 稳定性当前仓库无日志 |

说明：

1. `docs/OSforInfra.md`、`docs/OtherTrick.md`、`docs/superpowers/...` 是设计/调研/计划文档，不直接改变运行逻辑。
2. `tests/test_bench_cli.py` 只覆盖 CLI 参数解析，未覆盖 GPU 正确性、输出一致性或性能回归。
3. `config.py` 新增配置项默认值保持保守，`prefill_policy="fcfs"`、`enable_prefix_late_merge=False`、`enforce_eager=False`。

## 5. 优化 1：Benchmark & Metrics

### 5.1 原始行为

原始仓库缺少系统化 benchmark 矩阵和内建 metrics 聚合。优化前难以回答：

1. 吞吐是否变化；
2. TTFT（Time To First Token）是否变化；
3. prefill step 与 decode step 的耗时分别如何；
4. KV block 水位是否变化；
5. prefix cache 命中与 late merge 成功是否变化；
6. CUDA Graph replay/fallback 是否变化。

这属于实验基础设施问题，不是直接推理优化。

### 5.2 优化动机

没有尺子时，任何“优化有效”都容易停留在主观判断。该项目先补 `bench.py`、engine metrics 和 JSON/JSONL 输出，是为了把后续 trick 放入同一 workload、同一 seed、同一模型参数下比较。

### 5.3 具体代码修改

文件：`bench.py`

| 函数 | 当前行号 | 修改后行为 |
| --- | ---: | --- |
| `make_random_workload()` | L47-L56 | 随机 prompt 长度和输出长度 |
| `make_mixed_lengths_workload()` | L59-L78 | 60% short、30% medium、10% long 的混合长度 prompt |
| `make_shared_prefix_workload()` | L81-L94 | 多请求共享固定 prefix，再拼接随机 suffix |
| `run_once()` | L117-L200 | warmup、reset metrics、加入所有请求、循环 step，输出 JSON summary |
| `summarize_results()` | L231-L265 | 聚合 throughput、TTFT、prefill/decode、KV block、prefix、graph 指标 |
| `parse_args()` | L373-L403 | 增加 workload、matrix、repeat、output-jsonl、prefill policy、late merge 等 CLI 参数 |

文件：`nanovllm/engine/llm_engine.py`

| 函数 | 当前行号 | 修改后行为 |
| --- | ---: | --- |
| `reset_metrics()` | L57-L67 | 重置 scheduler、model runner，并初始化 TTFT、prefill/decode latency 等 engine metrics |
| `metrics()` | L69-L84 | 聚合 engine、scheduler、block_manager、model_runner、cuda 指标 |
| `step()` | L86-L102 | 按 prefill/decode 记录 step latency，首次完成 token 时记录 TTFT |

文件：`nanovllm/engine/block_manager.py`

新增 `reset_metrics()` L37-L48 和 `_update_peak_used_blocks()` L50-L51，统计 allocation、reuse、prefix hits/misses、late merge、peak used blocks。

文件：`nanovllm/engine/model_runner.py`

新增 `reset_metrics()` L63-L68 和 `_record_graph_fallback()` L70-L73，统计 eager prefill runs、eager decode runs、graph replays、graph fallbacks 和 fallback reasons。

### 5.4 执行链路

```text
bench.py: main()
→ parse_args()
→ run_repeated_isolated() 或 run_repeated()
→ run_once()
→ LLM(...)
→ warmup generate()
→ llm.reset_metrics()
→ llm.add_request()
→ while not llm.is_finished(): llm.step()
→ llm.metrics()
→ JSON / JSONL 输出
```

Benchmark 本身不改变模型输出路径，除非通过参数打开 `prefill_policy=fair` 或 `enable_prefix_late_merge`。

### 5.5 核心数据结构和不变量

| 数据 | 来源 | 不变量 |
| --- | --- | --- |
| `metadata` | `bench.py:run_once()` | 记录 workload、seed、配置项，便于复现实验 |
| `summary.throughput_tok_s` | `total_output_tokens / elapsed` | 只统计生成 token，不统计 prompt token |
| `ttft_latencies` | `LLMEngine.step()` | 每个 seq 只在首次 completion token 出现时记录一次 |
| `prefill_step_latencies` | `LLMEngine.step()` | 只在 batch 包含 prefill 时记录 |
| `decode_step_latencies` | `LLMEngine.step()` | 只在纯 decode batch 时记录 |
| `block_manager.peak_used_blocks` | `BlockManager` | 是 BlockManager 观测水位，不等价于 GPU allocator peak memory |

### 5.6 为什么可能提升性能

该优化本身不提升推理性能。它的收益是工程上的：能定位后续改动影响的是 prefill、decode、TTFT、KV residency 还是 CUDA Graph replay。

### 5.7 复杂度

Benchmark 额外复杂度主要在 Python 侧：

1. 每次 step 增加 `time.perf_counter()` 调用，复杂度 O(1)。
2. latency list 按 step/request 数增长，空间复杂度 O(number_of_steps + number_of_requests)。
3. JSONL repeat 会把每次运行完整结果写入磁盘，空间复杂度 O(repeat * variants)。

### 5.8 正确性与边界

当前 benchmark 局限：

1. 未在每个 step 前后显式 `torch.cuda.synchronize()`，因此 step latency 可能受到 GPU 异步执行影响。
2. `throughput_tok_s` 包含 Python scheduler、engine loop 和模型执行 wall time，不是纯 GPU kernel throughput。
3. `tests/test_bench_cli.py` 只验证参数解析，不验证 benchmark 输出语义。
4. 当前仓库无原始 logs/jsonl，历史性能数字无法从本地文件复核。

### 5.9 与 vLLM 的关系

正式 vLLM 有更完整的 benchmark、serving metrics、Prometheus 指标和 profiling 工具链。本项目的 benchmark 是轻量复现实验脚手架，适合简历项目展示“先建立可测量指标再优化”，不能等同于 vLLM production observability。

## 6. 优化 2：Token-budget Fair Chunked Prefill Scheduler

### 6.1 原始 nano-VLLM 行为

原始调度器核心路径在 `Scheduler.schedule()`。当前版本保留了 FCFS 路径：

```text
nanovllm/engine/scheduler.py
Scheduler.schedule()       L42-L51
Scheduler._schedule_fcfs() L53-L87
```

FCFS prefill 逻辑在 token budget 内按等待队列顺序调度。若第一个 waiting sequence 是长 prompt，它可能占用大部分 `max_num_batched_tokens`，后续短请求需要等待。这个问题在 mixed-length workload 下更明显，属于调度和排队延迟问题，不是 attention kernel 本身的问题。

### 6.2 优化动机

Chunked Prefill 的核心思想是把长 prompt 的 prefill 拆成多个 chunk，让 scheduler 在一个 step 内或者多个 step 间更公平地分配 token budget。理论收益：

1. 降低单个超长 prefill step 的执行时间；
2. 避免长 prompt 长时间阻塞后续请求；
3. 让 batch 更接近 `max_num_batched_tokens` 的可控上限。

当前实现是 token-budget fair prefill 原型，不是成熟 serving scheduler。

### 6.3 具体代码修改

文件：`nanovllm/config.py`

```text
L15: prefill_policy: str = "fcfs"
L16: prefill_chunk_size: int = 1024
L28-L29: 校验 prefill_policy 和 prefill_chunk_size
```

文件：`nanovllm/engine/scheduler.py`

```text
类：Scheduler
当前行号：L10-L24

新增成员：
prefill_policy
prefill_chunk_size
metrics
```

核心函数：

```text
Scheduler.schedule()
当前行号：L42-L51

修改后行为：
1. waiting 非空且 prefill_policy == "fair" 时调用 _schedule_fair_prefill()
2. waiting 非空且 policy 为 fcfs 时调用 _schedule_fcfs()
3. waiting 为空时调用 _schedule_decode()
```

```text
Scheduler._schedule_fair_prefill()
当前行号：L89-L128

关键逻辑：
1. 扫描 waiting 队列一轮；
2. 如 sequence 尚未分配 block，则先调用 block_manager.can_allocate() 和 allocate()；
3. remaining_budget = max_num_batched_tokens - scheduled_tokens；
4. chunk_tokens = min(seq.num_tokens - seq.num_scheduled_tokens, prefill_chunk_size, remaining_budget)；
5. chunk 未完成则重新放回 waiting；
6. prompt prefill 完成后将 seq 状态改为 RUNNING 并进入 running 队列。
```

文件：`bench.py`

```text
L123-L127: mode == optimized 时自动启用 fair policy 和 prefix late merge
L211-L218: matrix 中 fair / late_merge / optimized 各自只打开对应变量
L392-L393: CLI 支持 --prefill-policy 和 --prefill-chunk-size
```

### 6.4 执行链路

```text
LLMEngine.add_request()
→ Sequence.num_scheduled_tokens 初始为 0
→ Scheduler.add() 将 sequence 放入 waiting
→ LLMEngine.step()
→ Scheduler.schedule()
→ Scheduler._schedule_fair_prefill()
→ 为多个 waiting sequence 分配 chunk token
→ ModelRunner.prepare_prefill()
→ Attention.forward() 写入对应 slot_mapping 的 KV
→ Scheduler.postprocess()
→ 若 prompt 未完成，sequence 留在 waiting；若完成，进入 running 参与 decode
```

### 6.5 核心数据结构和不变量

| 数据结构 | 文件 | 字段 | 不变量 |
| --- | --- | --- | --- |
| `Sequence` | `sequence.py` L18-L31 | `num_scheduled_tokens` | 表示该 sequence 已经被 prefill 计算过的 token 数，chunk 必须连续推进 |
| `Scheduler.waiting` | `scheduler.py` | deque | 未完成 prefill 的请求停留在 waiting |
| `Scheduler.running` | `scheduler.py` | deque | prefill 完成后进入 decode 阶段 |
| token budget | `scheduler.py` | `max_num_batched_tokens` | 每轮 schedule 的总 token 数不能超过 budget |
| chunk size | `config.py` L16 | `prefill_chunk_size` | 单个 sequence 单轮 prefill token 上限 |

正确性条件：

1. `num_scheduled_tokens` 只能递增，不能跳跃或回退。
2. chunk 边界必须和 `prepare_prefill()` 读取的 token 范围一致。
3. block allocation 必须覆盖该 sequence 后续写入的 physical slots。
4. prompt 未完成的 sequence 不应进入 running decode。
5. prefill 和 decode 的 token budget 统计不能重复。

### 6.6 为什么可能提升性能

收益来自调度形态变化：

1. 长 prompt 被限制在 `prefill_chunk_size` 内，降低单 step prefill 时间上限。
2. waiting 队列可以在一个 budget 内容纳多个请求的 prefill chunk。
3. mixed-length workload 下，短请求有机会更早完成 prefill。

但当前代码没有 decode-first 策略。如果 waiting 长期非空，`Scheduler.schedule()` 会优先 prefill，因此理论上仍可能造成 decode starvation。生产级 vLLM 通常会结合 chunked prefill、decode priority、max partial prefills、long prefill thresholds 等策略。

### 6.7 时间与空间复杂度

设 waiting 请求数为 `W`，本轮 token budget 为 `B`，chunk size 为 `C`。

| 项 | FCFS | fair prefill |
| --- | --- | --- |
| waiting 扫描 | 通常从队首推进，最坏 O(W) | 每轮扫描 waiting 一次，O(W) |
| 单请求 token | 可接近 `B` | 不超过 `min(C, B)` |
| 额外空间 | 无明显额外空间 | 主要是 queue 重新入队，O(1) 级别 |

代价：

1. 长 prompt 被拆多轮，可能增加 Python schedule 次数。
2. 多 sequence chunk 可能增加 block table 和 slot mapping 构造开销。
3. 如果 workload 本来都是短请求，fair policy 可能只有开销没有收益。

### 6.8 正确性与边界情况

| 边界 | 当前实现审计 |
| --- | --- |
| 空 waiting | `schedule()` 会走 decode 路径 |
| 长度不足一个 chunk | 一轮完成，进入 running |
| 长 prompt | 多轮 chunk，未完成时放回 waiting |
| decode starvation | 当前没有 decode-first guard，存在设计风险 |
| preemption | decode 路径有 `preempt()`，fair prefill 路径不涉及抢占 running |
| abort | 当前仓库未发现请求 abort API，无法验证 |
| 多线程 | Scheduler 当前按 engine loop 单线程调用，暂无锁；未来多线程 serving 需要同步保护 |

### 6.9 优化代价和适用边界

适合：

1. mixed-length prompt；
2. 存在少量超长 prompt 和大量短 prompt；
3. `max_num_batched_tokens` 足够容纳多个 chunk。

可能无收益或退化：

1. random workload 且长度分布不产生明显 head-of-line blocking；
2. 全部请求长度接近；
3. chunk size 过小导致 scheduler overhead 增大；
4. decode 已经是瓶颈时，prefill chunking 不能直接提升 decode throughput。

### 6.10 与 vLLM 的关系

正式 vLLM 的 chunked prefill 是 serving scheduler 的核心策略之一，会和 decode 调度、prefix cache、preemption、max partial prefills 等共同工作。本项目实现的是 nano-VLLM 中的简化 token-budget chunking，用于展示调度机制理解和可测量原型，不应声称达到 vLLM 生产实现完整度。

## 7. 优化 3：Shared-prefix Full-block KV Late Merge

### 7.1 原始 nano-VLLM 行为

原始 BlockManager 负责 physical KV block 的分配、引用计数和 per-sequence block table。已有 allocate-time prefix cache 会在 `can_allocate()` 里通过 hash 查询已缓存 block，命中则复用。

问题是：如果多个请求共享 prefix，但某些共享 block 在请求分配时还没有完成 hash 登记，或者因为并发 prefill 时序导致 allocate 阶段未命中，那么这些请求会先各自分配 physical block 并写入相同 KV。原始机制不会在写入后再把相同 full block 合并。

这属于 KV Cache residency 和内存管理问题，不是减少 attention 计算的问题。

### 7.2 优化动机

Shared-prefix workload 中，多请求拥有相同 prefix。若 block size 为 256，prefix 长度为 768，则理论上存在 3 个 full block 的共享机会。Late merge 希望在 block 已经写入并完成后，检查 full block 内容是否相同，并把重复 physical block 的后续引用重定向到 canonical block。

理论收益：

1. 降低 resident used KV blocks；
2. 降低后续 decode 访问的 KV cache footprint；
3. 提高同一 batch 内共享 prefix 的 capacity；
4. 为面试展示 PagedAttention、block table、ref_count、prefix cache 的工程理解。

### 7.3 具体代码修改

文件：`nanovllm/config.py`

```text
L17: enable_prefix_late_merge: bool = False
```

文件：`nanovllm/engine/block_manager.py`

```text
类：Block
当前行号：L8-L23

字段：
block_id
ref_count
hash
token_ids
```

```text
函数：BlockManager.compute_hash()
当前行号：L53-L59

行为：
1. 使用 xxhash64；
2. 如果 prefix != -1，则把 prefix hash 写入 hash 输入；
3. 再写入当前 block token ids；
4. 得到链式 block hash。
```

```text
函数：BlockManager.can_allocate()
当前行号：L79-L97

行为：
1. 对 range(seq.num_blocks - 1) 遍历 full block，不处理最后一个 partial block；
2. 逐 block 计算链式 hash；
3. 查询 hash_to_block_id；
4. 如果 hash 命中，还要比较 token_ids；
5. 统计 prefix probes/hits/misses。
```

```text
函数：BlockManager.hash_blocks()
当前行号：L134-L147

行为：
1. 从 newly full block 开始计算 hash；
2. 填写 block.hash 和 block.token_ids；
3. 若 enable_prefix_late_merge=True，调用 _late_merge_block()；
4. 否则把 hash 登记到 hash_to_block_id。
```

```text
函数：BlockManager._late_merge_block()
当前行号：L149-L174

行为：
1. 查询 canonical block；
2. 如果 canonical 与当前 block 不是同一个，且 token_ids 完全相同，则允许合并；
3. 将 seq.block_table[block_index] 重定向到 canonical_id；
4. canonical.ref_count += 1；
5. duplicate.ref_count -= 1；
6. duplicate.ref_count == 0 时回收 duplicate physical block；
7. 统计 late_merge_successes 和 reclaimed_blocks；
8. 调用 _assert_consistent() 做 ref_count 非负检查。
```

文件：`nanovllm/engine/scheduler.py`

```text
Scheduler.postprocess()
当前行号：L159-L170

行为：
1. model 输出后调用 block_manager.hash_blocks(seq)；
2. 更新 seq.num_cached_tokens；
3. 追加 sampled token；
4. sequence 完成则 deallocate。
```

### 7.4 执行链路

```text
LLMEngine.step()
→ Scheduler.schedule()
→ BlockManager.allocate()
→ ModelRunner.prepare_prefill()
→ Attention.forward()
→ store_kvcache() 已把当前 token 的 K/V 写入 physical block
→ Scheduler.postprocess()
→ BlockManager.hash_blocks(seq)
→ BlockManager._late_merge_block(seq, block_index, hash_value, token_ids)
→ seq.block_table[block_index] 从 duplicate block 改为 canonical block
→ duplicate block ref_count 归零后进入 free list
```

为什么叫 late merge：合并发生在 KV 写入之后，而不是 allocate 阶段命中 prefix cache 之前。

### 7.5 核心数据结构和不变量

| 数据结构 | 字段 | 不变量 |
| --- | --- | --- |
| `Block` | `block_id` | physical block id 全局唯一 |
| `Block` | `ref_count` | 被 block_table 引用的次数，不能为负 |
| `Block` | `hash` | 当前 full block 的链式 hash，free block 被重新分配时需要清理旧映射 |
| `Block` | `token_ids` | full block token 内容，用于 hash collision 后校验 |
| `Sequence.block_table` | list[int] | 逻辑 block index 到 physical block id 的映射，重定向不能改变逻辑 token 顺序 |
| `hash_to_block_id` | dict | hash 到 canonical physical block 的映射 |

关键正确性条件：

1. hash 相同不能直接共享，必须 `canonical.token_ids == token_ids`。
2. 只处理 full block，partial block 不参与共享，避免未完成 KV 被其他请求读到。
3. 重定向 block table 后，duplicate block 的 ref_count 必须减一。
4. canonical block 的 ref_count 必须加一，避免被提前释放。
5. free block 重新分配时，如果它是 hash map 中 canonical，必须删除旧映射。

### 7.6 为什么可能提升性能

Late merge 的直接收益不是减少 prefill FLOPs，而是减少 resident KV blocks。对于共享 prefix：

1. 多个 sequence 的相同 full prefix block 只保留一个 canonical physical block；
2. duplicate physical block 被回收到 free list；
3. 后续 decode 的 block table 指向同一 canonical block；
4. KV cache 可容纳更多请求，或在同样请求量下降低 block manager 观测水位。

如果历史实验中出现 “used KV blocks 降低约 25.12%”，从机制上与该优化吻合。但当前仓库缺少原始日志，不能在本文中把该数字标为已复核事实。

### 7.7 时间与空间复杂度

设每个 sequence 有 `L` 个 token，block size 为 `S=256`，full block 数为 `K=floor(L/S)`。

| 操作 | 复杂度 |
| --- | --- |
| 每个 full block hash | O(S)，因为要读当前 block token ids |
| hash table 查询 | 平均 O(1) |
| token equality check | 命中时 O(S) |
| block table redirect | O(1) |
| ref_count 更新 | O(1) |

代价：

1. 对每个完成的 full block 增加 hash 和可能的 token compare。
2. shared-prefix ratio 很低时，hash probe 只有 misses，收益小。
3. block size 越大，单次 token compare 成本越高，但 metadata 数量越少。
4. 当前只存 token ids，不存 KV 内容校验。正确性依赖同模型、同 token、同位置编码和同推理配置下 KV 确定性一致。

### 7.8 正确性与边界情况

| 边界 | 当前实现审计 |
| --- | --- |
| prefix 长度不足一个 full block | 不会 late merge |
| prefix 不是 block_size 整数倍 | 只 merge 完整 block，尾部 partial 不 merge |
| hash collision | 有 token_ids 校验，能避免不同 token 误共享 |
| 相同 token 不同模型/LoRA | 当前单模型假设下可接受；生产系统需把 model id、LoRA id、cache salt 加入 identity |
| RoPE position | 链式 hash 由 block 顺序隐含，且 token 位置由逻辑 block index 决定；生产中仍建议显式纳入 cache identity |
| sequence finish | `BlockManager.deallocate()` 遍历 block_table 降 ref_count，为 0 则释放 |
| abort | 当前仓库未发现 abort API，无法验证 |
| preemption | decode preemption 会 deallocate victim；late merge 后 ref_count 应能保护 canonical |
| 多线程竞争 | 当前 engine loop 单线程调度，未加锁；未来多线程 serving 需要保护 hash map、free list 和 ref_count |
| duplicate block 已被 consumer 使用 | 当前 merge 发生在 scheduler postprocess 后、下一轮使用前；单线程下风险较低 |

### 7.9 优化代价和适用边界

收益最大：

1. 多请求共享长 prefix；
2. shared prefix 至少覆盖一个或多个 full block；
3. 请求同时或近似同时进入系统，allocate-time prefix cache 未完全覆盖；
4. KV cache block pressure 是瓶颈。

基本无收益：

1. random prompt；
2. prefix 长度小于 block size；
3. 所有共享 block 已经被 allocate-time prefix cache 命中；
4. 输出短、decode 少、KV residency 不是瓶颈。

可能退化：

1. hash/compare 成本超过节省的 KV residency；
2. block size 过大导致 token compare 开销增加；
3. 高并发生产环境中缺少锁和 eviction 策略。

### 7.10 与 vLLM/SGLang 的关系

与 vLLM Automatic Prefix Caching（APC）相似点：

1. 都以 block 为单位复用 prefix KV；
2. 都需要 block hash、block table 和 ref_count；
3. 都必须处理 hash collision 和 cache identity。

差异：

1. vLLM APC 通常在 allocation/cache lookup 阶段复用，避免重复计算和写入；当前 late merge 是写入后合并，主要节省后续 resident KV blocks。
2. vLLM production 会处理 eviction、multi-tenant cache salt、LoRA、多模态、分布式一致性等；当前实现没有覆盖。
3. SGLang RadixAttention 更偏向 radix tree 管理 prefix 共享，能处理更细粒度的 prefix 结构；当前实现是 fixed-size full block hash map。

## 8. 优化 4：Safe CUDA Graph Eligibility / Fallback

### 8.1 原始行为

CUDA Graph 适合固定 shape 的 decode replay，但 dynamic batch size、context length、block table width 和 capture window 都可能导致 replay buffer shape 不匹配。原始路径如果 graph replay 条件检查不充分，长上下文或未 capture batch 可能触发失败。

### 8.2 优化动机

Decode 阶段通常每步只生成一个 token，kernel launch overhead 和 Python overhead 更明显。CUDA Graph 通过预捕获固定 shape 执行图，理论上能降低 launch overhead。但 graph replay 必须安全：

1. batch size 必须有对应 capture graph；
2. runtime tensors 不能超过 capture buffer shape；
3. context length 不能超过 capture window；
4. block table width 不能超过 capture 时最大宽度。

### 8.3 具体代码修改

文件：`nanovllm/config.py`

```text
L18: max_seq_len_to_capture: int | None = None
L32-L35: 默认等于 max_model_len，否则截断到 max_model_len
```

文件：`nanovllm/engine/model_runner.py`

```text
ModelRunner.__init__()
当前行号：L17-L41

新增：
max_seq_len_to_capture
metrics
如果 not enforce_eager，则 capture_cudagraph()
```

```text
ModelRunner.run_model()
当前行号：L231-L265

关键逻辑：
1. prefill 一律 eager，统计 eager_prefill_runs；
2. decode 如果 enforce_eager，则 eager fallback；
3. 如果 batch size 没有 captured graph，则 fallback；
4. 如果 runtime block_tables 宽度超过 graph buffer 宽度，则 fallback；
5. 如果 context_lens 最大值超过 max_seq_len_to_capture，则 fallback；
6. 否则清理 graph_vars 缓冲区并 replay graph；
7. 统计 graph_replays、graph_fallbacks、graph_fallback_reasons。
```

```text
ModelRunner.capture_cudagraph()
当前行号：L275-L310

关键逻辑：
1. max_num_blocks 基于 max_seq_len_to_capture 计算；
2. 预捕获 batch size [1, 2, 4, 8] 和 16 间隔 bucket；
3. 每个 graph bucket 维护 input_ids、positions、slot_mapping、context_lens、block_tables 等 graph_vars。
```

### 8.4 执行链路

```text
LLMEngine.step()
→ Scheduler.schedule() 产出 decode batch
→ ModelRunner.prepare_decode()
→ ModelRunner.run_model(input_ids, positions, is_prefill=False)
→ CUDA Graph eligibility checks
→ eligible: graph.replay()
→ ineligible: self.model(...) eager path
→ metrics 记录 replay 或 fallback reason
```

### 8.5 核心数据结构和不变量

| 数据 | 不变量 |
| --- | --- |
| `graphs` | key 为 captured batch size，只有对应 batch size 才能 replay |
| `graph_vars` | runtime tensor copy 不能超过 captured buffer shape |
| `context_lens` | 最大 context length 不能超过 `max_seq_len_to_capture` |
| `block_tables` | width 不能超过 captured `max_num_blocks` |
| `slot_mapping` | replay 前需要填充无效值，避免上次 replay 残留 |

### 8.6 为什么可能提升性能

该改动主要是稳定性收益，而不是必然的性能提升。安全 fallback 可以避免不符合 capture 条件的 decode batch 错误 replay；在 eligible batch 上继续使用 CUDA Graph，保留原有 fast path。

如果 benchmark 中 `graph_fallbacks=0`，只能说明测试 workload 没有触发 fallback，不代表 fallback 机制无价值。它的价值在长上下文、动态 shape 或 capture window 限制场景更明显。

### 8.7 复杂度

每次 decode 增加若干 O(1) 或 O(batch_size) 检查：

1. `bs not in graphs`：O(1)；
2. `block_tables.size(1)`：O(1)；
3. `context_lens.max()`：O(batch_size)；
4. buffer fill/copy：O(captured buffer size)。

代价一般小于错误 graph replay 的风险，但小 batch decode 下额外检查仍可能带来微小 overhead。

### 8.8 正确性与边界情况

| 边界 | 当前实现 |
| --- | --- |
| prefill | 不使用 CUDA Graph，直接 eager |
| enforce_eager | 直接 eager，并记录 fallback |
| batch size 未捕获 | fallback |
| context length 超 capture | fallback |
| block table width 超 capture | fallback |
| graph buffer 残留 | replay 前填充/清零 |
| fallback 统计完整性 | 覆盖 run_model 中当前检查路径，但无法证明覆盖未来新增路径 |

### 8.9 与 vLLM 的关系

正式 vLLM 中 CUDA Graph replay 同样依赖 capture sizes、shape constraints 和 fallback。当前实现是 nano-VLLM 的简化版本，体现了 graph fast path 必须有 runtime guardrail 的思想，但没有覆盖 production 中更多维度，如 attention backend、kv dtype、multi-GPU collective、spec decode、LoRA 和不同 model runner path。

## 9. 工程性补丁：Matrix Benchmark 子进程隔离

### 9.1 背景

历史运行中 `python bench.py --matrix --repeat 5 --output-jsonl ...` 曾报错或无法稳定结束。当前 commit `07962aa` 的提交信息说明根因判断是：同一 Python 进程内反复创建/销毁 LLM，CUDA allocator / NCCL 状态没有完全回到冷启动，第二轮开始 `allocate_kv_cache()` 的显存估算可能变成非正数。

### 9.2 代码修改

文件：`bench.py`

| 函数 | 行号 | 行为 |
| --- | ---: | --- |
| `build_child_command()` | L295-L333 | 为每个 variant/repeat 构造 `--single-run-json` 子进程命令 |
| `run_child_command()` | L336-L348 | 执行子进程、捕获 stdout/stderr、解析 JSON |
| `run_repeated_isolated()` | L351-L370 | matrix 或 repeat 场景下逐个子进程运行并写 JSONL |
| `main()` | L415 | `args.matrix or args.repeat > 1 or args.output_jsonl` 时走 isolated runner |

文件：`tests/test_bench_cli.py`

L20-L38 验证 `--matrix --repeat --output-jsonl` 能解析；L40-L45 验证 `--single-run-json` 参数能解析。

### 9.3 价值与边界

价值：

1. matrix 中不同 variant/repeat 更接近冷启动；
2. 避免前一个 LLM 实例的 CUDA/NCCL 生命周期影响后一个实验；
3. 让 benchmark 输出更适合 A/B 对比。

边界：

1. 子进程隔离增加总实验耗时；
2. 当前测试没有在 GPU 环境里验证 allocator/NCCL 问题彻底消失；
3. 子进程 cold start 会把模型加载开销排除在单次 `run_once()` 计时之外，但总 wall time 会变长。

## 10. Benchmark 设计审查

### 10.1 Workload 生成

| Workload | 代码 | 行号 | 设计 |
| --- | --- | ---: | --- |
| random | `make_random_workload()` | `bench.py` L47-L56 | 随机 prompt 和 output length，检验低 prefix-sharing 场景 |
| mixed_lengths | `make_mixed_lengths_workload()` | L59-L78 | 60% short、30% medium、10% long，检验长短请求混排 |
| shared_prefix | `make_shared_prefix_workload()` | L81-L94 | 统一 shared prefix 加随机 suffix，检验 prefix cache/late merge |

### 10.2 Variant 定义

`bench.py:variant_args()` L203-L221：

| 配置 | prefill_policy | enable_prefix_late_merge | mode |
| --- | --- | --- | --- |
| baseline | `fcfs` | False | baseline |
| fair | `fair` | False | baseline |
| late_merge | `fcfs` | True | baseline |
| optimized | `fair` | True | optimized |

这使矩阵实验可以分离 fair scheduler 和 late merge 的单独影响。

### 10.3 指标定义

| 指标 | 来源 | 定义 |
| --- | --- | --- |
| throughput_tok_s | `bench.py` L190-L193 | `total_output_tokens / elapsed_s` |
| TTFT | `LLMEngine.step()` L86-L102 | 请求加入后，到第一个 completion token 出现 |
| prefill_step_s | `LLMEngine.step()` | 包含 prefill batch 的 step wall time |
| decode_step_s | `LLMEngine.step()` | 纯 decode batch 的 step wall time |
| peak_used_blocks | `BlockManager` | BlockManager 观测到的 used block 水位 |
| prefix_hits/misses | `BlockManager.can_allocate()` | allocate-time prefix cache probe 结果 |
| late_merge_successes | `BlockManager._late_merge_block()` | late merge 成功次数 |
| graph_fallbacks | `ModelRunner.run_model()` | CUDA Graph fallback 次数 |

### 10.4 当前实验数据状态

当前仓库搜索：

```bash
rg --files -g '*.json' -g '*.jsonl' -g '*.csv' -g '*.log' -g '*.md' -g '*.sh' -g '*.py'
```

结果中没有 `logs/` 目录，也没有 `*.jsonl`、`*.csv`、`*.log` 原始 benchmark 输出。因此：

1. “used KV blocks 降低约 25.12%”无法从当前仓库复核；
2. “吞吐提升约 5.17%”无法从当前仓库复核；
3. “decode P95 latency 降低约 7.03%”无法从当前仓库复核；
4. “mixed-lengths 下 prefill step mean latency 降低约 12.5%”无法从当前仓库复核；
5. “random workload 吞吐波动约 -0.44%”无法从当前仓库复核；
6. “random workload TTFT P95 波动约 +0.59%”无法从当前仓库复核。

这些数字可以作为“历史外部运行记录中曾观察到的初步结果”讨论，但当前文档不能把它们标为仓库内可验证事实。

### 10.5 推荐补充实验命令

为了让简历表述更可审计，建议重新生成并提交或归档原始结果：

```bash
python bench.py --model ~/models/Qwen3-0.6B/ --matrix --workload random --num-seqs 128 --repeat 5 --output-jsonl logs/trick1_random_matrix.jsonl
python bench.py --model ~/models/Qwen3-0.6B/ --matrix --workload mixed_lengths --num-seqs 128 --repeat 5 --output-jsonl logs/trick1_mixed_matrix.jsonl
python bench.py --model ~/models/Qwen3-0.6B/ --matrix --workload shared_prefix --num-seqs 128 --repeat 5 --output-jsonl logs/trick1_shared_matrix.jsonl
```

运行后至少保存：

1. JSONL 原始文件；
2. stdout summary；
3. `nvidia-smi`、CUDA、PyTorch、GPU 型号；
4. commit id；
5. 是否启用 `--enforce-eager`；
6. 模型路径、dtype、max model length、block size。

## 11. 实验结果和可信度分析

### 11.1 当前仓库内可验证实验

当前仓库只有 `tests/test_bench_cli.py` 单元测试：

| 测试 | 行号 | 验证内容 |
| --- | ---: | --- |
| `test_matrix_repeat_and_jsonl_flags_parse()` | L20-L38 | `--matrix`、`--repeat`、`--output-jsonl` 参数可解析 |
| `test_single_run_json_flag_parse()` | L40-L45 | `--single-run-json` 参数可解析 |

这只能证明 CLI 入口支持相关参数，不能证明 GPU benchmark 结果、性能收益或输出正确性。

### 11.2 历史性能说法可信度分级

| 说法 | 代码机制是否支持 | 当前仓库实验是否可复核 | 文档建议 |
| --- | --- | --- | --- |
| shared-prefix 下 used KV blocks 降低约 25.12% | 支持，late merge 会释放 duplicate full block | 否 | 可写“历史实验观察到”，不能写“已在仓库复现” |
| shared-prefix 下吞吐提升约 5.17% | 机制上可能，但吞吐受多因素影响 | 否 | 谨慎写“在特定 workload 初步观察到” |
| shared-prefix 下 decode P95 降低约 7.03% | 可能与 KV footprint 降低相关，但缺少 profiling 因果证明 | 否 | 不能写成因果结论 |
| mixed-lengths 下 prefill mean 降低约 12.5% | fair chunking 机制支持降低单 step prefill | 否 | 可作为待复核结果 |
| random workload 吞吐约 -0.44% | 机制上 random 无 prefix 收益，轻微 overhead 合理 | 否 | 可作为预期风险说明 |
| CUDA Graph fallback 为 0 | 代码可统计 | 否 | 只能说当前代码具备统计能力 |

### 11.3 适合写入简历的结论

可写：

1. 设计并实现 `random / mixed_lengths / shared_prefix` benchmark matrix，支持 repeat、JSONL、A/B variants 和子进程隔离。
2. 实现 token-budget chunked prefill scheduler 原型，将长 prompt prefill 拆分为受 `prefill_chunk_size` 控制的 chunk。
3. 实现 shared-prefix full-block KV late merge 原型，使用链式 block hash、token 校验、ref_count 和 block table redirect 回收重复 physical blocks。
4. 为 decode CUDA Graph 增加 runtime eligibility checks 和 fallback reason metrics。

谨慎写：

1. “在历史 shared-prefix benchmark 中观察到 resident KV block 明显下降和吞吐改善”。需要附原始日志。
2. “初步实验显示 random workload 退化较小”。需要附 repeat JSONL。

不建议写：

1. “显著提升 vLLM 推理性能”，因为本项目是 nano-VLLM 原型，不是正式 vLLM。
2. “证明 decode P95 降低来自 KV block 减少”，缺少 profiling 因果链。
3. “生产级 Prefix Caching”，当前没有 eviction、cache salt、多 GPU、一致性和并发保护。

## 12. 正确性、风险和适用边界

### 12.1 Prefix Late Merge 风险

| 风险 | 当前处理 | 剩余问题 |
| --- | --- | --- |
| hash collision | token_ids 二次校验 | 未纳入 model id、LoRA id、cache salt |
| partial block 共享 | 不共享 partial block | prefix 尾部不能受益 |
| ref_count 错误 | 加减 ref_count，并 assert 非负 | 无完整 fuzz test |
| free block 悬挂映射 | `_allocate_block()` 清理旧 canonical mapping | 需要更多回归测试 |
| 多线程竞争 | 当前单线程 engine loop | production serving 需加锁或单线程 ownership |
| abort 路径 | 当前无 abort API | 无法验证异常释放 |

### 12.2 Fair Prefill 风险

| 风险 | 当前处理 | 剩余问题 |
| --- | --- | --- |
| decode starvation | 未专门处理 | waiting 长期非空时 decode 可能被推迟 |
| prefill starvation | 扫描 waiting 一轮并重新入队 | 未实现 aged priority |
| chunk size 调参 | 固定 `prefill_chunk_size` | 缺少 workload-aware 自适应 |
| TTFT 改善不稳定 | 代码能改变调度形态 | 当前仓库无 logs 支撑稳定改善 |

### 12.3 CUDA Graph 风险

| 风险 | 当前处理 | 剩余问题 |
| --- | --- | --- |
| batch size 未 capture | fallback | 未对 fallback 性能做压力测试 |
| context length 超 window | fallback | 没有 failure injection |
| block table width 超 buffer | fallback | 未覆盖多 backend |
| replay buffer 残留 | replay 前 fill/zero | 未做输出一致性测试 |

## 13. 与 vLLM/SGLang 工业机制对比

| 机制 | 本项目 | vLLM/SGLang 工业实现 |
| --- | --- | --- |
| PagedAttention | 使用 block table 和 physical KV blocks | vLLM 有完整 block manager、eviction、swap/preemption 等 |
| Prefix Caching | allocate-time cache + late merge 原型 | vLLM APC 通常在分配阶段命中，避免重复 KV 计算 |
| RadixAttention | 未实现 radix tree | SGLang 用 radix tree 管理 prefix 共享和路由 |
| Chunked Prefill | token-budget fair prefill 原型 | vLLM 有更完整的 decode/prefill 协同调度和参数 |
| CUDA Graph | decode graph eligibility/fallback | production 要覆盖更多 dynamic features 和分布式路径 |
| 分布式推理 | 当前未审计到 TP 优化实现 | vLLM/TensorRT-LLM 需处理 NCCL、TP/PP、rank 一致性 |
| Benchmark | 单机脚本、JSON/JSONL | production 需要 serving trace、并发 QPS、P99、SLO、profiling |

## 14. 字节 AML 面试高频问题及推荐回答

### 第一组：项目概述与动机

#### Q1：为什么选择 nano-VLLM，而不是直接修改 vLLM？

**推荐回答：** 我选择 nano-VLLM 是因为它保留了 PagedAttention、BlockManager、Scheduler、CUDA Graph 等核心概念，但代码规模更适合个人在短时间内完成端到端改造和审计。我的目标不是声称替代 vLLM，而是用小框架验证推理系统里的调度、KV Cache 和 graph fallback 机制，并能讲清楚实现链路。

**回答依据：** `scheduler.py`、`block_manager.py`、`model_runner.py` 分别对应调度、KV 管理和 CUDA Graph。

**容易踩坑的回答：** “nano-VLLM 比 vLLM 更好”。这无法证明。

**继续追问：** 如果迁移到 vLLM，要改哪些模块？如何处理 eviction 和多 GPU？

#### Q2：这个项目解决的核心问题是什么？

**推荐回答：** 核心是三个点：第一，建立 benchmark 和 metrics，让优化可测；第二，在 mixed-length workload 中用 token-budget chunked prefill 改善长 prompt 阻塞；第三，在 shared-prefix workload 中用 full-block KV late merge 降低重复 physical KV blocks。另外给 CUDA Graph replay 加 eligibility guardrail，减少 shape 不匹配风险。

**回答依据：** `bench.py` L47-L200，`scheduler.py` L89-L128，`block_manager.py` L149-L174，`model_runner.py` L231-L265。

**容易踩坑的回答：** 只说“提升吞吐”，不说明路径和 workload。

**继续追问：** 哪个优化最有工程价值？哪个收益最不稳定？

#### Q3：为什么认为 KV Cache 是重要瓶颈？

**推荐回答：** 在 decode 阶段，每生成一个 token 都需要读取历史 KV，长上下文和高并发下 KV Cache 占用会限制 batch 容量，并影响显存带宽和 cache residency。我的 late merge 不减少已经发生的 prefill 计算，但能在 shared-prefix 场景减少重复 physical KV blocks，让后续 decode 共享同一 prefix KV。

**回答依据：** `Attention.forward()` L71-L74 decode 使用 `flash_attn_with_kvcache`；`BlockManager._late_merge_block()` 重定向 block table。

**容易踩坑的回答：** “late merge 减少了 prefill 计算”。当前实现不是这样。

**继续追问：** 为什么 KV blocks 降低不一定等比例提升吞吐？

#### Q4：你的优化属于调度、内存管理还是算子优化？

**推荐回答：** 主要是调度和内存管理优化，不是算子优化。Fair prefill 改的是 scheduler 如何分配 token budget；late merge 改的是 BlockManager 如何合并重复 KV blocks；CUDA Graph 改的是 model runner 的安全 fast path。attention kernel 和 Triton `store_kvcache_kernel` 本身没有被改写。

**回答依据：** 变更集中在 `scheduler.py`、`block_manager.py`、`model_runner.py`，`attention.py` 未在 diff 中修改。

**容易踩坑的回答：** “优化了 FlashAttention 内核”。

**继续追问：** 如果要做算子层优化，你会从哪里下手？

#### Q5：为什么先做 benchmark？

**推荐回答：** 因为推理优化很容易被 workload 和测量方式误导。先做 random、mixed-length、shared-prefix 三类 workload 和 baseline/fair/late_merge/optimized 矩阵，可以区分低共享、长短混合和高共享场景，避免只用一个 workload 证明所有结论。

**回答依据：** `bench.py` L47-L104，L203-L221。

**容易踩坑的回答：** “跑一次快了就说明优化有效”。

**继续追问：** 当前 benchmark 还有哪些统计学不足？

### 第二组：代码实现细节

#### Q6：block hash 如何构造？

**推荐回答：** 当前 `BlockManager.compute_hash()` 使用 xxhash64。它先把父 block 的 hash 作为 prefix 写入 hash 输入，再写入当前 block token ids，因此形成链式 block hash。同样 token 但出现在不同前缀链路下，hash 会不同。

**回答依据：** `block_manager.py` L53-L59。

**容易踩坑的回答：** “只 hash 当前 token ids”。当前实现包含 prefix hash。

**继续追问：** 生产系统还应把哪些字段放进 cache key？

#### Q7：为什么 hash 相同还要 token 校验？

**推荐回答：** hash 只是加速查找，不能作为正确性依据。当前代码在 `can_allocate()` 和 `_late_merge_block()` 中都比较了 `token_ids`，只有 canonical block 的 token 内容完全一致才复用或合并，避免 hash collision 导致错误共享。

**回答依据：** `block_manager.py` L88-L91，L158-L160。

**容易踩坑的回答：** “xxhash 不会碰撞”。任何有限 hash 都可能碰撞。

**继续追问：** token 相同是否一定 KV 相同？

#### Q8：为什么只处理 full block？

**推荐回答：** full block 的 KV 已经完整写入，逻辑边界稳定，适合作为共享单位。partial block 仍可能继续追加 token，如果过早共享会导致一个 sequence 后续写入影响另一个 sequence 的逻辑视图。当前 `can_allocate()` 遍历 `range(seq.num_blocks - 1)`，明确跳过最后的 partial block。

**回答依据：** `block_manager.py` L82-L86，`Sequence.last_block_num_tokens` L59-L61。

**容易踩坑的回答：** “partial block 也可以直接共享”。风险很高。

**继续追问：** 如何支持 partial prefix 复用？

#### Q9：late merge 如何重定向 block table？

**推荐回答：** `_late_merge_block()` 找到 canonical block 后，把 `seq.block_table[block_index]` 从 duplicate block id 改为 canonical id，然后 canonical `ref_count += 1`，duplicate `ref_count -= 1`。如果 duplicate 引用归零，就释放回 free list。

**回答依据：** `block_manager.py` L161-L168。

**容易踩坑的回答：** “复制 KV 数据到 canonical”。当前实现不是复制，而是重定向引用。

**继续追问：** 如果 duplicate block 仍被其他 sequence 引用怎么办？

#### Q10：late merge 发生在 KV 写入前还是写入后？

**推荐回答：** 发生在写入后。`Attention.forward()` 会先调用 `store_kvcache()` 写 physical KV，之后 `Scheduler.postprocess()` 调用 `BlockManager.hash_blocks()`，再尝试 late merge。因此它不能节省已经发生的 prefill 计算，只能回收重复 physical block 并影响后续 decode residency。

**回答依据：** `attention.py` L62-L64，`scheduler.py` L159-L170。

**容易踩坑的回答：** “避免了重复 prefill”。当前代码不支持这个结论。

**继续追问：** allocate-time prefix cache 和 late merge 的区别？

#### Q11：Fair Chunked Prefill 的 token budget 在哪里计算？

**推荐回答：** 在 `Scheduler._schedule_fair_prefill()` 中计算。它维护 `scheduled_tokens`，每次用 `remaining_budget = max_num_batched_tokens - scheduled_tokens`，再用 `min(剩余 prompt tokens, prefill_chunk_size, remaining_budget)` 决定本轮给一个 sequence 的 chunk 大小。

**回答依据：** `scheduler.py` L97-L113。

**容易踩坑的回答：** “每个请求固定 1024 token”。实际还受剩余 prompt 和 batch budget 限制。

**继续追问：** chunk size 如何自适应？

#### Q12：chunk 后 sequence 状态如何迁移？

**推荐回答：** 如果 `seq.num_scheduled_tokens < seq.num_tokens`，说明 prompt 还没 prefill 完，sequence 会重新进入 waiting；如果完成，则状态设为 `RUNNING` 并进入 running 队列，后续参与 decode。

**回答依据：** `scheduler.py` L115-L123。

**容易踩坑的回答：** “chunk 之后直接 decode”。只有整个 prompt prefill 完成后才 decode。

**继续追问：** 多 chunk prefill 如何保证 position 连续？

#### Q13：CUDA Graph eligibility 检查了哪些条件？

**推荐回答：** 当前 `run_model()` 对 decode 检查：是否 `enforce_eager`、batch size 是否有 captured graph、runtime block table 宽度是否超过 graph buffer、context length 是否超过 `max_seq_len_to_capture`。不满足则 eager fallback 并记录 reason。

**回答依据：** `model_runner.py` L231-L265。

**容易踩坑的回答：** “只要是 decode 就一定 replay graph”。

**继续追问：** graph replay 成功和 eligibility 统计有什么区别？

#### Q14：为什么 replay 前要清理 graph buffer？

**推荐回答：** CUDA Graph 使用预分配 buffer。如果当前 batch 比 capture buffer 小，旧数据可能残留。当前代码在 replay 前对 `slot_mapping`、`context_lens`、`block_tables` 做 fill/zero，再 copy runtime tensors，减少残留 shape 内容影响。

**回答依据：** `model_runner.py` L254-L259。

**容易踩坑的回答：** “copy 新 tensor 会自动覆盖所有 buffer”。小 batch 不一定覆盖全部 buffer。

**继续追问：** 还有哪些 graph_vars 需要类似处理？

### 第三组：正确性与异常路径

#### Q15：如何避免 double free？

**推荐回答：** BlockManager 通过 `ref_count` 控制释放。late merge 时 canonical 加一，duplicate 减一，只有 duplicate `ref_count == 0` 才 `_deallocate_block()`。sequence 完成时 `deallocate()` 对 block_table 中每个 physical block 减引用，归零才释放。当前还有 `_assert_consistent()` 检查 ref_count 非负。

**回答依据：** `block_manager.py` L118-L125，L161-L174。

**容易踩坑的回答：** “Python list 会自动管理”。KV physical block 生命周期必须显式管理。

**继续追问：** 如果 block_table 中同一 physical block 出现多次怎么办？

#### Q16：hash collision 会导致错误共享吗？

**推荐回答：** 当前实现用 token equality 做二次校验，因此普通 hash collision 不会直接导致错误共享。但生产系统还应把模型、LoRA、cache salt、位置编码策略等纳入 identity，否则 token 相同但上下文配置不同也可能不应共享。

**回答依据：** `block_manager.py` L158-L160。

**容易踩坑的回答：** “token 相同就永远 KV 相同”。这依赖推理配置一致。

**继续追问：** 多租户 serving 如何隔离 prefix cache？

#### Q17：preemption 路径和 late merge 是否冲突？

**推荐回答：** 当前 decode preemption 会调用 `BlockManager.deallocate()` 释放 victim 的 block_table 引用。late merge 后 block_table 已指向 canonical，ref_count 应保护共享 block 不被提前释放。但当前没有专门的 preemption+late_merge 单测，因此只能说代码机制上兼容，未充分验证。

**回答依据：** `scheduler.py` L130-L157，`block_manager.py` L118-L125。

**容易踩坑的回答：** “完全没问题”。缺少测试覆盖。

**继续追问：** 如何设计这个单测？

#### Q18：abort 请求时如何释放？

**推荐回答：** 当前仓库没有明显的 public abort API，因此无法从当前代码验证 abort 路径。可以推测如果未来加 abort，需要复用 `BlockManager.deallocate()` 并确保 waiting/running 队列移除和 ref_count 一致，但这是设计建议，不是已实现事实。

**回答依据：** 当前审计未发现 abort 接口。

**容易踩坑的回答：** “abort 已支持”。无法验证。

**继续追问：** abort 发生在 chunked prefill 中间怎么办？

#### Q19：多线程下 hash map 和 ref_count 安全吗？

**推荐回答：** 当前 engine loop 是单线程调度模型，未看到对 BlockManager 的多线程并发访问，因此没有锁也暂时可运行。但如果进入 production serving，多 worker 同时访问 `hash_to_block_id`、free list 和 ref_count，就必须引入 ownership、锁或 actor 化管理。

**回答依据：** `LLMEngine.step()` 单路径调用 scheduler 和 model runner。

**容易踩坑的回答：** “Python GIL 可以保证正确性”。GPU/多进程/多线程场景不能这样假设。

**继续追问：** 你会用细粒度锁还是单线程 KV manager？

#### Q20：CUDA Graph fallback 是否覆盖所有异常？

**推荐回答：** 只覆盖当前 `run_model()` 中显式检查的条件，例如 batch size、block table width、context length、enforce eager。它不能覆盖未来 backend 变化、dtype、LoRA、分布式 collective 等所有条件，所以我会把它表述为 runtime guardrail 原型，而不是完整安全证明。

**回答依据：** `model_runner.py` L231-L265。

**容易踩坑的回答：** “有 fallback 就绝对安全”。

**继续追问：** 如何做 failure injection？

### 第四组：性能分析

#### Q21：为什么 KV blocks 降低 25%，吞吐可能只提升 5%？

**推荐回答：** KV block 数下降主要降低显存 residency 和后续 decode 访问 footprint，但整体吞吐还受 attention 计算、采样、Python scheduler、kernel launch、模型权重读写等影响。late merge 也不减少已经发生的 prefill 计算，所以吞吐不会和 KV blocks 等比例提升。

**回答依据：** late merge 在 `Scheduler.postprocess()` 后发生，`attention.py` 已完成 KV 写入。

**容易踩坑的回答：** “KV 降 25% 所以吞吐也应提升 25%”。

**继续追问：** 如何用 Nsight 验证瓶颈转移？

#### Q22：为什么 random workload 没有收益？

**推荐回答：** random workload 几乎没有共享 prefix，late merge 的 hash probe 多数 miss，fair prefill 也不一定改善排队。因此它更像 overhead 检查，预期收益很小甚至略退化。它的价值是证明优化没有只针对共享 prefix workload 造假。

**回答依据：** `make_random_workload()` L47-L56，late merge 需要 token 内容相同才合并。

**容易踩坑的回答：** “所有 workload 都应该提升”。

**继续追问：** random 退化多少才可以接受？

#### Q23：token compare 是否会抵消收益？

**推荐回答：** 有可能。在低 prefix-sharing 场景，hash probe 和 token compare 都是额外开销；在高共享场景，回收 KV blocks 的收益可能覆盖这些开销。当前实现只在 hash 命中时做 token compare，miss 时主要是 hash 和 dict lookup 成本。

**回答依据：** `block_manager.py` L88-L91，L158-L160。

**容易踩坑的回答：** “校验没有成本”。

**继续追问：** 如何优化 token compare？

#### Q24：如何证明提升不是 workload 造出来的？

**推荐回答：** 需要至少三类 workload：random 验证低共享场景，mixed-length 验证调度压力，shared-prefix 验证 prefix cache 场景；每类用 baseline/fair/late_merge/optimized 矩阵和 repeat，固定 seed 起点，保存 JSONL 原始数据。当前代码支持这些，但当前仓库缺少原始结果文件，所以还不能说已经完整证明。

**回答依据：** `bench.py` matrix 和 repeat 支持 L203-L221，L351-L370。

**容易踩坑的回答：** “shared-prefix 快了就证明优化通用”。

**继续追问：** 还需要真实 trace 吗？

#### Q25：当前计时方法有什么局限？

**推荐回答：** `bench.py` 用 `time.perf_counter()` 统计 wall time，`LLMEngine.step()` 也用 wall time 分 prefill/decode。它包含 Python 调度开销，但没有每步显式 `torch.cuda.synchronize()`，所以单 step latency 可能受 CUDA 异步影响。吞吐整体 loop 时间相对更可用，但 kernel 级结论需要 Nsight。

**回答依据：** `bench.py` L149-L157，`llm_engine.py` L86-L102。

**容易踩坑的回答：** “这个 latency 就是 GPU kernel latency”。

**继续追问：** 如何重新设计 latency measurement？

#### Q26：P95 样本量是否足够？

**推荐回答：** 当前代码支持 repeat 和每次多个请求，但是否足够取决于保存的 JSONL 样本量。当前仓库没有 logs/jsonl，所以无法判断 P95 稳定性。正式报告应展示每个 variant 的 mean、std、min/max 和 confidence interval，避免单次 P95 被噪声影响。

**回答依据：** `summarize_results()` L231-L265 会输出 mean/std/min/max。

**容易踩坑的回答：** “P95 一定稳定”。没有样本就不能这样说。

**继续追问：** P99 要多少请求才有意义？

### 第五组：与工业系统对比

#### Q27：与 vLLM APC 有什么区别？

**推荐回答：** vLLM APC 主要在 cache lookup/allocation 阶段复用 prefix KV，命中后可以避免重复 prefill KV 计算。我的 late merge 是写入后发现 full block 重复再合并，不能节省已经发生的计算，更像补偿 allocate-time miss 的去重原型。

**回答依据：** `block_manager.can_allocate()` 与 `_late_merge_block()` 的时序差异。

**容易踩坑的回答：** “这就是完整 APC”。

**继续追问：** 如何把 late merge 前移到 allocation 阶段？

#### Q28：与 SGLang RadixAttention 有什么区别？

**推荐回答：** SGLang RadixAttention 用 radix tree 表达 prefix 共享关系，可以更灵活地处理不同长度共享前缀。当前实现是 fixed-size full block hash map，只在完整 block 粒度复用，结构简单但表达能力弱。

**回答依据：** 当前只有 `hash_to_block_id`，没有 trie/radix tree 数据结构。

**容易踩坑的回答：** “功能等价”。不等价。

**继续追问：** 什么 workload 下 radix tree 更有优势？

#### Q29：production 中如何支持 multi-GPU / TP？

**推荐回答：** 当前审计没有发现 TP 场景下的 KV block 数 all-reduce 或分布式 prefix cache 实现。production 中需要确保各 rank block allocation 一致、cache identity 一致、eviction 一致，还要处理 NCCL 生命周期和 rank 间错误传播。这个项目目前还停留在单机单进程原型层面。

**回答依据：** `config.py` 有 `tensor_parallel_size`，但本轮 diff 未实现 TP KV block 同步。

**容易踩坑的回答：** “已经支持多卡优化”。

**继续追问：** 为什么各 rank num_kvcache_blocks 不一致会出错？

#### Q30：scheduler 是否线程安全？

**推荐回答：** 当前 scheduler 是 engine step 单线程驱动，waiting/running deque 没有锁。单线程下可接受，但 production serving 如果有多线程请求注入、abort、后台 eviction，就需要队列和 BlockManager 的并发控制。

**回答依据：** `LLMEngine.step()` 串行调用 `scheduler.schedule()` 和 `postprocess()`。

**容易踩坑的回答：** “deque 天然线程安全”。组合操作不是原子的。

**继续追问：** 你会如何设计并发模型？

### 第六组：进一步优化

#### Q31：下一步最值得做什么？

**推荐回答：** 我会先补可验证性：提交 JSONL 原始结果、输出一致性测试、late merge ref_count 单测、CUDA Graph fallback failure injection。然后做 adaptive chunk size 和 decode-first policy。最后再考虑把 late merge 前移为 allocation-time prefix cache 增强，或引入 radix tree。

**回答依据：** 当前仓库最大短板是缺少原始 benchmark logs 和 correctness tests。

**容易踩坑的回答：** “直接写 Triton kernel”。在证据不足前下钻算子可能收益不明确。

**继续追问：** adaptive chunk size 的 cost model 怎么设计？

#### Q32：如何减少 Python scheduler overhead？

**推荐回答：** 可以先用 profiling 找出 scheduler 时间占比，再考虑减少 per-step Python 对象操作、批量构造 block tables、复用 tensor buffer、把热点路径下沉到 C++/CUDA 或 Torch extension。当前项目还没提供 scheduler overhead profiling，所以这只是后续方向。

**回答依据：** `bench.py` 目前只能分 prefill/decode step wall time，没有 scheduler breakdown。

**容易踩坑的回答：** “Python 一定是瓶颈”。需要 profiling。

**继续追问：** 如何拆分 CPU scheduler 和 GPU execution 时间？

#### Q33：如何设计自适应 chunk size？

**推荐回答：** 可以根据当前 waiting/running 数、decode backlog、历史 prefill step time、目标 TTFT/SLO 动态调整 chunk size。decode backlog 高时减小 prefill chunk，让 decode 更快插队；GPU 利用率低时增大 chunk，提高吞吐。需要避免频繁震荡。

**回答依据：** 当前固定 `prefill_chunk_size` 在 `config.py` L16。

**容易踩坑的回答：** “chunk 越小越公平”。太小会增加调度和 kernel launch 开销。

**继续追问：** 用 PID 还是启发式阈值？

#### Q34：如何优化 `reshape_and_cache` 或 KV 写入？

**推荐回答：** 当前 KV 写入在 `attention.py` 的 Triton `store_kvcache_kernel`。下一步可以用 Nsight 看 memory throughput、coalescing、occupancy，再评估 fused projection+cache、向量化 store、减少 slot_mapping 间接访问或复用 vLLM 的 paged attention backend。没有 profiling 前不应声称这里是瓶颈。

**回答依据：** `attention.py` L10-L40。

**容易踩坑的回答：** “改 Triton 一定更快”。

**继续追问：** slot_mapping 随机性如何影响 store 合并？

#### Q35：如何做 workload-aware scheduling？

**推荐回答：** 需要把请求长度、prefix sharing ratio、decode backlog、SLO class、cache hit probability 纳入 cost model。比如共享 prefix 高的请求可以聚合调度以提高 cache 命中，长 prompt 可以 chunk，小 decode backlog 时提高 prefill chunk 维持吞吐。

**回答依据：** 当前 benchmark 已有三类 workload，但 scheduler 还没有 workload-aware feature。

**容易踩坑的回答：** “一个策略适合所有请求”。

**继续追问：** 如何估计 prefix sharing ratio？

#### Q36：如何把这个项目迁移到正式 vLLM？

**推荐回答：** 我会先对齐 vLLM 的 block manager、scheduler 和 APC 接口，避免另起一套 cache identity。late merge 更可能作为 APC miss 后的 dedup 或 debug feature，fair prefill 要接入 vLLM scheduler 的 decode priority 和 preemption 机制。还需要补充多 GPU、LoRA、多租户和 eviction。

**回答依据：** 当前实现只覆盖 nano-VLLM 简化路径。

**容易踩坑的回答：** “直接复制代码过去”。

**继续追问：** vLLM 中 block table 和 KV cache manager 的真实接口是什么？

## 15. 项目口述版本

### 15.1 30 秒简历概述

我在 nano-VLLM 上做了一个推理引擎优化原型，重点围绕 benchmark、调度和 KV Cache。先实现了 random、mixed-length、shared-prefix 三类 benchmark 矩阵和 TTFT、prefill/decode latency、KV block、CUDA Graph fallback 指标；然后实现 token-budget chunked prefill，缓解长 prompt 对调度的阻塞；再实现 shared-prefix full-block KV late merge，用链式 block hash、token 校验、ref_count 和 block table 重定向回收重复 KV blocks。最后给 decode CUDA Graph 加 runtime eligibility 和 fallback 统计。

### 15.2 2 分钟项目介绍

背景是大模型推理服务里，prefill 和 decode 的资源特征不同，KV Cache 又会随并发和上下文长度快速增长。nano-VLLM 虽然代码小，但有 Scheduler、BlockManager、PagedAttention 和 CUDA Graph 路径，适合做端到端原型。

我先补了 benchmark 和 metrics，支持 random、mixed-length、shared-prefix 三种 workload，以及 baseline、fair、late_merge、optimized 四种配置，指标包括吞吐、TTFT、prefill/decode step latency、KV block 水位、prefix hit 和 graph fallback。

实现上有两个核心优化。第一个是 token-budget chunked prefill，把长 prompt 拆成受 `prefill_chunk_size` 限制的 chunk，让 mixed-length 请求不会完全被单个长 prefill 占住。第二个是 shared-prefix full-block KV late merge，在 KV 写入后对已完成 full block 做链式 hash 和 token 校验，如果发现重复 block，就把 sequence 的 block table 重定向到 canonical block，并维护 ref_count 回收 duplicate block。

另外我给 decode CUDA Graph 增加了 eligibility 检查，覆盖 batch size、block table width 和 context length，ineligible 时 fallback eager 并记录 reason。

局限是当前原型还不是 production 级：没有多 GPU、eviction、LoRA/cache salt、abort、完整 correctness tests；当前仓库也缺少原始 benchmark logs，所以性能数字需要重新归档后才能作为强证据。

### 15.3 5 分钟深入介绍

原始 nano-VLLM 的链路是：请求进入 `LLMEngine.add_request()`，生成 `Sequence` 后进入 `Scheduler.waiting`；每次 `LLMEngine.step()` 调用 `Scheduler.schedule()` 产生 batch；`ModelRunner.prepare_prefill()` 或 `prepare_decode()` 构造 input ids、positions、slot mapping、block tables；`Attention.forward()` 写入 KV Cache 并调用 FlashAttention；最后 `Scheduler.postprocess()` 追加采样 token，完成时释放 block。

我做的第一步不是直接优化，而是补尺子。`bench.py` 支持三类 workload 和四个 variant，`LLMEngine`、`Scheduler`、`BlockManager`、`ModelRunner` 都增加了 metrics。这样可以分别看 TTFT、prefill step、decode step、KV block 水位和 CUDA Graph fallback。

核心优化一是 fair chunked prefill。原 FCFS prefill 可能让长 prompt 在 token budget 内占用过多计算，短请求等待。新路径 `_schedule_fair_prefill()` 会扫描 waiting 队列，每个 sequence 单轮最多调度 `prefill_chunk_size` 个 token，未完成的 sequence 回到 waiting，完成 prefill 后进入 running decode。这个优化的目标是调度公平性和 prefill step 长尾，而不是直接优化 kernel。

核心优化二是 prefix late merge。BlockManager 以 256 token 为 block。每个 full block 生成链式 hash，hash 输入包含父 block hash 和当前 block token。写入 KV 后，`hash_blocks()` 对新完成 block 尝试 `_late_merge_block()`：如果 hash 命中并且 token_ids 完全一致，就把当前 sequence 的 block table 从 duplicate block 重定向到 canonical block，canonical 引用加一，duplicate 引用减一并可能释放。这个机制只处理 full block，避免 partial block 后续写入造成错误共享。

CUDA Graph 方面，我没有声称重新优化 graph，而是增加安全回退。decode 时只有 batch size 已捕获、block table width 不超 buffer、context length 不超过 capture window 才 replay，否则 eager fallback 并记录 reason。

benchmark 设计上，random 检查低共享场景 overhead，mixed-length 检查调度，shared-prefix 检查 KV dedup。当前代码支持 repeat 和 JSONL，并用子进程隔离 matrix，避免 CUDA/NCCL 状态污染。但当前仓库缺少原始日志，所以性能数字需要重新运行保存，面试中我会把结果表述为“历史初步观察”，不会夸大成严格证明。

工业化不足包括：prefix cache identity 不含 LoRA/cache salt；没有 eviction 和 distributed consistency；scheduler 没有 decode-first 和自适应 chunk；CUDA Graph fallback 没有 failure injection；缺少输出一致性、ref_count fuzz 和多卡测试。下一步我会优先补测试和原始 benchmark，再做 adaptive scheduling 和更接近 vLLM APC 的 allocation-time reuse。

### 15.4 一句话项目定位

在 nano-VLLM 上实现可测量的调度与 KV Cache 优化原型。

## 16. 简历表述审查与推荐版本

### 16.1 逐句审查

| 原表述 | 代码能否证明 | 实验能否证明 | 风险 | 推荐修改 |
| --- | --- | --- | --- | --- |
| 实现 shared-prefix full-block KV late-merge | 能 | 当前仓库不能复核性能 | 准确，但需标注原型 | 设计并实现 shared-prefix full-block KV late-merge 原型 |
| 链式 block hash | 能 | 不需要实验 | 准确 | 使用父 block hash + 当前 block token 构造链式 block hash |
| token 校验 | 能 | 不需要实验 | 准确 | hash 命中后进行 token_ids 二次校验，降低 collision 误共享风险 |
| ref-count | 能 | 缺少单测 | 机制准确，验证不足 | 维护 canonical/duplicate block ref_count，并补充 ref_count 单测 |
| block table 重定向 | 能 | 不需要实验 | 准确 | 将重复 block 的逻辑映射重定向到 canonical physical block |
| 降低 used KV blocks | 机制能支持 | 当前仓库无 logs | 不能写强结论 | 在历史 shared-prefix 实验中观察到 used KV blocks 下降，需附 JSONL |
| 吞吐提升 | 代码不能直接证明 | 当前仓库无 logs | 容易夸大 | 在特定 shared-prefix workload 初步观察到吞吐改善 |
| decode P95 降低 | 代码不能直接证明因果 | 当前仓库无 logs | 因果不足 | 历史实验中 decode P95 有改善趋势，仍需 Nsight 验证原因 |
| 实现 token-budget chunked prefill scheduler | 能 | 当前仓库无 logs | 准确 | 实现 token-budget fair chunked prefill 原型 |
| CUDA Graph eligibility 统计 | 能 | fallback 非零实验缺失 | 准确但不要夸大 | 为 decode CUDA Graph 增加 eligibility checks、eager fallback 和 reason metrics |

### 16.2 推荐简历版本

```latex
\datedsubsection{\textbf{nano-VLLM 推理引擎调度与 KV Cache 优化原型} \quad|\quad \textit{进行中}}{2026.05 -- 至今}
\textit{技术栈：Python, PyTorch, CUDA Graph, PagedAttention, KV Cache, Benchmark/Profiling}
\vspace{-0.3ex}
\begin{itemize}[leftmargin=1.5em, itemsep=0.15ex, parsep=0.15ex, topsep=0.2ex]
    \item \textbf{Benchmark 与指标体系：} 重构 \texttt{bench.py}，支持 random / mixed-length / shared-prefix 三类 workload 及 baseline / fair / late-merge / optimized 矩阵对比，输出吞吐、TTFT、prefill/decode step latency、KV block 水位、prefix hit 与 CUDA Graph fallback 等指标，并通过子进程隔离 repeat 实验，降低 CUDA/NCCL 生命周期残留对 A/B 测试的影响。
    \item \textbf{Token-budget Chunked Prefill：} 在 Scheduler 中实现 \texttt{prefill\_policy=fair} 原型，基于 \texttt{max\_num\_batched\_tokens} 与 \texttt{prefill\_chunk\_size} 将长 prompt prefill 拆分为多轮 chunk，缓解 mixed-length workload 下长请求对 prefill token budget 的独占；当前实现仍保留 FCFS 默认路径，并明确区分调度收益与 kernel 优化。
    \item \textbf{Shared-prefix KV Late Merge：} 在 BlockManager 中实现 full-block 级 KV 去重原型，使用父 block hash + 当前 block token 构造链式 hash，hash 命中后进行 token 校验，并通过 block table 重定向与 ref-count 维护回收重复 physical KV blocks；该优化主要面向 shared-prefix workload，当前为单机单模型原型，尚未覆盖 eviction、LoRA/cache salt 与多 GPU 一致性。
    \item \textbf{CUDA Graph 安全回退：} 为 decode CUDA Graph 增加 runtime eligibility checks，覆盖 batch size、context length、block table width 与 capture window，不满足条件时回退 eager 并记录 fallback reason，提升长上下文和动态 shape 场景下 graph replay 的可诊断性。
\end{itemize}
```

更短版本：

```text
nano-VLLM 推理优化原型：实现 benchmark 矩阵、token-budget chunked prefill、shared-prefix full-block KV late merge 与 CUDA Graph eligibility/fallback；围绕 TTFT、prefill/decode latency、KV block 水位和 graph fallback 建立 A/B 指标体系，当前为单机单模型原型。
```

## 17. 当前仍未解决的问题

1. 当前仓库缺少原始 benchmark logs/jsonl，性能数字不可复核。
2. 缺少输出一致性测试，不能证明 late merge 与 graph fallback 对生成结果无影响。
3. 缺少 ref_count、block table redirect、hash collision、preemption 的单元测试。
4. 缺少 CUDA Graph fallback failure injection。
5. Fair prefill 没有 decode-first、SLO class、自适应 chunk size。
6. Prefix cache identity 没有纳入 model id、LoRA id、cache salt、多模态 hash。
7. 没有 eviction、LRU、cache capacity pressure 策略。
8. 没有多 GPU / TP 一致性验证。
9. 没有 Nsight Systems / Compute profiling，无法证明 kernel 或 memory bandwidth 级因果。

## 18. 下一步最有价值的改进方向

优先级建议：

1. 补证据：重新运行 matrix benchmark，保存 JSONL、stdout、环境信息和 commit id。
2. 补 correctness：输出一致性测试、late merge ref_count 单测、hash collision 人工构造测试。
3. 补 fallback：构造 context length 超 capture、block table width 超 buffer、uncaptured batch size 的 CUDA Graph fallback 测试。
4. 调度增强：decode-first 或 bounded-prefill policy，避免 waiting 长期非空时 decode starvation。
5. 自适应 chunk：根据 decode backlog、prefill step latency 和目标 TTFT 动态调整 chunk size。
6. Prefix cache 工业化：cache identity 加入 model/LoRA/cache salt，设计 eviction 和多 GPU 一致性。
7. Profiling：用 Nsight Systems 分离 CPU scheduler、kernel launch、attention kernel 和 KV write/read 时间。

## 19. 附录：代码位置索引

| 主题 | 文件 | 类/函数 | 当前行号 | 关键定位词 | 作用 |
| --- | --- | --- | ---: | --- | --- |
| LLM 入口 | `nanovllm/llm.py` | `LLM` | L4 | `class LLM(LLMEngine)` | API 入口 |
| 配置项 | `nanovllm/config.py` | `Config` | L6-L35 | `prefill_policy`、`enable_prefix_late_merge` | 新增优化开关 |
| Sequence 状态 | `nanovllm/engine/sequence.py` | `Sequence` | L18-L31 | `num_scheduled_tokens` | chunked prefill 进度 |
| block 数 | `nanovllm/engine/sequence.py` | `num_blocks` | L55-L57 | `block_size` | logical block 计算 |
| Engine 初始化 | `nanovllm/engine/llm_engine.py` | `__init__` | L18-L38 | `Scheduler`、`ModelRunner` | 组件连接 |
| 请求加入 | `nanovllm/engine/llm_engine.py` | `add_request` | L49-L55 | `_request_start_times` | 记录 TTFT 起点 |
| metrics 聚合 | `nanovllm/engine/llm_engine.py` | `metrics` | L69-L84 | `block_manager`、`model_runner` | 输出指标 |
| 单步执行 | `nanovllm/engine/llm_engine.py` | `step` | L86-L102 | `perf_counter` | step latency 和 TTFT |
| 调度入口 | `nanovllm/engine/scheduler.py` | `schedule` | L42-L51 | `prefill_policy` | 选择调度策略 |
| FCFS prefill | `nanovllm/engine/scheduler.py` | `_schedule_fcfs` | L53-L87 | `num_scheduled_tokens` | 原始路径保留 |
| Fair prefill | `nanovllm/engine/scheduler.py` | `_schedule_fair_prefill` | L89-L128 | `prefill_chunk_size` | token-budget chunking |
| Decode 调度 | `nanovllm/engine/scheduler.py` | `_schedule_decode` | L130-L150 | `can_append` | decode batch |
| preemption | `nanovllm/engine/scheduler.py` | `preempt` | L152-L157 | `preemptions` | decode 资源不足时抢占 |
| postprocess | `nanovllm/engine/scheduler.py` | `postprocess` | L159-L170 | `hash_blocks` | late merge 触发点 |
| Block 数据 | `nanovllm/engine/block_manager.py` | `Block` | L8-L23 | `ref_count`、`hash` | physical block 元数据 |
| BlockManager metrics | `nanovllm/engine/block_manager.py` | `reset_metrics` | L37-L48 | `late_merge_successes` | KV 指标 |
| 链式 hash | `nanovllm/engine/block_manager.py` | `compute_hash` | L53-L59 | `xxhash`、`prefix` | block hash |
| block 分配 | `nanovllm/engine/block_manager.py` | `_allocate_block` | L61-L71 | `hash_to_block_id` | 分配并清理旧映射 |
| prefix cache lookup | `nanovllm/engine/block_manager.py` | `can_allocate` | L79-L97 | `prefix_hits` | allocate-time 复用 |
| block allocate | `nanovllm/engine/block_manager.py` | `allocate` | L99-L116 | `num_cached_tokens` | 建立 block_table |
| block 释放 | `nanovllm/engine/block_manager.py` | `deallocate` | L118-L125 | `ref_count` | sequence 完成释放 |
| late merge 入口 | `nanovllm/engine/block_manager.py` | `hash_blocks` | L134-L147 | `enable_prefix_late_merge` | full block hash 与 merge |
| late merge 实现 | `nanovllm/engine/block_manager.py` | `_late_merge_block` | L149-L174 | `canonical_id` | block table redirect |
| model metrics | `nanovllm/engine/model_runner.py` | `reset_metrics` | L63-L68 | `graph_fallbacks` | graph 指标 |
| graph fallback | `nanovllm/engine/model_runner.py` | `_record_graph_fallback` | L70-L73 | `fallback_reason` | fallback reason |
| KV cache 分配 | `nanovllm/engine/model_runner.py` | `allocate_kv_cache` | L117-L151 | `num_kvcache_blocks` | 显存估算和 KV tensor |
| prefill 准备 | `nanovllm/engine/model_runner.py` | `prepare_prefill` | L165-L206 | `slot_mapping` | prefill 输入 |
| decode 准备 | `nanovllm/engine/model_runner.py` | `prepare_decode` | L208-L224 | `block_tables` | decode 输入 |
| graph eligibility | `nanovllm/engine/model_runner.py` | `run_model` | L231-L265 | `graph_replays` | replay/fallback |
| graph capture | `nanovllm/engine/model_runner.py` | `capture_cudagraph` | L275-L310 | `max_seq_len_to_capture` | 预捕获 decode graph |
| KV 写入 | `nanovllm/layers/attention.py` | `store_kvcache` | L33-L40 | `slot_mapping` | 写 physical KV |
| Attention | `nanovllm/layers/attention.py` | `Attention.forward` | L59-L75 | `flash_attn_with_kvcache` | prefill/decode attention |
| workload | `bench.py` | `make_*_workload` | L47-L104 | `shared_prefix` | 生成实验负载 |
| 单次 benchmark | `bench.py` | `run_once` | L117-L200 | `llm.reset_metrics` | 单次运行和 JSON summary |
| matrix variant | `bench.py` | `variant_args` | L203-L221 | `fair`、`late_merge` | A/B 配置 |
| 结果聚合 | `bench.py` | `summarize_results` | L231-L265 | `metric_paths` | repeat 汇总 |
| 子进程命令 | `bench.py` | `build_child_command` | L295-L333 | `--single-run-json` | 实验隔离 |
| 子进程运行 | `bench.py` | `run_repeated_isolated` | L351-L370 | `output_jsonl` | matrix/repeat 隔离 |
| CLI 参数 | `bench.py` | `parse_args` | L373-L403 | `--matrix` | benchmark 参数 |
| CLI 测试 | `tests/test_bench_cli.py` | `BenchCliTest` | L8-L49 | `parse_args` | 参数解析测试 |

## 20. 附录：Git commit 与功能对应关系

| Commit | 文件 | 功能 |
| --- | --- | --- |
| `9f06488` | `docs/superpowers/specs/...design.md` | 设计规格文档 |
| `ea73880` | `bench.py` | 初版 benchmark、workload、JSON 输出 |
| `ea73880` | `nanovllm/config.py` | 新增 prefill、late merge、capture 配置 |
| `ea73880` | `nanovllm/engine/scheduler.py` | fair chunked prefill |
| `ea73880` | `nanovllm/engine/block_manager.py` | late merge、block metrics |
| `ea73880` | `nanovllm/engine/model_runner.py` | CUDA Graph eligibility/fallback |
| `ea73880` | `nanovllm/engine/llm_engine.py` | metrics 聚合与 step latency |
| `e7a3d91` | `bench.py` | repeat、JSONL 输出 |
| `e7a3d91` | `nanovllm/engine/llm_engine.py` | exit 幂等性增强 |
| `e7a3d91` | `tests/test_bench_cli.py` | CLI 单测 |
| `07962aa` | `bench.py` | 子进程隔离 matrix/repeat |
| `07962aa` | `tests/test_bench_cli.py` | `--single-run-json` 单测 |

## 21. 附录：本次审计使用的命令

```bash
git status --short --branch
git remote -v
git branch -a
git log --oneline --decorate --graph --all --max-count=80
git tag
git diff --stat ffa7349..HEAD
git diff --name-status ffa7349..HEAD
git log --stat --oneline ffa7349..HEAD
git diff --stat ffa7349..HEAD -- nanovllm/engine/block_manager.py nanovllm/engine/scheduler.py nanovllm/engine/model_runner.py bench.py
rg --files -g '*.json' -g '*.jsonl' -g '*.csv' -g '*.log' -g '*.md' -g '*.sh' -g '*.py'
nl -ba bench.py
nl -ba nanovllm/config.py
nl -ba nanovllm/engine/sequence.py
nl -ba nanovllm/engine/llm_engine.py
nl -ba nanovllm/engine/scheduler.py
nl -ba nanovllm/engine/block_manager.py
nl -ba nanovllm/engine/model_runner.py
nl -ba nanovllm/layers/attention.py
nl -ba tests/test_bench_cli.py
```

## 22. 最终自检

| 检查项 | 结果 |
| --- | --- |
| 是否确定 baseline | 是，`ffa7349`，并说明选择理由 |
| 是否覆盖全部 commits | 是，覆盖 `9f06488..07962aa` |
| 是否区分功能修改与统计修改 | 是 |
| 是否有未提供代码证据的结论 | 有性能数字，但已明确标注当前仓库无法验证 |
| 是否虚构指标 | 否，未把缺失日志中的数字作为已复核事实 |
| 是否把相关性写成因果 | 否，decode P95 等只写机制可能性 |
| 是否分析 abnormal path | 是，覆盖 collision、partial block、preemption、abort、OOM/graph fallback 等 |
| 是否分析优化代价 | 是 |
| 是否说明 benchmark 局限 | 是 |
| 是否给出可口述面试答案 | 是，共 36 个问题 |
| 是否仅生成文档 | 是，本次未修改业务代码 |

