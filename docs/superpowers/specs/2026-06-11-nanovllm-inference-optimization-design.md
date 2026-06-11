# nano-vLLM Inference Optimization Design

Date: 2026-06-11

## Goal

Build a credible AI Infra resume project on top of this nano-vLLM codebase by adding measurable inference optimizations in a conservative order:

1. Benchmark & Metrics
2. Fair Chunked Prefill
3. Prefix Cache Late Merge
4. Safe CUDA Graph

The project must produce A/B data that can survive interview questions. The primary claims should be about measurable latency, KV cache efficiency, and fast-path stability, not vague "faster inference" claims.

Target machine:

- Single NVIDIA RTX PRO 4000 Blackwell GPU with 24GB VRAM
- Intel i9-14900, 32 logical CPUs
- 188GiB RAM
- Single NUMA node
- NVMe storage
- No Nsight installed at project start

## Non-Goals

- Do not implement CPU/NVMe KV offload in this phase.
- Do not change attention kernels or add FlashInfer in this phase.
- Do not add new model families.
- Do not claim model quality improvements.
- Do not optimize for multi-GPU tensor parallelism in this phase.

These are valid later extensions, but they would dilute the first measurable milestone.

## Success Metrics

The benchmark suite must report these metrics for every run:

- End-to-end output throughput in tokens/s
- Time to first token, including P50, P90, P95, and P99
- Decode step latency, including P50, P90, P95, and P99
- Prefill step latency
- Peak CUDA allocated and reserved memory
- KV block allocation high-water mark
- Prefix cache probe count, hit count, miss count, and hit rate
- Late-merge attempts, successful merges, and reclaimed blocks
- CUDA graph replay count, fallback count, and fallback reasons

Target outcomes:

- Mixed long/short prompt workload: reduce P95 TTFT by at least 10%.
- Shared-prefix workload: reduce peak KV block usage by at least 10%.
- Random workload: avoid meaningful throughput regression; target no worse than 3% regression.
- Safe CUDA Graph: zero shape mismatch failures, with explicit fallback counters.

The benchmark should also print enough run metadata to make the results reproducible: seed, model path, workload type, request count, prompt length distribution, max output length distribution, max batched tokens, max sequences, block size, eager/graph mode, and git commit if available.

## Step 1: Benchmark & Metrics

### Design

Create a benchmark harness before changing runtime behavior. This avoids measuring a moving target and makes every later optimization testable.

The harness should support three workloads:

- `random`: matches the current `bench.py` style with random prompt and output lengths.
- `mixed_lengths`: combines many short prompts, some medium prompts, and a smaller number of long prompts to expose head-of-line blocking in prefill.
- `shared_prefix`: generates many prompts with a shared system prefix and divergent suffixes to expose prefix cache behavior.

The harness should support at least two modes:

- `baseline`: current behavior, with optimization flags disabled.
- `optimized`: selected optimization flags enabled.

For implementation simplicity, the first version can extend or replace `bench.py` with CLI flags. It can use synthetic token IDs instead of text prompts, because this project measures engine behavior, scheduler behavior, and KV cache behavior.

### Required Runtime Instrumentation

Add lightweight counters to the engine:

- Scheduler metrics: number of prefill batches, decode batches, scheduled prefill tokens, scheduled decode tokens, and per-sequence first-token timestamps.
- Block manager metrics: allocations, deallocations, peak used blocks, prefix probes, prefix hits, prefix misses, late-merge attempts, late-merge successes, and reclaimed duplicate blocks.
- Model runner metrics: graph replays, graph fallbacks, and fallback reason counts.

These counters should be low overhead and easy to reset between benchmark runs.

### Interview Defense

Question: How do you prove the improvement is not workload fabrication?

Answer: Run all changes against three workloads. `random` protects general throughput, `mixed_lengths` targets scheduler fairness, and `shared_prefix` targets cache reuse. Each optimization must improve the workload it is designed for without materially regressing the neutral workload. The benchmark uses fixed seeds and prints workload distributions, so the result is reproducible.

## Step 2: Fair Chunked Prefill

### Current Problem

`Scheduler.schedule()` currently lets only the first sequence receive chunked prefill when remaining token budget is smaller than the next prompt. If a long prompt sits at the front, shorter prompts behind it can wait even though scheduling a small slice for them would improve TTFT.

This is a classic head-of-line blocking problem.

### Design

Introduce an optional fair prefill policy controlled by config, for example:

- `prefill_policy="fcfs"` for current behavior
- `prefill_policy="fair"` for the new behavior
- `prefill_chunk_size`, defaulting to a conservative value such as 512 or 1024 tokens

The fair policy should:

1. Admit multiple waiting sequences into one prefill batch when budget allows.
2. Cap each scheduled prefill slice by `prefill_chunk_size`.
3. Prefer completing short prompts when they fit within the remaining budget.
4. Keep admission safe by preserving `BlockManager.can_allocate()` checks.
5. Keep the decode path unchanged.

A practical first algorithm:

- Scan the waiting queue while there is budget and capacity.
- For each sequence, allocate blocks if needed.
- Schedule `min(remaining_prompt_tokens, prefill_chunk_size, remaining_budget)`.
- If a sequence finishes prefill, move it to running.
- If it does not finish, keep it waiting with updated `num_cached_tokens` after postprocess.

The policy should avoid repeatedly scheduling only the same long sequence. A simple rotating deque is sufficient for the first version.

### Why It Should Not Lower Throughput

Fair chunked prefill does not reduce the global prefill token budget. It changes how the budget is divided among waiting requests. On a saturated batch, the GPU should still process roughly the same number of prefill tokens. The intended gain is lower TTFT for short and medium prompts, with similar total throughput.

There can be a small throughput cost if more fragmented prefill batches increase overhead. That is why the design includes:

- A bounded chunk size rather than tiny slices.
- A random workload guardrail with at most 3% allowed throughput regression.
- Metrics for prefill step time and scheduled tokens per step.

### Interview Defense

Question: Why does fair chunked prefill not reduce throughput?

Answer: It keeps the same token budget and batch size constraints, so the GPU still receives large prefill batches. The optimization redistributes prefill tokens to reduce head-of-line blocking. The benchmark verifies this by checking random workload throughput and scheduled prefill tokens per step.

## Step 3: Prefix Cache Late Merge

### Current Problem

`BlockManager.hash_blocks()` maps a full block hash to one physical block ID. If multiple sequences in the same scheduling step produce identical full blocks, they may occupy separate physical blocks before the hash table canonicalizes future requests. This wastes KV blocks and weakens prefix cache results in shared-prefix workloads.

### Design

Add an optional late-merge pass after full-block hashing:

1. Hash newly completed full blocks as today.
2. Detect when a newly hashed block has the same hash and exact token IDs as an existing canonical block.
3. Redirect the sequence block table from the duplicate block to the canonical block.
4. Increment the canonical block ref count.
5. Decrement and reclaim the duplicate block when its ref count reaches zero.

The exact-token check is required even when hashes match. This keeps the optimization safe against hash collisions.

The first implementation should merge only full blocks. Partial-block COW is out of scope for this phase because it has more correctness risk and more subtle write-sharing behavior.

### Correctness Rules

- Merge only blocks with identical hash and identical `token_ids`.
- Merge only full blocks produced by prompt/prefill hashing, not mutable partial decode tails.
- Never merge a block with itself.
- Preserve `ref_count` invariants.
- Reclaim a duplicate block only when its `ref_count` reaches zero.
- Keep the block table length unchanged; only physical block IDs change.
- Add internal assertions or tests for ref counts and used/free block consistency.

### Interview Defense

Question: How does Prefix Late Merge guarantee it does not incorrectly share KV?

Answer: It requires both hash equality and exact token equality. The hash is only a fast lookup key. The actual correctness check is token sequence equality on full immutable blocks. It does not merge partial mutable blocks, so later decode writes cannot corrupt another sequence's prefix.

## Step 4: Safe CUDA Graph

### Current Problem

The decode graph fast path assumes the runtime batch and block table shape fit the captured graph buffers. If a runtime sequence needs a wider block table than the graph buffer, replay setup can fail or write invalid shapes.

### Design

Add runtime eligibility checks before graph replay:

- Graph mode must be enabled.
- The call must be decode, not prefill.
- Batch size must fit a captured graph bucket.
- Runtime `context.block_tables.size(1)` must fit the captured graph block table width.
- Runtime context length must not exceed the graph capture length.

If any check fails, run the eager decode path and increment a fallback counter with a reason.

Also clear graph block table buffers before replay or fill unused entries with `-1`, so stale block IDs from previous larger batches cannot leak into smaller batches.

Optional config:

- `max_seq_len_to_capture`, defaulting to `max_model_len`

### Why It Has Value

Safe CUDA Graph is not mainly a throughput feature in this project. It is a stability and fast-path correctness feature. It lets the engine keep using CUDA graph when eligible while falling back cleanly for long-context edge cases. That is the behavior expected in a production inference engine.

### Interview Defense

Question: Why is Safe CUDA Graph valuable if it can fall back to eager?

Answer: CUDA graph is a fast path, not a correctness requirement. The safe path preserves graph replay for common decode batches and prevents shape-related crashes for out-of-envelope batches. The fallback counter proves how often the system leaves the fast path and why.

## Implementation Boundaries

Expected touched files:

- `nanovllm/config.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/llm_engine.py`
- `bench.py` or a new benchmark script

Keep changes small and flag-gated:

- Existing behavior remains available through baseline config.
- Optimized behavior is enabled explicitly by benchmark flags.
- Counters should be resettable and printable.

## Implementation Milestones

Milestone 1: Benchmark & Metrics

- Add benchmark CLI and synthetic workloads.
- Add runtime metrics counters.
- Validate that baseline runs produce stable JSON metrics.
- Acceptance: `random`, `mixed_lengths`, and `shared_prefix` workloads run without changing optimization behavior.

Milestone 2: Fair Chunked Prefill

- Add config flags for prefill policy and chunk size.
- Implement fair prefill scheduling behind the new flag.
- Compare baseline vs fair policy on `mixed_lengths` and `random`.
- Acceptance: P95 TTFT improves on `mixed_lengths`; `random` throughput remains within the regression budget.

Milestone 3: Prefix Cache Late Merge

- Add block manager metrics for prefix probes and late merge.
- Implement full-block duplicate detection and canonical block redirection.
- Add invariant checks for block ref counts and used/free pools.
- Compare baseline vs late merge on `shared_prefix`.
- Acceptance: peak used KV blocks decreases on `shared_prefix`, with no output or ref-count correctness failures.

Milestone 4: Safe CUDA Graph

- Add graph eligibility checks and fallback reasons.
- Clear graph buffers before replay.
- Add config for graph capture sequence length if needed.
- Compare graph replay/fallback behavior on random and long-context workloads.
- Acceptance: no shape mismatch failures; graph replay remains active for eligible decode batches.

## Testing Strategy

Unit-level tests or smoke checks:

- Fair scheduler preserves sequence completion and does not exceed token budget.
- Late merge preserves block contents and ref counts.
- Safe graph falls back when a fake runtime block table is wider than capture capacity.

Benchmark tests:

- Run `random` baseline vs optimized.
- Run `mixed_lengths` baseline vs optimized.
- Run `shared_prefix` baseline vs optimized.

Recommended repeated runs:

- At least 5 repetitions during development.
- At least 20 repetitions for final reported numbers if runtime allows.

Statistical reporting:

- Print per-run metrics as JSON lines or a JSON summary.
- Report mean and median.
- Report percent change from baseline.
- Keep raw output files under a benchmark results directory that is not committed unless explicitly requested.

## Resume Narrative

Project summary:

Implemented measurable inference optimizations in a compact vLLM-style engine: a reproducible benchmark harness, fair chunked prefill scheduling, prefix-cache duplicate block merging, and safe CUDA graph replay guards.

Defensible claims:

- Reduced P95 TTFT in mixed long/short prompt workloads by improving prefill fairness.
- Reduced peak KV block usage in shared-prefix workloads through full-block late merge.
- Preserved random workload throughput within a defined regression budget.
- Added graph replay eligibility and fallback metrics to stabilize the CUDA graph fast path.

The story is intentionally systems-oriented: measure first, optimize scheduler behavior, improve KV cache sharing, then harden the fast path.
