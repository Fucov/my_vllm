# nano-vLLM Inference Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement measurable nano-vLLM inference optimizations in the order Benchmark & Metrics, Fair Chunked Prefill, Prefix Cache Late Merge, and Safe CUDA Graph.

**Architecture:** Add flag-gated runtime behavior and lightweight metrics to the existing compact engine. Keep baseline behavior available, expose metrics through `LLMEngine`, and use `bench.py` as the reproducible A/B harness.

**Tech Stack:** Python, PyTorch, CUDA graph, nano-vLLM scheduler/block manager/model runner.

---

### Task 1: Benchmark & Metrics

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/engine/block_manager.py`
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `nanovllm/engine/llm_engine.py`
- Modify: `bench.py`

- [x] Add config flags for optimization policy and benchmark metric toggles.
- [x] Add scheduler, block manager, and model runner metrics dictionaries.
- [x] Add `LLMEngine.metrics()` and `LLMEngine.reset_metrics()`.
- [x] Replace the one-off benchmark with a CLI supporting `random`, `mixed_lengths`, and `shared_prefix`.
- [x] Verify `python -m py_compile bench.py nanovllm/config.py nanovllm/engine/*.py`.

### Task 2: Fair Chunked Prefill

**Files:**
- Modify: `nanovllm/engine/scheduler.py`
- Modify: `nanovllm/config.py`
- Modify: `bench.py`

- [x] Add `prefill_policy` and `prefill_chunk_size` config fields.
- [x] Preserve current FCFS behavior under `prefill_policy="fcfs"`.
- [x] Implement fair prefill under `prefill_policy="fair"` by scanning waiting requests, slicing prefill work by chunk size, and keeping decode unchanged.
- [x] Verify scheduled prefill tokens never exceed `max_num_batched_tokens`.
- [x] Verify compile checks pass.

### Task 3: Prefix Cache Late Merge

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/engine/block_manager.py`
- Modify: `nanovllm/engine/scheduler.py`

- [x] Add `enable_prefix_late_merge` config field.
- [x] Track prefix cache probes, hits, and misses.
- [x] Add full-block canonicalization after hashing newly completed full blocks.
- [x] Merge only when hash and exact token IDs match.
- [x] Preserve ref-count and used/free block invariants.
- [x] Verify compile checks pass.

### Task 4: Safe CUDA Graph

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `bench.py`

- [x] Add `max_seq_len_to_capture` config field.
- [x] Add graph replay eligibility checks.
- [x] Add eager fallback counters and fallback reasons.
- [x] Clear graph block table buffers before replay.
- [x] Verify compile checks pass.

### Task 5: Final Verification

**Files:**
- Read/Run: repository root

- [x] Run `python -m py_compile bench.py nanovllm/config.py nanovllm/engine/*.py`.
- [x] Run a tiny CPU-free syntax/config smoke where possible without loading the model.
- [x] Summarize exact commands and any benchmark commands for the server.
