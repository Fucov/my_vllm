import argparse
import json
import os
import random
import statistics
import subprocess
import time

import torch

from nanovllm import LLM, SamplingParams


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_random_workload(args: argparse.Namespace) -> tuple[list[list[int]], list[SamplingParams]]:
    prompts = [
        [random.randint(0, args.vocab_size - 1) for _ in range(random.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=random.randint(args.min_output_len, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return prompts, sampling_params


def make_mixed_lengths_workload(args: argparse.Namespace) -> tuple[list[list[int]], list[SamplingParams]]:
    prompts = []
    short_count = int(args.num_seqs * 0.6)
    medium_count = int(args.num_seqs * 0.3)
    long_count = args.num_seqs - short_count - medium_count
    buckets = (
        (short_count, 64, min(512, args.max_model_len - 1)),
        (medium_count, 768, min(1536, args.max_model_len - 1)),
        (long_count, 2048, min(args.max_input_len, args.max_model_len - 1)),
    )
    for count, low, high in buckets:
        low = min(low, high)
        for _ in range(count):
            prompts.append([random.randint(0, args.vocab_size - 1) for _ in range(random.randint(low, high))])
    random.shuffle(prompts)
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=random.randint(args.min_output_len, args.max_output_len))
        for _ in prompts
    ]
    return prompts, sampling_params


def make_shared_prefix_workload(args: argparse.Namespace) -> tuple[list[list[int]], list[SamplingParams]]:
    prefix_len = min(args.shared_prefix_len, args.max_model_len - 2)
    suffix_max = max(1, min(args.max_input_len - prefix_len, args.max_model_len - prefix_len - 1))
    shared_prefix = [random.randint(0, args.vocab_size - 1) for _ in range(prefix_len)]
    prompts = []
    for _ in range(args.num_seqs):
        suffix_len = random.randint(1, suffix_max)
        suffix = [random.randint(0, args.vocab_size - 1) for _ in range(suffix_len)]
        prompts.append(shared_prefix + suffix)
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=random.randint(args.min_output_len, args.max_output_len))
        for _ in prompts
    ]
    return prompts, sampling_params


def make_workload(args: argparse.Namespace) -> tuple[list[list[int]], list[SamplingParams]]:
    if args.workload == "random":
        return make_random_workload(args)
    if args.workload == "mixed_lengths":
        return make_mixed_lengths_workload(args)
    if args.workload == "shared_prefix":
        return make_shared_prefix_workload(args)
    raise ValueError(f"unknown workload: {args.workload}")


def length_summary(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": percentile([float(v) for v in values], 50),
        "p95": percentile([float(v) for v in values], 95),
    }


def run_once(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompts, sampling_params = make_workload(args)
    model_path = os.path.expanduser(args.model)

    prefill_policy = args.prefill_policy
    enable_prefix_late_merge = args.enable_prefix_late_merge
    if args.mode == "optimized":
        prefill_policy = "fair"
        enable_prefix_late_merge = True

    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        prefill_policy=prefill_policy,
        prefill_chunk_size=args.prefill_chunk_size,
        enable_prefix_late_merge=enable_prefix_late_merge,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
    )
    llm.generate(["Benchmark warmup"], SamplingParams(ignore_eos=True, max_tokens=1), use_tqdm=False)
    llm.reset_metrics()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for prompt, params in zip(prompts, sampling_params):
        llm.add_request(prompt, params)
    outputs = {}
    while not llm.is_finished():
        output, _ = llm.step()
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
    elapsed = time.perf_counter() - start
    metrics = llm.metrics()

    total_output_tokens = sum(len(tokens) for tokens in outputs.values())
    engine_metrics = metrics["engine"]
    prompt_lens = [len(prompt) for prompt in prompts]
    output_lens = [params.max_tokens for params in sampling_params]
    summary = {
        "metadata": {
            "git_commit": git_commit(),
            "mode": args.mode,
            "workload": args.workload,
            "seed": args.seed,
            "model": model_path,
            "num_seqs": args.num_seqs,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "kvcache_block_size": args.kvcache_block_size,
            "prefill_policy": prefill_policy,
            "prefill_chunk_size": args.prefill_chunk_size,
            "enable_prefix_late_merge": enable_prefix_late_merge,
            "enforce_eager": args.enforce_eager,
            "max_seq_len_to_capture": args.max_seq_len_to_capture,
            "prompt_lengths": length_summary(prompt_lens),
            "requested_output_lengths": length_summary(output_lens),
        },
        "summary": {
            "elapsed_s": elapsed,
            "total_output_tokens": total_output_tokens,
            "throughput_tok_s": total_output_tokens / elapsed,
            "completed_requests": len(outputs),
            "ttft_s": summarize_latencies(engine_metrics["ttft_latencies"]),
            "prefill_step_s": summarize_latencies(engine_metrics["prefill_step_latencies"]),
            "decode_step_s": summarize_latencies(engine_metrics["decode_step_latencies"]),
        },
        "metrics": metrics,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-vLLM benchmark with reproducible workloads and runtime metrics.")
    parser.add_argument("--model", default="~/models/Qwen3-0.6B/")
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--workload", choices=("random", "mixed_lengths", "shared_prefix"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--shared-prefix-len", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--prefill-policy", choices=("fcfs", "fair"), default="fcfs")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--enable-prefix-late-merge", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_once(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
