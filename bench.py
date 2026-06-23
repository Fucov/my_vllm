import argparse
import json
import os
import random
import copy
import sys
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

# 随机 prompt 长度和输出长度
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

# 60% short、30% medium、10% long 的混合长度 prompt
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

# 多请求共享固定 prefix，再拼接随机 suffix
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

# warmup、reset metrics、加入所有请求、循环 step，输出 JSON summary
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

    llm = None
    try:
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
    finally:
        if llm is not None:
            llm.exit()

    total_output_tokens = sum(len(tokens) for tokens in outputs.values())
    engine_metrics = metrics["engine"]
    prompt_lens = [len(prompt) for prompt in prompts]
    output_lens = [params.max_tokens for params in sampling_params]
    summary = {
        "metadata": {
            "git_commit": git_commit(),
            "mode": args.mode,
            "run_label": getattr(args, "run_label", args.mode),
            "repeat_index": getattr(args, "repeat_index", 0),
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


def variant_args(args: argparse.Namespace, label: str, seed: int, repeat_index: int) -> argparse.Namespace:
    variant = copy.copy(args)
    variant.seed = seed
    variant.repeat_index = repeat_index
    variant.run_label = label
    variant.mode = "baseline"
    variant.prefill_policy = "fcfs"
    variant.enable_prefix_late_merge = False
    if label == "fair":
        variant.prefill_policy = "fair"
    elif label == "late_merge":
        variant.enable_prefix_late_merge = True
    elif label == "optimized":
        variant.mode = "optimized"
        variant.prefill_policy = "fair"
        variant.enable_prefix_late_merge = True
    elif label != "baseline":
        raise ValueError(f"unknown matrix label: {label}")
    return variant


def metric_path(result: dict, path: str) -> float:
    value = result
    for part in path.split("."):
        value = value[part]
    return float(value)

# 聚合 throughput、TTFT、prefill/decode、KV block、prefix、graph 指标
def summarize_results(results: list[dict]) -> dict:
    metric_paths = {
        "throughput_tok_s": "summary.throughput_tok_s",
        "ttft_p50_s": "summary.ttft_s.p50",
        "ttft_p95_s": "summary.ttft_s.p95",
        "ttft_p99_s": "summary.ttft_s.p99",
        "prefill_mean_s": "summary.prefill_step_s.mean",
        "decode_p95_s": "summary.decode_step_s.p95",
        "peak_used_blocks": "metrics.block_manager.peak_used_blocks",
        "prefix_hits": "metrics.block_manager.prefix_hits",
        "prefix_misses": "metrics.block_manager.prefix_misses",
        "late_merge_successes": "metrics.block_manager.late_merge_successes",
        "late_merge_reclaimed_blocks": "metrics.block_manager.late_merge_reclaimed_blocks",
        "graph_replays": "metrics.model_runner.graph_replays",
        "graph_fallbacks": "metrics.model_runner.graph_fallbacks",
    }
    grouped: dict[str, list[dict]] = {}
    for result in results:
        label = result["metadata"].get("run_label", result["metadata"]["mode"])
        grouped.setdefault(label, []).append(result)

    summary = {}
    for label, items in grouped.items():
        label_summary = {"runs": len(items)}
        for name, path in metric_paths.items():
            values = [metric_path(item, path) for item in items]
            label_summary[name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        summary[label] = label_summary
    return summary


def run_repeated(args: argparse.Namespace) -> list[dict]:
    labels = ("baseline", "fair", "late_merge", "optimized") if args.matrix else (getattr(args, "run_label", args.mode),)
    results = []
    jsonl_file = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        jsonl_file = open(args.output_jsonl, "w")
    try:
        for repeat_index in range(args.repeat):
            seed = args.seed + repeat_index
            for label in labels:
                run_args = variant_args(args, label, seed, repeat_index) if args.matrix else copy.copy(args)
                if not args.matrix:
                    run_args.seed = seed
                    run_args.repeat_index = repeat_index
                    run_args.run_label = label
                result = run_once(run_args)
                results.append(result)
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(result, sort_keys=True) + "\n")
                    jsonl_file.flush()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
    return results


def build_child_command(args: argparse.Namespace, label: str, seed: int, repeat_index: int) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-run-json",
        "--model", args.model,
        "--mode", "baseline",
        "--workload", args.workload,
        "--seed", str(seed),
        "--num-seqs", str(args.num_seqs),
        "--vocab-size", str(args.vocab_size),
        "--min-input-len", str(args.min_input_len),
        "--max-input-len", str(args.max_input_len),
        "--min-output-len", str(args.min_output_len),
        "--max-output-len", str(args.max_output_len),
        "--shared-prefix-len", str(args.shared_prefix_len),
        "--temperature", str(args.temperature),
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kvcache-block-size", str(args.kvcache_block_size),
        "--prefill-chunk-size", str(args.prefill_chunk_size),
        "--run-label", label,
        "--repeat-index", str(repeat_index),
    ]
    if args.max_seq_len_to_capture is not None:
        cmd.extend(["--max-seq-len-to-capture", str(args.max_seq_len_to_capture)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if label == "fair":
        cmd.extend(["--prefill-policy", "fair"])
    elif label == "late_merge":
        cmd.append("--enable-prefix-late-merge")
    elif label == "optimized":
        cmd.extend(["--mode", "optimized"])
    else:
        cmd.extend(["--prefill-policy", "fcfs"])
    return cmd


def run_child_command(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"child benchmark failed with exit code {completed.returncode}")
    text = completed.stdout
    start = text.find("{")
    if start == -1:
        raise RuntimeError("child benchmark did not print JSON")
    return json.loads(text[start:])


def run_repeated_isolated(args: argparse.Namespace) -> list[dict]:
    labels = ("baseline", "fair", "late_merge", "optimized") if args.matrix else (args.run_label,)
    results = []
    jsonl_file = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        jsonl_file = open(args.output_jsonl, "w")
    try:
        for repeat_index in range(args.repeat):
            seed = args.seed + repeat_index
            for label in labels:
                result = run_child_command(build_child_command(args, label, seed, repeat_index))
                results.append(result)
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(result, sort_keys=True) + "\n")
                    jsonl_file.flush()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
    return results

# 增加 workload、matrix、repeat、output-jsonl、prefill policy、late merge 等 CLI 参数
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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--single-run-json", action="store_true")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--repeat-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.run_label is None:
        args.run_label = args.mode
    if args.single_run_json:
        print(json.dumps(run_once(args), indent=2, sort_keys=True))
        return
    results = run_repeated_isolated(args) if args.matrix or args.repeat > 1 or args.output_jsonl else run_repeated(args)
    if len(results) == 1 and not args.matrix:
        print(json.dumps(results[0], indent=2, sort_keys=True))
        return
    output = {
        "metadata": {
            "workload": args.workload,
            "repeat": args.repeat,
            "matrix": args.matrix,
            "output_jsonl": args.output_jsonl,
            "seed_start": args.seed,
            "num_seqs": args.num_seqs,
            "model": os.path.expanduser(args.model),
            "git_commit": git_commit(),
        },
        "summary": summarize_results(results),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
