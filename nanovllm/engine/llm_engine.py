import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.reset_metrics()
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self._request_start_times[seq.seq_id] = perf_counter()
        self.scheduler.add(seq)
        return seq.seq_id

    def reset_metrics(self):
        self.scheduler.reset_metrics()
        self.model_runner.call("reset_metrics")
        self._request_start_times = {}
        self._first_token_seq_ids = set()
        self._engine_metrics = {
            "prefill_step_latencies": [],
            "decode_step_latencies": [],
            "ttft_latencies": [],
            "completed_requests": 0,
        }

    def metrics(self):
        cuda_metrics = {}
        if torch.cuda.is_available():
            cuda_metrics = {
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved(),
                "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
                "cuda_max_memory_reserved": torch.cuda.max_memory_reserved(),
            }
        return {
            "engine": self._engine_metrics,
            "scheduler": self.scheduler.metrics,
            "block_manager": self.scheduler.block_manager.metrics,
            "model_runner": self.model_runner.metrics,
            "cuda": cuda_metrics,
        }

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        t = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        latency = perf_counter() - t
        key = "prefill_step_latencies" if is_prefill else "decode_step_latencies"
        self._engine_metrics[key].append(latency)
        now = perf_counter()
        for seq in seqs:
            if seq.seq_id not in self._first_token_seq_ids and seq.num_completion_tokens > 0:
                self._first_token_seq_ids.add(seq.seq_id)
                self._engine_metrics["ttft_latencies"].append(now - self._request_start_times[seq.seq_id])
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        self._engine_metrics["completed_requests"] += len(outputs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
