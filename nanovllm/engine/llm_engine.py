import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
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
        atexit.register(self.exit)

    def exit(self):
        # 退出时通知所有子进程退出,并等待它们结束。
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # add_request 的意思是把用户的输入 prompt 和采样参数封装成一个 Sequence 对象, 然后放到 Scheduler 的 waiting 队列里等待被 schedule。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    """
    四种scheduler调度模式：

    1. vLLM 混合批 / Chunked Prefill:
    在同一个 worker 内采用 decode-priority 调度：优先调度已有请求的 decode token，
    再用剩余 token budget 调度 prefill；长 prompt 会被切成 prefill chunk。
    因此同一轮 forward 里可以同时包含 decode seq 和 prefill chunk。
    目标是利用 prefill compute-bound、decode memory-bound 的互补性，
    提高 GPU 利用率，并改善 decode 的 ITL/TPOT。

    2. nano-vLLM 分割批:
    为了实现简单，scheduler 每个 step 只返回一种 phase：
    要么是 prefill batch，要么是 decode batch。
    同一个 forward batch 内不会混合 prefill 和 decode。
    它不是严格“所有请求 prefill 完才 decode”，而是 batch/step 级别的 phase separation。
    优点是逻辑清晰、调度和模型执行路径简单；缺点是 GPU 利用率和尾延迟优化空间有限。

    3. SGLang Chunked Prefill + Mixed Batch:
    SGLang 支持 chunked prefill；开启 --enable-mixed-chunk 后，
    允许在 chunked prefill 场景下把 prefill chunk 和 decode 请求混合到同一个 batch。
    它和 vLLM / Sarathi 的思想类似：用受限大小的 prefill chunk 补足 decode batch，
    避免长 prompt prefill 阻塞 decode，同时提高 GPU 利用率。
    但不是“每个 step 必定都有 decode”，而是有 decode 请求且调度预算允许时才混合。

    4. SGLang PD 分离:
    将 prefill 和 decode 拆到不同的物理 GPU / worker / server 实例上运行。
    Prefill worker 专门处理 prompt，生成 KV cache；Decode worker 专门执行逐 token 生成。
    二者通过 KV cache transfer / connector / 通信后端传递 KV cache。
    目标不是继续混合执行，而是隔离 prefill 对 decode 的干扰，
    分别优化 TTFT、ITL/TPOT、p95/p99 延迟和资源配比。
    """
    def step(self):
        # step 的意思是执行 Scheduler 里已经 schedule 好的 Sequence, 包括调用 ModelRunner 来运行模型, 然后把生成的 token_ids 传回 Scheduler 进行后处理。
        # 它会返回一个列表, 每个元素是一个 tuple, 包含 seq_id 和对应的生成 token_ids 列表。
        # 和generate函数的区别在于, step函数是 generate函数内部调用的一个子函数, 用来执行每一步的生成逻辑; 而 generate函数是对外暴露的接口, 用来接收用户输入并返回生成结果的。
        # 执行步骤如下:
        # 1. 调用 scheduler.schedule() 来选出本步要处理的 Sequence 列表和一个布尔值 is_prefill 来表示当前阶段是 prefill 还是 decode。
        # 2. 根据 is_prefill 来统计本步计划写入 KV Cache 的 Token 数量 num_tokens, 这个数量在 prefill 阶段是正数, 在 decode 阶段是负数(因为 decode 阶段的 token_ids 是本步实际生成的 Token ID 列表,而不是计划写入 KV Cache 的 Token ID 列表)。
        # 3. 调用 model_runner.call("run", seqs, is_prefill) 来执行模型的 forward, 得到本步生成的 token_ids 列表。
        # 4. 调用 scheduler.postprocess(seqs, token_ids, is_prefill) 来让 Scheduler 根据生成的 token_ids 来更新 Sequence 的状态和 KV Cache 的分配情况。
        # 5. 从 seqs 中筛选出已经完成的 Sequence, 把它们的 seq_id 和 completion_token_ids 组成一个列表返回。
        # 这个 step 函数在 generate 的主循环里被不断调用, 直到 Scheduler 里所有 Sequence 都完成了。

        # prefill和decode阶段在attention内核区别如下：
        # 1. prefill阶段的输入是prompt tokens(或者切片seq长度), decode阶段的输入是已经生成的tokens（长度1）。两者k:q比值不同。
        # 2. prefill阶段的输入会写入KV Cache, decode阶段的输入不写入KV Cache？（因为decode阶段的输入tokens已经在之前的step写入过KV Cache了, 现在生成的新token才需要写入KV Cache）
        # 3. KV来源不同：prefill阶段的KV来自于当前step的输入tokens经过attention计算得到的KV, decode阶段的KV来自于之前step写入KV Cache的tokens经过attention计算得到的KV。
        seqs, is_prefill = self.scheduler.schedule() # 选出待处理的序列
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs) 
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        # is_finished 的意思是判断 Scheduler 里是否还有未完成的 Sequence, 也就是 waiting 和 running 队列是否都空了。这个函数在 generate 的主循环里被调用, 用来决定什么时候结束生成。
        # 和 Scheduler 的 is_finished 不同, LLMEngine 的 is_finished 是对外暴露的接口, 让用户可以在 generate 之外的地方也能检查生成是否完成。
        # 和 exit 函数区别在于, exit函数是当用户强制退出程序时被调用的, 用来清理资源和通知子进程退出的; 而 is_finished 是在正常的生成流程中被调用的, 用来检查生成是否完成的。
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # generate 的意思是根据用户输入的 prompts 和 sampling_params 来生成文本。它会先把 prompts 和 sampling_params 传给 add_request 来创建 Sequence 并放到 Scheduler 里, 然后进入一个循环不断调用 step 来执行生成, 直到 Scheduler 里所有 Sequence 都完成了。它还会使用 tqdm 来显示生成的进度和速度。
        # 执行步骤如下:
        # 1. 使用 tqdm 来创建一个进度条, total 设置为 prompts 的长度, desc 设置为 "Generating", dynamic_ncols=True 来让进度条根据终端宽度自动调整, disable 设置为 not use_tqdm 来根据参数决定是否显示进度条。
        # 2. 如果 sampling_params 不是一个列表, 就把它扩展成一个和 prompts 长度相同的列表, 这样每个 prompt 都有对应的采样参数。
        # 3. 使用 zip 来同时遍历 prompts 和 sampling_params, 对每个 prompt 和对应的采样参数调用 add_request 来把它们添加到 Scheduler 里。
        # 4. 创建一个空字典 outputs 来存储生成的结果, 初始化 prefill_throughput 和 decode_throughput 为 0。
        # 5. 进入一个循环, 条件是 not self.is_finished(),也就是 Scheduler 里还有未完成的 Sequence。
        # 6. 在循环里,先记录当前时间 t, 然后调用 step 来执行生成, 得到本步的输出 output 和本步计划写入 KV Cache 的 Token 数量 num_tokens。
        # 7. 根据 num_tokens 的正负来计算 prefill_throughput 和 decode_throughput, prefill_throughput 是 num_tokens 除以本步执行的时间, decode_throughput 是 -num_tokens 除以本步执行的时间(因为 decode 阶段的 num_tokens 是负数)。
        # 8. 使用 pbar.set_postfix 来更新进度条的后缀信息, 显示当前的 prefill 和 decode 速度。
        # 9. 从 step 的输出 output 中遍历每个 seq_id 和对应的 token_ids, 把它们存到 outputs 字典里, 并且调用 pbar.update(1) 来更新进度条。
        # 10. 循环结束后, 调用 pbar.close() 来关闭进度条, 然后把 outputs 字典里的结果按照 seq_id 的顺序取出来, 解码成文本, 最后返回一个列表, 每个元素是一个字典, 包含生成的文本和对应的 token_ids。

        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter() # 记录当前时间, 用来计算本步的执行时间和吞吐量
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
