from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256          # 类变量，由 LLMEngine 启动时根据 Config 改写
    counter = count()         # 自增 seq_id 来源

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # —— 用户请求载体 ——
        self.seq_id = next(Sequence.counter) # 全局唯一的序列 ID, 由一个全局自增计数器提供
        self.token_ids = copy(token_ids) # 完整的Token序列, 包含 prompt 和 completion 两部分
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids) # prompt 部分的 Token 数, 由用户请求时提供, 用于区分 prompt 和 completion
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens # completion 部分的最大 Token 数, 由用户请求时提供, 用于判断序列何时完成
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略 EOS token, 由用户请求时提供, 用于判断序列何时完成

        # —— 调度最小单位 ——
        self.status = SequenceStatus.WAITING # 当前状态, 由 LLMEngine 调度时更新, WAITING: 等待调度, RUNNING: 正在推理, FINISHED: 已完成   
        self.num_scheduled_tokens = 0 # step计划的 Token 数, 由 LLMEngine 调度时更新
        self.is_prefill = True

        # —— KV 块持有者 ——
        self.block_table = [] # KV块号列表，长度 = ceil(num_tokens / block_size)
        self.num_tokens = len(self.token_ids) # token_ids 中的长度
        self.num_cached_tokens = 0 # 已写入 KV 块的 Token 数, 由 LLMEngine 调度时更新
        

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:] # completion 部分的 Token ID 列表, 由 token_ids 和 num_prompt_tokens 派生, 用于区分 prompt 和 completion

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size # 向上取整，得到 KV 块的数量

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size # 最后一个 KV 块中的 Token 数, 可能不足 block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]  # 取第 i 个 KV 块的 Token ID 列表, (用于hashing)

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
