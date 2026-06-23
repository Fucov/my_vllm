import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    prefill_policy: str = "fcfs" #prefill策略
    prefill_chunk_size: int = 1024
    enable_prefix_late_merge: bool = False
    max_seq_len_to_capture: int | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.prefill_policy in ("fcfs", "fair")
        assert self.prefill_chunk_size > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        else:
            self.max_seq_len_to_capture = min(self.max_seq_len_to_capture, self.max_model_len)
