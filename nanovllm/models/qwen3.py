import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,              # 每个 token 的隐藏向量维度，也就是模型宽度。
        num_heads: int,                # 全模型 query heads 数；TP 后每张卡只保留其中一部分。
        num_kv_heads: int,             # 全模型 key/value heads 数；小于 num_heads 时就是 GQA/MQA。
        max_position: int = 4096 * 32, # RoPE 预计算的最大位置，必须覆盖运行时最大上下文长度。
        head_dim: int | None = None,   # 单个 attention head 的维度；缺省时 hidden_size / num_heads。
        rms_norm_eps: float = 1e-06,   # Q/K RMSNorm 的数值稳定项，只在无 QKV bias 的 Qwen3 路径使用。
        qkv_bias: bool = False,        # 是否给 q/k/v projection 加 bias；也决定是否启用 Q/K norm。
        rope_theta: float = 10000,     # RoPE 的频率基数，越大通常支持越长的位置外推。
        rope_scaling: dict | None = None, # HF 配置里的 RoPE 扩展参数；这里只读取可能覆盖的 rope_theta。
    ) -> None:
        super().__init__()
        # Tensor Parallel: 权重按 head 维切开，因此总 head 数必须能被进程数整除。
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 当前 TP rank 本地负责的 query heads。
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本地 K/V heads；GQA 下通常少于 Q heads。
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 每个 head 的通道数。
        self.q_size = self.num_heads * self.head_dim                    # 本地 Q 投影输出宽度。
        self.kv_size = self.num_kv_heads * self.head_dim                # 本地 K 或 V 投影输出宽度。
        self.scaling = self.head_dim ** -0.5                            # QK^T 缩放因子，保持 softmax 数值稳定。
        self.qkv_bias = qkv_bias

        # 一次线性层同时产出 Q/K/V；QKVParallelLinear 会按 TP rank 装载各自的 head shard。
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # RowParallelLinear 接收本地 heads 拼接后的结果，内部 all_reduce 合并 TP rank 的部分输出。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # RoPE 作用在 Q/K 的 head_dim 上；positions 是每个 token 的绝对位置索引。
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # Attention 封装了 prefill/decode 分支：prefill 走 varlen flash-attn，decode 走 paged KV cache。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # Qwen3 无 attention bias 时通常对 Q/K 做 per-head RMSNorm，V 不参与归一化。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states 形状通常是 [num_tokens, hidden_size]，num_tokens 是本轮 batch 内被展平的 token 数。
        qkv = self.qkv_proj(hidden_states)
        # 本地 qkv 最后一维布局为 [Q heads, K heads, V heads]，GQA 下 K/V 宽度可以小于 Q。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # RoPE 只旋转 Q/K，让注意力分数携带位置信息；V 保持原值用于加权求和。
        q, k = self.rotary_emb(positions, q, k)
        # Attention 内部会根据全局 context 把 K/V 写入 paged KV cache，并执行 prefill 或 decode attention。
        o = self.attn(q, k, v)
        # 多个 heads 的输出先在本 rank 展平，再通过 o_proj 回到 hidden_size。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,       # Transformer 主干维度，MLP 输入/输出都回到这个宽度。
        intermediate_size: int, # FFN 扩展维度，通常大于 hidden_size。
        hidden_act: str,        # Qwen3 这里要求 silu，用于 SwiGLU: silu(gate) * up。
    ) -> None:
        super().__init__()
        # gate_proj 和 up_proj 合并成一个列并行线性层，减少一次 matmul/通信调度开销。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # down_proj 把 intermediate_size 投回 hidden_size；RowParallelLinear 会聚合 TP 分片。
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        # SiluAndMul 会把最后一维切成 gate/up 两半，并计算 silu(gate) * up。
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config, # HF 的模型结构配置，权重 loader 也按这些名字和形状对齐。
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层还没有累计 residual：归一化后的值进 attention，原 hidden_states 作为残差起点。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层使用融合 add + RMSNorm：先 hidden_states + residual，再返回归一化结果和新的 residual。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # attention 输出先加回 residual 并归一化，得到 MLP 输入；同时更新 residual 给下一次残差相加。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # 返回的是“尚未加残差的 MLP 输出”和“已经包含 attention 残差的 residual”。
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词表按 vocab 维做并行切分；每个 rank 只保存一段 embedding/lm_head 权重。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # num_hidden_layers 决定 DecoderLayer 堆叠深度；所有层共享同一份结构配置但权重各自独立。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # positions 与 input_ids 一一对应；调度器会把不同序列的 token 展平成一个批次传进来。
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层 MLP 输出还要再与 residual 相加并做最终 RMSNorm，得到用于 lm_head 的 hidden states。
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # HuggingFace 权重名到本实现合并模块的映射；loader 用它把分开的 q/k/v、gate/up 权重塞进合并层。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # tie_word_embeddings=True 时输出层和输入 embedding 共享同一块权重，节省显存并保持 HF 语义。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # forward 只返回最后一层 hidden states；采样前再显式调用 compute_logits。
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # lm_head 把 hidden_size 投到 vocab_size，各 rank 负责局部词表 logits。
        return self.lm_head(hidden_states)
