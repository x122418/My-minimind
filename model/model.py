from transformers import PretrainedConfig

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = 0,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN


# 继承nn.Module类  RMS是一层
class RMSNorm(nn.Module):
    # __init__
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(x**2, keepdim=True, dim=-1) + self.eps)
        return x / rms

    # forward
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


# 预计算旋转表
def precompute_freqs_cis(dim: int, end: int, rope_base, rope_scaling: dict | None):
    # 初始化RoPE参数   长度end取值 eg:32k
    # freq 在这里指rope空间波的频率 单位距离转过的角度
    # rope_base 原文取 10000
    freqs = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    attn_factor = 1.0

    # 划分高低维度（低高频率 也就是 高低波长）的指标是 b = L / lambda_i    L是训练长度
    # lambda_i（波长）为 2 * pi/freq   = 2*pi*rope_base **(2i/dim)
    # b = L / 2*pi*rope_base **(2i/dim)     rope_base **(2i/dim) = L/(b * 2*pi)
    # 2i / dim * ln(rope_base) = lnL - lnb - ln(2pi)
    # i = dim/2 * (lnL - lnb - ln(2pi)) / ln(rope_base)

    # 根据rope_scaling 规则获取超参数
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        # 如果推理长度大于训练长度 需要缩放
        if end > orig_max:
            # b 到 i的映射
            inv_dim = lambda b: (
                dim / 2 * math.log(orig_max / (b * 2 * math.pi)) / math.log(rope_base)
            )
            # 划分高低频 的i边界
            # i 是 i个旋转对 or i种旋转角度（同一个位置上的） 一共 dim//2 对
            # low 高频 小的i  不需要缩放
            # high 低频 大的i 需要缩放

            high = min(dim // 2 - 1, math.ceil(inv_dim(beta_slow)))
            low = max(0, math.floor(inv_dim(beta_fast)))
            # 相当于给定两个点计算一个线性函数
            # 借助一个ramp实现 使得ramp是在0，1 之间插值的
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(0.001, high - low),
                0,
                1,
            )
            # 取low -> ramp = 0 -> freq = freq * 1
            # 取high -> ramp = 1 -> freq = freq / factor
            # 已有两个点 (0, 1)  (1, 1/factor) 求一次函数
            freqs = freqs * (1 - (1 - 1 / factor) * ramp)

    # 根据end计算 位置序列
    t = torch.arange(end, device=freqs.device).float()

    # 考虑不同位置的freqs表格 是 seq_length * Hidden_size//2 维度的 (S, H//2)
    freqs = torch.outer(t, freqs).float()

    # 但实际旋转时 又要cat一下（因为 (S,H) 尽管H上的两两一对的角度相同，但也需要旋转
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


# 编写RoPE函数 应用旋转表到 q, k 矩阵
# 旋转二维向量 [a, b]  A角度  转变为 [a*cosA - b*sinA, a*sinA + b*cosA]
# 工程上 每一个平面 取的基 并不是相邻的 实际是 abcd,ABCD 这种x和x+lenght/2 的对应
# 这个对应需要注意于cos矩阵的cat方式一致
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # [a, b] -> [-b, a] 方便后续旋转的计算

    def rotate_half(x):
        mid_idx = x.shape[-1] // 2

        return torch.cat([-x[..., mid_idx:], x[..., :mid_idx]], dim=-1)

    # 考虑q [B, S, N, H],  cos [S, H]
    # roteteed_vec = [a, b] * cos + [-b, a] * sin
    # 旋转后向量在 [a, b] [-b, a] 这组垂直基下的坐标自然是 [cos, sin]  相当于重构了坐标系
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(
        unsqueeze_dim
    )
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(
        unsqueeze_dim
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, S, N, H = x.shape
    if n_rep == 1:
        return x

    return x[:, :, :, None, :].expand(B, S, N, n_rep, H).reshape(B, S, N * n_rep, H)


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is None
            else args.num_attention_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_local_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_local_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.is_causal = True
        self.drop = args.dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    # 投影 计算kqv
    # 拆分多头
    # q k 使用rope
    # k v 使用repeat  需要注意KV cache
    # 进行 attention计算 Q @ k^T / sqrt(d)
    # 最后拼接头 返回结果

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: (
            torch.Tensor | None
        ) = None,  # [batch_size, kv_len or S]  假设1有效 0无效
    ):
        B, S, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(B, S, self.n_local_heads, self.head_dim)
        xk = xk.view(B, S, self.num_key_value_heads, self.head_dim)
        xv = xv.view(B, S, self.num_key_value_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            # 这意味着直接拼接过去的kv 如果是逐token生成 则要求 当前的kv 是seq_len = 1
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # xk, xv [B, S + psat_s, N*, H]  (before the next line)
        xk, xv = repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(
            xv, self.n_rep
        ).transpose(1, 2)
        # xk, xv [B, N*, S + psat_s, H]  (before the next line)

        # xq [B, S, N, H]
        xq = xq.transpose(1, 2)
        # xq [B, N, S, H]

        if (
            S > 1
            and self.flash
            and (attention_mask == None or torch.all(attention_mask == 1))
            and past_key_value == None
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.drop if self.training else 0.0,
                is_causal=self.is_causal,
            )

        # 手动实现attention计算   q @ K^T  / sqrt(H)
        else:
            scores = xq @ xk.transpose(-1, -2) / math.sqrt(self.head_dim)
            # [B, N, S, S + past_s]
            if self.is_causal == True:
                scores[:, :, :, -S:] += torch.triu(
                    torch.full((S, S), float("-inf"), device=scores.device), diagonal=1
                )
                # [B, N, S, S] 对这部分scores加causal掩码    [B, N, S, past_s] 不加掩码 全部看见

            # attention_mask [batch_size, S + past_s]
            if attention_mask is not None:
                scores += (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * float("-inf")
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)
            output = attn_weights @ xv  # [B, N, S, S+past_s] @ [B, N, S+past_s, H]
            # [B, N, S, H]
            output = output.transpose(1, 2).reshape(B, S, -1)  # [B, S, N, H]
            output = self.o_proj(output)
            output = self.residual_dropout(output)

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        # 升维 实际验证最好用2.66倍  并且上取整到64的倍数
        if args.intermediate_size == 0:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * (intermediate_size + 63) // 64
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MinimindBlock(nn.Module):
    def __init__(self, layer_id: int , config: MokioMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.input_rmsnorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attn_output_rmsnorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embedding,
        past_key_value,
        use_cache,
        attention_mask,
    ):
        residuals = hidden_states

        hidden_states, past_key_value = self.attn(
            self.input_rmsnorm(hidden_states),  # pre-norm
            position_embedding,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = hidden_states + residuals
        # post-attn norm  + residuals
        output = self.mlp(self.attn_output_rmsnorm(hidden_states)) + hidden_states

        return output, past_key_value
    
class MokioMiniModel(nn.Module):
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MinimindBlock(l, config)  for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size//config.num_attention_heads,
            config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor|None = None,
                attn_mask:torch.Tensor|None = None, 
                past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
                 use_cache:bool = False, **kwargs):
        
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        B, S = input_ids.shape
        
        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        # past_key_values 为 不同（pastk, pastv) 组成的列表 （一共k层数 block的重复次数）
        past_key_values = past_key_values or ([None] * len(self.layers)) # type: ignore[assignment]

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0# type: ignore[assignment]

        # text embedding + dropout
        hidden_states = self.dropout(self.embed(input_ids))

        position_embeddings = (self.freqs_cos[start_pos:start_pos+S],
                               self.freqs_sin[start_pos:start_pos+S])

        presents = []
        for layer_id, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):# type: ignore[assignment]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attn_mask,
            )
            presents.append(present)        

        return hidden_states, presents