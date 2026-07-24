"""A compact decoder-only transformer trained from scratch.

The architecture follows the nanoGPT lineage with modern components: RMSNorm,
rotary position embeddings, SwiGLU feed-forward, optional grouped-query
attention, and flash attention through scaled dot product attention. No
pretrained weights or configurations are ever loaded, so the model starts with
no inherited knowledge. Size is fully parameterized; presets provide convenient
points up to roughly one billion parameters.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


PRESETS = {
    'smoke': dict(number_of_layers=2, number_of_heads=2,
                  embedding_dimension=128, block_size=256),
    'pico': dict(number_of_layers=2, number_of_heads=4,
                 embedding_dimension=128, block_size=512),
    'nano': dict(number_of_layers=4, number_of_heads=4,
                 embedding_dimension=192, block_size=512),
    'micro': dict(number_of_layers=6, number_of_heads=6,
                  embedding_dimension=288, block_size=512),
    'mini': dict(number_of_layers=8, number_of_heads=8,
                 embedding_dimension=384, block_size=1024),
    'poc-60m': dict(number_of_layers=10, number_of_heads=10,
                    embedding_dimension=640, block_size=1024),
    'poc-150m': dict(number_of_layers=12, number_of_heads=12,
                     embedding_dimension=960, block_size=1024),
    'poc-350m': dict(number_of_layers=24, number_of_heads=16,
                     embedding_dimension=1024, block_size=1024),
    'poc-760m': dict(number_of_layers=24, number_of_heads=16,
                     embedding_dimension=1536, block_size=2048),
    'poc-1b': dict(number_of_layers=24, number_of_heads=16,
                   embedding_dimension=1792, block_size=2048),
}


@dataclass
class GPTConfig:
    vocabulary_size: int
    number_of_layers: int = 12
    number_of_heads: int = 12
    number_of_key_value_heads: int = None
    embedding_dimension: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.number_of_key_value_heads is None:
            self.number_of_key_value_heads = self.number_of_heads
        if self.embedding_dimension % self.number_of_heads != 0:
            raise ValueError('embedding_dimension must divide by number_of_heads')
        if self.number_of_heads % self.number_of_key_value_heads != 0:
            raise ValueError('number_of_heads must divide by key_value heads')


def build_config(model_config, vocabulary_size):
    """Merge a preset and explicit overrides into a concrete GPTConfig."""
    base = {}
    if model_config.preset:
        if model_config.preset not in PRESETS:
            raise ValueError('unknown preset %r' % model_config.preset)
        base.update(PRESETS[model_config.preset])
    override_fields = [
        'number_of_layers',
        'number_of_heads',
        'number_of_key_value_heads',
        'embedding_dimension',
        'block_size',
    ]
    for name in override_fields:
        value = getattr(model_config, name, None)
        if value is not None:
            base[name] = value
    return GPTConfig(
        vocabulary_size=vocabulary_size,
        dropout=model_config.dropout,
        mlp_ratio=model_config.mlp_ratio,
        rope_theta=model_config.rope_theta,
        tie_embeddings=model_config.tie_embeddings,
        **base,
    )


class RMSNorm(nn.Module):
    def __init__(self, dimension, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight


def build_rotary_cache(sequence_length, head_dimension, theta, device, dtype):
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dimension, 2, device=device).float()
                  / head_dimension)
    )
    positions = torch.arange(sequence_length, device=device).float()
    frequencies = torch.outer(positions, inverse_frequency)
    cosine = frequencies.cos()[None, None, :, :]
    sine = frequencies.sin()[None, None, :, :]
    return cosine.to(dtype), sine.to(dtype)


def apply_rotary(tensor, cosine, sine, offset=0):
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    sequence_length = tensor.size(2)
    cosine = cosine[..., offset:offset + sequence_length, :]
    sine = sine[..., offset:offset + sequence_length, :]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_of_heads = config.number_of_heads
        self.number_of_key_value_heads = config.number_of_key_value_heads
        self.head_dimension = config.embedding_dimension // config.number_of_heads
        self.dropout = config.dropout
        query_dimension = self.number_of_heads * self.head_dimension
        key_value_dimension = self.number_of_key_value_heads * self.head_dimension
        self.query_projection = nn.Linear(
            config.embedding_dimension, query_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            config.embedding_dimension, key_value_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            config.embedding_dimension, key_value_dimension, bias=False
        )
        self.output_projection = nn.Linear(
            config.embedding_dimension, config.embedding_dimension, bias=False
        )

    def forward(self, hidden_states, cosine, sine, past=None, offset=0):
        """Attend over the input, optionally continuing a decode cache.

        Without past this is ordinary causal attention over the whole
        sequence. With past, hidden_states must be a single new position:
        its key and value are appended to the cached ones and the lone
        query attends over everything, which is what makes incremental
        generation cheap. Returns the output and the (pre-expansion)
        key/value cache for the next step.
        """
        batch, sequence_length, _ = hidden_states.shape
        queries = self.query_projection(hidden_states).view(
            batch, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)
        keys = self.key_projection(hidden_states).view(
            batch, sequence_length, self.number_of_key_value_heads,
            self.head_dimension,
        ).transpose(1, 2)
        values = self.value_projection(hidden_states).view(
            batch, sequence_length, self.number_of_key_value_heads,
            self.head_dimension,
        ).transpose(1, 2)

        queries = apply_rotary(queries, cosine, sine, offset)
        keys = apply_rotary(keys, cosine, sine, offset)

        if past is not None:
            past_keys, past_values = past
            keys = torch.cat((past_keys, keys), dim=2)
            values = torch.cat((past_values, values), dim=2)
        new_past = (keys, values)

        if self.number_of_key_value_heads != self.number_of_heads:
            repeats = self.number_of_heads // self.number_of_key_value_heads
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)

        attention = functional.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past is None,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch, sequence_length, -1
        )
        return self.output_projection(attention), new_past


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.mlp_ratio * config.embedding_dimension * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate_projection = nn.Linear(
            config.embedding_dimension, hidden, bias=False
        )
        self.up_projection = nn.Linear(
            config.embedding_dimension, hidden, bias=False
        )
        self.down_projection = nn.Linear(
            hidden, config.embedding_dimension, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        gated = functional.silu(self.gate_projection(hidden_states))
        projected = gated * self.up_projection(hidden_states)
        return self.dropout(self.down_projection(projected))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dimension)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.embedding_dimension)
        self.feed_forward = SwiGLU(config)

    def forward(self, hidden_states, cosine, sine, past=None, offset=0):
        attended, new_past = self.attention(
            self.attention_norm(hidden_states), cosine, sine, past, offset
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states, new_past


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.embedding_dimension
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.number_of_layers)]
        )
        self.final_norm = RMSNorm(config.embedding_dimension)
        self.language_model_head = nn.Linear(
            config.embedding_dimension, config.vocabulary_size, bias=False
        )
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embedding.weight

        self.rotary_cache = None
        self.apply(self._initialize_weights)
        residual_scale = 0.02 / math.sqrt(2 * config.number_of_layers)
        for name, parameter in self.named_parameters():
            if name.endswith('output_projection.weight') or name.endswith(
                'down_projection.weight'
            ):
                nn.init.normal_(parameter, mean=0.0, std=residual_scale)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, non_embedding=False):
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total = total - self.token_embedding.weight.numel()
            if not self.config.tie_embeddings:
                total = total - self.language_model_head.weight.numel()
        return total

    def _rotary(self, sequence_length, device, dtype):
        head_dimension = (
            self.config.embedding_dimension // self.config.number_of_heads
        )
        if (
            self.rotary_cache is None
            or self.rotary_cache[0].size(2) < sequence_length
            or self.rotary_cache[0].device != device
        ):
            self.rotary_cache = build_rotary_cache(
                max(sequence_length, self.config.block_size),
                head_dimension, self.config.rope_theta, device, dtype,
            )
        return self.rotary_cache

    def forward(self, input_ids, targets=None, ignore_index=-100, past=None,
                offset=0, return_cache=False):
        _, sequence_length = input_ids.shape
        hidden_states = self.dropout(self.token_embedding(input_ids))
        cosine, sine = self._rotary(
            offset + sequence_length, input_ids.device, hidden_states.dtype
        )
        new_past = []
        for index, block in enumerate(self.blocks):
            layer_past = past[index] if past is not None else None
            hidden_states, layer_new = block(
                hidden_states, cosine, sine, layer_past, offset
            )
            new_past.append(layer_new)
        hidden_states = self.final_norm(hidden_states)

        if targets is not None:
            logits = self.language_model_head(hidden_states)
            loss = functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
            result = (logits, loss)
        else:
            logits = self.language_model_head(hidden_states[:, [-1], :])
            result = (logits, None)
        if return_cache:
            return result + (new_past,)
        return result

    def _adjust_logits(self, logits, input_ids, top_k, top_p,
                       repetition_penalty, repetition_window):
        if repetition_penalty and repetition_penalty != 1.0:
            recent = input_ids[:, -repetition_window:]
            for row in range(input_ids.size(0)):
                seen = torch.unique(recent[row])
                row_logits = logits[row, seen]
                logits[row, seen] = torch.where(
                    row_logits > 0,
                    row_logits / repetition_penalty,
                    row_logits * repetition_penalty,
                )
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float('inf')
        if top_p is not None:
            logits = filter_top_p(logits, top_p)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, eos_id=None, repetition_penalty=1.0,
                 repetition_window=64):
        """Sample continuations, cached when the result fits the block.

        When the prompt plus the new tokens fit inside block_size, the
        prompt is prefilled once and each new token is decoded
        incrementally against the key/value cache. Longer requests fall
        back to recomputing the cropped context per token, preserving the
        old sliding-window behavior.
        """
        if input_ids.size(1) + max_new_tokens <= self.config.block_size:
            return self._generate_cached(
                input_ids, max_new_tokens, temperature, top_k, top_p,
                eos_id, repetition_penalty, repetition_window,
            )
        return self._generate_recompute(
            input_ids, max_new_tokens, temperature, top_k, top_p, eos_id,
            repetition_penalty, repetition_window,
        )

    def _generate_cached(self, input_ids, max_new_tokens, temperature,
                         top_k, top_p, eos_id, repetition_penalty,
                         repetition_window):
        logits, _, past = self(input_ids, past=None, offset=0,
                               return_cache=True)
        for step in range(max_new_tokens):
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            step_logits = self._adjust_logits(
                step_logits, input_ids, top_k, top_p, repetition_penalty,
                repetition_window,
            )
            probabilities = functional.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
            if step < max_new_tokens - 1:
                logits, _, past = self(
                    next_token, past=past, offset=input_ids.size(1) - 1,
                    return_cache=True,
                )
        return input_ids

    def _generate_recompute(self, input_ids, max_new_tokens, temperature,
                            top_k, top_p, eos_id, repetition_penalty,
                            repetition_window):
        for _ in range(max_new_tokens):
            conditioned = input_ids[:, -self.config.block_size:]
            logits, _ = self(conditioned)
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            step_logits = self._adjust_logits(
                step_logits, input_ids, top_k, top_p, repetition_penalty,
                repetition_window,
            )
            probabilities = functional.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return input_ids

    def configure_optimizers(self, weight_decay, learning_rate, betas,
                             device_type):
        decay_parameters = []
        no_decay_parameters = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dim() >= 2:
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)
        groups = [
            {'params': decay_parameters, 'weight_decay': weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0},
        ]
        use_fused = device_type == 'cuda'
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, fused=use_fused
        )


def filter_top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = functional.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits[remove] = -float('inf')
    restored = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return restored
