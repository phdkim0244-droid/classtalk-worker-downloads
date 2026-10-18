"""Fused encoder attention for the pinned mlx-whisper 0.4.3 model layout.

The decoder's cross-attention remains untouched because word alignment uses
its attention scores. Encoder blocks discard those scores, so they can use
MLX's fused kernel without allocating the full 1500 x 1500 attention matrix.
"""
import mlx.core as mx
from mlx_whisper.whisper import MultiHeadAttention


class EncoderAttention(MultiHeadAttention):
    def qkv_attention(self, q, k, v, mask=None):
        batch, length, state = q.shape
        head_dim = state // self.n_head
        q = q.reshape(batch, length, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        if mask is not None:
            mask = mask[:length, :k.shape[2]]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(batch, length, state), None


def accelerate_encoder(model):
    for block in model.encoder.blocks:
        if isinstance(block.attn, EncoderAttention):
            continue
        original = block.attn
        replacement = EncoderAttention(model.dims.n_audio_state, original.n_head)
        replacement.update(original.parameters())
        block.attn = replacement
