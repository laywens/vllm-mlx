# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-vlm's Qwen3.5 attention to support BatchKVCache.

mlx-vlm's Qwen3_5Attention uses cache.offset directly for kv_seq_len
computation and mask slicing. BatchKVCache stores offset as mx.array
(per-batch-item), not int, causing:

    mask = mask[..., :kv_seq_len]
    ValueError: Slice indices must be integers or None.

This patch replaces Qwen3_5Attention.__call__ with a version that converts
cache.offset to int before using it for arithmetic/slicing, while leaving the
actual cache.offset untouched so update_and_fetch still works with per-batch
offsets.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _cache_offset_to_int(cache) -> int:
    """Extract cache offset as int, handling BatchKVCache mx.array offset."""
    if cache is None:
        return 0
    off = cache.offset
    if isinstance(off, int):
        return off
    if isinstance(off, mx.array):
        return int(off.max().item()) if off.ndim > 0 else int(off.item())
    return int(off)


def patch_qwen35_attention_for_batching() -> bool:
    """Monkey-patch Qwen3_5Attention.__call__ to handle BatchKVCache.

    Returns True if the patch was applied, False if mlx-vlm is not installed
    or the Qwen3.5 module is not available.
    """
    try:
        from mlx_lm.models.base import scaled_dot_product_attention
        from mlx_vlm.models.qwen3_5.language import (
            Qwen3_5Attention,
            apply_multimodal_rotary_pos_emb,
        )
    except ImportError:
        logger.debug("[Qwen3.5 patch] mlx-vlm Qwen3.5 module not available")
        return False

    if getattr(Qwen3_5Attention, "_batch_patched", False):
        logger.debug("[Qwen3.5 patch] Already patched")
        return True

    def _patched_call(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache=None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len, _hidden_dim = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(batch_size, seq_len, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, seq_len, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, seq_len, self.num_key_value_heads, -1
        ).transpose(
            0,
            2,
            1,
            3,
        )

        kv_seq_len = keys.shape[-2]

        # Convert cache.offset to int for slice compatibility. BatchKVCache
        # stores offset as mx.array, but kv_seq_len must be an int for slicing.
        offset = _cache_offset_to_int(cache)

        # mlx-vlm can cache position_ids on the language model object. If a
        # later request has a different prompt length, stale position_ids can
        # be shorter than the current chunk and crash rotary application.
        if position_ids is not None and position_ids.shape[-1] != seq_len:
            logger.debug(
                "[Qwen3.5 patch] Recomputing stale position_ids: got %s, expected %s",
                position_ids.shape[-1],
                seq_len,
            )
            position_ids = None

        if position_ids is None:
            kv_seq_len += offset + 1
            position_ids = mx.arange(offset, offset + seq_len)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += offset + 1 if cache is not None else 0

        cos, sin = self.rotary_emb(values, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., :kv_seq_len]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.o_proj(output * mx.sigmoid(gate))

    Qwen3_5Attention.__call__ = _patched_call
    Qwen3_5Attention._batch_patched = True
    logger.info("[Qwen3.5 patch] Attention patched for BatchKVCache support")
    return True
