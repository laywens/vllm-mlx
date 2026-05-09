# SPDX-License-Identifier: Apache-2.0
"""Focused tests for SpecPrefill helper behavior."""


def test_rope_helpers_support_rotary_emb_attribute():
    from vllm_mlx import specprefill

    original_rope = object()
    replacement_rope = object()

    class Attention:
        rotary_emb = original_rope

    attention = Attention()

    assert specprefill._get_rope(attention) is original_rope

    specprefill._set_rope(attention, replacement_rope)

    assert attention.rotary_emb is replacement_rope


def test_cleanup_rope_restores_rotary_emb_wrappers():
    from vllm_mlx import specprefill

    original_rope = object()

    class Attention:
        rotary_emb = specprefill._OffsetAdjustedRoPE(original_rope, adjustment=3)

    class Layer:
        self_attn = Attention()

    class Model:
        layers = [Layer()]

    model = Model()

    specprefill.cleanup_rope(model)

    assert model.layers[0].self_attn.rotary_emb is original_rope


def test_query_extractor_selection_uses_model_type_before_attributes():
    from types import SimpleNamespace

    from vllm_mlx import specprefill

    class Rope:
        pass

    class Attention:
        q_norm = object()
        rope = Rope()

    unknown_model = SimpleNamespace(config=SimpleNamespace(model_type="unknown"))
    qwen_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
    nemotron_model = SimpleNamespace(config=SimpleNamespace(model_type="nemotron_h"))

    assert (
        specprefill._select_query_extractor(unknown_model, Attention())
        is specprefill._llama_extract_queries
    )
    assert (
        specprefill._select_query_extractor(qwen_model, Attention())
        is specprefill._qwen35_extract_queries
    )
    assert (
        specprefill._select_query_extractor(nemotron_model, Attention())
        is specprefill._nemotron_h_extract_queries
    )
