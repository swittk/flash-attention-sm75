import pytest
import torch
import torch.nn.functional as F

from flash_attn import get_last_flash_attn_backend, reset_last_flash_attn_backend

from sm75_test_utils import enable_native_only, require_sm75


def test_transformers_llama_fa2_prefill_and_cached_decode(monkeypatch):
    transformers = pytest.importorskip("transformers")
    enable_native_only(monkeypatch)
    device = require_sm75()

    def reject_sdpa(*args, **kwargs):
        raise AssertionError("Transformers used PyTorch SDPA instead of flash_attention_2")

    monkeypatch.setattr(F, "scaled_dot_product_attention", reject_sdpa)

    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "flash_attention_2"
    model = transformers.LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    input_ids = torch.tensor([[1, 5, 9, 7, 3, 4, 8, 6]], device=device)

    reset_last_flash_attn_backend()
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True)
    prefill_backend = get_last_flash_attn_backend()
    assert prefill.logits.shape == (1, input_ids.shape[1], config.vocab_size)
    assert prefill_backend is not None and prefill_backend.startswith("sm75_native")

    reset_last_flash_attn_backend()
    with torch.inference_mode():
        decode = model(
            input_ids=torch.tensor([[11]], device=device),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    decode_backend = get_last_flash_attn_backend()
    assert decode.logits.shape == (1, 1, config.vocab_size)
    assert decode_backend is not None and decode_backend.startswith("sm75_native")
