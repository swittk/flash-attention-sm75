def test_common_downstream_imports_resolve():
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

    assert callable(flash_attn_func)
    assert callable(flash_attn_varlen_func)
    assert callable(flash_attn_with_kvcache)
    assert callable(flash_attn_kvpacked_func)
    assert callable(flash_attn_qkvpacked_func)
    assert callable(pad_input)
    assert callable(unpad_input)
