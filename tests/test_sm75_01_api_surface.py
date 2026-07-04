def test_public_api_surface_exists():
    import flash_attn
    from flash_attn import bert_padding
    from flash_attn import flash_attn_interface

    public_names = [
        "flash_attn_func",
        "flash_attn_qkvpacked_func",
        "flash_attn_kvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
        "flash_attn_with_kvcache",
    ]
    for name in public_names:
        assert callable(getattr(flash_attn, name))
        assert hasattr(flash_attn_interface, name)

    for name in [
        "index_first_axis",
        "pad_input",
        "unpad_input",
        "unpad_input_for_concatenated_sequences",
    ]:
        assert callable(getattr(bert_padding, name))

    assert callable(flash_attn.get_last_flash_attn_backend)
    assert callable(flash_attn.reset_last_flash_attn_backend)
