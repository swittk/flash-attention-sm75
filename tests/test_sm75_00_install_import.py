import importlib.metadata

import pytest


def test_flash_attn_imports_with_official_package_name():
    import flash_attn

    assert flash_attn.__name__ == "flash_attn"
    assert callable(flash_attn.flash_attn_func)
    try:
        version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("package metadata is only available after installation")
    assert version


def test_sm75_extension_import_if_installed():
    pytest.importorskip("flash_attn_sm75_cuda")
