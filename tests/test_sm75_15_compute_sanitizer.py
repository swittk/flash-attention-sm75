import os
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("compute-sanitizer") is None, reason="compute-sanitizer is not installed")
def test_compute_sanitizer_sm75_smoke_command_exists():
    env = os.environ.copy()
    env["FLASH_ATTN_SM75_NATIVE_ONLY"] = "1"
    env["FLASH_ATTN_SM75_BAN_FALLBACKS"] = "1"
    result = subprocess.run(
        ["compute-sanitizer", "--version"],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
