import pytest
import torch

from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

from sm75_test_utils import enable_native_only, make_qkv, make_varlen


@pytest.mark.parametrize(
    "api",
    ["dense", "qkvpacked", "kvpacked", "varlen", "varlen_qkvpacked", "varlen_kvpacked"],
)
def test_dropout_rejected_on_sm75(monkeypatch, api):
    enable_native_only(monkeypatch)

    with pytest.raises(RuntimeError, match="SM75 attention dropout is disabled"):
        if api == "dense":
            q, k, v = make_qkv(headdim=64)
            flash_attn_func(q, k, v, dropout_p=0.1)
        elif api == "qkvpacked":
            q, k, v = make_qkv(headdim=64)
            flash_attn_qkvpacked_func(torch.stack((q, k, v), dim=2), dropout_p=0.1)
        elif api == "kvpacked":
            q, k, v = make_qkv(nheads=4, nheads_k=2, headdim=64)
            flash_attn_kvpacked_func(q, torch.stack((k, v), dim=2), dropout_p=0.1)
        elif api == "varlen":
            q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(headdim=64)
            flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.1)
        elif api == "varlen_qkvpacked":
            q, k, v = make_qkv(batch=1, headdim=64)
            q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
            cu = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
            flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1), cu, q.shape[0], dropout_p=0.1
            )
        else:
            q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(headdim=64)
            flash_attn_varlen_kvpacked_func(
                q, torch.stack((k, v), dim=1), cu_q, cu_k, max_q, max_k, dropout_p=0.1
            )
