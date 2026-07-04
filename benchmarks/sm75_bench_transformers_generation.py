import argparse
import gc
import json
import time

import torch


def _load_model(model_id, attn_implementation):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_implementation,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return tok, model


def _prompt(tok, prompt_len):
    text = "GPUs accelerate attention. "
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        raise RuntimeError("tokenizer returned no tokens")
    repeat = (prompt_len + ids.numel() - 1) // ids.numel()
    input_ids = ids.repeat(repeat)[:prompt_len].unsqueeze(0)
    return {"input_ids": input_ids.to("cuda"), "attention_mask": torch.ones_like(input_ids, device="cuda")}


def _generate_once(model, inputs, new_tokens):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return out, total_s, peak


def _prefill_once(model, inputs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _backend():
    try:
        from flash_attn import get_last_flash_attn_backend

        return get_last_flash_attn_backend()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"backend_unavailable:{exc}"


def _run_variant(model_id, prompt_len, new_tokens, attn_impl, warmup):
    tok, model = _load_model(model_id, attn_impl)
    inputs = _prompt(tok, prompt_len)
    for _ in range(warmup):
        _generate_once(model, inputs, min(4, new_tokens))
    prefill_s = _prefill_once(model, inputs)
    out, total_s, peak = _generate_once(model, inputs, new_tokens)
    decoded = tok.decode(out[0, -min(new_tokens, out.shape[1]) :], skip_special_tokens=True)
    backend = _backend() if attn_impl == "flash_attention_2" else "transformers_sdpa"
    del model, tok, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "attn_implementation": attn_impl,
        "prefill_s": prefill_s,
        "total_s": total_s,
        "time_to_first_token_s": prefill_s,
        "decode_tokens_per_s": new_tokens / max(1e-9, total_s - prefill_s),
        "total_tokens_per_s": (prompt_len + new_tokens) / max(1e-9, total_s),
        "peak_vram_mb": peak,
        "backend": backend,
        "sample_text": decoded,
    }


def run_case(model_id, prompt_len, new_tokens, warmup):
    name = f"{model_id}_p{prompt_len}_new{new_tokens}"
    print(f"running {name}", flush=True)
    try:
        sdpa = _run_variant(model_id, prompt_len, new_tokens, "sdpa", warmup)
    except Exception as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "name": name,
            "family": "transformers_generation",
            "model": model_id,
            "prompt_len": prompt_len,
            "new_tokens": new_tokens,
            "status": "failed",
            "failed_stage": "transformers_sdpa",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        fa2 = _run_variant(model_id, prompt_len, new_tokens, "flash_attention_2", warmup)
    except Exception as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "name": name,
            "family": "transformers_generation",
            "model": model_id,
            "prompt_len": prompt_len,
            "new_tokens": new_tokens,
            "baseline": "transformers_sdpa",
            "sdpa": sdpa,
            "status": "failed",
            "failed_stage": "flash_attention_2",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "name": name,
        "family": "transformers_generation",
        "model": model_id,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "status": "passed",
        "baseline": "transformers_sdpa",
        "sdpa": sdpa,
        "flash_attention_2": fa2,
        "backend": fa2["backend"],
        "speedup_vs_baseline": sdpa["total_s"] / fa2["total_s"],
        "decode_speedup_vs_baseline": fa2["decode_tokens_per_s"] / max(1e-9, sdpa["decode_tokens_per_s"]),
        "peak_vram_native_mb": fa2["peak_vram_mb"],
        "peak_vram_baseline_mb": sdpa["peak_vram_mb"],
        "fallback_attempted": not str(fa2["backend"]).startswith("sm75_native"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128])
    parser.add_argument("--new-tokens", nargs="+", type=int, default=[32])
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", "--output", dest="out", default=f"sm75_transformers_generation_{int(time.time())}.json")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    cases = [(m, p, n) for m in args.models for p in args.prompt_lens for n in args.new_tokens]
    if args.max_cases:
        cases = cases[: args.max_cases]
    rows = [run_case(model_id, prompt_len, new_tokens, args.warmup) for model_id, prompt_len, new_tokens in cases]
    payload = {
        "matrix": "transformers_generation",
        "row_count": len(rows),
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
