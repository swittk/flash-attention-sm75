import argparse
import gc
import json
import os
import subprocess
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USE_INSTALLED_FLASH_ATTN = os.getenv("FLASH_ATTN_BENCH_USE_INSTALLED", "0").upper() in {
    "1",
    "TRUE",
    "YES",
    "ON",
}
if not USE_INSTALLED_FLASH_ATTN and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_model(model_id, attn_implementation):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
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
        out = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
        )
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


def _decode_once(model, inputs, new_tokens):
    from transformers.cache_utils import DynamicCache

    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True)
    past_key_values = prefill.past_key_values
    if isinstance(past_key_values, tuple):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    del prefill
    input_ids = torch.zeros(
        (inputs["input_ids"].shape[0], 1),
        device=inputs["input_ids"].device,
        dtype=inputs["input_ids"].dtype,
    )
    full_attention_mask = torch.ones(
        (
            inputs["attention_mask"].shape[0],
            inputs["attention_mask"].shape[1] + new_tokens,
        ),
        device=inputs["attention_mask"].device,
        dtype=inputs["attention_mask"].dtype,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(new_tokens):
            output = model(
                input_ids=input_ids,
                attention_mask=full_attention_mask[:, : inputs["attention_mask"].shape[1] + step + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del output, past_key_values
    return elapsed, new_tokens / elapsed


def _backend():
    try:
        from flash_attn import get_last_flash_attn_backend

        return get_last_flash_attn_backend()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"backend_unavailable:{exc}"


def _flash_attn_import_path():
    try:
        import flash_attn

        return getattr(flash_attn, "__file__", "")
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"import_path_unavailable:{exc}"


def _run_variant(model_id, prompt_len, new_tokens, attn_impl, warmup):
    tok, model = _load_model(model_id, attn_impl)
    inputs = _prompt(tok, prompt_len)
    for _ in range(warmup):
        _generate_once(model, inputs, min(4, new_tokens))
    _prefill_once(model, inputs)
    prefill_s = _prefill_once(model, inputs)
    out, total_s, peak = _generate_once(model, inputs, new_tokens)
    actual_new_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
    decode_s, decode_tps = _decode_once(model, inputs, new_tokens)
    decoded = tok.decode(out[0, -actual_new_tokens:], skip_special_tokens=True) if actual_new_tokens else ""
    backend = _backend() if attn_impl == "flash_attention_2" else "transformers_sdpa"
    del model, tok, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return {
        "attn_implementation": attn_impl,
        "prefill_s": prefill_s,
        "total_s": total_s,
        "time_to_first_token_s": prefill_s,
        "decode_time_s": decode_s,
        "decode_time_valid": True,
        "decode_time_method": "direct_cached_forward_loop",
        "decode_tokens_per_s": decode_tps,
        "requested_new_tokens": new_tokens,
        "actual_new_tokens": actual_new_tokens,
        "generation_complete": actual_new_tokens == new_tokens,
        "total_tokens_per_s": (prompt_len + actual_new_tokens) / max(1e-9, total_s),
        "peak_vram_mb": peak,
        "backend": backend,
        "flash_attn_import_path": _flash_attn_import_path() if attn_impl == "flash_attention_2" else None,
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
    decode_speedup = None
    if sdpa["decode_tokens_per_s"] is not None and fa2["decode_tokens_per_s"] is not None:
        decode_speedup = fa2["decode_tokens_per_s"] / sdpa["decode_tokens_per_s"]
    comparable = (
        sdpa["generation_complete"]
        and fa2["generation_complete"]
        and sdpa["actual_new_tokens"] == fa2["actual_new_tokens"]
    )
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
        "comparable": comparable,
        "speedup_vs_baseline": sdpa["total_s"] / fa2["total_s"] if comparable else None,
        "decode_speedup_vs_baseline": decode_speedup if comparable else None,
        "peak_vram_native_mb": fa2["peak_vram_mb"],
        "peak_vram_baseline_mb": sdpa["peak_vram_mb"],
        "fallback_attempted": not str(fa2["backend"]).startswith("sm75_native"),
    }


def _write_payload(path, rows, complete):
    payload = {
        "matrix": "transformers_generation",
        "row_count": len(rows),
        "complete": complete,
        "rows": rows,
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
    return payload


def _run_case_subprocess(model_id, prompt_len, new_tokens, warmup, out_path, idx):
    case_path = f"{out_path}.case{idx}.json"
    cmd = [
        sys.executable,
        __file__,
        "--models",
        model_id,
        "--prompt-lens",
        str(prompt_len),
        "--new-tokens",
        str(new_tokens),
        "--warmup",
        str(warmup),
        "--out",
        case_path,
        "--no-isolate-cases",
    ]
    env = os.environ.copy()
    if not USE_INSTALLED_FLASH_ATTN:
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = REPO_ROOT if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        return {
            "name": f"{model_id}_p{prompt_len}_new{new_tokens}",
            "family": "transformers_generation",
            "model": model_id,
            "prompt_len": prompt_len,
            "new_tokens": new_tokens,
            "status": "failed",
            "failed_stage": "benchmark_subprocess",
            "error_type": "CalledProcessError",
            "error": f"exit code {proc.returncode}",
        }
    with open(case_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        os.remove(case_path)
    except OSError:
        pass
    return payload["rows"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128])
    parser.add_argument("--new-tokens", nargs="+", type=int, default=[32])
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", "--output", dest="out", default=f"sm75_transformers_generation_{int(time.time())}.json")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--no-isolate-cases", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    cases = [(m, p, n) for m in args.models for p in args.prompt_lens for n in args.new_tokens]
    if args.max_cases:
        cases = cases[: args.max_cases]
    rows = []
    for idx, (model_id, prompt_len, new_tokens) in enumerate(cases):
        if args.no_isolate_cases:
            row = run_case(model_id, prompt_len, new_tokens, args.warmup)
        else:
            row = _run_case_subprocess(model_id, prompt_len, new_tokens, args.warmup, args.out, idx)
        rows.append(row)
        _write_payload(args.out, rows, False)
    payload = _write_payload(args.out, rows, True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
