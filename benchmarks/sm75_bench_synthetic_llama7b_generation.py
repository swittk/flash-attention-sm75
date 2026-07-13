import argparse
import gc
import json
import math
import os
import sys
import time

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _write_payload(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _backend():
    try:
        from flash_attn import get_last_flash_attn_backend

        return get_last_flash_attn_backend()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"backend_unavailable:{exc}"


def _build_model(attn_implementation):
    from accelerate import init_empty_weights
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=16384,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = attn_implementation

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with init_empty_weights():
            model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(default_dtype)

    model.to_empty(device="cuda")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for buffer in model.buffers():
            if buffer.is_floating_point():
                buffer.zero_()
    model.eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


def _inputs(prompt_len):
    input_ids = torch.full((1, prompt_len), 1, device="cuda", dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _prefill_once(model, inputs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - start, torch.cuda.max_memory_allocated() / 1024**2


def _generate_once(model, inputs, new_tokens):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return output, elapsed, peak


def _median(values):
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def _run_variant(attn_implementation, prompt_lens, new_tokens, warmup, iters):
    gc.collect()
    torch.cuda.empty_cache()
    model = _build_model(attn_implementation)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    attention_class = type(model.model.layers[0].self_attn).__name__
    rows = []
    for prompt_len in prompt_lens:
        print(f"running {attn_implementation} prompt={prompt_len}", flush=True)
        inputs = _inputs(prompt_len)
        for _ in range(warmup):
            output, _, _ = _generate_once(model, inputs, min(2, new_tokens))
            del output
        prefill_samples = []
        total_samples = []
        prefill_peak = 0.0
        peak = 0.0
        output = None
        for _ in range(iters):
            prefill_s, sample_prefill_peak = _prefill_once(model, inputs)
            output, total_s, sample_peak = _generate_once(model, inputs, new_tokens)
            prefill_samples.append(prefill_s)
            total_samples.append(total_s)
            prefill_peak = max(prefill_peak, sample_prefill_peak)
            peak = max(peak, sample_peak)
        prefill_s = _median(prefill_samples)
        total_s = _median(total_samples)
        decode_s = total_s - prefill_s
        decode_time_valid = decode_s > 0.0
        rows.append(
            {
                "prompt_len": prompt_len,
                "new_tokens": new_tokens,
                "prefill_s": prefill_s,
                "total_s": total_s,
                "prefill_samples_s": prefill_samples,
                "total_samples_s": total_samples,
                "decode_s": decode_s,
                "decode_time_valid": decode_time_valid,
                "decode_tokens_per_s": new_tokens / decode_s if decode_time_valid else None,
                "prefill_peak_vram_mb": prefill_peak,
                "peak_vram_mb": peak,
                "backend": _backend() if attn_implementation == "flash_attention_2" else "transformers_sdpa",
                "last_token": int(output[0, -1].item()),
            }
        )
        del inputs, output
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "attn_implementation": attn_implementation,
        "parameter_count": parameter_count,
        "attention_class": attention_class,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 512, 2048, 4096])
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    if max(args.prompt_lens) + args.new_tokens > 16384:
        raise SystemExit("prompt length plus generated tokens must not exceed 16384")

    variants = {}
    for attn_implementation in ("sdpa", "flash_attention_2"):
        variants[attn_implementation] = _run_variant(
            attn_implementation,
            args.prompt_lens,
            args.new_tokens,
            args.warmup,
            args.iters,
        )

    sdpa_by_prompt = {row["prompt_len"]: row for row in variants["sdpa"]["rows"]}
    rows = []
    for fa2 in variants["flash_attention_2"]["rows"]:
        sdpa = sdpa_by_prompt[fa2["prompt_len"]]
        rows.append(
            {
                "name": f"synthetic_llama7b_p{fa2['prompt_len']}_new{args.new_tokens}",
                "prompt_len": fa2["prompt_len"],
                "new_tokens": args.new_tokens,
                "sdpa": sdpa,
                "flash_attention_2": fa2,
                "speedup_vs_baseline": sdpa["total_s"] / fa2["total_s"],
                "prefill_speedup_vs_baseline": sdpa["prefill_s"] / fa2["prefill_s"],
                "decode_speedup_vs_baseline": (
                    fa2["decode_tokens_per_s"] / sdpa["decode_tokens_per_s"]
                    if fa2["decode_tokens_per_s"] is not None
                    and sdpa["decode_tokens_per_s"] is not None
                    else None
                ),
                "fallback_attempted": not str(fa2["backend"]).startswith("sm75_native"),
            }
        )
    speedups = [row["speedup_vs_baseline"] for row in rows]
    payload = {
        "matrix": "synthetic_llama7b_generation",
        "architecture": {
            "layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "attention_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 32000,
            "weights": "deterministic zeros",
        },
        "parameter_count": variants["sdpa"]["parameter_count"],
        "attention_classes": {
            key: value["attention_class"] for key, value in variants.items()
        },
        "geomean_speedup_vs_baseline": math.exp(
            sum(math.log(speedup) for speedup in speedups) / len(speedups)
        ),
        "rows": rows,
    }
    _write_payload(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
