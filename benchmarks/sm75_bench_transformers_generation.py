import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    transformers = __import__("transformers")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    inputs = tok("SM75 FlashAttention benchmark", return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=args.tokens)
    torch.cuda.synchronize()
    print({"model": args.model, "tokens": args.tokens, "seconds": time.perf_counter() - start})


if __name__ == "__main__":
    main()
