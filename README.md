# FlashAttention for NVIDIA Turing (SM75)

This is an unofficial experimental fork of
[Dao-AILab FlashAttention](https://github.com/Dao-AILab/flash-attention). It
adds native CUDA attention kernels for NVIDIA Turing GPUs with compute
capability 7.5 while keeping the `flash-attn` distribution name and
`flash_attn` Python imports for compatibility.

The fork has been tested primarily on an RTX 2080 Ti. Other SM75 GPUs,
including the T4, still need independent testing. This project is not
affiliated with or endorsed by Dao-AILab or NVIDIA.

## What This Fork Is

The first release is an experimental, inference-first SM75 backend compatible
with the FlashAttention 2 public Python API. Supported SM75 calls run through
CUDA code in this repository. They do not silently fall back to PyTorch SDPA or
xFormers.

For Ampere, Ada, Hopper, Blackwell, ROCm, FA3, FA4, and upstream training
documentation, use the
[upstream FlashAttention project](https://github.com/Dao-AILab/flash-attention).

## Supported And Tested Hardware

- Primary test GPU: RTX 2080 Ti, compute capability 7.5.
- Target GPU architecture: NVIDIA Turing, SM75.
- Current build environment: CUDA 12.x.
- Native compute dtype: FP16.
- BF16 compatibility: inputs are converted to FP16 for computation and outputs
  are converted back to BF16.
- Supported head dimensions: `32, 64, 80, 96, 112, 128, 160, 192, 224, 256`.

The PyTorch and CUDA versions used to build the wheel must match the runtime
environment. Support for a head dimension does not imply support for every
upstream option at that dimension.

## Installation

Install the build dependency first, then build from this checkout without
allowing pip to replace the installed PyTorch build:

```sh
python3 -m pip install einops
python3 -m pip install --no-deps --no-build-isolation .
```

To build and install a wheel:

```sh
python3 -m pip install einops
FLASH_ATTN_BUILD_SM75_ONLY=TRUE python3 -m pip wheel . --no-build-isolation --no-deps -w dist-sm75
python3 -m pip install --force-reinstall --no-deps dist-sm75/flash_attn*.whl
```

Wheel builds from this repository always compile the checked-out source. This
fork does not download cached Dao-AILab release wheels.

## Verify The SM75 Backend

```python
import flash_attn
import flash_attn_sm75_cuda
from flash_attn import flash_attn_func, get_last_flash_attn_backend

# Run a supported SM75 CUDA call before checking the backend.
print(flash_attn.__version__)
print(get_last_flash_attn_backend())
```

The backend string for a successful SM75 call starts with `sm75_native`.
Native-only test mode turns any unsupported route into an error:

```sh
FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 pytest -q tests/test_sm75_*.py
```

## Compatibility

| Area | SM75 status |
| --- | --- |
| Dense attention | Native FP16 CUDA paths for the listed head dimensions. |
| Packed QKV and KV | Native routes; some use strided Q/K/V views into the dense kernels. |
| Variable length | Native routes; some modes repack data into native variable-length buffers. |
| KV cache | Native contiguous and paged cache routes, cache update, rotary, `cache_batch_idx`, and `cache_leftpad`. |
| BF16 | Compatibility conversion to FP16 compute, then conversion back to BF16. |
| Dropout | Not supported on SM75 in this release. `dropout_p > 0` raises an error. |
| Backward | Tested only with `dropout_p == 0`; production training support is not claimed. |
| Runtime fallbacks | Supported SM75 calls do not use SDPA or xFormers fallbacks. |

## Benchmarks

On the disclosed RTX 2080 Ti benchmark matrix, the SM75 public attention
operations achieved a 1.723x geomean against the included PyTorch SDPA-based
reference pipelines, with a 1.144x minimum. KV-cache update and decode achieved
a 2.894x geomean against the included PyTorch reference pipeline, with a 1.184x
minimum.

These are public-operation pipeline comparisons, not pure kernel timings. The
GQA reference includes K/V head expansion. The variable-length reference uses
per-sequence Python calls and concatenation. The KV-cache reference includes
cache update, page-to-dense conversion where needed, rotary, concatenation,
left-padding handling, GQA expansion, and SDPA.

Installed-wheel Transformers generation achieved about a 1.100x end-to-end
geomean, with a 0.963x minimum. Attention is only part of whole-model runtime,
so the component benchmark gains should not be read as whole-model gains.

## Limitations

- CUDA stream and device handling still needs a complete audit and
  non-default-stream testing.
- Dropout is disabled because the SM75 code does not implement compatible CUDA
  RNG and Philox state handling.
- The first release is intended for experimental inference use.
- Hardware testing outside the RTX 2080 Ti is incomplete.

## Tests And Reproducibility

Release results must come from an installed wheel, not imports from the source
tree. The release report must record the Python, PyTorch, CUDA, Transformers,
driver, compiler, model revisions, commands, warmups, sample counts, and raw
benchmark output.

Current local validation commands include:

```sh
make sm75-ci
make sm75-sanitize
```

A small release report and its benchmark data will be committed under
`docs/release/`. Larger raw logs can be attached to the GitHub prerelease.

## Provenance And Licenses

The upstream Dao-AILab code remains under its upstream license. Parts of the
SM75 forward and backward kernel implementation are derived from
[SSIU's flash-attention-turing](https://github.com/ssiu/flash-attention-turing).
See `csrc/flash_attn_sm75/NOTICE.md` for attribution.

## AI-Assisted Development

Development was substantially assisted by OpenAI coding agents. Project
direction, hardware operation, benchmark design, acceptance criteria, review,
and release validation were performed by the maintainer. Published claims are
tied to included reproducible tests and benchmark artifacts.

## Upstream Credits

FlashAttention and FlashAttention-2 were created by Tri Dao and collaborators.
Please cite the upstream work when using this fork:

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)

Upstream source and full documentation:
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
