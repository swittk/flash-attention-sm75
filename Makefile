
PYTHON ?= python3
SM75_ATTENTION_RESULTS ?= /tmp/sm75_attention_micro.json
SM75_KVCACHE_RESULTS ?= /tmp/sm75_kvcache_decode.json
SM75_EVIDENCE_DIR ?= /tmp/flash-attn-sm75-evidence
SM75_GRIND_ATTENTION_RESULTS ?= $(SM75_EVIDENCE_DIR)/sm75_grind_attention.json
SM75_GRIND_KVCACHE_RESULTS ?= $(SM75_EVIDENCE_DIR)/sm75_grind_kvcache.json
SM75_GRIND_NUM_SHARDS ?= 1
SM75_GRIND_SHARD_INDEX ?= 0
SM75_RELEASE_ROOT ?= /tmp/flash-attn-sm75-release
SM75_RELEASE_WHEEL_DIR ?= $(SM75_RELEASE_ROOT)/wheel
SM75_RELEASE_VENV ?= $(SM75_RELEASE_ROOT)/venv
SM75_RELEASE_ATTENTION_RESULTS ?= $(SM75_RELEASE_ROOT)/attention.json
SM75_RELEASE_KVCACHE_RESULTS ?= $(SM75_RELEASE_ROOT)/kvcache.json
SM75_RELEASE_TRANSFORMERS_RESULTS ?= $(SM75_RELEASE_ROOT)/transformers-synthetic-7b.json

.PHONY: sm75-install-dev sm75-test-smoke sm75-test-correctness sm75-test-native-only sm75-bench sm75-sanitize sm75-profile sm75-ci sm75-grind
.PHONY: sm75-ci-quick sm75-release-clean sm75-release-wheel sm75-release-installed-wheel sm75-release-correctness sm75-release-sanitize sm75-release-benchmarks sm75-release-audit
.NOTPARALLEL: sm75-release-audit

sm75-install-dev:
	FLASH_ATTN_FORCE_BUILD=TRUE pip install -e . --no-build-isolation

sm75-test-smoke:
	pytest -q tests/test_sm75_00_install_import.py tests/test_sm75_01_api_surface.py tests/test_sm75_02_no_fallback_allowed.py

sm75-test-correctness:
	pytest -q tests/test_sm75_03_dtype_policy.py tests/test_sm75_04_dense_correctness.py tests/test_sm75_05_qkvpacked_correctness.py tests/test_sm75_06_varlen_correctness.py tests/test_sm75_07_backward_correctness.py

sm75-test-native-only:
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 pytest -q tests/test_sm75_*.py

sm75-bench:
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) benchmarks/sm75_bench_attention_micro.py --output $(SM75_ATTENTION_RESULTS)
	$(PYTHON) benchmarks/compare_sm75_results.py $(SM75_ATTENTION_RESULTS) --mode prefill --prefill-gate 1.20
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) benchmarks/sm75_bench_kvcache_decode.py --output $(SM75_KVCACHE_RESULTS)
	$(PYTHON) benchmarks/compare_sm75_results.py $(SM75_KVCACHE_RESULTS) --mode decode --decode-gate 1.15

sm75-sanitize:
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 compute-sanitizer --target-processes all $(PYTHON) -m pytest -q tests/test_sm75_04_dense_correctness.py

sm75-profile:
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 ncu --set full $(PYTHON) benchmarks/sm75_bench_attention_micro.py --quick

sm75-ci: sm75-test-smoke sm75-test-correctness sm75-test-native-only sm75-bench

sm75-ci-quick:
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) -m pytest -q \
		tests/test_sm75_00_install_import.py \
		tests/test_sm75_01_api_surface.py \
		tests/test_sm75_02_no_fallback_allowed.py \
		tests/test_sm75_04_dense_correctness.py \
		tests/test_sm75_05_qkvpacked_correctness.py \
		tests/test_sm75_10_kvcache_correctness.py \
		tests/test_sm75_12_transformers_fa2_unmodified.py \
		tests/test_sm75_17_stream_device_semantics.py

sm75-release-clean:
	test -z "$$(git status --porcelain)" || { echo "release audit requires a clean Git worktree"; git status --short; exit 1; }

sm75-release-wheel: sm75-release-clean
	mkdir -p $(SM75_RELEASE_WHEEL_DIR)
	rm -f $(SM75_RELEASE_WHEEL_DIR)/flash_attn*.whl
	MAX_JOBS=2 NVCC_THREADS=2 FLASH_ATTN_BUILD_SM75_ONLY=TRUE TORCH_CUDA_ARCH_LIST="7.5" \
		FLASH_ATTN_LOCAL_VERSION=$$($(PYTHON) -c 'import torch; cu=(torch.version.cuda or "none").replace(".", ""); tv=".".join(torch.__version__.split("+")[0].split(".")[:2]); abi=int(torch._C._GLIBCXX_USE_CXX11_ABI); print(f"cu{cu}torch{tv}cxx11abi{abi}")') \
		$(PYTHON) -m pip wheel . --no-build-isolation --no-deps -w $(SM75_RELEASE_WHEEL_DIR)
	sha256sum $(SM75_RELEASE_WHEEL_DIR)/flash_attn*.whl | tee $(SM75_RELEASE_ROOT)/wheel.sha256
	git rev-parse HEAD | tee $(SM75_RELEASE_ROOT)/commit.txt

sm75-release-installed-wheel: sm75-release-wheel
	rm -rf $(SM75_RELEASE_VENV)
	$(PYTHON) -m venv --system-site-packages $(SM75_RELEASE_VENV)
	$(SM75_RELEASE_VENV)/bin/python -m pip install --force-reinstall --no-deps $(SM75_RELEASE_WHEEL_DIR)/flash_attn*.whl
	cd /tmp && PYTHONPATH= $(abspath $(SM75_RELEASE_VENV))/bin/python -c 'import pathlib,sys,flash_attn,flash_attn_sm75_cuda; p=pathlib.Path(flash_attn.__file__).resolve(); assert pathlib.Path(sys.prefix).resolve() in p.parents, p; print(p); print(flash_attn.__version__)'

sm75-release-correctness: sm75-release-installed-wheel
	cd /tmp && PYTHONPATH= FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 \
		$(abspath $(SM75_RELEASE_VENV))/bin/python -m pytest --rootdir=/tmp -q $(abspath tests)/test_sm75_*.py

sm75-release-sanitize: sm75-release-installed-wheel
	cd /tmp && PYTHONPATH= FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 \
		compute-sanitizer --error-exitcode=1 --target-processes all \
		$(abspath $(SM75_RELEASE_VENV))/bin/python -m pytest --rootdir=/tmp -q \
		$(abspath tests)/test_sm75_04_dense_correctness.py \
		$(abspath tests)/test_sm75_05_qkvpacked_correctness.py \
		$(abspath tests)/test_sm75_06_varlen_correctness.py \
		$(abspath tests)/test_sm75_10_kvcache_correctness.py \
		$(abspath tests)/test_sm75_11_paged_kvcache_correctness.py \
		$(abspath tests)/test_sm75_17_stream_device_semantics.py

sm75-release-benchmarks: sm75-release-installed-wheel
	cd /tmp && PYTHONPATH= FLASH_ATTN_BENCH_USE_INSTALLED=1 FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 \
		$(abspath $(SM75_RELEASE_VENV))/bin/python $(abspath benchmarks)/sm75_bench_attention_micro.py \
		--matrix final-broad --warmup 20 --iters 100 --output $(abspath $(SM75_RELEASE_ATTENTION_RESULTS))
	cd /tmp && PYTHONPATH= FLASH_ATTN_BENCH_USE_INSTALLED=1 FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 \
		$(abspath $(SM75_RELEASE_VENV))/bin/python $(abspath benchmarks)/sm75_bench_kvcache_decode.py \
		--matrix final-broad --warmup 20 --iters 100 --output $(abspath $(SM75_RELEASE_KVCACHE_RESULTS))
	cd /tmp && PYTHONPATH= FLASH_ATTN_BENCH_USE_INSTALLED=1 FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 \
		$(abspath $(SM75_RELEASE_VENV))/bin/python $(abspath benchmarks)/sm75_bench_synthetic_llama7b_generation.py \
		--prompt-lens 128 512 --new-tokens 16 --warmup 2 --iters 5 --out $(abspath $(SM75_RELEASE_TRANSFORMERS_RESULTS))

sm75-release-audit: sm75-release-correctness sm75-release-sanitize sm75-release-benchmarks

sm75-grind:
	mkdir -p $(SM75_EVIDENCE_DIR)
	MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_SM75_ONLY=TRUE TORCH_CUDA_ARCH_LIST="7.5" $(PYTHON) setup.py build_ext --inplace
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) -m pytest -q tests/test_sm75_04_dense_correctness.py tests/test_sm75_05_qkvpacked_correctness.py tests/test_sm75_06_varlen_correctness.py tests/test_sm75_16_matrix_coverage.py --tb=short
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) benchmarks/sm75_bench_attention_micro.py --matrix final-broad --num-shards $(SM75_GRIND_NUM_SHARDS) --shard-index $(SM75_GRIND_SHARD_INDEX) --warmup 50 --iters 200 --output $(SM75_GRIND_ATTENTION_RESULTS)
	$(PYTHON) -c 'import json,sys; p=sys.argv[1]; rows=json.load(open(p))["rows"]; rows=sorted(rows,key=lambda r:r.get("speedup_vs_sdpa",r.get("speedup_vs_baseline",0))); print("\n".join("{:.4f} {} {}".format(r.get("speedup_vs_sdpa",r.get("speedup_vs_baseline",0)), r.get("name"), r.get("backend")) for r in rows[:30])); raise SystemExit(1 if rows and rows[0].get("speedup_vs_sdpa",rows[0].get("speedup_vs_baseline",0)) < 0.85 else 0)' $(SM75_GRIND_ATTENTION_RESULTS)
	$(PYTHON) benchmarks/compare_sm75_results.py $(SM75_GRIND_ATTENTION_RESULTS) --mode prefill --prefill-gate 1.20
	FLASH_ATTN_SM75_NATIVE_ONLY=1 FLASH_ATTN_SM75_BAN_FALLBACKS=1 $(PYTHON) benchmarks/sm75_bench_kvcache_decode.py --matrix final-broad --num-shards $(SM75_GRIND_NUM_SHARDS) --shard-index $(SM75_GRIND_SHARD_INDEX) --warmup 50 --iters 200 --output $(SM75_GRIND_KVCACHE_RESULTS)
	$(PYTHON) -c 'import json,sys; p=sys.argv[1]; rows=json.load(open(p))["rows"]; rows=sorted(rows,key=lambda r:r.get("speedup_vs_sdpa",r.get("speedup_vs_baseline",0))); print("\n".join("{:.4f} {} {}".format(r.get("speedup_vs_sdpa",r.get("speedup_vs_baseline",0)), r.get("name"), r.get("backend")) for r in rows[:30])); raise SystemExit(1 if rows and rows[0].get("speedup_vs_sdpa",rows[0].get("speedup_vs_baseline",0)) < 0.85 else 0)' $(SM75_GRIND_KVCACHE_RESULTS)
	$(PYTHON) benchmarks/compare_sm75_results.py $(SM75_GRIND_KVCACHE_RESULTS) --mode decode --decode-gate 1.15
