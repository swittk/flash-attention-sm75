
PYTHON ?= python3
SM75_ATTENTION_RESULTS ?= /tmp/sm75_attention_micro.json
SM75_KVCACHE_RESULTS ?= /tmp/sm75_kvcache_decode.json
SM75_EVIDENCE_DIR ?= ../evidence-sm75
SM75_GRIND_ATTENTION_RESULTS ?= $(SM75_EVIDENCE_DIR)/sm75_grind_attention.json
SM75_GRIND_KVCACHE_RESULTS ?= $(SM75_EVIDENCE_DIR)/sm75_grind_kvcache.json
SM75_GRIND_NUM_SHARDS ?= 1
SM75_GRIND_SHARD_INDEX ?= 0

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	$(PYTHON) setup.py sdist

upload_package: create_dist
	twine upload dist/*

.PHONY: sm75-install-dev sm75-test-smoke sm75-test-correctness sm75-test-native-only sm75-bench sm75-sanitize sm75-profile sm75-ci sm75-grind

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
