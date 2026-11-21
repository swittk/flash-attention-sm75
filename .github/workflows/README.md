# SM75 release automation

The inherited upstream B200/FA4 self-hosted CI and generic wheel workflows were removed; they were not usable for SM75 validation.

For an experimental Turing release, build and test on a real SM75 GPU:

```sh
make sm75-ci-quick
make sm75-release-audit
```

`sm75-release-audit` generates a source-built, installed-wheel release matrix under `/tmp/flash-attn-sm75-release`. Attach the final wheel and evidence to a manually created GitHub prerelease. No PyPI publication is configured.
