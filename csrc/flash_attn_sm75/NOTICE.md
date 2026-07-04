SM75 backend provenance
=======================

This directory contains a small, manually isolated subset of the
flash-attention-turing implementation from:

https://github.com/ssiu/flash-attention-turing

Source revision inspected during import:
52a67d7c8d85d2363d3aa2bf491898f1e826ea69

Only the FP16 SM75 kernel files needed for the native Turing backend were
copied into the normal FlashAttention source tree. The sibling reference
checkout is not a submodule, subtree, package dependency, or vendored repo.
