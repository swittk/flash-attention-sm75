SM75 backend attribution
========================

Parts of the SM75 forward and backward kernel implementation are derived from
SSIU's flash-attention-turing project:

https://github.com/ssiu/flash-attention-turing

This fork adds its own SM75 API integration, dispatch, generic and specialized
kernels, packed and variable-length paths, and KV-cache support.
