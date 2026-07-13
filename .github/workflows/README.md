# GitHub workflows in this fork

Automatic PyPI publishing inherited from upstream is disabled in this fork.
Pushing a tag must not publish either `flash-attn` or `flash-attn-4`.

The remaining wheel workflows are manual build tools. Public SM75 artifacts
should be attached to an experimental GitHub prerelease only after the local
release audit passes and the generated wheel has been reviewed.
