# experimental

This directory is the repo-only incubator for new indicator ideas.

Rules:
- keep experimental work out of the published `v1indicators` package
- do not let published package modules import from `experimental/`
- promote stable work from here into `v1indicators/foundational/...` or `v1indicators/derived/...`

Because this directory sits outside the `v1indicators` package, it is not included in the PyPI build.
