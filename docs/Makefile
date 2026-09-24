# Minimal Sphinx Makefile. Local: non-strict (-n shows missing refs as warnings).
# CI builds with -W (warnings are errors) via SPHINXOPTS.
SPHINXOPTS    ?= -n --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" api/_generated

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
