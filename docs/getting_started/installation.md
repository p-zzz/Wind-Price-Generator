# Installation

Requires Python **3.11 or newer**; the shipped models were tested on 3.12.

```bash
git clone https://github.com/p-zzz/Wind-Price-Generator.git
cd Wind-Price-Generator
python -m venv venv
source venv/bin/activate
pip install -e .
```

On Windows, activate with `venv\Scripts\activate` instead.

## With Anaconda

Open a terminal (on Windows: **Anaconda Prompt**) and run:

```bash
conda create -n dk1gen python=3.12
conda activate dk1gen
git clone https://github.com/p-zzz/Wind-Price-Generator.git
cd Wind-Price-Generator
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

Use `pip` inside the conda environment as shown, not `conda install`, for the
package's dependencies: the exact versions it needs are pinned for pip. Without
git, download the repository as a ZIP from GitHub (**Code -> Download ZIP**), unzip
it, and `cd` into the folder instead of cloning. Later sessions only need
`conda activate dk1gen`.

## Pinned dependencies

The dependencies in `pyproject.toml` are **pinned exactly** on purpose. The fitted
models in `models/` contain pickled statsmodels/patsy objects that break across major
version bumps (for example, statsmodels 0.15 with pandas 3 fails to load them).

## CPU-only PyTorch

The generator runs on CPU; a GPU is optional and not needed. By default `pip`
installs the CUDA build of PyTorch, which is several GB. On a machine without an
NVIDIA GPU, install the CPU build instead:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

## Check the install

```bash
python -c "import torch, pandas, statsmodels; print(torch.__version__, pandas.__version__, statsmodels.__version__)"
python examples/run_example.py
```

The example runs a one-year scenario in well under a minute on a laptop CPU and
writes a CSV to `outputs/`.

## Building these docs

```bash
pip install -e ".[docs]"
cd docs && make html     # open docs/_build/html/index.html
```
