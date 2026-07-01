# ML4T Project - Claude Code Instructions

## Project

Machine Learning for Algorithmic Trading (3rd Edition) by Stefan Jansen.
Repository of Jupyter notebooks covering data science, ML, deep learning,
NLP, and reinforcement learning applied to trading strategies.

## Environment Setup

**Before doing any work**, check if the Python environment is set up:

```bash
# Check if .venv exists and has packages
.venv/bin/python -c "import pandas; print('Environment OK')" 2>/dev/null || \
.venv/Scripts/python.exe -c "import pandas; print('Environment OK')" 2>/dev/null
```

If the environment is NOT set up, follow the full instructions in
**`docs/environment-setup.md`**. Here is the summary for quick reference:

### Quick Setup (macOS)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 2. Install Python 3.12 and create venv
uv python install 3.12
uv venv --python 3.12 .venv

# 3. Temporarily edit pyproject.toml (required - project says >=3.14 but we use 3.12)
sed -i.bak 's/requires-python = ">=3.14"/requires-python = ">=3.12"/' pyproject.toml

# 4. Install all dependencies
uv pip install -e "." --python .venv/bin/python

# 5. Restore pyproject.toml
mv pyproject.toml.bak pyproject.toml

# 6. Verify
.venv/bin/python -c "import torch, sklearn, lightgbm, xgboost; print('All OK')"
```

### Quick Setup (Windows - Git Bash / MINGW64)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install 3.12
uv venv --python 3.12 .venv
sed -i.bak 's/requires-python = ">=3.14"/requires-python = ">=3.12"/' pyproject.toml
uv pip install -e "." --python .venv/Scripts/python.exe
mv pyproject.toml.bak pyproject.toml
```

### Important Notes

- **Do NOT use `uv sync`** -- it enforces the lockfile which targets Python
  3.14 and will fail to build packages without a C compiler.
- Use `uv pip install -e .` instead, which resolves dependencies fresh.
- The `.env` file is already in the repo with blank API keys. Edit it to add
  keys only if needed for specific chapters.
- If `import talib` fails on macOS, run: `brew install ta-lib`

## Project Structure

- `01_*` through `27_*` -- Chapter directories with notebooks
- `data/` -- Data directory (datasets downloaded at runtime)
- `utils/` -- Shared utility modules
- `case_studies/` -- End-to-end case study pipelines
- `docs/` -- Documentation including `environment-setup.md`
- `.env` -- Environment variables and API keys (blank by default)
- `pyproject.toml` -- Python dependencies (194 packages)

## Running Notebooks

```bash
# Launch Jupyter Lab
.venv/bin/jupyter lab                    # macOS/Linux
.venv/Scripts/jupyter.exe lab            # Windows

# Run a specific notebook via papermill
.venv/bin/python <chapter_dir>/<notebook>.py
```

## Key Technical Details

- Python version: 3.12 (not 3.14 as pyproject.toml states -- see docs/environment-setup.md)
- Package manager: uv (not pip, not conda)
- Virtual environment: .venv/ in project root
- Total packages: ~474
- PyTorch: CPU on Windows, MPS (Metal) on Apple Silicon
