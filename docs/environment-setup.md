# Environment Setup Guide (Local, without Docker)

This file documents the exact steps used to set up the ML4T project environment
on a local machine **without Docker** and **without admin rights**. It is
intended to be read by Claude Code to replicate the environment on another
machine (e.g. a Mac at home).

---

## Context and Constraints

- The project's `pyproject.toml` specifies `requires-python = ">=3.14"`, but
  many packages lack pre-built wheels for Python 3.14 on Windows (and some on
  macOS). Building from source requires a C compiler (MSVC on Windows, Xcode
  CLI tools on macOS).
- **Workaround**: use **Python 3.12** via `uv`, which has near-universal
  pre-built wheel coverage across all platforms. Install with `uv pip install`
  (which does not enforce the `requires-python` metadata) rather than
  `uv sync` (which does).
- The project's `uv.lock` is **not used** in this setup because it was
  generated for Python 3.14. Dependency resolution happens fresh via
  `uv pip install -e .` against the dependency list in `pyproject.toml`.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Git | To clone the repository |
| Internet access | To download `uv`, Python, and packages |
| ~3 GB disk | For `.venv` with all 474 packages |
| No admin rights needed | `uv` installs to `~/.local/bin` |

---

## Step-by-Step Instructions

### 1. Install uv

`uv` is a fast Python package manager written in Rust. It replaces pip, venv,
pyenv, and pip-tools in a single binary.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

**Windows (Git Bash / MINGW64):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Restart the shell or add ~/.local/bin to PATH
```

Verify:
```bash
uv --version
# Expected: uv 0.11.x or later
```

### 2. Clone the repository

```bash
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading
```

### 3. Install Python 3.12

`uv` can download and manage Python versions without admin rights.

```bash
uv python install 3.12
```

This installs a standalone CPython 3.12.x build into uv's managed directory.

### 4. Create the virtual environment

```bash
uv venv --python 3.12 .venv
```

You will see a warning about Python 3.12 being incompatible with the project's
`requires-python >= 3.14`. This is expected and safe to ignore.

**Activate the venv (optional but useful for IDE integration):**

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows (Git Bash):
```bash
source .venv/Scripts/activate
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### 5. Temporarily relax the Python version constraint

The `pyproject.toml` must be edited temporarily so `uv pip install` can resolve
dependencies for Python 3.12:

```bash
# On macOS/Linux:
sed -i.bak 's/requires-python = ">=3.14"/requires-python = ">=3.12"/' pyproject.toml

# On Windows (Git Bash):
sed -i.bak 's/requires-python = ">=3.14"/requires-python = ">=3.12"/' pyproject.toml
```

### 6. Install all dependencies

```bash
uv pip install -e "." --python .venv/bin/python    # macOS/Linux
uv pip install -e "." --python .venv/Scripts/python.exe  # Windows
```

This resolves ~474 packages and installs them. On a good connection this
completes in 2-5 minutes.

### 7. Restore pyproject.toml

```bash
mv pyproject.toml.bak pyproject.toml
# Or: git checkout pyproject.toml
```

### 8. Set up the .env file

```bash
cp .env.example .env
```

Edit `.env` to add any API keys you need. All keys are optional. The most
commonly useful ones are:

| Key | Purpose | Where to get it |
|-----|---------|-----------------|
| `FRED_API_KEY` | Federal Reserve economic data | https://fred.stlouisfed.org/docs/api/api_key.html |
| `EDGAR_IDENTITY` | SEC filings (Ch04) | Your name + email, e.g. `"Jane Doe jane@example.org"` |
| `ANTHROPIC_API_KEY` | LLM agent demos (Ch24) | https://console.anthropic.com/ |

### 9. Verify the installation

```bash
# Use the venv python directly (no activation needed):
.venv/bin/python -c "
import sys
print(f'Python: {sys.version}')
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy; print(f'NumPy: {numpy.__version__}')
import pandas; print(f'Pandas: {pandas.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import lightgbm; print(f'LightGBM: {lightgbm.__version__}')
import xgboost; print(f'XGBoost: {xgboost.__version__}')
import polars; print(f'Polars: {polars.__version__}')
import statsmodels; print(f'Statsmodels: {statsmodels.__version__}')
import shap; print(f'SHAP: {shap.__version__}')
import catboost; print(f'CatBoost: {catboost.__version__}')
import plotly; print(f'Plotly: {plotly.__version__}')
import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
print('All OK')
"
```

### 10. Launch Jupyter Lab

```bash
.venv/bin/jupyter lab    # macOS/Linux
.venv/Scripts/jupyter.exe lab  # Windows
# Opens at http://localhost:8888
```

---

## macOS-Specific Notes

### Apple Silicon (M1/M2/M3/M4)

- `uv venv --python 3.12` will download a native ARM64 Python build.
- PyTorch installs the MPS (Metal Performance Shaders) backend automatically.
  To verify GPU acceleration:
  ```python
  import torch
  print(torch.backends.mps.is_available())  # Should be True
  ```
- All 474 packages have ARM64 wheels for Python 3.12.

### macOS Intel

- Same steps as above. No special considerations.

### TA-Lib C library (if needed)

If `import talib` fails with a missing library error, install the C library:

```bash
# With Homebrew (requires Homebrew, no admin needed):
brew install ta-lib

# Then reinstall the Python wrapper:
uv pip install --force-reinstall ta-lib --python .venv/bin/python
```

On Windows, `ta-lib` 0.6.8+ bundles the C library in the wheel, so this step
is not needed.

---

## Corporate Proxy

If behind a corporate proxy, set these before running any `uv` or `curl`
commands:

```bash
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"
# Some proxies also need:
export NO_PROXY="localhost,127.0.0.1"
```

On Windows PowerShell:
```powershell
$env:HTTP_PROXY = "http://your-proxy:port"
$env:HTTPS_PROXY = "http://your-proxy:port"
```

---

## What Was Installed (Reference)

Environment created on 2026-07-01:

| Component | Version |
|-----------|---------|
| uv | 0.11.26 |
| Python | 3.12.13 |
| Total packages | 474 |
| PyTorch | 2.10.0 (CPU on Windows, MPS on Apple Silicon) |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| Polars | 1.42.1 |
| scikit-learn | 1.6.1 |
| LightGBM | 4.6.0 |
| XGBoost | 3.3.0 |
| CatBoost | 1.2.10 |
| Statsmodels | 0.14.6 |
| SHAP | 0.48.0 |
| Transformers | 4.57.6 |
| Plotly | 6.8.0 |
| Matplotlib | 3.11.0 |

---

## Troubleshooting

### `uv pip install` fails with `requires-python >= 3.14`
You forgot step 5. Edit `pyproject.toml` to change `requires-python` to
`">=3.12"` before installing.

### `uv sync` fails with build errors on Windows
Don't use `uv sync`. Use `uv pip install -e .` as described above. `uv sync`
enforces the lockfile which targets Python 3.14 and will try to build packages
from source.

### Package import errors after install
Run the verification script in step 9. If a specific package fails, try:
```bash
uv pip install --force-reinstall <package-name> --python .venv/bin/python
```

### Permission denied on macOS
If `uv` install fails with permission errors on macOS:
```bash
chmod +x ~/.local/bin/uv
```
