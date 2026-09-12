#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup.sh --check
       bash scripts/setup.sh --install analysis|mac|cuda

--help and --check are offline. No arguments prints this help.
--install downloads Python packages into .venv, never models/data/papers.
Requires an existing Python 3.11+; set PYTHON_BIN to choose an executable.
USAGE
}

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$1" == "--check" && $# -eq 1 ]]; then
  exec "$python_bin" "$repo_root/scripts/doctor.py"
fi

if [[ "$1" != "--install" || $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

profile="$2"
case "$profile" in
  analysis|mac|cuda) ;;
  *) usage >&2; exit 2 ;;
esac

"$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or newer is required")'
if [[ "$profile" == "mac" && "$(uname -s)" != "Darwin" ]]; then
  printf '%s\n' 'The mac profile requires macOS.' >&2
  exit 2
fi
if [[ "$profile" == "cuda" && "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' 'The cuda profile is intended for Linux with an existing NVIDIA driver.' >&2
  exit 2
fi

cd "$repo_root"
if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi
venv_python="$repo_root/.venv/bin/python"
"$venv_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Existing .venv requires Python 3.11+")'

requirements_file="requirements/research.txt"
if [[ "$profile" == "analysis" ]]; then
  requirements_file="requirements/analysis.txt"
fi

"$venv_python" -m pip install -r "$requirements_file"
"$venv_python" -m pip install --no-deps -e .
"$venv_python" -m pip check
"$venv_python" scripts/record_environment.py
if [[ "$profile" == "analysis" ]]; then
  "$venv_python" scripts/doctor.py --backend analysis
else
  "$venv_python" scripts/doctor.py --backend "$profile" --require-research
fi
printf '%s\n' 'Setup complete. No model, dataset, or paper was downloaded.'
