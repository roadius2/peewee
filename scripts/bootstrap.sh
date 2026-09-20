#!/usr/bin/env bash
# Set up ~/dev_projects/ultra_laya as a working clone for a Claude Code session on a GPU box.
#   curl -fsSL https://raw.githubusercontent.com/roadius2/ultra_laya/claude/main/scripts/bootstrap.sh | bash
# or, from an existing clone:  scripts/bootstrap.sh
set -euo pipefail
DEST="${ULTRA_LAYA_DIR:-$HOME/dev_projects/ultra_laya}"
BRANCH="${ULTRA_LAYA_BRANCH:-claude/main}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"   # set to .../whl/cpu on a CPU box

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/roadius2/ultra_laya "$DEST"
fi
cd "$DEST"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
git config user.name "${GIT_USER_NAME:-roadius2}"
git config user.email "${GIT_USER_EMAIL:-roadius2@users.noreply.github.com}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install torch --index-url "$TORCH_INDEX"
pip install -e ".[dev,server,onnx]" datasets
python -m pytest -q -p no:warnings

cat <<MSG

ultra_laya is ready in $DEST (branch $BRANCH, venv .venv).
Next:  cd $DEST && . .venv/bin/activate && claude
Then read CLAUDE.md, docs/HANDOFF.md and docs/GPU_VALIDATION.md, and run:
       python scripts/gpu_validate.py --out reports/\$(hostname)-\$(date +%Y%m%d)
MSG
