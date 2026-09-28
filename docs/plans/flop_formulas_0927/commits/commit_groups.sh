#!/usr/bin/env bash
# Two commits, from the flop-formula round:
#
#   1  one canonical logical cost per operation, and flop formulas as the
#      channel a library declares them through
#   2  a flop formula registered on every custom operator, and the test
#      that every operator has one
#
#   commit_groups.sh --dry-run     show what each group would commit
#   commit_groups.sh               make every group, in order
#   commit_groups.sh 2             make only group 2
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
# ROOT comes from where this script *lives*, not where it is run from, so
# calling it from another checkout would commit into this repository instead.
CALLER_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ "$CALLER_ROOT" != "$ROOT" ]]; then
  echo "refusing: this script commits into $ROOT" >&2
  echo "          but you are in ${CALLER_ROOT:-a non-repository}." >&2
  echo "          To replay elsewhere, copy commits/ into that tree and run the copy." >&2
  exit 1
fi
cd "$ROOT"
COUNT=2

#  1  one canonical logical cost per operation; flop_formula and
#     has_flop_formula in dispatch; the docs that describe them

GROUP_1=(
  docs/API_REFERENCE.md
  docs/ARCHITECTURE.md
  docs/EXTENDING.md
  docs/OPS.md
  docs/PROVIDERS.md
  docs/plans/flop_formulas_0927/
  src/mlops/dispatch/__init__.py
  src/mlops/dispatch/costs.py
  src/mlops/dispatch/logical_costs.py
)

#  2  a flop formula on every custom operator, shared where variants
#     share a cost, and the test that none is missing

GROUP_2=(
  src/mlops/preparation/packed_sequence.py
  src/mlops/providers/builtin/adamw.py
  src/mlops/providers/builtin/cross_entropy.py
  src/mlops/providers/builtin/dsa.py
  src/mlops/providers/builtin/embedding.py
  src/mlops/providers/builtin/flash_attention.py
  src/mlops/providers/builtin/gelu.py
  src/mlops/providers/builtin/head.py
  src/mlops/providers/builtin/layer_norm.py
  src/mlops/providers/builtin/moe.py
  src/mlops/providers/builtin/moe_composed.py
  src/mlops/providers/builtin/partial_rope.py
  src/mlops/providers/builtin/rms_norm.py
  src/mlops/providers/builtin/rope.py
  src/mlops/providers/builtin/swiglu.py
  src/mlops/providers/fla/hybrid.py
  src/mlops/providers/liger/rms_norm.py
  src/mlops/providers/scattermoe/moe.py
  tests/test_flop_formulas.py
)

group_paths() { local name="GROUP_$1[@]"; printf '%s\n' "${!name}"; }

if [[ "${1-}" == "--dry-run" ]]; then
  for index in $(seq 1 "$COUNT"); do
    echo "=== group $index: $(head -1 "$HERE/messages/0$index.txt")"
    group_paths "$index" | sed 's/^/    /'
  done
  echo
  echo "paths in no group (must be empty):"
  comm -23 <(git status --porcelain | awk '{print $NF}' | sort -u) \
           <(for index in $(seq 1 "$COUNT"); do group_paths "$index"; done | sort -u) | sed 's/^/    /'
  echo
  echo "paths in more than one group (must be empty):"
  for index in $(seq 1 "$COUNT"); do group_paths "$index"; done | sort | uniq -d | sed 's/^/    /'
  exit 0
fi

make_group() {
  local index="$1"
  local message="$HERE/messages/0$index.txt"
  [[ -f "$message" ]] || { echo "no message for group $index" >&2; exit 1; }
  local paths=()
  while IFS= read -r path; do paths+=("$path"); done < <(group_paths "$index")
  for path in "${paths[@]}"; do
    # A group lists both names of a rename. The old one matches nothing to
    # add, because the rename is already staged; the commit below still
    # records its deletion, because the old name matches in HEAD. A name
    # that matches nothing anywhere fails there rather than here.
    if [[ -e "$path" ]] || git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
      git add -A -- "$path"
    fi
  done
  git commit --only -F "$message" -- "${paths[@]}"
}

if [[ $# -eq 1 ]]; then
  make_group "$1"
else
  for index in $(seq 1 "$COUNT"); do make_group "$index"; done
fi
