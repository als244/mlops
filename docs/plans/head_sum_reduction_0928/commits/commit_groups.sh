#!/usr/bin/env bash
# One commit, from the head-sum-reduction round:
#
#   1  head_loss returns the summed cross entropy on request, for a caller
#      that divides by a total of its own
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
COUNT=1

#  1  head_loss(reduction="sum") on both providers, documented and tested

GROUP_1=(
  docs/OPS.md
  docs/plans/head_sum_reduction_0928/
  src/mlops/head.py
  src/mlops/providers/builtin/head.py
  src/mlops/providers/native_torch/head.py
  tests/test_head.py
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
