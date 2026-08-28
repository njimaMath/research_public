#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Building QuantitativeStrictAT"
lake build QuantitativeStrictAT

echo "Checking Main.lean"
lake env lean Main.lean

echo "Scanning project Lean sources for placeholders and local axioms"
pattern='(^|[^[:alnum:]_])(sorry|admit|sorryAx)([^[:alnum:]_]|$)|^[[:space:]]*axiom[[:space:]]'
found=0
while IFS= read -r -d '' source_file; do
  if grep -En "$pattern" "$source_file"; then
    found=1
  fi
done < <(find . -path './.lake' -prune -o -type f -name '*.lean' -print0)

if (( found != 0 )); then
  echo "Verification failed: inappropriate placeholders or project-local axioms were found above." >&2
  exit 1
fi

echo "Verification succeeded: build, public endpoint, and source-integrity checks passed."
