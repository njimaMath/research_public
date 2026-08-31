#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Building QuantitativeStrictAT"
lake build QuantitativeStrictAT

echo "Checking Main.lean and the public ConcreteMain API"
{
  cat Main.lean
  cat <<'LEAN'
example (β : ℝ) {h : ℝ} (hh : 0 < h) :
    ConcreteMain.ReplicaSymmetricFixedPointClaim β h :=
  ConcreteMain.replicaSymmetricFixedPointClaim_of_pos_field β hh

example : ConcreteMain.QuantitativeStrictATClaim :=
  ConcreteMain.quantitativeStrictATClaim

example (β h : ℝ) : ConcreteMain.OverlapCLTClaim β h :=
  ConcreteMain.overlapCLTClaim β h
LEAN
} | lake env lean --stdin

echo "Scanning project Lean sources for placeholders and local axioms"
pattern='(^|[^[:alnum:]_])(sorry|admit|sorryAx|axiom)([^[:alnum:]_]|$)'
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

echo "Verification succeeded: build, public endpoint, API, and source-integrity checks passed."
