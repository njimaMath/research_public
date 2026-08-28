import Lake

open Lake DSL

package QuantitativeStrictAT where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.32.1"

lean_lib QuantitativeStrictAT where
  globs := #[
    .submodules `Lemmas
  ]
