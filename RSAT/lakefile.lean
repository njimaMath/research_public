import Lake

open Lake DSL

package QuantitativeStrictAT where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.32.1"

lean_lib GeneralizedSupport where
  srcDir := "../generalizedLatala"
  globs := #[
    .one `SpinGlass.Mathlib.Probability.Distributions.GaussianIntegrationByParts,
    .one `SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert,
    .one `SpinGlass.Defs
  ]

lean_lib QuantitativeStrictAT where
  globs := #[
    .submodules `Lemmas
  ]
