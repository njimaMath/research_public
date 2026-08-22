import Lake

open Lake DSL

package LatalaMeetsAT where
  packagesDir := "../../.lake/packages"

require mathlib from "../../.lake/packages/mathlib"

lean_lib GeneralizedSupport where
  srcDir := "../generalizedLatala"
  globs := #[
    .one `SpinGlass.Mathlib.Probability.Distributions.GaussianIntegrationByParts,
    .one `SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert,
    .one `SpinGlass.Defs
  ]

lean_lib LatalaMeetsAT where
  globs := #[
    .one `SpinGlass.SKModel,
    .one `SpinGlass.GuerraBound,
    .one `SpinGlass.Calculus,
    .one `SpinGlass.Replicas,
    .submodules `SpinGlassAT,
    .submodules `Lemmas,
    .one `Latala_AT
  ]
