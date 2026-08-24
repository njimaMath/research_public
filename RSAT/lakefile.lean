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
    .one `SpinGlass.Replicas,
    .one `SpinGlass.AT.Defs,
    .one `SpinGlass.AT.Algebra,
    .one `SpinGlass.AT.SKModel,
    .one `SpinGlass.AT.Calculus,
    .one `SpinGlass.AT.GuerraBound,
    .one `SpinGlass.AT.Gaussian_concentration.gaussian_concentration,
    .one `SpinGlass.AT.Mathlib.Probability.Distributions.GaussianIntegrationByParts,
    .one `SpinGlass.AT.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert,
    .submodules `Lemmas
  ]
