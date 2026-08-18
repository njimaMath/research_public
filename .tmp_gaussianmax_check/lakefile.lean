import Lake

open Lake DSL

package GaussianMaxCheck where
  packagesDir := "../../.lake/packages"

require mathlib from "../../.lake/packages/mathlib"

lean_lib ATGaussianConcentration where
  srcDir := "../latalaStrictAlmeidaThouless/SpinGlassAT/Gaussian_concentration"
  globs := #[
    .one `GaussianConcentrationAux,
    .one `GaussianCovSmooth,
    .one `GaussianCovMollify,
    .one `gaussian_concentration
  ]

lean_lib StrictAT where
  srcDir := "../latalaStrictAlmeidaThouless"
  globs := #[.one `Lemmas.GaussianMax]
