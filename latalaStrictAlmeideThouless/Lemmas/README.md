# Strict-AT formalization status

This directory is the dependency-safe blueprint for the quantitative strict-AT theorem.
`MainResult.lean` is the integration target and `Latala_AT.lean` imports it.

The modules compile with localized proof placeholders for the analytic inputs.  The axiom audit
for `SpinGlass.AT.quantitative_strictAT` therefore reports `sorryAx` together with ordinary Lean
foundations.  No project-specific typeclasses, `axiom` declarations, or `admit`s are present.

The smart path now separates centered disorder from the deterministic external field and uses the
finite-volume covariance from the paper.  The scalar Latała kernel, signed matrix path, half-mass
profile, GT terminal condition and functional, and quadratic coupled pressure use the formulas in
`refs/paper.tex`.  Missing analytic constructions are stated as ordinary lemmas with `sorry`
rather than being exposed as typeclass assumptions or replaced by simpler surrogate formulas.
Their Gaussian differentiation, concentration, and last-spin interpolation proofs remain to be
constructed.

The development currently gives the AT modules a small finite-volume model independent of the
legacy `SpinGlass.SKModel` import.  The legacy Hilbert Gaussian files do not compile with the
repository's current Lean 4.32 toolchain.  Once those files are ported, this local model should
be replaced by a bridge to `SKModel`, and the `RSSmartPathDisorder` constructor from independent
SK and simple disorders should be supplied there.

Suggested proof order:

1. `Replicas`, `RSParameters`, and `UniformData`.
2. `GaussianDerivative` and `FreeEnergyDerivative`.
3. `Cavity/Coefficients` and `Cavity/Stability`.
4. The remaining cavity modules.
5. The scalar sign modules.
6. The GT modules, coupled pressure, and fixed deviation.
7. Absorption, free energy, replicon, and compactness.

After each group, rebuild `LatalaMeetsAT` and inspect the printed axioms of
`SpinGlass.AT.quantitative_strictAT`.
