# Strict-AT formalization status

This directory is the dependency-safe blueprint for the quantitative strict-AT theorem.
`MainResult.lean` is the integration target and `Latala_AT.lean` imports it.

The current milestone is structural, not certified.  The modules compile, but named analytic
and probabilistic proof obligations still use `sorry`.  Consequently the axiom audit for
`SpinGlass.AT.quantitative_strictAT` reports `sorryAx` together with ordinary Lean foundations.
No project-specific `axiom` declarations and no `admit`s are present.

Every current `sorry` now has an adjacent proof-route comment keyed to the notation and equation
labels in `refs/paper.tex`.  Comments headed with a repair requirement or API gap record a
prerequisite that must be resolved before the indicated proof is attempted.  In particular, the
comments flag the centered versus full Hamiltonian mismatch, the placeholder quadratic pressure
and GT functional, the missing last-spin interpolation, and hypotheses absent from several scalar
and cavity statements.

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
