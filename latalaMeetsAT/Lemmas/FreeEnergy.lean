import Lemmas.Absorption

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def rsFreeEnergy (β h : ℝ) : ℝ :=
  rsPathValue β h (rsQ β h) 1

noncomputable def skFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : ℝ := pathFreeEnergy path 1

theorem rs_freeEnergy_error {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      0 ≤ rsFreeEnergy β h - skFreeEnergy path ∧
      rsFreeEnergy β h - skFreeEnergy path ≤ M / N := by
  -- Paper route, equation (freeenergyidentity): integrate
  -- `smartPath_freeEnergy_deriv` from `0` to `1` and use the product endpoint
  -- identity
  -- `pathFreeEnergy path 0 = log 2 + E log cosh (h + β*sqrt q*Z)`.
  -- Rearrangement gives exactly
  -- `rsFreeEnergy - skFreeEnergy =
  --   β^2/4 * ∫ s in 0..1, overlapSecondMoment path s`.
  -- Nonnegativity of the integrand proves the lower bound.  Apply
  -- `uniform_secondMoment`, the uniform bound on `β`, and interval length one
  -- for the upper bound.
  --
  -- The product endpoint identity is false for the present centered-only
  -- `path.H`, which omits the deterministic external field.  Repair the smart
  -- path model as noted in `FreeEnergyDerivative`, then add endpoint continuity
  -- so the fundamental theorem of calculus applies on the closed interval.
  sorry

end SpinGlass.AT
