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
    (data : UniformATData K) (C : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data C) :
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
  -- `fullPathHamiltonian` now includes the deterministic external field, but
  -- the endpoint factorization and closed-interval derivative extension are
  -- not yet proved.
  -- BLOCKED: the open-interval derivative theorem is unfinished and neither
  -- endpoint identity has a proved continuity extension.
  -- NEEDED: `D_N(0)=0`, endpoint continuity, and the interval fundamental
  -- theorem of calculus applied to `rsGap_deriv`.
  -- BLUEPRINT: equation `freeenergyidentity`.
  sorry

end SpinGlass.AT
