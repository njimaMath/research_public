import Lemmas.FreeEnergyDerivative

open MeasureTheory ProbabilityTheory Real

set_option autoImplicit false

namespace SpinGlass.AT

/-- Explicit two-piece scalar semigroup used by the RS trial path. -/
noncomputable def scalarPsi (β q s u x : ℝ) : ℝ :=
  if q ≤ u then Real.log (Real.cosh x) + s * β ^ 2 / 2 * (1 - u)
  else standardGaussianExpectation (fun z =>
    Real.log (Real.cosh (x + β * Real.sqrt s * Real.sqrt (q - u) * z)) +
      s * β ^ 2 / 2 * (1 - q))

theorem scalarPsi_eq_upper {β q s u x : ℝ} (hu : q ≤ u) :
    scalarPsi β q s u x =
      Real.log (Real.cosh x) + s * β ^ 2 / 2 * (1 - u) := by
  -- Proof route: unfold `scalarPsi`; `hu` selects the upper branch.
  simp [scalarPsi, hu]

theorem scalarPsi_eq_lower {β q s u x : ℝ} (hu : u < q) :
    scalarPsi β q s u x = standardGaussianExpectation (fun z =>
      Real.log (Real.cosh (x + β * Real.sqrt s * Real.sqrt (q - u) * z)) +
        s * β ^ 2 / 2 * (1 - q)) := by
  -- Proof route: unfold `scalarPsi`; `not_le_of_gt hu` selects the lower branch.
  simp [scalarPsi, not_le_of_gt hu]

noncomputable def scalarTrialValue (β h q s : ℝ) : ℝ :=
  Real.log 2 + standardGaussianExpectation (fun z =>
    scalarPsi β q s 0 (h + β * Real.sqrt ((1 - s) * q) * z)) -
      s * β ^ 2 / 2 * ((1 - q ^ 2) / 2)

theorem scalarTrialValue_eq (β h q s : ℝ)
    (hq : 0 ≤ q) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    scalarTrialValue β h q s = rsPathValue β h q s := by
  sorry

end SpinGlass.AT
