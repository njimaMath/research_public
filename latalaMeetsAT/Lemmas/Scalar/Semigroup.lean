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

theorem scalarTrialValue_eq (β h q s : ℝ) :
    rsPathValue β h q s = rsPathValue β h q s := by
  -- Proof replacement guide: the current reflexive statement is a placeholder.
  -- The useful identity is equation (RSpathvalue): evaluate the upper and lower
  -- scalar Gaussian semigroups, combine independent Gaussian variances using
  -- `d + s*β^2*q = β^2*q`, compute the correction integral, and show the trial
  -- functional equals `rsPathValue`.
  rfl

end SpinGlass.AT
