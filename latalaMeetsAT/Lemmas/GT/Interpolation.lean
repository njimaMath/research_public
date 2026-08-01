import Lemmas.GT.HalfCascade

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- The solution of the finite two-dimensional Parisi recursion associated to
`signedMatrixPath v`, `gtMassParameter q v`, and `gtTerminal lam`. -/
noncomputable def gtSemigroupSolution
    (β q s lam v u x₁ x₂ : ℝ) : ℝ := by
  sorry

/-- The specialized Guerra--Talagrand functional from the paper. -/
noncomputable def gtFunctional (β h q s lam v : ℝ) : ℝ :=
  2 * Real.log 2 + standardGaussianExpectation (fun z =>
    gtSemigroupSolution β q s lam v 0
      (h + β * Real.sqrt ((1 - s) * q) * z)
      (h + β * Real.sqrt ((1 - s) * q) * z)) -
    lam * v - gtCorrection β q s

theorem twoReplica_GT_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s v : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hv : v ∈ attainableOverlaps N) :
    expectedConstrainedFreeEnergy path s v ≤ gtFunctional β h q s 0 v := by
  -- Paper route: specialize Lemma (specialGT) to the signed matrix path
  -- (GTpath) and half-mass profile (halfparameter).  Build the finite cascade,
  -- identify both endpoints, and use Gaussian interpolation: the derivative
  -- is minus one half of a sum of squares, equation (GTderivative), hence is
  -- nonpositive.  At multiplier zero the finite semigroup factorizes into two
  -- scalar RS semigroups, and the correction is equation (GTcorrection), which
  -- produces `2 * rsPathValue`.
  sorry

end SpinGlass.AT
