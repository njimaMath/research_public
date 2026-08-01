import Lemmas.Cavity.Coefficients

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def cavityRemainder {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : Fin 3 → ℝ := by
  sorry

theorem cavity_system {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    cavityVector path s - s • (cavityMatrix β q (rsR β h)).mulVec (cavityVector path s) =
      (1 / (N : ℝ)) • theta q (rsR β h) + cavityRemainder path s := by
  -- Construct the remainder by the last-spin interpolation and its integral
  -- Taylor remainder.  It is not defined as the residual of this equation.
  sorry

theorem cavityRemainder_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    ‖cavityRemainder path s‖ ≤
      10 * ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s) := by
  -- Paper route, Proposition (cavity) and equations
  -- (cavitydecomposition)--(cavitydiagonalerror): use site exchangeability to
  -- separate a cavity term and an `N⁻¹` diagonal term for each of `A,B,C`.
  -- Taylor-expand the last-spin interpolation at `u=0`.  Its zeroth-order
  -- cavity terms vanish; the first derivatives give `s*cavityMatrix*x` by the
  -- coefficient table, while the diagonal expectations give `theta/N`.
  -- Replace decoupled quadratic overlaps by endpoint overlaps using
  -- (cavityendpointcomparison).  Bound the integral Taylor remainder by the
  -- corrected second-derivative lemma and the diagonal change using Young's
  -- inequality.  Collect the three coordinate bounds into the function norm.
  --
  -- The numerical constant `10` is not established in the paper, which uses
  -- an unspecified `C_K`, and this statement has no compact-set, fixed-point,
  -- positivity, or `N > 0` assumptions.  Replace `10` by data containing a
  -- uniform constant and add those hypotheses.  The centered/full-Hamiltonian
  -- repair and an explicit cavity disorder extension are also prerequisites.
  sorry

end SpinGlass.AT
