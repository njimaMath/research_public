import Lemmas.Concentration_Coupled
import Lemmas.GTFlatness
import Lemmas.GTbound.GTBound
import GaussianMax


open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Lemma 3.1: a uniform sublinear bound for the quadratically coupled
pressure along the replica-symmetric smart path. Do not change the claim-/
theorem quadraticCoupledPressure_sublinear
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ ρ₀ > 0, ∃ C > 0,
      ∀ {N : ℕ}, 0 < N →
      ∀ {β h : ℝ}, (β, h) ∈ K →
      ∀ s : Set.Icc (0 : ℝ) 1,
      ∀ path :
        RSSmartPathDisorder Ω N β h (rsQ β h),
        quadraticCoupledPressure path s.1 ρ₀ ≤
          rsPathValue β h (rsQ β h) s.1 +
            C * Real.sqrt
              (Real.log ((N : ℝ) + 1) / (N : ℝ)) := by
  sorry

end SpinGlass.AT
