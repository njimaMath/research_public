import Lemmas.GTbound.Endpoint
import Lemmas.GTbound.HalfEndpoint

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

/-- Canonical finite-volume GT comparison in the branch `|v| < q`. -/
theorem canonical_constrained_le_gtFunctional_of_abs_lt_q
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Ioo (0 : ℝ) 1)
    (hv : v ∈ attainableOverlaps N) (hrq : |v| < q) :
    (∫ x, coupledConstrainedLogPartition N β h q s v x
        ∂SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N)) ≤
      (N : ℝ) * gtFunctional β h q s lam v := by
  have hone := gtOrdinaryPressure_one_eq_canonical
    (N := N) (β := β) (h := h) (q := q) (s := s) (v := v) (lam := lam)
    hN hs ⟨hq.1.le, hq.2.le⟩ hv
  have hcompare := gtConstrainedOrdinaryPressure_one_le_zero
    (N := N) (β := β) (h := h) (q := q) (s := s) (v := v) (lam := lam)
    hN hs hq.1.le hrq.le hv
  have hrelax := gtConstrainedOrdinaryPressure_zero_le_unconstrained
    (N := N) (β := β) (h := h) (q := q) (s := s) (v := v) (lam := lam) hv
  have hend := gtUnconstrainedOrdinaryPressure_zero_add_gap_eq_gtFunctional
    (N := N) (β := β) (h := h) (q := q) (s := s) (v := v) (lam := lam)
    hN hs.1 hq.1 hq.2.le hrq
  rw [hone.symm]
  calc
    gtConstrainedOrdinaryPressure N β h q s v lam hv 1 ≤
        gtConstrainedOrdinaryPressure N β h q s v lam hv 0 +
          (N : ℝ) * (s * β ^ 2 * (1 - q) ^ 2) / 2 := hcompare
    _ ≤ gtUnconstrainedOrdinaryPressure N β h q s v lam 0 +
          (N : ℝ) * (s * β ^ 2 * (1 - q) ^ 2) / 2 :=
        add_le_add hrelax le_rfl
    _ = (N : ℝ) * gtFunctional β h q s lam v := hend

end SpinGlass.AT
