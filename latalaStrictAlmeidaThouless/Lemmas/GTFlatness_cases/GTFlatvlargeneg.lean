import Lemmas.GTFlatnessCore

open MeasureTheory ProbabilityTheory Set

namespace SpinGlass.AT

/-! ### Overlaps `-1 ≤ v ≤ -q` -/

/-- On the large-negative-overlap branch, the distance to `q` is at most two. -/
lemma sub_sq_le_four_of_negative_overlap
    {q v : ℝ}
    (hq0 : 0 ≤ q)
    (hv : v ∈ Icc (-1 : ℝ) (-q)) :
    (v - q) ^ 2 ≤ 4 := by
  have hv_lower : -1 ≤ v := hv.1
  have hv_upper : v ≤ -q := hv.2
  have hq1 : q ≤ 1 := by
    linarith
  have hdiff_lower : -2 ≤ v - q := by
    linarith
  have hdiff_upper : v - q ≤ 0 := by
    linarith
  have hprod :
      0 ≤ ((v - q) + 2) * (2 - (v - q)) := by
    apply mul_nonneg
    · linarith
    · linarith
  nlinarith

end SpinGlass.AT
