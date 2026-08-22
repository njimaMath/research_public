import Lemmas.GTFlatness_cases.GTFlatnessCore
import Lemmas.GTFlatness_cases.GTFlat_endpoints
import Lemmas.GTFlatness_cases.GTFlat_small_positive
import Lemmas.GTFlatness_cases.GTFlat_small_negative
import Lemmas.GTFlatness_cases.GTFlat_large_negative

/-!
# GT flatness

This public module collects the shared GT-flatness theory and its
case-specific consequences. Case modules depend only on
`GTFlatnessCore`, avoiding an import cycle while preserving the original
single-import API.
-/

namespace SpinGlass.AT

open Set

/-- A uniform quadratic improvement of the GT functional away from the
replica-symmetric overlap.  The branchwise estimates used to establish this
statement are exported by the case modules above. -/
theorem gtFunctional_uniform_quadratic_gap {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam ∈ Icc (-1 : ℝ) 1, gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  sorry

end SpinGlass.AT
