import Lemmas.GTFlatness

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Overlaps `0 ≤ v < q` -/

/-- Strict positivity of the endpoint multiplier derivative on the small
positive-overlap branch. -/
lemma flatness_deriv_gtFunctional_zero_pos_of_mem_Ico_zero_q
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    {β h q s v : ℝ}
    (hp : (β, h) ∈ K)
    (hq : q = rsQ β h)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ Set.Ico 0 q) :
    0 <
      deriv
        (fun lam => gtFunctional β h q s lam v) 0 := by
  subst q
  exact
    (flatness_deriv_gtFunctional_zero_sign
      data hp hs).1 v hv

end SpinGlass.AT
