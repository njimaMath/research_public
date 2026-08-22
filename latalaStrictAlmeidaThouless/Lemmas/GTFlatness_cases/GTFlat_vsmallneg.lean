import Lemmas.GTFlatnessCore

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Interior negative-overlap derivatives -/

/-- Extracts the numerical derivative from a supplied derivative formula for
the negative-overlap correlation. -/
lemma flatnessTildeGDeriv_eq_deriv
    (β h s v D : ℝ)
    (hD :
      HasDerivAt
        (fun u => flatnessTildeG β h (rsQ β h) s u)
        D v) :
    deriv
        (fun u => flatnessTildeG β h (rsQ β h) s u) v = D := by
  exact hD.deriv

/-- On `-q ≤ v < 0`, the endpoint multiplier derivative is
`flatnessTildeG β h q s v - v`. -/
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg
    (β h q s v : ℝ)
    (hq : q ∈ Set.Ioo (0 : ℝ) 1)
    (hv : v ∈ Set.Ico (-q) 0) :
    deriv (fun lam => gtFunctional β h q s lam v) 0 =
      flatnessTildeG β h q s v - v := by
  by_cases hvleft : v = -q
  · subst v
    have habs : |(-q : ℝ)| = q := by
      rw [abs_neg, abs_of_pos hq.1]
    have hqabs : q ≤ |(-q : ℝ)| := by
      rw [habs]
    have habs1 : |(-q : ℝ)| < 1 := by
      rw [habs]
      exact hq.2
    rw [flatness_deriv_gtFunctional_zero_q_le_abs_v_lt_one
      β h q s (-q) hq.1 hqabs habs1]
    apply congrArg (fun y : ℝ => y - (-q))
    unfold flatnessTildeG
    rw [habs]
    have hzero : gtIncrementScale β s q q = 0 := by
      simp [gtIncrementScale]
    rw [hzero]
    simp [standardGaussianExpectation]
  · have hvneg : v < 0 := hv.2
    have hvne : v ≠ 0 := ne_of_lt hvneg
    have hv0 : 0 < |v| := by
      exact abs_pos.mpr hvne
    have hminusqv : -q < v := by
      exact lt_of_le_of_ne hv.1 (Ne.symm hvleft)
    have hvq : |v| < q := by
      rw [abs_of_neg hvneg]
      linarith
    simpa [flatnessTildeG] using
      (flatness_deriv_gtFunctional_zero_abs_v_lt_q
        β h q s v hv0 hvq)

/-- Canonical form of the small-negative-overlap derivative formula. -/
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg_rsQ
    (β h s v : ℝ)
    (hβ : 0 < β) (hh : 0 < h)
    (hv : v ∈ Set.Ico (-(rsQ β h)) 0) :
    deriv
        (fun lam =>
          gtFunctional β h (rsQ β h) s lam v) 0 =
      flatnessTildeG β h (rsQ β h) s v - v := by
  exact
    flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg
      β h (rsQ β h) s v
      ⟨rsQ_pos hβ hh, rsQ_lt_one hβ hh⟩
      hv

end SpinGlass.AT
