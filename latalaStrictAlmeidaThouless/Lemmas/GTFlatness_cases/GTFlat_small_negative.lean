import Lemmas.GTFlatness_cases.GTFlat_small_positive

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Small negative overlaps `-q ≤ v ≤ 0` -/

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
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg_of_mem_Ico
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
    flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg_of_mem_Ico
      β h (rsQ β h) s v
      ⟨rsQ_pos hβ hh, rsQ_lt_one hβ hh⟩
      hv

/-- On `-q ≤ v ≤ 0`, the endpoint multiplier derivative is
`flatnessTildeG β h q s v - v`. -/
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg
    (β h q s v : ℝ)
    (hq : q ∈ Ioo (0 : ℝ) 1)
    (_hs : s ∈ Icc (0 : ℝ) 1)
    (hv : v ∈ Icc (-q) 0) :
    deriv (fun lam => gtFunctional β h q s lam v) 0 =
      flatnessTildeG β h q s v - v := by
  by_cases hvzero : v = 0
  · subst v
    rw [flatness_deriv_gtFunctional_zero_abs_v_eq_zero β h q s 0 hq.1 abs_zero]
    simp [flatnessTildeG, gtIncrementScale, standardGaussianExpectation]
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
  · have hvneg : v < 0 := lt_of_le_of_ne hv.2 hvzero
    have hvne : v ≠ 0 := ne_of_lt hvneg
    have hv0 : 0 < |v| := abs_pos.mpr hvne
    have hminusqv : -q < v := lt_of_le_of_ne hv.1 (Ne.symm hvleft)
    have hvq : |v| < q := by
      rw [abs_of_neg hvneg]
      linarith
    simpa [flatnessTildeG] using
      (flatness_deriv_gtFunctional_zero_abs_v_lt_q
        β h q s v hv0 hvq)

/-- At the boundary `v = 0`, `flatnessTildeG` is the multiplier derivative
of the GT functional. -/
lemma flatnessTildeG_zero_eq_deriv_gtFunctional_zero
    (β h q s : ℝ)
    (hq : 0 < q) :
    flatnessTildeG β h q s 0 =
      deriv
        (fun lam => gtFunctional β h q s lam 0) 0 := by
  rw [flatness_deriv_gtFunctional_zero_abs_v_eq_zero
    β h q s 0 hq abs_zero]
  simp [flatnessTildeG, gtIncrementScale, standardGaussianExpectation]

/-- Uniform linear separation of the endpoint multiplier derivative from zero
on the negative-overlap branch. -/
lemma flatness_deriv_gtFunctional_zero_negative_global_separation
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c : ℝ, 0 < c ∧
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-q) 0 →
        c * |v - q| ≤
          |deriv (fun lam =>
            gtFunctional β h q s lam v) 0| := by
  obtain ⟨c₀, hc₀, hsep⟩ := scalarOrderParameterCorrect_global_separation data
  refine ⟨min c₀ data.gap, lt_min hc₀ data.gap_pos, ?_⟩
  intro β h q s v hp hq hs hv
  subst q
  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp
  have hqpos : 0 < rsQ β h := rsQ_pos hβ hh
  let f : ℝ → ℝ := fun u => flatnessTildeG β h (rsQ β h) s u
  have hf0 : f 0 = scalarOrderParameterCorrect β h s 0 := by
    dsimp [f]
    rw [flatnessTildeG_zero_eq_deriv_gtFunctional_zero β h (rsQ β h) s hqpos,
      flatness_deriv_gtFunctional_zero_eq_g_sub β h s 0 hβ hh hs
        ⟨le_rfl, zero_le_one⟩]
    ring
  have hzero : 0 < f 0 := by
    rw [hf0]
    simpa using
      (scalarOrderParameterCorrect_sign data hp hs).1 0 ⟨le_rfl, hqpos⟩
  have hbase : c₀ * rsQ β h ≤ f 0 := by
    have hseparation := hsep hp hs (show (0 : ℝ) ∈ Icc (0 : ℝ) 1 by
      exact ⟨le_rfl, zero_le_one⟩)
    rw [← hf0, sub_zero, abs_of_pos hzero] at hseparation
    have habs : |(0 : ℝ) - rsQ β h| = rsQ β h := by
      rw [abs_of_neg (by linarith : 0 - rsQ β h < 0)]
      ring
    rwa [habs] at hseparation
  by_cases hv0 : v = 0
  · subst v
    rw [flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg β h (rsQ β h) s 0
      ⟨hqpos, rsQ_lt_one hβ hh⟩ hs ⟨by linarith, le_rfl⟩]
    change min c₀ data.gap * |0 - rsQ β h| ≤ |f 0 - 0|
    rw [sub_zero, abs_of_pos hzero]
    have hmin : min c₀ data.gap ≤ c₀ := min_le_left _ _
    calc
      min c₀ data.gap * |0 - rsQ β h| = min c₀ data.gap * rsQ β h := by
        rw [abs_of_neg (by linarith : 0 - rsQ β h < 0)]
        ring
      _ ≤ c₀ * rsQ β h :=
        mul_le_mul_of_nonneg_right hmin hqpos.le
      _ ≤ f 0 := hbase
  · have hvneg : v < 0 := lt_of_le_of_ne hv.2 hv0
    have hcont : ContinuousOn f (Icc v 0) := by
      apply (flatnessTildeG_continuousOn_neg β h (rsQ β h) s).mono
      intro u hu
      exact ⟨le_trans hv.1 hu.1, hu.2⟩
    have hdiff : DifferentiableOn ℝ f (Ioo v 0) := by
      intro u hu
      obtain ⟨D, hD⟩ := flatnessTildeG_hasDerivAt_neg β h (rsQ β h) s u
        hβ.le hs hqpos ⟨lt_of_le_of_lt hv.1 hu.1, hu.2⟩
      exact hD.differentiableAt.differentiableWithinAt
    obtain ⟨u, hu, hslope⟩ := exists_deriv_eq_slope f hvneg hcont hdiff
    have hderiv := flatnessTildeG_deriv_lt_one_neg data hp rfl hs
      ⟨lt_of_le_of_lt hv.1 hu.1, hu.2⟩
    have hratio : (f 0 - f v) / (0 - v) ≤ 1 - data.gap := by
      rwa [← hslope]
    have hden : 0 < 0 - v := by linarith
    have hstep := (div_le_iff₀ hden).mp hratio
    have hmain : min c₀ data.gap * (rsQ β h - v) ≤ f v - v := by
      have hmin₀ : min c₀ data.gap ≤ c₀ := min_le_left _ _
      have hminGap : min c₀ data.gap ≤ data.gap := min_le_right _ _
      have hqnonneg : 0 ≤ rsQ β h := hqpos.le
      nlinarith
    rw [flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg β h (rsQ β h) s v
      ⟨hqpos, rsQ_lt_one hβ hh⟩ hs hv]
    have hmain0 : 0 ≤ f v - v := by
      apply le_trans ?_ hmain
      exact mul_nonneg (le_of_lt (lt_min hc₀ data.gap_pos))
        (by linarith : 0 ≤ rsQ β h - v)
    rw [abs_of_nonneg hmain0]
    calc
      min c₀ data.gap * |v - rsQ β h| =
          min c₀ data.gap * (rsQ β h - v) := by
        rw [abs_of_nonpos (by linarith : v - rsQ β h ≤ 0)]
        ring
      _ ≤ f v - v := hmain

end SpinGlass.AT
