import Lemmas.GTFlatness_cases.GTFlatnessCore

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Small positive overlaps `0 ≤ v < q` -/

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

/-- Uniform derivative gap on the lower positive-away region
`0 ≤ v < q - ε`. -/
lemma flatness_deriv_gtFunctional_zero_lower_away
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c : ℝ, 0 < c ∧
      ∀ {β h q s v ε : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Set.Icc (0 : ℝ) 1 →
        0 < ε →
        0 ≤ v →
        v < q - ε →
        c * ε ≤
          deriv (fun lam => gtFunctional β h q s lam v) 0 := by
  obtain ⟨c, hc, hsep⟩ :=
    scalarOrderParameterCorrect_global_separation data
  refine ⟨c, hc, ?_⟩
  intro β h q s v ε hp hq hs hε hv0 hvqε
  subst q
  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp
  have hq1 : rsQ β h ≤ 1 :=
    (rsQ_mem_Icc β h).2
  have hv1 : v ≤ 1 := by
    linarith
  have hvIcc : v ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨hv0, hv1⟩
  have hvq : v < rsQ β h := by
    linarith
  have hmain := hsep hp hs hvIcc
  rw [flatness_deriv_gtFunctional_zero_eq_g_sub
    β h s v hβ hh hs hvIcc]
  have hsign :
      0 ≤ scalarOrderParameterCorrect β h s v - v := by
    have h :=
      (scalarOrderParameterCorrect_sign data hp hs).1 v ⟨hv0, hvq⟩
    exact h.le
  rw [abs_of_nonpos (by linarith : v - rsQ β h ≤ 0),
    abs_of_nonneg hsign] at hmain
  have hmain' :
      c * (rsQ β h - v) ≤ scalarOrderParameterCorrect β h s v - v := by
    nlinarith [hmain]
  have hdist : ε ≤ rsQ β h - v := by
    linarith
  exact
    (mul_le_mul_of_nonneg_left hdist hc.le).trans hmain'

/-- Uniform derivative gap on the upper positive-away region
`q + ε < v ≤ 1`. -/
lemma flatness_deriv_gtFunctional_zero_upper_away
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c : ℝ, 0 < c ∧
      ∀ {β h q s v ε : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Set.Icc (0 : ℝ) 1 →
        0 < ε →
        q + ε < v →
        v ≤ 1 →
        c * ε ≤
          -deriv (fun lam => gtFunctional β h q s lam v) 0 := by
  obtain ⟨c, hc, hsep⟩ :=
    scalarOrderParameterCorrect_global_separation data
  refine ⟨c, hc, ?_⟩
  intro β h q s v ε hp hq hs hε hqεv hv1
  subst q
  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp
  have hq0 : 0 ≤ rsQ β h :=
    (rsQ_mem_Icc β h).1
  have hv0 : 0 ≤ v := by
    linarith
  have hvIcc : v ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨hv0, hv1⟩
  have hqv : rsQ β h < v := by
    linarith
  have hmain := hsep hp hs hvIcc
  have hsign :
      scalarOrderParameterCorrect β h s v - v ≤ 0 := by
    have h :=
      (scalarOrderParameterCorrect_sign data hp hs).2.2 v ⟨hqv, hv1⟩
    exact h.le
  rw [abs_of_nonneg (by linarith : 0 ≤ v - rsQ β h),
    abs_of_nonpos hsign] at hmain
  have hdist : ε ≤ v - rsQ β h := by
    linarith
  calc
    c * ε ≤ c * (v - rsQ β h) :=
      mul_le_mul_of_nonneg_left hdist hc.le
    _ ≤ -(scalarOrderParameterCorrect β h s v - v) := hmain
    _ = -deriv
        (fun lam => gtFunctional β h (rsQ β h) s lam v) 0 := by
      rw [flatness_deriv_gtFunctional_zero_eq_g_sub
        β h s v hβ hh hs hvIcc]

/-- Uniform linear separation of the multiplier derivative from zero on the
entire nonnegative-overlap branch. -/
lemma flatness_deriv_gtFunctional_zero_positive_global_separation
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c : ℝ, 0 < c ∧
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Set.Icc (0 : ℝ) 1 →
        v ∈ Set.Icc (0 : ℝ) 1 →
        c * |v - q| ≤
          |deriv (fun lam => gtFunctional β h q s lam v) 0| := by
  obtain ⟨c, hc, hsep⟩ :=
    scalarOrderParameterCorrect_global_separation data
  refine ⟨c, hc, ?_⟩
  intro β h q s v hp hq hs hv
  subst q
  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp
  rw [flatness_deriv_gtFunctional_zero_eq_g_sub
    β h s v hβ hh hs hv]
  exact hsep hp hs hv

end SpinGlass.AT
