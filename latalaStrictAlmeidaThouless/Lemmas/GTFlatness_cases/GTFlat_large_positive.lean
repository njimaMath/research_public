import Lemmas.GTFlatness_cases.GTFlat_small_positive

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Large positive overlaps `q < v ≤ 1` -/

/-- Uniform quadratic gap on the upper positive-away region `q + ε < v ≤ 1`. -/
lemma gtFunctional_upper_positive_away_quadratic_gap
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h q s v ε : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      0 < ε →
      q + ε < v →
      v ≤ 1 →
      ∃ lam ∈ Icc (-1 : ℝ) 1,
        gtFunctional β h q s lam v ≤
          2 * rsPathValue β h q s - c * ε ^ 2 := by
  obtain ⟨a, ha, haway⟩ :=
    flatness_deriv_gtFunctional_zero_upper_away data

  refine ⟨a ^ 2 / 5, by positivity, ?_⟩
  intro β h q s v ε hp hq hs hε hqεv hv1

  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp

  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp

  have hq0 : 0 ≤ q := by
    rw [hq]
    exact (rsQ_mem_Icc β h).1

  have hq1 : q ≤ 1 := by
    rw [hq]
    exact (rsQ_mem_Icc β h).2

  have hqpos : 0 < q := by
    rw [hq]
    exact rsQ_pos hβ hh

  have hq_lt_one : q < 1 := by
    rw [hq]
    exact rsQ_lt_one hβ hh

  have hv0 : 0 ≤ v := by
    linarith

  have hvpos : 0 < v := by
    linarith

  have hqv : q < v := by
    linarith

  have hvIcc : v ∈ Icc (-1 : ℝ) 1 := by
    exact ⟨by linarith, hv1⟩

  have hvIcc01 : v ∈ Icc (0 : ℝ) 1 := by
    exact ⟨hv0, hv1⟩

  let F : ℝ → ℝ := fun l =>
    gtFunctional β h q s l v

  let d : ℝ := deriv F 0

  let lam : ℝ := -(2 / 5 : ℝ) * d

  have hgap : a * ε ≤ -d := by
    simpa [d, F] using
      haway hp hq hs hε hqεv hv1

  have hdneg : d < 0 := by
    have hae : 0 < a * ε := mul_pos ha hε
    linarith

  have hd : |d| ≤ 2 := by
    dsimp [d, F]
    exact
      abs_deriv_gtFunctional_le_two β h q s 0 v
        (by simpa [abs_le] using hvIcc)

  have hd_bounds : -2 ≤ d ∧ d ≤ 2 :=
    abs_le.mp hd

  have hlam : lam ∈ Icc (-1 : ℝ) 1 := by
    dsimp [lam]
    constructor <;>
      nlinarith [hd_bounds.1, hd_bounds.2]

  have ht :=
    flatness_gtFunctional_taylor_upper
      β h q s v lam

  change
    F lam ≤
      F 0 + d * lam + (5 / 4 : ℝ) * lam ^ 2
    at ht

  have hopt :
      d * lam + (5 / 4 : ℝ) * lam ^ 2 =
        -(d ^ 2) / 5 := by
    dsimp [lam]
    ring

  have ht' :
      F lam ≤ F 0 - d ^ 2 / 5 := by
    nlinarith [ht, hopt]

  have hsq :
      a ^ 2 * ε ^ 2 ≤ d ^ 2 := by
    have hmul :=
      mul_self_le_mul_self
        (mul_nonneg ha.le hε.le)
        hgap
    calc
      a ^ 2 * ε ^ 2
          = (a * ε) * (a * ε) := by
              ring
      _ ≤ (-d) * (-d) := hmul
      _ = d ^ 2 := by
              ring

  have hzero0 :
      gtFunctional β h q s 0 v =
        2 * rsPathValue β h q s := by
    apply flatness_gtFunctional_zero_eq_two_rsPathValue
    all_goals
      first
      | assumption
      | exact hqpos
      | exact hq_lt_one
      | exact ⟨hqpos, hq_lt_one⟩
      | exact hvIcc
      | exact hvIcc01
      | linarith

  have hzero :
      F 0 = 2 * rsPathValue β h q s := by
    simpa [F] using hzero0

  refine ⟨lam, hlam, ?_⟩

  change F lam ≤
    2 * rsPathValue β h q s -
      (a ^ 2 / 5) * ε ^ 2

  nlinarith [ht', hsq]

end SpinGlass.AT
