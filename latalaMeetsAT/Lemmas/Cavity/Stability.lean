import Lemmas.Cavity.System

set_option autoImplicit false

namespace SpinGlass.AT

def stabilityOperator (β q r s : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  1 - s • cavityMatrix β q r

theorem cavityMatrix_determinant {β h s : ℝ} :
    Matrix.det (stabilityOperator β (rsQ β h) (rsR β h) s) =
      (1 - s * atParameter β h) *
        (1 - s * β ^ 2 * (1 - 4 * rsQ β h + 3 * rsR β h)) ^ 2 := by
  classical
  simp [stabilityOperator, cavityMatrix, atParameter, rsA,
    Matrix.det_fin_three]
  ring

theorem one_sub_anomalous_lower_bound {K : Set (ℝ × ℝ)}
    (data : UniformATData K) {β h s : ℝ}
    (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    data.gap ≤ 1 - s * atParameter β h := by
  -- Proof route: this is the uniform path gap already proved in `UniformData`.
  exact path_gap data hp hs

set_option maxHeartbeats 800000 in
theorem cavityMatrix_inverse_uniform {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ M > 0, ∀ {β h s : ℝ}, (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ i j, |(stabilityOperator β (rsQ β h) (rsR β h) s)⁻¹ i j| ≤ M := by
  let L : ℝ := 1 + 19 * data.βmax ^ 2
  let M : ℝ := 1 + 2 * L ^ 2 / data.gap ^ 3
  refine ⟨M, ?_, ?_⟩
  · dsimp [M, L]
    have hden : 0 < data.gap ^ 3 := pow_pos data.gap_pos 3
    have hnum : 0 ≤ 2 * (1 + 19 * data.βmax ^ 2) ^ 2 := by positivity
    nlinarith [div_nonneg hnum hden.le]
  intro β h s hp hs i j
  let q := rsQ β h
  let r := rsR β h
  let S := stabilityOperator β q r s
  have hβ : 0 < β := data.β_pos (β, h) hp
  have hh : 0 < h := data.h_pos (β, h) hp
  have hβle : β ≤ data.βmax := data.β_bound (β, h) hp
  have hq0 : 0 ≤ q := (rsQ_mem_Icc β h).1
  have hq1 : q ≤ 1 := (rsQ_mem_Icc β h).2
  have hr0 : 0 ≤ r := by
    dsimp [r, rsR, standardGaussianExpectation]
    exact MeasureTheory.integral_nonneg fun z => by positivity
  have hrq : r ≤ q := rsR_le_rsQ hβ hh
  have hr1 : r ≤ 1 := hrq.trans hq1
  have hβsq : β ^ 2 ≤ data.βmax ^ 2 := by nlinarith
  have hL0 : 0 ≤ L := by
    dsimp [L]
    positivity
  have hb2 : |β ^ 2 * (1 - q ^ 2)| ≤ data.βmax ^ 2 := by
    rw [abs_of_nonneg (mul_nonneg (sq_nonneg β) (by nlinarith))]
    nlinarith [mul_nonneg (sub_nonneg.mpr hq1) hq0]
  have hb1 : |β ^ 2 * (q - q ^ 2)| ≤ data.βmax ^ 2 := by
    rw [abs_of_nonneg (mul_nonneg (sq_nonneg β) (by nlinarith))]
    nlinarith [mul_nonneg (sub_nonneg.mpr hq1) hq0]
  have hb0 : |β ^ 2 * (r - q ^ 2)| ≤ data.βmax ^ 2 := by
    rw [abs_mul, abs_of_nonneg (sq_nonneg β)]
    have habs : |r - q ^ 2| ≤ 1 := by
      rw [abs_le]
      constructor <;> nlinarith [sq_nonneg q, mul_self_le_mul_self hq0 hq1]
    exact (mul_le_of_le_one_right (sq_nonneg β) habs).trans
      (by nlinarith [hβsq, sq_nonneg data.βmax])
  have hsb2 : |s * (β ^ 2 * (1 - q ^ 2))| ≤ data.βmax ^ 2 := by
    rw [abs_mul, abs_of_nonneg hs.1]
    calc
      s * |β ^ 2 * (1 - q ^ 2)| ≤ 1 * data.βmax ^ 2 :=
        mul_le_mul hs.2 hb2 (abs_nonneg _) (by norm_num)
      _ = data.βmax ^ 2 := one_mul _
  have hsb1 : |s * (β ^ 2 * (q - q ^ 2))| ≤ data.βmax ^ 2 := by
    rw [abs_mul, abs_of_nonneg hs.1]
    calc
      s * |β ^ 2 * (q - q ^ 2)| ≤ 1 * data.βmax ^ 2 :=
        mul_le_mul hs.2 hb1 (abs_nonneg _) (by norm_num)
      _ = data.βmax ^ 2 := one_mul _
  have hsb0 : |s * (β ^ 2 * (r - q ^ 2))| ≤ data.βmax ^ 2 := by
    rw [abs_mul, abs_of_nonneg hs.1]
    calc
      s * |β ^ 2 * (r - q ^ 2)| ≤ 1 * data.βmax ^ 2 :=
        mul_le_mul hs.2 hb0 (abs_nonneg _) (by norm_num)
      _ = data.βmax ^ 2 := one_mul _
  have hsb2' := abs_le.mp hsb2
  have hsb1' := abs_le.mp hsb1
  have hsb0' := abs_le.mp hsb0
  have hentry : ∀ a b, |S a b| ≤ L := by
    intro a b
    rw [abs_le]
    constructor
    · dsimp [S, L]
      fin_cases a <;> fin_cases b <;>
        simp [stabilityOperator, cavityMatrix] <;>
        nlinarith [hsb2'.1, hsb2'.2, hsb1'.1, hsb1'.2, hsb0'.1, hsb0'.2]
    · dsimp [S, L]
      fin_cases a <;> fin_cases b <;>
        simp [stabilityOperator, cavityMatrix] <;>
        nlinarith [hsb2'.1, hsb2'.2, hsb1'.1, hsb1'.2, hsb0'.1, hsb0'.2]
  have hrep : data.gap ≤ 1 - s * atParameter β h := path_gap data hp hs
  have hanom0 : data.gap ≤
      1 - s * (β ^ 2 * (1 - 4 * q + 3 * r)) := by
    dsimp [q, r]
    nlinarith [mul_nonneg hs.1
      (sub_nonneg.mpr (anomalous_eigenvalue_le_replicon hβ hh))]
  have hdet : data.gap ^ 3 ≤ Matrix.det S := by
    rw [show Matrix.det S =
        (1 - s * atParameter β h) *
          (1 - s * β ^ 2 * (1 - 4 * q + 3 * r)) ^ 2 by
      simpa [S, q, r] using cavityMatrix_determinant (β := β) (h := h) (s := s)]
    have hrep0 : 0 ≤ 1 - s * atParameter β h := le_trans data.gap_pos.le hrep
    have hsq : data.gap ^ 2 ≤
        (1 - s * (β ^ 2 * (1 - 4 * q + 3 * r))) ^ 2 := by
      simpa [pow_two] using mul_self_le_mul_self data.gap_pos.le hanom0
    calc
      data.gap ^ 3 = data.gap * data.gap ^ 2 := by ring
      _ ≤ (1 - s * atParameter β h) *
          (1 - s * (β ^ 2 * (1 - 4 * q + 3 * r))) ^ 2 :=
        mul_le_mul hrep hsq (sq_nonneg data.gap) hrep0
      _ = (1 - s * atParameter β h) *
          (1 - s * β ^ 2 * (1 - 4 * q + 3 * r)) ^ 2 := by ring
  have hgap3 : 0 < data.gap ^ 3 := pow_pos data.gap_pos 3
  have hdetpos : 0 < Matrix.det S := lt_of_lt_of_le hgap3 hdet
  have hminor (a b c d e f g k : Fin 3) :
      |S a b * S c d - S e f * S g k| ≤ 2 * L ^ 2 := by
    calc
      |S a b * S c d - S e f * S g k| ≤
          |S a b * S c d| + |S e f * S g k| := abs_sub _ _
      _ = |S a b| * |S c d| + |S e f| * |S g k| := by rw [abs_mul, abs_mul]
      _ ≤ L * L + L * L := by
        gcongr
        all_goals exact hentry _ _
      _ = 2 * L ^ 2 := by ring
  have hadj : ∀ a b, |S.adjugate a b| ≤ 2 * L ^ 2 := by
    intro a b
    fin_cases a <;> fin_cases b
    · simpa [Matrix.adjugate_fin_three] using hminor 1 1 2 2 1 2 2 1
    · simpa [Matrix.adjugate_fin_three, sub_eq_add_neg, add_comm] using
        hminor 0 2 2 1 0 1 2 2
    · simpa [Matrix.adjugate_fin_three] using hminor 0 1 1 2 0 2 1 1
    · simpa [Matrix.adjugate_fin_three, sub_eq_add_neg, add_comm] using
        hminor 1 2 2 0 1 0 2 2
    · simpa [Matrix.adjugate_fin_three] using hminor 0 0 2 2 0 2 2 0
    · simpa [Matrix.adjugate_fin_three, sub_eq_add_neg, add_comm] using
        hminor 0 2 1 0 0 0 1 2
    · simpa [Matrix.adjugate_fin_three] using hminor 1 0 2 1 1 1 2 0
    · simpa [Matrix.adjugate_fin_three, sub_eq_add_neg, add_comm] using
        hminor 0 1 2 0 0 0 2 1
    · simpa [Matrix.adjugate_fin_three] using hminor 0 0 1 1 0 1 1 0
  have hunit : IsUnit (Matrix.det S) :=
    isUnit_iff_ne_zero.mpr (ne_of_gt hdetpos)
  rw [Matrix.nonsing_inv_apply S hunit]
  change |(↑hunit.unit⁻¹ : ℝ) * S.adjugate i j| ≤ M
  rw [abs_mul]
  have hdetinv : |((↑hunit.unit⁻¹ : ℝ))| ≤ 1 / data.gap ^ 3 := by
    rw [show (↑hunit.unit⁻¹ : ℝ) = (Matrix.det S)⁻¹ by
      rw [Units.val_inv_eq_inv_val, hunit.unit_spec]]
    rw [abs_inv, abs_of_pos hdetpos]
    simpa [one_div] using one_div_le_one_div_of_le hgap3 hdet
  calc
    |(↑hunit.unit⁻¹ : ℝ)| * |S.adjugate i j| ≤
        (1 / data.gap ^ 3) * (2 * L ^ 2) :=
      mul_le_mul hdetinv (hadj i j) (abs_nonneg _) (by positivity)
    _ ≤ M := by
      dsimp [M]
      rw [show (1 / data.gap ^ 3) * (2 * L ^ 2) =
          2 * L ^ 2 / data.gap ^ 3 by ring]
      linarith

theorem replicon_leftEigenvector (β h s : ℝ) :
    let ell : Fin 3 → ℝ := ![1, -2, 1]
    Matrix.vecMul ell (stabilityOperator β (rsQ β h) (rsR β h) s) =
      (1 - s * atParameter β h) • ell := by
  dsimp [stabilityOperator, cavityMatrix, atParameter, rsA]
  funext j
  fin_cases j <;>
    simp [Matrix.vecMul_eq_sum, Fin.sum_univ_three, Matrix.one_apply] <;>
    ring

end SpinGlass.AT
