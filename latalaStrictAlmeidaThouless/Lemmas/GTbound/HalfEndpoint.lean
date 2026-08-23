import Lemmas.GTbound.Endpoint
import Lemmas.GTbound.HalfStep

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace SpinGlass.AT

/-- Trial coordinates below `q` in the branch `q ≤ |v|`. -/
noncomputable def gtHalfOuterTrialCoefficient
    (N : ℕ) (β q s v : ℝ)
    (p : SpinGlass.Config N × SpinGlass.Config N) :
    EuclideanSpace ℝ (GTOrdinaryIndex N) :=
  let sign := gtPathSign v
  let e := β * Real.sqrt ((1 - s) * q)
  let a := gtIncrementScale β s 0 q
  WithLp.toLp 2 fun k =>
    match k with
    | Sum.inl _ => 0
    | Sum.inr (Sum.inl _) => 0
    | Sum.inr (Sum.inr (i, j)) =>
        if j = 0 then
          e * (SpinGlass.spin N p.1 i + SpinGlass.spin N p.2 i)
        else if j = 1 then
          a * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i)
        else 0

/-- The rank-one trial increment from `q` to `|v|`. -/
noncomputable def gtHalfInnerTrialCoefficient
    (N : ℕ) (β q s v : ℝ)
    (p : SpinGlass.Config N × SpinGlass.Config N) :
    EuclideanSpace ℝ (Fin N) :=
  let b := gtIncrementScale β s q |v|
  WithLp.toLp 2 fun i =>
    b * (SpinGlass.spin N p.1 i +
      gtPathSign v * SpinGlass.spin N p.2 i)

lemma gtHalfOuter_coefficients_orthogonal
    (N : ℕ) (β q s v : ℝ)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p)
      (gtHalfOuterTrialCoefficient N β q s v r) = 0 := by
  classical
  rw [PiLp.inner_apply, Fintype.sum_sum_type]
  simp [gtOrdinaryBarePhysicalCoefficient, gtHalfOuterTrialCoefficient,
    RCLike.inner_apply]

lemma gtHalfOuterTrialCoefficient_inner
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtHalfOuterTrialCoefficient N β q s v p)
      (gtHalfOuterTrialCoefficient N β q s v r) =
      (N : ℝ) * ∑ a : Fin 2, ∑ b : Fin 2,
        gtCovarianceMatrix β q s v q a b * pairOverlapMatrix p r a b := by
  classical
  let sign : ℝ := gtPathSign v
  let e : ℝ := β * Real.sqrt ((1 - s) * q)
  let a : ℝ := gtIncrementScale β s 0 q
  have he : e ^ 2 = β ^ 2 * (1 - s) * q := by
    dsimp [e]
    rw [mul_pow, Real.sq_sqrt (mul_nonneg (sub_nonneg.mpr hs.2) hq0)]
    ring
  have ha : a ^ 2 = β ^ 2 * s * q := by
    simpa using gtIncrementScale_sq (β := β) (s := s)
      (lower := 0) (upper := q) hs.1 hq0
  have hsignsq : sign ^ 2 = 1 := by
    simpa [sign] using gtPathSign_sq v
  rw [PiLp.inner_apply, Fintype.sum_sum_type]
  simp only [gtHalfOuterTrialCoefficient, RCLike.inner_apply, conj_trivial,
    mul_zero, Finset.sum_const_zero, zero_add]
  rw [Fintype.sum_sum_type]
  simp only [mul_zero, Finset.sum_const_zero, zero_add]
  rw [Fintype.sum_prod_type]
  simp only [Fin.sum_univ_four]
  simp only [show (0 : Fin 4) = 0 by rfl, if_true,
    show (1 : Fin 4) ≠ 0 by decide, if_false,
    show (1 : Fin 4) = 1 by rfl,
    show (2 : Fin 4) ≠ 0 by decide, show (2 : Fin 4) ≠ 1 by decide,
    show (3 : Fin 4) ≠ 0 by decide, show (3 : Fin 4) ≠ 1 by decide]
  simp only [zero_mul, add_zero]
  change (∑ i : Fin N, (
      e * (SpinGlass.spin N r.1 i + SpinGlass.spin N r.2 i) *
          (e * (SpinGlass.spin N p.1 i + SpinGlass.spin N p.2 i)) +
        a * (SpinGlass.spin N r.1 i + sign * SpinGlass.spin N r.2 i) *
          (a * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i)))) = _
  rw [show (∑ i : Fin N, (
      e * (SpinGlass.spin N r.1 i + SpinGlass.spin N r.2 i) *
          (e * (SpinGlass.spin N p.1 i + SpinGlass.spin N p.2 i)) +
        a * (SpinGlass.spin N r.1 i + sign * SpinGlass.spin N r.2 i) *
          (a * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i)))) =
      e ^ 2 * ((∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.1 i) +
        (∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.2 i) +
        (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.1 i) +
        (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.2 i)) +
      a ^ 2 * ((∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.1 i) +
        sign * (∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.2 i) +
        sign * (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.1 i) +
        sign ^ 2 * (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.2 i)) by
    simp_rw [mul_add, add_mul, Finset.sum_add_distrib, Finset.mul_sum]
    repeat rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring]
  rw [he, ha, hsignsq]
  simp_rw [spin_sum_eq_mul_overlap hN]
  simp only [pairOverlapMatrix, pairConfig, Fin.sum_univ_two]
  unfold gtCovarianceMatrix signedMatrixPath
  simp only [min_eq_left hqr, Matrix.add_apply, Matrix.smul_apply]
  simp [sign]
  ring

lemma gtHalfInnerTrialCoefficient_inner
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs0 : 0 ≤ s) (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtHalfInnerTrialCoefficient N β q s v p)
      (gtHalfInnerTrialCoefficient N β q s v r) =
      (N : ℝ) * (s * β ^ 2) * ∑ a : Fin 2, ∑ b : Fin 2,
        (signedMatrixPath v |v| a b - signedMatrixPath v q a b) *
          pairOverlapMatrix p r a b := by
  classical
  let sign : ℝ := gtPathSign v
  let b : ℝ := gtIncrementScale β s q |v|
  have hb : b ^ 2 = β ^ 2 * s * (|v| - q) := by
    simpa [b] using gtIncrementScale_sq (β := β) (s := s)
      (lower := q) (upper := |v|) hs0 hqr
  have hsignsq : sign ^ 2 = 1 := by
    simpa [sign] using gtPathSign_sq v
  rw [PiLp.inner_apply]
  simp only [gtHalfInnerTrialCoefficient, RCLike.inner_apply, conj_trivial]
  change (∑ i : Fin N,
      b * (SpinGlass.spin N r.1 i + sign * SpinGlass.spin N r.2 i) *
        (b * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i))) = _
  rw [show (∑ i : Fin N,
      b * (SpinGlass.spin N r.1 i + sign * SpinGlass.spin N r.2 i) *
        (b * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i))) =
      b ^ 2 * ((∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.1 i) +
        sign * (∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N r.2 i) +
        sign * (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.1 i) +
        sign ^ 2 * (∑ i, SpinGlass.spin N p.2 i * SpinGlass.spin N r.2 i)) by
    simp_rw [mul_add, add_mul, Finset.sum_add_distrib, Finset.mul_sum]
    repeat rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring]
  rw [hb, hsignsq]
  simp_rw [spin_sum_eq_mul_overlap hN]
  simp only [pairOverlapMatrix, pairConfig, Fin.sum_univ_two]
  unfold signedMatrixPath
  simp only [min_eq_left (abs_nonneg v), min_eq_left hqr,
    gtPathSign_mul_abs]
  simp [sign]
  ring

lemma gtHalfTotalTrialCovariance
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtHalfOuterTrialCoefficient N β q s v p)
        (gtHalfOuterTrialCoefficient N β q s v r) +
      inner ℝ (gtHalfInnerTrialCoefficient N β q s v p)
        (gtHalfInnerTrialCoefficient N β q s v r) =
      (N : ℝ) * ∑ a : Fin 2, ∑ b : Fin 2,
        gtCovarianceMatrix β q s v |v| a b * pairOverlapMatrix p r a b := by
  rw [gtHalfOuterTrialCoefficient_inner hN hs hq0 hqr,
    gtHalfInnerTrialCoefficient_inner hN hs.1 hqr]
  unfold gtCovarianceMatrix
  simp only [Matrix.add_apply, Matrix.smul_apply]
  simp_rw [Finset.mul_sum]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro b _
  ring

/-- Square completion at the lower breakpoint `q`. -/
lemma gtHalfOuter_covariance_square_completion
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p)
        (gtOrdinaryBarePhysicalCoefficient N β q s r) +
      (N : ℝ) * gtScalarVariance β s v q -
      inner ℝ (gtHalfOuterTrialCoefficient N β q s v p)
        (gtHalfOuterTrialCoefficient N β q s v r) =
      (N : ℝ) * (s * β ^ 2 / 2) *
        ∑ a : Fin 2, ∑ b : Fin 2,
          (pairOverlapMatrix p r a b - signedMatrixPath v q a b) ^ 2 := by
  rw [gtOrdinaryBarePhysicalCoefficient_inner hN hs hq0,
    gtHalfOuterTrialCoefficient_inner hN hs hq0 hqr,
    gtScalarVariance_eq_matrix_sum hq0]
  unfold gtCovarianceFunction gtCovarianceMatrix
  simp only [Matrix.add_apply, Matrix.smul_apply, Fin.sum_univ_two]
  norm_num
  ring

/-- Square completion at the signed breakpoint `|v|`. -/
lemma gtHalfTotal_covariance_square_completion
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p)
        (gtOrdinaryBarePhysicalCoefficient N β q s r) +
      (N : ℝ) * gtScalarVariance β s v |v| -
      (inner ℝ (gtHalfOuterTrialCoefficient N β q s v p)
          (gtHalfOuterTrialCoefficient N β q s v r) +
        inner ℝ (gtHalfInnerTrialCoefficient N β q s v p)
          (gtHalfInnerTrialCoefficient N β q s v r)) =
      (N : ℝ) * (s * β ^ 2 / 2) *
        ∑ a : Fin 2, ∑ b : Fin 2,
          (pairOverlapMatrix p r a b - signedMatrixPath v |v| a b) ^ 2 := by
  rw [gtOrdinaryBarePhysicalCoefficient_inner hN hs hq0,
    gtHalfTotalTrialCovariance hN hs hq0 hqr,
    gtScalarVariance_eq_matrix_sum (abs_nonneg v)]
  unfold gtCovarianceFunction gtCovarianceMatrix
  simp only [Matrix.add_apply, Matrix.smul_apply, Fin.sum_univ_two]
  norm_num
  ring

lemma gtHalfUpperGap_self
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|) (hv : v ∈ attainableOverlaps N)
    (p : ConstrainedPair N v) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p.1)
        (gtOrdinaryBarePhysicalCoefficient N β q s p.1) +
      (N : ℝ) * gtScalarVariance β s v |v| -
      (inner ℝ (gtHalfOuterTrialCoefficient N β q s v p.1)
          (gtHalfOuterTrialCoefficient N β q s v p.1) +
        inner ℝ (gtHalfInnerTrialCoefficient N β q s v p.1)
          (gtHalfInnerTrialCoefficient N β q s v p.1)) =
      (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2) := by
  rw [gtHalfTotal_covariance_square_completion hN hs hq0 hqr]
  rw [pairOverlapMatrix_self_eq_signedMatrixPath_one hN hv p]
  have hv1 : |v| ≤ 1 := abs_le.2 (gtAttainableOverlap_mem_Icc hN hv)
  have hmatrix :
      (∑ a : Fin 2, ∑ b : Fin 2,
        (signedMatrixPath v 1 a b - signedMatrixPath v |v| a b) ^ 2) =
        2 * (1 - |v|) ^ 2 := by
    unfold signedMatrixPath
    simp [Fin.sum_univ_two, min_eq_right hv1, min_eq_left hv1,
      gtPathSign_mul_abs]
    ring
  rw [hmatrix]
  ring

/-- Specialized pointwise derivative bound in the branch `q ≤ |v|`. -/
lemma gtConstrainedHalfDerivativeExpression_le
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q)
    (hqr : q ≤ |v|) (hv : v ∈ attainableOverlaps N)
    (t : ℝ) (z₀ : EuclideanSpace ℝ (GTOrdinaryIndex N)) :
    letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
    gtHalfDerivativeExpression
        (fun p : ConstrainedPair N v => gtPairPotential N h lam v p.1)
        (fun p : ConstrainedPair N v =>
          gtOrdinaryBarePhysicalCoefficient N β q s p.1)
        (fun p : ConstrainedPair N v =>
          gtHalfOuterTrialCoefficient N β q s v p.1)
        (fun p : ConstrainedPair N v =>
          gtHalfInnerTrialCoefficient N β q s v p.1)
        0 t z₀ ≤
      ((N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2)) / 2 +
        ((N : ℝ) * gtScalarVariance β s v q -
          (N : ℝ) * gtScalarVariance β s v |v|) / 4 := by
  letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
  let lowerGap : ConstrainedPair N v → ConstrainedPair N v → ℝ :=
    fun p r => (N : ℝ) * (s * β ^ 2 / 2) *
      ∑ a : Fin 2, ∑ b : Fin 2,
        (pairOverlapMatrix p.1 r.1 a b - signedMatrixPath v q a b) ^ 2
  let upperGap : ConstrainedPair N v → ConstrainedPair N v → ℝ :=
    fun p r => (N : ℝ) * (s * β ^ 2 / 2) *
      ∑ a : Fin 2, ∑ b : Fin 2,
        (pairOverlapMatrix p.1 r.1 a b - signedMatrixPath v |v| a b) ^ 2
  apply gtHalfDerivativeExpression_le
    (shiftQ := (N : ℝ) * gtScalarVariance β s v q)
    (shiftR := (N : ℝ) * gtScalarVariance β s v |v|)
    (diagGap := (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2))
    (lowerGap := lowerGap) (upperGap := upperGap)
  · intro p r
    exact gtHalfOuter_covariance_square_completion hN hs hq0 hqr p.1 r.1
  · intro p r
    dsimp [upperGap]
    have h := gtHalfTotal_covariance_square_completion (N := N) (β := β)
      hN hs hq0 hqr p.1 r.1
    convert h using 1 <;> ring
  · intro p
    dsimp [upperGap]
    rw [pairOverlapMatrix_self_eq_signedMatrixPath_one hN hv p]
    have hv1 : |v| ≤ 1 := abs_le.2 (gtAttainableOverlap_mem_Icc hN hv)
    unfold signedMatrixPath
    simp [Fin.sum_univ_two, min_eq_right hv1, min_eq_left hv1,
      gtPathSign_mul_abs]
    ring
  · intro p r
    dsimp [lowerGap]
    apply mul_nonneg
    · exact mul_nonneg (by positivity)
        (div_nonneg (mul_nonneg hs.1 (sq_nonneg β)) (by norm_num))
    · positivity
  · intro p r
    dsimp [upperGap]
    apply mul_nonneg
    · exact mul_nonneg (by positivity)
        (div_nonneg (mul_nonneg hs.1 (sq_nonneg β)) (by norm_num))
    · positivity

end SpinGlass.AT
