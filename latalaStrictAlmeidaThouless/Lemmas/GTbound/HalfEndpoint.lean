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

/-! ## An ordinary comparison field stopped at `|v|`

The last coordinate below is the increment from `q` to `|v|`.  It is kept
as an ordinary Gaussian during the comparison and converted to mass `1/2`
at the independent endpoint by Jensen's inequality. -/

noncomputable def gtStoppedTrialCoefficient
    (N : ℕ) (β q s v : ℝ)
    (p : SpinGlass.Config N × SpinGlass.Config N) :
    EuclideanSpace ℝ (GTOrdinaryIndex N) :=
  let sign := gtPathSign v
  let e := β * Real.sqrt ((1 - s) * q)
  let a := gtIncrementScale β s 0 q
  let b := gtIncrementScale β s q |v|
  WithLp.toLp 2 fun k =>
    match k with
    | Sum.inl _ => 0
    | Sum.inr (Sum.inl _) => 0
    | Sum.inr (Sum.inr (i, j)) =>
        if j = 0 then
          e * (SpinGlass.spin N p.1 i + SpinGlass.spin N p.2 i)
        else if j = 1 then
          a * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i)
        else if j = 2 then
          b * (SpinGlass.spin N p.1 i + sign * SpinGlass.spin N p.2 i)
        else 0

lemma gtStopped_coefficients_orthogonal
    (N : ℕ) (β q s v : ℝ)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p)
      (gtStoppedTrialCoefficient N β q s v r) = 0 := by
  classical
  rw [PiLp.inner_apply, Fintype.sum_sum_type]
  simp [gtOrdinaryBarePhysicalCoefficient, gtStoppedTrialCoefficient,
    RCLike.inner_apply]

lemma gtStoppedTrialCoefficient_inner
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q) (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtStoppedTrialCoefficient N β q s v p)
      (gtStoppedTrialCoefficient N β q s v r) =
      (N : ℝ) * ∑ a : Fin 2, ∑ b : Fin 2,
        gtCovarianceMatrix β q s v |v| a b * pairOverlapMatrix p r a b := by
  classical
  have hsplit :
      inner ℝ (gtStoppedTrialCoefficient N β q s v p)
          (gtStoppedTrialCoefficient N β q s v r) =
        inner ℝ (gtHalfOuterTrialCoefficient N β q s v p)
            (gtHalfOuterTrialCoefficient N β q s v r) +
          inner ℝ (gtHalfInnerTrialCoefficient N β q s v p)
            (gtHalfInnerTrialCoefficient N β q s v r) := by
    rw [PiLp.inner_apply, PiLp.inner_apply, PiLp.inner_apply]
    repeat rw [Fintype.sum_sum_type]
    simp only [gtStoppedTrialCoefficient, gtHalfOuterTrialCoefficient,
      gtHalfInnerTrialCoefficient, RCLike.inner_apply, conj_trivial,
      mul_zero, Finset.sum_const_zero, zero_add]
    repeat rw [Fintype.sum_sum_type]
    repeat rw [Fintype.sum_prod_type]
    simp only [Fin.sum_univ_four]
    simp only [show (0 : Fin 4) = 0 by rfl, if_true,
      show (1 : Fin 4) ≠ 0 by decide, if_false,
      show (1 : Fin 4) = 1 by rfl,
      show (2 : Fin 4) ≠ 0 by decide, show (2 : Fin 4) ≠ 1 by decide,
      show (2 : Fin 4) = 2 by rfl,
      show (3 : Fin 4) ≠ 0 by decide, show (3 : Fin 4) ≠ 1 by decide,
      show (3 : Fin 4) ≠ 2 by decide]
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hsplit, gtHalfTotalTrialCovariance hN hs hq0 hqr]

lemma gtStopped_covariance_square_completion
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q) (hqr : q ≤ |v|)
    (p r : SpinGlass.Config N × SpinGlass.Config N) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p)
        (gtOrdinaryBarePhysicalCoefficient N β q s r) +
      (N : ℝ) * gtScalarVariance β s v |v| -
      inner ℝ (gtStoppedTrialCoefficient N β q s v p)
        (gtStoppedTrialCoefficient N β q s v r) =
      (N : ℝ) * (s * β ^ 2 / 2) *
        ∑ a : Fin 2, ∑ b : Fin 2,
          (pairOverlapMatrix p r a b - signedMatrixPath v |v| a b) ^ 2 := by
  rw [gtOrdinaryBarePhysicalCoefficient_inner hN hs hq0,
    gtStoppedTrialCoefficient_inner hN hs hq0 hqr,
    gtScalarVariance_eq_matrix_sum (abs_nonneg v)]
  unfold gtCovarianceFunction gtCovarianceMatrix
  simp only [Matrix.add_apply, Matrix.smul_apply, Fin.sum_univ_two]
  norm_num
  ring

lemma gtStoppedUpperGap_self
    {N : ℕ} (hN : 0 < N) {β q s v : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q) (hqr : q ≤ |v|)
    (hv : v ∈ attainableOverlaps N) (p : ConstrainedPair N v) :
    inner ℝ (gtOrdinaryBarePhysicalCoefficient N β q s p.1)
        (gtOrdinaryBarePhysicalCoefficient N β q s p.1) +
      (N : ℝ) * gtScalarVariance β s v |v| -
      inner ℝ (gtStoppedTrialCoefficient N β q s v p.1)
        (gtStoppedTrialCoefficient N β q s v p.1) =
      (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2) := by
  rw [gtStopped_covariance_square_completion hN hs hq0 hqr]
  rw [pairOverlapMatrix_self_eq_signedMatrixPath_one hN hv p]
  have hv1 : |v| ≤ 1 := abs_le.2 (gtAttainableOverlap_mem_Icc hN hv)
  unfold signedMatrixPath
  simp [Fin.sum_univ_two, min_eq_right hv1, gtPathSign_mul_abs]
  ring

noncomputable def gtConstrainedStoppedPressure
    (N : ℕ) (β h q s v lam : ℝ) (hv : v ∈ attainableOverlaps N)
    (t : ℝ) : ℝ := by
  letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
  exact gtOrdinaryPressure
    (fun p : ConstrainedPair N v => gtPairPotential N h lam v p.1)
    (fun p : ConstrainedPair N v =>
      gtOrdinaryBarePhysicalCoefficient N β q s p.1)
    (fun p : ConstrainedPair N v => gtStoppedTrialCoefficient N β q s v p.1)
    0 t

noncomputable def gtUnconstrainedStoppedPressure
    (N : ℕ) (β h q s v lam t : ℝ) : ℝ :=
  gtOrdinaryPressure
    (fun p : SpinGlass.Config N × SpinGlass.Config N => gtPairPotential N h lam v p)
    (fun p => gtOrdinaryBarePhysicalCoefficient N β q s p)
    (fun p => gtStoppedTrialCoefficient N β q s v p) 0 t

lemma gtConstrainedStoppedPressure_one_le_zero
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 ≤ q) (hqr : q ≤ |v|)
    (hv : v ∈ attainableOverlaps N) :
    gtConstrainedStoppedPressure N β h q s v lam hv 1 ≤
      gtConstrainedStoppedPressure N β h q s v lam hv 0 +
        (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2) / 2 := by
  letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
  unfold gtConstrainedStoppedPressure
  apply gtOrdinaryPressure_one_le_zero_add_shiftedDiagonalGap
    (hAB := fun p r => gtStopped_coefficients_orthogonal N β q s v p.1 r.1)
    (shift := (N : ℝ) * gtScalarVariance β s v |v|)
    (gap := (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2))
  · intro p
    have hself := gtStoppedUpperGap_self (β := β) hN hs hq0 hqr hv p
    linarith
  · intro p r
    have hsq := gtStopped_covariance_square_completion (β := β)
      hN hs hq0 hqr p.1 r.1
    have hnonneg : 0 ≤ (N : ℝ) * (s * β ^ 2 / 2) *
        ∑ a : Fin 2, ∑ b : Fin 2,
          (pairOverlapMatrix p.1 r.1 a b - signedMatrixPath v |v| a b) ^ 2 := by
      exact mul_nonneg
        (mul_nonneg (Nat.cast_nonneg N)
          (div_nonneg (mul_nonneg hs.1 (sq_nonneg β)) (by norm_num)))
        (Finset.sum_nonneg fun a _ => Finset.sum_nonneg fun b _ => sq_nonneg _)
    linarith

lemma gtConstrainedStoppedPressure_one_eq_canonical
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ attainableOverlaps N) :
    gtConstrainedStoppedPressure N β h q s v lam hv 1 =
      ∫ x, coupledConstrainedLogPartition N β h q s v x
        ∂SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N) := by
  letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
  have heq : gtConstrainedStoppedPressure N β h q s v lam hv 1 =
      gtConstrainedOrdinaryPressure N β h q s v lam hv 1 := by
    unfold gtConstrainedStoppedPressure gtConstrainedOrdinaryPressure
    unfold gtOrdinaryPressure
    apply integral_congr_ae
    filter_upwards with z
    congr 1
    unfold gtOrdinaryField
    simp
  rw [heq, gtOrdinaryPressure_one_eq_canonical hN hs hq hv]

lemma gtConstrainedStoppedPressure_zero_le_unconstrained
    {N : ℕ} {β h q s v lam : ℝ} (hv : v ∈ attainableOverlaps N) :
    gtConstrainedStoppedPressure N β h q s v lam hv 0 ≤
      gtUnconstrainedStoppedPressure N β h q s v lam 0 := by
  classical
  letI : Nonempty (ConstrainedPair N v) := constrainedPair_nonempty hv
  unfold gtConstrainedStoppedPressure gtUnconstrainedStoppedPressure
  unfold gtOrdinaryPressure
  apply integral_mono
  · exact integrable_gtStateLogPartition_gtOrdinaryField _ _ _ 0 0
  · exact integrable_gtStateLogPartition_gtOrdinaryField _ _ _ 0 0
  · intro z
    unfold gtStateLogPartition
    apply Real.log_le_log
    · exact gtStatePartition_pos _ _
    · unfold gtStatePartition
      simp only [gtOrdinaryField, Real.sqrt_zero, zero_smul, sub_zero,
        Real.sqrt_one, one_smul, zero_add, add_zero, gtCoefficientCLM_apply]
      rw [← Finset.sum_subtype
        (p := fun p : SpinGlass.Config N × SpinGlass.Config N =>
          SpinGlass.overlap N p.1 p.2 = v)
        (Finset.univ.filter fun p : SpinGlass.Config N × SpinGlass.Config N =>
          SpinGlass.overlap N p.1 p.2 = v) (by simp)
        (fun p : SpinGlass.Config N × SpinGlass.Config N =>
          Real.exp (inner ℝ (gtStoppedTrialCoefficient N β q s v p)
            (WithLp.toLp 2 z) + gtPairPotential N h lam v p))]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
        (fun _ _ _ => Real.exp_nonneg _)

noncomputable def gtStoppedTrialFieldOne
    (N : ℕ) (β h q s v : ℝ) (z : GTOrdinarySiteIndex N → ℝ)
    (i : Fin N) : ℝ :=
  h + β * Real.sqrt ((1 - s) * q) * z (i, 0) +
    gtIncrementScale β s 0 q * z (i, 1) +
    gtIncrementScale β s q |v| * z (i, 2)

noncomputable def gtStoppedTrialFieldTwo
    (N : ℕ) (β h q s v : ℝ) (z : GTOrdinarySiteIndex N → ℝ)
    (i : Fin N) : ℝ :=
  h + β * Real.sqrt ((1 - s) * q) * z (i, 0) +
    gtPathSign v * gtIncrementScale β s 0 q * z (i, 1) +
    gtPathSign v * gtIncrementScale β s q |v| * z (i, 2)

lemma gtUnconstrainedStopped_zero_integrand_eq_terminal_sum
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (z : GTOrdinaryIndex N → ℝ) :
    gtStateLogPartition
        (fun p : SpinGlass.Config N × SpinGlass.Config N =>
          gtPairPotential N h lam v p)
        (gtOrdinaryField
          (fun p => gtOrdinaryBarePhysicalCoefficient N β q s p)
          (fun p => gtStoppedTrialCoefficient N β q s v p)
          0 0 (WithLp.toLp 2 z)) =
      2 * (N : ℝ) * Real.log 2 +
        ∑ i : Fin N,
          gtTerminal lam
            (gtStoppedTrialFieldOne N β h q s v
              (fun ij => z (Sum.inr (Sum.inr ij))) i)
            (gtStoppedTrialFieldTwo N β h q s v
              (fun ij => z (Sum.inr (Sum.inr ij))) i) -
        lam * (N : ℝ) * v := by
  classical
  have hpart :
      gtStatePartition
          (fun p : SpinGlass.Config N × SpinGlass.Config N =>
            gtPairPotential N h lam v p)
          (gtOrdinaryField
            (fun p => gtOrdinaryBarePhysicalCoefficient N β q s p)
            (fun p => gtStoppedTrialCoefficient N β q s v p)
            0 0 (WithLp.toLp 2 z)) =
        pairFieldPartition N lam v
          (gtStoppedTrialFieldOne N β h q s v
            (fun ij => z (Sum.inr (Sum.inr ij))))
          (gtStoppedTrialFieldTwo N β h q s v
            (fun ij => z (Sum.inr (Sum.inr ij)))) := by
    unfold gtStatePartition pairFieldPartition gtOrdinaryField
    simp only [Real.sqrt_zero, zero_smul, sub_zero, Real.sqrt_one, one_smul,
      zero_add, add_zero, gtCoefficientCLM_apply]
    apply Finset.sum_congr rfl
    intro p _
    apply congrArg Real.exp
    rw [PiLp.inner_apply]
    simp only [gtStoppedTrialCoefficient, RCLike.inner_apply, conj_trivial,
      PiLp.toLp_apply, gtPairPotential, gtStoppedTrialFieldOne,
      gtStoppedTrialFieldTwo]
    rw [Fintype.sum_sum_type]
    simp only [mul_zero, Finset.sum_const_zero, zero_add]
    rw [Fintype.sum_sum_type]
    simp only [mul_zero, Finset.sum_const_zero, zero_add]
    rw [Fintype.sum_prod_type]
    simp only [Fin.sum_univ_four]
    simp only [show (0 : Fin 4) = 0 by rfl, if_true,
      show (1 : Fin 4) ≠ 0 by decide, if_false,
      show (1 : Fin 4) = 1 by rfl,
      show (2 : Fin 4) ≠ 0 by decide, show (2 : Fin 4) ≠ 1 by decide,
      show (2 : Fin 4) = 2 by rfl,
      show (3 : Fin 4) ≠ 0 by decide, show (3 : Fin 4) ≠ 1 by decide,
      show (3 : Fin 4) ≠ 2 by decide]
    simp only [mul_add, mul_sub, Finset.sum_add_distrib, Finset.mul_sum]
    repeat rw [← Finset.sum_add_distrib]
    apply add_left_cancel (a := lam * (N : ℝ) * v)
    ring_nf
    repeat rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  unfold gtStateLogPartition
  rw [hpart, log_pairFieldPartition]

lemma gtUnconstrainedStoppedPressure_zero_eq_four_integrals
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ} :
    gtUnconstrainedStoppedPressure N β h q s v lam 0 =
      2 * (N : ℝ) * Real.log 2 - lam * (N : ℝ) * v +
        (N : ℝ) * ∫ z₀, ∫ z₁, ∫ z₂, ∫ _z₃,
          gtTerminal lam
            (h + β * Real.sqrt ((1 - s) * q) * z₀ +
              gtIncrementScale β s 0 q * z₁ +
              gtIncrementScale β s q |v| * z₂)
            (h + β * Real.sqrt ((1 - s) * q) * z₀ +
              gtPathSign v * gtIncrementScale β s 0 q * z₁ +
              gtPathSign v * gtIncrementScale β s q |v| * z₂)
          ∂gaussianReal 0 1 ∂gaussianReal 0 1
          ∂gaussianReal 0 1 ∂gaussianReal 0 1 := by
  classical
  let μ := Measure.pi (fun _ : GTOrdinaryIndex N => gaussianReal 0 1)
  let g : (Fin 4 → ℝ) → ℝ := fun z =>
    gtTerminal lam
      (h + β * Real.sqrt ((1 - s) * q) * z 0 +
        gtIncrementScale β s 0 q * z 1 + gtIncrementScale β s q |v| * z 2)
      (h + β * Real.sqrt ((1 - s) * q) * z 0 +
        gtPathSign v * gtIncrementScale β s 0 q * z 1 +
        gtPathSign v * gtIncrementScale β s q |v| * z 2)
  have hg : Integrable g (Measure.pi (fun _ : Fin 4 => gaussianReal 0 1)) := by
    let c₁ : EuclideanSpace ℝ (Fin 4) := WithLp.toLp 2
      ![β * Real.sqrt ((1 - s) * q), gtIncrementScale β s 0 q,
        gtIncrementScale β s q |v|, 0]
    let c₂ : EuclideanSpace ℝ (Fin 4) := WithLp.toLp 2
      ![β * Real.sqrt ((1 - s) * q),
        gtPathSign v * gtIncrementScale β s 0 q,
        gtPathSign v * gtIncrementScale β s q |v|, 0]
    have hi := integrable_gtTerminal_affine_gaussianProduct lam h h c₁ c₂
    apply hi.congr
    filter_upwards with z
    dsimp [g, c₁, c₂]
    simp only [PiLp.inner_apply, Fin.sum_univ_four, RCLike.inner_apply,
      conj_trivial, PiLp.toLp_apply]
    apply congrArg₂ (gtTerminal lam) <;> simp [Fin.sum_univ_four] <;> ring
  let f : Fin N → (GTOrdinaryIndex N → ℝ) → ℝ := fun i z =>
    gtTerminal lam
      (gtStoppedTrialFieldOne N β h q s v
        (fun ij => z (Sum.inr (Sum.inr ij))) i)
      (gtStoppedTrialFieldTwo N β h q s v
        (fun ij => z (Sum.inr (Sum.inr ij))) i)
  have hf (i : Fin N) : Integrable (f i) μ := by
    let e : Fin 4 → GTOrdinaryIndex N := fun a => Sum.inr (Sum.inr (i, a))
    have he : Function.Injective e := by
      intro a b hab
      exact congrArg Prod.snd (Sum.inr.inj (Sum.inr.inj hab))
    let φ : (GTOrdinaryIndex N → ℝ) → (Fin 4 → ℝ) := fun z a => z (e a)
    have hmp : MeasurePreserving φ μ
        (Measure.pi (fun _ : Fin 4 => gaussianReal 0 1)) := by
      refine ⟨by fun_prop, ?_⟩
      simpa [φ, μ] using gaussianProduct_restrict_map e he
    have hc : Integrable (fun z : GTOrdinaryIndex N → ℝ => g (φ z)) μ :=
      (hmp.integrable_comp hg.1).2 hg
    simpa [f, g, e, φ, gtStoppedTrialFieldOne,
      gtStoppedTrialFieldTwo] using hc
  unfold gtUnconstrainedStoppedPressure gtOrdinaryPressure
  simp_rw [gtUnconstrainedStopped_zero_integrand_eq_terminal_sum hN]
  have hpoint (z : GTOrdinaryIndex N → ℝ) :
      2 * (N : ℝ) * Real.log 2 + ∑ i, f i z - lam * (N : ℝ) * v =
        (2 * (N : ℝ) * Real.log 2 - lam * (N : ℝ) * v) + ∑ i, f i z := by ring
  rw [integral_congr_ae (ae_of_all _ hpoint)]
  rw [integral_add (integrable_const _) (integrable_finset_sum _ fun i _ => hf i)]
  simp only [integral_const, Measure.real, measure_univ, ENNReal.toReal_one, one_smul]
  rw [integral_finset_sum Finset.univ (fun i _ => hf i)]
  have hone (i : Fin N) : (∫ z, f i z ∂μ) =
      ∫ z : Fin 4 → ℝ, g z ∂Measure.pi (fun _ : Fin 4 => gaussianReal 0 1) := by
    let e : Fin 4 → GTOrdinaryIndex N := fun a => Sum.inr (Sum.inr (i, a))
    have he : Function.Injective e := by
      intro a b hab
      exact congrArg Prod.snd (Sum.inr.inj (Sum.inr.inj hab))
    have hr := integral_gaussianProduct_restrict e he g hg.1
    simpa [μ, f, e, g, gtStoppedTrialFieldOne, gtStoppedTrialFieldTwo] using hr
  simp_rw [hone]
  rw [integral_gaussianFinFour_eq_iterated g hg]
  dsimp [g]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp

lemma gaussian_terminal_expectation_le_half_step
    (lam x₁ x₂ scale sign : ℝ) :
    (∫ z, gtTerminal lam (x₁ + scale * z)
        (x₂ + sign * scale * z) ∂gaussianReal 0 1) ≤
      gtRankOneStep (1 / 2) scale sign (gtTerminal lam) x₁ x₂ := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => gtTerminal l x.1 x.2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  have hgood : GTFrame.GoodFam F D := by
    simpa [F, D] using (GTFrame.goodFam_fLbase (P := Unit))
  have hlin : Integrable (fun z => (1 / 2 : ℝ) *
      F () lam (x₁ + scale * z, x₂ + sign * scale * z))
      (gaussianReal 0 1) :=
    (hgood.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam scale (sign * scale) (x₁, x₂)).const_mul _
  have hexp : Integrable (fun z => Real.exp ((1 / 2 : ℝ) *
      F () lam (x₁ + scale * z, x₂ + sign * scale * z)))
      (gaussianReal 0 1) :=
    GTFrame.integrable_expShift (m := (1 / 2 : ℝ))
      (GTFrame.expMoments_gaussianReal 0 1) hgood (by norm_num)
      () lam scale (sign * scale) (x₁, x₂)
  have hj :
      Real.exp (∫ z, (1 / 2 : ℝ) *
          F () lam (x₁ + scale * z, x₂ + sign * scale * z)
          ∂gaussianReal 0 1) ≤
        ∫ z, Real.exp ((1 / 2 : ℝ) *
          F () lam (x₁ + scale * z, x₂ + sign * scale * z))
          ∂gaussianReal 0 1 := by
    exact convexOn_exp.map_integral_le continuousOn_exp isClosed_univ
      (by simp) hlin hexp
  have hpos : 0 < ∫ z, Real.exp ((1 / 2 : ℝ) *
      F () lam (x₁ + scale * z, x₂ + sign * scale * z))
      ∂gaussianReal 0 1 := lt_of_lt_of_le (Real.exp_pos _) hj
  have hlog := Real.log_le_log (Real.exp_pos _) hj
  rw [Real.log_exp] at hlog
  unfold gtRankOneStep standardGaussianExpectation
  simp only [show (1 / 2 : ℝ) ≠ 0 by norm_num, if_false]
  dsimp [F] at hlog ⊢
  rw [integral_const_mul] at hlog
  nlinarith

lemma gtStopped_four_integral_le_half_cascade
    (β h q s v lam : ℝ) :
    (∫ z₀, ∫ z₁, ∫ z₂, ∫ _z₃,
      gtTerminal lam
        (h + β * Real.sqrt ((1 - s) * q) * z₀ +
          gtIncrementScale β s 0 q * z₁ +
          gtIncrementScale β s q |v| * z₂)
        (h + β * Real.sqrt ((1 - s) * q) * z₀ +
          gtPathSign v * gtIncrementScale β s 0 q * z₁ +
          gtPathSign v * gtIncrementScale β s q |v| * z₂)
      ∂gaussianReal 0 1 ∂gaussianReal 0 1
      ∂gaussianReal 0 1 ∂gaussianReal 0 1) ≤
    standardGaussianExpectation (fun z₀ =>
      gtRankOneStep 0 (gtIncrementScale β s 0 q) (gtPathSign v)
        (gtRankOneStep (1 / 2) (gtIncrementScale β s q |v|)
          (gtPathSign v) (gtTerminal lam))
        (h + β * Real.sqrt ((1 - s) * q) * z₀)
        (h + β * Real.sqrt ((1 - s) * q) * z₀)) := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => gtTerminal l x.1 x.2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  let a : Unit → ℝ := fun _ => gtIncrementScale β s 0 q
  let b : Unit → ℝ := fun _ => gtIncrementScale β s q |v|
  let sg : Unit → ℝ := fun _ => gtPathSign v
  let e : ℝ := β * Real.sqrt ((1 - s) * q)
  have hF : GTFrame.GoodFam F D := by
    simpa [F, D] using (GTFrame.goodFam_fLbase (P := Unit))
  have h0b := GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1)
    hF (α := b) (β := fun u => sg u * b u) (hα := continuous_const)
      (hβ := by dsimp [sg, b]; fun_prop)
  have hMb := GTFrame.stepM_good (m := (1 / 2 : ℝ))
    (GTFrame.expMoments_gaussianReal 0 1) hF (by norm_num)
      (α := b) (β := fun u => sg u * b u) (hα := continuous_const)
      (hβ := by dsimp [sg, b]; fun_prop)
  have h0a0b := GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1)
    h0b (α := a) (β := fun u => sg u * a u) (hα := continuous_const)
      (hβ := by dsimp [sg, a]; fun_prop)
  have h0aMb := GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1)
    hMb (α := a) (β := fun u => sg u * a u) (hα := continuous_const)
      (hβ := by dsimp [sg, a]; fun_prop)
  have hleft : Integrable (fun z₀ =>
      GTFrame.step0 (gaussianReal 0 1) a (fun u => sg u * a u)
        (GTFrame.step0 (gaussianReal 0 1) b (fun u => sg u * b u) F)
        () lam (h + e * z₀, h + e * z₀)) (gaussianReal 0 1) :=
    h0a0b.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam e e (h, h)
  have hright : Integrable (fun z₀ =>
      GTFrame.step0 (gaussianReal 0 1) a (fun u => sg u * a u)
        (GTFrame.stepM (gaussianReal 0 1) (1 / 2) b (fun u => sg u * b u) F)
        () lam (h + e * z₀, h + e * z₀)) (gaussianReal 0 1) :=
    h0aMb.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam e e (h, h)
  simp only [integral_const, Measure.real, measure_univ, ENNReal.toReal_one,
    one_smul]
  unfold standardGaussianExpectation gtRankOneStep
  simp only [if_pos rfl, show (1 / 2 : ℝ) ≠ 0 by norm_num, if_false]
  change (∫ z₀, GTFrame.step0 (gaussianReal 0 1) a (fun u => sg u * a u)
      (GTFrame.step0 (gaussianReal 0 1) b (fun u => sg u * b u) F)
      () lam (h + e * z₀, h + e * z₀) ∂gaussianReal 0 1) ≤
    ∫ z₀, GTFrame.step0 (gaussianReal 0 1) a (fun u => sg u * a u)
      (GTFrame.stepM (gaussianReal 0 1) (1 / 2) b (fun u => sg u * b u) F)
      () lam (h + e * z₀, h + e * z₀) ∂gaussianReal 0 1
  apply integral_mono hleft hright
  intro z₀
  unfold GTFrame.step0
  apply integral_mono
  · exact h0b.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam (a ()) (sg () * a ()) (h + e * z₀, h + e * z₀)
  · exact hMb.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam (a ()) (sg () * a ()) (h + e * z₀, h + e * z₀)
  · intro z₁
    have hj := gaussian_terminal_expectation_le_half_step lam
      (h + e * z₀ + a () * z₁)
      (h + e * z₀ + sg () * a () * z₁) (b ()) (sg ())
    unfold gtRankOneStep standardGaussianExpectation at hj
    simp only [show (1 / 2 : ℝ) ≠ 0 by norm_num, if_false] at hj
    dsimp [GTFrame.stepM, F, a, b, sg] at hj ⊢
    norm_num at hj ⊢
    convert hj using 1 <;> ring

lemma gtRankOneStep_half_terminal_add_const
    (scale sign lam k x₁ x₂ : ℝ) :
    gtRankOneStep (1 / 2) scale sign
        (fun y₁ y₂ => gtTerminal lam y₁ y₂ + k) x₁ x₂ =
      gtRankOneStep (1 / 2) scale sign (gtTerminal lam) x₁ x₂ + k := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => gtTerminal l x.1 x.2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  have hgood : GTFrame.GoodFam F D := by
    simpa [F, D] using (GTFrame.goodFam_fLbase (P := Unit))
  have hpos : 0 < ∫ z, Real.exp ((1 / 2 : ℝ) *
      gtTerminal lam (x₁ + scale * z) (x₂ + sign * scale * z))
      ∂gaussianReal 0 1 := by
    simpa [F] using GTFrame.integral_expShift_pos
      (m := (1 / 2 : ℝ)) (GTFrame.expMoments_gaussianReal 0 1)
      hgood (by norm_num) () lam scale (sign * scale) (x₁, x₂)
  unfold gtRankOneStep standardGaussianExpectation
  simp only [show (1 / 2 : ℝ) ≠ 0 by norm_num, if_false]
  have heq :
      (∫ z, Real.exp ((1 / 2 : ℝ) *
          (gtTerminal lam (x₁ + scale * z) (x₂ + sign * scale * z) + k))
          ∂gaussianReal 0 1) =
        Real.exp ((1 / 2 : ℝ) * k) *
          ∫ z, Real.exp ((1 / 2 : ℝ) *
            gtTerminal lam (x₁ + scale * z) (x₂ + sign * scale * z))
            ∂gaussianReal 0 1 := by
    simp_rw [mul_add, Real.exp_add]
    rw [integral_mul_const]
    ring
  rw [heq, Real.log_mul (Real.exp_ne_zero _) hpos.ne', Real.log_exp]
  ring

lemma gtRankOneStep_zero_half_add_const
    (outer inner sign lam k x₁ x₂ : ℝ) :
    gtRankOneStep 0 outer sign
        (fun y₁ y₂ =>
          gtRankOneStep (1 / 2) inner sign (gtTerminal lam) y₁ y₂ + k)
        x₁ x₂ =
      gtRankOneStep 0 outer sign
        (gtRankOneStep (1 / 2) inner sign (gtTerminal lam)) x₁ x₂ + k := by
  let G : GTTwoField := gtRankOneStep (1 / 2) inner sign (gtTerminal lam)
  change gtRankOneStep 0 outer sign (fun y₁ y₂ => G y₁ y₂ + k) x₁ x₂ =
    gtRankOneStep 0 outer sign G x₁ x₂ + k
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => gtTerminal l x.1 x.2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  let b : Unit → ℝ := fun _ => inner
  let sg : Unit → ℝ := fun _ => sign
  have hF : GTFrame.GoodFam F D := by
    simpa [F, D] using (GTFrame.goodFam_fLbase (P := Unit))
  have hMb := GTFrame.stepM_good (m := (1 / 2 : ℝ))
    (GTFrame.expMoments_gaussianReal 0 1) hF (by norm_num)
      (α := b) (β := fun u => sg u * b u) (hα := continuous_const)
      (hβ := by dsimp [sg, b]; fun_prop)
  have hi : Integrable (fun z =>
      G (x₁ + outer * z) (x₂ + sign * outer * z))
      (gaussianReal 0 1) := by
    have h := hMb.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam outer (sign * outer) (x₁, x₂)
    simpa [G, GTFrame.stepM, F, b, sg, gtRankOneStep,
      standardGaussianExpectation] using h
  unfold gtRankOneStep
  simp only [if_pos rfl, standardGaussianExpectation]
  rw [integral_add hi (integrable_const k)]
  simp

/- The ordinary stopped comparison alone has too large a diagonal remainder;
the final theorem below instead uses the half-mass interpolation.  The
calculation is retained temporarily while the endpoint identity is assembled.
lemma gtUnconstrainedStoppedPressure_zero_add_gap_le_gtFunctional
    {N : ℕ} (hN : 0 < N) {β h q s v lam : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq0 : 0 < q) (hq1 : q ≤ 1)
    (hqr : q ≤ |v|) (hr1 : |v| ≤ 1) :
    gtUnconstrainedStoppedPressure N β h q s v lam 0 +
        (N : ℝ) * (s * β ^ 2 * (1 - |v|) ^ 2) / 2 ≤
      (N : ℝ) * gtFunctional β h q s lam v := by
  let sign : ℝ := gtPathSign v
  let e : ℝ := β * Real.sqrt ((1 - s) * q)
  let a : ℝ := gtIncrementScale β s 0 q
  let b : ℝ := gtIncrementScale β s q |v|
  let c : ℝ := gtIncrementScale β s |v| 1
  let innerF : GTTwoField :=
    gtRankOneStep (1 / 2) b sign (gtTerminal lam)
  let cascade : GTTwoField := gtRankOneStep 0 a sign innerF
  let C : ℝ := standardGaussianExpectation fun z => cascade (h + e * z) (h + e * z)
  have hj := gtStopped_four_integral_le_half_cascade β h q s v lam
  change _ ≤ C at hj
  have hc : c ^ 2 = β ^ 2 * s * (1 - |v|) := by
    dsimp [c]
    simpa using gtIncrementScale_sq (β := β) (s := s)
      hs.1 hr1
  have hupper_fun :
      gtDiagonalStep 1 c (gtTerminal lam) =
        fun x₁ x₂ => gtTerminal lam x₁ x₂ + c ^ 2 := by
    funext x₁ x₂
    exact gtDiagonalStep_one_terminal c lam x₁ x₂
  have hqnot : ¬ q ≤ (0 : ℝ) := not_le.mpr hq0
  have hrpos : 0 < |v| := lt_of_lt_of_le hq0 hqr
  have hrnot : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos
  have hfunctional :
      gtFunctional β h q s lam v =
        2 * Real.log 2 +
          standardGaussianExpectation (fun z =>
            gtRankOneStep 0 a sign
              (gtRankOneStep (1 / 2) b sign
                (gtDiagonalStep 1 c (gtTerminal lam)))
              (h + e * z) (h + e * z)) -
          lam * v - gtCorrection β q s := by
    unfold gtFunctional gtSemigroupSolution
    simp [hqr, hqnot, hrnot, a, b, c, e, sign]
  have hcascade_add (x₁ x₂ : ℝ) :
      gtRankOneStep 0 a sign
          (gtRankOneStep (1 / 2) b sign
            (gtDiagonalStep 1 c (gtTerminal lam))) x₁ x₂ =
        cascade x₁ x₂ + c ^ 2 := by
    rw [hupper_fun]
    have hinner_fun :
        gtRankOneStep (1 / 2) b sign
            (fun y₁ y₂ => gtTerminal lam y₁ y₂ + c ^ 2) =
          fun y₁ y₂ => innerF y₁ y₂ + c ^ 2 := by
      funext y₁ y₂
      exact gtRankOneStep_half_terminal_add_const b sign lam (c ^ 2) y₁ y₂
    rw [hinner_fun]
    exact gtRankOneStep_zero_half_add_const a b sign lam (c ^ 2) x₁ x₂
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => gtTerminal l x.1 x.2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  let aa : Unit → ℝ := fun _ => a
  let bb : Unit → ℝ := fun _ => b
  let sg : Unit → ℝ := fun _ => sign
  have hF : GTFrame.GoodFam F D := by
    simpa [F, D] using (GTFrame.goodFam_fLbase (P := Unit))
  have hMb := GTFrame.stepM_good (m := (1 / 2 : ℝ))
    (GTFrame.expMoments_gaussianReal 0 1) hF (by norm_num)
      (α := bb) (β := fun u => sg u * bb u) (hα := continuous_const)
      (hβ := by dsimp [sg, bb]; fun_prop)
  have h0aMb := GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1)
    hMb (α := aa) (β := fun u => sg u * aa u) (hα := continuous_const)
      (hβ := by dsimp [sg, aa]; fun_prop)
  have hCint : Integrable (fun z => cascade (h + e * z) (h + e * z))
      (gaussianReal 0 1) := by
    have hi := h0aMb.integrable_shift (GTFrame.expMoments_gaussianReal 0 1)
      () lam e e (h, h)
    simpa [cascade, innerF, GTFrame.step0, GTFrame.stepM, F, aa, bb, sg,
      gtRankOneStep, standardGaussianExpectation] using hi
  have htrial :
      standardGaussianExpectation (fun z =>
          gtRankOneStep 0 a sign
            (gtRankOneStep (1 / 2) b sign
              (gtDiagonalStep 1 c (gtTerminal lam)))
            (h + e * z) (h + e * z)) = C + c ^ 2 := by
    simp_rw [hcascade_add]
    unfold standardGaussianExpectation C
    rw [integral_add hCint (integrable_const (c ^ 2))]
    simp [standardGaussianExpectation]
  rw [gtUnconstrainedStoppedPressure_zero_eq_four_integrals hN]
  rw [hfunctional, htrial]
  unfold gtCorrection
  have hdiff : 0 ≤ (|v| - q) * (|v| + q) :=
    mul_nonneg (sub_nonneg.mpr hqr) (add_nonneg (abs_nonneg v) hq0.le)
  have hN0 : 0 ≤ (N : ℝ) := Nat.cast_nonneg N
  have hsβ : 0 ≤ s * β ^ 2 := mul_nonneg hs.1 (sq_nonneg β)
  have hjN := mul_le_mul_of_nonneg_left hj hN0
  rw [hc]
  ring_nf
-/
end SpinGlass.AT
