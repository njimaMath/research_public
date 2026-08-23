import Lemmas.GTbound.HalfCalculus
import Lemmas.GTbound.HalfEndpoint

open MeasureTheory ProbabilityTheory Real BigOperators Filter Topology
open scoped ContDiff

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace SpinGlass.AT

noncomputable def gtHalfBaseTransform
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) : ℝ :=
  2 * Real.log (gtHalfBaseDenominator V L H)

lemma hasFDerivAt_gtHalfBaseTransform
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    HasFDerivAt (gtHalfBaseTransform V L)
      (fderiv ℝ (gtHalfBaseTransform V L) H) H := by
  have hd := hasFDerivAt_gtHalfBaseDenominator V L H
  have hl := hd.log (gtHalfBaseDenominator_pos V L H).ne'
  have hc := hl.const_mul 2
  change HasFDerivAt (fun H => 2 * Real.log (gtHalfBaseDenominator V L H))
    (fderiv ℝ (fun H => 2 * Real.log (gtHalfBaseDenominator V L H)) H) H
  exact hc.differentiableAt.hasFDerivAt

lemma fderiv_gtHalfBaseTransform_apply
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H K : GTStateSpace S) :
    fderiv ℝ (gtHalfBaseTransform V L) H K =
      ∑ ξ : S, gtHalfBaseWeight V L H ξ * K ξ := by
  let D := gtHalfBaseDenominator V L H
  have hD : D ≠ 0 := (gtHalfBaseDenominator_pos V L H).ne'
  have hd := hasFDerivAt_gtHalfBaseDenominator V L H
  have heq := congrArg (fun T : GTStateSpace S →L[ℝ] ℝ => T K) hd.fderiv
  rw [gtHalfBaseDenominator_fderiv_apply] at heq
  have ht := hasFDerivAt_gtHalfBaseTransform V L H
  have hchain := (hd.log hD).const_mul 2
  have hu := ht.unique hchain
  rw [hu]
  simp only [ContinuousLinearMap.smul_apply, smul_eq_mul]
  rw [← hd.fderiv]
  rw [heq]
  unfold gtHalfBaseWeight
  field_simp [hD]
  rw [Finset.sum_div]

lemma norm_fderiv_gtHalfBaseTransform_le_one
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    ‖fderiv ℝ (gtHalfBaseTransform V L) H‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro K
  rw [Real.norm_eq_abs, fderiv_gtHalfBaseTransform_apply]
  calc
    |∑ ξ : S, gtHalfBaseWeight V L H ξ * K ξ| ≤
        ∑ ξ : S, |gtHalfBaseWeight V L H ξ * K ξ| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ ξ : S, gtHalfBaseWeight V L H ξ * |K ξ| := by
      apply Finset.sum_congr rfl
      intro ξ _
      rw [abs_mul, abs_of_nonneg (gtHalfBaseWeight_nonneg V L H ξ)]
    _ ≤ ∑ ξ : S, gtHalfBaseWeight V L H ξ * ‖K‖ := by
      apply Finset.sum_le_sum
      intro ξ _
      exact mul_le_mul_of_nonneg_left
        (by simpa [Real.norm_eq_abs] using
          PiLp.norm_apply_le (p := (2 : ENNReal)) K ξ)
        (gtHalfBaseWeight_nonneg V L H ξ)
    _ = ‖K‖ := by rw [← Finset.sum_mul, sum_gtHalfBaseWeight, one_mul]
    _ = 1 * ‖K‖ := by ring

lemma lipschitzWith_gtHalfBaseTransform
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S) :
    LipschitzWith 1 (gtHalfBaseTransform V L) := by
  apply lipschitzWith_of_nnnorm_fderiv_le (𝕜 := ℝ)
  · intro H
    exact (hasFDerivAt_gtHalfBaseTransform V L H).differentiableAt
  · intro H
    rw [← NNReal.coe_le_coe]
    simpa using norm_fderiv_gtHalfBaseTransform_le_one V L H

lemma gtHalfTransform_eq_base
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    (t : ℝ) (z₀ : EuclideanSpace ℝ I₀) :
    gtHalfTransform V A B₀ B₁ H₀ t z₀ =
      gtHalfBaseTransform V (gtHalfInnerCLM B₁ t)
        (gtHalfOuterCLM A B₀ t z₀ + H₀) := by
  unfold gtHalfTransform gtHalfBaseTransform
  rw [gtHalfDenominator_eq_base]

lemma integrable_gtHalfTransform
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S) (t : ℝ) :
    Integrable (fun z₀ : I₀ → ℝ =>
      gtHalfTransform V A B₀ B₁ H₀ t (WithLp.toLp 2 z₀))
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) := by
  let L := gtHalfInnerCLM B₁ t
  let M := gtHalfOuterCLM A B₀ t
  let C : ℝ := |gtHalfBaseTransform V L 0| + ‖H₀‖
  let bound : (I₀ → ℝ) → ℝ := fun z₀ =>
    C + ‖M‖ * ‖(WithLp.toLp 2 z₀ : EuclideanSpace ℝ I₀)‖
  have hbound : Integrable bound
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) := by
    exact (integrable_const C).add
      ((integrable_norm_gaussianProduct (I := I₀)).const_mul ‖M‖)
  apply hbound.mono'
  · have hc : Continuous (fun z₀ : I₀ → ℝ =>
        gtHalfBaseTransform V L (M (WithLp.toLp 2 z₀) + H₀)) :=
      (lipschitzWith_gtHalfBaseTransform V L).continuous.comp (by fun_prop)
    apply hc.aestronglyMeasurable.congr
    filter_upwards with z₀
    simpa [L, M] using
      (gtHalfTransform_eq_base V A B₀ B₁ H₀ t (WithLp.toLp 2 z₀)).symm
  · filter_upwards with z₀
    let zE : EuclideanSpace ℝ I₀ := WithLp.toLp 2 z₀
    rw [Real.norm_eq_abs]
    rw [gtHalfTransform_eq_base]
    change |gtHalfBaseTransform V L (M zE + H₀)| ≤ bound z₀
    have hlip := (lipschitzWith_gtHalfBaseTransform V L).norm_sub_le
      (M zE + H₀) 0
    have hM := M.le_opNorm zE
    have htri := norm_add_le (M zE) H₀
    dsimp [bound, C]
    have habs := abs_sub_abs_le_abs_sub
      (gtHalfBaseTransform V L (M zE + H₀)) (gtHalfBaseTransform V L 0)
    rw [show ‖gtHalfBaseTransform V L (M zE + H₀) -
        gtHalfBaseTransform V L 0‖ =
      |gtHalfBaseTransform V L (M zE + H₀) -
        gtHalfBaseTransform V L 0| by rfl] at hlip
    norm_num at hlip
    nlinarith

/-- Derivative of the two-level field on the open interpolation interval. -/
noncomputable def gtHalfFieldDeriv
    {S I₀ I₁ : Type*} [Fintype S] [Fintype I₀] [Fintype I₁]
    (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (t : ℝ)
    (z₀ : EuclideanSpace ℝ I₀) (z₁ : EuclideanSpace ℝ I₁) :
    GTStateSpace S :=
  (1 / (2 * Real.sqrt t)) • gtCoefficientCLM A z₀ -
    (1 / (2 * Real.sqrt (1 - t))) • gtCoefficientCLM B₀ z₀ -
    (1 / (2 * Real.sqrt (1 - t))) • gtCoefficientCLM B₁ z₁

lemma hasDerivAt_gtHalfField
    {S I₀ I₁ : Type*} [Fintype S] [Fintype I₀] [Fintype I₁]
    (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) (z₁ : EuclideanSpace ℝ I₁) :
    HasDerivAt (fun u => gtHalfField A B₀ B₁ H₀ u z₀ z₁)
      (gtHalfFieldDeriv A B₀ B₁ t z₀ z₁) t := by
  have hA := (Real.hasDerivAt_sqrt ht.1.ne').smul_const
    (gtCoefficientCLM A z₀)
  have hsub : HasDerivAt (fun u : ℝ => (1 : ℝ) - u) (-1) t := by
    simpa using HasDerivAt.const_sub (c := (1 : ℝ)) (hasDerivAt_id t)
  have hB := ((Real.hasDerivAt_sqrt (ne_of_gt (sub_pos.mpr ht.2))).comp t hsub
    ).smul_const (gtCoefficientCLM B₀ z₀ + gtCoefficientCLM B₁ z₁)
  have hsum := (hA.add hB).add_const H₀
  have hd :
      (1 / (2 * Real.sqrt t)) • gtCoefficientCLM A z₀ +
          (1 / (2 * Real.sqrt (1 - t)) * (-1)) •
            (gtCoefficientCLM B₀ z₀ + gtCoefficientCLM B₁ z₁) =
        gtHalfFieldDeriv A B₀ B₁ t z₀ z₁ := by
    ext ξ
    simp [gtHalfFieldDeriv]
    ring
  have hout := hsum.congr_deriv hd
  simpa [gtHalfField] using hout

lemma hasDerivAt_gtHalfDensity
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) (z₁ : EuclideanSpace ℝ I₁) :
    HasDerivAt
      (fun u => gtHalfDensity V A B₀ B₁ H₀ u z₀ z₁)
      (gtHalfDensity V A B₀ B₁ H₀ t z₀ z₁ * (1 / 2 : ℝ) *
        ∑ ξ : S,
          gtStateGibbs V (gtHalfField A B₀ B₁ H₀ t z₀ z₁) ξ *
            gtHalfFieldDeriv A B₀ B₁ t z₀ z₁ ξ) t := by
  have hlog : HasFDerivAt (gtStateLogPartition V)
      (fderiv ℝ (gtStateLogPartition V)
        (gtHalfField A B₀ B₁ H₀ t z₀ z₁))
      (gtHalfField A B₀ B₁ H₀ t z₀ z₁) :=
    ((contDiff_gtStateLogPartition V).differentiable (by simp)
      ).differentiableAt.hasFDerivAt
  have hc := hlog.comp_hasDerivAt t
    (hasDerivAt_gtHalfField A B₀ B₁ H₀ ht z₀ z₁)
  have he := (hc.const_mul (1 / 2 : ℝ)).exp
  rw [fderiv_gtStateLogPartition_apply] at he
  convert he using 1 <;> simp [gtHalfDensity] <;> ring

/-- Differentiate the inner half-mass normalizing integral. -/
lemma hasDerivAt_gtHalfDenominator
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) :
    HasDerivAt (fun u => gtHalfDenominator V A B₀ B₁ H₀ u z₀)
      (∫ z₁ : I₁ → ℝ,
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
          ∑ ξ : S,
            gtStateGibbs V
                (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ
        ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) t := by
  classical
  let ε : ℝ := min t (1 - t) / 2
  have hε : 0 < ε := by
    dsimp [ε]
    have hm : 0 < min t (1 - t) := lt_min ht.1 (sub_pos.mpr ht.2)
    linarith
  have hball : ∀ x ∈ Metric.ball t ε, x ∈ Set.Ioo (0 : ℝ) 1 := by
    intro x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hεt : ε ≤ t / 2 := by
      dsimp [ε]
      gcongr
      exact min_le_left _ _
    have hε1t : ε ≤ (1 - t) / 2 := by
      dsimp [ε]
      gcongr
      exact min_le_right _ _
    constructor
    · have := (abs_sub_lt_iff.1 hxt).2
      nlinarith
    · have := (abs_sub_lt_iff.1 hxt).1
      nlinarith
  let cA : ℝ := 1 / (2 * Real.sqrt (t / 2))
  let cB : ℝ := 1 / (2 * Real.sqrt ((1 - t) / 2))
  have hcA : 0 ≤ cA := by dsimp [cA]; positivity
  have hcB : 0 ≤ cB := by dsimp [cB]; positivity
  have hcoeffA : ∀ x ∈ Metric.ball t ε,
      |1 / (2 * Real.sqrt x)| ≤ cA := by
    intro x hx
    have hxI := hball x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hεt : ε ≤ t / 2 := by
      dsimp [ε]
      gcongr
      exact min_le_left _ _
    have htx : t / 2 ≤ x := by
      have := (abs_sub_lt_iff.1 hxt).2
      nlinarith
    have ht2 : 0 < t / 2 := by linarith [ht.1]
    have hp : 0 < 2 * Real.sqrt (t / 2) := by
      exact mul_pos (by norm_num) (Real.sqrt_pos.2 ht2)
    have hle : 2 * Real.sqrt (t / 2) ≤ 2 * Real.sqrt x := by
      nlinarith [Real.sqrt_le_sqrt htx]
    have hi : 1 / (2 * Real.sqrt x) ≤ cA := by
      dsimp [cA]
      simpa [one_div] using one_div_le_one_div_of_le hp hle
    rw [abs_of_nonneg (by positivity : 0 ≤ 1 / (2 * Real.sqrt x))]
    exact hi
  have hcoeffB : ∀ x ∈ Metric.ball t ε,
      |1 / (2 * Real.sqrt (1 - x))| ≤ cB := by
    intro x hx
    have hxI := hball x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hε1t : ε ≤ (1 - t) / 2 := by
      dsimp [ε]
      gcongr
      exact min_le_right _ _
    have htx : (1 - t) / 2 ≤ 1 - x := by
      have := (abs_sub_lt_iff.1 hxt).1
      nlinarith
    have ht2 : 0 < (1 - t) / 2 := by linarith [ht.2]
    have hp : 0 < 2 * Real.sqrt ((1 - t) / 2) := by
      exact mul_pos (by norm_num) (Real.sqrt_pos.2 ht2)
    have hle : 2 * Real.sqrt ((1 - t) / 2) ≤
        2 * Real.sqrt (1 - x) := by
      nlinarith [Real.sqrt_le_sqrt htx]
    have hi : 1 / (2 * Real.sqrt (1 - x)) ≤ cB := by
      dsimp [cB]
      simpa [one_div] using one_div_le_one_div_of_le hp hle
    rw [abs_of_nonneg (by positivity : 0 ≤ 1 / (2 * Real.sqrt (1 - x)))]
    exact hi
  let LA := gtCoefficientCLM A
  let LB₀ := gtCoefficientCLM B₀
  let LB₁ := gtCoefficientCLM B₁
  let R₀ : ℝ := ‖LA z₀‖ + ‖LB₀ z₀‖ + ‖H₀‖
  let R₁ : ℝ := ‖LB₁‖
  let D₀ : ℝ := cA * ‖LA z₀‖ + cB * ‖LB₀ z₀‖
  let D₁ : ℝ := cB * ‖LB₁‖
  let C : ℝ := (hasModerateGrowth_gtStateLogPartition V).C / 2
  let E₀ : ℝ := Real.exp (C * (1 + R₀))
  let bound : (I₁ → ℝ) → ℝ := fun z₁ =>
    (1 / 2 : ℝ) * (D₀ + D₁ * ‖(WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁)‖) *
      (E₀ * Real.exp ((C * R₁) *
        ‖(WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁)‖))
  have hC : 0 ≤ C := by
    dsimp [C]
    exact div_nonneg (hasModerateGrowth_gtStateLogPartition V).Cpos.le (by norm_num)
  have hR₀ : 0 ≤ R₀ := by dsimp [R₀]; positivity
  have hR₁ : 0 ≤ R₁ := by dsimp [R₁]; positivity
  have hD₀ : 0 ≤ D₀ := by dsimp [D₀]; positivity
  have hD₁ : 0 ≤ D₁ := by dsimp [D₁]; positivity
  have hE₀ : 0 ≤ E₀ := by dsimp [E₀]; positivity
  have hbound_int : Integrable bound
      (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
    let c := C * R₁
    have he := integrable_exp_mul_norm_gaussianProduct (I := I₁) c
    have hne := integrable_norm_mul_exp_mul_norm_gaussianProduct (I := I₁) c
    have hs := (he.const_mul D₀).add (hne.const_mul D₁)
    have hs' := (hs.const_mul E₀).const_mul (1 / 2 : ℝ)
    convert hs' using 1
    funext z₁
    dsimp [bound, c]
    ring
  let F : ℝ → (I₁ → ℝ) → ℝ := fun x z₁ =>
    gtHalfDensity V A B₀ B₁ H₀ x z₀ (WithLp.toLp 2 z₁)
  let F' : ℝ → (I₁ → ℝ) → ℝ := fun x z₁ =>
    gtHalfDensity V A B₀ B₁ H₀ x z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
      ∑ ξ : S,
        gtStateGibbs V
            (gtHalfField A B₀ B₁ H₀ x z₀ (WithLp.toLp 2 z₁)) ξ *
          gtHalfFieldDeriv A B₀ B₁ x z₀ (WithLp.toLp 2 z₁) ξ
  have hF_meas : ∀ᶠ x in nhds t,
      AEStronglyMeasurable (F x)
        (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
    refine Filter.Eventually.of_forall (fun x => ?_)
    exact (integrable_gtHalfDenominator_integrand V A B₀ B₁ H₀ x z₀).1
  have hF_int : Integrable (F t)
      (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) :=
    integrable_gtHalfDenominator_integrand V A B₀ B₁ H₀ t z₀
  have hF'_meas : AEStronglyMeasurable (F' t)
      (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
    have hfield : Continuous (fun z₁ : I₁ → ℝ =>
        gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) := by
      unfold gtHalfField
      fun_prop
    have hderiv : Continuous (fun z₁ : I₁ → ℝ =>
        gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁)) := by
      unfold gtHalfFieldDeriv
      fun_prop
    have hdensity : Continuous (fun z₁ : I₁ → ℝ =>
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) := by
      unfold gtHalfDensity
      exact Real.continuous_exp.comp
        (continuous_const.mul ((contDiff_gtStateLogPartition V).continuous.comp hfield))
    have hc : Continuous (fun z₁ : I₁ → ℝ =>
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
          ∑ ξ : S,
            gtStateGibbs V
                (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ) := by
      apply (hdensity.mul continuous_const).mul
      apply continuous_finset_sum
      intro ξ _
      exact (((contDiff_gtStateGibbs V ξ).continuous.comp hfield).mul
        ((EuclideanSpace.proj ξ).continuous.comp hderiv))
    exact hc.aestronglyMeasurable
  have h_bound : ∀ᵐ z₁ ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1),
      ∀ x ∈ Metric.ball t ε, ‖F' x z₁‖ ≤ bound z₁ := by
    refine ae_of_all _ (fun z₁ x hx => ?_)
    let zE : EuclideanSpace ℝ I₁ := WithLp.toLp 2 z₁
    let H := gtHalfField A B₀ B₁ H₀ x z₀ zE
    let K := gtHalfFieldDeriv A B₀ B₁ x z₀ zE
    have hxI := hball x hx
    have hH : ‖H‖ ≤ R₀ + R₁ * ‖zE‖ := by
      have hsx : Real.sqrt x ≤ 1 := Real.sqrt_le_one.mpr hxI.2.le
      have hs1x : Real.sqrt (1 - x) ≤ 1 :=
        Real.sqrt_le_one.mpr (by linarith [hxI.1])
      calc
        ‖H‖ ≤ ‖Real.sqrt x • LA z₀‖ +
              ‖Real.sqrt (1-x) • (LB₀ z₀ + LB₁ zE)‖ + ‖H₀‖ := by
          dsimp [H, gtHalfField, LA, LB₀, LB₁]
          calc
            _ ≤ ‖Real.sqrt x • LA z₀ +
                Real.sqrt (1-x) • (LB₀ z₀ + LB₁ zE)‖ + ‖H₀‖ := norm_add_le _ _
            _ ≤ _ := by gcongr; exact norm_add_le _ _
        _ ≤ ‖LA z₀‖ + (‖LB₀ z₀‖ + ‖LB₁‖ * ‖zE‖) + ‖H₀‖ := by
          rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
            abs_of_nonneg (Real.sqrt_nonneg _), abs_of_nonneg (Real.sqrt_nonneg _)]
          have hinner : ‖LB₀ z₀ + LB₁ zE‖ ≤
              ‖LB₀ z₀‖ + ‖LB₁‖ * ‖zE‖ := by
            calc
              _ ≤ ‖LB₀ z₀‖ + ‖LB₁ zE‖ := norm_add_le _ _
              _ ≤ _ := by gcongr; exact LB₁.le_opNorm zE
          have hfirst : Real.sqrt x * ‖LA z₀‖ ≤ ‖LA z₀‖ := by
            nlinarith [norm_nonneg (LA z₀)]
          have hsecond : Real.sqrt (1-x) * ‖LB₀ z₀ + LB₁ zE‖ ≤
              ‖LB₀ z₀‖ + ‖LB₁‖ * ‖zE‖ := by
            calc
              _ ≤ 1 * ‖LB₀ z₀ + LB₁ zE‖ :=
                mul_le_mul_of_nonneg_right hs1x (norm_nonneg _)
              _ ≤ _ := by simpa using hinner
          linarith
        _ = R₀ + R₁ * ‖zE‖ := by dsimp [R₀, R₁]; ring
    have hK : ‖K‖ ≤ D₀ + D₁ * ‖zE‖ := by
      calc
        ‖K‖ ≤ |1 / (2 * Real.sqrt x)| * ‖LA z₀‖ +
              |1 / (2 * Real.sqrt (1-x))| * ‖LB₀ z₀‖ +
              |1 / (2 * Real.sqrt (1-x))| * ‖LB₁ zE‖ := by
          dsimp [K, gtHalfFieldDeriv, LA, LB₀, LB₁]
          let a := (1 / (2 * Real.sqrt x)) • LA z₀
          let b := (1 / (2 * Real.sqrt (1-x))) • LB₀ z₀
          let c := (1 / (2 * Real.sqrt (1-x))) • LB₁ zE
          calc
            ‖a - b - c‖ ≤ ‖a - b‖ + ‖c‖ := norm_sub_le _ _
            _ ≤ (‖a‖ + ‖b‖) + ‖c‖ := by
              gcongr
              exact norm_sub_le _ _
            _ = _ := by
              dsimp [a, b, c]
              simp only [norm_smul, Real.norm_eq_abs]
              ring
        _ ≤ cA * ‖LA z₀‖ + cB * ‖LB₀ z₀‖ +
              cB * (‖LB₁‖ * ‖zE‖) := by
          gcongr
          · exact hcoeffA x hx
          · exact hcoeffB x hx
          · exact hcoeffB x hx
          · exact LB₁.le_opNorm zE
        _ = D₀ + D₁ * ‖zE‖ := by dsimp [D₀, D₁]; ring
    have havg : |∑ ξ : S, gtStateGibbs V H ξ * K ξ| ≤ ‖K‖ := by
      simpa [fderiv_gtStateLogPartition_apply, Real.norm_eq_abs] using
        (ContinuousLinearMap.le_opNorm
          (fderiv ℝ (gtStateLogPartition V) H) K |>.trans
            (mul_le_mul_of_nonneg_right
              (norm_fderiv_gtStateLogPartition_le_one V H) (norm_nonneg K)))
    have hdensity : gtHalfDensity V A B₀ B₁ H₀ x z₀ zE ≤
        E₀ * Real.exp ((C * R₁) * ‖zE‖) := by
      let base := hasModerateGrowth_gtStateLogPartition V
      have hg := base.F_bound H
      have hm : base.m = 1 := by rfl
      rw [hm, pow_one] at hg
      unfold gtHalfDensity
      rw [show gtHalfField A B₀ B₁ H₀ x z₀ zE = H by rfl]
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      have hlog : (1 / 2 : ℝ) * gtStateLogPartition V H ≤ C * (1 + ‖H‖) := by
        have := le_abs_self (gtStateLogPartition V H)
        dsimp [C]
        nlinarith
      calc
        (1 / 2 : ℝ) * gtStateLogPartition V H ≤ C * (1 + ‖H‖) := hlog
        _ ≤ C * (1 + (R₀ + R₁ * ‖zE‖)) :=
          mul_le_mul_of_nonneg_left (by linarith) hC
        _ = C * (1 + R₀) + C * R₁ * ‖zE‖ := by ring
    rw [Real.norm_eq_abs]
    dsimp [F', bound]
    rw [abs_mul, abs_mul]
    change |gtHalfDensity V A B₀ B₁ H₀ x z₀ zE| * |(1 / 2 : ℝ)| *
        |∑ ξ : S, gtStateGibbs V H ξ * K ξ| ≤ _
    have hdensity0 : 0 ≤ gtHalfDensity V A B₀ B₁ H₀ x z₀ zE := by
      unfold gtHalfDensity
      exact Real.exp_nonneg _
    rw [abs_of_nonneg hdensity0,
      abs_of_nonneg (by norm_num : (0:ℝ)≤1/2)]
    calc
      gtHalfDensity V A B₀ B₁ H₀ x z₀ zE * (1 / 2) *
          |∑ ξ : S, gtStateGibbs V H ξ * K ξ| ≤
        gtHalfDensity V A B₀ B₁ H₀ x z₀ zE * (1 / 2) *
          (D₀ + D₁ * ‖zE‖) := by
            gcongr
            exact havg.trans hK
      _ ≤
        (E₀ * Real.exp ((C * R₁) * ‖zE‖)) * (1 / 2) *
          (D₀ + D₁ * ‖zE‖) := by gcongr
      _ = (1 / 2) * (D₀ + D₁ * ‖zE‖) *
          (E₀ * Real.exp (C * R₁ * ‖zE‖)) := by ring
  have h_diff : ∀ᵐ z₁ ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1),
      ∀ x ∈ Metric.ball t ε, HasDerivAt (fun u => F u z₁) (F' x z₁) x := by
    refine ae_of_all _ (fun z₁ x hx => ?_)
    exact hasDerivAt_gtHalfDensity V A B₀ B₁ H₀ (hball x hx) z₀
      (WithLp.toLp 2 z₁)
  have hmain :=
    (hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := Measure.pi (fun _ : I₁ => gaussianReal 0 1))
      (F := F) (F' := F') (x₀ := t) (bound := bound)
      (s := Metric.ball t ε) (Metric.ball_mem_nhds t hε)
      hF_meas hF_int hF'_meas h_bound hbound_int h_diff).2
  change HasDerivAt
    (fun u => gtHalfDenominator V A B₀ B₁ H₀ u z₀) _ t
  exact hmain

/-- Pointwise derivative of the half transform, before either Gaussian
integration by parts. -/
lemma hasDerivAt_gtHalfTransform_raw
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) :
    HasDerivAt (fun u => gtHalfTransform V A B₀ B₁ H₀ u z₀)
      (2 * ((∫ z₁ : I₁ → ℝ,
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
            ∑ ξ : S,
              gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
                gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) /
          gtHalfDenominator V A B₀ B₁ H₀ t z₀)) t := by
  have hd := hasDerivAt_gtHalfDenominator V A B₀ B₁ H₀ ht z₀
  have hl := hd.log (gtHalfDenominator_pos V A B₀ B₁ H₀ t z₀).ne'
  have hout := hl.const_mul 2
  simpa [gtHalfTransform] using hout

/-- The same pointwise derivative, separated into the outer coordinates and
the normalized inner Gaussian expectation. -/
lemma hasDerivAt_gtHalfTransform_before_ibp
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) :
    HasDerivAt (fun u => gtHalfTransform V A B₀ B₁ H₀ u z₀)
      (∑ ξ : S,
          gtHalfGibbsWeight V A B₀ B₁ H₀ t z₀ ξ *
            ((1 / (2 * Real.sqrt t)) * inner ℝ (A ξ) z₀ -
              (1 / (2 * Real.sqrt (1-t))) * inner ℝ (B₀ ξ) z₀) -
        (1 / (2 * Real.sqrt (1-t))) * ∑ ξ : S,
          ∫ z₁ : I₁ → ℝ,
            inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ) *
              (gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
                gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ /
                    gtHalfDenominator V A B₀ B₁ H₀ t z₀)
            ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) t := by
  classical
  have hraw := hasDerivAt_gtHalfTransform_raw V A B₀ B₁ H₀ ht z₀
  apply hraw.congr_deriv
  let D := gtHalfDenominator V A B₀ B₁ H₀ t z₀
  have hD : D ≠ 0 := (gtHalfDenominator_pos V A B₀ B₁ H₀ t z₀).ne'
  have hcoord (ξ : S) : Integrable (fun z₁ : I₁ → ℝ =>
      gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
        (gtStateGibbs V
          (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
          gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ))
      (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
    let LA := gtCoefficientCLM A
    let LB₀ := gtCoefficientCLM B₀
    let LB₁ := gtCoefficientCLM B₁
    let R₀ : ℝ := Real.sqrt t * ‖LA z₀‖ +
      Real.sqrt (1-t) * ‖LB₀ z₀‖ + ‖H₀‖
    let R₁ : ℝ := Real.sqrt (1-t) * ‖LB₁‖
    let D₀ : ℝ := |1 / (2 * Real.sqrt t)| * ‖LA z₀‖ +
      |1 / (2 * Real.sqrt (1-t))| * ‖LB₀ z₀‖
    let D₁ : ℝ := |1 / (2 * Real.sqrt (1-t))| * ‖LB₁‖
    let C : ℝ := (hasModerateGrowth_gtStateLogPartition V).C / 2
    let M : ℝ := Real.exp (C * (1 + R₀))
    let c : ℝ := C * R₁
    let bound : (I₁ → ℝ) → ℝ := fun z₁ =>
      (1 / 2 : ℝ) * M *
        (D₀ * Real.exp (c * ‖(WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁)‖) +
          D₁ * (‖(WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁)‖ *
            Real.exp (c * ‖(WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁)‖)))
    have hbound : Integrable bound
        (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
      have he := integrable_exp_mul_norm_gaussianProduct (I := I₁) c
      have hne := integrable_norm_mul_exp_mul_norm_gaussianProduct (I := I₁) c
      have hi := ((he.const_mul D₀).add (hne.const_mul D₁)).const_mul M |>.const_mul (1/2)
      convert hi using 1
      funext z₁
      dsimp [bound]
      ring
    apply hbound.mono'
    · have hfield : Continuous (fun z₁ : I₁ → ℝ =>
          gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) := by
        unfold gtHalfField
        fun_prop
      have hderiv : Continuous (fun z₁ : I₁ → ℝ =>
          gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁)) := by
        unfold gtHalfFieldDeriv
        fun_prop
      exact ((((Real.continuous_exp.comp
        (continuous_const.mul ((contDiff_gtStateLogPartition V).continuous.comp hfield))).mul
          continuous_const).mul
            (((contDiff_gtStateGibbs V ξ).continuous.comp hfield).mul
              ((EuclideanSpace.proj ξ).continuous.comp hderiv))).aestronglyMeasurable)
    · filter_upwards with z₁
      let zE : EuclideanSpace ℝ I₁ := WithLp.toLp 2 z₁
      let H := gtHalfField A B₀ B₁ H₀ t z₀ zE
      let K := gtHalfFieldDeriv A B₀ B₁ t z₀ zE
      have hR₀ : 0 ≤ R₀ := by dsimp [R₀]; positivity
      have hR₁ : 0 ≤ R₁ := by dsimp [R₁]; positivity
      have hD₀ : 0 ≤ D₀ := by dsimp [D₀]; positivity
      have hD₁ : 0 ≤ D₁ := by dsimp [D₁]; positivity
      have hC : 0 ≤ C := by
        dsimp [C]
        exact div_nonneg (hasModerateGrowth_gtStateLogPartition V).Cpos.le (by norm_num)
      have hH : ‖H‖ ≤ R₀ + R₁ * ‖zE‖ := by
        calc
          ‖H‖ ≤ (‖Real.sqrt t • LA z₀‖ +
                ‖Real.sqrt (1-t) • (LB₀ z₀ + LB₁ zE)‖) + ‖H₀‖ := by
            dsimp [H, gtHalfField, LA, LB₀, LB₁]
            calc
              _ ≤ ‖Real.sqrt t • gtCoefficientCLM A z₀ +
                    Real.sqrt (1-t) •
                      (gtCoefficientCLM B₀ z₀ + gtCoefficientCLM B₁ zE)‖ +
                    ‖H₀‖ := norm_add_le _ _
              _ ≤ _ := by gcongr; exact norm_add_le _ _
          _ ≤ ‖Real.sqrt t • LA z₀‖ +
                (‖Real.sqrt (1-t) • LB₀ z₀‖ +
                  Real.sqrt (1-t) * (‖LB₁‖ * ‖zE‖)) + ‖H₀‖ := by
            have hinner := norm_add_le (LB₀ z₀) (LB₁ zE)
            have hop := LB₁.le_opNorm zE
            have hscaled : ‖Real.sqrt (1-t) • (LB₀ z₀ + LB₁ zE)‖ ≤
                ‖Real.sqrt (1-t) • LB₀ z₀‖ +
                  Real.sqrt (1-t) * (‖LB₁‖ * ‖zE‖) := by
              rw [norm_smul, norm_smul, Real.norm_eq_abs,
                abs_of_nonneg (Real.sqrt_nonneg _)]
              exact mul_le_mul_of_nonneg_left
                (hinner.trans (add_le_add_right hop _)) (Real.sqrt_nonneg _) |>.trans_eq (by
                  rw [mul_add])
            linarith
          _ = R₀ + R₁ * ‖zE‖ := by
            dsimp [R₀, R₁]
            simp only [norm_smul, Real.norm_eq_abs,
              abs_of_nonneg (Real.sqrt_nonneg _)]
            ring
      have hKξ : |K ξ| ≤ D₀ + D₁ * ‖zE‖ := by
        have hcξ : |K ξ| ≤ ‖K‖ := by
          simpa [Real.norm_eq_abs] using PiLp.norm_apply_le (p := (2 : ENNReal)) K ξ
        have hKn : ‖K‖ ≤ D₀ + D₁ * ‖zE‖ := by
          calc
            ‖K‖ ≤ |1 / (2 * Real.sqrt t)| * ‖LA z₀‖ +
                |1 / (2 * Real.sqrt (1-t))| * ‖LB₀ z₀‖ +
                |1 / (2 * Real.sqrt (1-t))| * ‖LB₁ zE‖ := by
              dsimp [K, gtHalfFieldDeriv, LA, LB₀, LB₁]
              calc
                _ ≤ ‖(1 / (2 * Real.sqrt t)) • gtCoefficientCLM A z₀ -
                      (1 / (2 * Real.sqrt (1-t))) • gtCoefficientCLM B₀ z₀‖ +
                    ‖(1 / (2 * Real.sqrt (1-t))) • gtCoefficientCLM B₁ zE‖ :=
                  norm_sub_le _ _
                _ ≤ (‖(1 / (2 * Real.sqrt t)) • gtCoefficientCLM A z₀‖ +
                      ‖(1 / (2 * Real.sqrt (1-t))) • gtCoefficientCLM B₀ z₀‖) +
                    ‖(1 / (2 * Real.sqrt (1-t))) • gtCoefficientCLM B₁ zE‖ := by
                  gcongr
                  exact norm_sub_le _ _
                _ = _ := by simp only [norm_smul, Real.norm_eq_abs]
            _ ≤ D₀ + D₁ * ‖zE‖ := by
              dsimp [D₀, D₁]
              nlinarith [LB₁.le_opNorm zE,
                abs_nonneg (1 / (2 * Real.sqrt (1-t)))]
        exact hcξ.trans hKn
      have hdensity : gtHalfDensity V A B₀ B₁ H₀ t z₀ zE ≤
          M * Real.exp (c * ‖zE‖) := by
        let base := hasModerateGrowth_gtStateLogPartition V
        have hg := base.F_bound H
        have hm : base.m = 1 := by rfl
        rw [hm, pow_one] at hg
        unfold gtHalfDensity
        rw [show gtHalfField A B₀ B₁ H₀ t z₀ zE = H by rfl, ← Real.exp_add]
        apply Real.exp_le_exp.mpr
        have hlog : (1 / 2 : ℝ) * gtStateLogPartition V H ≤ C * (1 + ‖H‖) := by
          have := le_abs_self (gtStateLogPartition V H)
          dsimp [C]
          nlinarith
        calc
          _ ≤ C * (1 + ‖H‖) := hlog
          _ ≤ C * (1 + (R₀ + R₁ * ‖zE‖)) :=
            mul_le_mul_of_nonneg_left (by linarith) hC
          _ = C * (1 + R₀) + c * ‖zE‖ := by dsimp [c]; ring
      have hg0 := gtStateGibbs_nonneg V H ξ
      have hg1 := gtStateGibbs_le_one V H ξ
      have hd0 : 0 ≤ gtHalfDensity V A B₀ B₁ H₀ t z₀ zE := by
        unfold gtHalfDensity
        positivity
      rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_mul,
        abs_of_nonneg hd0, abs_of_nonneg (by norm_num : (0:ℝ)≤1/2),
        abs_of_nonneg hg0]
      dsimp [bound]
      have hprod : gtHalfDensity V A B₀ B₁ H₀ t z₀ zE *
          gtStateGibbs V H ξ * |K ξ| ≤
          (M * Real.exp (c * ‖zE‖)) * (D₀ + D₁ * ‖zE‖) := by
        calc
          _ ≤ gtHalfDensity V A B₀ B₁ H₀ t z₀ zE * 1 *
              (D₀ + D₁ * ‖zE‖) := by gcongr
          _ ≤ _ := by
            simpa using mul_le_mul_of_nonneg_right hdensity
              (by positivity : 0 ≤ D₀ + D₁ * ‖zE‖)
      calc
        gtHalfDensity V A B₀ B₁ H₀ t z₀ zE * (1/2) *
            (gtStateGibbs V H ξ * |K ξ|) ≤
          (1/2) * ((M * Real.exp (c * ‖zE‖)) *
            (D₀ + D₁ * ‖zE‖)) := by nlinarith
        _ = (1/2) * M *
            (D₀ * Real.exp (c * ‖zE‖) +
              D₁ * (‖zE‖ * Real.exp (c * ‖zE‖))) := by ring
  rw [show (fun z₁ : I₁ → ℝ =>
      gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
        ∑ ξ : S,
          gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
            gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ) =
      fun z₁ => ∑ ξ : S,
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
          (gtStateGibbs V
            (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
            gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ) by
        funext z₁
        simp only [Finset.mul_sum]]
  rw [integral_finset_sum]
  · have hbase (ξ : S) : Integrable (fun z₁ : I₁ → ℝ =>
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
          gtStateGibbs V
            (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ)
        (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) :=
      integrable_gtHalfDensity_mul_gibbs V A B₀ B₁ H₀ t z₀ ξ
    let cA : ℝ := 1 / (2 * Real.sqrt t)
    let cB : ℝ := 1 / (2 * Real.sqrt (1-t))
    have hsqrtB : Real.sqrt (1-t) ≠ 0 :=
      (Real.sqrt_pos.2 (sub_pos.mpr ht.2)).ne'
    have hcB : cB ≠ 0 := by
      dsimp [cB]
      exact one_div_ne_zero (mul_ne_zero (by norm_num) hsqrtB)
    have hinner (ξ : S) : Integrable (fun z₁ : I₁ → ℝ =>
        gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
          gtStateGibbs V
            (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
          inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ))
        (Measure.pi (fun _ : I₁ => gaussianReal 0 1)) := by
      let outer : ℝ := cA * inner ℝ (A ξ) z₀ - cB * inner ℝ (B₀ ξ) z₀
      have ho := (hbase ξ).mul_const ((1 / 2 : ℝ) * outer)
      have hd := (hcoord ξ).sub ho
      have hs := hd.const_mul (-2 / cB)
      apply hs.congr
      filter_upwards with z₁
      simp [gtHalfFieldDeriv, gtCoefficientCLM_apply, cA, cB, outer]
      field_simp [hcB, hsqrtB]
      ring_nf
      rw [real_inner_comm]
    have hone (ξ : S) :
        2 * ((∫ z₁ : I₁ → ℝ,
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
            (gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ)
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) / D) =
          ((∫ z₁ : I₁ → ℝ,
              gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
                gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ
              ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) / D) *
            (cA * inner ℝ (A ξ) z₀ - cB * inner ℝ (B₀ ξ) z₀) -
          cB * ∫ z₁ : I₁ → ℝ,
            inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ) *
              (gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
                gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ / D)
            ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1) := by
      let outer : ℝ := cA * inner ℝ (A ξ) z₀ - cB * inner ℝ (B₀ ξ) z₀
      rw [show (fun z₁ : I₁ → ℝ =>
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
            (gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ)) =
          fun z₁ => (1 / 2 : ℝ) * outer *
              (gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
                gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ) -
            (1 / 2 : ℝ) * cB *
              (gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
                gtStateGibbs V
                  (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
                inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ)) by
            funext z₁
            simp [gtHalfFieldDeriv, gtCoefficientCLM_apply, outer, cA, cB]
            rw [real_inner_comm (B₁ ξ) (WithLp.toLp 2 z₁)]
            ring]
      rw [integral_sub ((hbase ξ).const_mul ((1/2) * outer))
        ((hinner ξ).const_mul ((1/2) * cB))]
      rw [integral_const_mul, integral_const_mul]
      rw [show (∫ z₁ : I₁ → ℝ,
          inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ) *
            (gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
              gtStateGibbs V
                (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ / D)
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) =
        (∫ z₁ : I₁ → ℝ,
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) *
            gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
            inner ℝ (WithLp.toLp 2 z₁ : EuclideanSpace ℝ I₁) (B₁ ξ)
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) / D by
          rw [← integral_div]
          apply integral_congr_ae
          filter_upwards with z₁
          ring]
      field_simp [hD]
      ring
    rw [show 2 * ((∑ ξ : S, ∫ z₁ : I₁ → ℝ,
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
            (gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ)
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) / D) =
        ∑ ξ : S, 2 * ((∫ z₁ : I₁ → ℝ,
          gtHalfDensity V A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁) * (1 / 2 : ℝ) *
            (gtStateGibbs V
              (gtHalfField A B₀ B₁ H₀ t z₀ (WithLp.toLp 2 z₁)) ξ *
              gtHalfFieldDeriv A B₀ B₁ t z₀ (WithLp.toLp 2 z₁) ξ)
          ∂Measure.pi (fun _ : I₁ => gaussianReal 0 1)) / D) by
            rw [Finset.sum_div, Finset.mul_sum]]
    simp_rw [hone]
    unfold gtHalfGibbsWeight
    dsimp [cA, cB]
    rw [Finset.sum_sub_distrib, Finset.mul_sum]
  · intro ξ _
    exact hcoord ξ

/-- The pointwise derivative after integration by parts in the inner
one-half-mass coordinate, but before integration by parts in the outer
mass-zero coordinate. -/
noncomputable def gtHalfDerivativeBeforeOuterIBP
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    (t : ℝ) (z₀ : EuclideanSpace ℝ I₀) : ℝ :=
  ∑ ξ : S,
      gtHalfGibbsWeight V A B₀ B₁ H₀ t z₀ ξ *
        ((1 / (2 * Real.sqrt t)) * inner ℝ (A ξ) z₀ -
          (1 / (2 * Real.sqrt (1-t))) * inner ℝ (B₀ ξ) z₀) -
    (1 / 2 : ℝ) * ∑ ξ : S,
      gtHalfGibbsWeight V A B₀ B₁ H₀ t z₀ ξ * inner ℝ (B₁ ξ) (B₁ ξ) +
    (1 / 4 : ℝ) * ∑ ξ : S, ∑ η : S,
      gtHalfInnerPairWeight V A B₀ B₁ H₀ t z₀ ξ η *
        inner ℝ (B₁ ξ) (B₁ η)

lemma hasDerivAt_gtHalfTransform_inner_ibp
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1)
    (z₀ : EuclideanSpace ℝ I₀) :
    HasDerivAt (fun u => gtHalfTransform V A B₀ B₁ H₀ u z₀)
      (gtHalfDerivativeBeforeOuterIBP V A B₀ B₁ H₀ t z₀) t := by
  classical
  apply (hasDerivAt_gtHalfTransform_before_ibp V A B₀ B₁ H₀ ht z₀).congr_deriv
  have hs : Real.sqrt (1 - t) ≠ 0 :=
    (Real.sqrt_pos.2 (sub_pos.mpr ht.2)).ne'
  simp_rw [gtHalf_inner_stein V A B₀ B₁ H₀ t z₀]
  unfold gtHalfDerivativeBeforeOuterIBP gtHalfInnerCLM
  simp only [smul_apply]
  field_simp [hs]
  ring_nf
  ring

lemma hasDerivAt_gtHalfPressure_before_outer_ibp
    {S I₀ I₁ : Type*} [Fintype S] [Nonempty S]
    [Fintype I₀] [Fintype I₁]
    (V : S → ℝ) (A B₀ : S → EuclideanSpace ℝ I₀)
    (B₁ : S → EuclideanSpace ℝ I₁) (H₀ : GTStateSpace S)
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (gtHalfPressure V A B₀ B₁ H₀)
      (∫ z₀ : I₀ → ℝ,
        gtHalfDerivativeBeforeOuterIBP V A B₀ B₁ H₀ t (WithLp.toLp 2 z₀)
        ∂Measure.pi (fun _ : I₀ => gaussianReal 0 1)) t := by
  classical
  let ε : ℝ := min t (1 - t) / 2
  have hε : 0 < ε := by
    dsimp [ε]
    have : 0 < min t (1 - t) := lt_min ht.1 (sub_pos.mpr ht.2)
    linarith
  have hball : ∀ x ∈ Metric.ball t ε, x ∈ Set.Ioo (0 : ℝ) 1 := by
    intro x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hεt : ε ≤ t / 2 := by dsimp [ε]; gcongr; exact min_le_left _ _
    have hε1t : ε ≤ (1 - t) / 2 := by
      dsimp [ε]; gcongr; exact min_le_right _ _
    constructor
    · have := (abs_sub_lt_iff.1 hxt).2
      nlinarith
    · have := (abs_sub_lt_iff.1 hxt).1
      nlinarith
  let cA : ℝ := 1 / (2 * Real.sqrt (t / 2))
  let cB : ℝ := 1 / (2 * Real.sqrt ((1 - t) / 2))
  let C₀ : ℝ := ∑ ξ : S, (cA * ‖A ξ‖ + cB * ‖B₀ ξ‖)
  let C₁ : ℝ := (1 / 2 : ℝ) * ∑ ξ : S, |inner ℝ (B₁ ξ) (B₁ ξ)| +
    (1 / 4 : ℝ) * ∑ ξ : S, ∑ η : S, |inner ℝ (B₁ ξ) (B₁ η)|
  let bound : (I₀ → ℝ) → ℝ := fun z =>
    C₀ * ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I₀)‖ + C₁
  have hcA : 0 ≤ cA := by dsimp [cA]; positivity
  have hcB : 0 ≤ cB := by dsimp [cB]; positivity
  have hC₀ : 0 ≤ C₀ := by
    dsimp [C₀]
    exact Finset.sum_nonneg fun ξ _ => add_nonneg
      (mul_nonneg hcA (norm_nonneg _)) (mul_nonneg hcB (norm_nonneg _))
  have hC₁ : 0 ≤ C₁ := by dsimp [C₁]; positivity
  have hbound : Integrable bound
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) := by
    exact ((integrable_norm_gaussianProduct (I := I₀)).const_mul C₀).add
      (integrable_const C₁)
  have hcoeffA : ∀ x ∈ Metric.ball t ε,
      |1 / (2 * Real.sqrt x)| ≤ cA := by
    intro x hx
    have hxI := hball x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hεt : ε ≤ t / 2 := by dsimp [ε]; gcongr; exact min_le_left _ _
    have htx : t / 2 ≤ x := by
      have := (abs_sub_lt_iff.1 hxt).2
      nlinarith
    have hp : 0 < 2 * Real.sqrt (t / 2) := by
      have : 0 < t / 2 := by linarith [ht.1]
      positivity
    have hi : 1 / (2 * Real.sqrt x) ≤ 1 / (2 * Real.sqrt (t / 2)) := by
      simpa [one_div] using one_div_le_one_div_of_le hp
        (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt htx) (by norm_num))
    rw [abs_of_nonneg (by positivity : 0 ≤ 1 / (2 * Real.sqrt x))]
    exact hi
  have hcoeffB : ∀ x ∈ Metric.ball t ε,
      |1 / (2 * Real.sqrt (1-x))| ≤ cB := by
    intro x hx
    have hxI := hball x hx
    have hxt : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
    have hε1t : ε ≤ (1 - t) / 2 := by
      dsimp [ε]; gcongr; exact min_le_right _ _
    have htx : (1 - t) / 2 ≤ 1 - x := by
      have := (abs_sub_lt_iff.1 hxt).1
      nlinarith
    have hp : 0 < 2 * Real.sqrt ((1 - t) / 2) := by
      have : 0 < (1 - t) / 2 := by linarith [ht.2]
      positivity
    have hi : 1 / (2 * Real.sqrt (1-x)) ≤
        1 / (2 * Real.sqrt ((1-t)/2)) := by
      simpa [one_div] using one_div_le_one_div_of_le hp
        (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt htx) (by norm_num))
    rw [abs_of_nonneg (by positivity : 0 ≤ 1 / (2 * Real.sqrt (1-x)))]
    exact hi
  let F : ℝ → (I₀ → ℝ) → ℝ := fun x z =>
    gtHalfTransform V A B₀ B₁ H₀ x (WithLp.toLp 2 z)
  let F' : ℝ → (I₀ → ℝ) → ℝ := fun x z =>
    gtHalfDerivativeBeforeOuterIBP V A B₀ B₁ H₀ x (WithLp.toLp 2 z)
  have hFmeas : ∀ᶠ x in nhds t, AEStronglyMeasurable (F x)
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) := by
    filter_upwards with x
    exact (integrable_gtHalfTransform V A B₀ B₁ H₀ x).1
  have hFint : Integrable (F t)
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) :=
    integrable_gtHalfTransform V A B₀ B₁ H₀ t
  have hF'meas : AEStronglyMeasurable (F' t)
      (Measure.pi (fun _ : I₀ => gaussianReal 0 1)) := by
    have hw (ξ : S) : Continuous (fun z : EuclideanSpace ℝ I₀ =>
        gtHalfGibbsWeight V A B₀ B₁ H₀ t z ξ) := by
      rw [show (fun z : EuclideanSpace ℝ I₀ =>
          gtHalfGibbsWeight V A B₀ B₁ H₀ t z ξ) =
        fun z => gtHalfBaseWeight V (gtHalfInnerCLM B₁ t)
          (gtHalfOuterCLM A B₀ t z + H₀) ξ by
            funext z; rw [gtHalfGibbsWeight_eq_base]]
      exact (continuous_gtHalfBaseWeight V (gtHalfInnerCLM B₁ t) ξ).comp
        ((gtHalfOuterCLM A B₀ t).continuous.add continuous_const)
    have hu (ξ η : S) : Continuous (fun z : EuclideanSpace ℝ I₀ =>
        gtHalfInnerPairWeight V A B₀ B₁ H₀ t z ξ η) := by
      rw [show (fun z : EuclideanSpace ℝ I₀ =>
          gtHalfInnerPairWeight V A B₀ B₁ H₀ t z ξ η) =
        fun z => gtHalfBasePairWeight V (gtHalfInnerCLM B₁ t)
          (gtHalfOuterCLM A B₀ t z + H₀) ξ η by
            funext z; rw [gtHalfInnerPairWeight_eq_base]]
      exact (continuous_gtHalfBasePairWeight V (gtHalfInnerCLM B₁ t) ξ η).comp
        ((gtHalfOuterCLM A B₀ t).continuous.add continuous_const)
    have hc : Continuous (fun z : EuclideanSpace ℝ I₀ =>
        gtHalfDerivativeBeforeOuterIBP V A B₀ B₁ H₀ t z) := by
      unfold gtHalfDerivativeBeforeOuterIBP
      fun_prop
    exact (hc.comp (by fun_prop)).aestronglyMeasurable
  have hdom : ∀ᵐ z ∂Measure.pi (fun _ : I₀ => gaussianReal 0 1),
      ∀ x ∈ Metric.ball t ε, ‖F' x z‖ ≤ bound z := by
    refine ae_of_all _ (fun z x hx => ?_)
    let zE : EuclideanSpace ℝ I₀ := WithLp.toLp 2 z
    have hw0 (ξ : S) : 0 ≤ gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ :=
      gtHalfGibbsWeight_nonneg V A B₀ B₁ H₀ x zE ξ
    have hw1 (ξ : S) : gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ ≤ 1 := by
      rw [gtHalfGibbsWeight_eq_base]
      exact gtHalfBaseWeight_le_one V _ _ ξ
    have hu0 (ξ η : S) : 0 ≤ gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η :=
      gtHalfInnerPairWeight_nonneg V A B₀ B₁ H₀ x zE ξ η
    have hu1 (ξ η : S) : gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η ≤ 1 := by
      rw [gtHalfInnerPairWeight_eq_base]
      exact gtHalfBasePairWeight_le_one V _ _ ξ η
    rw [Real.norm_eq_abs]
    dsimp [F', gtHalfDerivativeBeforeOuterIBP]
    calc
      |_ - _ + _| ≤
          |∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
            ((1 / (2 * Real.sqrt x)) * inner ℝ (A ξ) zE -
              (1 / (2 * Real.sqrt (1-x))) * inner ℝ (B₀ ξ) zE)| +
          |(1/2:ℝ) * ∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
            inner ℝ (B₁ ξ) (B₁ ξ)| +
          |(1/4:ℝ) * ∑ ξ : S, ∑ η : S,
            gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η *
              inner ℝ (B₁ ξ) (B₁ η)| := by
            linarith [abs_add_le
              ((∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                ((1 / (2 * Real.sqrt x)) * inner ℝ (A ξ) zE -
                  (1 / (2 * Real.sqrt (1-x))) * inner ℝ (B₀ ξ) zE)) -
                (1/2:ℝ) * ∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                  inner ℝ (B₁ ξ) (B₁ ξ))
              ((1/4:ℝ) * ∑ ξ : S, ∑ η : S,
                gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η *
                  inner ℝ (B₁ ξ) (B₁ η)),
              abs_sub_le
                (∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                  ((1 / (2 * Real.sqrt x)) * inner ℝ (A ξ) zE -
                    (1 / (2 * Real.sqrt (1-x))) * inner ℝ (B₀ ξ) zE))
                ((1/2:ℝ) * ∑ ξ : S, gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                  inner ℝ (B₁ ξ) (B₁ ξ))]
      _ ≤ C₀ * ‖zE‖ + C₁ := by
        rw [abs_mul, abs_mul, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/2),
          abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/4)]
        apply add_le_add
        · calc
            |∑ ξ : S, _| ≤ ∑ ξ : S, |gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                ((1 / (2 * Real.sqrt x)) * inner ℝ (A ξ) zE -
                  (1 / (2 * Real.sqrt (1-x))) * inner ℝ (B₀ ξ) zE)| :=
              Finset.abs_sum_le_sum_abs _ _
            _ ≤ ∑ ξ : S, (cA * ‖A ξ‖ + cB * ‖B₀ ξ‖) * ‖zE‖ := by
              apply Finset.sum_le_sum
              intro ξ _
              rw [abs_mul, abs_of_nonneg (hw0 ξ)]
              calc
                gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                    |(1 / (2 * Real.sqrt x)) * inner ℝ (A ξ) zE -
                      (1 / (2 * Real.sqrt (1-x))) * inner ℝ (B₀ ξ) zE| ≤
                    1 * (|1 / (2 * Real.sqrt x)| * |inner ℝ (A ξ) zE| +
                      |1 / (2 * Real.sqrt (1-x))| * |inner ℝ (B₀ ξ) zE|) := by
                        gcongr
                        exact abs_sub _ _
                _ ≤ (cA * ‖A ξ‖ + cB * ‖B₀ ξ‖) * ‖zE‖ := by
                  have hA := abs_real_inner_le_norm (A ξ) zE
                  have hB := abs_real_inner_le_norm (B₀ ξ) zE
                  gcongr
                  · exact hcoeffA x hx
                  · exact hA
                  · exact hcoeffB x hx
                  · exact hB
            _ = C₀ * ‖zE‖ := by rw [← Finset.sum_mul]; rfl
        · dsimp [C₁]
          apply add_le_add
          · gcongr
            calc
              |∑ ξ : S, _| ≤ ∑ ξ : S,
                  |gtHalfGibbsWeight V A B₀ B₁ H₀ x zE ξ *
                    inner ℝ (B₁ ξ) (B₁ ξ)| := Finset.abs_sum_le_sum_abs _ _
              _ ≤ ∑ ξ : S, |inner ℝ (B₁ ξ) (B₁ ξ)| := by
                apply Finset.sum_le_sum
                intro ξ _
                rw [abs_mul, abs_of_nonneg (hw0 ξ)]
                exact mul_le_of_le_one_left (abs_nonneg _) (hw1 ξ)
          · gcongr
            calc
              |∑ ξ : S, ∑ η : S, _| ≤ ∑ ξ : S, |∑ η : S,
                  gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η *
                    inner ℝ (B₁ ξ) (B₁ η)| := Finset.abs_sum_le_sum_abs _ _
              _ ≤ ∑ ξ : S, ∑ η : S, |inner ℝ (B₁ ξ) (B₁ η)| := by
                apply Finset.sum_le_sum
                intro ξ _
                calc
                  |∑ η : S, _| ≤ ∑ η : S,
                      |gtHalfInnerPairWeight V A B₀ B₁ H₀ x zE ξ η *
                        inner ℝ (B₁ ξ) (B₁ η)| := Finset.abs_sum_le_sum_abs _ _
                  _ ≤ _ := by
                    apply Finset.sum_le_sum
                    intro η _
                    rw [abs_mul, abs_of_nonneg (hu0 ξ η)]
                    exact mul_le_of_le_one_left (abs_nonneg _) (hu1 ξ η)
      _ = bound z := rfl
  have hdiff : ∀ᵐ z ∂Measure.pi (fun _ : I₀ => gaussianReal 0 1),
      ∀ x ∈ Metric.ball t ε, HasDerivAt (fun u => F u z) (F' x z) x := by
    refine ae_of_all _ (fun z x hx => ?_)
    exact hasDerivAt_gtHalfTransform_inner_ibp V A B₀ B₁ H₀ (hball x hx)
      (WithLp.toLp 2 z)
  have hm := (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := Measure.pi (fun _ : I₀ => gaussianReal 0 1))
    (F := F) (F' := F') (x₀ := t) (bound := bound)
    (s := Metric.ball t ε) (Metric.ball_mem_nhds t hε)
    hFmeas hFint hF'meas hdom hbound hdiff).2
  exact hm

end SpinGlass.AT
