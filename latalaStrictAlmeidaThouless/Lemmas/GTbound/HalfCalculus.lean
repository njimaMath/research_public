import Lemmas.GTbound.HalfStep
import Mathlib.Analysis.Calculus.FDeriv.Mul

open MeasureTheory ProbabilityTheory Real BigOperators Filter Topology
open scoped ContDiff

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace SpinGlass.AT

/-- The half-mass density, viewed directly as a function of the finite-state
field.  This small wrapper keeps the calculus formulas readable. -/
noncomputable def gtHalfBaseDensity
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H : GTStateSpace S) : ℝ :=
  Real.exp ((1 / 2 : ℝ) * gtStateLogPartition V H)

lemma contDiff_gtHalfBaseDensity
    {S : Type*} [Fintype S] [Nonempty S] (V : S → ℝ) :
    ContDiff ℝ ∞ (gtHalfBaseDensity V) := by
  unfold gtHalfBaseDensity
  exact (contDiff_const.mul (contDiff_gtStateLogPartition V)).exp

lemma gtHalfBaseDensity_pos
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H : GTStateSpace S) :
    0 < gtHalfBaseDensity V H := by
  unfold gtHalfBaseDensity
  positivity

lemma fderiv_gtHalfBaseDensity_apply
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H K : GTStateSpace S) :
    fderiv ℝ (gtHalfBaseDensity V) H K =
      gtHalfBaseDensity V H * (1 / 2 : ℝ) *
        ∑ ξ : S, gtStateGibbs V H ξ * K ξ := by
  have hd : HasFDerivAt (gtStateLogPartition V)
      (fderiv ℝ (gtStateLogPartition V) H) H :=
    ((contDiff_gtStateLogPartition V).differentiable (by simp)
      ).differentiableAt.hasFDerivAt
  have he := (hd.const_mul (1 / 2 : ℝ)).exp
  have happ := congrArg
    (fun L : GTStateSpace S →L[ℝ] ℝ => L K) he.fderiv
  change fderiv ℝ (gtHalfBaseDensity V) H K = _ at happ
  rw [happ]
  simp only [ContinuousLinearMap.smul_apply, smul_eq_mul]
  rw [fderiv_gtStateLogPartition_apply]
  unfold gtHalfBaseDensity
  ring

lemma contDiff_gtHalfBaseDensity_mul_gibbs
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (ξ : S) :
    ContDiff ℝ ∞ (fun H : GTStateSpace S =>
      gtHalfBaseDensity V H * gtStateGibbs V H ξ) :=
  (contDiff_gtHalfBaseDensity V).mul (contDiff_gtStateGibbs V ξ)

lemma fderiv_gtHalfBaseDensity_mul_gibbs_apply
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H K : GTStateSpace S) (ξ : S) :
    fderiv ℝ (fun H' : GTStateSpace S =>
        gtHalfBaseDensity V H' * gtStateGibbs V H' ξ) H K =
      gtHalfBaseDensity V H * gtStateGibbs V H ξ *
        (K ξ - (1 / 2 : ℝ) *
          ∑ η : S, gtStateGibbs V H η * K η) := by
  have hd : HasFDerivAt (gtHalfBaseDensity V)
      (fderiv ℝ (gtHalfBaseDensity V) H) H :=
    ((contDiff_gtHalfBaseDensity V).differentiable (by simp)
      ).differentiableAt.hasFDerivAt
  have hg : HasFDerivAt
      (fun H' : GTStateSpace S => gtStateGibbs V H' ξ)
      (fderiv ℝ (fun H' : GTStateSpace S => gtStateGibbs V H' ξ) H) H :=
    ((contDiff_gtStateGibbs V ξ).differentiable (by simp)
      ).differentiableAt.hasFDerivAt
  have hp := hd.mul hg
  have happ := congrArg
    (fun L : GTStateSpace S →L[ℝ] ℝ => L K) hp.fderiv
  change fderiv ℝ (fun H' : GTStateSpace S =>
      gtHalfBaseDensity V H' * gtStateGibbs V H' ξ) H K = _ at happ
  rw [happ]
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
    smul_eq_mul]
  rw [fderiv_gtHalfBaseDensity_apply, fderiv_gtStateGibbs_apply]
  ring

lemma norm_fderiv_gtHalfBaseDensity_le
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H : GTStateSpace S) :
    ‖fderiv ℝ (gtHalfBaseDensity V) H‖ ≤
      (1 / 2 : ℝ) * gtHalfBaseDensity V H := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by
    exact mul_nonneg (by norm_num) (Real.exp_nonneg _))
  intro K
  rw [Real.norm_eq_abs, fderiv_gtHalfBaseDensity_apply]
  have havg : |∑ ξ : S, gtStateGibbs V H ξ * K ξ| ≤ ‖K‖ := by
    simpa [fderiv_gtStateLogPartition_apply, Real.norm_eq_abs] using
      (ContinuousLinearMap.le_opNorm
        (fderiv ℝ (gtStateLogPartition V) H) K |>.trans
          (mul_le_mul_of_nonneg_right
            (norm_fderiv_gtStateLogPartition_le_one V H) (norm_nonneg K)))
  rw [abs_mul, abs_mul, abs_of_nonneg (gtHalfBaseDensity_pos V H).le,
    abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
  calc
    gtHalfBaseDensity V H * (1 / 2) *
        |∑ ξ : S, gtStateGibbs V H ξ * K ξ| ≤
      gtHalfBaseDensity V H * (1 / 2) * ‖K‖ :=
        mul_le_mul_of_nonneg_left havg
          (mul_nonneg (gtHalfBaseDensity_pos V H).le (by norm_num))
    _ = (1 / 2) * gtHalfBaseDensity V H * ‖K‖ := by ring

lemma norm_fderiv_gtHalfBaseDensity_mul_gibbs_le
    {S : Type*} [Fintype S] [Nonempty S]
    (V : S → ℝ) (H : GTStateSpace S) (ξ : S) :
    ‖fderiv ℝ (fun H' : GTStateSpace S =>
        gtHalfBaseDensity V H' * gtStateGibbs V H' ξ) H‖ ≤
      2 * gtHalfBaseDensity V H := by
  apply ContinuousLinearMap.opNorm_le_bound _
    (mul_nonneg (by norm_num) (gtHalfBaseDensity_pos V H).le)
  intro K
  rw [Real.norm_eq_abs, fderiv_gtHalfBaseDensity_mul_gibbs_apply]
  have hKξ : |K ξ| ≤ ‖K‖ := by
    simpa [Real.norm_eq_abs] using
      PiLp.norm_apply_le (p := (2 : ENNReal)) K ξ
  have havg : |∑ η : S, gtStateGibbs V H η * K η| ≤ ‖K‖ := by
    simpa [fderiv_gtStateLogPartition_apply, Real.norm_eq_abs] using
      (ContinuousLinearMap.le_opNorm
        (fderiv ℝ (gtStateLogPartition V) H) K |>.trans
          (mul_le_mul_of_nonneg_right
            (norm_fderiv_gtStateLogPartition_le_one V H) (norm_nonneg K)))
  have hg0 := gtStateGibbs_nonneg V H ξ
  have hg1 := gtStateGibbs_le_one V H ξ
  rw [abs_mul, abs_mul, abs_of_nonneg (gtHalfBaseDensity_pos V H).le,
    abs_of_nonneg hg0]
  have hd := abs_sub (K ξ)
    ((1 / 2 : ℝ) * ∑ η : S, gtStateGibbs V H η * K η)
  rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)] at hd
  have hd2 :
      |K ξ - (1 / 2) * ∑ η : S, gtStateGibbs V H η * K η| ≤
        (3 / 2 : ℝ) * ‖K‖ := by
    linarith
  have hρ : 0 ≤ gtHalfBaseDensity V H := (gtHalfBaseDensity_pos V H).le
  calc
    gtHalfBaseDensity V H * gtStateGibbs V H ξ *
        |K ξ - (1 / 2) * ∑ η : S, gtStateGibbs V H η * K η| ≤
      (gtHalfBaseDensity V H * 1) * ((3 / 2 : ℝ) * ‖K‖) :=
        mul_le_mul (mul_le_mul_of_nonneg_left hg1 hρ) hd2
          (abs_nonneg _) (by positivity)
    _ ≤ (2 * gtHalfBaseDensity V H) * ‖K‖ := by
      nlinarith [norm_nonneg K]

/-- Local exponential domination for the half density along an affine
Gaussian field. -/
lemma gtHalfBaseDensity_affine_local_bound
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H H' : GTStateSpace S) (z : EuclideanSpace ℝ I)
    (hnear : ‖H' - H‖ < 1) :
    gtHalfBaseDensity V (L z + H') ≤
      Real.exp
        ((hasModerateGrowth_gtStateLogPartition V).C *
          (2 + ‖L‖ + ‖H‖) * (1 + ‖z‖)) := by
  let base := hasModerateGrowth_gtStateLogPartition V
  have hH' : ‖H'‖ ≤ ‖H‖ + 1 := by
    have heq : H' = (H' - H) + H := by abel
    calc
      ‖H'‖ = ‖(H' - H) + H‖ := congrArg norm heq
      _ ≤ ‖H' - H‖ + ‖H‖ := norm_add_le _ _
      _ ≤ ‖H‖ + 1 := by linarith
  have hL := L.le_opNorm z
  have hadd := norm_add_le (L z) H'
  have hnorm : 1 + ‖L z + H'‖ ≤
      (2 + ‖L‖ + ‖H‖) * (1 + ‖z‖) := by
    nlinarith [norm_nonneg L, norm_nonneg H, norm_nonneg z]
  have hb := base.F_bound (L z + H')
  have hm : base.m = 1 := by rfl
  rw [hm, pow_one] at hb
  unfold gtHalfBaseDensity
  apply Real.exp_le_exp.mpr
  calc
    (1 / 2 : ℝ) * gtStateLogPartition V (L z + H') ≤
        base.C * (1 + ‖L z + H'‖) := by
      have := le_abs_self (gtStateLogPartition V (L z + H'))
      have hc0 : 0 ≤ base.C * (1 + ‖L z + H'‖) :=
        mul_nonneg base.Cpos.le (by positivity)
      nlinarith
    _ ≤ base.C * ((2 + ‖L‖ + ‖H‖) * (1 + ‖z‖)) :=
      mul_le_mul_of_nonneg_left hnorm base.Cpos.le
    _ = base.C * (2 + ‖L‖ + ‖H‖) * (1 + ‖z‖) := by ring

lemma integrable_gtHalfBaseDensity_affine_local_bound
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    Integrable (fun z : I → ℝ =>
      Real.exp
        ((hasModerateGrowth_gtStateLogPartition V).C *
          (2 + ‖L‖ + ‖H‖) *
            (1 + ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖)))
      (Measure.pi (fun _ : I => gaussianReal 0 1)) := by
  let c : ℝ := (hasModerateGrowth_gtStateLogPartition V).C *
    (2 + ‖L‖ + ‖H‖)
  have hi := integrable_exp_mul_norm_gaussianProduct (I := I) c
  have hc : (fun z : I → ℝ => Real.exp
      (c * (1 + ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖))) =
      fun z => Real.exp c * Real.exp
        (c * ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖) := by
    funext z
    rw [mul_add, mul_one, Real.exp_add]
  rw [hc]
  exact hi.const_mul _

/-- Gaussian convolution of the half density as a function of its base
finite-state field. -/
noncomputable def gtHalfBaseDenominator
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) : ℝ :=
  ∫ z : I → ℝ,
    gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H)
    ∂Measure.pi (fun _ : I => gaussianReal 0 1)

noncomputable def gtHalfBaseNumerator
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) (ξ : S) : ℝ :=
  ∫ z : I → ℝ,
    gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H) *
      gtStateGibbs V (L (WithLp.toLp 2 z) + H) ξ
    ∂Measure.pi (fun _ : I => gaussianReal 0 1)

lemma integrable_gtHalfBaseDenominator_integrand
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    Integrable (fun z : I → ℝ =>
      gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H))
      (Measure.pi (fun _ : I => gaussianReal 0 1)) := by
  simpa [gtHalfBaseDensity] using
    integrable_exp_mul_gtStateLogPartition_affine V (1 / 2 : ℝ) L H

lemma integrable_gtHalfBaseNumerator_integrand
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) (ξ : S) :
    Integrable (fun z : I → ℝ =>
      gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H) *
        gtStateGibbs V (L (WithLp.toLp 2 z) + H) ξ)
      (Measure.pi (fun _ : I => gaussianReal 0 1)) := by
  have hd := integrable_gtHalfBaseDenominator_integrand V L H
  apply hd.mono'
  · have hc : Continuous (fun z : I → ℝ =>
        gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H) *
          gtStateGibbs V (L (WithLp.toLp 2 z) + H) ξ) :=
      ((contDiff_gtHalfBaseDensity V).continuous.comp (by fun_prop)).mul
        ((contDiff_gtStateGibbs V ξ).continuous.comp (by fun_prop))
    exact hc.aestronglyMeasurable
  · filter_upwards with z
    rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (gtHalfBaseDensity_pos V _).le,
      abs_of_nonneg (gtStateGibbs_nonneg V _ ξ)]
    exact mul_le_of_le_one_right (gtHalfBaseDensity_pos V _).le
      (gtStateGibbs_le_one V _ ξ)

lemma hasFDerivAt_gtHalfBaseDenominator
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    HasFDerivAt (gtHalfBaseDenominator V L)
      (∫ z : I → ℝ,
        fderiv ℝ (gtHalfBaseDensity V)
          (L (WithLp.toLp 2 z) + H)
        ∂Measure.pi (fun _ : I => gaussianReal 0 1)) H := by
  let μ : Measure (I → ℝ) := Measure.pi (fun _ : I => gaussianReal 0 1)
  let F : GTStateSpace S → (I → ℝ) → ℝ := fun H' z =>
    gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H')
  let F' : GTStateSpace S → (I → ℝ) →
      (GTStateSpace S →L[ℝ] ℝ) := fun H' z =>
    fderiv ℝ (gtHalfBaseDensity V) (L (WithLp.toLp 2 z) + H')
  let c : ℝ := (hasModerateGrowth_gtStateLogPartition V).C *
    (2 + ‖L‖ + ‖H‖)
  let bound : (I → ℝ) → ℝ := fun z =>
    (1 / 2 : ℝ) * Real.exp
      (c * (1 + ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖))
  have hbound : Integrable bound μ := by
    have hi := integrable_gtHalfBaseDensity_affine_local_bound V L H
    exact hi.const_mul (1 / 2 : ℝ)
  have hmain := hasFDerivAt_integral_of_dominated_of_fderiv_le
    (μ := μ) (F := F) (F' := F') (x₀ := H)
    (bound := bound) (s := Metric.ball H 1)
    (Metric.ball_mem_nhds H (by norm_num))
    (Filter.Eventually.of_forall fun H' => by
      exact (integrable_gtHalfBaseDenominator_integrand V L H').1)
    (integrable_gtHalfBaseDenominator_integrand V L H)
    (by
      have hc : Continuous (F' H) := by
        dsimp [F']
        have hfd : ContDiff ℝ 0 (fderiv ℝ (gtHalfBaseDensity V)) :=
          (contDiff_gtHalfBaseDensity V).fderiv_right
            (m := (0 : WithTop ℕ∞)) (by simp)
        exact hfd.continuous.comp (by fun_prop)
      exact (hc.measurable.comp (by fun_prop)).aestronglyMeasurable)
    (ae_of_all μ fun z H' hH' => by
      have hnear : ‖H' - H‖ < 1 := by
        simpa [Metric.mem_ball, dist_eq_norm] using hH'
      calc
        ‖F' H' z‖ ≤ (1 / 2 : ℝ) *
            gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H') :=
          norm_fderiv_gtHalfBaseDensity_le V _
        _ ≤ bound z := by
          dsimp [bound, c]
          gcongr
          exact gtHalfBaseDensity_affine_local_bound V L H H'
            (WithLp.toLp 2 z) hnear)
    hbound
    (ae_of_all μ fun z H' _ => by
      have hd : HasFDerivAt (gtHalfBaseDensity V)
          (fderiv ℝ (gtHalfBaseDensity V)
            (L (WithLp.toLp 2 z) + H'))
          (L (WithLp.toLp 2 z) + H') :=
        ((contDiff_gtHalfBaseDensity V).differentiable (by simp)
          ).differentiableAt.hasFDerivAt
      exact hd.comp H'
        ((ContinuousLinearMap.id ℝ (GTStateSpace S)).hasFDerivAt.const_add
          (L (WithLp.toLp 2 z))))
  change HasFDerivAt
    (fun H' => ∫ z : I → ℝ,
      gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H')
      ∂Measure.pi (fun _ : I => gaussianReal 0 1))
    (∫ z : I → ℝ,
      fderiv ℝ (gtHalfBaseDensity V) (L (WithLp.toLp 2 z) + H)
      ∂Measure.pi (fun _ : I => gaussianReal 0 1)) H
  exact hmain

lemma hasFDerivAt_gtHalfBaseNumerator
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) (ξ : S) :
    HasFDerivAt (fun H' => gtHalfBaseNumerator V L H' ξ)
      (∫ z : I → ℝ,
        fderiv ℝ (fun K : GTStateSpace S =>
          gtHalfBaseDensity V K * gtStateGibbs V K ξ)
          (L (WithLp.toLp 2 z) + H)
        ∂Measure.pi (fun _ : I => gaussianReal 0 1)) H := by
  let μ : Measure (I → ℝ) := Measure.pi (fun _ : I => gaussianReal 0 1)
  let G : GTStateSpace S → ℝ := fun K =>
    gtHalfBaseDensity V K * gtStateGibbs V K ξ
  let F : GTStateSpace S → (I → ℝ) → ℝ := fun H' z =>
    G (L (WithLp.toLp 2 z) + H')
  let F' : GTStateSpace S → (I → ℝ) →
      (GTStateSpace S →L[ℝ] ℝ) := fun H' z =>
    fderiv ℝ G (L (WithLp.toLp 2 z) + H')
  let c : ℝ := (hasModerateGrowth_gtStateLogPartition V).C *
    (2 + ‖L‖ + ‖H‖)
  let bound : (I → ℝ) → ℝ := fun z =>
    2 * Real.exp
      (c * (1 + ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖))
  have hbound : Integrable bound μ := by
    have hi := integrable_gtHalfBaseDensity_affine_local_bound V L H
    exact hi.const_mul 2
  have hmain := hasFDerivAt_integral_of_dominated_of_fderiv_le
    (μ := μ) (F := F) (F' := F') (x₀ := H)
    (bound := bound) (s := Metric.ball H 1)
    (Metric.ball_mem_nhds H (by norm_num))
    (Filter.Eventually.of_forall fun H' => by
      exact (integrable_gtHalfBaseNumerator_integrand V L H' ξ).1)
    (integrable_gtHalfBaseNumerator_integrand V L H ξ)
    (by
      have hfd : ContDiff ℝ 0 (fderiv ℝ G) :=
        (contDiff_gtHalfBaseDensity_mul_gibbs V ξ).fderiv_right
          (m := (0 : WithTop ℕ∞)) (by simp)
      exact ((hfd.continuous.comp (by fun_prop)).measurable.comp
        (by fun_prop)).aestronglyMeasurable)
    (ae_of_all μ fun z H' hH' => by
      have hnear : ‖H' - H‖ < 1 := by
        simpa [Metric.mem_ball, dist_eq_norm] using hH'
      calc
        ‖F' H' z‖ ≤ 2 *
            gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H') :=
          norm_fderiv_gtHalfBaseDensity_mul_gibbs_le V _ ξ
        _ ≤ bound z := by
          dsimp [bound, c]
          gcongr
          exact gtHalfBaseDensity_affine_local_bound V L H H'
            (WithLp.toLp 2 z) hnear)
    hbound
    (ae_of_all μ fun z H' _ => by
      have hd : HasFDerivAt G
          (fderiv ℝ G (L (WithLp.toLp 2 z) + H'))
          (L (WithLp.toLp 2 z) + H') :=
        ((contDiff_gtHalfBaseDensity_mul_gibbs V ξ).differentiable (by simp)
          ).differentiableAt.hasFDerivAt
      exact hd.comp H'
        ((ContinuousLinearMap.id ℝ (GTStateSpace S)).hasFDerivAt.const_add
          (L (WithLp.toLp 2 z))))
  change HasFDerivAt
    (fun H' => ∫ z : I → ℝ,
      gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H') *
        gtStateGibbs V (L (WithLp.toLp 2 z) + H') ξ
      ∂Measure.pi (fun _ : I => gaussianReal 0 1))
    (∫ z : I → ℝ,
      fderiv ℝ (fun K : GTStateSpace S =>
        gtHalfBaseDensity V K * gtStateGibbs V K ξ)
        (L (WithLp.toLp 2 z) + H)
      ∂Measure.pi (fun _ : I => gaussianReal 0 1)) H
  exact hmain

lemma gtHalfBaseDenominator_pos
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) :
    0 < gtHalfBaseDenominator V L H := by
  unfold gtHalfBaseDenominator
  rw [integral_pos_iff_support_of_nonneg
    (fun z => (gtHalfBaseDensity_pos V _).le)
    (integrable_gtHalfBaseDenominator_integrand V L H)]
  have hsupp : Function.support (fun z : I → ℝ =>
      gtHalfBaseDensity V (L (WithLp.toLp 2 z) + H)) = Set.univ := by
    ext z
    simp [Function.mem_support, (gtHalfBaseDensity_pos V _).ne']
  rw [hsupp]
  simp

/-- A Gibbs coordinate after the normalized inner half-mass tilt, expressed
as a function of the deterministic base field. -/
noncomputable def gtHalfBaseWeight
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) (ξ : S) : ℝ :=
  gtHalfBaseNumerator V L H ξ / gtHalfBaseDenominator V L H

lemma hasFDerivAt_gtHalfBaseWeight
    {S I : Type*} [Fintype S] [Nonempty S] [Fintype I]
    (V : S → ℝ) (L : EuclideanSpace ℝ I →L[ℝ] GTStateSpace S)
    (H : GTStateSpace S) (ξ : S) :
    HasFDerivAt (fun H' => gtHalfBaseWeight V L H' ξ)
      (fderiv ℝ (fun H' => gtHalfBaseWeight V L H' ξ) H) H := by
  have hn := hasFDerivAt_gtHalfBaseNumerator V L H ξ
  have hd := hasFDerivAt_gtHalfBaseDenominator V L H
  have hi := hd.differentiableAt.inv (gtHalfBaseDenominator_pos V L H).ne'
  have hm := hn.differentiableAt.mul hi
  change HasFDerivAt
    ((fun H' => gtHalfBaseNumerator V L H' ξ) *
      (gtHalfBaseDenominator V L)⁻¹)
    (fderiv ℝ ((fun H' => gtHalfBaseNumerator V L H' ξ) *
      (gtHalfBaseDenominator V L)⁻¹) H) H
  exact hm.hasFDerivAt

end SpinGlass.AT
