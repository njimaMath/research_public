import GeneralizedLatala.Coupled.Calculus

/-!
# Gaussian integration by parts for the coupled path

The joint Gaussian trace calculation for the differentiated coupled free energy.

Main declarations:
- `coupledFreeEnergy_time_ibp_trace_workspace`

Dependencies:
- the coupled calculus layer and Hilbert-space Gaussian integration by parts

This file corresponds to the relevant part of `blueprint_latala.txt`.
-/

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ENNReal NNReal Topology

set_option maxHeartbeats 800000

namespace SpinGlass
namespace GeneralizedLatala

universe uΩ uι

variable {Ω : Type uΩ} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

variable (N : ℕ) [NeZero N] (β h q : ℝ)
variable (sk : SKDisorder.{uΩ} (Ω := Ω) N β h)
variable (sim : SimpleDisorder.{uΩ} (Ω := Ω) N β q)

/-! ## Gaussian-IBP trace layer -/

private noncomputable def coupledIBPJointCLM (a b : ℝ) :
    WithLp 2 (EnergySpace N × EnergySpace N) →L[ℝ] EnergySpace N :=
  LinearMap.toContinuousLinearMap
    { toFun := fun p => a • (WithLp.ofLp p).1 + b • (WithLp.ofLp p).2
      map_add' := by intro x y; simp; abel
      map_smul' := by intro c x; simp [smul_add, smul_smul, mul_comm] }

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma abs_tiltedReplicaAverageDet_workspace_le
    (H : EnergySpace N) (coupling M : ℝ) (f : ReplicaFun N 2)
    (hM : 0 ≤ M) (hf : ∀ σs, |f σs| ≤ M) :
    |tiltedReplicaAverageDet (N := N) (q := q) H coupling f| ≤ M := by
  classical
  let W : ReplicaSpace N 2 → ℝ := fun σs =>
    Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) *
      ∏ l, gibbs_pmf N H (σs l)
  have hW (σs : ReplicaSpace N 2) : 0 ≤ W σs := by
    exact mul_nonneg (Real.exp_nonneg _)
      (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg N H (σs l))
  have hsum : 0 < ∑ σs : ReplicaSpace N 2, W σs := by
    simpa [W, tiltedReplicaPartitionDet, gibbs_average_n_det, mul_assoc] using
      tiltedReplicaPartitionDet_pos (N := N) (q := q) H coupling
  have hform :
      tiltedReplicaAverageDet (N := N) (q := q) H coupling f =
        (∑ σs : ReplicaSpace N 2, f σs * W σs) / ∑ σs, W σs := by
    simp [tiltedReplicaAverageDet, tiltedReplicaPartitionDet,
      gibbs_average_n_det, W, mul_assoc]
  rw [hform, abs_div, abs_of_pos hsum]
  apply (div_le_iff₀ hsum).2
  calc
    |∑ σs : ReplicaSpace N 2, f σs * W σs| ≤
        ∑ σs : ReplicaSpace N 2, |f σs * W σs| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ σs : ReplicaSpace N 2, |f σs| * W σs := by
      apply Finset.sum_congr rfl
      intro σs _
      rw [abs_mul, abs_of_nonneg (hW σs)]
    _ ≤ ∑ σs : ReplicaSpace N 2, M * W σs := by
      exact Finset.sum_le_sum fun σs _ => mul_le_mul_of_nonneg_right (hf σs) (hW σs)
    _ = M * ∑ σs : ReplicaSpace N 2, W σs := by rw [Finset.mul_sum]

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma abs_coupledHessianDet_workspace_le
    (H : EnergySpace N) (coupling : ℝ) (u v : EnergySpace N) :
    |coupledHessianDet (N := N) (q := q) H coupling u v| ≤
      (4 / (N : ℝ)) * ‖u‖ * ‖v‖ := by
  have hu (σs : ReplicaSpace N 2) : |pairEval (N := N) u σs| ≤ 2 * ‖u‖ := by
    unfold pairEval
    calc
      |u (σs 0) + u (σs 1)| ≤ |u (σs 0)| + |u (σs 1)| := abs_add_le _ _
      _ ≤ ‖u‖ + ‖u‖ := add_le_add (abs_apply_le_norm N u _) (abs_apply_le_norm N u _)
      _ = 2 * ‖u‖ := by ring
  have hv (σs : ReplicaSpace N 2) : |pairEval (N := N) v σs| ≤ 2 * ‖v‖ := by
    unfold pairEval
    calc
      |v (σs 0) + v (σs 1)| ≤ |v (σs 0)| + |v (σs 1)| := abs_add_le _ _
      _ ≤ ‖v‖ + ‖v‖ := add_le_add (abs_apply_le_norm N v _) (abs_apply_le_norm N v _)
      _ = 2 * ‖v‖ := by ring
  have huv (σs : ReplicaSpace N 2) :
      |pairEval (N := N) u σs * pairEval (N := N) v σs| ≤ 4 * ‖u‖ * ‖v‖ := by
    rw [abs_mul]
    nlinarith [hu σs, hv σs, abs_nonneg (pairEval N u σs),
      abs_nonneg (pairEval N v σs), norm_nonneg u, norm_nonneg v]
  have hxy := abs_tiltedReplicaAverageDet_workspace_le
    (N := N) (q := q) H coupling (4 * ‖u‖ * ‖v‖)
    (fun σs => pairEval N u σs * pairEval N v σs) (by positivity) huv
  have hx := abs_tiltedReplicaAverageDet_workspace_le
    (N := N) (q := q) H coupling (2 * ‖u‖) (pairEval N u) (by positivity) hu
  have hy := abs_tiltedReplicaAverageDet_workspace_le
    (N := N) (q := q) H coupling (2 * ‖v‖) (pairEval N v) (by positivity) hv
  have hN : 0 < (N : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  unfold coupledHessianDet
  rw [abs_mul]
  have hc : |1 / (2 * (N : ℝ))| = 1 / (2 * (N : ℝ)) := abs_of_pos (by positivity)
  rw [hc]
  calc
    1 / (2 * (N : ℝ)) *
        |tiltedReplicaAverageDet N q H coupling
            (fun σs => pairEval N u σs * pairEval N v σs) -
          tiltedReplicaAverageDet N q H coupling (pairEval N u) *
            tiltedReplicaAverageDet N q H coupling (pairEval N v)|
      ≤ 1 / (2 * (N : ℝ)) *
          (|tiltedReplicaAverageDet N q H coupling
              (fun σs => pairEval N u σs * pairEval N v σs)| +
            |tiltedReplicaAverageDet N q H coupling (pairEval N u)| *
              |tiltedReplicaAverageDet N q H coupling (pairEval N v)|) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        calc
          |tiltedReplicaAverageDet N q H coupling
                (fun σs => pairEval N u σs * pairEval N v σs) -
              tiltedReplicaAverageDet N q H coupling (pairEval N u) *
                tiltedReplicaAverageDet N q H coupling (pairEval N v)|
              ≤ |tiltedReplicaAverageDet N q H coupling
                  (fun σs => pairEval N u σs * pairEval N v σs)| +
                |tiltedReplicaAverageDet N q H coupling (pairEval N u) *
                  tiltedReplicaAverageDet N q H coupling (pairEval N v)| := abs_sub _ _
          _ = _ := by rw [abs_mul]
    _ ≤ 1 / (2 * (N : ℝ)) *
          (4 * ‖u‖ * ‖v‖ + (2 * ‖u‖) * (2 * ‖v‖)) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        exact add_le_add hxy
          (mul_le_mul hx hy (abs_nonneg _) (by positivity))
    _ = (4 / (N : ℝ)) * ‖u‖ * ‖v‖ := by field_simp; ring

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma fderiv_coupledFirstVariation_affine_workspace
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (A : E →L[ℝ] EnergySpace N) (field u : EnergySpace N)
    (x y : E) (Λ : ℝ) :
    fderiv ℝ
        (fun z : E => fderiv ℝ
          (fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ)
          (A z + field) u) x y =
      coupledHessianDet (N := N) (q := q) (A x + field) (Λ / 2) u (A y) := by
  have hΦ := contDiff_coupledFreeEnergyDet_beforeIBP (N := N) (q := q) Λ
  have hgrad : ContDiff ℝ 1
      (fderiv ℝ (fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ)) :=
    hΦ.fderiv_right (m := (1 : WithTop ℕ∞)) (by
      change (↑(2 : ℕ∞) : WithTop ℕ∞) ≤ ↑(⊤ : ℕ∞)
      exact WithTop.coe_le_coe.mpr le_top)
  have hscalar : ContDiff ℝ 1
      (fun H : EnergySpace N =>
        fderiv ℝ
          (fun K : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) K Λ) H u) :=
    hgrad.clm_apply contDiff_const
  have hc := (hscalar.differentiable (by norm_num)).differentiableAt.hasFDerivAt.comp
    x (A.hasFDerivAt.add_const field)
  change (fderiv ℝ
    ((fun H : EnergySpace N =>
      fderiv ℝ (fun K : EnergySpace N => coupledFreeEnergyDet N q K Λ) H u) ∘
      fun z => A z + field) x) y = _
  rw [hc.fderiv]
  simp only [ContinuousLinearMap.comp_apply]
  exact fderiv_coupledFirstVariation_apply_workspace
    (N := N) (q := q) (A x + field) u (A y) Λ

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private noncomputable def coupledFirstVariationModerateGrowth_workspace
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (A : E →L[ℝ] EnergySpace N) (field u : EnergySpace N) (Λ : ℝ) :
    PhysLean.Probability.GaussianIBP.HasModerateGrowth
      (fun x : E => fderiv ℝ
        (fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ)
        (A x + field) u) := by
  let C := 1 + (1 / (N : ℝ)) * ‖u‖ + (4 / (N : ℝ)) * ‖A‖ * ‖u‖
  refine ⟨C, 0, ?_, ?_, ?_⟩
  · dsimp [C]; positivity
  · intro x
    have hop := ContinuousLinearMap.le_opNorm
      (fderiv ℝ (fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ)
        (A x + field)) u
    have hb := opNorm_fderiv_coupledFreeEnergyDet_le_beforeIBP
      (N := N) (q := q) (A x + field) Λ
    rw [Real.norm_eq_abs] at hop
    calc
      |(fderiv ℝ (fun H => coupledFreeEnergyDet N q H Λ) (A x + field)) u|
          ≤ (1 / (N : ℝ)) * ‖u‖ :=
        hop.trans (mul_le_mul_of_nonneg_right hb (norm_nonneg u))
      _ ≤ C := by
        dsimp [C]
        have hn : 0 ≤ (4 / (N : ℝ)) * ‖A‖ * ‖u‖ := by positivity
        linarith
      _ = C * (1 + ‖x‖) ^ 0 := by simp
  · intro x
    simp only [pow_zero, mul_one]
    refine ContinuousLinearMap.opNorm_le_bound _ (by dsimp [C]; positivity) ?_
    intro y
    rw [fderiv_coupledFirstVariation_affine_workspace (N := N) (q := q)]
    have hb := abs_coupledHessianDet_workspace_le
      (N := N) (q := q) (A x + field) (Λ / 2) u (A y)
    rw [Real.norm_eq_abs]
    calc
      |coupledHessianDet N q (A x + field) (Λ / 2) u (A y)|
          ≤ (4 / (N : ℝ)) * ‖u‖ * ‖A y‖ := hb
      _ ≤ C * ‖y‖ := by
        have hAy := A.le_opNorm y
        have hfac : 0 ≤ 4 / (N : ℝ) * ‖u‖ := by positivity
        have hmul := mul_le_mul_of_nonneg_left hAy hfac
        calc
          (4 / (N : ℝ)) * ‖u‖ * ‖A y‖
              ≤ (4 / (N : ℝ)) * ‖u‖ * (‖A‖ * ‖y‖) := hmul
          _ = ((4 / (N : ℝ)) * ‖A‖ * ‖u‖) * ‖y‖ := by ring
          _ ≤ C * ‖y‖ := by
            apply mul_le_mul_of_nonneg_right _ (norm_nonneg y)
            dsimp [C]
            have h₁ : 0 ≤ (1 / (N : ℝ)) * ‖u‖ := by positivity
            linarith

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma coupledHessian_eq_secondFDeriv_workspace
    (H u v : EnergySpace N) (Λ : ℝ) :
    coupledHessianDet (N := N) (q := q) H (Λ / 2) u v =
      (fderiv ℝ
        (fderiv ℝ
          (fun K : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) K Λ)) H v) u := by
  have hΦ := contDiff_coupledFreeEnergyDet_beforeIBP (N := N) (q := q) Λ
  have hgrad : Differentiable ℝ
      (fderiv ℝ (fun K : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) K Λ)) :=
    (hΦ.fderiv_right (m := (1 : WithTop ℕ∞)) (by
      change (↑(2 : ℕ∞) : WithTop ℕ∞) ≤ ↑(⊤ : ℕ∞)
      exact WithTop.coe_le_coe.mpr le_top)).differentiable (by norm_num)
  have hc := hgrad.differentiableAt.hasFDerivAt.clm_apply
    (hasFDerivAt_const (x := H) (c := u))
  have happ := congrArg (fun L : EnergySpace N →L[ℝ] ℝ => L v) hc.fderiv
  have hcoupled := fderiv_coupledFirstVariation_apply_workspace
    (N := N) (q := q) H u v Λ
  simpa [ContinuousLinearMap.comp_apply] using hcoupled.symm.trans happ

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma coupledGaussianTrace_eq_stdBasis_workspace
    {Ω' : Type*} [MeasureSpace Ω'] [IsProbabilityMeasure (volume : Measure Ω')]
    (g : Ω' → EnergySpace N)
    (hg : PhysLean.Probability.GaussianIBP.IsGaussianHilbert g)
    (H : EnergySpace N) (Λ : ℝ) :
    (∑ i : hg.ι, (hg.τ i : ℝ) *
      coupledHessianDet (N := N) (q := q) H (Λ / 2) (hg.w i) (hg.w i)) =
      ∑ σ : Config N, ∑ τ : Config N,
        inner ℝ ((PhysLean.Probability.GaussianIBP.covOp (g := g) hg)
          (std_basis N σ)) (std_basis N τ) *
        coupledHessianDet (N := N) (q := q) H (Λ / 2)
          (std_basis N σ) (std_basis N τ) := by
  classical
  let D := fderiv ℝ
    (fderiv ℝ (fun K : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) K Λ)) H
  have hB (u v : EnergySpace N) :
      coupledHessianDet N q H (Λ / 2) u v = (D v) u := by
    exact coupledHessian_eq_secondFDeriv_workspace (N := N) (q := q) H u v Λ
  have hrepr (v : EnergySpace N) :
      v = ∑ σ : Config N, v σ • std_basis N σ := by
    ext τ
    simp [std_basis]
  simp_rw [hB]
  simp only [PhysLean.Probability.GaussianIBP.covOp_apply]
  have hinner (σ τ : Config N) :
      inner ℝ (∑ i : hg.ι,
        ((hg.τ i : ℝ) * inner ℝ (std_basis N σ) (hg.w i)) • hg.w i)
        (std_basis N τ) =
      ∑ i : hg.ι, (hg.τ i : ℝ) * (hg.w i) σ * (hg.w i) τ := by
    rw [sum_inner]
    apply Finset.sum_congr rfl
    intro i _
    rw [inner_smul_left, inner_std_basis_apply]
    rw [real_inner_comm, inner_std_basis_apply]
    simp
  simp_rw [hinner]
  have hexpand (i : hg.ι) :
      (D (hg.w i)) (hg.w i) =
        ∑ σ : Config N, ∑ τ : Config N,
          (hg.w i) σ * (hg.w i) τ * (D (std_basis N τ)) (std_basis N σ) := by
    let S := ∑ σ : Config N, (hg.w i) σ • std_basis N σ
    have hwi : hg.w i = S := hrepr (hg.w i)
    calc
      (D (hg.w i)) (hg.w i) = (D S) S :=
        congrArg₂ (fun v u => (D v) u) hwi hwi
      _ = _ := by
        simp [S, std_basis, map_sum, map_smul, smul_eq_mul,
          Finset.mul_sum, Finset.sum_mul]
        ring
  simp_rw [hexpand]
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro σ _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro τ _
  apply Finset.sum_congr rfl
  intro i _
  ring

private lemma coupledJointTrace_split_workspace
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    (a b a' b' : ℝ) (H : EnergySpace N) (Λ : ℝ) :
    (∑ i : (isGaussianHilbert_UV (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) hIndep).ι,
      ((isGaussianHilbert_UV (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) hIndep).τ i : ℝ) *
      coupledHessianDet N q H (Λ / 2)
        (coupledIBPJointCLM (N := N) a b
          ((isGaussianHilbert_UV (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) hIndep).w i))
        (coupledIBPJointCLM (N := N) a' b'
          ((isGaussianHilbert_UV (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) hIndep).w i))) =
      (a * a') * ∑ i : sk.hU.ι, (sk.hU.τ i : ℝ) *
        coupledHessianDet N q H (Λ / 2) (sk.hU.w i) (sk.hU.w i) +
      (b * b') * ∑ i : sim.hV.ι, (sim.hV.τ i : ℝ) *
        coupledHessianDet N q H (Λ / 2) (sim.hV.w i) (sim.hV.w i) := by
  classical
  have hU (i : sk.hU.ι) :
      coupledIBPJointCLM (N := N) a b (WithLp.toLp 2 (sk.hU.w i, 0)) =
        a • sk.hU.w i := by simp [coupledIBPJointCLM]
  have hU' (i : sk.hU.ι) :
      coupledIBPJointCLM (N := N) a' b' (WithLp.toLp 2 (sk.hU.w i, 0)) =
        a' • sk.hU.w i := by simp [coupledIBPJointCLM]
  have hV (i : sim.hV.ι) :
      coupledIBPJointCLM (N := N) a b (WithLp.toLp 2 (0, sim.hV.w i)) =
        b • sim.hV.w i := by simp [coupledIBPJointCLM]
  have hV' (i : sim.hV.ι) :
      coupledIBPJointCLM (N := N) a' b' (WithLp.toLp 2 (0, sim.hV.w i)) =
        b' • sim.hV.w i := by simp [coupledIBPJointCLM]
  simp only [isGaussianHilbert_UV,
    OrthonormalBasis.prod_apply, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    LinearMap.inl_apply, LinearMap.inr_apply, Function.comp_apply]
  simp_rw [hU, hU', hV, hV']
  simp_rw [coupledHessian_eq_secondFDeriv_workspace (N := N) (q := q)]
  simp only [map_smul, ContinuousLinearMap.smul_apply, smul_eq_mul]
  simp only [Finset.mul_sum]
  apply congrArg₂ (· + ·)
  · apply Finset.sum_congr rfl
    intro i _
    ring
  · apply Finset.sum_congr rfl
    intro i _
    ring

/-- Joint Gaussian IBP for the smart-path derivative, expressed as a canonical
configuration-basis covariance trace.

This should be proved with `UV`, `isGaussianHilbert_UV`, and
`gaussian_integration_by_parts_hilbert_cov_op`, following the existing proof of
`pressure_derivative_ibp_trace` and the scratch development in
`.tmp_gaussian_interp.lean`.
-/
lemma coupledFreeEnergy_time_ibp_trace_workspace
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    {t Λ : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    (∫ w,
        fderiv ℝ
          (fun H : EnergySpace N =>
            coupledFreeEnergyDet (N := N) (q := q) H Λ)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t w)
          (dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t w)
        ∂ℙ) =
      (1 / 2) *
        ∫ w,
          (∑ σ : Config N, ∑ τ : Config N,
            (mixedCovKernel N sk.ξ σ τ -
              referenceCovKernel N β σ τ) *
              coupledHessianDet
                (N := N) (q := q)
                (H_t
                  (N := N) (β := β) (h := h) (q := q)
                  (sk := sk) (sim := sim) t w)
                (Λ / 2)
                (std_basis N σ) (std_basis N τ))
          ∂ℙ := by
  /-
  Recommended structure:

  1. set `a = sqrt t`, `b = sqrt (1-t)`,
     `a' = 1/(2*sqrt t)`, `b' = -1/(2*sqrt (1-t))`;
  2. package `(sk.U, sim.V)` using `isGaussianHilbert_UV hIndep`;
  3. apply Gaussian IBP to the first variation of `coupledFreeEnergyDet`;
  4. rewrite the derivative using
     `fderiv_coupledFirstVariation_apply_workspace`;
  5. split the product-Hilbert eigenbasis trace into the U and V blocks;
  6. change each eigenbasis trace to the canonical `std_basis` trace;
  7. use `sk.cov_eq` and `sim.cov_eq`;
  8. simplify `a*a' = 1/2` and `b*b' = -1/2`.
  -/
  classical
  have ht0 : 0 < t := ht.1
  have ht1 : t < 1 := ht.2
  set a := Real.sqrt t
  set b := Real.sqrt (1 - t)
  set a' := 1 / (2 * Real.sqrt t)
  set b' := -1 / (2 * Real.sqrt (1 - t))
  let field := H_field (N := N) (h := h)
  let hg := isGaussianHilbert_UV (N := N) (β := β) (h := h) (q := q)
    (sk := sk) (sim := sim) hIndep
  let A := coupledIBPJointCLM (N := N) a b
  let B := coupledIBPJointCLM (N := N) a' b'
  let Φ := fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ
  have hFi_diff : ∀ i : hg.ι, ContDiff ℝ 1
      (fun x => fderiv ℝ Φ (A x + field) (B (hg.w i))) := by
    intro i
    have hΦ := contDiff_coupledFreeEnergyDet_beforeIBP (N := N) (q := q) Λ
    have hgrad : ContDiff ℝ 1 (fderiv ℝ Φ) :=
      hΦ.fderiv_right (m := (1 : WithTop ℕ∞)) (by
        change (↑(2 : ℕ∞) : WithTop ℕ∞) ≤ ↑(⊤ : ℕ∞)
        exact WithTop.coe_le_coe.mpr le_top)
    exact (hgrad.comp (A.contDiff.add contDiff_const)).clm_apply contDiff_const
  have hFi_growth : ∀ i : hg.ι,
      PhysLean.Probability.GaussianIBP.HasModerateGrowth
        (fun x => fderiv ℝ Φ (A x + field) (B (hg.w i))) := by
    intro i
    exact coupledFirstVariationModerateGrowth_workspace
      (N := N) (q := q) A field (B (hg.w i)) Λ
  have hmain := gaussian_ibp_gradient_linear
    (g := UV (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim))
    (hg := hg) A B field Φ hFi_diff hFi_growth
  have hraw :
      (∫ w, fderiv ℝ Φ
        (a • sk.U w + b • sim.V w + field)
        (a' • sk.U w + b' • sim.V w) ∂ℙ) =
      ∫ w, ∑ i : hg.ι, (hg.τ i : ℝ) *
        coupledHessianDet N q
          (a • sk.U w + b • sim.V w + field) (Λ / 2)
          (B (hg.w i)) (A (hg.w i)) ∂ℙ := by
    change (∫ w, fderiv ℝ Φ
      (A (UV (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) w) + field)
      (B (UV (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) w)) ∂ℙ) = _
    rw [hmain]
    apply MeasureTheory.integral_congr_ae
    filter_upwards with w
    apply Finset.sum_congr rfl
    intro i _
    rw [fderiv_coupledFirstVariation_affine_workspace
      (N := N) (q := q) A field (B (hg.w i))
      (UV (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) w)
      (hg.w i) Λ]
    simp [A, B, UV, coupledIBPJointCLM]
  have haa : a * a' = 1 / 2 := by
    dsimp [a, a']
    field_simp [ne_of_gt (Real.sqrt_pos.mpr ht0)]
  have hbb : b * b' = -(1 / 2) := by
    dsimp [b, b']
    field_simp [ne_of_gt (Real.sqrt_pos.mpr (sub_pos.mpr ht1))]
  have hH : H_t (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) t =
      fun w => a • sk.U w + b • sim.V w + field := by
    unfold H_t H_gauss
    simp [a, b, field]
  have hdH : dH_t (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) t =
      fun w => a' • sk.U w + b' • sim.V w := by
    unfold dH_t
    ext w
    simp [a', b']
    ring
  simp only [hH, hdH]
  rw [hraw]
  rw [← MeasureTheory.integral_const_mul]
  apply MeasureTheory.integral_congr_ae
  filter_upwards with w
  rw [coupledJointTrace_split_workspace
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
    hIndep a' b' a b (a • sk.U w + b • sim.V w + field) Λ]
  rw [mul_comm a' a, mul_comm b' b, haa, hbb]
  rw [coupledGaussianTrace_eq_stdBasis_workspace
    (N := N) (q := q) sk.U sk.hU
    (a • sk.U w + b • sim.V w + field) Λ]
  rw [coupledGaussianTrace_eq_stdBasis_workspace
    (N := N) (q := q) sim.V sim.hV
    (a • sk.U w + b • sim.V w + field) Λ]
  simp_rw [sk.cov_eq, sim.cov_eq]
  simp_rw [sub_mul]
  simp only [Finset.sum_sub_distrib]
  ring


end GeneralizedLatala
end SpinGlass
