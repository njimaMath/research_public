import GeneralizedLatala.Interpolation.Pressure

/-!
# Coupled smart path

The coupled free energy, coupling derivatives, time regularity, and the pre-IBP time derivative.

Main declarations:
- `coupledFreeEnergy_hasDerivAt_coupling_formula`
- `coupledFreeEnergy_hasDerivAt_time_before_ibp`
- `coupledHessianDet`

Dependencies:
- ordinary pressure interpolation and tilted observables

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

/-! ## Coupled smart path and its characteristic

The lemmas in this section deliberately use `HasDerivAt`.  Thus the differential identities
also carry the regularity needed by the later chain-rule and endpoint arguments.
-/

private lemma tiltedLog_hasDerivAt_coupling
    (H : EnergySpace N) (coupling : ℝ) :
    HasDerivAt
      (fun c => Real.log (tiltedReplicaPartitionDet (N := N) (q := q) H c))
      ((N : ℝ) * tiltedCenteredOverlapSqDet (N := N) (q := q) H coupling)
      coupling := by
  classical
  let A : ReplicaSpace N 2 → ℝ := fun σs =>
    (N : ℝ) * centeredOverlapSq N q σs
  let W : ReplicaSpace N 2 → ℝ := fun σs =>
    ∏ l, gibbs_pmf N H (σs l)
  have hterm (σs : ReplicaSpace N 2) :
      HasDerivAt (fun c : ℝ => Real.exp (c * A σs) * W σs)
        (A σs * Real.exp (coupling * A σs) * W σs) coupling := by
    have hi : HasDerivAt (fun c : ℝ => c * A σs) (A σs) coupling := by
      simpa using (hasDerivAt_id coupling).mul_const (A σs)
    simpa [Function.comp_def, mul_comm, mul_left_comm] using
      ((Real.hasDerivAt_exp _).comp coupling hi).mul_const (W σs)
  have hpart : HasDerivAt
      (fun c => tiltedReplicaPartitionDet (N := N) (q := q) H c)
      (∑ σs : ReplicaSpace N 2,
        A σs * Real.exp (coupling * A σs) * W σs) coupling := by
    simpa [tiltedReplicaPartitionDet, gibbs_average_n_det, A, W, mul_assoc] using
      (HasDerivAt.fun_sum (u := (Finset.univ : Finset (ReplicaSpace N 2)))
        (A := fun σs => fun c : ℝ => Real.exp (c * A σs) * W σs)
        (A' := fun σs => A σs * Real.exp (coupling * A σs) * W σs)
        (x := coupling) (fun σs _ => hterm σs))
  have hlog := (Real.hasDerivAt_log
    (ne_of_gt (tiltedReplicaPartitionDet_pos (N := N) (q := q) H coupling))).comp
      coupling hpart
  simpa [Function.comp_def, tiltedCenteredOverlapSqDet,
    tiltedReplicaPartitionDet, gibbs_average_n_det, A, W, div_eq_mul_inv,
    Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc] using hlog

lemma norm_tiltedLog_deriv_le
    (H : EnergySpace N) (coupling : ℝ) :
    ‖(N : ℝ) * tiltedCenteredOverlapSqDet (N := N) (q := q) H coupling‖ ≤
      ∑ σs : ReplicaSpace N 2, (N : ℝ) * centeredOverlapSq N q σs := by
  classical
  let A : ReplicaSpace N 2 → ℝ := fun σs =>
    (N : ℝ) * centeredOverlapSq N q σs
  let P : ReplicaSpace N 2 → ℝ := fun σs =>
    Real.exp (coupling * A σs) * ∏ l, gibbs_pmf N H (σs l)
  have hA (σs : ReplicaSpace N 2) : 0 ≤ A σs :=
    mul_nonneg (Nat.cast_nonneg N) (sq_nonneg _)
  have hP (σs : ReplicaSpace N 2) : 0 ≤ P σs :=
    mul_nonneg (Real.exp_nonneg _) (Finset.prod_nonneg fun l _ =>
      gibbs_pmf_nonneg (N := N) (H := H) (σ := σs l))
  have hsum : 0 < ∑ σs : ReplicaSpace N 2, P σs := by
    simpa [P, A, tiltedReplicaPartitionDet, gibbs_average_n_det,
      mul_comm, mul_left_comm, mul_assoc] using
      tiltedReplicaPartitionDet_pos (N := N) (q := q) H coupling
  have hnonneg : 0 ≤ (N : ℝ) *
      tiltedCenteredOverlapSqDet (N := N) (q := q) H coupling := by
    apply mul_nonneg (Nat.cast_nonneg N)
    unfold tiltedCenteredOverlapSqDet gibbs_average_n_det
    exact div_nonneg (Finset.sum_nonneg fun σs _ =>
      mul_nonneg (mul_nonneg (sq_nonneg _) (Real.exp_nonneg _))
        (Finset.prod_nonneg fun l _ =>
          gibbs_pmf_nonneg (N := N) (H := H) (σ := σs l)))
      (le_of_lt (tiltedReplicaPartitionDet_pos (N := N) (q := q) H coupling))
  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg]
  have hratio (σs : ReplicaSpace N 2) : P σs / (∑ τ, P τ) ≤ 1 :=
    (div_le_one hsum).2
      (Finset.single_le_sum (fun τ _ => hP τ) (Finset.mem_univ σs))
  have hle : (∑ σs : ReplicaSpace N 2, A σs * (P σs / ∑ τ, P τ)) ≤
      ∑ σs : ReplicaSpace N 2, A σs := by
    apply Finset.sum_le_sum
    intro σs _
    simpa using mul_le_mul_of_nonneg_left (hratio σs) (hA σs)
  simpa [tiltedCenteredOverlapSqDet, tiltedReplicaPartitionDet,
    gibbs_average_n_det, A, P, div_eq_mul_inv, Finset.mul_sum,
    mul_comm, mul_left_comm, mul_assoc] using hle

/-- Coupling derivative of the logarithmic quadratic moment.

The factor `N` comes from differentiating `exp (coupling * N * Q₁₂²)`.  The quotient defining
`tiltedCenteredOverlapSq` is legitimate by `tiltedReplicaPartitionDet_pos`. -/
lemma logQuadraticMoment_hasDerivAt_coupling_formula (t coupling : ℝ) :
    HasDerivAt
      (fun c => logQuadraticMoment
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t c)
      ((N : ℝ) * tiltedCenteredOverlapSq
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        t coupling) coupling := by
  classical
  let F : ℝ → Ω → ℝ := fun c ω => Real.log
    (tiltedReplicaPartitionDet (N := N) (q := q)
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t ω) c)
  let F' : ℝ → Ω → ℝ := fun c ω => (N : ℝ) *
    tiltedCenteredOverlapSqDet (N := N) (q := q)
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t ω) c
  let B : ℝ := ∑ σs : ReplicaSpace N 2, (N : ℝ) * centeredOverlapSq N q σs
  have hdiff (c : ℝ) (ω : Ω) : HasDerivAt (F · ω) (F' c ω) c := by
    simpa [F, F'] using tiltedLog_hasDerivAt_coupling (N := N) (q := q)
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t ω) c
  have hbound (c : ℝ) (ω : Ω) : ‖F' c ω‖ ≤ B := by
    simpa [F', B] using norm_tiltedLog_deriv_le (N := N) (q := q)
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t ω) c
  have hU_meas : Measurable sk.U := sk.hU.repr_measurable
  have hV_meas : Measurable sim.V := sim.hV.repr_measurable
  have hHt_meas : Measurable
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t) := by
    have h1 : Measurable (fun ω => (Real.sqrt t) • sk.U ω) :=
      hU_meas.const_smul (Real.sqrt t)
    have h2 : Measurable (fun ω => (Real.sqrt (1 - t)) • sim.V ω) :=
      hV_meas.const_smul (Real.sqrt (1 - t))
    have h3 : Measurable (fun _ω : Ω => H_field (N := N) (h := h)) := measurable_const
    exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) t
  have hpmf_meas (σ : Config N) : Measurable fun ω =>
      gibbs_pmf N
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω) σ :=
    (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σ)).continuous.measurable.comp hHt_meas
  have hpart_meas (c : ℝ) : Measurable fun ω =>
      tiltedReplicaPartitionDet (N := N) (q := q)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω) c := by
    unfold tiltedReplicaPartitionDet gibbs_average_n_det
    apply Finset.measurable_sum
    intro σs _
    apply measurable_const.mul
    apply Finset.measurable_prod
    intro l _
    exact hpmf_meas (σs l)
  have hnum_meas (c : ℝ) : Measurable fun ω =>
      gibbs_average_n_det (N := N) (n := 2)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω)
        (fun σs => centeredOverlapSq N q σs *
          Real.exp (c * (N : ℝ) * centeredOverlapSq N q σs)) := by
    unfold gibbs_average_n_det
    apply Finset.measurable_sum
    intro σs _
    apply measurable_const.mul
    apply Finset.measurable_prod
    intro l _
    exact hpmf_meas (σs l)
  have hF_meas (c : ℝ) : AEStronglyMeasurable (F c) ℙ := by
    exact ((hpart_meas c).log).aestronglyMeasurable
  have hF'_meas (c : ℝ) : AEStronglyMeasurable (F' c) ℙ := by
    apply Measurable.aestronglyMeasurable
    dsimp only [F', tiltedCenteredOverlapSqDet]
    exact measurable_const.mul ((hnum_meas c).div (hpart_meas c))
  have hzero (ω : Ω) : F 0 ω = 0 := by
    dsimp only [F]
    rw [show tiltedReplicaPartitionDet (N := N) (q := q)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω) 0 = 1 by
      unfold tiltedReplicaPartitionDet gibbs_average_n_det
      simp only [zero_mul, Real.exp_zero, one_mul]
      exact sum_prod_gibbs_pmf_eq_one (N := N) (n := 2) _]
    exact Real.log_one
  have hF_int : Integrable (F coupling) ℙ := by
    apply Integrable.of_bound (hF_meas coupling) (B * ‖coupling‖)
    filter_upwards with ω
    have hm := convex_univ.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun c => F c ω) (f' := fun c => F' c ω)
      (C := B) (x := 0) (y := coupling)
      (fun c _ => (hdiff c ω).hasDerivWithinAt)
      (fun c _ => hbound c ω) (by simp) (by simp)
    simpa [hzero] using hm
  have hmain :=
    (hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := (ℙ : Measure Ω)) (F := F) (F' := F') (x₀ := coupling)
      (bound := fun _ => B) (s := Set.univ) Filter.univ_mem
      (Filter.Eventually.of_forall hF_meas) hF_int (hF'_meas coupling)
      (ae_of_all _ fun ω c _ => hbound c ω) (integrable_const B)
      (ae_of_all _ fun ω c _ => hdiff c ω)).2
  rw [show logQuadraticMoment
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t =
      fun c => ∫ ω, F c ω ∂ℙ by
        funext c
        rfl]
  rw [tiltedCenteredOverlapSq, ← integral_const_mul]
  exact hmain

/-- Coupling derivative of the normalized coupled free energy.

The physical coupling is `Λ`, while the exponential coupling is `Λ / 2`.  The chain-rule
factor `1 / 2` and the normalization `1 / (2N)` turn the preceding derivative into `1 / 4`
times the annealed tilted overlap. -/
lemma coupledFreeEnergy_hasDerivAt_coupling_formula (t Λ : ℝ) :
    HasDerivAt
      (fun L => coupledFreeEnergy
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t L)
      ((1 / 4) * tiltedCenteredOverlapSq
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        t (Λ / 2)) Λ := by
  have hlog := logQuadraticMoment_hasDerivAt_coupling_formula
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t (Λ / 2)
  have hinner : HasDerivAt (fun L : ℝ => L / 2) (1 / 2) Λ := by
    simpa using (hasDerivAt_id Λ).div_const 2
  have hcomp : HasDerivAt
      (fun L => logQuadraticMoment
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t (L / 2))
      (((N : ℝ) * tiltedCenteredOverlapSq
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        t (Λ / 2)) * (1 / 2)) Λ :=
    by
      change HasDerivAt
        ((fun c => logQuadraticMoment
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t c) ∘
          fun L : ℝ => L / 2) _ Λ
      exact hlog.comp Λ hinner
  have hscaled := hcomp.const_mul (1 / (2 * (N : ℝ)))
  have hcoeff :
      (1 / (2 * (N : ℝ))) *
          (((N : ℝ) * tiltedCenteredOverlapSq
            (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
            t (Λ / 2)) * (1 / 2)) =
        (1 / 4) * tiltedCenteredOverlapSq
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
          t (Λ / 2) := by
    have hN : (N : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne N
    field_simp [hN]
    ring
  rw [hcoeff] at hscaled
  simpa only [coupledFreeEnergy, coupledExcess, physicalLogQuadraticMoment] using
    hscaled.const_add
      (interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t)

/-- Evaluation of a Hamiltonian direction on two replicas. -/
private noncomputable def pairEval_beforeIBP
    (u : EnergySpace N) : ReplicaFun N 2 :=
  fun σs => u (σs 0) + u (σs 1)

/-- Expectation under the normalized quadratically tilted two-replica law. -/
private noncomputable def tiltedReplicaAverageDet_beforeIBP
    (H : EnergySpace N) (coupling : ℝ)
    (f : ReplicaFun N 2) : ℝ :=
  gibbs_average_n_det (N := N) (n := 2) H
      (fun σs =>
        f σs *
          Real.exp
            (coupling * (N : ℝ) * centeredOverlapSq N q σs)) /
    tiltedReplicaPartitionDet (N := N) (q := q) H coupling

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma fderiv_tiltedReplicaPartitionDet_apply_beforeIBP
    (H u : EnergySpace N) (coupling : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          tiltedReplicaPartitionDet (N := N) (q := q) K coupling)
        H u =
      2 * (∑ τ : Config N, gibbs_pmf N H τ * u τ) *
          tiltedReplicaPartitionDet (N := N) (q := q) H coupling -
        gibbs_average_n_det (N := N) (n := 2) H
          (fun σs =>
            pairEval_beforeIBP (N := N) u σs *
              Real.exp
                (coupling * (N : ℝ) *
                  centeredOverlapSq N q σs)) := by
  unfold gibbs_average_n_det pairEval_beforeIBP
  unfold gibbs_pmf tiltedReplicaPartitionDet
  rw [fderiv_gibbs_average_n_det_apply]
  unfold gibbs_average_n_det gibbs_pmf
  simp +decide [
    Fin.sum_univ_two,
    mul_sub,
    sub_mul,
    mul_assoc,
    mul_comm,
    mul_left_comm,
    Finset.mul_sum _ _ _,
    Finset.sum_mul
  ]

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma fderiv_coupledFreeEnergyDet_apply_beforeIBP
    (H u : EnergySpace N) (Λ : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          coupledFreeEnergyDet (N := N) (q := q) K Λ)
        H u =
      -(1 / (2 * (N : ℝ))) *
        tiltedReplicaAverageDet_beforeIBP
          (N := N) (q := q) H (Λ / 2)
          (pairEval_beforeIBP (N := N) u) := by
  erw [fderiv_add] <;>
    norm_num [fderiv_free_energy_density_apply]
  · erw [fderiv_mul, fderiv.log] <;>
      norm_num [fderiv_tiltedReplicaPartitionDet_apply_beforeIBP]
    · unfold tiltedReplicaAverageDet_beforeIBP
      ring
      rw [
        mul_inv_cancel_right₀
          (ne_of_gt
            (tiltedReplicaPartitionDet_pos
              (N := N) (q := q) H (Λ * (1 / 2))))
      ]
      ring
    · have hdiff :
          ∀ σs : ReplicaSpace N 2,
            DifferentiableAt ℝ
              (fun K : EnergySpace N =>
                Real.exp
                    (Λ / 2 * (N : ℝ) *
                      centeredOverlapSq N q σs) *
                  ∏ l,
                    Real.exp (-K.ofLp (σs l)) / Z N K)
              H := by
        intro σs
        have hpmf :
            ∀ l : Fin 2,
              DifferentiableAt ℝ
                (fun K : EnergySpace N =>
                  Real.exp (-K.ofLp (σs l)) / Z N K)
                H :=
          fun l => differentiableAt_gibbs_pmf N H (σs l)
        fun_prop
      exact DifferentiableAt.fun_sum fun i _ => hdiff i
    · exact
        ne_of_gt
          (tiltedReplicaPartitionDet_pos
            (N := N) (q := q) H (Λ / 2))
    · refine DifferentiableAt.log ?_ ?_
      · unfold tiltedReplicaPartitionDet gibbs_average_n_det
        unfold gibbs_pmf
        norm_num [
          Real.exp_ne_zero,
          Finset.prod_eq_zero_iff,
          Real.differentiableAt_exp,
          differentiableAt_pi
        ]
        have hdiff :
            ∀ x : ReplicaSpace N 2,
              DifferentiableAt ℝ
                (fun K : EnergySpace N =>
                  Real.exp (-K.ofLp (x 0)) *
                      Real.exp (-K.ofLp (x 1)) /
                    Z N K ^ 2)
                H := by
          intro x
          apply_rules [
            DifferentiableAt.div,
            DifferentiableAt.mul,
            DifferentiableAt.exp,
            differentiableAt_id,
            differentiableAt_const
          ]
          · fun_prop
          · fun_prop
          · apply_rules [
              DifferentiableAt.inv,
              DifferentiableAt.pow,
              differentiableAt_id
            ]
            · unfold Z
              fun_prop
            · exact
                ne_of_gt
                  (sq_pos_of_pos
                    (Z_pos (N := N) H))
        fun_prop
      · exact
          ne_of_gt
            (tiltedReplicaPartitionDet_pos
              (N := N) (q := q) H (Λ / 2))
  · apply_rules [
      DifferentiableAt.mul,
      DifferentiableAt.log
    ] <;> norm_num
    · unfold Z
      fun_prop
    · exact
        ne_of_gt
          (Finset.sum_pos
            (fun _ _ => Real.exp_pos _)
            Finset.univ_nonempty)
  · apply_rules [
      DifferentiableAt.mul,
      DifferentiableAt.log
    ] <;>
      norm_num [tiltedReplicaPartitionDet_pos]
    · unfold tiltedReplicaPartitionDet
      unfold gibbs_average_n_det
      norm_num [gibbs_average_n, gibbs_pmf]
      have hdiff :
          DifferentiableAt ℝ
            (fun x : EnergySpace N =>
              ∑ σ : Config N, Real.exp (-x σ))
            H := by
        fun_prop
      simp_all +decide [
        ← mul_div_assoc,
        ← Finset.sum_div _ _ _
      ]
      refine DifferentiableAt.mul ?_ ?_
      · fun_prop
      · exact
          DifferentiableAt.inv
            (hdiff.pow 2)
            (ne_of_gt
              (sq_pos_of_pos
                (Finset.sum_pos
                  (fun _ _ => Real.exp_pos _)
                  Finset.univ_nonempty)))
    · exact
        ne_of_gt
          (tiltedReplicaPartitionDet_pos
            (N := N) (q := q) H (Λ / 2))

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma abs_pairEval_beforeIBP_le
    (u : EnergySpace N) (σs : ReplicaSpace N 2) :
    |pairEval_beforeIBP (N := N) u σs| ≤ 2 * ‖u‖ := by
  unfold pairEval_beforeIBP
  calc
    |u (σs 0) + u (σs 1)|
        ≤ |u (σs 0)| + |u (σs 1)| := abs_add_le _ _
    _ ≤ ‖u‖ + ‖u‖ := by
      exact add_le_add
        (abs_apply_le_norm (N := N) u (σs 0))
        (abs_apply_le_norm (N := N) u (σs 1))
    _ = 2 * ‖u‖ := by ring

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma abs_tiltedReplicaAverageDet_pairEval_beforeIBP_le
    (H : EnergySpace N) (coupling : ℝ) (u : EnergySpace N) :
    |tiltedReplicaAverageDet_beforeIBP
        (N := N) (q := q) H coupling
        (pairEval_beforeIBP (N := N) u)|
      ≤ 2 * ‖u‖ := by
  classical
  let W : ReplicaSpace N 2 → ℝ := fun σs =>
    Real.exp
        (coupling * (N : ℝ) *
          centeredOverlapSq N q σs) *
      ∏ l, gibbs_pmf N H (σs l)

  have hW (σs : ReplicaSpace N 2) : 0 ≤ W σs := by
    exact mul_nonneg
      (Real.exp_nonneg _)
      (Finset.prod_nonneg fun l _ =>
        gibbs_pmf_nonneg
          (N := N) (H := H) (σ := σs l))

  have hsum :
      0 < ∑ σs : ReplicaSpace N 2, W σs := by
    simpa [
      W,
      tiltedReplicaPartitionDet,
      gibbs_average_n_det,
      mul_assoc
    ] using
      tiltedReplicaPartitionDet_pos
        (N := N) (q := q) H coupling

  have hform :
      tiltedReplicaAverageDet_beforeIBP
          (N := N) (q := q) H coupling
          (pairEval_beforeIBP (N := N) u) =
        (∑ σs : ReplicaSpace N 2,
            pairEval_beforeIBP (N := N) u σs * W σs) /
          (∑ σs : ReplicaSpace N 2, W σs) := by
    simp [
      tiltedReplicaAverageDet_beforeIBP,
      tiltedReplicaPartitionDet,
      gibbs_average_n_det,
      W,
      mul_assoc
    ]

  rw [hform, abs_div, abs_of_pos hsum]
  apply (div_le_iff₀ hsum).2
  calc
    |∑ σs : ReplicaSpace N 2,
        pairEval_beforeIBP (N := N) u σs * W σs|
        ≤
      ∑ σs : ReplicaSpace N 2,
        |pairEval_beforeIBP (N := N) u σs * W σs| := by
          simpa using
            (Finset.abs_sum_le_sum_abs
              (s := Finset.univ)
              (f := fun σs : ReplicaSpace N 2 =>
                pairEval_beforeIBP (N := N) u σs * W σs))
    _ =
      ∑ σs : ReplicaSpace N 2,
        |pairEval_beforeIBP (N := N) u σs| * W σs := by
          apply Finset.sum_congr rfl
          intro σs _
          rw [abs_mul, abs_of_nonneg (hW σs)]
    _ ≤
      ∑ σs : ReplicaSpace N 2,
        (2 * ‖u‖) * W σs := by
          apply Finset.sum_le_sum
          intro σs _
          exact mul_le_mul_of_nonneg_right
            (abs_pairEval_beforeIBP_le
              (N := N) u σs)
            (hW σs)
    _ =
      (2 * ‖u‖) *
        ∑ σs : ReplicaSpace N 2, W σs := by
          rw [Finset.mul_sum]

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma opNorm_fderiv_coupledFreeEnergyDet_le_beforeIBP
    (H : EnergySpace N) (Λ : ℝ) :
    ‖fderiv ℝ
        (fun K : EnergySpace N =>
          coupledFreeEnergyDet (N := N) (q := q) K Λ)
        H‖
      ≤ 1 / (N : ℝ) := by
  have hNr : 0 < (N : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)

  have hcoef :
      0 ≤ 1 / (2 * (N : ℝ)) := by
    positivity

  refine ContinuousLinearMap.opNorm_le_bound
    _
    (by positivity)
    ?_

  intro u
  rw [
    fderiv_coupledFreeEnergyDet_apply_beforeIBP
      (N := N) (q := q)
  ]

  have havg :=
    abs_tiltedReplicaAverageDet_pairEval_beforeIBP_le
      (N := N) (q := q)
      H (Λ / 2) u

  calc
    ‖-(1 / (2 * (N : ℝ))) *
        tiltedReplicaAverageDet_beforeIBP
          (N := N) (q := q) H (Λ / 2)
          (pairEval_beforeIBP (N := N) u)‖
        =
      (1 / (2 * (N : ℝ))) *
        |tiltedReplicaAverageDet_beforeIBP
          (N := N) (q := q) H (Λ / 2)
          (pairEval_beforeIBP (N := N) u)| := by
            simp [
              Real.norm_eq_abs,
              abs_mul,
              abs_of_nonneg hcoef
            ]
    _ ≤
      (1 / (2 * (N : ℝ))) * (2 * ‖u‖) :=
        mul_le_mul_of_nonneg_left havg hcoef
    _ =
      (1 / (N : ℝ)) * ‖u‖ := by
        field_simp [ne_of_gt hNr]

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma contDiff_tiltedReplicaPartitionDet_beforeIBP
    (coupling : ℝ) :
    ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
      (fun H : EnergySpace N =>
        tiltedReplicaPartitionDet
          (N := N) (q := q) H coupling) := by
  unfold tiltedReplicaPartitionDet gibbs_average_n_det
  apply ContDiff.sum
  intro σs _
  apply ContDiff.mul
  · exact contDiff_const
  · have hpmf :
        ∀ l : Fin 2,
          ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
            (fun H : EnergySpace N =>
              gibbs_pmf N H (σs l)) :=
      fun l =>
        contDiff_gibbs_pmf
          (N := N) (σ := σs l)
    fun_prop

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma contDiff_coupledFreeEnergyDet_beforeIBP
    (Λ : ℝ) :
    ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
      (fun H : EnergySpace N =>
        coupledFreeEnergyDet
          (N := N) (q := q) H Λ) := by
  have hpart :=
    contDiff_tiltedReplicaPartitionDet_beforeIBP
      (N := N) (q := q) (Λ / 2)

  have hlog :
      ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
        (fun H : EnergySpace N =>
          Real.log
            (tiltedReplicaPartitionDet
              (N := N) (q := q) H (Λ / 2))) :=
    hpart.log fun H =>
      ne_of_gt
        (tiltedReplicaPartitionDet_pos
          (N := N) (q := q) H (Λ / 2))

  have hscaled :
      ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
        (fun H : EnergySpace N =>
          (1 / (2 * (N : ℝ))) *
            Real.log
              (tiltedReplicaPartitionDet
                (N := N) (q := q) H (Λ / 2))) := by
    simpa [smul_eq_mul] using
      (ContDiff.const_smul
        (𝕜 := ℝ) (n := (↑(⊤ : ℕ∞) : WithTop ℕ∞)) (R := ℝ)
        (c := 1 / (2 * (N : ℝ))) hlog)

  simpa [coupledFreeEnergyDet] using
    (contDiff_free_energy_density (N := N)).add hscaled

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma tiltedLog_hasDerivAt_coupling_beforeIBP
    (H : EnergySpace N) (coupling : ℝ) :
    HasDerivAt
      (fun c =>
        Real.log
          (tiltedReplicaPartitionDet
            (N := N) (q := q) H c))
      ((N : ℝ) *
        tiltedCenteredOverlapSqDet
          (N := N) (q := q) H coupling)
      coupling := by
  classical
  let A : ReplicaSpace N 2 → ℝ := fun σs =>
    (N : ℝ) * centeredOverlapSq N q σs
  let W : ReplicaSpace N 2 → ℝ := fun σs =>
    ∏ l, gibbs_pmf N H (σs l)

  have hterm (σs : ReplicaSpace N 2) :
      HasDerivAt
        (fun c : ℝ =>
          Real.exp (c * A σs) * W σs)
        (A σs *
          Real.exp (coupling * A σs) *
          W σs)
        coupling := by
    have hi :
        HasDerivAt
          (fun c : ℝ => c * A σs)
          (A σs)
          coupling := by
      simpa using
        (hasDerivAt_id coupling).mul_const (A σs)
    simpa [
      Function.comp_def,
      mul_comm,
      mul_left_comm
    ] using
      ((Real.hasDerivAt_exp _).comp coupling hi).mul_const
        (W σs)

  have hpart :
      HasDerivAt
        (fun c =>
          tiltedReplicaPartitionDet
            (N := N) (q := q) H c)
        (∑ σs : ReplicaSpace N 2,
          A σs *
            Real.exp (coupling * A σs) *
            W σs)
        coupling := by
    simpa [
      tiltedReplicaPartitionDet,
      gibbs_average_n_det,
      A,
      W,
      mul_assoc
    ] using
      (HasDerivAt.fun_sum
        (u := Finset.univ)
        (A := fun σs =>
          fun c : ℝ =>
            Real.exp (c * A σs) * W σs)
        (A' := fun σs =>
          A σs *
            Real.exp (coupling * A σs) *
            W σs)
        (x := coupling)
        (fun σs _ => hterm σs))

  have hlog :=
    (Real.hasDerivAt_log
      (ne_of_gt
        (tiltedReplicaPartitionDet_pos
          (N := N) (q := q) H coupling))).comp
      coupling hpart

  simpa [
    Function.comp_def,
    tiltedCenteredOverlapSqDet,
    tiltedReplicaPartitionDet,
    gibbs_average_n_det,
    A,
    W,
    div_eq_mul_inv,
    Finset.mul_sum,
    mul_comm,
    mul_left_comm,
    mul_assoc
  ] using hlog

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
private lemma norm_tiltedLog_deriv_le_beforeIBP
    (H : EnergySpace N) (coupling : ℝ) :
    ‖(N : ℝ) *
        tiltedCenteredOverlapSqDet
          (N := N) (q := q) H coupling‖
      ≤
    ∑ σs : ReplicaSpace N 2,
      (N : ℝ) * centeredOverlapSq N q σs := by
  classical
  let A : ReplicaSpace N 2 → ℝ := fun σs =>
    (N : ℝ) * centeredOverlapSq N q σs
  let P : ReplicaSpace N 2 → ℝ := fun σs =>
    Real.exp (coupling * A σs) *
      ∏ l, gibbs_pmf N H (σs l)

  have hA (σs : ReplicaSpace N 2) :
      0 ≤ A σs :=
    mul_nonneg
      (Nat.cast_nonneg N)
      (sq_nonneg _)

  have hP (σs : ReplicaSpace N 2) :
      0 ≤ P σs :=
    mul_nonneg
      (Real.exp_nonneg _)
      (Finset.prod_nonneg fun l _ =>
        gibbs_pmf_nonneg
          (N := N) (H := H) (σ := σs l))

  have hsum :
      0 < ∑ σs : ReplicaSpace N 2, P σs := by
    simpa [
      P,
      A,
      tiltedReplicaPartitionDet,
      gibbs_average_n_det,
      mul_comm,
      mul_left_comm,
      mul_assoc
    ] using
      tiltedReplicaPartitionDet_pos
        (N := N) (q := q) H coupling

  have hnonneg :
      0 ≤
        (N : ℝ) *
          tiltedCenteredOverlapSqDet
            (N := N) (q := q) H coupling := by
    apply mul_nonneg (Nat.cast_nonneg N)
    unfold tiltedCenteredOverlapSqDet gibbs_average_n_det
    exact div_nonneg
      (Finset.sum_nonneg fun σs _ =>
        mul_nonneg
          (mul_nonneg
            (sq_nonneg _)
            (Real.exp_nonneg _))
          (Finset.prod_nonneg fun l _ =>
            gibbs_pmf_nonneg
              (N := N) (H := H) (σ := σs l)))
      (le_of_lt
        (tiltedReplicaPartitionDet_pos
          (N := N) (q := q) H coupling))

  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg]

  have hratio (σs : ReplicaSpace N 2) :
      P σs / (∑ τ, P τ) ≤ 1 :=
    (div_le_one hsum).2
      (Finset.single_le_sum
        (fun τ _ => hP τ)
        (Finset.mem_univ σs))

  have hle :
      (∑ σs : ReplicaSpace N 2,
        A σs * (P σs / ∑ τ, P τ))
        ≤
      ∑ σs : ReplicaSpace N 2, A σs := by
    apply Finset.sum_le_sum
    intro σs _
    simpa using
      mul_le_mul_of_nonneg_left
        (hratio σs)
        (hA σs)

  simpa [
    tiltedCenteredOverlapSqDet,
    tiltedReplicaPartitionDet,
    gibbs_average_n_det,
    A,
    P,
    div_eq_mul_inv,
    Finset.mul_sum,
    mul_comm,
    mul_left_comm,
    mul_assoc
  ] using hle

private lemma measurable_H_t_beforeIBP (s : ℝ) :
    Measurable
      (H_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s) := by
  have hU :=
    sk.hU.repr_measurable.const_smul
      (Real.sqrt s)
  have hV :=
    sim.hV.repr_measurable.const_smul
      (Real.sqrt (1 - s))
  exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
    (sk := sk) (sim := sim) s

private lemma measurable_dH_t_beforeIBP (s : ℝ) :
    Measurable
      (fun w =>
        dH_t
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w) := by
  have hU :=
    sk.hU.repr_measurable.const_smul
      (1 / (2 * Real.sqrt s))
  have hV :=
    sim.hV.repr_measurable.const_smul
      (1 / (2 * Real.sqrt (1 - s)))
  exact measurable_dH_t_updated (N := N) (β := β) (h := h) (q := q)
    (sk := sk) (sim := sim) s

private lemma integrable_freeEnergy_H_t_beforeIBP
    (s : ℝ) :
    Integrable
      (fun w =>
        free_energy_density (N := N)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) s w))
      ℙ := by
  let C : ℝ :=
    (SpinGlass.hasModerateGrowth_free_energy_density N).C
  let aU : ℝ := |Real.sqrt s|
  let aV : ℝ := |Real.sqrt (1 - s)|

  let boundFun : Ω → ℝ := fun w =>
    C *
      (1 +
        aU * ‖sk.U w‖ +
        aV * ‖sim.V w‖ +
        ‖H_field (N := N) (h := h)‖)

  have hmeas :
      AEStronglyMeasurable
        (fun w =>
          free_energy_density (N := N)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) s w))
        ℙ :=
    ((contDiff_free_energy_density (N := N)).continuous.measurable.comp
      (measurable_H_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s)).aestronglyMeasurable

  have hU_int :
      Integrable (fun w => ‖sk.U w‖) ℙ :=
    PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian
      (g := sk.U) sk.hU

  have hV_int :
      Integrable (fun w => ‖sim.V w‖) ℙ :=
    PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian
      (g := sim.V) sim.hV

  have hbound_int :
      Integrable boundFun ℙ := by
    dsimp only [boundFun]
    apply Integrable.const_mul
    exact
      ((((integrable_const (1 : ℝ)).add
          (hU_int.const_mul aU)).add
          (hV_int.const_mul aV)).add
          (integrable_const _))

  refine hbound_int.mono' hmeas ?_
  filter_upwards with w

  have hnorm :
      ‖H_t
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w‖
        ≤
      aU * ‖sk.U w‖ +
        aV * ‖sim.V w‖ +
        ‖H_field (N := N) (h := h)‖ := by
    calc
      ‖H_t
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w‖
          ≤
          ‖(Real.sqrt s) • sk.U w‖ +
          ‖(Real.sqrt (1 - s)) • sim.V w‖ +
          ‖H_field (N := N) (h := h)‖ := by
            simp only [H_t, H_gauss]
            exact (norm_add_le
                ((Real.sqrt s) • sk.U w +
                  (Real.sqrt (1 - s)) • sim.V w)
                (H_field (N := N) (h := h))).trans
                (by
                  gcongr
                  exact norm_add_le
                    ((Real.sqrt s) • sk.U w)
                    ((Real.sqrt (1 - s)) • sim.V w))
      _ =
        aU * ‖sk.U w‖ +
          aV * ‖sim.V w‖ +
          ‖H_field (N := N) (h := h)‖ := by
            simp [
              aU,
              aV,
              norm_smul,
              Real.norm_eq_abs
            ]

  have hgrowth :=
    (SpinGlass.hasModerateGrowth_free_energy_density N).F_bound
      (H_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s w)

  have hm :
      (SpinGlass.hasModerateGrowth_free_energy_density N).m = 1 := by
    rfl

  rw [hm, pow_one] at hgrowth
  rw [Real.norm_eq_abs]

  have hinside :
      1 +
          ‖H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) s w‖
        ≤
      1 +
        aU * ‖sk.U w‖ +
        aV * ‖sim.V w‖ +
        ‖H_field (N := N) (h := h)‖ := by
    linarith

  have hmul :=
    mul_le_mul_of_nonneg_left hinside
      (le_of_lt
        (SpinGlass.hasModerateGrowth_free_energy_density N).Cpos)

  exact hgrowth.trans
    (by simpa only [C, boundFun] using hmul)

private lemma integrable_tiltedLog_H_t_beforeIBP
    (s coupling : ℝ) :
    Integrable
      (fun w =>
        Real.log
          (tiltedReplicaPartitionDet
            (N := N) (q := q)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) s w)
            coupling))
      ℙ := by
  let B : ℝ :=
    ∑ σs : ReplicaSpace N 2,
      (N : ℝ) * centeredOverlapSq N q σs

  have hlog_cont :
      Continuous
        (fun H : EnergySpace N =>
          Real.log
            (tiltedReplicaPartitionDet
              (N := N) (q := q) H coupling)) := by
    exact
      ((contDiff_tiltedReplicaPartitionDet_beforeIBP
          (N := N) (q := q) coupling).log
        (fun H =>
          ne_of_gt
            (tiltedReplicaPartitionDet_pos
              (N := N) (q := q) H coupling))).continuous

  have hmeas :
      AEStronglyMeasurable
        (fun w =>
          Real.log
            (tiltedReplicaPartitionDet
              (N := N) (q := q)
              (H_t
                (N := N) (β := β) (h := h) (q := q)
                (sk := sk) (sim := sim) s w)
              coupling))
        ℙ :=
    (hlog_cont.measurable.comp
      (measurable_H_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s)).aestronglyMeasurable

  apply Integrable.of_bound hmeas (B * ‖coupling‖)
  filter_upwards with w

  let H :=
    H_t
      (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) s w

  have hzero :
      Real.log
          (tiltedReplicaPartitionDet
            (N := N) (q := q) H 0)
        = 0 := by
    rw [show
      tiltedReplicaPartitionDet
          (N := N) (q := q) H 0
        = 1 by
      unfold tiltedReplicaPartitionDet gibbs_average_n_det
      simp only [zero_mul, Real.exp_zero, one_mul]
      exact
        sum_prod_gibbs_pmf_eq_one
          (N := N) (n := 2) H]
    exact Real.log_one

  have hm :=
    convex_univ.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun c =>
        Real.log
          (tiltedReplicaPartitionDet
            (N := N) (q := q) H c))
      (f' := fun c =>
        (N : ℝ) *
          tiltedCenteredOverlapSqDet
            (N := N) (q := q) H c)
      (C := B)
      (x := 0)
      (y := coupling)
      (fun c _ =>
        (tiltedLog_hasDerivAt_coupling_beforeIBP
          (N := N) (q := q) H c).hasDerivWithinAt)
      (fun c _ =>
        norm_tiltedLog_deriv_le_beforeIBP
          (N := N) (q := q) H c)
      (by simp)
      (by simp)

  simpa [H, hzero] using hm

private lemma coupledFreeEnergy_eq_integral_det_beforeIBP
    (s Λ : ℝ) :
    coupledFreeEnergy
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s Λ
      =
    ∫ w,
      coupledFreeEnergyDet
        (N := N) (q := q)
        (H_t
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w)
        Λ
      ∂ℙ := by
  have hfree :=
    integrable_freeEnergy_H_t_beforeIBP
      (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) s

  have hlog :=
    integrable_tiltedLog_H_t_beforeIBP
      (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) s (Λ / 2)

  change
    (∫ w,
      free_energy_density (N := N)
        (H_t
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w)
      ∂ℙ) +
      (1 / (2 * (N : ℝ))) *
        (∫ w,
          Real.log
            (tiltedReplicaPartitionDet
              (N := N) (q := q)
              (H_t
                (N := N) (β := β) (h := h) (q := q)
                (sk := sk) (sim := sim) s w)
              (Λ / 2))
          ∂ℙ)
      =
    ∫ w,
      free_energy_density (N := N)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) s w) +
        (1 / (2 * (N : ℝ))) *
          Real.log
            (tiltedReplicaPartitionDet
              (N := N) (q := q)
              (H_t
                (N := N) (β := β) (h := h) (q := q)
                (sk := sk) (sim := sim) s w)
              (Λ / 2))
      ∂ℙ

  rw [← integral_const_mul]
  rw [← integral_add hfree (hlog.const_mul _)]

private lemma integral_coupledFreeEnergyDet_hasDerivAt_beforeIBP
    {t Λ : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt
      (fun s =>
        ∫ w,
          coupledFreeEnergyDet
            (N := N) (q := q)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) s w)
            Λ
          ∂ℙ)
      (∫ w,
        fderiv ℝ
          (fun H : EnergySpace N =>
            coupledFreeEnergyDet
              (N := N) (q := q) H Λ)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t w)
          (dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t w)
        ∂ℙ)
      t := by
  classical

  have ht0 : 0 < t := ht.1
  have ht1 : t < 1 := ht.2
  have h1t0 : 0 < 1 - t := by
    linarith

  let ε : ℝ := min t (1 - t) / 2

  have hε_pos : 0 < ε := by
    have hmin : 0 < min t (1 - t) :=
      lt_min ht0 h1t0
    dsimp only [ε]
    linarith

  have hball_Ioo :
      ∀ x ∈ Metric.ball t ε,
        x ∈ Set.Ioo (0 : ℝ) 1 := by
    intro x hx

    have hx' : |x - t| < ε := by
      simpa [
        Metric.mem_ball,
        Real.dist_eq,
        abs_sub_comm
      ] using hx

    have hxleft : x - t < ε :=
      (abs_sub_lt_iff.1 hx').1
    have hxright : t - x < ε :=
      (abs_sub_lt_iff.1 hx').2

    have hε_le_t : ε ≤ t / 2 := by
      have hmin : min t (1 - t) ≤ t :=
        min_le_left _ _
      dsimp only [ε]
      linarith

    have hε_le_1t : ε ≤ (1 - t) / 2 := by
      have hmin : min t (1 - t) ≤ 1 - t :=
        min_le_right _ _
      dsimp only [ε]
      linarith

    constructor
    · have hxlower : t - ε < x := by
        linarith
      have : 0 < t - ε := by
        linarith
      exact lt_trans this hxlower
    · have hxupper : x < t + ε := by
        linarith
      have : t + ε < 1 := by
        linarith
      exact lt_trans hxupper this

  let F : ℝ → Ω → ℝ := fun s w =>
    coupledFreeEnergyDet
      (N := N) (q := q)
      (H_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s w)
      Λ

  let F' : ℝ → Ω → ℝ := fun s w =>
    fderiv ℝ
      (fun H : EnergySpace N =>
        coupledFreeEnergyDet
          (N := N) (q := q) H Λ)
      (H_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s w)
      (dH_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s w)

  have hΦ :
      ContDiff ℝ (↑(⊤ : ℕ∞) : WithTop ℕ∞)
        (fun H : EnergySpace N =>
          coupledFreeEnergyDet
            (N := N) (q := q) H Λ) :=
    contDiff_coupledFreeEnergyDet_beforeIBP
      (N := N) (q := q) Λ

  have hF_meas :
      ∀ᶠ s in nhds t,
        AEStronglyMeasurable (F s) ℙ := by
    refine Filter.Eventually.of_forall ?_
    intro s
    exact
      (hΦ.continuous.measurable.comp
        (measurable_H_t_beforeIBP
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s)).aestronglyMeasurable

  have hF_int :
      Integrable (F t) ℙ := by
    have hfree :=
      integrable_freeEnergy_H_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t

    have hlog :=
      integrable_tiltedLog_H_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t (Λ / 2)

    change Integrable (fun w =>
      free_energy_density (N := N)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) +
        (1 / (2 * (N : ℝ))) * Real.log
          (tiltedReplicaPartitionDet (N := N) (q := q)
            (H_t (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) t w) (Λ / 2))) ℙ
    exact (hfree.add (hlog.const_mul (1 / (2 * (N : ℝ))))).congr
      (ae_of_all _ fun w => by rfl)

  let Cf : ℝ := 1 / (N : ℝ)
  let cU : ℝ := 1 / (2 * Real.sqrt (t / 2))
  let cV : ℝ :=
    1 / (2 * Real.sqrt ((1 - t) / 2))

  let bound : Ω → ℝ := fun w =>
    Cf *
      (cU * ‖sk.U w‖ +
        cV * ‖sim.V w‖)

  have hCf_nonneg : 0 ≤ Cf := by
    positivity
  have hcU_nonneg : 0 ≤ cU := by
    positivity
  have hcV_nonneg : 0 ≤ cV := by
    positivity

  have hbound_int :
      Integrable bound ℙ := by
    have hU_int :
        Integrable (fun w => ‖sk.U w‖) ℙ :=
      PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian
        (g := sk.U) sk.hU

    have hV_int :
        Integrable (fun w => ‖sim.V w‖) ℙ :=
      PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian
        (g := sim.V) sim.hV

    have h1 :
        Integrable
          (fun w => cU * ‖sk.U w‖)
          ℙ :=
      hU_int.const_mul cU

    have h2 :
        Integrable
          (fun w => cV * ‖sim.V w‖)
          ℙ :=
      hV_int.const_mul cV

    have hsum :
        Integrable
          (fun w =>
            cU * ‖sk.U w‖ +
              cV * ‖sim.V w‖)
          ℙ :=
      h1.add h2

    simpa [bound, Cf, mul_add, mul_assoc] using
      hsum.const_mul Cf

  have hF'_meas :
      AEStronglyMeasurable (F' t) ℙ := by
    have hHt_meas :=
      measurable_H_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t

    have hdHt_meas :=
      measurable_dH_t_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t

    have hfderiv_cont :
        Continuous
          (fun p : EnergySpace N × EnergySpace N =>
            fderiv ℝ
              (fun H : EnergySpace N =>
                coupledFreeEnergyDet
                  (N := N) (q := q) H Λ)
              p.1 p.2) := by
      have hcont :
          Continuous
            (fun H : EnergySpace N =>
              fderiv ℝ
                (fun K : EnergySpace N =>
                  coupledFreeEnergyDet
                    (N := N) (q := q) K Λ)
                H) :=
        hΦ.continuous_fderiv (by simp)

      exact
        ((hcont.comp continuous_fst).clm_apply continuous_snd)

    have hpair :
        Measurable
          (fun w =>
            (H_t
                (N := N) (β := β) (h := h) (q := q)
                (sk := sk) (sim := sim) t w,
              dH_t
                (N := N) (β := β) (h := h) (q := q)
                (sk := sk) (sim := sim) t w)) :=
      hHt_meas.prodMk hdHt_meas

    exact
      (hfderiv_cont.measurable.comp hpair).aestronglyMeasurable

  have h_bound :
      ∀ᵐ w ∂ℙ,
        ∀ x ∈ Metric.ball t ε,
          ‖F' x w‖ ≤ bound w := by
    refine ae_of_all _ ?_
    intro w x hx

    have hxIoo :
        x ∈ Set.Ioo (0 : ℝ) 1 :=
      hball_Ioo x hx

    have hCoeffU :
        |1 / (2 * Real.sqrt x)| ≤ cU := by
      have hx_lower : t / 2 ≤ x := by
        have hx' : |x - t| < ε := by
          simpa [
            Metric.mem_ball,
            Real.dist_eq,
            abs_sub_comm
          ] using hx
        have hxright : t - x < ε :=
          (abs_sub_lt_iff.1 hx').2

        have hε_le_t : ε ≤ t / 2 := by
          have hmin : min t (1 - t) ≤ t :=
            min_le_left _ _
          dsimp only [ε]
          linarith

        linarith

      have hsqrt_le :
          Real.sqrt (t / 2) ≤ Real.sqrt x :=
        Real.sqrt_le_sqrt hx_lower

      have hpos :
          0 < 2 * Real.sqrt (t / 2) := by
        have : 0 < Real.sqrt (t / 2) :=
          Real.sqrt_pos.2 (by linarith)
        linarith

      have hle :
          2 * Real.sqrt (t / 2) ≤
            2 * Real.sqrt x := by
        linarith

      have hdiv :
          1 / (2 * Real.sqrt x) ≤
            1 / (2 * Real.sqrt (t / 2)) := by
        simpa [one_div] using
          one_div_le_one_div_of_le hpos hle

      have hnonneg :
          0 ≤ 1 / (2 * Real.sqrt x) := by
        positivity

      simpa [
        cU,
        abs_of_nonneg hnonneg,
        abs_of_nonneg (Real.sqrt_nonneg x)
      ] using hdiv

    have hCoeffV :
        |1 / (2 * Real.sqrt (1 - x))| ≤ cV := by
      have h1x_lower :
          (1 - t) / 2 ≤ 1 - x := by
        have hx' : |x - t| < ε := by
          simpa [
            Metric.mem_ball,
            Real.dist_eq,
            abs_sub_comm
          ] using hx

        have hxleft : x - t < ε :=
          (abs_sub_lt_iff.1 hx').1

        have hε_le_1t :
            ε ≤ (1 - t) / 2 := by
          have hmin :
              min t (1 - t) ≤ 1 - t :=
            min_le_right _ _
          dsimp only [ε]
          linarith

        linarith

      have hsqrt_le :
          Real.sqrt ((1 - t) / 2) ≤
            Real.sqrt (1 - x) :=
        Real.sqrt_le_sqrt h1x_lower

      have hpos :
          0 <
            2 * Real.sqrt ((1 - t) / 2) := by
        have :
            0 <
              Real.sqrt ((1 - t) / 2) :=
          Real.sqrt_pos.2 (by linarith)
        linarith

      have hle :
          2 * Real.sqrt ((1 - t) / 2) ≤
            2 * Real.sqrt (1 - x) := by
        linarith

      have hdiv :
          1 / (2 * Real.sqrt (1 - x)) ≤
            1 /
              (2 * Real.sqrt ((1 - t) / 2)) := by
        simpa [one_div] using
          one_div_le_one_div_of_le hpos hle

      have hnonneg :
          0 ≤
            1 / (2 * Real.sqrt (1 - x)) := by
        positivity

      simpa [
        cV,
        abs_of_nonneg hnonneg,
        abs_of_nonneg (Real.sqrt_nonneg (1 - x))
      ] using hdiv

    have hdH_norm :
        ‖dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w‖
          ≤
        cU * ‖sk.U w‖ +
          cV * ‖sim.V w‖ := by
      have htri :
          ‖dH_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w‖
            ≤
          |1 / (2 * Real.sqrt x)| * ‖sk.U w‖ +
            |1 / (2 * Real.sqrt (1 - x))| *
              ‖sim.V w‖ := by
        simpa [
          dH_t,
          sub_eq_add_neg,
          norm_smul,
          abs_mul
        ] using
          (norm_add_le
            ((1 / (2 * Real.sqrt x)) • sk.U w)
            (-(1 / (2 * Real.sqrt (1 - x))) • sim.V w))

      exact htri.trans
        (by
          gcongr)

    have hop :
        ‖fderiv ℝ
            (fun H : EnergySpace N =>
              coupledFreeEnergyDet
                (N := N) (q := q) H Λ)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w)‖
          ≤ Cf := by
      simpa [Cf] using
        opNorm_fderiv_coupledFreeEnergyDet_le_beforeIBP
          (N := N) (q := q)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w)
          Λ

    have happ :
        ‖F' x w‖
          ≤
        Cf *
          ‖dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w‖ := by
      have hle :=
        ContinuousLinearMap.le_opNorm
          (fderiv ℝ
            (fun H : EnergySpace N =>
              coupledFreeEnergyDet
                (N := N) (q := q) H Λ)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w))
          (dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w)

      have hmul :=
        mul_le_mul_of_nonneg_right hop
          (norm_nonneg
            (dH_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w))

      simpa [F'] using hle.trans hmul

    have :
        ‖F' x w‖
          ≤
        Cf *
          (cU * ‖sk.U w‖ +
            cV * ‖sim.V w‖) :=
      happ.trans
        (mul_le_mul_of_nonneg_left
          hdH_norm hCf_nonneg)

    simpa [bound] using this

  have h_diff :
      ∀ᵐ w ∂ℙ,
        ∀ x ∈ Metric.ball t ε,
          HasDerivAt
            (fun s => F s w)
            (F' x w)
            x := by
    refine ae_of_all _ ?_
    intro w x hx

    have hxIoo :
        x ∈ Set.Ioo (0 : ℝ) 1 :=
      hball_Ioo x hx

    have hHt_diff :
        HasDerivAt
          (fun s =>
            H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) s w)
          (dH_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w)
          x :=
      hasDerivAt_H_t
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) x hxIoo w

    have houter :
        HasFDerivAt
          (fun H : EnergySpace N =>
            coupledFreeEnergyDet
              (N := N) (q := q) H Λ)
          (fderiv ℝ
            (fun H : EnergySpace N =>
              coupledFreeEnergyDet
                (N := N) (q := q) H Λ)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w))
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) x w) :=
      (hΦ.differentiable (by simp)).differentiableAt.hasFDerivAt

    change HasDerivAt
      ((fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ) ∘
        fun s => H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w)
      (F' x w) x
    simpa [F'] using houter.comp_hasDerivAt x hHt_diff

  have hmain :=
    (hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := (ℙ : Measure Ω))
      (F := F)
      (F' := F')
      (x₀ := t)
      (bound := bound)
      (s := Metric.ball t ε)
      (hs := Metric.ball_mem_nhds t hε_pos)
      hF_meas
      hF_int
      hF'_meas
      h_bound
      hbound_int
      h_diff).2

  simpa [F] using hmain

/-- Differentiate the coupled smart path before Gaussian integration by parts.

This isolates differentiation under the disorder integral from the covariance calculation.
The intended proof repeats `pressure_derivative_before_ibp` with
`coupledFreeEnergyDet (N := N) (q := q) · Λ`; positivity of
`tiltedReplicaPartitionDet` handles the logarithm. -/
lemma coupledFreeEnergy_hasDerivAt_time_before_ibp
    {t Λ : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt
      (fun s => coupledFreeEnergy
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s Λ)
      (∫ ω,
        fderiv ℝ (fun H : EnergySpace N => coupledFreeEnergyDet (N := N) (q := q) H Λ)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
          (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
        ∂ℙ) t := by
  have hraw :=
    integral_coupledFreeEnergyDet_hasDerivAt_beforeIBP
      (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) (Λ := Λ) ht

  have hfun :
      (fun s =>
        coupledFreeEnergy
          (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s Λ)
        =
      (fun s =>
        ∫ w,
          coupledFreeEnergyDet
            (N := N) (q := q)
            (H_t
              (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) s w)
            Λ
          ∂ℙ) := by
    funext s
    exact
      coupledFreeEnergy_eq_integral_det_beforeIBP
        (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s Λ

  rw [hfun]
  exact hraw

/-- Evaluation of a Hamiltonian direction on the two replicas. -/
noncomputable def pairEval
    (u : EnergySpace N) : ReplicaFun N 2 :=
  fun σs => u (σs 0) + u (σs 1)

/-- Expectation under the normalized quadratically tilted two-replica Gibbs law. -/
noncomputable def tiltedReplicaAverageDet
    (H : EnergySpace N) (coupling : ℝ)
    (f : ReplicaFun N 2) : ℝ :=
  gibbs_average_n_det (N := N) (n := 2) H
      (fun σs =>
        f σs *
          Real.exp
            (coupling * (N : ℝ) * centeredOverlapSq N q σs)) /
    tiltedReplicaPartitionDet (N := N) (q := q) H coupling

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma tiltedReplicaAverageDet_one
    (H : EnergySpace N) (coupling : ℝ) :
    tiltedReplicaAverageDet
        (N := N) (q := q) H coupling (fun _ => 1) = 1 := by
  unfold tiltedReplicaAverageDet tiltedReplicaPartitionDet
  simp only [one_mul]
  exact div_self
    (ne_of_gt
      (tiltedReplicaPartitionDet_pos
        (N := N) (q := q) H coupling))

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma tiltedReplicaAverageDet_centeredOverlapSq
    (H : EnergySpace N) (coupling : ℝ) :
    tiltedReplicaAverageDet
        (N := N) (q := q) H coupling
        (centeredOverlapSq N q) =
      tiltedCenteredOverlapSqDet
        (N := N) (q := q) H coupling := by
  rfl

/-- Explicit Hessian of the normalized coupled two-replica free energy.

The formula is the covariance of `pairEval u` and `pairEval v` under the tilted law,
with normalization `1 / (2N)`.
-/
noncomputable def coupledHessianDet
    (H : EnergySpace N) (coupling : ℝ)
    (u v : EnergySpace N) : ℝ :=
  (1 / (2 * (N : ℝ))) *
    (tiltedReplicaAverageDet
        (N := N) (q := q) H coupling
        (fun σs =>
          pairEval (N := N) u σs * pairEval (N := N) v σs) -
      tiltedReplicaAverageDet
        (N := N) (q := q) H coupling
        (pairEval (N := N) u) *
      tiltedReplicaAverageDet
        (N := N) (q := q) H coupling
        (pairEval (N := N) v))


end GeneralizedLatala
end SpinGlass
