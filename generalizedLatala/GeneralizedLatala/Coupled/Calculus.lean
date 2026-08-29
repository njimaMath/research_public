import GeneralizedLatala.Coupled.Core

/-!
# Calculus for coupled observables

Finite-dimensional derivative formulas for tilted replica averages and coupled first variations.

Main declarations:
- `fderiv_tiltedReplicaPartitionDet_apply_workspace`
- `fderiv_coupledFirstVariation_apply_workspace`

Dependencies:
- the coupled smart-path definitions and regularity lemmas

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

/-! ## Calculus layer

These are the first lemmas to prove. They use finite sums, the quotient rule,
`fderiv_gibbs_average_n_det_apply`, and positivity of
`tiltedReplicaPartitionDet`.
-/

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma fderiv_tiltedReplicaPartitionDet_apply_workspace
    (H u : EnergySpace N) (coupling : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          tiltedReplicaPartitionDet (N := N) (q := q) K coupling)
        H u =
      2 * (∑ τ : Config N, gibbs_pmf N H τ * u τ) *
          tiltedReplicaPartitionDet (N := N) (q := q) H coupling -
        gibbs_average_n_det (N := N) (n := 2) H
          (fun σs =>
            pairEval (N := N) u σs *
              Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs)) := by
  unfold gibbs_average_n_det pairEval;
  unfold gibbs_pmf tiltedReplicaPartitionDet;
  rw [ fderiv_gibbs_average_n_det_apply ];
  unfold gibbs_average_n_det gibbs_pmf;
  simp +decide [ Fin.sum_univ_two, mul_sub, sub_mul, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, Finset.sum_mul ]

/-
First Hamiltonian derivative of the deterministic coupled free energy.
-/
lemma fderiv_coupledFreeEnergyDet_apply_workspace
    (H u : EnergySpace N) (Λ : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          coupledFreeEnergyDet (N := N) (q := q) K Λ)
        H u =
      -(1 / (2 * (N : ℝ))) *
        tiltedReplicaAverageDet
          (N := N) (q := q) H (Λ / 2)
          (pairEval (N := N) u) := by
  /-
  Suggested proof:

  * unfold `coupledFreeEnergyDet`;
  * differentiate `free_energy_density` using
    `fderiv_free_energy_density_apply`;
  * differentiate the logarithm of the tilted partition function;
  * use `fderiv_gibbs_average_n_det_apply` for its Hamiltonian derivative;
  * collect the two ordinary Gibbs-average terms, which cancel;
  * divide by the positive tilted partition function.
  -/
  erw [ fderiv_add ] <;> norm_num [ fderiv_free_energy_density_apply ];
  · erw [ fderiv_mul, fderiv.log ] <;> norm_num [ fderiv_tiltedReplicaPartitionDet_apply_workspace ];
    · unfold tiltedReplicaAverageDet; ring;
      rw [ mul_inv_cancel_right₀ ( ne_of_gt ( tiltedReplicaPartitionDet_pos _ _ _ _ ) ) ] ; ring;
    · -- The sum of differentiable functions is differentiable.
      have h_diff : ∀ σs : ReplicaSpace N 2, DifferentiableAt ℝ (fun K : EnergySpace N => Real.exp (Λ / 2 * N * centeredOverlapSq N q σs) * ∏ l, Real.exp (-K.ofLp (σs l)) / Z N K) H := by
        intro σs;
        have h_diff : ∀ l : Fin 2, DifferentiableAt ℝ (fun K : EnergySpace N => Real.exp (-K.ofLp (σs l)) / Z N K) H := by
          exact fun l => differentiableAt_gibbs_pmf N H (σs l)
        fun_prop
      exact DifferentiableAt.fun_sum fun i _ => h_diff i
    · exact ne_of_gt (tiltedReplicaPartitionDet_pos N q H (Λ / 2));
    · refine' DifferentiableAt.log _ _;
      · unfold tiltedReplicaPartitionDet gibbs_average_n_det;
        unfold gibbs_pmf; norm_num [ Real.exp_ne_zero, Finset.prod_eq_zero_iff, Real.differentiableAt_exp, differentiableAt_pi ] ;
        have h_diff : ∀ x : ReplicaSpace N 2, DifferentiableAt ℝ (fun K : EnergySpace N => Real.exp (-K.ofLp (x 0)) * Real.exp (-K.ofLp (x 1)) / Z N K ^ 2) H := by
          intro x;
          apply_rules [ DifferentiableAt.div, DifferentiableAt.mul, DifferentiableAt.exp, differentiableAt_id, differentiableAt_const ];
          · fun_prop;
          · fun_prop;
          · apply_rules [ DifferentiableAt.inv, DifferentiableAt.pow, differentiableAt_id ];
            · unfold Z;
              fun_prop;
            · exact ne_of_gt ( sq_pos_of_pos ( Z_pos N H ) );
        fun_prop;
      · exact ne_of_gt ( tiltedReplicaPartitionDet_pos N q H ( Λ / 2 ) );
  · apply_rules [ DifferentiableAt.mul, DifferentiableAt.log ] <;> norm_num;
    · unfold Z ;
      fun_prop;
    · exact ne_of_gt ( Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) Finset.univ_nonempty );
  · apply_rules [ DifferentiableAt.mul, DifferentiableAt.log ] <;> norm_num [ tiltedReplicaPartitionDet_pos ];
    · unfold tiltedReplicaPartitionDet;
      unfold gibbs_average_n_det; norm_num [ gibbs_average_n, gibbs_pmf ] ;
      have h_diff : DifferentiableAt ℝ (fun x : EnergySpace N => (∑ σ : Config N, Real.exp (-x σ))) H := by
        fun_prop;
      simp_all +decide [ ← mul_div_assoc, ← Finset.sum_div _ _ _ ];
      refine' DifferentiableAt.mul _ _;
      · fun_prop;
      · exact DifferentiableAt.inv ( h_diff.pow 2 ) ( ne_of_gt ( sq_pos_of_pos ( Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) ( Finset.univ_nonempty ) ) ) );
    · exact ne_of_gt (tiltedReplicaPartitionDet_pos N q H (Λ / 2))

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma differentiableAt_tiltedReplicaAverageDet_workspace
    (H : EnergySpace N) (coupling : ℝ) (f : ReplicaFun N 2) :
    DifferentiableAt ℝ
      (fun K : EnergySpace N =>
        tiltedReplicaAverageDet (N := N) (q := q) K coupling f) H := by
  refine' DifferentiableAt.congr_of_eventuallyEq _ _;
  exact fun K => (∑ σs : ReplicaSpace N 2, (∏ l : Fin 2, gibbs_pmf N K (σs l)) * f σs * Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs)) / (∑ σs : ReplicaSpace N 2, (∏ l : Fin 2, gibbs_pmf N K (σs l)) * Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs));
  · refine' DifferentiableAt.mul _ _;
    · have h_diff : ∀ σs : ReplicaSpace N 2, DifferentiableAt ℝ (fun K : EnergySpace N => ∏ l : Fin 2, gibbs_pmf N K (σs l)) H := by
        exact fun σs => differentiableAt_prod_gibbs_pmf N 2 H σs;
      fun_prop;
    · refine' DifferentiableAt.inv _ _;
      · have h_diff : ∀ σ : Config N, DifferentiableAt ℝ (fun K : EnergySpace N => gibbs_pmf N K σ) H := by
          exact fun σ => differentiableAt_gibbs_pmf N H σ;
        fun_prop (disch := norm_num);
      · refine' ne_of_gt ( lt_of_lt_of_le _ ( Finset.single_le_sum ( fun x _ => _ ) ( Finset.mem_univ ( fun _ => fun _ => Bool.true ) ) ) );
        · exact mul_pos ( Finset.prod_pos fun _ _ => gibbs_pmf_pos _ _ _ ) ( Real.exp_pos _ );
        · exact mul_nonneg ( Finset.prod_nonneg fun _ _ => div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ ) ) ( Real.exp_nonneg _ );
  · filter_upwards [ ] with K ; unfold tiltedReplicaAverageDet gibbs_average_n_det tiltedReplicaPartitionDet ; simp +decide [ Finset.prod_mul_distrib, mul_assoc ] ;
    unfold gibbs_average_n_det; simp +decide [ mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ] ;

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma fderiv_tiltedReplicaAverageDet_apply_workspace
    (H u v : EnergySpace N) (coupling : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          tiltedReplicaAverageDet (N := N) (q := q) K coupling
            (pairEval (N := N) u))
        H v =
      - (tiltedReplicaAverageDet (N := N) (q := q) H coupling
            (fun σs => pairEval (N := N) u σs * pairEval (N := N) v σs) -
          tiltedReplicaAverageDet (N := N) (q := q) H coupling
            (pairEval (N := N) u) *
          tiltedReplicaAverageDet (N := N) (q := q) H coupling
            (pairEval (N := N) v)) := by
  unfold tiltedReplicaAverageDet;
  erw [ fderiv_mul ];
  · erw [ fderiv_fun_comp (𝕜 := ℝ) (x := H)
      (f := fun K : EnergySpace N => tiltedReplicaPartitionDet N q K coupling)
      (g := fun x : ℝ => x⁻¹)
      (differentiableAt_inv
        (ne_of_gt (tiltedReplicaPartitionDet_pos (N := N) (q := q) H coupling)))
      (by
        apply_rules [ ContDiff.differentiable ];
        apply_rules [ ContDiff.sum, ContDiff.mul, ContDiff.exp, contDiff_const, contDiff_id ];
        any_goals exact ⊤;
        · intro i hi; apply_rules [ ContDiff.mul, ContDiff.exp, contDiff_const, contDiff_id ] ;
          · fun_prop;
          · refine' ContDiff.inv _ _;
            · refine' ContDiff.sum fun σ _ => ContDiff.exp _;
              fun_prop;
            · exact fun x => ne_of_gt <| Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) Finset.univ_nonempty;
          · fun_prop;
          · refine' ContDiff.inv _ _;
            · refine' ContDiff.sum fun σ _ => ContDiff.exp _;
              fun_prop;
            · exact fun x => ne_of_gt <| Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) Finset.univ_nonempty;
        · norm_num) ];
    simp +decide [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, fderiv_tiltedReplicaPartitionDet_apply_workspace, fderiv_gibbs_average_n_det_apply ];
    unfold gibbs_average_n_det; ring;
    unfold pairEval; simp +decide [ Finset.sum_add_distrib, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, Finset.sum_mul _ _ _ ] ; ring;
    by_cases h : tiltedReplicaPartitionDet N q H coupling = 0 <;> simp_all +decide [ sq, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ] ; ring;
    simp +decide [ Finset.sum_add_distrib, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ] ; ring;
  · unfold gibbs_average_n_det;
    simp +decide [ gibbs_pmf ];
    have h_diff : DifferentiableAt ℝ (fun K : EnergySpace N => Z N K) H := by
      unfold Z ;
      fun_prop (disch := norm_num);
    have h_diff : ∀ x : ReplicaSpace N 2, DifferentiableAt ℝ (fun K : EnergySpace N => Real.exp (-K.ofLp (x 0)) * Real.exp (-K.ofLp (x 1)) / Z N K ^ 2) H := by
      intro x;
      refine' DifferentiableAt.mul _ _;
      · fun_prop;
      · exact DifferentiableAt.inv ( h_diff.pow 2 ) ( by exact ne_of_gt ( sq_pos_of_pos ( Z_pos ( N := N ) H ) ) );
    fun_prop;
  · apply DifferentiableAt.inv;
    · unfold tiltedReplicaPartitionDet;
      unfold gibbs_average_n_det;
      unfold gibbs_pmf; norm_num [ Finset.prod_mul_distrib, Real.exp_ne_zero ] ;
      have h_diff : DifferentiableAt ℝ (fun K : EnergySpace N => Z N K) H := by
        unfold Z
        fun_prop
      have h_diff : ∀ x : ReplicaSpace N 2, DifferentiableAt ℝ (fun K : EnergySpace N => Real.exp (-K.ofLp (x 0)) * Real.exp (-K.ofLp (x 1)) / Z N K ^ 2) H := by
        intro x
        refine' DifferentiableAt.mul _ _
        · fun_prop
        · exact DifferentiableAt.inv (h_diff.pow 2)
            (by exact ne_of_gt (sq_pos_of_pos (Z_pos (N := N) H)))
      fun_prop;
    · refine' ne_of_gt ( _ );
      exact tiltedReplicaPartitionDet_pos _ _ _ _

/-
Second Hamiltonian derivative of the deterministic coupled free energy.
-/
lemma fderiv_coupledFirstVariation_apply_workspace
    (H u v : EnergySpace N) (Λ : ℝ) :
    fderiv ℝ
        (fun K : EnergySpace N =>
          fderiv ℝ
            (fun L : EnergySpace N =>
              coupledFreeEnergyDet (N := N) (q := q) L Λ)
            K u)
        H v =
      coupledHessianDet
        (N := N) (q := q) H (Λ / 2) u v := by
  /-
  Rewrite the inner derivative with
  `fderiv_coupledFreeEnergyDet_apply_workspace` and differentiate the
  normalized tilted expectation. The quotient rule gives exactly the
  tilted covariance in `coupledHessianDet`.
  -/
  have h_deriv : (fderiv ℝ (fun K => (fderiv ℝ (fun L => coupledFreeEnergyDet N q L Λ) K) u) H) v = -(1 / (2 * (N : ℝ))) * (fderiv ℝ (fun K => tiltedReplicaAverageDet N q K (Λ / 2) (pairEval N u)) H) v := by
    rw [ show ( fun K => ( fderiv ℝ ( fun L => coupledFreeEnergyDet N q L Λ ) K ) u ) = fun K => - ( 1 / ( 2 * N ) ) * tiltedReplicaAverageDet N q K ( Λ / 2 ) ( pairEval N u ) from funext fun K => fderiv_coupledFreeEnergyDet_apply_workspace N q K u Λ ];
    rw [ fderiv_const_mul ] ; norm_num [ differentiableAt_tiltedReplicaAverageDet_workspace ];
    exact differentiableAt_tiltedReplicaAverageDet_workspace N q H (Λ / 2) (pairEval N u)
  rw [ h_deriv, fderiv_tiltedReplicaAverageDet_apply_workspace ] ; unfold coupledHessianDet ; ring


end GeneralizedLatala
end SpinGlass
