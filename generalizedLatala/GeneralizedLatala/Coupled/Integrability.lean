import GeneralizedLatala.Coupled.ReplicaAlgebra

/-!
# Integrability of coupled observables

Uniform finite-state bounds and integrability results for tilted and cross-replica observables.

Main declarations:
- `integrable_tiltedCenteredOverlapSqDet_Ht_workspace`
- `integrable_coupledCrossBregmanDet_Ht_workspace`

Dependencies:
- finite replica algebra and Gaussian moderate-growth estimates

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

/-! ## Integrability of the normalized finite-state observables -/

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma fourReplicaTiltWeight_sum_workspace
    (H : EnergySpace N) (coupling : ℝ) :
    (∑ σs : ReplicaSpace N 4,
      Real.exp (coupling * (N : ℝ) *
        ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (1 : Fin 4) σs) ^ 2 +
          (centeredOverlap (N := N) (q := q) (2 : Fin 4) (3 : Fin 4) σs) ^ 2)) *
        ∏ l, gibbs_pmf N H (σs l)) =
      (tiltedReplicaPartitionDet (N := N) (q := q) H coupling) ^ 2 := by
  rw [ sq, tiltedReplicaPartitionDet ];
  unfold gibbs_average_n_det;
  simp +decide only [Fin.prod_univ_two, Finset.sum_mul];
  simp +decide only [Finset.mul_sum _ _ _];
  rw [ ← Finset.sum_product' ];
  refine' Finset.sum_bij ( fun x _ => ( fun i => x ( if i = 0 then 0 else 1 ), fun i => x ( if i = 0 then 2 else 3 ) ) ) _ _ _ _ <;> simp +decide;
  · simp +decide [ funext_iff, Fin.forall_fin_succ ];
    tauto;
  · exact fun a b => ⟨ fun i => if i = 0 then a 0 else if i = 1 then a 1 else if i = 2 then b 0 else b 1, by ext i; fin_cases i <;> rfl, by ext i; fin_cases i <;> rfl ⟩;
  · simp +decide [ Fin.prod_univ_four, centeredOverlapSq ];
    simp +decide [ centeredOverlap, overlap ] ; intros ; ring_nf;
    simpa only [ mul_assoc, ← Real.exp_add ] using by ring;

lemma measurable_H_t_workspace (t : ℝ) :
    Measurable
      (H_t (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t) := by
  have hU : Measurable sk.U := sk.hU.repr_measurable
  have hV : Measurable sim.V := sim.hV.repr_measurable
  exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
    (sk := sk) (sim := sim) t

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma measurable_coupledCrossMomentDet_workspace (coupling : ℝ) :
    Measurable
      (fun H : EnergySpace N =>
        coupledCrossMomentDet (N := N) (q := q) H coupling) := by
  refine' Measurable.mul _ _;
  · apply_rules [ Finset.measurable_sum, Finset.measurable_prod ];
    refine' fun σ _ => Measurable.mul _ _;
    · fun_prop;
    · exact Finset.measurable_prod _ fun _ _ => ( contDiff_gibbs_pmf N ( σ _ ) |> ContDiff.continuous |> Continuous.measurable );
  · refine' Measurable.inv ( Measurable.pow_const _ _ );
    refine' Finset.measurable_sum _ fun σs _ => _;
    refine' Measurable.mul _ _;
    · exact measurable_const;
    · exact Finset.measurable_prod _ fun _ _ => ( contDiff_gibbs_pmf ( N := N ) ( σ := σs _ ) |> ContDiff.continuous |> Continuous.measurable )

lemma integrable_tiltedCenteredOverlapSqDet_Ht_workspace
    (t coupling : ℝ) :
    Integrable
      (fun ω =>
        tiltedCenteredOverlapSqDet
          (N := N) (q := q)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t ω)
          coupling) ℙ := by
  /-
  The tilted quantity is a normalized expectation of a fixed observable on a
  finite state space. Bound it by

    `∑ σs : ReplicaSpace N 2, |centeredOverlapSq N q σs|`.
  -/
  refine' MeasureTheory.Integrable.mono' _ _ _;
  refine' fun ω => ( ∑ σs : ReplicaSpace N 2, ( N : ℝ ) * centeredOverlapSq N q σs );
  · norm_num;
  · have h_measurable : Measurable (fun H : EnergySpace N => tiltedCenteredOverlapSqDet (N := N) (q := q) H coupling) := by
      refine' Measurable.div _ _;
      · refine' Finset.measurable_sum _ fun σs _ => _;
        refine' Measurable.mul _ _;
        · fun_prop;
        · refine' Finset.measurable_prod _ fun i _ => _;
          refine' Measurable.div _ _;
          · fun_prop;
          · refine' Finset.measurable_sum _ fun σ _ => _;
            fun_prop;
      · refine' Finset.measurable_sum _ fun σs _ => _;
        refine' Measurable.mul _ _;
        · exact measurable_const;
        · refine' Finset.measurable_prod _ fun i _ => _;
          refine' Measurable.div _ _;
          · fun_prop;
          · exact Finset.measurable_sum _ fun _ _ => Real.continuous_exp.measurable.comp ( measurable_neg.comp ( by measurability ) );
    have h_measurable : Measurable (fun ω => H_t (N := N) (β := β) (h := h)
        (q := q) (sk := sk) (sim := sim) t ω) :=
      measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t
    exact Measurable.aestronglyMeasurable ( by measurability );
  · refine' Filter.Eventually.of_forall fun ω => _;
    rw [ tiltedCenteredOverlapSqDet ];
    rw [ gibbs_average_n_det, tiltedReplicaPartitionDet ];
    rw [ gibbs_average_n_det ];
    rw [ Real.norm_of_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => mul_nonneg ( mul_nonneg ( by exact sq_nonneg _ ) ( Real.exp_nonneg _ ) ) ( Finset.prod_nonneg fun _ _ => by exact div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ ) ) ) ( Finset.sum_nonneg fun _ _ => mul_nonneg ( Real.exp_nonneg _ ) ( Finset.prod_nonneg fun _ _ => by exact div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ ) ) ) ) ];
    rw [ div_le_iff₀ ];
    · rw [ Finset.sum_mul _ _ _ ];
      refine' Finset.sum_le_sum fun i _ => _;
      refine' le_trans _ ( mul_le_mul_of_nonneg_left ( Finset.single_le_sum ( fun a _ => mul_nonneg ( Real.exp_nonneg _ ) ( Finset.prod_nonneg fun b _ => _ ) ) ( Finset.mem_univ i ) ) _ );
      · rw [ mul_assoc ];
        gcongr;
        · exact mul_nonneg ( Real.exp_nonneg _ ) ( Finset.prod_nonneg fun _ _ => div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ ) );
        · exact le_mul_of_one_le_left ( sq_nonneg _ ) ( mod_cast NeZero.pos N );
      · exact div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ );
      · exact mul_nonneg ( Nat.cast_nonneg _ ) ( sq_nonneg _ );
    · refine' Finset.sum_pos _ _ <;> simp +decide [ gibbs_pmf ];
      exact fun _ => mul_pos ( Real.exp_pos _ ) ( div_pos ( mul_pos ( Real.exp_pos _ ) ( Real.exp_pos _ ) ) ( sq_pos_of_pos ( Z_pos _ _ ) ) )

lemma integrable_coupledCrossMomentDet_Ht_workspace
    (t coupling : ℝ) :
    Integrable
      (fun ω =>
        coupledCrossMomentDet
          (N := N) (q := q)
          (H_t
            (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t ω)
          coupling) ℙ := by
  /-
  Again use the normalized finite four-replica law and bound by the finite sum
  of `|crossPairCenteredOverlapSq|`.
  -/
  refine' MeasureTheory.Integrable.mono' _ _ _;
  refine' fun ω => ∑ σs : ReplicaSpace N 4, |crossPairCenteredOverlapSq N q σs|;
  · norm_num;
  · exact Measurable.aestronglyMeasurable ( by exact Measurable.comp ( measurable_coupledCrossMomentDet_workspace N q coupling ) ( measurable_H_t_workspace N β h q sk sim t ) );
  · refine' Filter.Eventually.of_forall fun ω => _;
    unfold coupledCrossMomentDet gibbs_average_n_det;
    rw [ norm_div ];
    refine' div_le_of_le_mul₀ _ _ _;
    · positivity;
    · exact Finset.sum_nonneg fun _ _ => abs_nonneg _;
    · refine' le_trans ( norm_sum_le _ _ ) _;
      rw [ Finset.sum_mul _ _ _ ];
      refine' Finset.sum_le_sum fun σs _ => _;
      rw [ ← fourReplicaTiltWeight_sum_workspace ];
      refine' le_trans _ ( mul_le_mul_of_nonneg_left ( le_abs_self _ ) ( abs_nonneg _ ) );
      refine' le_trans _ ( mul_le_mul_of_nonneg_left ( Finset.single_le_sum ( fun σs _ => _ ) ( Finset.mem_univ σs ) ) ( abs_nonneg _ ) );
      · simp +decide [ abs_mul, abs_of_nonneg, Real.exp_nonneg, gibbs_pmf_nonneg ];
        rw [ mul_assoc ];
      · exact mul_nonneg ( Real.exp_nonneg _ ) ( Finset.prod_nonneg fun _ _ => div_nonneg ( Real.exp_nonneg _ ) ( Finset.sum_nonneg fun _ _ => Real.exp_nonneg _ ) )

lemma integrable_tiltedBregmanDet_Ht_workspace (t coupling : ℝ) :
    Integrable
      (fun ω => tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω) coupling) ℙ := by
  let B : ℝ := ∑ σs : ReplicaSpace N 2,
    |bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk) σs|
  apply (integrable_const B).mono'
  · apply Measurable.aestronglyMeasurable
    have hHt := measurable_H_t_workspace N β h q sk sim t
    have hf : Measurable (fun H : EnergySpace N =>
        tiltedBregmanDet N β h q sk H coupling) := by
      unfold tiltedBregmanDet tiltedReplicaPartitionDet gibbs_average_n_det
      apply Measurable.div
      · apply Finset.measurable_sum
        intro σs _
        apply measurable_const.mul
        apply Finset.measurable_prod
        intro l _
        exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous.measurable
      · apply Finset.measurable_sum
        intro σs _
        apply measurable_const.mul
        apply Finset.measurable_prod
        intro l _
        exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous.measurable
    exact hf.comp hHt
  · filter_upwards with ω
    let H := H_t N β h q sk sim t ω
    let w : ReplicaSpace N 2 → ℝ := fun σs =>
      Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) *
        ∏ l, gibbs_pmf N H (σs l)
    have hw (σs : ReplicaSpace N 2) : 0 ≤ w σs :=
      mul_nonneg (Real.exp_nonneg _) (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg N H (σs l))
    have hZ : 0 < ∑ σs : ReplicaSpace N 2, w σs := by
      change 0 < tiltedReplicaPartitionDet (N := N) (q := q) H coupling
      exact tiltedReplicaPartitionDet_pos N q H coupling
    have hbound : ‖(∑ σs, bregmanOverlap N β h q sk σs * w σs) /
        (∑ σs, w σs)‖ ≤ B := by
      rw [norm_div, Real.norm_of_nonneg hZ.le]
      apply (div_le_iff₀ hZ).2
      calc
        ‖∑ σs, bregmanOverlap N β h q sk σs * w σs‖
            ≤ ∑ σs, ‖bregmanOverlap N β h q sk σs * w σs‖ := norm_sum_le _ _
        _ ≤ ∑ σs, |bregmanOverlap N β h q sk σs| * (∑ ρs, w ρs) := by
          apply Finset.sum_le_sum
          intro σs _
          rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (hw σs)]
          exact mul_le_mul_of_nonneg_left
            (Finset.single_le_sum (fun ρs _ => hw ρs) (Finset.mem_univ σs)) (abs_nonneg _)
        _ = B * ∑ σs, w σs := by rw [Finset.sum_mul]
    simpa only [tiltedBregmanDet, tiltedReplicaPartitionDet, gibbs_average_n_det,
      w, H, mul_assoc] using hbound

lemma integrable_coupledCrossBregmanDet_Ht_workspace (t coupling : ℝ) :
    Integrable
      (fun ω => coupledCrossBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t ω) coupling) ℙ := by
  let B : ℝ := ∑ σs : ReplicaSpace N 4,
    |crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs|
  apply (integrable_const B).mono'
  · apply Measurable.aestronglyMeasurable
    have hHt := measurable_H_t_workspace N β h q sk sim t
    have hf : Measurable (fun H : EnergySpace N =>
        coupledCrossBregmanDet N β h q sk H coupling) := by
      unfold coupledCrossBregmanDet tiltedReplicaPartitionDet gibbs_average_n_det
      apply Measurable.div
      · apply Finset.measurable_sum
        intro σs _
        apply measurable_const.mul
        apply Finset.measurable_prod
        intro l _
        exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous.measurable
      · apply Measurable.pow_const
        apply Finset.measurable_sum
        intro σs _
        apply measurable_const.mul
        apply Finset.measurable_prod
        intro l _
        exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous.measurable
    exact hf.comp hHt
  · filter_upwards with ω
    let H := H_t N β h q sk sim t ω
    let w : ReplicaSpace N 4 → ℝ := fun σs =>
      Real.exp (coupling * (N : ℝ) *
        ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (1 : Fin 4) σs) ^ 2 +
          (centeredOverlap (N := N) (q := q) (2 : Fin 4) (3 : Fin 4) σs) ^ 2)) *
        ∏ l, gibbs_pmf N H (σs l)
    have hw (σs : ReplicaSpace N 4) : 0 ≤ w σs :=
      mul_nonneg (Real.exp_nonneg _) (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg N H (σs l))
    have hZ : 0 < ∑ σs : ReplicaSpace N 4, w σs := by
      rw [fourReplicaTiltWeight_sum_workspace (N := N) (q := q) H coupling]
      exact sq_pos_of_pos (tiltedReplicaPartitionDet_pos N q H coupling)
    have hbound : ‖(∑ σs, crossPairBregman N β h q sk σs * w σs) /
        (∑ σs, w σs)‖ ≤ B := by
      rw [norm_div, Real.norm_of_nonneg hZ.le]
      apply (div_le_iff₀ hZ).2
      calc
        ‖∑ σs, crossPairBregman N β h q sk σs * w σs‖
            ≤ ∑ σs, ‖crossPairBregman N β h q sk σs * w σs‖ := norm_sum_le _ _
        _ ≤ ∑ σs, |crossPairBregman N β h q sk σs| * (∑ ρs, w ρs) := by
          apply Finset.sum_le_sum
          intro σs _
          rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (hw σs)]
          exact mul_le_mul_of_nonneg_left
            (Finset.single_le_sum (fun ρs _ => hw ρs) (Finset.mem_univ σs)) (abs_nonneg _)
        _ = B * ∑ σs, w σs := by rw [Finset.sum_mul]
    simp only [w, H] at hbound
    rw [fourReplicaTiltWeight_sum_workspace (N := N) (q := q)
      (H_t N β h q sk sim t ω) coupling] at hbound
    simpa only [coupledCrossBregmanDet, tiltedReplicaPartitionDet, gibbs_average_n_det,
      mul_assoc] using hbound


end GeneralizedLatala
end SpinGlass
