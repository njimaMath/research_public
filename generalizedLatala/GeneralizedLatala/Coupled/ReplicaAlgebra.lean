import GeneralizedLatala.Coupled.GaussianIBP

/-!
# Finite replica algebra

Finite-sum covariance identities that rewrite the coupled Gaussian trace in overlap variables.

Main declarations:
- `covKernelDiff_eq_bregman_workspace`
- `coupled_trace_algebra_workspace`

Dependencies:
- the coupled Gaussian trace formula

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

/-! ## Finite replica algebra -/

private lemma weighted_sum_sub_constant
    {ι : Type*} [Fintype ι] (c a : ℝ) (f weight : ι → ℝ) :
    (∑ i, c * (f i - a) * weight i) =
      c * (∑ i, f i * weight i) - c * a * ∑ i, weight i := by
  calc
    (∑ i, c * (f i - a) * weight i) =
        ∑ i, (c * f i * weight i - c * a * weight i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by
      rw [Finset.sum_sub_distrib]
      refine congrArg₂ (· - ·) ?_ ?_
      · calc
          (∑ i, c * f i * weight i) = ∑ i, c * (f i * weight i) := by
            apply Finset.sum_congr rfl
            intro i _
            ring
          _ = _ := (Finset.mul_sum _ _ _).symm
      · exact (Finset.mul_sum _ _ _).symm

lemma covKernelDiff_eq_bregman_workspace
    (σ τ : Config N) :
    mixedCovKernel N sk.ξ σ τ - referenceCovKernel N β σ τ =
      (N : ℝ) * (bregmanRemainder sk.ξ β q (overlap N σ τ) +
        (sk.ξ q - β * q)) := by
  simp [mixedCovKernel, referenceCovKernel, bregmanRemainder]
  ring

lemma sum_crossPairCenteredOverlapSq_workspace
    (σs : ReplicaSpace N 4) :
    (centeredOverlap
        (N := N) (q := q)
        (0 : Fin 4) (2 : Fin 4) σs) ^ 2 +
      (centeredOverlap
        (N := N) (q := q)
        (0 : Fin 4) (3 : Fin 4) σs) ^ 2 +
      (centeredOverlap
        (N := N) (q := q)
        (1 : Fin 4) (2 : Fin 4) σs) ^ 2 +
      (centeredOverlap
        (N := N) (q := q)
        (1 : Fin 4) (3 : Fin 4) σs) ^ 2 =
      4 * crossPairCenteredOverlapSq
        (N := N) (q := q) σs := by
  unfold crossPairCenteredOverlapSq
  ring

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma sum_pairEval_std_basis_product_workspace
    (D : Config N → Config N → ℝ) (σs : ReplicaSpace N 2) :
    (∑ σ : Config N, ∑ τ : Config N,
      D σ τ * pairEval N (std_basis N σ) σs *
        pairEval N (std_basis N τ) σs) =
      D (σs 0) (σs 0) + D (σs 0) (σs 1) +
        D (σs 1) (σs 0) + D (σs 1) (σs 1) := by
  simp only [pairEval, std_basis]
  ring_nf
  simp_rw [Finset.sum_add_distrib]
  simp

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma sum_pairEval_std_basis_cross_workspace
    (D : Config N → Config N → ℝ)
    (σs ρs : ReplicaSpace N 2) :
    (∑ σ : Config N, ∑ τ : Config N,
      D σ τ * pairEval N (std_basis N σ) σs *
        pairEval N (std_basis N τ) ρs) =
      D (σs 0) (ρs 0) + D (σs 0) (ρs 1) +
        D (σs 1) (ρs 0) + D (σs 1) (ρs 1) := by
  simp only [pairEval, std_basis]
  ring_nf
  simp_rw [Finset.sum_add_distrib]
  simp

/-- Pointwise finite-volume trace identity.

This is the main algebraic goal. Unfold `coupledHessianDet`; the first tilted
expectation gives `(1-q)^2 + tilted Q₁₂²`, while the product of tilted means is
represented by four replicas and gives `2 * coupledCrossMomentDet` after all
normalizations are collected.
-/
lemma coupled_trace_algebra_workspace
    (hN : 0 < N)
    (H : EnergySpace N) (coupling : ℝ) :
    (1 / 2) *
        (∑ σ : Config N, ∑ τ : Config N,
          (mixedCovKernel N sk.ξ σ τ - referenceCovKernel N β σ τ) *
            coupledHessianDet
              (N := N) (q := q) H coupling
              (std_basis N σ) (std_basis N τ)) =
      (1 / 2) *
        (bregmanRemainder sk.ξ β q 1 +
          tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling -
          2 * coupledCrossBregmanDet
            (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling) := by
  /-
  Useful ingredients:

  * `covKernelDiff_eq_centered_sq_workspace`;
  * `overlap_self hN`;
  * `sum_gibbs_pmf` and `sum_prod_gibbs_pmf_eq_one`;
  * `tiltedReplicaPartitionDet_pos`;
  * an explicit equivalence
    `ReplicaSpace N 4 ≃ ReplicaSpace N 2 × ReplicaSpace N 2`;
  * `sum_crossPairCenteredOverlapSq_workspace`.
  -/
  classical

    let c : ℝ := sk.ξ q - β * q
    let D : Config N → Config N → ℝ := fun σ τ =>
      bregmanRemainder sk.ξ β q (overlap N σ τ) + c

    let W₂ : ReplicaSpace N 2 → ℝ := fun σs =>
      Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) *
        ∏ l, gibbs_pmf N H (σs l)

    let W₄ : ReplicaSpace N 4 → ℝ := fun σs =>
      Real.exp (coupling * (N : ℝ) *
        ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (1 : Fin 4) σs) ^ 2 +
        (centeredOverlap (N := N) (q := q) (2 : Fin 4) (3 : Fin 4) σs) ^ 2)) *
        ∏ l, gibbs_pmf N H (σs l)

    let e : ReplicaSpace N 4 ≃ ReplicaSpace N 2 × ReplicaSpace N 2 :=
      { toFun := fun σs =>
          (fun i => σs (if i = 0 then 0 else 1),
          fun i => σs (if i = 0 then 2 else 3))
        invFun := fun p i =>
          if i = 0 then p.1 0
          else if i = 1 then p.1 1
          else if i = 2 then p.2 0
          else p.2 1
        left_inv := by
          intro σs
          ext i
          fin_cases i <;> rfl
        right_inv := by
          intro p
          rcases p with ⟨σs, ρs⟩
          apply Prod.ext
          · ext i
            fin_cases i <;> rfl
          · ext i
            fin_cases i <;> rfl }

    have hN0 : (N : ℝ) ≠ 0 := by
      exact_mod_cast hN.ne'

    have hoverlap_comm (σ τ : Config N) :
        overlap N σ τ = overlap N τ σ := by
      unfold overlap
      congr 1
      apply Finset.sum_congr rfl
      intro i _
      ring

    have hpart :
        tiltedReplicaPartitionDet (N := N) (q := q) H coupling =
          ∑ σs : ReplicaSpace N 2, W₂ σs := by
      rfl

    have hZpos : 0 < ∑ σs : ReplicaSpace N 2, W₂ σs := by
      rw [← hpart]
      exact tiltedReplicaPartitionDet_pos
        (N := N) (q := q) H coupling

    have hZ0 : (∑ σs : ReplicaSpace N 2, W₂ σs) ≠ 0 :=
      ne_of_gt hZpos

    have htilt (f : ReplicaFun N 2) :
        tiltedReplicaAverageDet (N := N) (q := q) H coupling f =
          (∑ σs : ReplicaSpace N 2, f σs * W₂ σs) /
            (∑ σs : ReplicaSpace N 2, W₂ σs) := by
      unfold tiltedReplicaAverageDet gibbs_average_n_det
      rw [hpart]
      congr 1
      apply Finset.sum_congr rfl
      intro σs _
      dsimp only [W₂]
      ring

    have htiltedBregman :
        tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling =
          (∑ σs : ReplicaSpace N 2,
              bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) * W₂ σs) /
            (∑ σs : ReplicaSpace N 2, W₂ σs) := by
      unfold tiltedBregmanDet bregmanOverlap gibbs_average_n_det
      rw [hpart]
      congr 1
      apply Finset.sum_congr rfl
      intro σs _
      dsimp only [W₂]
      ring

    have hcross :
        coupledCrossBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling =
          (∑ σs : ReplicaSpace N 4,
              crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs * W₄ σs) /
            (∑ σs : ReplicaSpace N 2, W₂ σs) ^ 2 := by
      unfold coupledCrossBregmanDet gibbs_average_n_det
      rw [hpart]
      congr 1
      apply Finset.sum_congr rfl
      intro σs _
      dsimp only [W₄]
      ring

    have hsplit_weight (σs : ReplicaSpace N 4) :
        W₄ σs = W₂ (e σs).1 * W₂ (e σs).2 := by
      change W₄ σs =
        W₂ (fun i : Fin 2 => σs (if i = 0 then 0 else 1)) *
          W₂ (fun i : Fin 2 => σs (if i = 0 then 2 else 3))
      dsimp only [W₄, W₂, centeredOverlapSq, centeredOverlap]
      simp +decide only [Fin.prod_univ_two, Fin.prod_univ_four, if_true, if_false]
      rw [show coupling * (N : ℝ) *
            ((overlap N (σs 0) (σs 1) - q) ^ 2 +
              (overlap N (σs 2) (σs 3) - q) ^ 2) =
          coupling * (N : ℝ) * (overlap N (σs 0) (σs 1) - q) ^ 2 +
            coupling * (N : ℝ) * (overlap N (σs 2) (σs 3) - q) ^ 2 by
        ring]
      rw [Real.exp_add]
      ring_nf

    have hweight_sq :
        (∑ σs : ReplicaSpace N 4, W₄ σs) =
          (∑ σs : ReplicaSpace N 2, W₂ σs) ^ 2 := by
      calc
        (∑ σs : ReplicaSpace N 4, W₄ σs) =
            ∑ p : ReplicaSpace N 2 × ReplicaSpace N 2,
              W₂ p.1 * W₂ p.2 := by
          exact Fintype.sum_equiv e W₄
            (fun p => W₂ p.1 * W₂ p.2) hsplit_weight
        _ = (∑ σs : ReplicaSpace N 2, W₂ σs) ^ 2 := by
          simp only [Fintype.sum_prod_type]
          rw [sq, Finset.sum_mul]
          simp only [Finset.mul_sum]

    have hfour_pair :
        (∑ σs : ReplicaSpace N 4,
            (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs + c)) *
              W₄ σs) =
          ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
            (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk)
                (e.symm (σs, ρs)) + c)) * W₂ σs * W₂ ρs := by
      calc
        (∑ σs : ReplicaSpace N 4,
            (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs + c)) *
              W₄ σs) =
            ∑ p : ReplicaSpace N 2 × ReplicaSpace N 2,
              (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) (e.symm p) + c)) *
                W₂ p.1 * W₂ p.2 := by
          exact Fintype.sum_equiv e
            (fun σs =>
              (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs + c)) *
                W₄ σs)
            (fun p =>
              (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) (e.symm p) + c)) *
                W₂ p.1 * W₂ p.2)
            (fun σs => by
              simpa only [hsplit_weight σs, Equiv.symm_apply_apply, mul_assoc])
        _ = _ := by
          simp only [Fintype.sum_prod_type]

    have hwithin_point (σs : ReplicaSpace N 2) :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ * pairEval N (std_basis N σ) σs *
            pairEval N (std_basis N τ) σs) =
          2 * (bregmanRemainder sk.ξ β q 1 + c) +
            2 * (bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) + c) := by
      rw [sum_pairEval_std_basis_product_workspace
        (N := N) (D := D) σs]
      dsimp only [D]
      rw [overlap_self (N := N) hN (σs 0),
        overlap_self (N := N) hN (σs 1),
        hoverlap_comm (σs 1) (σs 0)]
      dsimp only [bregmanRemainder]
      ring

    have hcross_point (σs ρs : ReplicaSpace N 2) :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ * pairEval N (std_basis N σ) σs *
            pairEval N (std_basis N τ) ρs) =
          4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk)
              (e.symm (σs, ρs)) + c) := by
      rw [sum_pairEval_std_basis_cross_workspace
        (N := N) (D := D) σs ρs]
      simp +decide [D, c, e, crossPairBregman, bregmanRemainder]
      ring

    have hwithin_num :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (∑ σs : ReplicaSpace N 2,
              (pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs) * W₂ σs)) =
          ∑ σs : ReplicaSpace N 2,
            (2 * (bregmanRemainder sk.ξ β q 1 + c) +
              2 * (bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) + c)) *
              W₂ σs := by
      calc
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (∑ σs : ReplicaSpace N 2,
              (pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs) * W₂ σs)) =
            ∑ σ : Config N, ∑ τ : Config N, ∑ σs : ReplicaSpace N 2,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs * W₂ σs := by
          apply Finset.sum_congr rfl
          intro σ _
          apply Finset.sum_congr rfl
          intro τ _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro σs _
          ring
        _ = ∑ σ : Config N, ∑ σs : ReplicaSpace N 2, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs * W₂ σs := by
          apply Finset.sum_congr rfl
          intro σ _
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2, ∑ σ : Config N, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs * W₂ σs := by
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2,
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs) * W₂ σs := by
          apply Finset.sum_congr rfl
          intro σs _
          calc
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs * W₂ σs) =
                ∑ σ : Config N,
                  (∑ τ : Config N,
                    D σ τ * pairEval N (std_basis N σ) σs *
                      pairEval N (std_basis N τ) σs) * W₂ σs := by
              apply Finset.sum_congr rfl
              intro σ _
              rw [Finset.sum_mul]
            _ = _ := by
              rw [Finset.sum_mul]
        _ = _ := by
          apply Finset.sum_congr rfl
          intro σs _
          rw [hwithin_point σs]

    have hcross_num :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (∑ σs : ReplicaSpace N 2,
              pairEval N (std_basis N σ) σs * W₂ σs) *
            (∑ ρs : ReplicaSpace N 2,
              pairEval N (std_basis N τ) ρs * W₂ ρs)) =
          ∑ σs : ReplicaSpace N 4,
            (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs + c)) *
              W₄ σs := by
      calc
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (∑ σs : ReplicaSpace N 2,
              pairEval N (std_basis N σ) σs * W₂ σs) *
            (∑ ρs : ReplicaSpace N 2,
              pairEval N (std_basis N τ) ρs * W₂ ρs)) =
            ∑ σ : Config N, ∑ τ : Config N,
              ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
                D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                  pairEval N (std_basis N τ) ρs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σ _
          apply Finset.sum_congr rfl
          intro τ _
          calc
            D σ τ *
                  (∑ σs : ReplicaSpace N 2,
                    pairEval N (std_basis N σ) σs * W₂ σs) *
                  (∑ ρs : ReplicaSpace N 2,
                    pairEval N (std_basis N τ) ρs * W₂ ρs) =
                (∑ σs : ReplicaSpace N 2,
                  D σ τ * (pairEval N (std_basis N σ) σs * W₂ σs)) *
                  (∑ ρs : ReplicaSpace N 2,
                    pairEval N (std_basis N τ) ρs * W₂ ρs) := by
              congr 1
              rw [Finset.mul_sum]
            _ = ∑ σs : ReplicaSpace N 2,
                  (D σ τ * (pairEval N (std_basis N σ) σs * W₂ σs)) *
                    (∑ ρs : ReplicaSpace N 2,
                      pairEval N (std_basis N τ) ρs * W₂ ρs) := by
              rw [Finset.sum_mul]
            _ = ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
                  D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                    pairEval N (std_basis N τ) ρs * W₂ ρs := by
              apply Finset.sum_congr rfl
              intro σs _
              rw [Finset.mul_sum]
              apply Finset.sum_congr rfl
              intro ρs _
              ring
        _ = ∑ σ : Config N, ∑ σs : ReplicaSpace N 2,
              ∑ τ : Config N, ∑ ρs : ReplicaSpace N 2,
                D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                  pairEval N (std_basis N τ) ρs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σ _
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2, ∑ σ : Config N,
              ∑ τ : Config N, ∑ ρs : ReplicaSpace N 2,
                D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                  pairEval N (std_basis N τ) ρs * W₂ ρs := by
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2, ∑ σ : Config N,
              ∑ ρs : ReplicaSpace N 2, ∑ τ : Config N,
                D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                  pairEval N (std_basis N τ) ρs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σs _
          apply Finset.sum_congr rfl
          intro σ _
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
              ∑ σ : Config N, ∑ τ : Config N,
                D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                  pairEval N (std_basis N τ) ρs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σs _
          rw [Finset.sum_comm]
        _ = ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) ρs) * W₂ σs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σs _
          apply Finset.sum_congr rfl
          intro ρs _
          calc
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ * pairEval N (std_basis N σ) σs * W₂ σs *
                pairEval N (std_basis N τ) ρs * W₂ ρs) =
                ∑ σ : Config N, ∑ τ : Config N,
                  (D σ τ * pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) ρs) * (W₂ σs * W₂ ρs) := by
              apply Finset.sum_congr rfl
              intro σ _
              apply Finset.sum_congr rfl
              intro τ _
              ring
            _ = ∑ σ : Config N,
                  (∑ τ : Config N,
                    D σ τ * pairEval N (std_basis N σ) σs *
                      pairEval N (std_basis N τ) ρs) * (W₂ σs * W₂ ρs) := by
              apply Finset.sum_congr rfl
              intro σ _
              rw [Finset.sum_mul]
            _ = (∑ σ : Config N, ∑ τ : Config N,
                  D σ τ * pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) ρs) * (W₂ σs * W₂ ρs) := by
              rw [Finset.sum_mul]
            _ = _ := by
              ring
        _ = ∑ σs : ReplicaSpace N 2, ∑ ρs : ReplicaSpace N 2,
            (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk)
                (e.symm (σs, ρs)) + c)) * W₂ σs * W₂ ρs := by
          apply Finset.sum_congr rfl
          intro σs _
          apply Finset.sum_congr rfl
          intro ρs _
          rw [hcross_point σs ρs]
        _ = _ := hfour_pair.symm

    have hwithin :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (fun σs => pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs)) =
          2 * (bregmanRemainder sk.ξ β q 1 + c) +
            2 * tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling +
            2 * c := by
      calc
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (fun σs => pairEval N (std_basis N σ) σs *
                pairEval N (std_basis N τ) σs)) =
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                (∑ σs : ReplicaSpace N 2,
                  (pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) σs) * W₂ σs)) /
              (∑ σs : ReplicaSpace N 2, W₂ σs) := by
          simp_rw [htilt]
          simp only [div_eq_mul_inv]
          calc
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                ((∑ σs : ReplicaSpace N 2,
                  (pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) σs) * W₂ σs) *
                  (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹)) =
                ∑ σ : Config N, ∑ τ : Config N,
                  (D σ τ *
                    (∑ σs : ReplicaSpace N 2,
                      (pairEval N (std_basis N σ) σs *
                        pairEval N (std_basis N τ) σs) * W₂ σs)) *
                    (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹ := by
              apply Finset.sum_congr rfl
              intro σ _
              apply Finset.sum_congr rfl
              intro τ _
              ring
            _ = ∑ σ : Config N,
                  (∑ τ : Config N,
                    D σ τ *
                      (∑ σs : ReplicaSpace N 2,
                        (pairEval N (std_basis N σ) σs *
                          pairEval N (std_basis N τ) σs) * W₂ σs)) *
                    (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹ := by
              apply Finset.sum_congr rfl
              intro σ _
              rw [Finset.sum_mul]
            _ = (∑ σ : Config N, ∑ τ : Config N,
                  D σ τ *
                    (∑ σs : ReplicaSpace N 2,
                      (pairEval N (std_basis N σ) σs *
                        pairEval N (std_basis N τ) σs) * W₂ σs)) *
                  (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹ := by
              rw [Finset.sum_mul]
        _ = (∑ σs : ReplicaSpace N 2,
              (2 * (bregmanRemainder sk.ξ β q 1 + c) +
                2 * (bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) + c)) *
                W₂ σs) /
              (∑ σs : ReplicaSpace N 2, W₂ σs) := by
          rw [hwithin_num]
        _ = 2 * (bregmanRemainder sk.ξ β q 1 + c) +
            2 * tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling +
            2 * c := by
          rw [htiltedBregman]
          field_simp [hZ0]
          calc
            (∑ σs : ReplicaSpace N 2,
                2 * ((bregmanRemainder sk.ξ β q 1 + c) +
                  (bregmanRemainder sk.ξ β q
                    (overlap N (σs 0) (σs 1)) + c)) * W₂ σs) =
                ∑ σs : ReplicaSpace N 2,
                  ((2 * (bregmanRemainder sk.ξ β q 1 + c) + 2 * c) * W₂ σs +
                    2 * (bregmanRemainder sk.ξ β q
                      (overlap N (σs 0) (σs 1)) * W₂ σs)) := by
              apply Finset.sum_congr rfl
              intro σs _
              ring
            _ = (∑ σs : ReplicaSpace N 2,
                  (2 * (bregmanRemainder sk.ξ β q 1 + c) + 2 * c) * W₂ σs) +
                ∑ σs : ReplicaSpace N 2,
                  2 * (bregmanRemainder sk.ξ β q
                    (overlap N (σs 0) (σs 1)) * W₂ σs) :=
              Finset.sum_add_distrib
            _ = _ := by
              rw [← Finset.mul_sum, ← Finset.mul_sum]
              ring

    have hbetween :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (pairEval N (std_basis N σ)) *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (pairEval N (std_basis N τ))) =
          4 * (coupledCrossBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling + c) := by
      calc
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (pairEval N (std_basis N σ)) *
            tiltedReplicaAverageDet (N := N) (q := q) H coupling
              (pairEval N (std_basis N τ))) =
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                (∑ σs : ReplicaSpace N 2,
                  pairEval N (std_basis N σ) σs * W₂ σs) *
                (∑ ρs : ReplicaSpace N 2,
                  pairEval N (std_basis N τ) ρs * W₂ ρs)) /
              (∑ σs : ReplicaSpace N 2, W₂ σs) ^ 2 := by
          simp_rw [htilt]
          simp only [div_eq_mul_inv]
          rw [← inv_pow]
          calc
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                ((∑ σs : ReplicaSpace N 2,
                  pairEval N (std_basis N σ) σs * W₂ σs) *
                  (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹) *
                ((∑ ρs : ReplicaSpace N 2,
                  pairEval N (std_basis N τ) ρs * W₂ ρs) *
                  (∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹)) =
                ∑ σ : Config N, ∑ τ : Config N,
                  (D σ τ *
                    (∑ σs : ReplicaSpace N 2,
                      pairEval N (std_basis N σ) σs * W₂ σs) *
                    (∑ ρs : ReplicaSpace N 2,
                      pairEval N (std_basis N τ) ρs * W₂ ρs)) *
                    ((∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹) ^ 2 := by
              apply Finset.sum_congr rfl
              intro σ _
              apply Finset.sum_congr rfl
              intro τ _
              ring
            _ = ∑ σ : Config N,
                  (∑ τ : Config N,
                    D σ τ *
                      (∑ σs : ReplicaSpace N 2,
                        pairEval N (std_basis N σ) σs * W₂ σs) *
                      (∑ ρs : ReplicaSpace N 2,
                        pairEval N (std_basis N τ) ρs * W₂ ρs)) *
                    ((∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹) ^ 2 := by
              apply Finset.sum_congr rfl
              intro σ _
              rw [Finset.sum_mul]
            _ = (∑ σ : Config N, ∑ τ : Config N,
                  D σ τ *
                    (∑ σs : ReplicaSpace N 2,
                      pairEval N (std_basis N σ) σs * W₂ σs) *
                    (∑ ρs : ReplicaSpace N 2,
                      pairEval N (std_basis N τ) ρs * W₂ ρs)) *
                  ((∑ σs : ReplicaSpace N 2, W₂ σs)⁻¹) ^ 2 := by
              rw [Finset.sum_mul]
        _ = (∑ σs : ReplicaSpace N 4,
              (4 * (crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs + c)) *
                W₄ σs) /
              (∑ σs : ReplicaSpace N 2, W₂ σs) ^ 2 := by
          rw [hcross_num]
        _ = 4 * (coupledCrossBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling + c) := by
          rw [hcross]
          field_simp [hZ0]
          rw [← hweight_sq]
          simp_rw [show ∀ a b w : ℝ, 4 * (a + b) * w = 4 * a * w + 4 * b * w by
            intro a b w
            ring]
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]
          have hfactorCross : (∑ x : ReplicaSpace N 4,
              4 * crossPairBregman N β h q sk x * W₄ x) =
              4 * ∑ x : ReplicaSpace N 4, crossPairBregman N β h q sk x * W₄ x := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro x _
            ring
          rw [hfactorCross]
          ring

    have hcore :
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (fun σs => pairEval N (std_basis N σ) σs *
                  pairEval N (std_basis N τ) σs) -
              tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (pairEval N (std_basis N σ)) *
              tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (pairEval N (std_basis N τ)))) =
          2 * (bregmanRemainder sk.ξ β q 1 +
            tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling -
            2 * coupledCrossBregmanDet
              (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling) := by
      calc
        (∑ σ : Config N, ∑ τ : Config N,
          D σ τ *
            (tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (fun σs => pairEval N (std_basis N σ) σs *
                  pairEval N (std_basis N τ) σs) -
              tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (pairEval N (std_basis N σ)) *
              tiltedReplicaAverageDet (N := N) (q := q) H coupling
                (pairEval N (std_basis N τ)))) =
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                tiltedReplicaAverageDet (N := N) (q := q) H coupling
                  (fun σs => pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) σs)) -
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                tiltedReplicaAverageDet (N := N) (q := q) H coupling
                  (pairEval N (std_basis N σ)) *
                tiltedReplicaAverageDet (N := N) (q := q) H coupling
                  (pairEval N (std_basis N τ))) := by
          simp only [mul_sub, Finset.sum_sub_distrib, mul_assoc]
        _ = _ := by
          rw [hwithin, hbetween]
          ring

    simp_rw [covKernelDiff_eq_bregman_workspace
      (N := N) (β := β) (h := h) (q := q) (sk := sk)]
    unfold coupledHessianDet

    have hfactor :
        (∑ σ : Config N, ∑ τ : Config N,
          ((N : ℝ) *
            (bregmanRemainder sk.ξ β q (overlap N σ τ) + (sk.ξ q - β * q))) *
            ((1 / (2 * (N : ℝ))) *
              (tiltedReplicaAverageDet (N := N) (q := q) H coupling
                  (fun σs => pairEval N (std_basis N σ) σs *
                    pairEval N (std_basis N τ) σs) -
                tiltedReplicaAverageDet (N := N) (q := q) H coupling
                    (pairEval N (std_basis N σ)) *
                  tiltedReplicaAverageDet (N := N) (q := q) H coupling
                    (pairEval N (std_basis N τ))))) =
          ((N : ℝ) * (1 / (2 * (N : ℝ)))) *
            (∑ σ : Config N, ∑ τ : Config N,
              D σ τ *
                (tiltedReplicaAverageDet (N := N) (q := q) H coupling
                    (fun σs => pairEval N (std_basis N σ) σs *
                      pairEval N (std_basis N τ) σs) -
                  tiltedReplicaAverageDet (N := N) (q := q) H coupling
                      (pairEval N (std_basis N σ)) *
                    tiltedReplicaAverageDet (N := N) (q := q) H coupling
                      (pairEval N (std_basis N τ)))) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro σ _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro τ _
      dsimp only [D]
      ring

    rw [hfactor, hcore]
    field_simp [hN0]


end GeneralizedLatala
end SpinGlass
