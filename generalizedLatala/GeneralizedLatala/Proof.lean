import GeneralizedLatala.Consequences

/-!
# Generalized Latala theorem

Final assembly of the finite-volume overlap and replica-symmetric pressure estimates.

Main declarations:
- `generalized_latala`

Dependencies:
- overlap concentration and the replica-symmetric pressure sum rule

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

/-- Generalized Latała bound for a finite-volume convex mixed p-spin model.

At `t = 1`, `H_t` is the mixed disorder plus the external-field vector.  The theorem gives both
the `O(1/N)` centered-overlap estimate and the corresponding replica-symmetric pressure error.
-/
theorem generalized_latala
    (hN : 0 < N) (hβ0 : 0 ≤ β) (hΓ0 : 0 ≤ Γ)
    (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hfp : IsRSFixedPoint β h q)
    (hΔ : BregmanBounds sk.ξ β q Γ)
    (hρ : rho Γ q < 1)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω)) :
    overlapVariance
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1
        ≤ quadraticConstant Γ q / (lambdaStar Γ q * (N : ℝ)) ∧
      0 ≤ rsPressure sk.ξ β h q -
        interpolatedPressure
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1 ∧
      rsPressure sk.ξ β h q -
        interpolatedPressure
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1
        ≤ (Γ * quadraticConstant Γ q) /
            (4 * lambdaStar Γ q * (N : ℝ)) := by
  let C : ℝ := quadraticConstant Γ q / (lambdaStar Γ q * (N : ℝ))
  have hoverlap :=
    overlap_concentration_uniform
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hβ0 hΓ0 hq0 hq1 hfp hΔ hρ hIndep (t := (1 : ℝ)) (by simp)
  have hsum :=
    replica_symmetric_sum_rule
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hβ0 hIndep
  have hbreg0 : ∀ t : ℝ, 0 ≤ bregmanAverage
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t :=
    fun t => bregmanAverage_nonneg
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hΔ t
  have hint0 : 0 ≤ ∫ t in Set.Icc (0 : ℝ) 1,
      bregmanAverage
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t :=
    integral_nonneg hbreg0
  have hpressure0 : 0 ≤ rsPressure sk.ξ β h q -
      interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1 := by
    rw [hsum.2]
    exact mul_nonneg (by norm_num) hint0
  have hbound : ∀ t ∈ Set.Icc (0 : ℝ) 1,
      bregmanAverage
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t
        ≤ (Γ / 2) * C := by
    intro t ht
    calc
      bregmanAverage N β h q sk sim t
          ≤ (Γ / 2) * overlapVariance N β h q sk sim t :=
        bregmanAverage_le
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
          hN hΔ t
      _ ≤ (Γ / 2) * C := by
        apply mul_le_mul_of_nonneg_left _ (div_nonneg hΓ0 zero_le_two)
        dsimp only [C]
        exact overlap_concentration_uniform
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
          hN hβ0 hΓ0 hq0 hq1 hfp hΔ hρ hIndep ht
  have hconstInt : MeasureTheory.IntegrableOn
      (fun _ : ℝ => (Γ / 2) * C) (Set.Icc (0 : ℝ) 1)
      (MeasureTheory.volume : Measure ℝ) :=
    MeasureTheory.integrableOn_const (hs := by
      rw [Real.volume_Icc]
      finiteness)
  have hint_le :
      (∫ t in Set.Icc (0 : ℝ) 1,
          bregmanAverage
            (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t)
        ≤ (Γ / 2) * C := by
    calc
      (∫ t in Set.Icc (0 : ℝ) 1,
          bregmanAverage
            (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t)
          ≤ ∫ _t in Set.Icc (0 : ℝ) 1, (Γ / 2) * C := by
              exact integral_mono_ae hsum.1 hconstInt
                (ae_restrict_of_forall_mem measurableSet_Icc hbound)
      _ = (Γ / 2) * C := by
        norm_num [MeasureTheory.integral_const, Measure.restrict_apply_univ, Real.volume_Icc]
  refine ⟨hoverlap, hpressure0, ?_⟩
  rw [hsum.2]
  calc
    (1 / 2) *
          ∫ t in Set.Icc (0 : ℝ) 1,
            bregmanAverage
              (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t
        ≤ (1 / 2) * ((Γ / 2) * C) :=
          mul_le_mul_of_nonneg_left hint_le (by norm_num)
    _ = (Γ * quadraticConstant Γ q) /
          (4 * lambdaStar Γ q * (N : ℝ)) := by
      simp only [C]
      ring


end GeneralizedLatala
end SpinGlass
