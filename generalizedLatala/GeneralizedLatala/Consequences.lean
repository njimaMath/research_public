import GeneralizedLatala.Interpolation.QuadraticCoupling

/-!
# Overlap and pressure consequences

Overlap concentration and the replica-symmetric pressure sum rule obtained from the quadratic estimate.

Main declarations:
- `overlap_concentration_uniform`
- `replica_symmetric_sum_rule`

Dependencies:
- the uniform quadratic-coupling estimate and ordinary pressure interpolation

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

/-! ## Consequences -/

/-
Integrated finite-volume Jensen inequality for the centered overlap square.
-/
lemma scaled_overlapVariance_le_logQuadraticMoment
    (coupling : ℝ) (hcoupling : 0 ≤ coupling) (t : ℝ) :
    coupling * (N : ℝ) *
        overlapVariance
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t
      ≤ logQuadraticMoment
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
          t coupling := by
  refine' trans _ ( MeasureTheory.integral_mono_of_nonneg _ _ _ );
  case refine'_2 => exact fun ω => coupling * N * gibbs_average_n_det N 2 ( H_t N β h q sk sim t ω ) ( centeredOverlapSq N q );
  · rw [ MeasureTheory.integral_const_mul ] ; rfl;
  · refine' Filter.Eventually.of_forall fun ω => mul_nonneg ( mul_nonneg hcoupling ( Nat.cast_nonneg _ ) ) _;
    refine' Finset.sum_nonneg fun σs _ => mul_nonneg _ _;
    · exact sq_nonneg _;
    · exact Finset.prod_nonneg fun _ _ => div_nonneg ( Real.exp_nonneg _ ) ( Z_pos _ _ |> le_of_lt );
  · have h_integrable : Integrable (fun ω => gibbs_average_n N β h q sk sim 2 t (fun σs => Real.exp (coupling * N * centeredOverlapSq N q σs)) ω) ℙ := by
      apply SpinGlass.integrable_gibbs_average_n;
    refine' h_integrable.mono' _ _;
    · exact Real.measurable_log.comp_aemeasurable h_integrable.aemeasurable |> fun h => h.aestronglyMeasurable;
    · filter_upwards [ ] with ω;
      rw [ Real.norm_eq_abs, abs_of_nonneg ( Real.log_nonneg _ ) ];
      · refine' le_trans ( Real.log_le_sub_one_of_pos _ ) _;
        · refine' Finset.sum_pos _ _;
          · intro σs _;
            refine' mul_pos ( Real.exp_pos _ ) ( Finset.prod_pos fun l _ => _ );
            exact div_pos ( Real.exp_pos _ ) ( Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) ( Finset.univ_nonempty ) );
          · exact ⟨ fun _ => fun _ => Bool.true, Finset.mem_univ _ ⟩;
        · linarith;
      · have h_gibbs_exp : ∀ H : EnergySpace N, gibbs_average_n_det N 2 H (fun σs => Real.exp (coupling * N * centeredOverlapSq N q σs)) ≥ 1 := by
          intro H
          have h_gibbs_exp : gibbs_average_n_det N 2 H (fun σs => Real.exp (coupling * N * centeredOverlapSq N q σs)) ≥ Real.exp (gibbs_average_n_det N 2 H (fun σs => coupling * N * centeredOverlapSq N q σs)) := by
            apply gibbs_average_n_det_exp_jensen;
          refine' le_trans _ h_gibbs_exp;
          refine' Real.one_le_exp _;
          refine' Finset.sum_nonneg fun σs _ => mul_nonneg _ _;
          · exact mul_nonneg ( mul_nonneg hcoupling ( Nat.cast_nonneg _ ) ) ( sq_nonneg _ );
          · exact Finset.prod_nonneg fun _ _ => gibbs_pmf_nonneg N H _;
        exact h_gibbs_exp _;
  · filter_upwards [ ] with ω using scaled_centeredOverlapSq_le_log_gibbs_exp _ _ _ _

/-- Convexity of the log moment converts the quadratic exponential estimate into an overlap
second-moment estimate, uniformly along the smart path. -/
theorem overlap_concentration_uniform
    (hN : 0 < N) (hβ0 : 0 ≤ β) (hΓ0 : 0 ≤ Γ)
    (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hfp : IsRSFixedPoint β h q)
    (hΔ : BregmanBounds sk.ξ β q Γ) (hρ : rho Γ q < 1)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    {t : ℝ} (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    overlapVariance
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t
      ≤ quadraticConstant Γ q / (lambdaStar Γ q * (N : ℝ)) := by
  have hlambda : 0 < lambdaStar Γ q :=
    lambdaStar_pos (Γ := Γ) (q := q) hq0 hq1 hρ
  have hNreal : (0 : ℝ) < N := by
    exact_mod_cast hN
  have hlambdaN : 0 < lambdaStar Γ q * (N : ℝ) := mul_pos hlambda hNreal
  have hJensen :=
    scaled_overlapVariance_le_logQuadraticMoment
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      (coupling := lambdaStar Γ q) (le_of_lt hlambda) t
  have hquadratic :=
    uniform_quadratic_coupling
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hβ0 hΓ0 hq0 hq1 hfp hΔ hρ hIndep ht
  apply (le_div_iff₀ hlambdaN).2
  simpa [mul_assoc, mul_comm, mul_left_comm] using hJensen.trans hquadratic


private lemma overlapVariance_continuous : Continuous (overlapVariance
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)) := by
  let f : ReplicaFun N 2 := centeredOverlapSq N q
  let B : ℝ := ∑ σs : ReplicaSpace N 2, ‖f σs‖
  rw [continuous_iff_continuousAt]
  intro t
  apply MeasureTheory.continuousAt_of_dominated
  · filter_upwards with s
    exact (integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      (n := 2) (t := s) (f := f)).aestronglyMeasurable
  · filter_upwards with s
    filter_upwards with w
    simpa [B, Real.norm_eq_abs] using
      (abs_gibbs_average_n_le
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        (n := 2) (t := s) (f := f) w)
  · exact integrable_const B
  · filter_upwards with w
    have hHt : Continuous (fun t =>
        H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) := by
      simp only [H_t, H_gauss]
      fun_prop
    have hg : Continuous (fun H : EnergySpace N =>
        gibbs_average_n_det (N := N) (n := 2) H f) := by
      simp only [gibbs_average_n_det]
      apply continuous_finset_sum
      intro σs _
      apply Continuous.mul continuous_const
      apply continuous_finset_prod
      intro l _
      exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous
    exact (hg.comp hHt).continuousAt

private lemma bregmanAverage_continuous : Continuous (bregmanAverage
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)) := by
  let f : ReplicaFun N 2 := bregmanOverlap N β h q sk
  let B : ℝ := ∑ σs : ReplicaSpace N 2, ‖f σs‖
  rw [continuous_iff_continuousAt]
  intro t
  apply MeasureTheory.continuousAt_of_dominated
  · filter_upwards with s
    exact (integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      (n := 2) (t := s) (f := f)).aestronglyMeasurable
  · filter_upwards with s
    filter_upwards with w
    simpa [B, Real.norm_eq_abs] using
      (abs_gibbs_average_n_le
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        (n := 2) (t := s) (f := f) w)
  · exact integrable_const B
  · filter_upwards with w
    have hHt : Continuous (fun t =>
        H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) := by
      simp only [H_t, H_gauss]
      fun_prop
    have hg : Continuous (fun H : EnergySpace N =>
        gibbs_average_n_det (N := N) (n := 2) H f) := by
      simp only [gibbs_average_n_det]
      apply continuous_finset_sum
      intro σs _
      apply Continuous.mul continuous_const
      apply continuous_finset_prod
      intro l _
      exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous
    exact (hg.comp hHt).continuousAt

private lemma free_energy_siteEnergy_eq (N : ℕ) (a : Fin N → ℝ) :
    free_energy_density (N := N) (siteEnergy N a) =
      (1 / (N : ℝ)) * ∑ i : Fin N, (Real.log 2 + Real.log (Real.cosh (a i))) := by
  rw [free_energy_density, Z_siteEnergy]
  rw [Real.log_prod]
  · congr 1
    apply Finset.sum_congr rfl
    intro i _
    rw [show (∑ b : Bool, Real.exp (-(a i * boolSpin b))) =
        2 * Real.cosh (a i) by
      simp [boolSpin, Real.cosh_eq]
      ring]
    rw [Real.log_mul]
    · norm_num
    · exact ne_of_gt (Real.cosh_pos _)
  · intro i _
    exact ne_of_gt (Finset.sum_pos (fun b _ => Real.exp_pos _) Finset.univ_nonempty)

private lemma integrable_log_cosh_affine (h a : ℝ) : Integrable
    (fun z => Real.log (Real.cosh (h + a * z))) (gaussianReal 0 1) := by
  have hplus : Integrable (fun z => Real.exp (h + a * z)) (gaussianReal 0 1) := by
    simpa [Real.exp_add] using
      (ProbabilityTheory.integrable_exp_mul_gaussianReal (μ := 0) (v := 1) a).const_mul
        (Real.exp h)
  have hminus : Integrable (fun z => Real.exp (-(h + a * z))) (gaussianReal 0 1) := by
    have hi :=
      (ProbabilityTheory.integrable_exp_mul_gaussianReal (μ := 0) (v := 1) (-a)).const_mul
        (Real.exp (-h))
    simpa [Real.exp_add, mul_comm] using hi
  have hbound : Integrable
      (fun z => Real.exp (h + a * z) + Real.exp (-(h + a * z)))
      (gaussianReal 0 1) := hplus.add hminus
  apply hbound.mono'
  · have hc : Continuous (fun z => Real.cosh (h + a * z)) := by fun_prop
    exact (hc.log (fun z => ne_of_gt (Real.cosh_pos _))).aestronglyMeasurable
  · filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (Real.log_nonneg (Real.one_le_cosh _))]
    calc
      Real.log (Real.cosh (h + a * z))
          ≤ Real.cosh (h + a * z) - 1 :=
        Real.log_le_sub_one_of_pos (Real.cosh_pos _)
      _ ≤ Real.exp (h + a * z) + Real.exp (-(h + a * z)) := by
        rw [Real.cosh_eq]
        nlinarith [Real.exp_pos (h + a * z), Real.exp_pos (-(h + a * z))]

private lemma endpoint_pressure
    (hN : 0 < N) (hβ0 : 0 ≤ β) :
    interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 =
      Real.log 2 + standardGaussianExpectation
        (fun z => Real.log (Real.cosh (h + Real.sqrt β * z))) := by
  letI : IsProbabilityMeasure (gaussianProduct N) := by
    rw [gaussianProduct]
    infer_instance
  let F : EnergySpace N → ℝ := fun H =>
    free_energy_density (N := N) (H + H_field (N := N) (h := h))
  have hFcont : Continuous F :=
    (SpinGlass.contDiff_free_energy_density (N := N)).continuous.comp
      (continuous_id.add continuous_const)
  have hHt0 (ω : Ω) :
      H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω =
        sim.V ω + H_field (N := N) (h := h) := by
    simp [H_t, H_gauss]
  have hrefLaw := referenceField_hasGaussianLaw N β q
  calc
    interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 =
        ∫ ω, F (sim.V ω) ∂ℙ := by
          rw [interpolatedPressure]
          apply integral_congr_ae
          filter_upwards with ω
          rw [hHt0]
    _ = ∫ H, F H ∂Measure.map sim.V ℙ := by
          rw [integral_map sim.hV.repr_measurable.aemeasurable
            hFcont.aestronglyMeasurable]
    _ = ∫ H, F H ∂Measure.map (referenceField N β q) (gaussianProduct N) := by
          rw [simpleDisorder_law_eq_reference N β q sim hN hβ0]
    _ = ∫ z, F (referenceField N β q z) ∂gaussianProduct N := by
          rw [integral_map hrefLaw.aemeasurable hFcont.aestronglyMeasurable]
    _ = Real.log 2 + standardGaussianExpectation
        (fun z => Real.log (Real.cosh (h + Real.sqrt β * z))) := by
      let g : ℝ → ℝ := fun z =>
        Real.log (Real.cosh (h + Real.sqrt β * z))
      have hg : Integrable g (gaussianReal 0 1) :=
        integrable_log_cosh_affine h (Real.sqrt β)
      have hcoord (i : Fin N) : Integrable (fun z : Fin N → ℝ => g (z i))
          (gaussianProduct N) := by
        exact ((measurePreserving_eval (fun _ : Fin N => gaussianReal 0 1) i).integrable_comp
          hg.aestronglyMeasurable).2 hg
      rw [show (∫ z, F (referenceField N β q z) ∂gaussianProduct N) =
          ∫ z, (1 / (N : ℝ)) * ∑ i : Fin N, (Real.log 2 + g (z i))
            ∂gaussianProduct N by
        apply integral_congr_ae
        filter_upwards with z
        simp only [F]
        change free_energy_density (N := N)
          (referenceField N β q z + magnetic_field_vector (N := N) h) = _
        rw [reference_add_field_eq_siteEnergy, free_energy_siteEnergy_eq]]
      rw [integral_const_mul]
      rw [show (∫ z : Fin N → ℝ, ∑ i : Fin N, (Real.log 2 + g (z i))
            ∂gaussianProduct N) =
          ∫ z : Fin N → ℝ, ((N : ℝ) * Real.log 2 + ∑ i : Fin N, g (z i))
            ∂gaussianProduct N by
        apply integral_congr_ae
        filter_upwards with z
        simp [Finset.sum_add_distrib]]
      rw [integral_add (integrable_const _)
        (integrable_finset_sum Finset.univ (fun i _ => hcoord i))]
      rw [integral_finset_sum Finset.univ (fun i _ => hcoord i)]
      simp only [integral_const, probReal_univ, one_smul]
      have hcoord_integral (i : Fin N) :
          (∫ z : Fin N → ℝ, g (z i) ∂gaussianProduct N) =
            ∫ z, g z ∂gaussianReal 0 1 :=
        integral_comp_eval hg.aestronglyMeasurable
      simp_rw [hcoord_integral]
      simp only [standardGaussianExpectation, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin]
      have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
      field_simp
      ring

private lemma interpolatedPressure_continuousOn :
    ContinuousOn
      (interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim))
      (Set.Icc (0 : ℝ) 1) := by
  let C : ℝ := (SpinGlass.hasModerateGrowth_free_energy_density N).C
  let B : Ω → ℝ := fun w => C *
    (1 + ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖)
  apply MeasureTheory.continuousOn_of_dominated
  · intro t _
    have hHt_meas : Measurable
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t) := by
      have hU := sk.hU.repr_measurable.const_smul (Real.sqrt t)
      have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
      exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t
    exact ((SpinGlass.contDiff_free_energy_density (N := N)).continuous.measurable.comp
      hHt_meas).aestronglyMeasurable
  · intro t ht
    filter_upwards with w
    have hsqrtt0 : 0 ≤ Real.sqrt t := Real.sqrt_nonneg _
    have hsqrtt1 : Real.sqrt t ≤ 1 := Real.sqrt_le_one.mpr ht.2
    have hsqrt1t0 : 0 ≤ Real.sqrt (1 - t) := Real.sqrt_nonneg _
    have hsqrt1t1 : Real.sqrt (1 - t) ≤ 1 := Real.sqrt_le_one.mpr (by linarith [ht.1])
    have hnorm : ‖H_t
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w‖ ≤
        ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖ := by
      calc
        ‖H_t (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t w‖
            ≤ ‖(Real.sqrt t) • sk.U w‖ + ‖(Real.sqrt (1 - t)) • sim.V w‖ +
                ‖H_field (N := N) (h := h)‖ := by
              simp only [H_t, H_gauss]
              exact (norm_add_le
                ((Real.sqrt t) • sk.U w + (Real.sqrt (1 - t)) • sim.V w)
                (H_field (N := N) (h := h))).trans
                (by
                  gcongr
                  exact norm_add_le ((Real.sqrt t) • sk.U w)
                    ((Real.sqrt (1 - t)) • sim.V w))
        _ ≤ ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖ := by
              rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
                abs_of_nonneg hsqrtt0, abs_of_nonneg hsqrt1t0]
              gcongr
              · exact mul_le_of_le_one_left (norm_nonneg _) hsqrtt1
              · exact mul_le_of_le_one_left (norm_nonneg _) hsqrt1t1
    have hgrowth :=
      (SpinGlass.hasModerateGrowth_free_energy_density N).F_bound
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w)
    have hm : (SpinGlass.hasModerateGrowth_free_energy_density N).m = 1 := by rfl
    rw [hm, pow_one] at hgrowth
    change |free_energy_density (N := N)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w)| ≤
      C * (1 + ‖H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w‖) at hgrowth
    have hinside :
        1 + ‖H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w‖ ≤
        1 + ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖ := by
      linarith
    have hmul := mul_le_mul_of_nonneg_left hinside
      (le_of_lt (SpinGlass.hasModerateGrowth_free_energy_density N).Cpos)
    rw [Real.norm_eq_abs]
    exact hgrowth.trans (by simpa only [C] using hmul)
  · apply Integrable.const_mul
    exact (((integrable_const (1 : ℝ)).add
      (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sk.U) sk.hU)).add
      (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sim.V) sim.hV)).add
        (integrable_const _)
  · filter_upwards with w
    have hHt : Continuous (fun t =>
        H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) := by
      simp only [H_t, H_gauss]
      fun_prop
    exact ((SpinGlass.contDiff_free_energy_density (N := N)).continuous.comp hHt).continuousOn

/-- Integrated Guerra sum rule, including evaluation of the independent endpoint. -/
lemma replica_symmetric_sum_rule
    (hN : 0 < N) (hβ0 : 0 ≤ β)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω)) :
    MeasureTheory.IntegrableOn
        (bregmanAverage
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim))
        (Set.Icc (0 : ℝ) 1) (MeasureTheory.volume : Measure ℝ) ∧
      rsPressure sk.ξ β h q -
          interpolatedPressure
            (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1
        = (1 / 2) *
            ∫ t in Set.Icc (0 : ℝ) 1,
              bregmanAverage
                (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  let P : ℝ → ℝ := interpolatedPressure
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
  let v : ℝ → ℝ := bregmanAverage
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
  let g : ℝ → ℝ := fun t => (1 / 2) * (bregmanRemainder sk.ξ β q 1 - v t)
  have hvcont : Continuous v := bregmanAverage_continuous
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
  have hvint : IntegrableOn v (Set.Icc (0 : ℝ) 1) := hvcont.integrableOn_Icc
  have hgcont : Continuous g := by
    dsimp only [g]
    fun_prop
  have hPcont : ContinuousOn P (Set.Icc (0 : ℝ) 1) :=
    interpolatedPressure_continuousOn
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
  have hderiv : ∀ t ∈ Set.Ioo (0 : ℝ) 1, HasDerivAt P (g t) t := by
    intro t ht
    exact pressure_derivative
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hIndep ht
  have hFTC : (∫ t in (0 : ℝ)..1, g t) = P 1 - P 0 := by
    exact intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le zero_le_one hPcont
      hderiv (hgcont.intervalIntegrable 0 1)
  have hinterval :
      (∫ t in (0 : ℝ)..1, g t) =
        (1 / 2) * (bregmanRemainder sk.ξ β q 1 - ∫ t in (0 : ℝ)..1, v t) := by
    simp only [g]
    rw [intervalIntegral.integral_const_mul]
    rw [intervalIntegral.integral_sub
      (intervalIntegrable_const :
        IntervalIntegrable (fun _ : ℝ => bregmanRemainder sk.ξ β q 1) volume 0 1)
      (hvcont.intervalIntegrable 0 1)]
    norm_num
  have hset :
      (∫ t in Set.Icc (0 : ℝ) 1, v t) = ∫ t in (0 : ℝ)..1, v t := by
    rw [MeasureTheory.integral_Icc_eq_integral_Ioc,
      intervalIntegral.integral_of_le zero_le_one]
  have hP0 : P 0 = Real.log 2 + standardGaussianExpectation
      (fun z => Real.log (Real.cosh (h + Real.sqrt β * z))) :=
    endpoint_pressure
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) hN hβ0
  have hrel : P 1 - P 0 =
      (1 / 2) *
        (bregmanRemainder sk.ξ β q 1 - ∫ t in (0 : ℝ)..1, v t) :=
    hFTC.symm.trans hinterval
  refine ⟨hvint, ?_⟩
  rw [rsPressure, show interpolatedPressure
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 1 = P 1 by rfl,
    hset, ← hP0]
  linear_combination -hrel


end GeneralizedLatala
end SpinGlass
