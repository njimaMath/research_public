import GeneralizedLatala.Endpoint.Estimates
import GeneralizedLatala.Endpoint.GaussianAffineIBP

/-!
# Pressure interpolation

Differentiation and Gaussian integration by parts for the ordinary smart-path pressure.

Main declarations:
- `pressure_derivative_before_ibp`
- `pressure_derivative_ibp`
- `pressure_derivative`

Dependencies:
- independent-endpoint estimates and affine Gaussian integration by parts

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

/-! ## Gaussian interpolation and quadratic coupling -/

/-- Differentiation of the smart-path pressure before Gaussian integration by parts. -/
lemma pressure_derivative_before_ibp
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt
      (interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim))
      (∫ w,
        fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
          (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
        ∂ℙ) t := by
  classical
  have ht0 : 0 < t := ht.1
  have ht1 : t < 1 := ht.2
  have h1t0 : 0 < 1 - t := by linarith
  let ε : ℝ := (min t (1 - t)) / 2
  have hε_pos : 0 < ε := by
    have hmin : 0 < min t (1 - t) := lt_min ht0 h1t0
    have : 0 < (min t (1 - t)) / 2 := by linarith
    simpa [ε] using this
  have hball_Ioo : ∀ x ∈ Metric.ball t ε, x ∈ Set.Ioo (0 : ℝ) 1 := by
    intro x hx
    have hx' : |x - t| < ε := by
      simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm, ε] using hx
    have hx1 : x - t < ε := (abs_sub_lt_iff.1 hx').1
    have hx2 : t - x < ε := (abs_sub_lt_iff.1 hx').2
    have hε_le_t : ε ≤ t / 2 := by
      have : min t (1 - t) ≤ t := min_le_left _ _
      have : (min t (1 - t)) / 2 ≤ t / 2 := by nlinarith
      simpa [ε] using this
    have hε_le_1t : ε ≤ (1 - t) / 2 := by
      have : min t (1 - t) ≤ (1 - t) := min_le_right _ _
      have : (min t (1 - t)) / 2 ≤ (1 - t) / 2 := by nlinarith
      simpa [ε] using this
    have hx_lower : t / 2 < x := by
      have ht_eps : t / 2 ≤ t - ε := by nlinarith [hε_le_t]
      have hx_gt : t - ε < x := by linarith
      exact lt_of_le_of_lt ht_eps hx_gt
    have hx_gt0 : 0 < x := by
      have ht_eps : t - ε ≥ t / 2 := by nlinarith [hε_le_t]
      have hx_gt : t - ε < x := by linarith
      have : t / 2 < x := lt_of_le_of_lt ht_eps hx_gt
      have : 0 < t / 2 := by nlinarith [ht0]
      exact Std.lt_trans this hx_lower
    have hx_lt1 : x < 1 := by
      have hx_lt : x < t + ε := by linarith
      have ht_eps : t + ε ≤ (1 + t) / 2 := by nlinarith [hε_le_1t]
      have : x < (1 + t) / 2 := lt_of_lt_of_le hx_lt ht_eps
      have : (1 + t) / 2 < 1 := by nlinarith [ht1]
      simp; grind
    exact ⟨hx_gt0, hx_lt1⟩
  let F : ℝ → Ω → ℝ :=
    fun s w => free_energy_density (N := N) (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s w)
  let F' : ℝ → Ω → ℝ :=
    fun s w =>
      fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s w)
        (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s w)
  have hF_meas : ∀ᶠ s in nhds t, AEStronglyMeasurable (F s) (ℙ : Measure Ω) := by
    refine Filter.Eventually.of_forall (fun s => ?_)
    have hH_meas : Measurable (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s) := by
      have hU := sk.hU.repr_measurable.const_smul (Real.sqrt s)
      have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - s))
      exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) s
    exact ((contDiff_free_energy_density (N := N)).continuous.measurable.comp
      hH_meas).aestronglyMeasurable
  have hF_int : Integrable (F t) (ℙ : Measure Ω) := by
    let C : ℝ := (SpinGlass.hasModerateGrowth_free_energy_density N).C
    have hH_meas : Measurable
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t) := by
      have hU := sk.hU.repr_measurable.const_smul (Real.sqrt t)
      have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
      exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t
    have hF_meas : AEStronglyMeasurable (F t) (ℙ : Measure Ω) :=
      ((contDiff_free_energy_density (N := N)).continuous.measurable.comp
        hH_meas).aestronglyMeasurable
    let boundFun : Ω → ℝ := fun w => C * (1 + ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖)
    have hbound_int : Integrable boundFun (ℙ : Measure Ω) := by
      apply Integrable.const_mul
      exact (((integrable_const (1 : ℝ)).add
        (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sk.U) sk.hU)).add
          (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sim.V) sim.hV)).add
            (integrable_const _)
    refine MeasureTheory.Integrable.mono' hbound_int hF_meas ?_
    have hsqrtt0 : 0 ≤ Real.sqrt t := Real.sqrt_nonneg _
    have hsqrtt1 : Real.sqrt t ≤ 1 := Real.sqrt_le_one.mpr (le_of_lt ht1)
    have hsqrt1t0 : 0 ≤ Real.sqrt (1 - t) := Real.sqrt_nonneg _
    have hsqrt1t1 : Real.sqrt (1 - t) ≤ 1 := Real.sqrt_le_one.mpr (by linarith [ht0])
    filter_upwards with w
    have hnorm : ‖H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w‖ ≤
        ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖ := by
      calc
        ‖H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w‖
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
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
    have hm : (SpinGlass.hasModerateGrowth_free_energy_density N).m = 1 := by rfl
    rw [hm, pow_one] at hgrowth
    rw [Real.norm_eq_abs]
    have hinside : 1 + ‖H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w‖ ≤
        1 + ‖sk.U w‖ + ‖sim.V w‖ + ‖H_field (N := N) (h := h)‖ := by linarith
    have hmul := mul_le_mul_of_nonneg_left hinside
      (le_of_lt (SpinGlass.hasModerateGrowth_free_energy_density N).Cpos)
    exact hgrowth.trans (by simpa only [C] using hmul)
  -- Define the bound
  let Cf : ℝ := 1 / (N : ℝ)
  let cU : ℝ := 1 / (2 * Real.sqrt (t / 2))
  let cV : ℝ := 1 / (2 * Real.sqrt ((1 - t) / 2))
  let bound : Ω → ℝ := fun w => Cf * (cU * ‖sk.U w‖ + cV * ‖sim.V w‖)
  have hCf_nonneg : 0 ≤ Cf := by positivity
  have hcU_nonneg : 0 ≤ cU := by positivity
  have hcV_nonneg : 0 ≤ cV := by positivity
  have hbound_int : Integrable bound (ℙ : Measure Ω) := by
    have hU_int : Integrable (fun w => ‖sk.U w‖) (ℙ : Measure Ω) :=
      (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sk.U) sk.hU)
    have hV_int : Integrable (fun w => ‖sim.V w‖) (ℙ : Measure Ω) :=
      (PhysLean.Probability.GaussianIBP.integrable_norm_of_gaussian (g := sim.V) sim.hV)
    have h1 : Integrable (fun w => cU * ‖sk.U w‖) (ℙ : Measure Ω) := (hU_int.const_mul cU)
    have h2 : Integrable (fun w => cV * ‖sim.V w‖) (ℙ : Measure Ω) := (hV_int.const_mul cV)
    have hsum : Integrable (fun w => cU * ‖sk.U w‖ + cV * ‖sim.V w‖) (ℙ : Measure Ω) := h1.add h2
    simpa [bound, Cf, mul_add, mul_assoc] using hsum.const_mul Cf
  have hF'_meas : AEStronglyMeasurable (F' t) (ℙ : Measure Ω) := by
    have hdH_meas : Measurable (fun w => dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) := by
      simp only [dH_t]
      have hU := sk.hU.repr_measurable.const_smul ((1 : ℝ) / (2 * Real.sqrt t))
      have hV := sim.hV.repr_measurable.const_smul ((1 : ℝ) / (2 * Real.sqrt (1 - t)))
      exact measurable_dH_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t
    have hHM : Measurable (fun w => H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) := by
      have hU := sk.hU.repr_measurable.const_smul (Real.sqrt t)
      have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
      exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
        (sk := sk) (sim := sim) t
    have hfderiv_cont : Continuous (fun p : EnergySpace N × EnergySpace N =>
        fderiv ℝ (fun H => free_energy_density (N := N) H) p.1 p.2) := by
      have hcd := contDiff_free_energy_density (N := N)
      have hfderiv_cont' : Continuous (fun H => fderiv ℝ (fun H => free_energy_density (N := N) H) H) :=
        hcd.continuous_fderiv (by simp)
      exact ((hfderiv_cont'.comp continuous_fst).clm_apply continuous_snd)
    have hpair : Measurable (fun w => (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w,
        dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)) :=
      hHM.prodMk hdH_meas
    exact (hfderiv_cont.measurable.comp hpair).aestronglyMeasurable
  have h_bound :
      ∀ᵐ w ∂(ℙ : Measure Ω), ∀ x ∈ Metric.ball t ε, ‖F' x w‖ ≤ bound w := by
    refine ae_of_all _ (fun w => ?_)
    intro x hx
    have hxIoo : x ∈ Set.Ioo (0 : ℝ) 1 := hball_Ioo x hx
    -- Bound the operator norm of the derivative of free_energy_density
    have h_op :
        ‖fderiv ℝ (fun H' => free_energy_density (N := N) H')
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)‖ ≤ (1 / (N : ℝ)) := by
      refine ContinuousLinearMap.opNorm_le_bound _ hCf_nonneg ?_
      intro v
      have h_eval :
          (fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)) v =
            -(1 / (N : ℝ)) * ∑ σ : Config N, (gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * v σ :=
        fderiv_free_energy_density_apply (N := N)
          (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) (h := v)
      have hs1 : (∑ σ : Config N, gibbs_pmf N
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) = 1 :=
        sum_gibbs_pmf (N := N)
          (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)
      have hsum_bound :
          |∑ σ : Config N, gibbs_pmf N
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ| ≤ ‖v‖ := by
        have h_abs_le :
            |∑ σ : Config N, gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ|
              ≤ ∑ σ : Config N, |gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ| := by
          simpa using
            (Finset.abs_sum_le_sum_abs
              (f := fun σ : Config N => gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ)
              (s := (Finset.univ : Finset (Config N))))
        have h_abs_term :
            (∑ σ : Config N, |gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ|)
              = ∑ σ : Config N, (gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * |v σ| := by
          refine Finset.sum_congr rfl ?_
          intro σ _hσ
          have hg : 0 ≤ gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ :=
            gibbs_pmf_nonneg (N := N)
              (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ
          simp [abs_mul, abs_of_nonneg hg]
        have hsum_le :
            (∑ σ : Config N, (gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * |v σ|)
              ≤ (∑ σ : Config N, gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * ‖v‖ := by
          have hterm : ∀ σ : Config N, (gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * |v σ|
                ≤ (gibbs_pmf N
                  (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * ‖v‖ := by
            intro σ
            have hσ : |v σ| ≤ ‖v‖ := (abs_apply_le_norm (N := N) v σ)
            exact mul_le_mul_of_nonneg_left hσ (gibbs_pmf_nonneg (N := N)
              (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ)
          have hsum' :=
            (Finset.sum_le_sum (s := (Finset.univ : Finset (Config N)))
              (fun σ _ => hterm σ))
          have hfactor :
              (∑ σ : Config N, (gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * ‖v‖)
                = (∑ σ : Config N, gibbs_pmf N
                  (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * ‖v‖ := by
            simpa using
              (Finset.sum_mul (s := (Finset.univ : Finset (Config N)))
                (f := fun σ : Config N => gibbs_pmf N
                  (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ)
                (a := ‖v‖)).symm
          simpa [hfactor] using hsum'
        calc
          |∑ σ : Config N, gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ|
            ≤ ∑ σ : Config N, |gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ| := h_abs_le
          _ = ∑ σ : Config N, gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * |v σ| := h_abs_term
          _ ≤ (∑ σ : Config N, gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ) * ‖v‖ := hsum_le
          _ = ‖v‖ := by simp [hs1]
      have : ‖(fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)) v‖
          ≤ (1 / (N : ℝ)) * ‖v‖ := by
        have :
            ‖(fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)) v‖
              = (1 / (N : ℝ)) * |∑ σ : Config N, gibbs_pmf N
                (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ| := by
          simp [h_eval, Real.norm_eq_abs]
        calc
          ‖(fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)) v‖
          = (1 / (N : ℝ)) * |∑ σ : Config N, gibbs_pmf N
              (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) σ * v σ| := this
          _ ≤ (1 / (N : ℝ)) * ‖v‖ := by
                exact mul_le_mul_of_nonneg_left hsum_bound hCf_nonneg
      simpa [mul_assoc, mul_comm, mul_left_comm] using this
    have hL :
        ‖fderiv ℝ (fun H' => free_energy_density (N := N) H')
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)‖ ≤ Cf := by
      simpa [Cf] using h_op
    -- Bound the coefficients
    have hCoeffU :
        |1 / (2 * Real.sqrt x)| ≤ cU := by
      have hx_gt0 : 0 < x := hxIoo.1
      have hx_lower : t / 2 ≤ x := by
        have hx' : |x - t| < ε := by
          simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
        have hx2 : t - x < ε := (abs_sub_lt_iff.1 hx').2
        have hε_le_t : ε ≤ t / 2 := by
          have : min t (1 - t) ≤ t := min_le_left _ _
          have : (min t (1 - t)) / 2 ≤ t / 2 := by nlinarith
          simpa [ε] using this
        have hx_gt : t - ε < x := by linarith
        have ht_eps : t / 2 ≤ t - ε := by nlinarith [hε_le_t]
        exact le_trans ht_eps (le_of_lt hx_gt)
      have hsqrt_le : Real.sqrt (t / 2) ≤ Real.sqrt x := Real.sqrt_le_sqrt hx_lower
      have hpos : 0 < 2 * Real.sqrt (t / 2) := by
        have : 0 < Real.sqrt (t / 2) := by
          have : 0 < t / 2 := by nlinarith [ht0]
          exact Real.sqrt_pos.2 this
        nlinarith
      have hle :
          2 * Real.sqrt (t / 2) ≤ 2 * Real.sqrt x := by nlinarith [hsqrt_le]
      have : 1 / (2 * Real.sqrt x) ≤ 1 / (2 * Real.sqrt (t / 2)) := by
        simpa [one_div] using (one_div_le_one_div_of_le hpos hle)
      have hnonneg : 0 ≤ 1 / (2 * Real.sqrt x) := by positivity
      have hnonneg' : 0 ≤ 1 / (2 * Real.sqrt (t / 2)) := by positivity
      simpa [cU, abs_of_nonneg hnonneg, abs_of_nonneg hnonneg', abs_of_nonneg (Real.sqrt_nonneg x), one_div]
        using this
    have hCoeffV :
        |1 / (2 * Real.sqrt (1 - x))| ≤ cV := by
      have hx_lt1 : x < 1 := hxIoo.2
      have h1x_pos : 0 < 1 - x := by linarith
      have h1x_lower : (1 - t) / 2 ≤ 1 - x := by
        have hx' : |x - t| < ε := by
          simpa [Metric.mem_ball, Real.dist_eq, abs_sub_comm] using hx
        have hx1 : x - t < ε := (abs_sub_lt_iff.1 hx').1
        have hε_le_1t : ε ≤ (1 - t) / 2 := by
          have : min t (1 - t) ≤ (1 - t) := min_le_right _ _
          have : (min t (1 - t)) / 2 ≤ (1 - t) / 2 := by nlinarith
          simpa [ε] using this
        have hx_le : x ≤ t + (1 - t) / 2 := by
          have hx_le' : x ≤ t + ε := by linarith
          exact le_trans hx_le' (by nlinarith [hε_le_1t])
        nlinarith [hx_le]
      have hsqrt_le : Real.sqrt ((1 - t) / 2) ≤ Real.sqrt (1 - x) := Real.sqrt_le_sqrt h1x_lower
      have hpos : 0 < 2 * Real.sqrt ((1 - t) / 2) := by
        have : 0 < (1 - t) / 2 := by nlinarith [h1t0]
        have : 0 < Real.sqrt ((1 - t) / 2) := Real.sqrt_pos.2 this
        nlinarith
      have hle :
          2 * Real.sqrt ((1 - t) / 2) ≤ 2 * Real.sqrt (1 - x) := by nlinarith [hsqrt_le]
      have : 1 / (2 * Real.sqrt (1 - x)) ≤ 1 / (2 * Real.sqrt ((1 - t) / 2)) := by
        simpa [one_div] using (one_div_le_one_div_of_le hpos hle)
      have hnonneg : 0 ≤ 1 / (2 * Real.sqrt (1 - x)) := by positivity
      have hnonneg' : 0 ≤ 1 / (2 * Real.sqrt ((1 - t) / 2)) := by positivity
      simpa [cV, abs_of_nonneg hnonneg, abs_of_nonneg hnonneg',
        abs_of_nonneg (Real.sqrt_nonneg (1 - x)), one_div] using this
    -- Bound ‖dH_t x w‖
    have hdH_norm :
        ‖dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w‖
          ≤ cU * ‖sk.U w‖ + cV * ‖sim.V w‖ := by
      have htri :
          ‖dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w‖
            ≤ |1 / (2 * Real.sqrt x)| * ‖sk.U w‖ +
              |1 / (2 * Real.sqrt (1 - x))| * ‖sim.V w‖ := by
        simpa [dH_t, sub_eq_add_neg, norm_add_le, norm_smul, abs_mul] using
          (norm_add_le ((1 / (2 * Real.sqrt x)) • sk.U w) (-(1 / (2 * Real.sqrt (1 - x))) • sim.V w))
      have : |1 / (2 * Real.sqrt x)| * ‖sk.U w‖ +
            |1 / (2 * Real.sqrt (1 - x))| * ‖sim.V w‖
          ≤ cU * ‖sk.U w‖ + cV * ‖sim.V w‖ := by
        gcongr
      exact le_trans htri this
    -- Combine bounds
    have hF'_bound :
        ‖F' x w‖ ≤ Cf * ‖dH_t (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) x w‖ := by
      have hop : ‖(fderiv ℝ (fun H' => free_energy_density (N := N) H')
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w))
            (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)‖
          ≤ ‖fderiv ℝ (fun H' => free_energy_density (N := N) H')
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)‖ *
            ‖dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w‖ :=
        ContinuousLinearMap.le_opNorm _ _
      have hmul :
          ‖fderiv ℝ (fun H' => free_energy_density (N := N) H')
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w)‖ *
            ‖dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w‖
          ≤ Cf * ‖dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w‖ :=
        mul_le_mul_of_nonneg_right hL (norm_nonneg _)
      simpa [F'] using le_trans hop hmul
    have : ‖F' x w‖ ≤ bound w := by
      have : ‖F' x w‖ ≤ Cf * (cU * ‖sk.U w‖ + cV * ‖sim.V w‖) := by
        exact le_trans hF'_bound (mul_le_mul_of_nonneg_left hdH_norm (hCf_nonneg))
      simpa [bound, mul_add, mul_assoc, mul_left_comm, mul_comm] using this
    exact this
  have h_diff :
      ∀ᵐ w ∂(ℙ : Measure Ω), ∀ x ∈ Metric.ball t ε,
        HasDerivAt (fun s => F s w) (F' x w) x := by
    refine ae_of_all _ (fun w => ?_)
    intro x hx
    have hxIoo : x ∈ Set.Ioo (0 : ℝ) 1 := hball_Ioo x hx
    -- Chain rule: F = free_energy_density ∘ H_t, so dF/ds = fderiv(free_energy_density) ∘ dH_t/ds
    have hHt_diff : HasDerivAt
        (fun s => H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) s w)
        (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) x :=
      hasDerivAt_H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x hxIoo w
    have hFed : HasFDerivAt (fun H => free_energy_density (N := N) H)
        (fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w))
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) x w) :=
      ((contDiff_free_energy_density (N := N)).differentiable (by simp) ).differentiableAt.hasFDerivAt
    have hcomp := hFed.comp_hasDerivAt x hHt_diff
    change HasDerivAt
      ((fun H : EnergySpace N => free_energy_density (N := N) H) ∘
        fun s => H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) s w)
      (F' x w) x
    simpa [F'] using hcomp
  have hMain :=
    (hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (μ := (ℙ : Measure Ω)) (F := F) (F' := F') (x₀ := t) (bound := bound)
      (s := Metric.ball t ε) (hs := Metric.ball_mem_nhds t hε_pos)
      hF_meas hF_int hF'_meas h_bound hbound_int h_diff).2
  change HasDerivAt (fun s => ∫ w, F s w ∂ℙ) (∫ w, F' t w ∂ℙ) t
  exact hMain

/-!
### How to invoke the Hilbert-space Gaussian IBP theorem

The theorem intended here is
`PhysLean.Probability.GaussianIBP.gaussian_integration_by_parts_hilbert_cov_op` from
`SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert`.  Its schematic form is

```
E[⟪g, e⟫ * F(g)] = E[(fderiv ℝ F (g)) ((covOp hg) e)].
```

It requires `hg : IsGaussianHilbert g`, `ContDiff ℝ 1 F`, and
`HasModerateGrowth F`.  The disorder structures already provide the Gaussian models
`sk.hU` and `sim.hV`, while `sk.cov_eq` and `sim.cov_eq` identify the matrix entries of their
covariance operators in the configuration basis.

There is one important formal point.  The first-variation test function depends on both
`sk.U` and `sim.V`, so calling the theorem on `sk.hU` while leaving `sim.V ω` inside the test
function is not valid.  A convenient bridge is a local lemma constructing

```
G ω := (sk.U ω, sim.V ω)
```

as an `IsGaussianHilbert` random variable on the product Hilbert space.  Build its basis from
the two component bases, and use `hIndep` to prove that the two coordinate families are jointly
independent.  Its covariance operator is block diagonal.  This bridge is the only additional
Gaussian-model construction needed by the affine joint-IBP lemma below.

For the SK term, set

```
Φ p := free_energy_density (N := N) (a • p.1 + b • p.2 + field)
Fσ p := (fderiv ℝ Φ p) (std_basis N σ, 0)
```

and expand the random direction in the configuration basis.  For each `σ`, the main call is
schematically

```
have hIBP :=
  PhysLean.Probability.GaussianIBP.gaussian_integration_by_parts_hilbert_cov_op
    (hg := hG) (h := (std_basis N σ, 0)) (F := Fσ)
    (hF_diff := hFσ_diff) (hF_growth := hFσ_growth)
```

The derivative of `Fσ` is the Hessian of the pressure.  Expand the block covariance vector in
the configuration basis, use `sk.cov_eq σ τ`, interchange the finite sums with the integral,
and collect `a * a'`.  The simple-disorder term uses `(0, std_basis N σ)` and
`sim.cov_eq σ τ` in exactly the same way.

The required smoothness follows from `contDiff_free_energy_density`.  For moderate growth,
prove a small helper for each `Fσ`; the explicit Gibbs Hessian is uniformly bounded in finite
volume, so a constant polynomial bound suffices.  The integrability helpers surrounding the
IBP theorem then justify every finite-sum and expectation interchange.
-/

/-- Measurability of a configuration-basis entry of the pressure Hessian.  This helper is
shared by both covariance traces. -/
lemma measurable_hessian_free_energy_std_basis (σ τ : Config N) :
    Measurable (fun H : EnergySpace N =>
      hessian_free_energy N H (std_basis N σ) (std_basis N τ)) := by
  simp_rw [hessian_free_energy]
  apply Measurable.mul measurable_const
  apply Measurable.sub
  · exact Finset.measurable_sum _ fun x _ => by
      apply Measurable.mul _ measurable_const
      apply Measurable.mul _ measurable_const
      exact (contDiff_gibbs_pmf (N := N) (σ := x)).continuous.measurable
  · apply Measurable.mul
    · exact Finset.measurable_sum _ fun x _ => by
        apply Measurable.mul
        · exact (contDiff_gibbs_pmf (N := N) (σ := x)).continuous.measurable
        · exact measurable_const
    · exact Finset.measurable_sum _ fun x _ => by
        apply Measurable.mul
        · exact (contDiff_gibbs_pmf (N := N) (σ := x)).continuous.measurable
        · exact measurable_const

/-- Uniform finite-volume bound for a configuration-basis entry of the pressure Hessian. -/
lemma abs_hessian_free_energy_std_basis_le
    (H : EnergySpace N) (σ τ : Config N) :
    |hessian_free_energy N H (std_basis N σ) (std_basis N τ)| ≤ 1 / (N : ℝ) := by
  classical
  have hσ0 : 0 ≤ gibbs_pmf N H σ := gibbs_pmf_nonneg N H σ
  have hτ0 : 0 ≤ gibbs_pmf N H τ := gibbs_pmf_nonneg N H τ
  have hσ1 : gibbs_pmf N H σ ≤ 1 := gibbs_pmf_le_one N H σ
  have hτ1 : gibbs_pmf N H τ ≤ 1 := gibbs_pmf_le_one N H τ
  by_cases hστ : σ = τ
  · subst τ
    simp [hessian_free_energy, std_basis]
    have hp : 0 ≤ gibbs_pmf N H σ - gibbs_pmf N H σ * gibbs_pmf N H σ := by
      nlinarith
    rw [abs_of_nonneg hp]
    have hN0 : (0 : ℝ) ≤ (N : ℝ) := Nat.cast_nonneg N
    have hp1 : gibbs_pmf N H σ - gibbs_pmf N H σ * gibbs_pmf N H σ ≤ 1 := by
      nlinarith
    calc
      (N : ℝ)⁻¹ * (gibbs_pmf N H σ - gibbs_pmf N H σ * gibbs_pmf N H σ)
          ≤ (N : ℝ)⁻¹ * 1 := mul_le_mul_of_nonneg_left hp1 (inv_nonneg.mpr hN0)
      _ = (N : ℝ)⁻¹ := mul_one _
  · simp [hessian_free_energy, std_basis, hστ]
    rw [abs_of_nonneg hσ0, abs_of_nonneg hτ0]
    calc
      (N : ℝ)⁻¹ * (gibbs_pmf N H σ * gibbs_pmf N H τ)
          ≤ (N : ℝ)⁻¹ * 1 := by
            have hN0 : (0 : ℝ) ≤ (N : ℝ) := Nat.cast_nonneg N
            exact mul_le_mul_of_nonneg_left (by nlinarith) (inv_nonneg.mpr hN0)
      _ = (N : ℝ)⁻¹ := by ring

/-- Gaussian integration by parts for an affine combination of two independent Gaussian
Hamiltonians, expressed in the canonical configuration basis.

This is the sole measure-theoretic Gaussian-IBP interface used by the ordinary smart path.
Construct the product-Hilbert Gaussian model described above, apply the operator-form theorem
along the two block basis directions, use block diagonality, and collect the coefficients.
Keeping both covariance traces in one statement avoids duplicating the conditional or product
law argument. -/
lemma independent_gaussian_affine_ibp
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    (a b a' b' : ℝ) (field : EnergySpace N) :
    (∫ w,
      fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
        (a • sk.U w + b • sim.V w + field) (a' • sk.U w + b' • sim.V w) ∂ℙ) =
      (a * a') * ∫ w, (∑ σ : Config N, ∑ τ : Config N,
        mixedCovKernel N sk.ξ σ τ * hessian_free_energy N
          (a • sk.U w + b • sim.V w + field)
          (std_basis N σ) (std_basis N τ)) ∂ℙ +
      (b * b') * ∫ w, (∑ σ : Config N, ∑ τ : Config N,
        referenceCovKernel N β σ τ * hessian_free_energy N
          (a • sk.U w + b • sim.V w + field)
          (std_basis N σ) (std_basis N τ)) ∂ℙ := by
  exact independent_gaussian_affine_ibp_reproved
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
    hIndep a b a' b' field

/-- Joint Gaussian integration by parts for the raw smart-path derivative, before evaluating
its two covariance traces. -/
lemma pressure_derivative_ibp_trace
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    (∫ w,
        fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
          (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
        ∂ℙ) =
      (1 / 2) * ∫ w,
        (∑ σ : Config N, ∑ τ : Config N,
          (mixedCovKernel N sk.ξ σ τ -
            referenceCovKernel N β σ τ) *
          hessian_free_energy N
            (H_t (N := N) (β := β) (h := h) (q := q)
              (sk := sk) (sim := sim) t w)
            (std_basis N σ) (std_basis N τ)) ∂ℙ := by
  have ht0 : t > 0 := ht.1
  have ht1 : t < 1 := ht.2
  -- Set up the IBP parameters
  set a := Real.sqrt t with ha_def
  set b := Real.sqrt (1 - t) with hb_def
  set a' := 1 / (2 * Real.sqrt t) with ha'_def
  set b' := -1 / (2 * Real.sqrt (1 - t)) with hb'_def
  -- Apply the independent_gaussian_affine_ibp lemma
  have h_ibp := independent_gaussian_affine_ibp (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) hIndep a b a' b' (H_field (N := N) (h := h))
  -- Show that a * a' = 1/2 and b * b' = -1/2
  have ha_aa' : a * a' = 1 / 2 := by
    simp [ha_def, ha'_def]
    field_simp [ne_of_gt (Real.sqrt_pos.mpr ht0)]
  have hb_bb' : b * b' = -(1 / 2) := by
    simp [hb_def, hb'_def]
    field_simp [ne_of_gt (Real.sqrt_pos.mpr (sub_pos.mpr ht1))]
  -- Show that a • sk.U w + b • sim.V w + H_field = H_t t w
  have h_eq_H : H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t =
      fun w => a • sk.U w + b • sim.V w + H_field (N := N) (h := h) := by
    unfold H_t H_gauss
    simp [ha_def, hb_def]
  -- Show that a' • sk.U w + b' • sim.V w = dH_t t w
  have h_eq_dH : dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t =
      fun w => a' • sk.U w + b' • sim.V w := by
    unfold dH_t
    ext w
    simp [ha'_def, hb'_def]
    ring
  -- Rewrite h_ibp using the equalities
  have h_ibp' : ∫ w, fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H) (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) ∂ℙ =
    (a * a') * ∫ w, (∑ σ : Config N, ∑ τ : Config N,
      mixedCovKernel N sk.ξ σ τ * hessian_free_energy N (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) (std_basis N σ) (std_basis N τ)) ∂ℙ +
    (b * b') * ∫ w, (∑ σ : Config N, ∑ τ : Config N,
      referenceCovKernel N β σ τ * hessian_free_energy N (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w) (std_basis N σ) (std_basis N τ)) ∂ℙ := by
    simp only [h_eq_H, h_eq_dH] at *
    convert h_ibp using 2
  -- Substitute a * a' = 1/2 and b * b' = -1/2
  rw [ha_aa', hb_bb'] at h_ibp'
  -- Combine the integrals
  convert h_ibp' using 1
  have integral_eq : ∀ w, ∑ σ, ∑ τ, (mixedCovKernel N sk.ξ σ τ - referenceCovKernel N β σ τ) *
      hessian_free_energy N (H_t N β h q sk sim t w) (std_basis N σ) (std_basis N τ) =
      (∑ σ, ∑ τ, mixedCovKernel N sk.ξ σ τ * hessian_free_energy N (H_t N β h q sk sim t w) (std_basis N σ) (std_basis N τ)) -
      (∑ σ, ∑ τ, referenceCovKernel N β σ τ * hessian_free_energy N (H_t N β h q sk sim t w) (std_basis N σ) (std_basis N τ)) := by
    intro w
    simp_rw [sub_mul]
    simp only [Finset.sum_sub_distrib]
  -- Bound on hessian_free_energy for standard basis
  have std_basis_apply : ∀ σ τ : Config N, (std_basis N σ) τ = if σ = τ then 1 else 0 := by
    intro σ τ
    simp [std_basis]
  -- Integrability of finite sums of bounded functions
  have h_int1 : MeasureTheory.Integrable
      (fun x => ∑ σ : Config N, ∑ τ : Config N,
        mixedCovKernel N sk.ξ σ τ * hessian_free_energy N (H_t N β h q sk sim t x) (std_basis N σ) (std_basis N τ))
      ℙ := by
    apply MeasureTheory.integrable_finset_sum _
    intro σ _
    apply MeasureTheory.integrable_finset_sum _
    intro τ _
    refine MeasureTheory.Integrable.const_mul ?_ (mixedCovKernel N sk.ξ σ τ)
    refine MeasureTheory.Integrable.mono' (MeasureTheory.integrable_const (1 / (N : ℝ))) ?_ ?_
    · have hH_meas : Measurable (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t) := by
        have hU := sk.hU.repr_measurable.const_smul (Real.sqrt t)
        have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
        exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t
      have hheff_meas : Measurable
          (fun H => hessian_free_energy N H (std_basis N σ) (std_basis N τ)) :=
        measurable_hessian_free_energy_std_basis (N := N) σ τ
      exact (hheff_meas.comp hH_meas).aestronglyMeasurable
    · filter_upwards with x
      exact abs_hessian_free_energy_std_basis_le
        (N := N)
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t x) σ τ
  have h_int2 : MeasureTheory.Integrable
      (fun x => ∑ σ : Config N, ∑ τ : Config N,
        referenceCovKernel N β σ τ * hessian_free_energy N (H_t N β h q sk sim t x) (std_basis N σ) (std_basis N τ))
      ℙ := by
    apply MeasureTheory.integrable_finset_sum _
    intro σ _
    apply MeasureTheory.integrable_finset_sum _
    intro τ _
    refine MeasureTheory.Integrable.const_mul ?_ (referenceCovKernel N β σ τ)
    refine MeasureTheory.Integrable.mono' (MeasureTheory.integrable_const (1 / (N : ℝ))) ?_ ?_
    · have hH_meas : Measurable (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t) := by
        have hU := sk.hU.repr_measurable.const_smul (Real.sqrt t)
        have hV := sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
        exact measurable_H_t_updated (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t
      have hheff_meas : Measurable
          (fun H => hessian_free_energy N H (std_basis N σ) (std_basis N τ)) :=
        measurable_hessian_free_energy_std_basis (N := N) σ τ
      exact (hheff_meas.comp hH_meas).aestronglyMeasurable
    · filter_upwards with x
      exact abs_hessian_free_energy_std_basis_le
        (N := N)
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t x) σ τ
  rw [funext integral_eq, MeasureTheory.integral_sub h_int1 h_int2]
  rw [mul_sub]
  ring

/- The covariance-trace difference is the Bregman remainder, pointwise in the disorder. -/
lemma pressure_trace_algebra
    (hN : 0 < N) (H : EnergySpace N) :
    (1 / 2) *
        (∑ σ : Config N, ∑ τ : Config N,
          (mixedCovKernel N sk.ξ σ τ - referenceCovKernel N β σ τ) *
          hessian_free_energy N H (std_basis N σ) (std_basis N τ)) =
      (1 / 2) * (bregmanRemainder sk.ξ β q 1 -
        gibbs_average_n_det (N := N) (n := 2) H
          (bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk))) := by
  classical
  let K : Config N → Config N → ℝ := fun σ τ =>
    mixedCovKernel N sk.ξ σ τ - referenceCovKernel N β σ τ
  let D : ℝ → ℝ := bregmanRemainder sk.ξ β q
  have hN0 : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have hp : ∑ σ : Config N, gibbs_pmf N H σ = 1 := sum_gibbs_pmf N H
  have hrep :
      gibbs_average_n_det (N := N) (n := 2) H
          (bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk)) =
        ∑ σ : Config N, ∑ τ : Config N,
          gibbs_pmf N H σ * gibbs_pmf N H τ * D (overlap N σ τ) := by
    unfold gibbs_average_n_det bregmanOverlap
    rw [← Finset.sum_product']
    simpa +decide [Fin.prod_univ_two, D, mul_assoc, mul_comm, mul_left_comm] using
      (Equiv.sum_comp (finTwoArrowEquiv (Config N))
        (fun p : Config N × Config N =>
          gibbs_pmf N H p.1 * gibbs_pmf N H p.2 * D (overlap N p.1 p.2)))
  rw [show (∑ σ : Config N, ∑ τ : Config N, K σ τ *
      hessian_free_energy N H (std_basis N σ) (std_basis N τ)) =
      (1 / (N : ℝ)) *
        ((∑ σ, gibbs_pmf N H σ * K σ σ) -
          ∑ σ, ∑ τ, gibbs_pmf N H σ * gibbs_pmf N H τ * K σ τ) by
        exact trace_formula N H K]
  rw [hrep]
  simp only [K, D, mixedCovKernel, referenceCovKernel]
  simp_rw [overlap_self (N := N) hN]
  simp only [bregmanRemainder]
  field_simp [hN0]
  have hp2 : (∑ σ : Config N, ∑ τ : Config N,
      gibbs_pmf N H σ * gibbs_pmf N H τ) = 1 := by
    simpa only [← Finset.mul_sum, hp, mul_one]
  let c : ℝ := sk.ξ q - β * q
  have hkernel (r : ℝ) : sk.ξ r - β * r =
      (sk.ξ r - sk.ξ q - β * (r - q)) + c := by
    dsimp only [c]
    ring
  have hkernelOne : sk.ξ 1 - β =
      (sk.ξ 1 - sk.ξ q - β * (1 - q)) + c := by
    dsimp only [c]
    ring
  simp_rw [hkernelOne]
  simp_rw [hkernel]
  let A : ℝ := sk.ξ 1 - sk.ξ q - β * (1 - q)
  let S : ℝ := ∑ σ : Config N, ∑ τ : Config N,
    gibbs_pmf N H σ * gibbs_pmf N H τ *
      (sk.ξ (overlap N σ τ) - sk.ξ q - β * (overlap N σ τ - q))
  have hsum1 : (∑ σ : Config N,
      (N : ℝ) * gibbs_pmf N H σ * (A + c)) = (N : ℝ) * (A + c) := by
    calc
      _ = ((N : ℝ) * (A + c)) * ∑ σ : Config N, gibbs_pmf N H σ := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro σ _
        ring
      _ = _ := by rw [hp, mul_one]
  have hsumAffine : (∑ σ : Config N, ∑ τ : Config N,
      gibbs_pmf N H σ * gibbs_pmf N H τ *
        ((sk.ξ (overlap N σ τ) - sk.ξ q - β * (overlap N σ τ - q)) + c)) =
      S + c := by
    calc
      _ = S + (∑ σ : Config N, ∑ τ : Config N,
          gibbs_pmf N H σ * gibbs_pmf N H τ * c) := by
        dsimp only [S]
        simp only [mul_add, Finset.sum_add_distrib]
      _ = S + c := by
        rw [show (∑ σ : Config N, ∑ τ : Config N,
            gibbs_pmf N H σ * gibbs_pmf N H τ * c) = c by
          calc
            _ = (∑ σ : Config N, ∑ τ : Config N,
                gibbs_pmf N H σ * gibbs_pmf N H τ) * c := by
              rw [Finset.sum_mul]
              apply Finset.sum_congr rfl
              intro σ _
              rw [Finset.sum_mul]
            _ = c := by rw [hp2, one_mul]]
  have hsum2 : (∑ σ : Config N, ∑ τ : Config N,
      (N : ℝ) * gibbs_pmf N H σ * gibbs_pmf N H τ *
        ((sk.ξ (overlap N σ τ) - sk.ξ q - β * (overlap N σ τ - q)) + c)) =
      (N : ℝ) * (S + c) := by
    calc
      _ = (N : ℝ) * (∑ σ : Config N, ∑ τ : Config N,
          gibbs_pmf N H σ * gibbs_pmf N H τ *
            ((sk.ξ (overlap N σ τ) - sk.ξ q - β * (overlap N σ τ - q)) + c)) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro σ _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro τ _
        ring
      _ = _ := by rw [hsumAffine]
  change (∑ σ : Config N, (N : ℝ) * gibbs_pmf N H σ * (A + c)) -
      (∑ σ : Config N, ∑ τ : Config N,
        (N : ℝ) * gibbs_pmf N H σ * gibbs_pmf N H τ *
          ((sk.ξ (overlap N σ τ) - sk.ξ q - β * (overlap N σ τ - q)) + c)) =
    (N : ℝ) * (A - S)
  rw [hsum1, hsum2]
  ring

/-- The annealed Gibbs average of the centered overlap square is `overlapVariance`. -/
lemma integral_centeredOverlapSq_eq_overlapVariance (t : ℝ) :
    (∫ w, gibbs_average_n_det (N := N) (n := 2)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w) (centeredOverlapSq N q) ∂ℙ) =
      overlapVariance
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  rfl

lemma integral_bregmanOverlap_eq_bregmanAverage (t : ℝ) :
    (∫ w, gibbs_average_n_det (N := N) (n := 2)
        (H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) t w)
        (bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk)) ∂ℙ) =
      bregmanAverage
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  rfl

/-
Gaussian integration by parts evaluates the raw smart-path pressure derivative.
-/
lemma pressure_derivative_ibp
    (hN : 0 < N)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    (∫ w,
        fderiv ℝ (fun H : EnergySpace N => free_energy_density (N := N) H)
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
          (dH_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t w)
        ∂ℙ) =
      (1 / 2) * (bregmanRemainder sk.ξ β q 1 -
        bregmanAverage
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t) := by
  have := @SpinGlass.GeneralizedLatala.pressure_derivative_ibp_trace;
  rw [ this N β h q sk sim hIndep ht, MeasureTheory.integral_congr_ae ( Filter.Eventually.of_forall fun w => ?_ ) ];
  any_goals exact fun w => (1 / 2) *
    (bregmanRemainder sk.ξ β q 1 - gibbs_average_n_det (N := N) (n := 2)
      (H_t N β h q sk sim t w)
      (bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk))) * 2;
  · rw [ MeasureTheory.integral_mul_const, MeasureTheory.integral_const_mul ];
    rw [ MeasureTheory.integral_sub ] <;> norm_num;
    · rw [integral_bregmanOverlap_eq_bregmanAverage] ; ring;
    · apply_rules [ SpinGlass.integrable_gibbs_average_n ];
  · grind +suggestions

/-- The ordinary Guerra smart-path sum-rule derivative.

The repository already provides the smart path and Hilbert-space Gaussian integration by parts.
This lemma records their specialization to the centered overlap square.
-/
lemma pressure_derivative
    (hN : 0 < N)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω))
    {t : ℝ} (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt
      (interpolatedPressure
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim))
      ((1 / 2) * (bregmanRemainder sk.ξ β q 1 -
        bregmanAverage
          (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t)) t := by
  rw [← pressure_derivative_ibp
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
    hN hIndep ht]
  exact pressure_derivative_before_ibp
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) ht


end GeneralizedLatala
end SpinGlass
