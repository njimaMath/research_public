import Lemmas.CLT.CLT_Main

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false

namespace SpinGlass.AT

universe u

lemma test_integral_quenchedReplicaMeasure
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (F : ReplicaFun N n) :
    (∫ σs, F σs ∂quenchedReplicaMeasure H) = quenchedReplicaAverage H F := by
  classical
  letI : IsProbabilityMeasure (quenchedReplicaMeasure (n := n) H) :=
    quenchedReplicaMeasure_isProbabilityMeasure (n := n) H hH
  rw [MeasureTheory.integral_fintype]
  · unfold quenchedReplicaMeasure quenchedReplicaAverage replicaGibbsAverage
    have hterms : ∀ σs ∈ (Finset.univ : Finset (Replicas N n)), Integrable
        (fun ω => (∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a)) * F σs)
        (volume : Measure Ω) := by
      intro σs _
      have hwint : Integrable
          (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a))
          (volume : Measure Ω) := by
        simpa [replicaGibbsAverage, SpinGlass.replicaGibbsWeightNNReal] using
          (integrable_replicaGibbsAverage_comp H hH
            (fun τ => if τ = σs then 1 else 0))
      simpa [mul_comm] using hwint.mul_const (F σs)
    rw [MeasureTheory.integral_finsetSum Finset.univ hterms]
    apply Finset.sum_congr rfl
    intro σs _
    rw [measureReal_def, Measure.bind_apply (MeasurableSet.singleton σs)
      ((measurable_replicaGibbsMeasure_comp H hH).aemeasurable)]
    simp only [SpinGlass.replicaGibbsMeasure, Measure.coe_finsetSum,
      Finset.sum_apply, Measure.smul_apply,
      Measure.dirac_apply' _ (MeasurableSet.singleton σs)]
    have hwint : Integrable
        (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a))
        (volume : Measure Ω) := by
      simpa [replicaGibbsAverage, SpinGlass.replicaGibbsWeightNNReal] using
        (integrable_replicaGibbsAverage_comp H hH
          (fun τ => if τ = σs then 1 else 0))
    have hsum (ω : Ω) :
        (∑ x, (SpinGlass.replicaGibbsWeightNNReal (N := N) (n := n) (H ω) x : ENNReal) •
          ({σs} : Set (Replicas N n)).indicator (1 : Replicas N n → ENNReal) x) =
          (SpinGlass.replicaGibbsWeightNNReal (N := N) (n := n) (H ω) σs : ENNReal) := by
      rw [Finset.sum_eq_single σs]
      · simp
      · intro x _ hx
        simp [Set.indicator, hx]
      · simp
    simp_rw [hsum]
    have hwint' : Integrable
        (fun ω => (SpinGlass.replicaGibbsWeightNNReal (N := N) (n := n) (H ω) σs : ℝ))
        (volume : Measure Ω) := by
      change Integrable (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a)) volume
      exact hwint
    rw [MeasureTheory.lintegral_coe_eq_integral _ hwint']
    have hw0 : 0 ≤ ∫ ω, (SpinGlass.replicaGibbsWeightNNReal (N := N) (n := n) (H ω) σs : ℝ) ∂volume :=
      integral_nonneg fun ω =>
        (SpinGlass.replicaGibbsWeightNNReal (N := N) (n := n) (H ω) σs).coe_nonneg
    simp only [ENNReal.toReal_ofReal hw0, smul_eq_mul]
    rw [MeasureTheory.integral_mul_const]
    rfl
  · exact MeasureTheory.Integrable.of_finite

end SpinGlass.AT

namespace SpinGlass.AT

universe u

example
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ}
    (hβ : 0 < β)
    (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h)) :
    let σ2 : ℝ :=
      3 * rsA β h / (1 - atParameter β h)
        - 2 * cavityKappa (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h))
        - cavityZeta (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) ^ 2
    0 ≤ σ2 ∧ Tendsto (fun N : ℕ => scaledOverlapLaw (paths N)) atTop
        (𝓝 (centeredGaussianLaw σ2)) := by
  dsimp only
  let σ2 : ℝ :=
    3 * rsA β h / (1 - atParameter β h)
      - 2 * cavityKappa (rsQ β h) (rsR β h) /
          (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h))
      - cavityZeta (rsQ β h) (rsR β h) /
          (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) ^ 2
  change 0 ≤ σ2 ∧ Tendsto (fun N : ℕ => scaledOverlapLaw (paths N)) atTop
      (𝓝 (centeredGaussianLaw σ2))
  have hchar := overlapCLT_characteristic hβ hh hAT paths
  have hH (N : ℕ) : Measurable (fullPathHamiltonian (paths N) 1) := by
    exact (((paths N).sk.hU.repr_measurable.const_smul (Real.sqrt 1)).add
      ((paths N).simple.hV.repr_measurable.const_smul (Real.sqrt (1 - 1)))).add
        measurable_const
  have hσ2 : 0 ≤ σ2 := by
    have hlim := (hchar 1).1
    have hle : ∀ N : ℕ,
        quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
          (fun σs : Replicas N.succ 2 => Real.cos
            (1 * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)) ≤ 1 := by
      intro N
      calc
        _ ≤ quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
              (fun _ : Replicas N.succ 2 => 1) :=
          quenchedReplicaAverage_mono _ (hH N) _ _ (fun σs => Real.cos_le_one _)
        _ = 1 := by
          unfold quenchedReplicaAverage replicaGibbsAverage
          simp_rw [mul_one, SpinGlass.sum_prod_gibbs_pmf_eq_one]
          simp
    have hexp : Real.exp (-((1 : ℝ) / 2) * σ2) ≤ 1 := by
      apply le_of_tendsto (by simpa using hlim)
      filter_upwards with N
      simpa using hle N
    rw [Real.exp_le_one_iff] at hexp
    linarith
  refine ⟨hσ2, ?_⟩
  have hcoe : (σ2.toNNReal : ℝ) = σ2 := Real.coe_toNNReal σ2 hσ2
  apply MeasureTheory.ProbabilityMeasure.tendsto_of_tendsto_charFun
  intro t
  have h_charFun_scaled (N : ℕ) :
      MeasureTheory.charFun (scaledOverlapLaw (paths N) : Measure ℝ) t =
        (quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
          (fun σs : Replicas N.succ 2 => Real.cos
            (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)) : ℂ) +
        (quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
          (fun σs : Replicas N.succ 2 => Real.sin
            (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)) : ℂ) *
          Complex.I := by
    rw [MeasureTheory.charFun_apply_real]
    have hlaw : (scaledOverlapLaw (paths N) : Measure ℝ) =
        Measure.map
          (fun σs : Replicas N.succ 2 =>
            Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)
          (quenchedReplicaMeasure (fullPathHamiltonian (paths N) 1)) := rfl
    rw [hlaw]
    rw [MeasureTheory.integral_map (by fun_prop) (by fun_prop)]
    letI : IsProbabilityMeasure
        (quenchedReplicaMeasure (n := 2) (fullPathHamiltonian (paths N) 1)) :=
      quenchedReplicaMeasure_isProbabilityMeasure (n := 2) _ (hH N)
    have hexpint : Integrable
        (fun σs : Replicas N.succ 2 => Complex.exp
          ((t : ℂ) * (Real.sqrt (N.succ : ℝ) *
            centeredOverlap (rsQ β h) σs 0 1 : ℝ) * Complex.I))
        (quenchedReplicaMeasure (n := 2) (fullPathHamiltonian (paths N) 1)) :=
      MeasureTheory.Integrable.of_finite
    apply Complex.ext
    · rw [← integral_re hexpint]
      simpa [Complex.exp_re] using test_integral_quenchedReplicaMeasure
        (fullPathHamiltonian (paths N) 1) (hH N)
        (fun σs : Replicas N.succ 2 => Real.cos
          (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1))
    · rw [← integral_im hexpint]
      simpa [Complex.exp_im] using test_integral_quenchedReplicaMeasure
        (fullPathHamiltonian (paths N) 1) (hH N)
        (fun σs : Replicas N.succ 2 => Real.sin
          (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1))
  rw [show (fun N => MeasureTheory.charFun (scaledOverlapLaw (paths N) : Measure ℝ) t) =
      fun N => _ by funext N; exact h_charFun_scaled N]
  have hc := (hchar t).1
  have hs := (hchar t).2
  have hz : Tendsto (fun N =>
      (quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
        (fun σs : Replicas N.succ 2 => Real.cos
          (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)) : ℂ) +
      (quenchedReplicaAverage (fullPathHamiltonian (paths N) 1)
        (fun σs : Replicas N.succ 2 => Real.sin
          (t * Real.sqrt (N.succ : ℝ) * centeredOverlap (rsQ β h) σs 0 1)) : ℂ) * Complex.I)
      atTop (𝓝 ((Real.exp (-((1 : ℝ) / 2) * σ2 * t ^ 2) : ℂ) + 0 * Complex.I)) :=
    (Complex.continuous_ofReal.tendsto _).comp hc |>.add
      ((Complex.continuous_ofReal.tendsto _).comp hs |>.mul_const Complex.I)
  have hgauss : MeasureTheory.charFun
      (centeredGaussianLaw σ2 : Measure ℝ) t =
      Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
    change MeasureTheory.charFun (gaussianReal 0 σ2.toNNReal) t = _
    rw [ProbabilityTheory.charFun_gaussianReal]
    congr 1
    norm_num [hcoe]
    ring
  rw [hgauss]
  have hexpeq : (Real.exp (-((1 : ℝ) / 2) * σ2 * t ^ 2) : ℂ) =
      Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
    rw [Complex.ofReal_exp]
    congr 1
    push_cast
    ring
  rw [← hexpeq]
  simpa using hz

end SpinGlass.AT
