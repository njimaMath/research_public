import Lemmas.MainResult
import Mathlib.MeasureTheory.Measure.LevyConvergence
import Mathlib.Probability.Distributions.Gaussian.Real

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-! ## The quenched law of a replica observable -/

/-- The disorder-averaged Gibbs measure on `n` replicas. -/
noncomputable def quenchedReplicaMeasure
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) : Measure (Replicas N n) :=
  (volume : Measure Ω).bind
    (fun ω => SpinGlass.replicaGibbsMeasure (N := N) (n := n) (H ω))

lemma measurable_replicaGibbsMeasure_comp
    {Ω : Type u} [MeasureSpace Ω]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H) :
    Measurable
      (fun ω => SpinGlass.replicaGibbsMeasure (N := N) (n := n) (H ω)) := by
  classical
  apply Measure.measurable_of_measurable_coe
  intro s hs
  simp only [SpinGlass.replicaGibbsMeasure, Measure.coe_finsetSum,
    Finset.sum_apply, Measure.smul_apply, Measure.dirac_apply' _ hs]
  apply Finset.measurable_sum
  intro σs _
  apply Measurable.mul
  · unfold SpinGlass.replicaGibbsWeightNNReal
    apply Measurable.coe_nnreal_ennreal
    apply Measurable.subtype_mk
    apply Finset.measurable_prod
    intro a _
    exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs a)).continuous.measurable.comp hH
  · exact measurable_const

lemma quenchedReplicaMeasure_isProbabilityMeasure
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H) :
    IsProbabilityMeasure (quenchedReplicaMeasure (n := n) H) := by
  apply MeasureTheory.isProbabilityMeasure_bind
  · exact (measurable_replicaGibbsMeasure_comp H hH).aemeasurable
  · exact Filter.Eventually.of_forall fun _ => inferInstance

/-- The law of the scaled centered overlap
`sqrt N * (R₁₂ - q)` under the quenched two-replica Gibbs measure. -/
noncomputable def scaledOverlapLaw
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : ProbabilityMeasure ℝ := by
  let H := fullPathHamiltonian path 1
  have hH : Measurable H := by
    exact ((path.sk.hU.repr_measurable.const_smul (Real.sqrt 1)).add
      (path.simple.hV.repr_measurable.const_smul (Real.sqrt (1 - 1)))).add measurable_const
  letI : IsProbabilityMeasure (quenchedReplicaMeasure (n := 2) H) :=
    quenchedReplicaMeasure_isProbabilityMeasure H hH
  exact ⟨Measure.map
      (fun σs : Replicas N 2 =>
        Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1)
      (quenchedReplicaMeasure H),
    Measure.isProbabilityMeasure_map (by fun_prop)⟩

/-- The centered Gaussian probability law with the given nonnegative part as variance. -/
noncomputable def centeredGaussianLaw (variance : ℝ) : ProbabilityMeasure ℝ :=
  ⟨gaussianReal 0 variance.toNNReal, inferInstance⟩

/--
Characteristic-function form of the central limit theorem for the overlap at
the SK endpoint of the replica-symmetric smart path.

The theorem is stated using cosine and sine separately because
`quenchedReplicaAverage` is real-valued. Together, the two limits say that
the characteristic function of

  `sqrt N * (R₁₂ - q)`

converges to the characteristic function of a centered Gaussian with variance

  `3 * rsA / (1 - α)
    - 2 * κ / (1 - β^2 κ)
    - ζ / (1 - β^2 κ)^2`,

where

  `q = rsQ β h`,
  `r = rsR β h`,
  `α = atParameter β h`,
  `κ = cavityKappa q r`,
  `ζ = cavityZeta q r`.

The sequence is indexed by `N`, while the physical system size is `N.succ`,
so no zero-size model enters the statement.
-/
theorem overlapCLT_characteristic
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ}
    (hβ : 0 < β)
    (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths :
      ∀ N : ℕ,
        RSSmartPathDisorder Ω N.succ β h (rsQ β h)) :
    let σ2 : ℝ :=
      3 * rsA β h / (1 - atParameter β h)
        - 2 * cavityKappa (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h))
        - cavityZeta (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) ^ 2
    ∀ t : ℝ,
      Tendsto
          (fun N : ℕ =>
            quenchedReplicaAverage
              (fullPathHamiltonian (paths N) 1)
              (fun σs : Replicas N.succ 2 =>
                Real.cos
                  (t * Real.sqrt (N.succ : ℝ) *
                    centeredOverlap (rsQ β h) σs 0 1)))
          atTop
          (𝓝 (Real.exp (-((1 : ℝ) / 2) * σ2 * t ^ 2)))
      ∧
      Tendsto
          (fun N : ℕ =>
            quenchedReplicaAverage
              (fullPathHamiltonian (paths N) 1)
              (fun σs : Replicas N.succ 2 =>
                Real.sin
                  (t * Real.sqrt (N.succ : ℝ) *
                    centeredOverlap (rsQ β h) σs 0 1)))
          atTop
          (𝓝 0) := by
  sorry

/--
Weak-convergence form of the central limit theorem. If

`X_N = sqrt (N + 1) * (R₁₂ - rsQ β h)`,

then the quenched laws of `X_N` converge weakly to the centered normal law
with variance `σ2`. The first conjunct records that the displayed expression
is a valid variance; consequently `σ2.toNNReal` has real value `σ2`.
-/
theorem overlapCLT_weak
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ}
    (hβ : 0 < β)
    (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths :
      ∀ N : ℕ,
        RSSmartPathDisorder Ω N.succ β h (rsQ β h)) :
    let σ2 : ℝ :=
      3 * rsA β h / (1 - atParameter β h)
        - 2 * cavityKappa (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h))
        - cavityZeta (rsQ β h) (rsR β h) /
            (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) ^ 2
    0 ≤ σ2 ∧
      Tendsto
        (fun N : ℕ => scaledOverlapLaw (paths N))
        atTop
        (𝓝 (centeredGaussianLaw σ2)) := by
  sorry

end SpinGlass.AT
