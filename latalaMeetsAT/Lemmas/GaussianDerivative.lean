import Lemmas.SmartPath
import SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert
import Mathlib.Probability.Distributions.Gaussian.CharFun
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence
import Mathlib.Analysis.InnerProductSpace.Spectrum

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

open scoped RealInnerProductSpace NNReal
open PhysLean.Probability.GaussianIBP

/-- A centered finite-dimensional Gaussian law admits the independent-coordinate
model required by the Hilbert-space integration-by-parts theorem. -/
private noncomputable def gaussianHilbertOfCentered
    {Ω E : Type*} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    [SecondCountableTopology E] [CompleteSpace E]
    (g : Ω → E) (hg : Measurable g)
    (hgauss : IsGaussian (Measure.map g (ℙ : Measure Ω)))
    (hmean : ∫ ω : Ω, g ω ∂(ℙ : Measure Ω) = 0) :
    IsGaussianHilbert g := by
  classical
  let μ : Measure E := Measure.map g (ℙ : Measure Ω)
  letI : IsGaussian μ := hgauss
  have hmeanμ : ∫ x : E, x ∂μ = 0 := by
    change ∫ x : E, id x ∂Measure.map g (ℙ : Measure Ω) = 0
    rw [integral_map hg.aemeasurable aestronglyMeasurable_id]
    exact hmean
  let T : E →ₗ[ℝ] E := (covarianceOperator μ).toLinearMap
  let hT : T.IsPositive := isPositive_covarianceOperator
  let w : OrthonormalBasis (Fin (Module.finrank ℝ E)) ℝ E :=
    hT.isSymmetric.eigenvectorBasis rfl
  let lam : Fin (Module.finrank ℝ E) → ℝ := hT.isSymmetric.eigenvalues rfl
  let c : Fin (Module.finrank ℝ E) → Ω → ℝ := fun i ω => inner ℝ (g ω) (w i)
  refine
    { ι := Fin (Module.finrank ℝ E)
      fintype_ι := inferInstance
      w := w
      τ := fun i => (lam i).toNNReal
      c := c
      c_meas := ?_
      c_gauss := ?_
      c_indep := ?_
      repr := ?_ }
  · intro i
    have hp : Continuous (fun x : E => (x, w i)) :=
      continuous_id.prodMk continuous_const
    exact (continuous_inner.comp hp).measurable.comp hg
  · intro i
    let L : E →L[ℝ] ℝ := InnerProductSpace.toDualMap ℝ E (w i)
    have hmap := IsGaussian.map_eq_gaussianReal (μ := μ) L
    have hmeanL : ∫ x : E, L x ∂μ = 0 := by
      rw [L.integral_comp_id_comm IsGaussian.integrable_id, hmeanμ]
      simp
    have hvar : Var[L; μ] = lam i := by
      change Var[(fun x : E => inner ℝ (w i) x); μ] = lam i
      rw [← covarianceBilin_self IsGaussian.memLp_two_id]
      rw [covarianceBilin_apply IsGaussian.memLp_two_id]
      simp only [id_eq, hmeanμ, sub_zero]
      rw [← covarianceOperator_inner IsGaussian.memLp_two_id]
      change inner ℝ (T (w i)) (w i) = lam i
      rw [hT.isSymmetric.apply_eigenvectorBasis rfl i]
      rw [inner_smul_left]
      simp [w, lam]
    rw [hmeanL, hvar] at hmap
    change Measure.map (c i) (ℙ : Measure Ω) = gaussianReal 0 (lam i).toNNReal
    rw [show c i = L ∘ g by
      funext ω
      simpa [c, L] using (real_inner_comm (g ω) (w i)).symm]
    rw [← Measure.map_map L.measurable hg]
    change Measure.map L μ = gaussianReal 0 (lam i).toNNReal
    exact hmap
  · let C : E →L[ℝ] (Fin (Module.finrank ℝ E) → ℝ) :=
      ContinuousLinearMap.pi (fun i => InnerProductSpace.toDualMap ℝ E (w i))
    have hgLaw : HasGaussianLaw g (ℙ : Measure Ω) := hgauss.hasGaussianLaw
    have hjoint : HasGaussianLaw (fun ω : Ω => C (g ω)) (ℙ : Measure Ω) := by
      simpa [Function.comp_def] using hgLaw.map C
    have hi : iIndepFun (fun i ω => C (g ω) i) (ℙ : Measure Ω) := by
      apply hjoint.iIndepFun_of_covariance_eq_zero
      intro i j hij
      let Xi : E → ℝ := fun x => C x i
      let Xj : E → ℝ := fun x => C x j
      have hcov : cov[Xi, Xj; μ] = cov[Xi ∘ g, Xj ∘ g; (ℙ : Measure Ω)] :=
        covariance_map
          (((continuous_apply i).comp C.continuous).measurable.aestronglyMeasurable)
          (((continuous_apply j).comp C.continuous).measurable.aestronglyMeasurable)
          hg.aemeasurable
      change cov[Xi ∘ g, Xj ∘ g; (ℙ : Measure Ω)] = 0
      rw [← hcov]
      change cov[(fun x : E => inner ℝ (w i) x),
        (fun x : E => inner ℝ (w j) x); μ] = 0
      rw [← covarianceBilin_apply_eq_cov IsGaussian.memLp_two_id]
      rw [covarianceBilin_apply IsGaussian.memLp_two_id]
      simp only [id_eq, hmeanμ, sub_zero]
      rw [← covarianceOperator_inner IsGaussian.memLp_two_id]
      change inner ℝ (T (w i)) (w j) = 0
      rw [hT.isSymmetric.apply_eigenvectorBasis rfl i]
      rw [inner_smul_left]
      simp [hij, w]
    simpa [C, c, real_inner_comm] using hi
  · funext x
    calc
      g x = ∑ i, (w.repr (g x)).ofLp i • w i := (w.sum_repr (g x)).symm
      _ = ∑ i, c i x • w i := by
        apply Finset.sum_congr rfl
        intro i _
        congr 1
        rw [w.repr_apply_apply]
        simp [c, real_inner_comm]

/-- Derivative in the interpolation parameter of the smart-path covariance. -/
noncomputable def smartPathCovDerivative (N : ℕ) (β q : ℝ)
    (σ τ : Config N) : ℝ :=
  (N : ℝ) * β ^ 2 / 2 * configOverlap N σ τ ^ 2 -
    (N : ℝ) * β ^ 2 * q * configOverlap N σ τ - β ^ 2 / 2

private def replicaFinCast {n : ℕ} (a : Fin n) : Fin (n + 2) :=
  ⟨a, by omega⟩

private def penultimateReplica (n : ℕ) : Fin (n + 2) := ⟨n, by omega⟩

private def lastReplica (n : ℕ) : Fin (n + 2) := ⟨n + 1, by omega⟩

/-- The covariance-derivative operator for a finite replica expectation.

This is the standard two-extra-replica Hessian contraction. The last two
replicas encode the derivatives of the Gibbs normalizing factors. -/
noncomputable def replicaCovarianceOperator {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ) (s : ℝ) : ℝ :=
  (1 / 2 : ℝ) * quenchedReplicaAverage (path.H s) (fun σs : Replicas N (n + 2) =>
    F (fun a => σs (replicaFinCast a)) *
      ((∑ a : Fin n, ∑ b : Fin n,
          smartPathCovDerivative N β q
            (σs (replicaFinCast a)) (σs (replicaFinCast b))) -
        2 * (n : ℝ) * ∑ a : Fin n,
          smartPathCovDerivative N β q
            (σs (replicaFinCast a)) (σs (penultimateReplica n)) +
        (n : ℝ) * (n + 1 : ℝ) *
          smartPathCovDerivative N β q
            (σs (penultimateReplica n)) (σs (lastReplica n)) -
        (n : ℝ) * smartPathCovDerivative N β q
          (σs (penultimateReplica n)) (σs (penultimateReplica n))))

/-- Finite-dimensional Gaussian covariance interpolation for normalized
finite Gibbs sums.  Its analytic proof is isolated here. -/
theorem quenchedGibbs_deriv_of_covariance_deriv {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    (hs : s ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun t => quenchedReplicaAverage (path.H t) F)
      (replicaCovarianceOperator path F s) s := by
  classical
  have hzero : (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by norm_num
  have hone : (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by norm_num
  letI hgauss0 : IsGaussian (Measure.map (path.H 0) volume) := path.gaussian 0 hzero
  letI hgauss1 : IsGaussian (Measure.map (path.H 1) volume) := path.gaussian 1 hone
  have hmean0 : ∫ ω, path.H 0 ω ∂(volume : Measure Ω) = 0 := by
    ext σ
    rw [ContinuousLinearMap.integral_comp_comm]
    exact path.centered 0 hzero σ
  have hmean1 : ∫ ω, path.H 1 ω ∂(volume : Measure Ω) = 0 := by
    ext σ
    rw [ContinuousLinearMap.integral_comp_comm]
    exact path.centered 1 hone σ
  let hg0 : IsGaussianHilbert (path.H 0) :=
    gaussianHilbertOfCentered (path.H 0) (path.measurable 0) hgauss0 hmean0
  let hg1 : IsGaussianHilbert (path.H 1) :=
    gaussianHilbertOfCentered (path.H 1) (path.measurable 1) hgauss1 hmean1
  sorry

end SpinGlass.AT
