import SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert
import Mathlib.Probability.Distributions.Gaussian.CharFun
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence
import Mathlib.Analysis.InnerProductSpace.Spectrum

#check ContinuousLinearMap.pi
#check ContinuousLinearMap.apply_apply
#check ProbabilityTheory.HasGaussianLaw.map
#check ProbabilityTheory.IsGaussian.hasGaussianLaw_id
#check ProbabilityTheory.HasGaussianLaw.iIndepFun_of_covariance_eq_zero
#check ProbabilityTheory.covarianceBilin_self
#check ProbabilityTheory.covarianceOperator_inner
#check ProbabilityTheory.isPositive_covarianceOperator
#check LinearMap.IsPositive.isSymmetric
#check LinearMap.IsSymmetric.eigenvectorBasis
#check LinearMap.IsSymmetric.apply_eigenvectorBasis
#check LinearMap.IsPositive.nonneg_eigenvalues
#check ProbabilityTheory.IsGaussian.map_eq_gaussianReal
#check InnerProductSpace.toDualMap
#check OrthonormalBasis.sum_repr
#check OrthonormalBasis.sum_repr_symm
#check MeasureTheory.integral_map
#check MeasureTheory.Measure.map_map
#check ProbabilityTheory.covariance_map
#check PiLp.continuousLinearEquiv
#check WithLp.toLp
#check WithLp.ofLp
#check contDiff_prod'
#check contDiff_sum
#check ContDiff.prod
#check ContDiff.sum
#check contDiff_exp
#check Real.contDiff_exp

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory RealInnerProductSpace NNReal
open PhysLean.Probability.GaussianIBP

noncomputable section

private noncomputable def testModel
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [FiniteDimensional ℝ E] [MeasureSpace E] [BorelSpace E]
    [SecondCountableTopology E] [CompleteSpace E]
    [IsGaussian (ℙ : Measure E)]
    (hmean : ∫ x : E, x ∂(ℙ : Measure E) = 0) :
    IsGaussianHilbert (fun x : E => x) := by
  classical
  let T : E →ₗ[ℝ] E := (covarianceOperator (ℙ : Measure E)).toLinearMap
  let hT : T.IsPositive := isPositive_covarianceOperator
  let w : OrthonormalBasis (Fin (Module.finrank ℝ E)) ℝ E :=
    hT.isSymmetric.eigenvectorBasis rfl
  let lam : Fin (Module.finrank ℝ E) → ℝ := hT.isSymmetric.eigenvalues rfl
  let c : Fin (Module.finrank ℝ E) → E → ℝ := fun i x => inner ℝ x (w i)
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
    exact (continuous_inner.comp hp).measurable
  · intro i
    let L : E →L[ℝ] ℝ := InnerProductSpace.toDualMap ℝ E (w i)
    have hmap := IsGaussian.map_eq_gaussianReal (μ := (ℙ : Measure E)) L
    have hmeanL : ∫ x : E, L x ∂(ℙ : Measure E) = 0 := by
      rw [L.integral_comp_id_comm IsGaussian.integrable_id, hmean]
      simp
    have hvar : Var[L; (ℙ : Measure E)] = lam i := by
      change Var[(fun x : E => inner ℝ (w i) x); (ℙ : Measure E)] = lam i
      rw [← covarianceBilin_self IsGaussian.memLp_two_id]
      rw [covarianceBilin_apply IsGaussian.memLp_two_id]
      simp only [id_eq, hmean, sub_zero]
      rw [← covarianceOperator_inner IsGaussian.memLp_two_id]
      change inner ℝ (T (w i)) (w i) = lam i
      rw [hT.isSymmetric.apply_eigenvectorBasis rfl i]
      rw [inner_smul_left]
      simp [w, lam]
    rw [hmeanL, hvar] at hmap
    change Measure.map (c i) (ℙ : Measure E) = gaussianReal 0 (lam i).toNNReal
    have hcL : c i = L := by
      funext x
      simpa [c, L] using (real_inner_comm x (w i)).symm
    rw [hcL]
    exact hmap
  · let C : E →L[ℝ] (Fin (Module.finrank ℝ E) → ℝ) :=
      ContinuousLinearMap.pi (fun i => InnerProductSpace.toDualMap ℝ E (w i))
    have hjoint : HasGaussianLaw (fun x : E => C x) (ℙ : Measure E) :=
      IsGaussian.hasGaussianLaw_id.map C
    have hi : iIndepFun (fun i x => C x i) (ℙ : Measure E) := by
      apply hjoint.iIndepFun_of_covariance_eq_zero
      intro i j hij
      change cov[(fun x : E => inner ℝ (w i) x),
        (fun x : E => inner ℝ (w j) x); (ℙ : Measure E)] = 0
      rw [← covarianceBilin_apply_eq_cov IsGaussian.memLp_two_id]
      rw [covarianceBilin_apply IsGaussian.memLp_two_id]
      simp only [id_eq, hmean, sub_zero]
      rw [← covarianceOperator_inner IsGaussian.memLp_two_id]
      change inner ℝ (T (w i)) (w j) = 0
      rw [hT.isSymmetric.apply_eigenvectorBasis rfl i]
      rw [inner_smul_left]
      simp [OrthonormalBasis.inner_eq_ite, hij, w]
    simpa [C, c, real_inner_comm] using hi
  · funext x
    calc
      x = ∑ i, (w.repr x).ofLp i • w i := (w.sum_repr x).symm
      _ = ∑ i, c i x • w i := by
        apply Finset.sum_congr rfl
        intro i _
        congr 1
        rw [w.repr_apply_apply]
        simp [c, real_inner_comm]
