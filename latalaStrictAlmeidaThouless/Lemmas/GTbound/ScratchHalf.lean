import Lemmas.GTbound.Comparison
import SpinGlass.AT.Gaussian_concentration.gaussian_concentration

open MeasureTheory ProbabilityTheory Real BigOperators Filter Topology

#check MeasureTheory.Integrable.comp_measurable
#check MeasureTheory.integrable_map_measure
#check SYK.integrable_exp_mul_norm
#check integral_pos_iff_support_of_nonneg
#check Real.exp_log

namespace SpinGlass.AT

lemma test_integrable_exp_norm_gaussianProduct
    {I : Type*} [Fintype I] (c : ℝ) :
    Integrable (fun z : I → ℝ =>
      Real.exp (c * ‖(WithLp.toLp 2 z : EuclideanSpace ℝ I)‖))
      (Measure.pi (fun _ : I => gaussianReal 0 1)) := by
  have hi := SYK.integrable_exp_mul_norm (ι := I) c
  unfold SYK.standardGaussianMeasureOnEuclidean at hi
  rw [MeasureTheory.integrable_map_measure (by fun_prop) (by fun_prop)] at hi
  exact hi

end SpinGlass.AT
