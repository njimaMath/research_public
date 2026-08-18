import Lemmas.ATDefs
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.MeasureTheory.Measure.Lebesgue.Integral
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import SpinGlass.Mathlib.Probability.Distributions.GaussianIntegrationByParts

open MeasureTheory ProbabilityTheory Real
open scoped MeasureTheory NNReal

#check hasDerivAt_integral_of_dominated_loc_of_deriv_le
#check antitoneOn_of_deriv_nonpos
#check ProbabilityTheory.gaussianReal_integration_by_parts
#check integral_comp_neg_Iic
#check integral_comp_neg_Ioi
#check setIntegral_union
#check MeasureTheory.Integrable.integrableOn
#check Real.hasDerivAt_cosh
#check Real.hasDerivAt_sqrt
#check Real.hasDerivAt_sqrt'
#check Real.tanh_neg
#check Real.cosh_neg
#check Real.strictMono_tanh
#check Real.abs_tanh_lt_one
#check Real.tanh_sq_add_sech_sq
#check Real.cosh_sq_sub_sinh_sq
#check Real.sq_sqrt
#check gaussianReal_map_const_mul
#check gaussianReal_map_add_const
#check gaussianReal_map_const_add
#check integral_map
#check MeasureTheory.integral_mono_measure
#check Integrable.integral_add
#check intervalIntegral.integral_eq_sub_of_hasDerivAt
#check Real.contDiff_cosh
#check ContDiff.inv
#check ProbabilityTheory.integrable_id_gaussianReal
#check ProbabilityTheory.memLp_id_gaussianReal

example (x : ℝ) : HasDerivAt (fun x : ℝ => (Real.cosh x)⁻¹ ^ 3)
    (-3 * (Real.cosh x)⁻¹ ^ 3 * Real.tanh x) x := by
  convert (((Real.hasDerivAt_cosh x).inv (Real.cosh_ne_zero x)).pow 3) using 1
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp [Real.cosh_ne_zero x]
  ring
