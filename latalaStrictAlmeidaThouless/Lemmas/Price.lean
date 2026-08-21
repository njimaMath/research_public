import Lemmas.Price.Cosh

/-!
# Gaussian covariance differentiation

Compatibility entry point for the modular Price development.
-/

namespace ProbabilityTheory.PriceTanh

/-- The derivative of `tanh`, in the form that occurs in Price's theorem. -/
noncomputable def sechSq (x : ℝ) : ℝ := 1 - Real.tanh x ^ 2

lemma tanh_hasDerivAt (x : ℝ) : HasDerivAt Real.tanh (sechSq x) x := by
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  rw [show Real.tanh = fun y => Real.sinh y / Real.cosh y by
    funext y
    exact Real.tanh_eq_sinh_div_cosh y]
  apply ((Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc).congr_deriv
  rw [sechSq, Real.tanh_eq_sinh_div_cosh]
  field_simp [hc]

/-- The partial derivative in the first variable of
`(x,y) ↦ tanh x * tanh y`. -/
lemma hasDerivAt_tanh_mul_tanh_left (x y : ℝ) :
    HasDerivAt (fun u => Real.tanh u * Real.tanh y)
      (sechSq x * Real.tanh y) x := by
  simpa using (tanh_hasDerivAt x).mul_const (Real.tanh y)

/-- The mixed derivative needed to apply Price's theorem to
`E[tanh(Y₁(t)) tanh(Y₂(t))]`:
`∂₂ ∂₁ (tanh x * tanh y) = (1 - tanh² x) (1 - tanh² y)`. -/
lemma hasDerivAt_tanh_mul_tanh_mixed (x y : ℝ) :
    HasDerivAt (fun v => sechSq x * Real.tanh v)
      (sechSq x * sechSq y) y := by
  simpa using (tanh_hasDerivAt y).const_mul (sechSq x)

/-- The mixed derivative appearing under the Gaussian expectation in Price's identity. -/
noncomputable def tanhPriceIntegrand (x y : ℝ) : ℝ := sechSq x * sechSq y

end ProbabilityTheory.PriceTanh
