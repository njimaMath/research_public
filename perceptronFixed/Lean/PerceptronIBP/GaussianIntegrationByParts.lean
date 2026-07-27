import SpinGlass.Mathlib.Probability.Distributions.GaussianIntegrationByParts

open MeasureTheory Filter Set Real
open scoped ProbabilityTheory NNReal ENNReal Filter Topology

namespace ProbabilityTheory

/-- Gaussian integration by parts for the standard normal under direct
integrability assumptions. This complements `gaussianReal_integration_by_parts`,
whose interface uses `HasModerateGrowth`. -/
lemma gaussianReal_integration_by_parts_of_integrable
    {F : ℝ → ℝ}
    (hF : Differentiable ℝ F)
    (hF_int : Integrable F (gaussianReal 0 1))
    (hF'_int : Integrable (fun x => deriv F x) (gaussianReal 0 1))
    (hxF_int : Integrable (fun x => x * F x) (gaussianReal 0 1)) :
    (∫ x, x * F x ∂(gaussianReal 0 1)) =
      ∫ x, deriv F x ∂(gaussianReal 0 1) := by
  let φ : ℝ → ℝ :=
    fun x => (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x ^ 2) / 2)
  have hv : (1 : ℝ≥0) ≠ 0 := by simp
  have hpdf :
      φ = gaussianPDFReal 0 (1 : ℝ≥0) := by
    funext x
    simp [φ, gaussianPDFReal, div_eq_mul_inv, mul_left_comm, mul_comm]
  have hf : Measurable (gaussianPDF (0 : ℝ) (1 : ℝ≥0)) :=
    measurable_gaussianPDF _ _
  have hflt :
      (∀ᵐ x ∂(volume : Measure ℝ), gaussianPDF (0 : ℝ) (1 : ℝ≥0) x < ∞) :=
    ae_of_all _ (fun _ => gaussianPDF_lt_top)

  have hF_int' :
      Integrable F (volume.withDensity (gaussianPDF (0 : ℝ) (1 : ℝ≥0))) := by
    simpa [gaussianReal_of_var_ne_zero (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) hv] using hF_int
  have hF'_int' :
      Integrable (fun x => deriv F x)
        (volume.withDensity (gaussianPDF (0 : ℝ) (1 : ℝ≥0))) := by
    simpa [gaussianReal_of_var_ne_zero (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) hv] using hF'_int
  have hxF_int' :
      Integrable (fun x => x * F x)
        (volume.withDensity (gaussianPDF (0 : ℝ) (1 : ℝ≥0))) := by
    simpa [gaussianReal_of_var_ne_zero (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) hv] using hxF_int

  have hFφ : Integrable (fun x : ℝ => F x * φ x) volume := by
    have h :=
      (integrable_withDensity_iff_integrable_smul'
        (μ := (volume : Measure ℝ))
        (f := gaussianPDF (0 : ℝ) (1 : ℝ≥0)) hf hflt (g := F)).1 hF_int'
    simpa [hpdf, smul_eq_mul, mul_assoc, mul_left_comm, mul_comm] using h
  have hF'φ : Integrable (fun x : ℝ => deriv F x * φ x) volume := by
    have h :=
      (integrable_withDensity_iff_integrable_smul'
        (μ := (volume : Measure ℝ))
        (f := gaussianPDF (0 : ℝ) (1 : ℝ≥0)) hf hflt
        (g := fun x => deriv F x)).1 hF'_int'
    simpa [hpdf, smul_eq_mul, mul_assoc, mul_left_comm, mul_comm] using h
  have hxFφ : Integrable (fun x : ℝ => (x * F x) * φ x) volume := by
    have h :=
      (integrable_withDensity_iff_integrable_smul'
        (μ := (volume : Measure ℝ))
        (f := gaussianPDF (0 : ℝ) (1 : ℝ≥0)) hf hflt
        (g := fun x => x * F x)).1 hxF_int'
    simpa [hpdf, smul_eq_mul, mul_assoc, mul_left_comm, mul_comm] using h

  have hu : ∀ x, HasDerivAt F (deriv F x) x :=
    fun x => (hF x).hasDerivAt
  have hvφ : ∀ x, HasDerivAt φ (-x * φ x) x := by
    intro x
    have hinner :
        HasDerivAt (fun u : ℝ => -(u ^ 2) / 2) (-x) x := by
      have hpow : HasDerivAt (fun u : ℝ => u ^ 2) (2 * x) x := by
        simpa using hasDerivAt_pow (n := 2) (x := x)
      simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using
        hpow.neg.div_const (2 : ℝ)
    have hexp :
        HasDerivAt (fun u : ℝ => Real.exp (-(u ^ 2) / 2))
          (-(x * Real.exp (-(x ^ 2) / 2))) x := by
      simpa [Function.comp, mul_assoc, mul_left_comm, mul_comm] using
        (Real.hasDerivAt_exp (x := (-(x ^ 2) / 2))).comp x hinner
    have hmul :=
      hexp.const_mul (1 / Real.sqrt (2 * Real.pi))
    simpa [φ, mul_assoc, mul_left_comm, mul_comm] using hmul

  have huv' : Integrable (fun x : ℝ => F x * (-x * φ x)) volume := by
    have hneg : Integrable (fun x : ℝ => -(x * (F x * φ x))) volume := by
      simpa [mul_assoc] using hxFφ.const_mul (-1 : ℝ)
    convert hneg using 1
    funext x
    ring
  have hibp :
      (∫ x : ℝ, F x * (-x * φ x)) =
        -∫ x : ℝ, deriv F x * φ x := by
    simpa using
      (integral_mul_deriv_eq_deriv_mul_of_integrable
        (u := F) (v := φ) (u' := fun x => deriv F x) (v' := fun x => -x * φ x)
        hu hvφ huv' hF'φ hFφ)
  have hibp' :
      (∫ x : ℝ, (x * F x) * φ x) =
        ∫ x : ℝ, deriv F x * φ x := by
    have hleft :
        (∫ x : ℝ, (x * F x) * φ x) =
          -∫ x : ℝ, F x * (-x * φ x) := by
      rw [← integral_neg]
      congr 1
      funext x
      ring
    rw [hleft]
    simpa using congrArg Neg.neg hibp

  have hL :
      (∫ x, x * F x ∂gaussianReal 0 1) =
        ∫ x : ℝ, (x * F x) * φ x := by
    rw [integral_gaussianReal_eq_integral_smul (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) hv]
    simp [hpdf, smul_eq_mul, mul_assoc, mul_comm]
  have hR :
      (∫ x, deriv F x ∂gaussianReal 0 1) =
        ∫ x : ℝ, deriv F x * φ x := by
    rw [integral_gaussianReal_eq_integral_smul (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) hv]
    simp [hpdf, smul_eq_mul, mul_comm]
  rw [hL, hR]
  exact hibp'

end ProbabilityTheory
