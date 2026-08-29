import SpinGlass.Inequalities.KearnsSaul
import Mathlib.Probability.Distributions.Gaussian.CharFun

/-!
# Basic scalar data

Scalar Gaussian identities and replica-symmetric parameters for the generalized Latala estimate.

Main declarations:
- `hubbard_stratonovich`
- `IsRSFixedPoint`
- `lambdaStar`

Dependencies:
- the Kearns--Saul coefficient and the standard real Gaussian

This file corresponds to the relevant part of `blueprint_latala.txt`.
-/

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ENNReal NNReal Topology

set_option maxHeartbeats 800000

namespace SpinGlass
namespace GeneralizedLatala

universe uΩ uι

variable {Ω : Type uΩ} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

/-!
**# Hubbard--Stratonovich identity**

This file records the scalar Gaussian identity used to linearize a positive
quadratic exponential.  It depends only on mathlib.
-/

/-- The moment-generating function identity for a standard real Gaussian,
written directly as an integral. -/
lemma integral_exp_mul_standardGaussian (t : ℝ) :
    ∫ z, Real.exp (t * z) ∂gaussianReal 0 1 = Real.exp (t ^ 2 / 2) := by
  simpa [mgf] using congrFun (mgf_id_gaussianReal (μ := 0) (v := 1)) t

/-- The scalar Hubbard--Stratonovich identity.  If `a` is nonnegative and
`Z` is a standard real Gaussian, then
`exp (a * x ^ 2 / 2) = E[exp (sqrt a * x * Z)]`. -/
lemma hubbard_stratonovich (a x : ℝ) (ha : 0 ≤ a) :
    Real.exp (a * x ^ 2 / 2) =
      ∫ z, Real.exp (Real.sqrt a * x * z) ∂gaussianReal 0 1 := by
  rw [integral_exp_mul_standardGaussian, mul_pow, Real.sq_sqrt ha]

/-! ## Scalar replica-symmetric data -/

/-- Expectation against a standard real Gaussian. -/
noncomputable def standardGaussianExpectation (f : ℝ → ℝ) : ℝ :=
  ∫ z, f z ∂ProbabilityTheory.gaussianReal 0 1

/-- The replica-symmetric fixed-point equation
`q = E[tanh (h + sqrt(β) Z)^2]`, where `β = ξ'(q)`. -/
def IsRSFixedPoint (β h q : ℝ) : Prop :=
  q = standardGaussianExpectation
    (fun z => Real.tanh (h + Real.sqrt β * z) ^ 2)

/-- The sharp Bernoulli sub-Gaussian coefficient used at the independent endpoint. -/
noncomputable def kappa (q : ℝ) : ℝ :=
  if q = 0 then 1 else q / Real.artanh q

/-- The Bregman remainder of `ξ` at `q`, with prescribed slope `d = ξ'(q)`. -/
noncomputable def bregmanRemainder (ξ : ℝ → ℝ) (d q r : ℝ) : ℝ :=
  ξ r - ξ q - d * (r - q)

/-- The two global Bregman bounds supplied by convexity and the definition of `Γ`. -/
def BregmanBounds (ξ : ℝ → ℝ) (d q Γ : ℝ) : Prop :=
  ∀ r ∈ Set.Icc (-1 : ℝ) 1,
    0 ≤ bregmanRemainder ξ d q r ∧
      bregmanRemainder ξ d q r ≤ (Γ / 2) * (r - q) ^ 2

/-- The improved high-temperature parameter `ρ = Γ κ(q)`. -/
noncomputable def rho (Γ q : ℝ) : ℝ :=
  Γ * kappa q

/-- Coupling strength used in the quadratic replica estimate. -/
noncomputable def lambdaStar (Γ q : ℝ) : ℝ :=
  ((kappa q)⁻¹ - Γ) / 4

/-- The constant on the right side of the uniform logarithmic quadratic estimate. -/
noncomputable def quadraticConstant (Γ q : ℝ) : ℝ :=
  (1 / 2) * Real.exp (2 * rho Γ q / (1 - rho Γ q)) *
    Real.log (2 / (1 - rho Γ q))

/-- The replica-symmetric free-energy prediction. -/
noncomputable def rsPressure (ξ : ℝ → ℝ) (β h q : ℝ) : ℝ :=
  Real.log 2 +
    standardGaussianExpectation
      (fun z => Real.log (Real.cosh (h + Real.sqrt β * z))) +
    (1 / 2) * bregmanRemainder ξ β q 1

lemma kappa_zero : kappa 0 = 1 := by
  simp [kappa]

lemma kappa_pos {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1) : 0 < kappa q := by
  by_cases hq : q = 0
  · simp [hq, kappa]
  · have hqpos : 0 < q := lt_of_le_of_ne hq0 (Ne.symm hq)
    have ha : 0 < Real.artanh q := Real.artanh_pos ⟨hqpos, hq1⟩
    simp only [kappa, if_neg hq]
    exact div_pos hqpos ha

lemma rho_eq (Γ q : ℝ) : rho Γ q = Γ * kappa q := by
  rfl

lemma lambdaStar_eq (Γ q : ℝ) :
    lambdaStar Γ q = ((kappa q)⁻¹ - Γ) / 4 := by
  rfl


end GeneralizedLatala
end SpinGlass
