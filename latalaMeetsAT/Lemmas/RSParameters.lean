import Lemmas.Replicas
import Mathlib.Probability.Distributions.Gaussian.Real

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

/-- Expectation with respect to a standard real Gaussian. -/
noncomputable def standardGaussianExpectation (f : ℝ → ℝ) : ℝ :=
  ∫ z, f z ∂gaussianReal 0 1

/-- Replica-symmetric fixed-point equation. -/
def IsRSFixedPoint (β h q : ℝ) : Prop :=
  q = standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt q * z) ^ 2)

/-- Canonical RS parameter.  The later existence and uniqueness theorem shows
that this infimum is the unique fixed point in `[0,1]`. -/
noncomputable def rsQ (β h : ℝ) : ℝ :=
  sInf {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧ IsRSFixedPoint β h q}

noncomputable def rsR (β h : ℝ) : ℝ :=
  standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt (rsQ β h) * z) ^ 4)

noncomputable def rsA (β h : ℝ) : ℝ :=
  1 - 2 * rsQ β h + rsR β h

noncomputable def atParameter (β h : ℝ) : ℝ := β ^ 2 * rsA β h

theorem rsQ_fixedPoint {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) :
    IsRSFixedPoint β h (rsQ β h) := by
  -- Paper route: use equation (q) and the standard existence and uniqueness
  -- theorem for the SK fixed point when `β > 0` and `h > 0`.  With the current
  -- `sInf` definition, first prove that the set in `rsQ` is nonempty, contained
  -- in `[0,1]`, and is a singleton.  `csInf_mem` can then identify its infimum
  -- with that unique member.  The paper cites this fixed-point theorem rather
  -- than proving it, so the Lean development needs it as a separate analytic
  -- lemma before this proof can be completed.
  sorry

theorem rsQ_pos {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) : 0 < rsQ β h := by
  -- Proof route from the paragraph after equation (qcompact): get the fixed
  -- point equation from `rsQ_fixedPoint` and first establish `0 ≤ rsQ β h`.
  -- If `rsQ β h = 0`, the equation reduces to
  -- `0 = tanh h ^ 2`; `hh` implies `0 < tanh h`, a contradiction.  The missing
  -- lower bound `0 ≤ rsQ` should come from the singleton/set-membership lemma
  -- built for `rsQ_fixedPoint`.
  sorry

theorem rsQ_lt_one {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) : rsQ β h < 1 := by
  -- Proof route from equation (qcompact): prove `rsQ β h ≤ 1` from membership
  -- in the fixed-point set.  Equality would say that the Gaussian expectation
  -- of `1 - tanh (...) ^ 2` is zero.  Rewrite this integrand as `sech (...) ^ 2`;
  -- it is strictly positive for every finite real argument, so its integral
  -- against the Gaussian probability measure is positive.  This requires the
  -- standard lemma that a positive almost-everywhere integrable function has
  -- positive integral.
  sorry

theorem rsA_eq_one_sub_two_q_add_r (β h : ℝ) :
    rsA β h = 1 - 2 * rsQ β h + rsR β h := by
  -- Proof route: this is exactly the definition of `rsA`.
  rfl

theorem rsA_eq_gaussian_sech_fourth {β h : ℝ}
    (hβ : 0 < β) (hh : 0 < h) :
    rsA β h = standardGaussianExpectation (fun z =>
      (Real.cosh (h + β * Real.sqrt (rsQ β h) * z))⁻¹ ^ 4) := by
  have hq := rsQ_fixedPoint hβ hh
  unfold IsRSFixedPoint at hq
  unfold standardGaussianExpectation at hq
  unfold rsA rsR standardGaussianExpectation
  let X : ℝ → ℝ := fun z => h + β * √(rsQ β h) * z
  have htanh : Continuous (fun x : ℝ => Real.tanh x) := by
    simp_rw [Real.tanh_eq]
    apply Continuous.div
    · fun_prop
    · fun_prop
    · intro x
      positivity
  have hInt2 : Integrable (fun z => Real.tanh (X z) ^ 2) (gaussianReal 0 1) := by
    apply Integrable.of_bound (C := 1)
    · exact (htanh.comp (by fun_prop)).pow 2 |>.aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_one₀ (abs_nonneg _) (le_of_lt (Real.abs_tanh_lt_one _))
  have hInt4 : Integrable (fun z => Real.tanh (X z) ^ 4) (gaussianReal 0 1) := by
    apply Integrable.of_bound (C := 1)
    · exact (htanh.comp (by fun_prop)).pow 4 |>.aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_one₀ (abs_nonneg _) (le_of_lt (Real.abs_tanh_lt_one _))
  have hIntConst : Integrable (fun _ : ℝ => (1 : ℝ)) (gaussianReal 0 1) :=
    integrable_const 1
  have hx : 1 - 2 * rsQ β h + (∫ z, Real.tanh (X z) ^ 4 ∂gaussianReal 0 1) =
      ∫ z, (Real.cosh (X z))⁻¹ ^ 4 ∂gaussianReal 0 1 := by
    calc
      _ = 1 - 2 * (∫ z, Real.tanh (X z) ^ 2 ∂gaussianReal 0 1) +
          (∫ z, Real.tanh (X z) ^ 4 ∂gaussianReal 0 1) := by
        simpa only [X] using congrArg
          (fun y => 1 - 2 * y + (∫ z, Real.tanh (X z) ^ 4 ∂gaussianReal 0 1)) hq
      _ = ∫ z, (1 - 2 * Real.tanh (X z) ^ 2 + Real.tanh (X z) ^ 4)
          ∂gaussianReal 0 1 := by
        calc
          _ = (∫ z, 1 - 2 * Real.tanh (X z) ^ 2 ∂gaussianReal 0 1) +
              (∫ z, Real.tanh (X z) ^ 4 ∂gaussianReal 0 1) := by
            rw [integral_sub hIntConst (hInt2.const_mul 2), integral_const_mul]
            simp
          _ = ∫ z, (1 - 2 * Real.tanh (X z) ^ 2) + Real.tanh (X z) ^ 4
              ∂gaussianReal 0 1 := by
            simpa only [Pi.add_apply, Pi.sub_apply] using
              (integral_add (hIntConst.sub (hInt2.const_mul 2)) hInt4).symm
      _ = _ := integral_congr_ae (ae_of_all _ fun z => by
        change 1 - 2 * Real.tanh (X z) ^ 2 + Real.tanh (X z) ^ 4 =
          (Real.cosh (X z))⁻¹ ^ 4
        rw [Real.tanh_eq_sinh_div_cosh]
        have hc : Real.cosh (X z) ≠ 0 := ne_of_gt (Real.cosh_pos (X z))
        rw [inv_pow]
        field_simp
        nlinarith [Real.cosh_sq (X z)])
  simpa only [X] using hx

theorem atParameter_nonneg {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) :
    0 ≤ atParameter β h := by
  rw [atParameter, rsA_eq_gaussian_sech_fourth hβ hh]
  apply mul_nonneg (sq_nonneg β)
  unfold standardGaussianExpectation
  exact integral_nonneg fun z => by positivity

theorem rsR_le_rsQ {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) :
    rsR β h ≤ rsQ β h := by
  rw [rsQ_fixedPoint hβ hh]
  unfold rsR standardGaussianExpectation
  have htanh : Continuous (fun x : ℝ => Real.tanh x) := by
    simp_rw [Real.tanh_eq]
    apply Continuous.div
    · fun_prop
    · fun_prop
    · intro x
      positivity
  apply integral_mono
  · apply Integrable.of_bound (C := 1)
    · exact (htanh.comp (by fun_prop)).pow 4 |>.aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_one₀ (abs_nonneg _) (le_of_lt (Real.abs_tanh_lt_one _))
  · apply Integrable.of_bound (C := 1)
    · exact (htanh.comp (by fun_prop)).pow 2 |>.aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_pow]
      exact pow_le_one₀ (abs_nonneg _) (le_of_lt (Real.abs_tanh_lt_one _))
  · intro z
    have ht := Real.tanh_sq_lt_one (h + β * √(rsQ β h) * z)
    have hn := sq_nonneg (Real.tanh (h + β * √(rsQ β h) * z))
    nlinarith [sq_nonneg (Real.tanh (h + β * √(rsQ β h) * z) ^ 2)]

/-- The anomalous cavity eigenvalue is bounded by the replicon one. -/
theorem anomalous_eigenvalue_le_replicon {β h : ℝ}
    (hβ : 0 < β) (hh : 0 < h) :
    β ^ 2 * (1 - 4 * rsQ β h + 3 * rsR β h) ≤ atParameter β h := by
  have hqr : 0 ≤ rsQ β h - rsR β h := sub_nonneg.mpr (rsR_le_rsQ hβ hh)
  have hm : 0 ≤ β ^ 2 * (rsQ β h - rsR β h) :=
    mul_nonneg (sq_nonneg β) hqr
  unfold atParameter rsA
  nlinarith

end SpinGlass.AT
