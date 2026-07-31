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
  -- Proof route for equation (ra): use `rsQ_fixedPoint` to replace `rsQ` by
  -- the expectation of `tanh ^ 2`, unfold `rsR` and `rsA`, and move the finite
  -- linear combination inside the Gaussian integral.  Apply the pointwise
  -- identity `(1 - tanh x ^ 2) ^ 2 = cosh x ⁻¹ ^ 4`; `ring_nf` finishes after
  -- the standard hyperbolic identity.  Boundedness of `tanh` supplies all
  -- integrability obligations.
  sorry

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
