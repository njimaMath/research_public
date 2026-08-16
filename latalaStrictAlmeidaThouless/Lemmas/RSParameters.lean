import Lemmas.Replicas
import Mathlib.Probability.Distributions.Gaussian.Real

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

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

private noncomputable def rsMap (β h q : ℝ) : ℝ :=
  standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt q * z) ^ 2)

private theorem continuous_tanh' : Continuous (fun x : ℝ => Real.tanh x) := by
  simp_rw [Real.tanh_eq]
  apply Continuous.div
  · fun_prop
  · fun_prop
  · intro x
    positivity

private theorem continuous_rsMap (β h : ℝ) : Continuous (rsMap β h) := by
  refine continuous_iff_continuousAt.2 fun q₀ => ?_
  have hmeas : ∀ᶠ q in 𝓝 q₀,
      AEStronglyMeasurable
        (fun z : ℝ => Real.tanh (h + β * Real.sqrt q * z) ^ 2)
        (gaussianReal 0 1) := by
    refine Filter.Eventually.of_forall fun q => ?_
    exact ((continuous_tanh'.comp (by fun_prop)).pow 2).aestronglyMeasurable
  have hbound : ∀ᶠ q in 𝓝 q₀, ∀ᵐ z ∂gaussianReal 0 1,
      ‖Real.tanh (h + β * Real.sqrt q * z) ^ 2‖ ≤ (1 : ℝ) := by
    refine Filter.Eventually.of_forall fun q => ae_of_all _ fun z => ?_
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    exact (Real.tanh_sq_lt_one _).le
  have hlim : ∀ᵐ z : ℝ ∂gaussianReal 0 1,
      Tendsto
        (fun q : ℝ => Real.tanh (h + β * Real.sqrt q * z) ^ 2)
        (𝓝 q₀)
        (𝓝 (Real.tanh (h + β * Real.sqrt q₀ * z) ^ 2)) := by
    refine ae_of_all _ fun z => ?_
    have harg : ContinuousAt (fun q : ℝ => h + β * Real.sqrt q * z) q₀ := by
      fun_prop
    exact ((ContinuousAt.comp (x := q₀)
      (f := fun q : ℝ => h + β * Real.sqrt q * z)
      (g := Real.tanh) continuous_tanh'.continuousAt harg).pow 2).tendsto
  have htend := tendsto_integral_filter_of_dominated_convergence
    (μ := gaussianReal 0 1) (l := 𝓝 q₀)
    (F := fun q : ℝ => fun z : ℝ =>
      Real.tanh (h + β * Real.sqrt q * z) ^ 2)
    (f := fun z : ℝ => Real.tanh (h + β * Real.sqrt q₀ * z) ^ 2)
    (bound := fun _ : ℝ => (1 : ℝ)) hmeas hbound (integrable_const 1) hlim
  change Tendsto
    (fun q : ℝ => ∫ z, Real.tanh (h + β * Real.sqrt q * z) ^ 2
      ∂gaussianReal 0 1)
    (𝓝 q₀)
    (𝓝 (∫ z, Real.tanh (h + β * Real.sqrt q₀ * z) ^ 2
      ∂gaussianReal 0 1))
  exact htend

private theorem rsMap_mem_Icc (β h q : ℝ) : rsMap β h q ∈ Set.Icc (0 : ℝ) 1 := by
  constructor
  · unfold rsMap standardGaussianExpectation
    exact integral_nonneg fun z => sq_nonneg _
  · unfold rsMap standardGaussianExpectation
    calc
      (∫ z, Real.tanh (h + β * Real.sqrt q * z) ^ 2 ∂gaussianReal 0 1) ≤
          ∫ _z : ℝ, (1 : ℝ) ∂gaussianReal 0 1 := by
        apply integral_mono
        · apply Integrable.of_bound (C := 1)
          · exact ((continuous_tanh'.comp (by fun_prop)).pow 2).aestronglyMeasurable
          · filter_upwards [] with z
            rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
            exact (Real.tanh_sq_lt_one _).le
        · exact integrable_const 1
        · intro z
          exact (Real.tanh_sq_lt_one _).le
      _ = 1 := by simp

private theorem rsFixedPointSet_nonempty (β h : ℝ) :
    {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧ IsRSFixedPoint β h q}.Nonempty := by
  obtain ⟨q, hq, hfix⟩ := exists_mem_Icc_isFixedPt
    (continuous_rsMap β h).continuousOn (by norm_num)
    (rsMap_mem_Icc β h 0).1 (rsMap_mem_Icc β h 1).2
  exact ⟨q, hq, by simpa [IsRSFixedPoint, rsMap] using hfix.symm⟩

theorem rsQ_mem_Icc (β h : ℝ) : rsQ β h ∈ Set.Icc (0 : ℝ) 1 := by
  let S := {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧ IsRSFixedPoint β h q}
  have hclosed : IsClosed S := by
    dsimp [S]
    exact isClosed_Icc.inter
      (isClosed_eq continuous_id (continuous_rsMap β h))
  have hnonempty : S.Nonempty := rsFixedPointSet_nonempty β h
  have hbdd : BddBelow S := ⟨0, fun q hq => hq.1.1⟩
  exact (hclosed.csInf_mem hnonempty hbdd).1

theorem rsQ_fixedPoint {β h : ℝ} (_hβ : 0 < β) (_hh : 0 < h) :
    IsRSFixedPoint β h (rsQ β h) := by
  -- Continuity and interval invariance give a fixed point in `[0,1]`.  The
  -- fixed-point set is closed and bounded below, hence contains its infimum.
  -- Uniqueness is not needed for this selection argument.
  let S := {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧ IsRSFixedPoint β h q}
  have hclosed : IsClosed S := by
    dsimp [S]
    exact isClosed_Icc.inter
      (isClosed_eq continuous_id (continuous_rsMap β h))
  have hnonempty : S.Nonempty := rsFixedPointSet_nonempty β h
  have hbdd : BddBelow S := ⟨0, fun q hq => hq.1.1⟩
  exact (hclosed.csInf_mem hnonempty hbdd).2

theorem rsQ_pos {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) : 0 < rsQ β h := by
  -- Proof route from the paragraph after equation (qcompact): get the fixed
  -- point equation from `rsQ_fixedPoint` and first establish `0 ≤ rsQ β h`.
  -- If `rsQ β h = 0`, the equation reduces to
  -- `0 = tanh h ^ 2`; `hh` implies `0 < tanh h`, a contradiction.  The missing
  -- lower bound `0 ≤ rsQ` should come from the singleton/set-membership lemma
  -- built for `rsQ_fixedPoint`.
  have hqnonneg := (rsQ_mem_Icc β h).1
  apply lt_of_le_of_ne hqnonneg
  intro hqzero
  have hfp := rsQ_fixedPoint hβ hh
  rw [← hqzero] at hfp
  simp [IsRSFixedPoint, standardGaussianExpectation] at hfp
  have htanh : 0 < Real.tanh h := by
    rw [Real.tanh_eq]
    exact div_pos (sub_pos.mpr (Real.exp_lt_exp.mpr (by linarith)))
      (add_pos (Real.exp_pos h) (Real.exp_pos (-h)))
  nlinarith

theorem rsQ_lt_one {β h : ℝ} (hβ : 0 < β) (hh : 0 < h) : rsQ β h < 1 := by
  -- Proof route from equation (qcompact): prove `rsQ β h ≤ 1` from membership
  -- in the fixed-point set.  Equality would say that the Gaussian expectation
  -- of `1 - tanh (...) ^ 2` is zero.  Rewrite this integrand as `sech (...) ^ 2`;
  -- it is strictly positive for every finite real argument, so its integral
  -- against the Gaussian probability measure is positive.  This requires the
  -- standard lemma that a positive almost-everywhere integrable function has
  -- positive integral.
  have hqle := (rsQ_mem_Icc β h).2
  apply lt_of_le_of_ne hqle
  intro hqone
  have hfp := rsQ_fixedPoint hβ hh
  rw [hqone] at hfp
  let X : ℝ → ℝ := fun z => h + β * z
  let g : ℝ → ℝ := fun z => 1 - Real.tanh (X z) ^ 2
  have hgcont : Continuous g := by
    dsimp [g, X]
    exact continuous_const.sub ((continuous_tanh'.comp (by fun_prop)).pow 2)
  have hgnonneg : 0 ≤ g := fun z => by
    dsimp [g]
    exact sub_nonneg.mpr (Real.tanh_sq_lt_one _).le
  have hgint : Integrable g (gaussianReal 0 1) := by
    apply Integrable.of_bound (C := 1)
    · exact hgcont.aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_of_nonneg (hgnonneg z)]
      dsimp [g]
      nlinarith [sq_nonneg (Real.tanh (X z))]
  have hgpos : 0 < ∫ z, g z ∂gaussianReal 0 1 := by
    rw [integral_pos_iff_support_of_nonneg hgnonneg hgint]
    have hsupp : Function.support g = Set.univ := by
      ext z
      simp only [Function.mem_support, Set.mem_univ, iff_true]
      exact ne_of_gt (by
        dsimp [g]
        exact sub_pos.mpr (Real.tanh_sq_lt_one _))
    rw [hsupp]
    simp
  have htanhInt : Integrable (fun z => Real.tanh (X z) ^ 2)
      (gaussianReal 0 1) := by
    apply Integrable.of_bound (C := 1)
    · exact ((continuous_tanh'.comp (by fun_prop)).pow 2).aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact (Real.tanh_sq_lt_one _).le
  have hgzero : ∫ z, g z ∂gaussianReal 0 1 = 0 := by
    have htint : ∫ z, Real.tanh (X z) ^ 2 ∂gaussianReal 0 1 = 1 := by
      simpa [IsRSFixedPoint, standardGaussianExpectation, X] using hfp.symm
    rw [show g = fun z => 1 - Real.tanh (X z) ^ 2 by rfl,
      integral_sub (integrable_const 1) htanhInt, htint]
    norm_num
  linarith

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
