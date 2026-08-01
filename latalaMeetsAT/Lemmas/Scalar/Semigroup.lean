import Lemmas.FreeEnergyDerivative
import Mathlib.MeasureTheory.Group.IntegralConvolution
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

open MeasureTheory ProbabilityTheory Real
open scoped MeasureTheory NNReal

set_option autoImplicit false

namespace SpinGlass.AT

/-- Explicit two-piece scalar semigroup used by the RS trial path. -/
noncomputable def scalarPsi (β q s u x : ℝ) : ℝ :=
  if q ≤ u then Real.log (Real.cosh x) + s * β ^ 2 / 2 * (1 - u)
  else standardGaussianExpectation (fun z =>
    Real.log (Real.cosh (x + β * Real.sqrt s * Real.sqrt (q - u) * z)) +
      s * β ^ 2 / 2 * (1 - q))

theorem scalarPsi_eq_upper {β q s u x : ℝ} (hu : q ≤ u) :
    scalarPsi β q s u x =
      Real.log (Real.cosh x) + s * β ^ 2 / 2 * (1 - u) := by
  -- Proof route: unfold `scalarPsi`; `hu` selects the upper branch.
  simp [scalarPsi, hu]

theorem scalarPsi_eq_lower {β q s u x : ℝ} (hu : u < q) :
    scalarPsi β q s u x = standardGaussianExpectation (fun z =>
      Real.log (Real.cosh (x + β * Real.sqrt s * Real.sqrt (q - u) * z)) +
        s * β ^ 2 / 2 * (1 - q)) := by
  -- Proof route: unfold `scalarPsi`; `not_le_of_gt hu` selects the lower branch.
  simp [scalarPsi, not_le_of_gt hu]

noncomputable def scalarTrialValue (β h q s : ℝ) : ℝ :=
  Real.log 2 + standardGaussianExpectation (fun z =>
    scalarPsi β q s 0 (h + β * Real.sqrt ((1 - s) * q) * z)) -
      s * β ^ 2 / 2 * ((1 - q ^ 2) / 2)

private lemma log_cosh_nonneg (x : ℝ) : 0 ≤ Real.log (Real.cosh x) :=
  Real.log_nonneg (Real.one_le_cosh x)

private lemma log_cosh_le_sq (x : ℝ) : Real.log (Real.cosh x) ≤ x ^ 2 / 2 := by
  exact (Real.log_le_iff_le_exp (Real.cosh_pos x)).2 (Real.cosh_le_exp_half_sq x)

private lemma integrable_log_cosh_add_gaussian (h m : ℝ) (v : ℝ≥0) :
    Integrable (fun x : ℝ => Real.log (Real.cosh (h + x))) (gaussianReal m v) := by
  have hid : Integrable (fun x : ℝ => |x| ^ 2) (gaussianReal m v) := by
    simpa only [Real.norm_eq_abs, id_eq] using
      (memLp_id_gaussianReal (μ := m) (v := v) (2 : ℝ≥0)).integrable_norm_pow'
  have hg : Integrable (fun x : ℝ => 2 * h ^ 2 + 2 * |x| ^ 2) (gaussianReal m v) := by
    exact (integrable_const (2 * h ^ 2)).add (hid.const_mul 2)
  have hcosh : Continuous (fun x : ℝ => Real.cosh (h + x)) :=
    Real.continuous_cosh.comp (continuous_const.add continuous_id)
  have hc : Continuous (fun x : ℝ => Real.log (Real.cosh (h + x))) :=
    hcosh.log fun x => (Real.cosh_pos (h + x)).ne'
  refine hg.mono' hc.aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (log_cosh_nonneg (h + x))]
  calc
    Real.log (Real.cosh (h + x)) ≤ (h + x) ^ 2 / 2 := log_cosh_le_sq _
    _ ≤ 2 * h ^ 2 + 2 * |x| ^ 2 := by nlinarith [sq_nonneg (h - x), sq_abs x]

private lemma gaussian_convolution_log_cosh_add_const (h k a b c : ℝ)
    (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    (∫ x, ∫ y, Real.log (Real.cosh (h + a * x + b * y)) + k ∂gaussianReal 0 1
      ∂gaussianReal 0 1) =
      ∫ z, Real.log (Real.cosh (h + c * z)) + k ∂gaussianReal 0 1 := by
  let va : ℝ≥0 := NNReal.mk (a ^ 2) (sq_nonneg a) * 1
  let vb : ℝ≥0 := NNReal.mk (b ^ 2) (sq_nonneg b) * 1
  let vc : ℝ≥0 := NNReal.mk (c ^ 2) (sq_nonneg c) * 1
  have hma : Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1) =
      gaussianReal 0 va := by
    simpa [va] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : ℝ≥0)) a)
  have hmb : Measure.map (fun x : ℝ => b * x) (gaussianReal 0 1) =
      gaussianReal 0 vb := by
    simpa [vb] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : ℝ≥0)) b)
  have hmc : Measure.map (fun x : ℝ => c * x) (gaussianReal 0 1) =
      gaussianReal 0 vc := by
    simpa [vc] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : ℝ≥0)) c)
  have hv : va + vb = vc := by
    apply NNReal.eq
    simp [va, vb, vc, hc]
  have hf : Integrable (fun z : ℝ => Real.log (Real.cosh (h + z)) + k)
      (gaussianReal 0 va ∗ gaussianReal 0 vb) := by
    rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    change Integrable ((fun z : ℝ => Real.log (Real.cosh (h + z))) + fun _ => k)
      (gaussianReal 0 vc)
    exact (integrable_log_cosh_add_gaussian h 0 vc).add (integrable_const k)
  have hprod : Integrable (fun p : ℝ × ℝ =>
      Real.log (Real.cosh (h + (p.1 + p.2))) + k)
      ((gaussianReal 0 va).prod (gaussianReal 0 vb)) := by
    rw [Measure.conv] at hf
    exact (integrable_map_measure hf.1 (by fun_prop)).mp hf
  have houter : AEStronglyMeasurable
      (fun x : ℝ => ∫ y, Real.log (Real.cosh (h + (x + y))) + k ∂gaussianReal 0 vb)
      (gaussianReal 0 va) := hprod.integral_prod_left.1
  have hinner (x : ℝ) :
      (∫ y, Real.log (Real.cosh (h + a * x + b * y)) + k ∂gaussianReal 0 1) =
        ∫ y, Real.log (Real.cosh (h + a * x + y)) + k ∂gaussianReal 0 vb := by
    have hc' : Continuous (fun y : ℝ => Real.log (Real.cosh (h + a * x + y)) + k) := by
      have hcosh : Continuous (fun y : ℝ => Real.cosh (h + a * x + y)) := by fun_prop
      exact (hcosh.log fun y => (Real.cosh_pos (h + a * x + y)).ne').add continuous_const
    have hm : AEStronglyMeasurable (fun y : ℝ =>
        Real.log (Real.cosh (h + a * x + y)) + k)
        (Measure.map (fun y : ℝ => b * y) (gaussianReal 0 1)) :=
      hc'.aestronglyMeasurable
    rw [← hmb, integral_map (by fun_prop) hm]
  have houter_map :
      (∫ x, ∫ y, Real.log (Real.cosh (h + a * x + y)) + k ∂gaussianReal 0 vb
        ∂gaussianReal 0 1) =
        ∫ x, ∫ y, Real.log (Real.cosh (h + x + y)) + k ∂gaussianReal 0 vb
          ∂gaussianReal 0 va := by
    have hm : AEStronglyMeasurable
        (fun x : ℝ => ∫ y, Real.log (Real.cosh (h + (x + y))) + k ∂gaussianReal 0 vb)
        (Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1)) := by
      simpa [hma] using houter
    rw [← hma]
    simpa only [add_assoc] using (integral_map (by fun_prop) hm).symm
  calc
    (∫ x, ∫ y, Real.log (Real.cosh (h + a * x + b * y)) + k ∂gaussianReal 0 1
        ∂gaussianReal 0 1) =
        ∫ x, ∫ y, Real.log (Real.cosh (h + x + y)) + k ∂gaussianReal 0 vb
          ∂gaussianReal 0 va := by
            rw [integral_congr_ae (Filter.Eventually.of_forall hinner)]
            exact houter_map
    _ = ∫ z, Real.log (Real.cosh (h + z)) + k
          ∂(gaussianReal 0 va ∗ gaussianReal 0 vb) := by
            simpa only [add_assoc] using (integral_conv hf).symm
    _ = ∫ z, Real.log (Real.cosh (h + z)) + k ∂gaussianReal 0 vc := by
          rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    _ = ∫ z, Real.log (Real.cosh (h + c * z)) + k ∂gaussianReal 0 1 := by
          rw [← hmc, integral_map (by fun_prop)]
          have hcosh : Continuous (fun z : ℝ => Real.cosh (h + z)) := by fun_prop
          exact ((hcosh.log fun z => (Real.cosh_pos (h + z)).ne').add
            continuous_const).aestronglyMeasurable

theorem scalarTrialValue_eq (β h q s : ℝ)
    (hq : 0 ≤ q) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    scalarTrialValue β h q s = rsPathValue β h q s := by
  by_cases hq0 : q = 0
  · subst q
    simp [scalarTrialValue, scalarPsi, rsPathValue, standardGaussianExpectation]
    ring
  have hqpos : 0 < q := lt_of_le_of_ne hq (Ne.symm hq0)
  have hs0 : 0 ≤ s := hs.1
  have h1s : 0 ≤ 1 - s := sub_nonneg.mpr hs.2
  have hsq : (β * Real.sqrt q) ^ 2 =
      (β * Real.sqrt ((1 - s) * q)) ^ 2 +
        (β * Real.sqrt s * Real.sqrt q) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hq, mul_pow, Real.sq_sqrt (mul_nonneg h1s hq),
      mul_pow, mul_pow, Real.sq_sqrt hs0, Real.sq_sqrt hq]
    ring
  let k := s * β ^ 2 / 2 * (1 - q)
  have hconv := gaussian_convolution_log_cosh_add_const h k
    (β * Real.sqrt ((1 - s) * q)) (β * Real.sqrt s * Real.sqrt q)
      (β * Real.sqrt q) hsq
  unfold scalarTrialValue rsPathValue standardGaussianExpectation
  simp only [scalarPsi, if_neg (not_le_of_gt hqpos), sub_zero]
  simp only [standardGaussianExpectation]
  dsimp [k] at hconv
  rw [hconv]
  rw [integral_add]
  · simp only [integral_const, probReal_univ, one_smul]
    ring
  · have hmap := integrable_log_cosh_add_gaussian h 0
      (NNReal.mk ((β * Real.sqrt q) ^ 2) (sq_nonneg _))
    have hm : Measure.map (fun z : ℝ => β * Real.sqrt q * z) (gaussianReal 0 1) =
        gaussianReal 0 (NNReal.mk ((β * Real.sqrt q) ^ 2) (sq_nonneg _)) := by
      simpa using (gaussianReal_map_const_mul (μ := 0) (v := (1 : ℝ≥0))
        (β * Real.sqrt q))
    rw [← hm] at hmap
    exact (integrable_map_measure hmap.1 (by fun_prop)).mp hmap
  · exact integrable_const k

end SpinGlass.AT
