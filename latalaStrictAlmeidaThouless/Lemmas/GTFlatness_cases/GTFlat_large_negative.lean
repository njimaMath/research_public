import Lemmas.GTFlatness_cases.GTFlatnessCore
import Mathlib.MeasureTheory.Integral.MeanInequalities

open MeasureTheory ProbabilityTheory Set

namespace SpinGlass.AT

/-! ### Large negative overlaps `-1 ≤ v ≤ -q` -/

/-- On the large-negative-overlap branch, the distance to `q` is at most two. -/
lemma sub_sq_le_four_of_negative_overlap
    {q v : ℝ}
    (hq0 : 0 ≤ q)
    (hv : v ∈ Icc (-1 : ℝ) (-q)) :
    (v - q) ^ 2 ≤ 4 := by
  have hv_lower : -1 ≤ v := hv.1
  have hv_upper : v ≤ -q := hv.2
  have hq1 : q ≤ 1 := by
    linarith
  have hdiff_lower : -2 ≤ v - q := by
    linarith
  have hdiff_upper : v - q ≤ 0 := by
    linarith
  have hprod :
      0 ≤ ((v - q) + 2) * (2 - (v - q)) := by
    apply mul_nonneg
    · linarith
    · linarith
  nlinarith

/-- The terminal-zero diagonal step at unit time has an explicit form. -/
lemma flatness_diagonal_one_terminal_zero
    (b x₁ x₂ : ℝ) :
    gtDiagonalStep 1 b (gtTerminal 0) x₁ x₂ =
      b ^ 2
        + Real.log (Real.cosh x₁)
        + Real.log (Real.cosh x₂) := by
  rw [gtDiagonalStep_one_terminal, gtTerminal_zero]
  ring

private lemma flatness_integrable_cosh_affine
    (x a : ℝ) :
    Integrable
      (fun z : ℝ => Real.cosh (x + a * z))
      (gaussianReal 0 1) := by
  have hp :
      Integrable
        (fun z : ℝ => Real.exp (x + a * z))
        (gaussianReal 0 1) := by
    have h :=
      integrable_exp_mul_gaussianReal
        (μ := 0) (v := (1 : NNReal)) a
    have h' := h.const_mul (Real.exp x)
    convert h' using 1
    funext z
    rw [Real.exp_add]

  have hm :
      Integrable
        (fun z : ℝ => Real.exp (-(x + a * z)))
        (gaussianReal 0 1) := by
    have h :=
      integrable_exp_mul_gaussianReal
        (μ := 0) (v := (1 : NNReal)) (-a)
    have h' := h.const_mul (Real.exp (-x))
    convert h' using 1
    funext z
    rw [show -(x + a * z) = -x + (-a) * z by ring, Real.exp_add]

  have hadd := hp.add hm
  have hhalf := hadd.const_mul (1 / 2 : ℝ)
  convert hhalf using 1
  funext z
  simp only [Pi.add_apply]
  rw [Real.cosh_eq]
  ring

private lemma flatness_standardGaussian_exp_mul
    (t : ℝ) :
    standardGaussianExpectation
        (fun z : ℝ => Real.exp (t * z)) =
      Real.exp (t ^ 2 / 2) := by
  unfold standardGaussianExpectation
  have h :=
    congrFun
      (mgf_id_gaussianReal
        (μ := 0) (v := (1 : NNReal))) t
  simpa [mgf] using h

private lemma flatness_standardGaussian_cosh
    (x a : ℝ) :
    standardGaussianExpectation
        (fun z : ℝ => Real.cosh (x + a * z)) =
      Real.exp (a ^ 2 / 2) * Real.cosh x := by
  have hp :
      standardGaussianExpectation
          (fun z : ℝ => Real.exp (x + a * z)) =
        Real.exp x * Real.exp (a ^ 2 / 2) := by
    unfold standardGaussianExpectation
    calc
      (∫ z, Real.exp (x + a * z) ∂gaussianReal 0 1) =
          ∫ z, Real.exp x * Real.exp (a * z) ∂gaussianReal 0 1 := by
            apply integral_congr_ae
            filter_upwards [] with z
            rw [← Real.exp_add]
      _ = Real.exp x * ∫ z, Real.exp (a * z) ∂gaussianReal 0 1 := by
          rw [integral_const_mul]
      _ = Real.exp x * Real.exp (a ^ 2 / 2) := by
          change Real.exp x *
          standardGaussianExpectation (fun z => Real.exp (a * z)) = _
          rw [flatness_standardGaussian_exp_mul]

  have hm :
      standardGaussianExpectation
          (fun z : ℝ => Real.exp (-(x + a * z))) =
        Real.exp (-x) * Real.exp (a ^ 2 / 2) := by
    unfold standardGaussianExpectation
    calc
      (∫ z, Real.exp (-(x + a * z)) ∂gaussianReal 0 1) =
          ∫ z, Real.exp (-x) * Real.exp ((-a) * z)
            ∂gaussianReal 0 1 := by
            apply integral_congr_ae
            filter_upwards [] with z
            rw [show -(x + a * z) = -x + (-a) * z by ring, Real.exp_add]
      _ = Real.exp (-x) * ∫ z, Real.exp ((-a) * z) ∂gaussianReal 0 1 := by
          rw [integral_const_mul]
      _ = Real.exp (-x) * Real.exp (a ^ 2 / 2) := by
          change Real.exp (-x) *
          standardGaussianExpectation (fun z => Real.exp ((-a) * z)) = _
          rw [flatness_standardGaussian_exp_mul]
          ring_nf

  have hplus := flatness_integrable_cosh_affine x a
  unfold standardGaussianExpectation at hp hm ⊢
  rw [show
      (fun z : ℝ => Real.cosh (x + a * z)) =
        fun z =>
          (1 / 2 : ℝ) *
            (Real.exp (x + a * z) + Real.exp (-(x + a * z))) by
      funext z
      rw [Real.cosh_eq]
      ring]
  rw [integral_const_mul]

  have hip :
      Integrable
        (fun z : ℝ => Real.exp (x + a * z))
        (gaussianReal 0 1) := by
    have h :=
      integrable_exp_mul_gaussianReal
        (μ := 0) (v := (1 : NNReal)) a
    have h' := h.const_mul (Real.exp x)
    convert h' using 1
    funext z
    rw [Real.exp_add]

  have him :
      Integrable
        (fun z : ℝ => Real.exp (-(x + a * z)))
        (gaussianReal 0 1) := by
    have h :=
      integrable_exp_mul_gaussianReal
        (μ := 0) (v := (1 : NNReal)) (-a)
    have h' := h.const_mul (Real.exp (-x))
    convert h' using 1
    funext z
    rw [show -(x + a * z) = -x + (-a) * z by ring, Real.exp_add]

  rw [integral_add hip him, hp, hm, Real.cosh_eq]
  ring

private lemma flatness_standardGaussian_cauchy_schwarz
    (f g : ℝ → ℝ)
    (hf : MemLp f 2 (gaussianReal 0 1))
    (hg : MemLp g 2 (gaussianReal 0 1))
    (hf0 : ∀ z, 0 ≤ f z)
    (hg0 : ∀ z, 0 ≤ g z) :
    standardGaussianExpectation (fun z => f z * g z) ≤
      Real.sqrt
          (standardGaussianExpectation (fun z => f z ^ 2)) *
        Real.sqrt
          (standardGaussianExpectation (fun z => g z ^ 2)) := by
  unfold standardGaussianExpectation
  have h22 : (2 : ℝ).HolderConjugate 2 := by
    exact Real.holderConjugate_iff.mpr
      ⟨by norm_num, by norm_num⟩
  have hf' :
      MemLp f (ENNReal.ofReal (2 : ℝ))
        (gaussianReal 0 1) := by
    simpa using hf
  have hg' :
      MemLp g (ENNReal.ofReal (2 : ℝ))
        (gaussianReal 0 1) := by
    simpa using hg
  have h :=
    integral_mul_le_Lp_mul_Lq_of_nonneg
      (μ := gaussianReal 0 1)
      h22
      (Filter.Eventually.of_forall hf0)
      (Filter.Eventually.of_forall hg0)
      hf' hg'
  simpa [Real.sqrt_eq_rpow] using h

lemma flatness_negative_half_step_le
    (a b x₁ x₂ : ℝ) :
    gtRankOneStep (1 / 2) a (-1)
        (gtDiagonalStep 1 b (gtTerminal 0))
        x₁ x₂
      ≤
    b ^ 2 + a ^ 2
      + Real.log (Real.cosh x₁)
      + Real.log (Real.cosh x₂) := by
  let f : ℝ → ℝ := fun z =>
    Real.exp ((1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)))
  let g : ℝ → ℝ := fun z =>
    Real.exp ((1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)))
  have hf_sq (z : ℝ) :
      f z ^ 2 = Real.cosh (x₁ + a * z) := by
    dsimp [f]
    rw [pow_two, ← Real.exp_add]
    rw [show
      (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) =
        Real.log (Real.cosh (x₁ + a * z)) by ring]
    exact Real.exp_log (Real.cosh_pos _)
  have hg_sq (z : ℝ) :
      g z ^ 2 = Real.cosh (x₂ + (-1) * a * z) := by
    dsimp [g]
    rw [pow_two, ← Real.exp_add]
    rw [show
      (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) =
        Real.log (Real.cosh (x₂ + (-1) * a * z) ) by ring]
    exact Real.exp_log (Real.cosh_pos _)
  have hfcont : Continuous f := by
    have hfun : f = fun z => Real.sqrt (Real.cosh (x₁ + a * z)) := by
      funext z
      dsimp [f]
      rw [Real.sqrt_eq_rpow,
        Real.rpow_def_of_pos (Real.cosh_pos (x₁ + a * z))]
      congr 1
      ring
    rw [hfun]
    exact Real.continuous_sqrt.comp (by fun_prop)
  have hgcont : Continuous g := by
    have hfun : g = fun z => Real.sqrt (Real.cosh (x₂ + (-1) * a * z)) := by
      funext z
      dsimp [g]
      rw [Real.sqrt_eq_rpow,
        Real.rpow_def_of_pos (Real.cosh_pos (x₂ + (-1) * a * z))]
      congr 1
      ring
    rw [hfun]
    exact Real.continuous_sqrt.comp (by fun_prop)
  have hf_sq_int :
      Integrable (fun z => f z ^ 2) (gaussianReal 0 1) := by
    have h := flatness_integrable_cosh_affine x₁ a
    exact h.congr (Filter.Eventually.of_forall fun z => (hf_sq z).symm)
  have hg_sq_int :
      Integrable (fun z => g z ^ 2) (gaussianReal 0 1) := by
    have h := flatness_integrable_cosh_affine x₂ (-a)
    exact h.congr (Filter.Eventually.of_forall fun z => by
      calc
        Real.cosh (x₂ + (-a) * z) =
            Real.cosh (x₂ + (-1) * a * z) := by
              congr 2
              ring
        _ = g z ^ 2 := (hg_sq z).symm)
  have hfmem : MemLp f 2 (gaussianReal 0 1) := by
    exact (memLp_two_iff_integrable_sq hfcont.aestronglyMeasurable).2 hf_sq_int
  have hgmem : MemLp g 2 (gaussianReal 0 1) := by
    exact (memLp_two_iff_integrable_sq hgcont.aestronglyMeasurable).2 hg_sq_int
  have hf0 (z : ℝ) : 0 ≤ f z := by
    dsimp [f]
    positivity
  have hg0 (z : ℝ) : 0 ≤ g z := by
    dsimp [g]
    positivity
  let I : ℝ := standardGaussianExpectation (fun z => f z * g z)
  have hcs :
      I ≤
        Real.sqrt (Real.exp (a ^ 2 / 2) * Real.cosh x₁) *
          Real.sqrt (Real.exp (a ^ 2 / 2) * Real.cosh x₂) := by
    have h := flatness_standardGaussian_cauchy_schwarz f g hfmem hgmem hf0 hg0
    have hfE :
        standardGaussianExpectation (fun z => f z ^ 2) =
          Real.exp (a ^ 2 / 2) * Real.cosh x₁ := by
      calc
        standardGaussianExpectation (fun z => f z ^ 2) =
            standardGaussianExpectation (fun z => Real.cosh (x₁ + a * z)) := by
              apply congrArg standardGaussianExpectation
              funext z
              exact hf_sq z
        _ = Real.exp (a ^ 2 / 2) * Real.cosh x₁ :=
          flatness_standardGaussian_cosh x₁ a
    have hgE :
        standardGaussianExpectation (fun z => g z ^ 2) =
          Real.exp (a ^ 2 / 2) * Real.cosh x₂ := by
      calc
        standardGaussianExpectation (fun z => g z ^ 2) =
            standardGaussianExpectation (fun z => Real.cosh (x₂ + (-a) * z)) := by
              apply congrArg standardGaussianExpectation
              funext z
              rw [hg_sq]
              congr 2
              ring
        _ = Real.exp ((-a) ^ 2 / 2) * Real.cosh x₂ :=
          flatness_standardGaussian_cosh x₂ (-a)
        _ = Real.exp (a ^ 2 / 2) * Real.cosh x₂ := by ring_nf
    simpa [I, hfE, hgE] using h
  let C : ℝ :=
    a ^ 2 + Real.log (Real.cosh x₁) + Real.log (Real.cosh x₂)
  have hR :
      Real.sqrt (Real.exp (a ^ 2 / 2) * Real.cosh x₁) *
          Real.sqrt (Real.exp (a ^ 2 / 2) * Real.cosh x₂) =
        Real.exp (C / 2) := by
    have hA : 0 < Real.exp (a ^ 2 / 2) * Real.cosh x₁ := by positivity
    have hB : 0 < Real.exp (a ^ 2 / 2) * Real.cosh x₂ := by positivity
    rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
    rw [Real.rpow_def_of_pos hA, Real.rpow_def_of_pos hB]
    rw [← Real.exp_add]
    congr 1
    rw [Real.log_mul (Real.exp_ne_zero _) (Real.cosh_pos x₁).ne']
    rw [Real.log_mul (Real.exp_ne_zero _) (Real.cosh_pos x₂).ne']
    rw [Real.log_exp]
    dsimp [C]
    ring
  have hIle : I ≤ Real.exp (C / 2) := by
    rw [← hR]
    exact hcs
  have hIint :
      Integrable (fun z => f z * g z) (gaussianReal 0 1) := by
    have hdom :
        Integrable
          (fun z => (1 / 2 : ℝ) * (f z ^ 2 + g z ^ 2))
          (gaussianReal 0 1) :=
      (hf_sq_int.add hg_sq_int).const_mul (1 / 2 : ℝ)
    refine hdom.mono' (hfcont.mul hgcont).aestronglyMeasurable ?_
    filter_upwards [] with z
    have hs := sq_nonneg (f z - g z)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hf0 z) (hg0 z))]
    nlinarith
  have hIpos : 0 < I := by
    dsimp [I, standardGaussianExpectation]
    rw [integral_pos_iff_support_of_nonneg
      (fun z => mul_nonneg (hf0 z) (hg0 z)) hIint]
    have hsupp : Function.support (fun z => f z * g z) = Set.univ := by
      ext z
      simp only [Function.mem_support, Set.mem_univ, iff_true]
      exact mul_ne_zero
        (by dsimp [f]; exact (Real.exp_pos _).ne')
        (by dsimp [g]; exact (Real.exp_pos _).ne')
    rw [hsupp]
    simp
  let J : ℝ :=
    standardGaussianExpectation (fun z =>
      Real.exp ((1 / 2 : ℝ) *
        gtDiagonalStep 1 b (gtTerminal 0)
          (x₁ + a * z) (x₂ + (-1) * a * z)))
  have hJ : J = Real.exp (b ^ 2 / 2) * I := by
    dsimp [J, I, standardGaussianExpectation]
    rw [← integral_const_mul]
    apply integral_congr_ae
    filter_upwards [] with z
    rw [gtDiagonalStep_one_terminal, gtTerminal_zero]
    dsimp [f, g]
    rw [show
      (1 / 2 : ℝ) *
          (Real.log (Real.cosh (x₁ + a * z)) +
            Real.log (Real.cosh (x₂ + (-1) * a * z)) + b ^ 2) =
        b ^ 2 / 2 +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) by ring]
    rw [Real.exp_add, Real.exp_add]
    ring
  have hJpos : 0 < J := by
    rw [hJ]
    exact mul_pos (Real.exp_pos _) hIpos
  have hJle : J ≤ Real.exp ((b ^ 2 + C) / 2) := by
    rw [hJ]
    calc
      Real.exp (b ^ 2 / 2) * I ≤
          Real.exp (b ^ 2 / 2) * Real.exp (C / 2) := by
            exact mul_le_mul_of_nonneg_left hIle (Real.exp_nonneg _)
      _ = Real.exp ((b ^ 2 + C) / 2) := by
          rw [← Real.exp_add]
          congr 1
          ring
  have hlog : Real.log J ≤ (b ^ 2 + C) / 2 :=
    (Real.log_le_iff_le_exp hJpos).2 hJle
  have hmain :
      2 * Real.log J ≤
        b ^ 2 + a ^ 2 + Real.log (Real.cosh x₁) + Real.log (Real.cosh x₂) := by
    dsimp [C] at hlog
    nlinarith
  simpa [gtRankOneStep, J] using hmain

private lemma flatness_negative_half_step_lt_of_nontrivial
    (a b x₁ x₂ : ℝ)
    (ha : a ≠ 0)
    (hx : x₁ ≠ -x₂) :
    gtRankOneStep (1 / 2) a (-1)
        (gtDiagonalStep 1 b (gtTerminal 0))
        x₁ x₂
      <
    b ^ 2 + a ^ 2
      + Real.log (Real.cosh x₁)
      + Real.log (Real.cosh x₂) := by
  let f : ℝ → ℝ := fun z =>
    Real.exp ((1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)))
  let g : ℝ → ℝ := fun z =>
    Real.exp ((1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)))
  have hf_sq (z : ℝ) :
      f z ^ 2 = Real.cosh (x₁ + a * z) := by
    dsimp [f]
    rw [pow_two, ← Real.exp_add]
    rw [show
      (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) =
        Real.log (Real.cosh (x₁ + a * z)) by ring]
    exact Real.exp_log (Real.cosh_pos _)
  have hg_sq (z : ℝ) :
      g z ^ 2 = Real.cosh (x₂ + (-1) * a * z) := by
    dsimp [g]
    rw [pow_two, ← Real.exp_add]
    rw [show
      (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) =
        Real.log (Real.cosh (x₂ + (-1) * a * z)) by ring]
    exact Real.exp_log (Real.cosh_pos _)
  have hfcont : Continuous f := by
    have hfun : f = fun z => Real.sqrt (Real.cosh (x₁ + a * z)) := by
      funext z
      dsimp [f]
      rw [Real.sqrt_eq_rpow,
        Real.rpow_def_of_pos (Real.cosh_pos (x₁ + a * z))]
      congr 1
      ring
    rw [hfun]
    exact Real.continuous_sqrt.comp (by fun_prop)
  have hgcont : Continuous g := by
    have hfun : g = fun z => Real.sqrt (Real.cosh (x₂ + (-1) * a * z)) := by
      funext z
      dsimp [g]
      rw [Real.sqrt_eq_rpow,
        Real.rpow_def_of_pos (Real.cosh_pos (x₂ + (-1) * a * z))]
      congr 1
      ring
    rw [hfun]
    exact Real.continuous_sqrt.comp (by fun_prop)
  have hf_sq_int :
      Integrable (fun z => f z ^ 2) (gaussianReal 0 1) := by
    have h := flatness_integrable_cosh_affine x₁ a
    exact h.congr (Filter.Eventually.of_forall fun z => (hf_sq z).symm)
  have hg_sq_int :
      Integrable (fun z => g z ^ 2) (gaussianReal 0 1) := by
    have h := flatness_integrable_cosh_affine x₂ (-a)
    exact h.congr (Filter.Eventually.of_forall fun z => by
      calc
        Real.cosh (x₂ + (-a) * z) =
            Real.cosh (x₂ + (-1) * a * z) := by
              congr 2 <;> ring
        _ = g z ^ 2 := (hg_sq z).symm)
  have hf0 (z : ℝ) : 0 ≤ f z := by
    dsimp [f]
    positivity
  have hg0 (z : ℝ) : 0 ≤ g z := by
    dsimp [g]
    positivity
  have hIint :
      Integrable (fun z => f z * g z) (gaussianReal 0 1) := by
    have hdom :
        Integrable (fun z => (1 / 2 : ℝ) * (f z ^ 2 + g z ^ 2))
          (gaussianReal 0 1) :=
      (hf_sq_int.add hg_sq_int).const_mul (1 / 2 : ℝ)
    refine hdom.mono' (hfcont.mul hgcont).aestronglyMeasurable ?_
    filter_upwards [] with z
    have hs := sq_nonneg (f z - g z)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hf0 z) (hg0 z))]
    have hright : 0 ≤ (1 / 2 : ℝ) * (f z ^ 2 + g z ^ 2) := by
      positivity
    rw [Real.norm_eq_abs, abs_of_nonneg hright]
    nlinarith
  let E₁ : ℝ := Real.exp (a ^ 2 / 2) * Real.cosh x₁
  let E₂ : ℝ := Real.exp (a ^ 2 / 2) * Real.cosh x₂
  have hE₁pos : 0 < E₁ := by
    dsimp [E₁]
    positivity
  have hE₂pos : 0 < E₂ := by
    dsimp [E₂]
    positivity
  have hfE : (∫ z, f z ^ 2 ∂gaussianReal 0 1) = E₁ := by
    change standardGaussianExpectation (fun z => f z ^ 2) = E₁
    calc
      standardGaussianExpectation (fun z => f z ^ 2) =
          standardGaussianExpectation (fun z => Real.cosh (x₁ + a * z)) := by
            apply congrArg standardGaussianExpectation
            funext z
            exact hf_sq z
      _ = Real.exp (a ^ 2 / 2) * Real.cosh x₁ :=
        flatness_standardGaussian_cosh x₁ a
      _ = E₁ := rfl
  have hgE : (∫ z, g z ^ 2 ∂gaussianReal 0 1) = E₂ := by
    change standardGaussianExpectation (fun z => g z ^ 2) = E₂
    calc
      standardGaussianExpectation (fun z => g z ^ 2) =
          standardGaussianExpectation (fun z => Real.cosh (x₂ + (-a) * z)) := by
            apply congrArg standardGaussianExpectation
            funext z
            rw [hg_sq]
            congr 2
            ring
      _ = Real.exp ((-a) ^ 2 / 2) * Real.cosh x₂ :=
        flatness_standardGaussian_cosh x₂ (-a)
      _ = E₂ := by
        dsimp [E₂]
        ring_nf
  let A : ℝ := Real.sqrt E₁
  let B : ℝ := Real.sqrt E₂
  have hApos : 0 < A := by
    dsimp [A]
    exact Real.sqrt_pos.2 hE₁pos
  have hBpos : 0 < B := by
    dsimp [B]
    exact Real.sqrt_pos.2 hE₂pos
  have hA_sq : A ^ 2 = E₁ := by
    dsimp [A]
    exact Real.sq_sqrt hE₁pos.le
  have hB_sq : B ^ 2 = E₂ := by
    dsimp [B]
    exact Real.sq_sqrt hE₂pos.le
  let I : ℝ := ∫ z, f z * g z ∂gaussianReal 0 1
  have hIpos : 0 < I := by
    dsimp [I]
    rw [integral_pos_iff_support_of_nonneg
      (fun z => mul_nonneg (hf0 z) (hg0 z)) hIint]
    have hsupp : Function.support (fun z => f z * g z) = Set.univ := by
      ext z
      simp only [Function.mem_support, Set.mem_univ, iff_true]
      exact mul_ne_zero
        (by dsimp [f]; exact (Real.exp_pos _).ne')
        (by dsimp [g]; exact (Real.exp_pos _).ne')
    rw [hsupp]
    simp
  let D : ℝ → ℝ := fun z => B * f z - A * g z
  have hDcont : Continuous D := by
    dsimp [D]
    fun_prop
  have hBf :
      Integrable (fun z => B ^ 2 * f z ^ 2) (gaussianReal 0 1) :=
    hf_sq_int.const_mul (B ^ 2)
  have hcross :
      Integrable (fun z => (2 * A * B) * (f z * g z)) (gaussianReal 0 1) :=
    hIint.const_mul (2 * A * B)
  have hAg :
      Integrable (fun z => A ^ 2 * g z ^ 2) (gaussianReal 0 1) :=
    hg_sq_int.const_mul (A ^ 2)
  have hDsq_int :
      Integrable (fun z => D z ^ 2) (gaussianReal 0 1) := by
    have h := (hBf.sub hcross).add hAg
    exact h.congr (Filter.Eventually.of_forall fun z => by
      dsimp [D]
      ring)
  have hD1ne : D 1 ≠ 0 := by
    intro hD1
    have hlin : B * f 1 = A * g 1 := by
      dsimp [D] at hD1
      linarith
    have hsq := congrArg (fun t : ℝ => t ^ 2) hlin
    rw [mul_pow, mul_pow, hB_sq, hA_sq, hf_sq, hg_sq] at hsq
    have hsq' :
        E₂ * Real.cosh (x₁ + a) = E₁ * Real.cosh (x₂ - a) := by
      simpa [sub_eq_add_neg] using hsq
    have hfactor :
        Real.exp (a ^ 2 / 2) * (Real.cosh x₂ * Real.cosh (x₁ + a)) =
          Real.exp (a ^ 2 / 2) * (Real.cosh x₁ * Real.cosh (x₂ - a)) := by
      simpa [E₁, E₂, mul_assoc] using hsq'
    have hcosh :
        Real.cosh x₂ * Real.cosh (x₁ + a) =
          Real.cosh x₁ * Real.cosh (x₂ - a) := by
      exact mul_left_cancel₀ (Real.exp_ne_zero (a ^ 2 / 2)) hfactor
    have hhyper :
        Real.cosh x₂ * Real.cosh (x₁ + a) -
            Real.cosh x₁ * Real.cosh (x₂ - a) =
          Real.sinh (x₁ + x₂) * Real.sinh a := by
      rw [show x₂ - a = x₂ + (-a) by ring]
      simp_rw [Real.cosh_eq, Real.sinh_eq]
      rw [show -(x₁ + a) = -x₁ + (-a) by ring]
      rw [show -(x₂ + -a) = -x₂ + a by ring]
      rw [show -(x₁ + x₂) = -x₁ + (-x₂) by ring]
      simp_rw [Real.exp_add]
      ring
    have hprod : Real.sinh (x₁ + x₂) * Real.sinh a = 0 := by
      rw [← hhyper, hcosh]
      ring
    rcases mul_eq_zero.mp hprod with hsum | ha'
    · have hsum0 : x₁ + x₂ = 0 := by
        apply Real.sinh_injective
        simpa using hsum
      apply hx
      linarith
    · have ha0 : a = 0 := by
        apply Real.sinh_injective
        simpa using ha'
      exact ha ha0
  have hDpos : 0 < ∫ z, D z ^ 2 ∂gaussianReal 0 1 := by
    exact integral_pos_of_integrable_nonneg_nonzero
      (μ := gaussianReal 0 1)
      (x := (1 : ℝ))
      (hDcont.pow 2)
      hDsq_int
      (fun z => sq_nonneg (D z))
      (pow_ne_zero 2 hD1ne)
  have hDformula :
      (∫ z, D z ^ 2 ∂gaussianReal 0 1) =
        B ^ 2 * E₁ - (2 * A * B) * I + A ^ 2 * E₂ := by
    calc
      (∫ z, D z ^ 2 ∂gaussianReal 0 1) =
          ∫ z, (B ^ 2 * f z ^ 2 - (2 * A * B) * (f z * g z) +
            A ^ 2 * g z ^ 2) ∂gaussianReal 0 1 := by
              apply integral_congr_ae
              filter_upwards [] with z
              dsimp [D]
              ring
      _ = B ^ 2 * (∫ z, f z ^ 2 ∂gaussianReal 0 1) -
          (2 * A * B) * (∫ z, f z * g z ∂gaussianReal 0 1) +
          A ^ 2 * (∫ z, g z ^ 2 ∂gaussianReal 0 1) := by
            rw [integral_add (hBf.sub hcross) hAg]
            rw [integral_sub hBf hcross]
            rw [integral_const_mul, integral_const_mul, integral_const_mul]
      _ = B ^ 2 * E₁ - (2 * A * B) * I + A ^ 2 * E₂ := by
            rw [hfE, hgE]
            rfl
  have hDformula' :
      (∫ z, D z ^ 2 ∂gaussianReal 0 1) =
        2 * (A * B) ^ 2 - 2 * (A * B) * I := by
    rw [hDformula, ← hA_sq, ← hB_sq]
    ring
  have hABpos : 0 < A * B := mul_pos hApos hBpos
  have hIlt : I < A * B := by
    have h := hDpos
    rw [hDformula'] at h
    nlinarith
  let C : ℝ :=
    a ^ 2 + Real.log (Real.cosh x₁) + Real.log (Real.cosh x₂)
  have hAB : A * B = Real.exp (C / 2) := by
    dsimp [A, B]
    rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
    rw [Real.rpow_def_of_pos hE₁pos, Real.rpow_def_of_pos hE₂pos]
    rw [← Real.exp_add]
    congr 1
    dsimp [E₁, E₂, C]
    rw [Real.log_mul (Real.exp_ne_zero _) (Real.cosh_pos x₁).ne']
    rw [Real.log_mul (Real.exp_ne_zero _) (Real.cosh_pos x₂).ne']
    rw [Real.log_exp, Real.log_exp]
    ring
  have hIexp : I < Real.exp (C / 2) := by
    rw [← hAB]
    exact hIlt
  let J : ℝ :=
    standardGaussianExpectation (fun z =>
      Real.exp ((1 / 2 : ℝ) *
        gtDiagonalStep 1 b (gtTerminal 0)
          (x₁ + a * z) (x₂ + (-1) * a * z)))
  have hJ : J = Real.exp (b ^ 2 / 2) * I := by
    dsimp [J, I, standardGaussianExpectation]
    rw [← integral_const_mul]
    apply integral_congr_ae
    filter_upwards [] with z
    rw [gtDiagonalStep_one_terminal, gtTerminal_zero]
    dsimp [f, g]
    rw [show
      (1 / 2 : ℝ) *
          (Real.log (Real.cosh (x₁ + a * z)) +
            Real.log (Real.cosh (x₂ + (-1) * a * z)) + b ^ 2) =
        b ^ 2 / 2 +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₁ + a * z)) +
          (1 / 2 : ℝ) * Real.log (Real.cosh (x₂ + (-1) * a * z)) by ring]
    rw [Real.exp_add, Real.exp_add]
    ring
  have hJpos : 0 < J := by
    rw [hJ]
    exact mul_pos (Real.exp_pos _) hIpos
  have hJlt : J < Real.exp ((b ^ 2 + C) / 2) := by
    rw [hJ]
    calc
      Real.exp (b ^ 2 / 2) * I <
          Real.exp (b ^ 2 / 2) * Real.exp (C / 2) := by
            exact mul_lt_mul_of_pos_left hIexp (Real.exp_pos _)
      _ = Real.exp ((b ^ 2 + C) / 2) := by
          rw [← Real.exp_add]
          congr 1
          ring
  have hlog : Real.log J < (b ^ 2 + C) / 2 :=
    (Real.log_lt_iff_lt_exp hJpos).2 hJlt
  have hmain :
      2 * Real.log J <
        b ^ 2 + a ^ 2 + Real.log (Real.cosh x₁) + Real.log (Real.cosh x₂) := by
    dsimp [C] at hlog
    nlinarith
  simpa [gtRankOneStep, J] using hmain

private lemma flatness_negative_half_step_eq_iff
    (a b x₁ x₂ : ℝ) :
    gtRankOneStep (1 / 2) a (-1)
        (gtDiagonalStep 1 b (gtTerminal 0))
        x₁ x₂
      =
    b ^ 2 + a ^ 2
      + Real.log (Real.cosh x₁)
      + Real.log (Real.cosh x₂)
    ↔
    a = 0 ∨ x₁ = -x₂ := by
  constructor
  · intro heq
    by_cases ha : a = 0
    · exact Or.inl ha
    right
    by_contra hx
    have hlt := flatness_negative_half_step_lt_of_nontrivial a b x₁ x₂ ha hx
    linarith
  · rintro (ha | hx)
    · subst a
      simp [gtRankOneStep, gtDiagonalStep_one_terminal, gtTerminal_zero,
        standardGaussianExpectation]
      ring
    · have hx₂ : x₂ = -x₁ := by
        linarith
      subst x₂
      let J : ℝ :=
        standardGaussianExpectation (fun z =>
          Real.exp ((1 / 2 : ℝ) *
            gtDiagonalStep 1 b (gtTerminal 0)
              (x₁ + a * z) (-x₁ + (-1) * a * z)))
      have hJ :
          J = Real.exp (b ^ 2 / 2) *
              (Real.exp (a ^ 2 / 2) * Real.cosh x₁) := by
        dsimp [J]
        calc
          standardGaussianExpectation (fun z =>
              Real.exp ((1 / 2 : ℝ) *
                gtDiagonalStep 1 b (gtTerminal 0)
                  (x₁ + a * z) (-x₁ + (-1) * a * z))) =
            standardGaussianExpectation (fun z =>
              Real.exp (b ^ 2 / 2) * Real.cosh (x₁ + a * z)) := by
                apply congrArg standardGaussianExpectation
                funext z
                rw [gtDiagonalStep_one_terminal, gtTerminal_zero]
                have hneg : -x₁ + (-1) * a * z = -(x₁ + a * z) := by
                  ring
                rw [hneg, Real.cosh_neg]
                rw [show
                  (1 / 2 : ℝ) *
                      (Real.log (Real.cosh (x₁ + a * z)) +
                        Real.log (Real.cosh (x₁ + a * z)) + b ^ 2) =
                    b ^ 2 / 2 + Real.log (Real.cosh (x₁ + a * z)) by ring]
                rw [Real.exp_add, Real.exp_log (Real.cosh_pos _)]
          _ = Real.exp (b ^ 2 / 2) *
              standardGaussianExpectation (fun z => Real.cosh (x₁ + a * z)) := by
                unfold standardGaussianExpectation
                rw [integral_const_mul]
          _ = Real.exp (b ^ 2 / 2) *
              (Real.exp (a ^ 2 / 2) * Real.cosh x₁) := by
                rw [flatness_standardGaussian_cosh]
      have hJpos : 0 < J := by
        rw [hJ]
        positivity
      have hmain :
          2 * Real.log J =
            b ^ 2 + a ^ 2 + Real.log (Real.cosh x₁) +
              Real.log (Real.cosh (-x₁)) := by
        rw [hJ]
        rw [Real.log_mul (Real.exp_ne_zero _)
          (mul_ne_zero (Real.exp_ne_zero _) (Real.cosh_pos x₁).ne')]
        rw [Real.log_exp]
        rw [Real.log_mul (Real.exp_ne_zero _) (Real.cosh_pos x₁).ne']
        rw [Real.log_exp, Real.cosh_neg]
        ring
      simpa [gtRankOneStep, J] using hmain

end SpinGlass.AT
