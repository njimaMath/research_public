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
#check Real.tanh_neg
#check Real.cosh_neg
#check Real.abs_tanh_lt_one
#check Real.cosh_sq_sub_sinh_sq
#check Real.sq_sqrt
#check gaussianReal_map_const_mul
#check gaussianReal_map_add_const
#check gaussianReal_map_const_add
#check integral_map
#check MeasureTheory.integral_mono_measure
#check intervalIntegral.integral_eq_sub_of_hasDerivAt
#check Real.contDiff_cosh
#check ContDiff.inv
#check ProbabilityTheory.memLp_id_gaussianReal
#check Integrable.comp_neg
#check Integrable.add
#check integral_add
#check Set.Iic_union_Ioi
#check Set.Iic_disjoint_Ioi
#check Set.Ioi_disjoint_Iic
#check setIntegral_univ
#check inv_le_one₀
#check inv_le_one_of_one_le₀
#check Real.one_le_cosh
#check Real.cosh_pos
#check HasDerivAt.add_const
#check HasDerivAt.const_add
#check HasDerivAt.mul_const
#check HasDerivAt.const_mul
#check inv_le_inv₀
#check one_div_le_one_div_of_le
#check Real.sqrt_le_sqrt
#check Real.sqrt_lt_sqrt

example {f : ℝ → ℝ} (hf : Integrable f) :
    ∫ x, f x = ∫ x in Set.Ioi (0 : ℝ), (f x + f (-x)) := by
  have hsplit := setIntegral_union (f := f)
    (Set.Iic_disjoint_Ioi (a := (0 : ℝ)) (b := 0) le_rfl)
    measurableSet_Ioi hf.integrableOn hf.integrableOn
  rw [Set.Iic_union_Ioi, setIntegral_univ] at hsplit
  have hneg : ∫ x in Set.Ioi (0 : ℝ), f (-x) = ∫ x in Set.Iic 0, f x := by
    simpa using integral_comp_neg_Ioi (0 : ℝ) f
  calc
    ∫ x, f x = (∫ x in Set.Iic 0, f x) + ∫ x in Set.Ioi 0, f x := hsplit
    _ = (∫ x in Set.Ioi 0, f (-x)) + ∫ x in Set.Ioi 0, f x := by rw [hneg]
    _ = ∫ x in Set.Ioi 0, (f (-x) + f x) :=
      (integral_add hf.comp_neg.integrableOn hf.integrableOn).symm
    _ = ∫ x in Set.Ioi 0, (f x + f (-x)) := by
      apply integral_congr_ae
      filter_upwards [] with x
      ring

namespace SpinGlass.AT

private noncomputable def sech (x : ℝ) : ℝ := (Real.cosh x)⁻¹
private noncomputable def sech3 (x : ℝ) : ℝ := sech x ^ 3
private noncomputable def sech4 (x : ℝ) : ℝ := sech x ^ 4

private lemma sech_pos (x : ℝ) : 0 < sech x := by
  exact inv_pos.mpr (Real.cosh_pos x)

private lemma sech_le_one (x : ℝ) : sech x ≤ 1 := by
  exact inv_le_one_of_one_le₀ (Real.one_le_cosh x)

private lemma abs_sech_le_one (x : ℝ) : |sech x| ≤ 1 := by
  rw [abs_of_pos (sech_pos x)]
  exact sech_le_one x

private lemma abs_tanh_le_one (x : ℝ) : |Real.tanh x| ≤ 1 :=
  (Real.abs_tanh_lt_one x).le

private lemma sech3_hasDerivAt (x : ℝ) : HasDerivAt sech3
    (-3 * sech3 x * Real.tanh x) x := by
  unfold sech3 sech
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  apply (((Real.hasDerivAt_cosh x).inv hc).pow 3).congr_deriv
  rw [Real.tanh_eq_sinh_div_cosh]
  norm_num only [Nat.cast_ofNat, Nat.reduceSub]
  simp only [Pi.inv_apply, inv_pow]
  field_simp [hc]

private lemma continuous_sech3 : Continuous sech3 := by
  unfold sech3 sech
  exact (Real.continuous_cosh.inv₀ fun x => (Real.cosh_pos x).ne').pow 3

private lemma continuous_tanh : Continuous (fun x : ℝ => Real.tanh x) := by
  simp_rw [Real.tanh_eq_sinh_div_cosh]
  exact Real.continuous_sinh.div₀ Real.continuous_cosh
    (fun x => (Real.cosh_pos x).ne')

private lemma sech3_abs_le_one (x : ℝ) : |sech3 x| ≤ 1 := by
  unfold sech3
  rw [abs_pow]
  exact pow_le_one₀ (abs_nonneg _) (abs_sech_le_one x)

private lemma sech3Deriv_abs_le_three (x : ℝ) :
    |-3 * sech3 x * Real.tanh x| ≤ 3 := by
  rw [abs_mul, abs_mul, abs_neg]
  norm_num
  calc
    3 * |sech3 x| * |Real.tanh x| ≤ 3 * 1 * 1 := by
      gcongr
      · exact sech3_abs_le_one x
      · exact abs_tanh_le_one x
    _ = 3 := by norm_num

private lemma integrable_sech3_affine (a b : ℝ) :
    Integrable (fun z : ℝ => sech3 (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · exact (continuous_sech3.comp
      (continuous_const.add (continuous_const.mul continuous_id)))
      |>.aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using sech3_abs_le_one (a + b * z)

private lemma integrable_sech3Deriv_affine (a b : ℝ) :
    Integrable (fun z : ℝ => -3 * sech3 (a + b * z) *
      Real.tanh (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 3)
  · exact ((continuous_const.mul (continuous_sech3.comp
      (continuous_const.add (continuous_const.mul continuous_id)))).mul
      (continuous_tanh.comp
        (continuous_const.add (continuous_const.mul continuous_id))))
      |>.aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using sech3Deriv_abs_le_three (a + b * z)

private noncomputable def smoothSech3 (r x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z => sech3 (x + Real.sqrt r * z))

private lemma smoothSech3_hasDerivAt_x (r x : ℝ) :
    HasDerivAt (smoothSech3 r)
      (standardGaussianExpectation (fun z =>
        -3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z))) x := by
  unfold smoothSech3 standardGaussianExpectation
  let F : ℝ → ℝ → ℝ := fun y z => sech3 (y + Real.sqrt r * z)
  let F' : ℝ → ℝ → ℝ := fun y z =>
    -3 * sech3 (y + Real.sqrt r * z) * Real.tanh (y + Real.sqrt r * z)
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal 0 1) (F := F) (F' := F') (x₀ := x)
    (s := Set.univ) (bound := fun _ => (3 : ℝ))
    (by simp)
    (Filter.Eventually.of_forall fun y => by
      exact (continuous_sech3.comp
        (continuous_const.add (continuous_const.mul continuous_id)))
        |>.aestronglyMeasurable)
    (by simpa [F] using integrable_sech3_affine x (Real.sqrt r))
    (by
      exact ((continuous_const.mul (continuous_sech3.comp
        (continuous_const.add (continuous_const.mul continuous_id)))).mul
        (continuous_tanh.comp
          (continuous_const.add (continuous_const.mul continuous_id))))
        |>.aestronglyMeasurable)
    (by
      filter_upwards [] with z
      intro y _
      simpa [F', Real.norm_eq_abs] using
        sech3Deriv_abs_le_three (y + Real.sqrt r * z))
    (integrable_const 3)
    (by
      filter_upwards [] with z
      intro y _
      simpa [F, F', Function.comp_def] using
        (sech3_hasDerivAt _).comp y
          ((hasDerivAt_id y).add_const (Real.sqrt r * z)))
  simpa [F, F'] using h.2

private lemma tanh_hasDerivAt (x : ℝ) : HasDerivAt (fun x : ℝ => Real.tanh x)
    (sech x ^ 2) x := by
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  rw [show (fun x : ℝ => Real.tanh x) = fun x => Real.sinh x / Real.cosh x by
    funext y
    exact Real.tanh_eq_sinh_div_cosh y]
  apply ((Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc).congr_deriv
  unfold sech
  simp only [Pi.inv_apply, inv_pow]
  field_simp [hc]
  nlinarith [Real.cosh_sq_sub_sinh_sq x]

private noncomputable def sech3Second (x : ℝ) : ℝ :=
  9 * sech3 x * Real.tanh x ^ 2 - 3 * sech3 x * sech x ^ 2

private lemma sech3Deriv_hasDerivAt (x : ℝ) :
    HasDerivAt (fun x => -3 * sech3 x * Real.tanh x) (sech3Second x) x := by
  have h := ((sech3_hasDerivAt x).const_mul (-3)).mul (tanh_hasDerivAt x)
  apply h.congr_deriv
  unfold sech3Second
  ring

private lemma sech3Second_abs_le_twelve (x : ℝ) : |sech3Second x| ≤ 12 := by
  have hs3 : |sech3 x| ≤ 1 := sech3_abs_le_one x
  have ht : |Real.tanh x| ≤ 1 := abs_tanh_le_one x
  have hs : |sech x| ≤ 1 := abs_sech_le_one x
  unfold sech3Second
  calc
    |9 * sech3 x * Real.tanh x ^ 2 - 3 * sech3 x * sech x ^ 2|
        ≤ |9 * sech3 x * Real.tanh x ^ 2| + |3 * sech3 x * sech x ^ 2| :=
      abs_sub _ _
    _ ≤ 9 * 1 * 1 ^ 2 + 3 * 1 * 1 ^ 2 := by
      simp only [abs_mul, abs_pow]
      gcongr <;> norm_num
    _ = 12 := by norm_num

private lemma integrable_sech3Second_affine (a b : ℝ) :
    Integrable (fun z : ℝ => sech3Second (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 12)
  · apply Continuous.aestronglyMeasurable
    unfold sech3Second
    exact (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
      ((continuous_tanh.comp (by fun_prop)).pow 2)).sub
      ((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
        (((Real.continuous_cosh.comp (by fun_prop)).inv₀
          (fun z => (Real.cosh_pos _).ne')).pow 2)))
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using sech3Second_abs_le_twelve (a + b * z)

private lemma smoothSech3_first_hasDerivAt_x (r x : ℝ) :
    HasDerivAt
      (fun y => standardGaussianExpectation (fun z =>
        -3 * sech3 (y + Real.sqrt r * z) *
          Real.tanh (y + Real.sqrt r * z)))
      (standardGaussianExpectation (fun z =>
        sech3Second (x + Real.sqrt r * z))) x := by
  unfold standardGaussianExpectation
  let F : ℝ → ℝ → ℝ := fun y z =>
    -3 * sech3 (y + Real.sqrt r * z) * Real.tanh (y + Real.sqrt r * z)
  let F' : ℝ → ℝ → ℝ := fun y z => sech3Second (y + Real.sqrt r * z)
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal 0 1) (F := F) (F' := F') (x₀ := x)
    (s := Set.univ) (bound := fun _ => (12 : ℝ))
    (by simp)
    (Filter.Eventually.of_forall fun y => by
      exact ((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
        (continuous_tanh.comp (by fun_prop))).aestronglyMeasurable)
    (by simpa [F] using integrable_sech3Deriv_affine x (Real.sqrt r))
    (by
      apply Continuous.aestronglyMeasurable
      unfold F' sech3Second
      exact (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
        ((continuous_tanh.comp (by fun_prop)).pow 2)).sub
        ((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
          (((Real.continuous_cosh.comp (by fun_prop)).inv₀
            (fun z => (Real.cosh_pos _).ne')).pow 2))))
    (by
      filter_upwards [] with z
      intro y _
      simpa [F', Real.norm_eq_abs] using
        sech3Second_abs_le_twelve (y + Real.sqrt r * z))
    (integrable_const 12)
    (by
      filter_upwards [] with z
      intro y _
      simpa [F, F', Function.comp_def] using
        (sech3Deriv_hasDerivAt _).comp y
          ((hasDerivAt_id y).add_const (Real.sqrt r * z)))
  simpa [F, F'] using h.2

private lemma smoothSech3_hasDerivAt_x_twice (r x : ℝ) :
    HasDerivAt (deriv (smoothSech3 r))
      (standardGaussianExpectation (fun z =>
        sech3Second (x + Real.sqrt r * z))) x := by
  have h₁ := smoothSech3_hasDerivAt_x r
  have heq : deriv (smoothSech3 r) = fun y => standardGaussianExpectation (fun z =>
      -3 * sech3 (y + Real.sqrt r * z) *
        Real.tanh (y + Real.sqrt r * z)) := by
    funext y
    exact (h₁ y).deriv
  rw [heq]
  exact smoothSech3_first_hasDerivAt_x r x

private lemma contDiff_sech : ContDiff ℝ ⊤ sech := by
  unfold sech
  exact Real.contDiff_cosh.inv (fun x => (Real.cosh_pos x).ne')

private lemma contDiff_sech3 : ContDiff ℝ ⊤ sech3 := by
  unfold sech3
  exact contDiff_sech.pow 3

private lemma contDiff_tanh : ContDiff ℝ ⊤ (fun x : ℝ => Real.tanh x) := by
  simp_rw [Real.tanh_eq_sinh_div_cosh]
  exact Real.contDiff_sinh.div Real.contDiff_cosh
    (fun x => (Real.cosh_pos x).ne')

private lemma contDiff_sech3Deriv :
    ContDiff ℝ ⊤ (fun x => -3 * sech3 x * Real.tanh x) := by
  exact (contDiff_const.mul contDiff_sech3).mul contDiff_tanh

private lemma sech3Deriv_comp_deriv (a b z : ℝ) :
    deriv (fun y => -3 * sech3 (a + b * y) * Real.tanh (a + b * y)) z =
      b * sech3Second (a + b * z) := by
  have harg : HasDerivAt (fun y : ℝ => a + b * y) b z := by
    exact ((hasDerivAt_id z).const_mul b).const_add a
  simpa [Function.comp_def, mul_comm] using
    ((sech3Deriv_hasDerivAt (a + b * z)).comp z harg).deriv

private lemma sech3Deriv_comp_moderate (a b : ℝ) :
    HasModerateGrowth
      (fun z => -3 * sech3 (a + b * z) * Real.tanh (a + b * z)) := by
  refine ⟨16 * (1 + |b|), 0, by positivity, ?_, ?_⟩
  · intro z
    simpa only [pow_zero, mul_one] using
      (show |-3 * sech3 (a + b * z) * Real.tanh (a + b * z)| ≤
          16 * (1 + |b|) by
        have h := sech3Deriv_abs_le_three (a + b * z)
        have hb : 0 ≤ |b| := abs_nonneg b
        nlinarith)
  · intro z
    rw [sech3Deriv_comp_deriv]
    simpa only [pow_zero, mul_one] using
      (show |b * sech3Second (a + b * z)| ≤ 16 * (1 + |b|) by
        rw [abs_mul]
        have hs := sech3Second_abs_le_twelve (a + b * z)
        have hb : 0 ≤ |b| := abs_nonneg b
        nlinarith [mul_le_mul_of_nonneg_left hs hb])

private lemma smoothSech3_hasDerivAt_r_raw {r x : ℝ} (hr : 0 < r) :
    HasDerivAt (fun t => smoothSech3 t x)
      (standardGaussianExpectation (fun z =>
        (-3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z)) *
            (1 / (2 * Real.sqrt r) * z))) r := by
  unfold smoothSech3 standardGaussianExpectation
  let F : ℝ → ℝ → ℝ := fun t z => sech3 (x + Real.sqrt t * z)
  let F' : ℝ → ℝ → ℝ := fun t z =>
    (-3 * sech3 (x + Real.sqrt t * z) *
      Real.tanh (x + Real.sqrt t * z)) * (1 / (2 * Real.sqrt t) * z)
  let c : ℝ := Real.sqrt (r / 2)
  have hhalf : 0 < r / 2 := by linarith
  have hc : 0 < c := Real.sqrt_pos.2 hhalf
  have hboundInt : Integrable (fun z : ℝ => 3 * c⁻¹ * |z|)
      (gaussianReal 0 1) := by
    have hz : Integrable (fun z : ℝ => |z|) (gaussianReal 0 1) := by
      simpa using integrable_abs_pow_gaussianReal_centered (1 : ℝ≥0) 1
    exact hz.const_mul (3 * c⁻¹)
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal 0 1) (F := F) (F' := F') (x₀ := r)
    (s := Set.Ioi (r / 2)) (bound := fun z => 3 * c⁻¹ * |z|)
    (Ioi_mem_nhds (by linarith))
    (Filter.Eventually.of_forall fun t =>
      (continuous_sech3.comp
        (continuous_const.add
          ((Real.continuous_sqrt.comp continuous_const).mul continuous_id)))
        |>.aestronglyMeasurable)
    (by simpa [F] using integrable_sech3_affine x (Real.sqrt r))
    (by
      apply Continuous.aestronglyMeasurable
      dsimp [F']
      exact (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
        (continuous_tanh.comp (by fun_prop))).mul
          (continuous_const.mul continuous_id)))
    (by
      filter_upwards [] with z
      intro t ht
      have htpos : 0 < t := lt_trans hhalf ht
      have hroot : 0 < Real.sqrt t := Real.sqrt_pos.2 htpos
      have hrootle : c ≤ Real.sqrt t := Real.sqrt_le_sqrt ht.le
      have hinv : (Real.sqrt t)⁻¹ ≤ c⁻¹ :=
        (inv_le_inv₀ hroot hc).2 hrootle
      have hcoef : |1 / (2 * Real.sqrt t)| ≤ c⁻¹ := by
        rw [abs_of_pos (by positivity : 0 < 1 / (2 * Real.sqrt t))]
        calc
          1 / (2 * Real.sqrt t) ≤ (Real.sqrt t)⁻¹ := by
            rw [one_div]
            exact (inv_le_inv₀ (by positivity) hroot).2 (by nlinarith)
          _ ≤ c⁻¹ := hinv
      dsimp [F']
      calc
        |(-3 * sech3 (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)) *
            (1 / (2 * Real.sqrt t) * z)|
            = |-3 * sech3 (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)| *
                |1 / (2 * Real.sqrt t)| * |z| := by rw [abs_mul, abs_mul]
        _
            ≤ 3 * c⁻¹ * |z| := by
              gcongr
              · exact sech3Deriv_abs_le_three _
              · exact hcoef
        _ = 3 * c⁻¹ * |z| := rfl)
    hboundInt
    (by
      filter_upwards [] with z
      intro t ht
      have htpos : 0 < t := lt_trans hhalf ht
      have hsqrt := Real.hasDerivAt_sqrt htpos.ne'
      have harg : HasDerivAt (fun t => x + Real.sqrt t * z)
          (1 / (2 * Real.sqrt t) * z) t := by
        exact (hsqrt.mul_const z).const_add x
      simpa [F, F', Function.comp_def] using
        (sech3_hasDerivAt _).comp t harg)
  simpa [F, F'] using h.2

private lemma smoothSech3_hasDerivAt_r {r x : ℝ} (hr : 0 < r) :
    HasDerivAt (fun t => smoothSech3 t x)
      ((1 / 2) * standardGaussianExpectation (fun z =>
        sech3Second (x + Real.sqrt r * z))) r := by
  apply (smoothSech3_hasDerivAt_r_raw (x := x) hr).congr_deriv
  unfold standardGaussianExpectation
  let F : ℝ → ℝ := fun z =>
    -3 * sech3 (x + Real.sqrt r * z) * Real.tanh (x + Real.sqrt r * z)
  have hcont : ContDiff ℝ 1 F := by
    exact (contDiff_sech3Deriv.of_le (by norm_num)).comp (by fun_prop)
  have hibp := gaussianReal_integration_by_parts (v := (1 : ℝ≥0)) one_ne_zero
    hcont (sech3Deriv_comp_moderate x (Real.sqrt r))
  have hderiv : deriv F = fun z => Real.sqrt r * sech3Second
      (x + Real.sqrt r * z) := by
    funext z
    exact sech3Deriv_comp_deriv x (Real.sqrt r) z
  rw [hderiv] at hibp
  simp only [NNReal.coe_one, one_mul] at hibp
  have hsqrt : Real.sqrt r ≠ 0 := (Real.sqrt_pos.2 hr).ne'
  calc
    ∫ z, F z * (1 / (2 * Real.sqrt r) * z) ∂gaussianReal 0 1
        = (1 / (2 * Real.sqrt r)) * ∫ z, z * F z ∂gaussianReal 0 1 := by
          rw [← integral_const_mul]
          apply integral_congr_ae
          filter_upwards [] with z
          ring
    _ = (1 / (2 * Real.sqrt r)) *
          ∫ z, Real.sqrt r * sech3Second (x + Real.sqrt r * z)
            ∂gaussianReal 0 1 := by rw [hibp]
    _ = (1 / 2) * ∫ z, sech3Second (x + Real.sqrt r * z)
          ∂gaussianReal 0 1 := by
          rw [integral_const_mul]
          field_simp [hsqrt]

end SpinGlass.AT
