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
#check integral_neg_eq_self
#check integrable_gaussianPDFReal

private lemma integral_eq_integral_Ioi_add_neg {f : ℝ → ℝ} (hf : Integrable f) :
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
    simpa only [id_eq, mul_one] using
      ((hasDerivAt_id z).const_mul b).const_add a
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
                |1 / (2 * Real.sqrt t)| * |z| := by
                  simp only [abs_mul]
                  ring
        _
            ≤ 3 * c⁻¹ * |z| := by
              have hp :
                  |-3 * sech3 (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)| *
                      |1 / (2 * Real.sqrt t)| ≤ 3 * c⁻¹ :=
                mul_le_mul (sech3Deriv_abs_le_three _) hcoef
                  (abs_nonneg _) (by norm_num)
              exact mul_le_mul_of_nonneg_right hp (abs_nonneg z))
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

private lemma sech_neg (x : ℝ) : sech (-x) = sech x := by
  unfold sech
  rw [Real.cosh_neg]

private lemma sech3_neg (x : ℝ) : sech3 (-x) = sech3 x := by
  unfold sech3
  rw [sech_neg]

private lemma sech3Deriv_neg (x : ℝ) :
    -3 * sech3 (-x) * Real.tanh (-x) =
      -(-3 * sech3 x * Real.tanh x) := by
  rw [sech3_neg, Real.tanh_neg]
  ring

private lemma integral_comp_neg_standard (f : ℝ → ℝ) (hf : Continuous f) :
    (∫ z, f (-z) ∂gaussianReal 0 1) = ∫ z, f z ∂gaussianReal 0 1 := by
  have hmap : Measure.map (fun z : ℝ => -z) (gaussianReal 0 1) =
      gaussianReal 0 1 := by simpa using gaussianReal_map_neg (μ := (0 : ℝ)) (v := (1 : ℝ≥0))
  calc
    (∫ z, f (-z) ∂gaussianReal 0 1) =
        ∫ z, f z ∂Measure.map (fun z : ℝ => -z) (gaussianReal 0 1) := by
          rw [integral_map (by fun_prop) hf.aestronglyMeasurable]
    _ = ∫ z, f z ∂gaussianReal 0 1 := by rw [hmap]

private lemma smoothSech3_neg (r x : ℝ) : smoothSech3 r (-x) = smoothSech3 r x := by
  unfold smoothSech3 standardGaussianExpectation
  calc
    ∫ z, sech3 (-x + Real.sqrt r * z) ∂gaussianReal 0 1 =
        ∫ z, sech3 (x + Real.sqrt r * (-z)) ∂gaussianReal 0 1 := by
          apply integral_congr_ae
          filter_upwards [] with z
          calc
            sech3 (-x + Real.sqrt r * z) =
                sech3 (-(x + Real.sqrt r * (-z))) := by congr 1 <;> ring
            _ = sech3 (x + Real.sqrt r * (-z)) := sech3_neg _
    _ = ∫ z, sech3 (x + Real.sqrt r * z) ∂gaussianReal 0 1 := by
      simpa using integral_comp_neg_standard
        (fun z => sech3 (x + Real.sqrt r * z))
        (continuous_sech3.comp (by fun_prop))

private lemma smoothSech3_first_neg (r x : ℝ) :
    standardGaussianExpectation (fun z =>
      -3 * sech3 (-x + Real.sqrt r * z) *
        Real.tanh (-x + Real.sqrt r * z)) =
      -standardGaussianExpectation (fun z =>
        -3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z)) := by
  unfold standardGaussianExpectation
  calc
    ∫ z, -3 * sech3 (-x + Real.sqrt r * z) *
        Real.tanh (-x + Real.sqrt r * z) ∂gaussianReal 0 1 =
        ∫ z, -(-3 * sech3 (x + Real.sqrt r * (-z)) *
          Real.tanh (x + Real.sqrt r * (-z))) ∂gaussianReal 0 1 := by
            apply integral_congr_ae
            filter_upwards [] with z
            calc
              -3 * sech3 (-x + Real.sqrt r * z) *
                  Real.tanh (-x + Real.sqrt r * z) =
                  -3 * sech3 (-(x + Real.sqrt r * (-z))) *
                    Real.tanh (-(x + Real.sqrt r * (-z))) := by
                      congr 2 <;> ring
              _ = -(-3 * sech3 (x + Real.sqrt r * (-z)) *
                    Real.tanh (x + Real.sqrt r * (-z))) := sech3Deriv_neg _
    _ = -∫ z, -3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z) ∂gaussianReal 0 1 := by
      rw [integral_neg]
      congr 1
      simpa using integral_comp_neg_standard
        (fun z => -3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z))
        (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
          (continuous_tanh.comp (by fun_prop))))

private lemma standard_affine_integral_eq_gaussian
    {r : ℝ} (hr : 0 ≤ r) (x : ℝ) {f : ℝ → ℝ} (hf : Continuous f) :
    (∫ z, f (x + Real.sqrt r * z) ∂gaussianReal 0 1) =
      ∫ y, f y ∂gaussianReal x ⟨r, hr⟩ := by
  let v : ℝ≥0 := ⟨r, hr⟩
  have hmul : Measure.map (fun z : ℝ => Real.sqrt r * z) (gaussianReal 0 1) =
      gaussianReal 0 v := by
    rw [gaussianReal_map_const_mul]
    simp only [mul_zero]
    apply congrArg (gaussianReal 0)
    apply NNReal.eq
    simp only [NNReal.coe_mul, NNReal.coe_mk, NNReal.coe_one, mul_one]
    exact Real.sq_sqrt hr
  have hadd : Measure.map (fun y : ℝ => x + y) (gaussianReal 0 v) =
      gaussianReal x v := by
    simpa using gaussianReal_map_const_add (μ := (0 : ℝ)) (v := v) x
  have hmap : Measure.map (fun z : ℝ => x + Real.sqrt r * z)
      (gaussianReal 0 1) = gaussianReal x v := by
    calc
      Measure.map (fun z : ℝ => x + Real.sqrt r * z) (gaussianReal 0 1) =
          Measure.map (fun y : ℝ => x + y)
            (Measure.map (fun z : ℝ => Real.sqrt r * z) (gaussianReal 0 1)) := by
              simpa [Function.comp_def] using
                (Measure.map_map
                  (μ := gaussianReal 0 1)
                  (g := fun y : ℝ => x + y)
                  (f := fun z : ℝ => Real.sqrt r * z)
                  (by fun_prop) (by fun_prop)).symm
      _ = Measure.map (fun y : ℝ => x + y) (gaussianReal 0 v) := by rw [hmul]
      _ = gaussianReal x v := hadd
  calc
    (∫ z, f (x + Real.sqrt r * z) ∂gaussianReal 0 1) =
        ∫ y, f y ∂Measure.map (fun z : ℝ => x + Real.sqrt r * z)
          (gaussianReal 0 1) := by
            rw [integral_map (by fun_prop) hf.aestronglyMeasurable]
    _ = ∫ y, f y ∂gaussianReal x ⟨r, hr⟩ := by rw [hmap]

private lemma gaussianPDFReal_neg_le_self {v : ℝ≥0} (hv : v ≠ 0)
    {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    gaussianPDFReal x v (-y) ≤ gaussianPDFReal x v y := by
  rw [gaussianPDFReal, gaussianPDFReal]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Real.exp_le_exp.mpr
  have hvpos : 0 < (v : ℝ) := by
    exact_mod_cast (bot_lt_iff_ne_bot.mpr hv)
  apply div_le_div_of_nonneg_right _ (by positivity : 0 ≤ 2 * (v : ℝ))
  nlinarith [sq_nonneg (y - x), sq_nonneg (y + x)]

private lemma smoothSech3_first_nonpos {r x : ℝ} (hr : 0 ≤ r) (hx : 0 ≤ x) :
    standardGaussianExpectation (fun z =>
      -3 * sech3 (x + Real.sqrt r * z) *
        Real.tanh (x + Real.sqrt r * z)) ≤ 0 := by
  by_cases hr0 : r = 0
  · subst r
    simp only [Real.sqrt_zero, zero_mul, add_zero, standardGaussianExpectation,
      integral_const, probReal_univ, one_smul]
    have ht : 0 ≤ Real.tanh x := by
      rw [Real.tanh_eq_sinh_div_cosh]
      exact div_nonneg ((Real.sinh_nonneg_iff).2 hx) (Real.cosh_pos x).le
    exact mul_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonpos_of_nonneg (by norm_num)
        (pow_nonneg (sech_pos x).le 3)) ht
  let v : ℝ≥0 := ⟨r, hr⟩
  have hvpos : 0 < v := by
    change 0 < r
    exact lt_of_le_of_ne hr (Ne.symm hr0)
  have hv : v ≠ 0 := hvpos.ne'
  let D : ℝ → ℝ := fun y => -3 * sech3 y * Real.tanh y
  have hshift : standardGaussianExpectation (fun z => D (x + Real.sqrt r * z)) =
      ∫ y, D y ∂gaussianReal x v := by
    unfold standardGaussianExpectation
    simpa [v] using standard_affine_integral_eq_gaussian hr x
      ((continuous_const.mul continuous_sech3).mul continuous_tanh)
  rw [hshift, integral_gaussianReal_eq_integral_smul hv]
  have hvol : Integrable (fun y => gaussianPDFReal x v y * D y) := by
    apply Integrable.mono' ((integrable_gaussianPDFReal x v).const_mul 3)
    · exact ((measurable_gaussianPDFReal x v).mul
        ((continuous_const.mul continuous_sech3).mul continuous_tanh).measurable)
        |>.aestronglyMeasurable
    · filter_upwards [] with y
      have hp := gaussianPDFReal_nonneg x v y
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hp]
      dsimp [D]
      calc
        gaussianPDFReal x v y *
            |-3 * sech3 y * Real.tanh y| ≤
            gaussianPDFReal x v y * 3 :=
          mul_le_mul_of_nonneg_left (sech3Deriv_abs_le_three y) hp
        _ = 3 * gaussianPDFReal x v y := by ring
  simp only [smul_eq_mul]
  rw [integral_eq_integral_Ioi_add_neg hvol]
  apply integral_nonpos_of_ae
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
  have hy0 : 0 ≤ y := hy.le
  have hD : D y ≤ 0 := by
    have ht : 0 ≤ Real.tanh y := by
      rw [Real.tanh_eq_sinh_div_cosh]
      exact div_nonneg ((Real.sinh_nonneg_iff).2 hy0) (Real.cosh_pos y).le
    exact mul_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonpos_of_nonneg (by norm_num)
        (pow_nonneg (sech_pos y).le 3)) ht
  have hp := gaussianPDFReal_neg_le_self hv hx hy0
  change gaussianPDFReal x v y * D y + gaussianPDFReal x v (-y) * D (-y) ≤ 0
  have hDneg : D (-y) = -D y := by
    exact sech3Deriv_neg y
  rw [hDneg]
  nlinarith [mul_nonpos_of_nonpos_of_nonneg hD (sub_nonneg.mpr hp)]

private noncomputable def smoothSech3First (r x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z =>
    -3 * sech3 (x + Real.sqrt r * z) *
      Real.tanh (x + Real.sqrt r * z))

private noncomputable def smoothSech3Second (r x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z => sech3Second (x + Real.sqrt r * z))

private lemma smoothSech3_nonneg (r x : ℝ) : 0 ≤ smoothSech3 r x := by
  unfold smoothSech3 standardGaussianExpectation
  apply integral_nonneg
  intro z
  exact pow_nonneg (sech_pos _).le 3

private lemma abs_smoothSech3_le_one (r x : ℝ) : |smoothSech3 r x| ≤ 1 := by
  rw [abs_of_nonneg (smoothSech3_nonneg r x)]
  unfold smoothSech3 standardGaussianExpectation
  calc
    (∫ z, sech3 (x + Real.sqrt r * z) ∂gaussianReal 0 1) ≤
        ∫ _z : ℝ, (1 : ℝ) ∂gaussianReal 0 1 := by
          apply integral_mono (integrable_sech3_affine x (Real.sqrt r)) (integrable_const 1)
          intro z
          exact le_trans (le_abs_self _) (sech3_abs_le_one _)
    _ = 1 := by simp

private lemma abs_smoothSech3First_le_three (r x : ℝ) :
    |smoothSech3First r x| ≤ 3 := by
  unfold smoothSech3First standardGaussianExpectation
  calc
    |∫ z, -3 * sech3 (x + Real.sqrt r * z) *
        Real.tanh (x + Real.sqrt r * z) ∂gaussianReal 0 1| ≤
        ∫ z, |-3 * sech3 (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z)| ∂gaussianReal 0 1 :=
      abs_integral_le_integral_abs
    _ ≤ ∫ _z : ℝ, (3 : ℝ) ∂gaussianReal 0 1 := by
      apply integral_mono
      · exact (integrable_sech3Deriv_affine x (Real.sqrt r)).abs
      · exact integrable_const 3
      · intro z
        exact sech3Deriv_abs_le_three _
    _ = 3 := by simp

private lemma abs_smoothSech3Second_le_twelve (r x : ℝ) :
    |smoothSech3Second r x| ≤ 12 := by
  unfold smoothSech3Second standardGaussianExpectation
  calc
    |∫ z, sech3Second (x + Real.sqrt r * z) ∂gaussianReal 0 1| ≤
        ∫ z, |sech3Second (x + Real.sqrt r * z)| ∂gaussianReal 0 1 :=
      abs_integral_le_integral_abs
    _ ≤ ∫ _z : ℝ, (12 : ℝ) ∂gaussianReal 0 1 := by
      apply integral_mono
      · exact (integrable_sech3Second_affine x (Real.sqrt r)).abs
      · exact integrable_const 12
      · intro z
        exact sech3Second_abs_le_twelve _
    _ = 12 := by simp

private lemma sech_hasDerivAt (x : ℝ) :
    HasDerivAt sech (-sech x * Real.tanh x) x := by
  unfold sech
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  apply ((Real.hasDerivAt_cosh x).inv hc).congr_deriv
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp [hc]

private lemma tanh_sq_add_sech_sq (x : ℝ) :
    Real.tanh x ^ 2 + sech x ^ 2 = 1 := by
  unfold sech
  rw [Real.tanh_eq_sinh_div_cosh]
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  simp only [div_pow, inv_pow]
  field_simp [hc]
  nlinarith [Real.cosh_sq_sub_sinh_sq x]

private noncomputable def tiltedSech4Value (r x : ℝ) : ℝ :=
  Real.exp (-r / 2) * smoothSech3 r x * sech x

private noncomputable def tiltedSech4First (r x : ℝ) : ℝ :=
  Real.exp (-r / 2) * sech x *
    (smoothSech3First r x - smoothSech3 r x * Real.tanh x)

private noncomputable def tiltedSech4Second (r x : ℝ) : ℝ :=
  Real.exp (-r / 2) * sech x *
    (smoothSech3Second r x - 2 * smoothSech3First r x * Real.tanh x +
      smoothSech3 r x * (Real.tanh x ^ 2 - sech x ^ 2))

private lemma tiltedSech4Value_hasDerivAt_x (r x : ℝ) :
    HasDerivAt (tiltedSech4Value r) (tiltedSech4First r x) x := by
  have ha : HasDerivAt (smoothSech3 r) (smoothSech3First r x) x := by
    simpa [smoothSech3First] using smoothSech3_hasDerivAt_x r x
  have hs := sech_hasDerivAt x
  unfold tiltedSech4Value tiltedSech4First
  apply ((ha.const_mul (Real.exp (-r / 2))).mul hs).congr_deriv
  ring

private lemma tiltedSech4First_hasDerivAt_x (r x : ℝ) :
    HasDerivAt (tiltedSech4First r) (tiltedSech4Second r x) x := by
  have ha : HasDerivAt (smoothSech3 r) (smoothSech3First r x) x := by
    simpa [smoothSech3First] using smoothSech3_hasDerivAt_x r x
  have ha₁ : HasDerivAt (smoothSech3First r) (smoothSech3Second r x) x := by
    change HasDerivAt
      (fun y => standardGaussianExpectation (fun z =>
        -3 * sech3 (y + Real.sqrt r * z) * Real.tanh (y + Real.sqrt r * z)))
      (standardGaussianExpectation (fun z => sech3Second (x + Real.sqrt r * z))) x
    exact smoothSech3_first_hasDerivAt_x r x
  have hs := sech_hasDerivAt x
  have ht := tanh_hasDerivAt x
  unfold tiltedSech4First tiltedSech4Second
  have hbracket := ha₁.sub (ha.mul ht)
  apply ((hs.const_mul (Real.exp (-r / 2))).mul hbracket).congr_deriv
  simp only [Pi.sub_apply, Pi.mul_apply]
  ring

private lemma tiltedSech4_generator (r x : ℝ) :
    (1 / 2) * tiltedSech4Second r x +
        Real.tanh x * tiltedSech4First r x =
      (1 / 2) * Real.exp (-r / 2) * sech x *
        (smoothSech3Second r x - smoothSech3 r x) := by
  have hid := tanh_sq_add_sech_sq x
  have ht : Real.tanh x ^ 2 = 1 - sech x ^ 2 := by linarith
  unfold tiltedSech4Second tiltedSech4First
  linear_combination
    -(Real.exp (-r / 2) * sech x * smoothSech3 r x / 2) * hid

private lemma tiltedSech4Value_hasDerivAt_r {r x : ℝ} (hr : 0 < r) :
    HasDerivAt (fun t => tiltedSech4Value t x)
      ((1 / 2) * Real.exp (-r / 2) * sech x *
        (smoothSech3Second r x - smoothSech3 r x)) r := by
  have he : HasDerivAt (fun t : ℝ => Real.exp (-t / 2))
      ((-1 / 2) * Real.exp (-r / 2)) r := by
    have hinner : HasDerivAt (fun t : ℝ => -t / 2) (-1 / 2) r := by
      rw [show (fun t : ℝ => -t / 2) = fun t => (-1 / 2) * t by
        funext t
        ring]
      simpa using (hasDerivAt_id r).const_mul (-1 / 2)
    simpa only [Function.comp_def, mul_comm] using
      (Real.hasDerivAt_exp (-r / 2)).comp r hinner
  have ha := smoothSech3_hasDerivAt_r (x := x) hr
  unfold tiltedSech4Value smoothSech3Second
  apply ((he.mul ha).mul_const (sech x)).congr_deriv
  ring

private lemma tiltedSech4First_nonpos {r x : ℝ} (hr : 0 ≤ r) (hx : 0 ≤ x) :
    tiltedSech4First r x ≤ 0 := by
  have ha₁ : smoothSech3First r x ≤ 0 :=
    smoothSech3_first_nonpos hr hx
  have ha : 0 ≤ smoothSech3 r x := smoothSech3_nonneg r x
  have ht : 0 ≤ Real.tanh x := by
    rw [Real.tanh_eq_sinh_div_cosh]
    exact div_nonneg ((Real.sinh_nonneg_iff).2 hx) (Real.cosh_pos x).le
  unfold tiltedSech4First
  exact mul_nonpos_of_nonneg_of_nonpos
    (mul_nonneg (Real.exp_pos _).le (sech_pos x).le)
    (by linarith [mul_nonneg ha ht])

private lemma tiltedSech4First_neg (r x : ℝ) :
    tiltedSech4First r (-x) = -tiltedSech4First r x := by
  unfold tiltedSech4First smoothSech3First
  rw [sech_neg, Real.tanh_neg, smoothSech3_neg, smoothSech3_first_neg]
  ring

private lemma continuous_smoothSech3 (r : ℝ) : Continuous (smoothSech3 r) := by
  rw [continuous_iff_continuousAt]
  intro x
  exact (smoothSech3_hasDerivAt_x r x).continuousAt

private lemma continuous_smoothSech3First (r : ℝ) : Continuous (smoothSech3First r) := by
  rw [continuous_iff_continuousAt]
  intro x
  change ContinuousAt
    (fun y => standardGaussianExpectation (fun z =>
      -3 * sech3 (y + Real.sqrt r * z) * Real.tanh (y + Real.sqrt r * z))) x
  exact (smoothSech3_first_hasDerivAt_x r x).continuousAt

private lemma continuous_smoothSech3Second (r : ℝ) : Continuous (smoothSech3Second r) := by
  unfold smoothSech3Second standardGaussianExpectation
  rw [continuous_iff_continuousAt]
  intro x
  have hmeas : ∀ᶠ y in nhds x,
      AEStronglyMeasurable (fun z => sech3Second (y + Real.sqrt r * z))
        (gaussianReal 0 1) := by
    exact Filter.Eventually.of_forall fun y =>
      (by
        apply Continuous.aestronglyMeasurable
        unfold sech3Second
        exact (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
          ((continuous_tanh.comp (by fun_prop)).pow 2)).sub
          ((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
            (((Real.continuous_cosh.comp (by fun_prop)).inv₀
              (fun z => (Real.cosh_pos _).ne')).pow 2))))
  have hbound : ∀ᶠ y in nhds x, ∀ᵐ z ∂gaussianReal 0 1,
      ‖sech3Second (y + Real.sqrt r * z)‖ ≤ (12 : ℝ) := by
    exact Filter.Eventually.of_forall fun y => ae_of_all _ fun z => by
      simpa [Real.norm_eq_abs] using sech3Second_abs_le_twelve (y + Real.sqrt r * z)
  have hlim : ∀ᵐ z ∂gaussianReal 0 1,
      Filter.Tendsto (fun y => sech3Second (y + Real.sqrt r * z)) (nhds x)
        (nhds (sech3Second (x + Real.sqrt r * z))) := by
    exact ae_of_all _ fun z => by
      apply ContinuousAt.tendsto
      unfold sech3Second
      exact (((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
        ((continuous_tanh.comp (by fun_prop)).pow 2)).sub
        ((continuous_const.mul (continuous_sech3.comp (by fun_prop))).mul
          (((Real.continuous_cosh.comp (by fun_prop)).inv₀
            (fun y => (Real.cosh_pos _).ne')).pow 2))).continuousAt
  exact tendsto_integral_filter_of_dominated_convergence
    (l := nhds x) (F := fun y z => sech3Second (y + Real.sqrt r * z))
    (f := fun z => sech3Second (x + Real.sqrt r * z))
    (bound := fun _ => (12 : ℝ)) hmeas hbound (integrable_const 12) hlim

private lemma continuous_tiltedSech4Value (r : ℝ) : Continuous (tiltedSech4Value r) := by
  rw [continuous_iff_continuousAt]
  intro x
  exact (tiltedSech4Value_hasDerivAt_x r x).continuousAt

private lemma continuous_tiltedSech4First (r : ℝ) : Continuous (tiltedSech4First r) := by
  rw [continuous_iff_continuousAt]
  intro x
  exact (tiltedSech4First_hasDerivAt_x r x).continuousAt

private lemma continuous_sech : Continuous sech := contDiff_sech.continuous

private lemma continuous_tiltedSech4Second (r : ℝ) : Continuous (tiltedSech4Second r) := by
  unfold tiltedSech4Second
  exact ((continuous_const.mul continuous_sech).mul
    (((continuous_smoothSech3Second r).sub
      ((continuous_const.mul (continuous_smoothSech3First r)).mul continuous_tanh)).add
      ((continuous_smoothSech3 r).mul
        ((continuous_tanh.pow 2).sub (continuous_sech.pow 2)))))

private lemma abs_tiltedSech4Value_le_one {r x : ℝ} (hr : 0 ≤ r) :
    |tiltedSech4Value r x| ≤ 1 := by
  unfold tiltedSech4Value
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos (sech_pos _)]
  have he : Real.exp (-r / 2) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (by linarith)
  calc
    Real.exp (-r / 2) * |smoothSech3 r x| * sech x ≤ 1 * 1 * 1 := by
      have h₁ : Real.exp (-r / 2) * |smoothSech3 r x| ≤ 1 * 1 :=
        mul_le_mul he (abs_smoothSech3_le_one r x) (abs_nonneg _) (by norm_num)
      exact mul_le_mul h₁ (sech_le_one x) (sech_pos x).le (by norm_num)
    _ = 1 := by norm_num

private lemma abs_tiltedSech4First_le_four {r x : ℝ} (hr : 0 ≤ r) :
    |tiltedSech4First r x| ≤ 4 := by
  unfold tiltedSech4First
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos (sech_pos _)]
  have he : Real.exp (-r / 2) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (by linarith)
  have hb : |smoothSech3First r x - smoothSech3 r x * Real.tanh x| ≤ 4 := by
    calc
      |smoothSech3First r x - smoothSech3 r x * Real.tanh x| ≤
          |smoothSech3First r x| + |smoothSech3 r x| * |Real.tanh x| := by
            simpa [abs_mul] using abs_sub (smoothSech3First r x)
              (smoothSech3 r x * Real.tanh x)
      _ ≤ 3 + 1 * 1 := by
        gcongr
        · exact abs_smoothSech3First_le_three r x
        · exact abs_smoothSech3_le_one r x
        · exact abs_tanh_le_one x
      _ = 4 := by norm_num
  calc
    Real.exp (-r / 2) * sech x *
        |smoothSech3First r x - smoothSech3 r x * Real.tanh x| ≤
        1 * 1 * 4 := by
          have h₁ : Real.exp (-r / 2) * sech x ≤ 1 * 1 :=
            mul_le_mul he (sech_le_one x) (sech_pos x).le (by norm_num)
          exact mul_le_mul h₁ hb (abs_nonneg _) (by norm_num)
    _ = 4 := by norm_num

private lemma abs_tiltedSech4Second_le_twenty {r x : ℝ} (hr : 0 ≤ r) :
    |tiltedSech4Second r x| ≤ 20 := by
  unfold tiltedSech4Second
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos (sech_pos _)]
  have he : Real.exp (-r / 2) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (by linarith)
  have hinside : |smoothSech3Second r x -
      2 * smoothSech3First r x * Real.tanh x +
      smoothSech3 r x * (Real.tanh x ^ 2 - sech x ^ 2)| ≤ 20 := by
    calc
      |smoothSech3Second r x - 2 * smoothSech3First r x * Real.tanh x +
          smoothSech3 r x * (Real.tanh x ^ 2 - sech x ^ 2)| ≤
          |smoothSech3Second r x| + 2 * |smoothSech3First r x| * |Real.tanh x| +
            |smoothSech3 r x| * (|Real.tanh x| ^ 2 + |sech x| ^ 2) := by
              calc
                _ ≤ |smoothSech3Second r x -
                    2 * smoothSech3First r x * Real.tanh x| +
                    |smoothSech3 r x * (Real.tanh x ^ 2 - sech x ^ 2)| :=
                  abs_add_le _ _
                _ ≤ (|smoothSech3Second r x| +
                    |2 * smoothSech3First r x * Real.tanh x|) +
                    |smoothSech3 r x| * |Real.tanh x ^ 2 - sech x ^ 2| := by
                      gcongr
                      · exact abs_sub _ _
                      · rw [abs_mul]
                _ ≤ _ := by
                  rw [abs_mul, abs_mul,
                    abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
                  gcongr
                  calc
                    |Real.tanh x ^ 2 - sech x ^ 2| =
                        |Real.tanh x ^ 2 + -(sech x ^ 2)| := by ring
                    _ ≤ |Real.tanh x ^ 2| + |-(sech x ^ 2)| := abs_add_le _ _
                    _ = |Real.tanh x| ^ 2 + |sech x| ^ 2 := by
                      rw [abs_neg, abs_pow, abs_pow]
      _ ≤ 12 + 2 * 3 * 1 + 1 * (1 ^ 2 + 1 ^ 2) := by
        gcongr
        · exact abs_smoothSech3Second_le_twelve r x
        · exact abs_smoothSech3First_le_three r x
        · exact abs_tanh_le_one x
        · exact abs_smoothSech3_le_one r x
        · exact abs_tanh_le_one x
        · exact abs_sech_le_one x
      _ = 20 := by norm_num
  calc
    Real.exp (-r / 2) * sech x *
        |smoothSech3Second r x - 2 * smoothSech3First r x * Real.tanh x +
          smoothSech3 r x * (Real.tanh x ^ 2 - sech x ^ 2)| ≤
        1 * 1 * 20 := by
          have h₁ : Real.exp (-r / 2) * sech x ≤ 1 * 1 :=
            mul_le_mul he (sech_le_one x) (sech_pos x).le (by norm_num)
          exact mul_le_mul h₁ hinside (abs_nonneg _) (by norm_num)
    _ = 20 := by norm_num

private lemma tiltedSech4First_deriv (r x : ℝ) :
    deriv (tiltedSech4First r) x = tiltedSech4Second r x :=
  (tiltedSech4First_hasDerivAt_x r x).deriv

private lemma contDiff_tiltedSech4First (r : ℝ) :
    ContDiff ℝ 1 (tiltedSech4First r) := by
  rw [contDiff_one_iff_deriv]
  refine ⟨fun x => (tiltedSech4First_hasDerivAt_x r x).differentiableAt, ?_⟩
  have heq : deriv (tiltedSech4First r) = tiltedSech4Second r := by
    funext x
    exact tiltedSech4First_deriv r x
  rw [heq]
  exact continuous_tiltedSech4Second r

private lemma tiltedSech4First_shift_moderate {r : ℝ} (hr : 0 ≤ r) (h : ℝ) :
    HasModerateGrowth (fun y => tiltedSech4First r (h + y)) := by
  refine ⟨21, 0, by norm_num, ?_, ?_⟩
  · intro y
    simpa using (abs_tiltedSech4First_le_four (x := h + y) hr).trans (by norm_num)
  · intro y
    have hderiv : deriv (fun y => tiltedSech4First r (h + y)) y =
        tiltedSech4Second r (h + y) := by
      simpa [Function.comp_def] using
        ((tiltedSech4First_hasDerivAt_x r (h + y)).comp y
          ((hasDerivAt_id y).const_add h)).deriv
    rw [hderiv]
    simpa using (abs_tiltedSech4Second_le_twenty (x := h + y) hr).trans (by norm_num)

private lemma continuous_smoothSech3_time (x : ℝ) :
    Continuous (fun r => smoothSech3 r x) := by
  unfold smoothSech3 standardGaussianExpectation
  rw [continuous_iff_continuousAt]
  intro r₀
  have hmeas : ∀ᶠ r in nhds r₀,
      AEStronglyMeasurable (fun z => sech3 (x + Real.sqrt r * z))
        (gaussianReal 0 1) := by
    exact Filter.Eventually.of_forall fun r =>
      (continuous_sech3.comp (by fun_prop)).aestronglyMeasurable
  have hbound : ∀ᶠ r in nhds r₀, ∀ᵐ z ∂gaussianReal 0 1,
      ‖sech3 (x + Real.sqrt r * z)‖ ≤ (1 : ℝ) := by
    exact Filter.Eventually.of_forall fun r => ae_of_all _ fun z => by
      simpa [Real.norm_eq_abs] using sech3_abs_le_one (x + Real.sqrt r * z)
  have hlim : ∀ᵐ z ∂gaussianReal 0 1,
      Filter.Tendsto (fun r => sech3 (x + Real.sqrt r * z)) (nhds r₀)
        (nhds (sech3 (x + Real.sqrt r₀ * z))) := by
    exact ae_of_all _ fun z =>
      (continuous_sech3.comp (by fun_prop)).continuousAt.tendsto
  exact tendsto_integral_filter_of_dominated_convergence
    (l := nhds r₀) (F := fun r z => sech3 (x + Real.sqrt r * z))
    (f := fun z => sech3 (x + Real.sqrt r₀ * z))
    (bound := fun _ => (1 : ℝ)) hmeas hbound (integrable_const 1) hlim

private lemma continuous_tiltedSech4Value_time (x : ℝ) :
    Continuous (fun r => tiltedSech4Value r x) := by
  unfold tiltedSech4Value
  exact ((Real.continuous_exp.comp (by fun_prop)).mul
    (continuous_smoothSech3_time x)).mul continuous_const

private lemma abs_tiltedSech4Value_le_exp (r x : ℝ) :
    |tiltedSech4Value r x| ≤ Real.exp (-r / 2) := by
  unfold tiltedSech4Value
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos (sech_pos _)]
  have h₁ : |smoothSech3 r x| * sech x ≤ 1 * 1 :=
    mul_le_mul (abs_smoothSech3_le_one r x) (sech_le_one x)
      (sech_pos x).le (by norm_num)
  simpa [mul_assoc] using
    mul_le_mul_of_nonneg_left h₁ (Real.exp_pos (-r / 2)).le

private noncomputable def tiltedSech4Average (v : ℝ≥0) (h r : ℝ) : ℝ :=
  ∫ x, tiltedSech4Value r x ∂gaussianReal h v

private lemma continuous_tiltedSech4Average (v : ℝ≥0) (h : ℝ) :
    Continuous (tiltedSech4Average v h) := by
  rw [continuous_iff_continuousAt]
  intro r₀
  unfold tiltedSech4Average
  let C : ℝ := Real.exp (-r₀ / 2) + 1
  have hC : 0 < C := by dsimp [C]; positivity
  have hexp : ∀ᶠ r in nhds r₀, Real.exp (-r / 2) < C := by
    have ht : Filter.Tendsto (fun r : ℝ => Real.exp (-r / 2)) (nhds r₀)
        (nhds (Real.exp (-r₀ / 2))) := by
      exact (Real.continuous_exp.comp (by fun_prop :
        Continuous fun r : ℝ => -r / 2)).continuousAt.tendsto
    exact ht.eventually (Iio_mem_nhds (by dsimp [C]; linarith))
  have hmeas : ∀ᶠ r in nhds r₀,
      AEStronglyMeasurable (tiltedSech4Value r) (gaussianReal h v) := by
    exact Filter.Eventually.of_forall fun r =>
      (continuous_tiltedSech4Value r).aestronglyMeasurable
  have hbound : ∀ᶠ r in nhds r₀, ∀ᵐ x ∂gaussianReal h v,
      ‖tiltedSech4Value r x‖ ≤ C := by
    filter_upwards [hexp] with r hr
    exact ae_of_all _ fun x => by
      rw [Real.norm_eq_abs]
      exact (abs_tiltedSech4Value_le_exp r x).trans hr.le
  have hlim : ∀ᵐ x ∂gaussianReal h v,
      Filter.Tendsto (fun r => tiltedSech4Value r x) (nhds r₀)
        (nhds (tiltedSech4Value r₀ x)) := by
    exact ae_of_all _ fun x => (continuous_tiltedSech4Value_time x).continuousAt.tendsto
  exact tendsto_integral_filter_of_dominated_convergence
    (l := nhds r₀) (F := fun r x => tiltedSech4Value r x)
    (f := tiltedSech4Value r₀) (bound := fun _ => C)
    hmeas hbound (integrable_const C) hlim

private lemma tiltedSech4Average_hasDerivAt {v : ℝ≥0} {h r : ℝ} (hr : 0 < r) :
    HasDerivAt (tiltedSech4Average v h)
      (∫ x, (1 / 2) * Real.exp (-r / 2) * sech x *
        (smoothSech3Second r x - smoothSech3 r x) ∂gaussianReal h v) r := by
  unfold tiltedSech4Average
  let F : ℝ → ℝ → ℝ := fun t x => tiltedSech4Value t x
  let F' : ℝ → ℝ → ℝ := fun t x =>
    (1 / 2) * Real.exp (-t / 2) * sech x *
      (smoothSech3Second t x - smoothSech3 t x)
  have hhalf : 0 < r / 2 := by linarith
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal h v) (F := F) (F' := F') (x₀ := r)
    (s := Set.Ioi (r / 2)) (bound := fun _ => (7 : ℝ))
    (Ioi_mem_nhds (by linarith))
    (Filter.Eventually.of_forall fun t =>
      (continuous_tiltedSech4Value t).aestronglyMeasurable)
    (by
      apply Integrable.of_bound (C := 1)
      · exact (continuous_tiltedSech4Value r).aestronglyMeasurable
      · filter_upwards [] with x
        simpa [Real.norm_eq_abs] using abs_tiltedSech4Value_le_one (x := x) hr.le)
    (by
      apply Continuous.aestronglyMeasurable
      dsimp [F']
      exact ((((continuous_const.mul continuous_const).mul continuous_sech).mul
        ((continuous_smoothSech3Second r).sub (continuous_smoothSech3 r)))))
    (by
      filter_upwards [] with x
      intro t ht
      have ht0 : 0 ≤ t := (lt_trans hhalf ht).le
      dsimp [F']
      rw [abs_mul, abs_mul, abs_mul,
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2),
        abs_of_pos (Real.exp_pos _), abs_of_pos (sech_pos _)]
      have he : Real.exp (-t / 2) ≤ 1 := by
        rw [← Real.exp_zero]
        exact Real.exp_le_exp.mpr (by linarith)
      have hd : |smoothSech3Second t x - smoothSech3 t x| ≤ 13 := by
        calc
          _ ≤ |smoothSech3Second t x| + |smoothSech3 t x| := abs_sub _ _
          _ ≤ 12 + 1 := by
            gcongr
            · exact abs_smoothSech3Second_le_twelve t x
            · exact abs_smoothSech3_le_one t x
          _ = 13 := by norm_num
      have hp : (1 / 2) * Real.exp (-t / 2) * sech x ≤ (1 / 2) * 1 * 1 := by
        have h₁ : (1 / 2) * Real.exp (-t / 2) ≤ (1 / 2) * 1 :=
          mul_le_mul_of_nonneg_left he (by norm_num)
        exact mul_le_mul h₁ (sech_le_one x) (sech_pos x).le (by norm_num)
      have hconst : (0 : ℝ) ≤ ((1 : ℝ) / 2) * 1 * 1 := by norm_num
      have hmul := mul_le_mul hp hd (abs_nonneg _) hconst
      nlinarith [hmul]
    (integrable_const 7)
    (by
      filter_upwards [] with x
      intro t ht
      exact tiltedSech4Value_hasDerivAt_r (x := x) (lt_trans hhalf ht))
  simpa [F, F'] using h.2

private lemma tiltedSech4Average_deriv_generator {v : ℝ≥0} {h r : ℝ} (hr : 0 < r) :
    HasDerivAt (tiltedSech4Average v h)
      (∫ x, (1 / 2) * tiltedSech4Second r x +
        Real.tanh x * tiltedSech4First r x ∂gaussianReal h v) r := by
  apply (tiltedSech4Average_hasDerivAt (v := v) (h := h) hr).congr_deriv
  apply integral_congr_ae
  filter_upwards [] with x
  exact (tiltedSech4_generator r x).symm

end SpinGlass.AT
