import Lemmas.Scalar.LatalaKernel

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

private theorem latalaH_reference_integral (t : ℝ) (ht : 0 ≤ t) :
    ∫ y in Set.Icc (0 : ℝ) 1, latalaH t y * referenceDensity y = 1 := by
  let f : ℝ → ℝ := fun u => 1 - u ^ 2
  let f' : ℝ → ℝ := fun u => -2 * u
  let g : ℝ → ℝ := fun y => latalaH t y * referenceDensity y
  have hsub := intervalIntegral.integral_comp_mul_deriv_of_deriv_nonpos
    (a := (0 : ℝ)) (b := 1) (f := f) (f' := f') (g := g)
    (by fun_prop)
    (by
      intro u hu
      simpa [f, f', pow_two] using
        ((hasDerivAt_const u 1).sub ((hasDerivAt_id u).pow 2)))
    (by
      intro u hu
      norm_num at hu
      dsimp [f']
      linarith [hu.1])
  have hcomp :
      (∫ u in (0 : ℝ)..1, (g ∘ f) u * f' u) =
        ∫ u in (0 : ℝ)..1, -latalaH t (1 - u ^ 2) := by
    apply intervalIntegral.integral_congr_ae
    filter_upwards [] with u
    intro hu
    have huIoc : u ∈ Set.Ioc (0 : ℝ) 1 := by simpa [Set.uIoc_of_le zero_le_one] using hu
    have huI : u ∈ Set.Icc (0 : ℝ) 1 := ⟨huIoc.1.le, huIoc.2⟩
    have hu0 : u ≠ 0 := ne_of_gt huIoc.1
    have hsqrt : Real.sqrt (u ^ 2) = u := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg huI.1]
    dsimp [g, f, f', referenceDensity]
    rw [show 1 - (1 - u ^ 2) = u ^ 2 by ring, hsqrt]
    field_simp
  rw [hcomp] at hsub
  simp only [f, g, intervalIntegral.integral_symm, sub_self, one_pow,
    one_sub_one, zero_pow (by decide : (2 : ℕ) ≠ 0)] at hsub
  rw [intervalIntegral.integral_of_le zero_le_one]
  rw [← neg_inj]
  simpa only [intervalIntegral.integral_neg] using hsub.symm

end SpinGlass.AT
