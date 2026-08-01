import Lemmas.GT.Defs

set_option autoImplicit false

namespace SpinGlass.AT

noncomputable def Tzero (f : ℝ → ℝ) (x : ℝ) : ℝ := f x
noncomputable def Thalf (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * Real.log (standardGaussianExpectation (fun z => Real.exp (f (x + z) / 2)))
noncomputable def Tone (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.log (standardGaussianExpectation (fun z => Real.exp (f (x + z))))

theorem Tzero_continuous {f : ℝ → ℝ} (hf : Continuous f) : Continuous (Tzero f) := by
  -- Proof route: `Tzero f` is definitionally `f`.
  simpa [Tzero]

end SpinGlass.AT
