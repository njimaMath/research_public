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

theorem gtSemigroup_dlambda_abs_le_one {f : ℝ → ℝ} {lam : ℝ} :
    |deriv (fun x => Thalf (fun y => f y + x) 0) lam| ≤ 1 := by
  -- For this additive special case, factor `exp (x/2)` out of the Gaussian
  -- integral and simplify `Thalf (fun y => f y + x) 0` to `x + constant`.
  -- Its derivative is exactly `1`, so the conclusion is `abs_one.le`.
  -- Add the assumption that the exponential integral is finite and positive;
  -- it is needed to justify `log (exp (x/2) * I) = x/2 + log I`.  For the
  -- paper's actual multiplier derivative, instead propagate
  -- `|partial_lam f_lam| ≤ 1` through tilted expectations as in the paragraph
  -- following equation (GTsecondderivative).
  sorry

theorem gtSemigroup_dlambda2_le_three {f : ℝ → ℝ} {lam : ℝ} :
    deriv (fun x => deriv (fun y => Thalf (fun z => f z + y) 0) x) lam ≤ 3 := by
  -- Proof route: under the same finite-positive-integral assumption as the preceding
  -- lemma, the inner derivative is the constant `1`; its derivative is `0`,
  -- and `norm_num` closes `0 ≤ 3`.  The nontrivial paper estimate uses
  -- `partial_lamlam T_m F = E_m[partial_lamlam F]
  --   + m * Var_m(partial_lam F)`.
  -- Starting from terminal bounds `|partial_lam f_lam| ≤ 1` and
  -- `0 ≤ partial_lamlam f_lam ≤ 1`, at most two positive levels with masses
  -- `1/2` and `1` yield the bound `3`.  That theorem needs a parameterized
  -- semigroup API rather than the additive placeholder used here.
  sorry

end SpinGlass.AT
