import Lemmas.Scalar.Semigroup

set_option autoImplicit false

namespace SpinGlass.AT

noncomputable def latalaH (t y : ℝ) : ℝ := (1 - t * y) ^ 2 / (1 + t ^ 2)

noncomputable def latalaF (lam y : ℝ) : ℝ := Real.exp (-lam * y) * latalaH lam y

noncomputable def referenceDensity (y : ℝ) : ℝ :=
  1 / (2 * Real.sqrt (1 - y))

theorem latalaH_deriv_nonpos {t y : ℝ} (ht : 0 ≤ t) (hy : 0 ≤ y) :
    deriv (fun x => latalaH x y) t ≤ 0 := by
  -- Statement repair required.  Direct differentiation gives
  -- `2 * (1 - t*y) * (-y - t) / (1 + t^2)^2`.  The claimed sign follows
  -- when `t*y ≤ 1`, but is reversed once `t*y > 1`; for example `t = 2` and
  -- `y = 1` contradict the present statement.  Add `t * y ≤ 1`, prove the
  -- derivative with `HasDerivAt` rules for division, rewrite `deriv` using
  -- `.deriv`, and discharge the sign by `positivity` and `nlinarith`.
  sorry

theorem latalaF_antitone {lam : ℝ} (hlam : 0 ≤ lam) :
    AntitoneOn (latalaF lam) (Set.Icc (0 : ℝ) 1) := by
  -- Paper route: this file should encode equations (Flambda)--(Fmeanone), where
  -- the decreasing kernel is
  -- `(1 + (4 - 3*y)*t) / (1 + y*t)^3` for fixed `t ≥ 0`.  Its derivative is
  -- `-6*t*(1 + (2-y)*t)/(1+y*t)^4 ≤ 0` on `[0,1]`.
  -- The current `latalaF = exp (-lam*y) * latalaH lam y` is not that kernel,
  -- and `latalaH` needs the extra `lam*y ≤ 1` condition noted above.  Replace
  -- `latalaF` by the paper's kernel, or add hypotheses strong enough to control
  -- both factors; then use `antitoneOn_of_deriv_nonpos` on `[0,1]`.
  sorry

theorem opposite_monotone_covariance_le
    {f g : ℝ → ℝ} (hf : Monotone f) (hg : Antitone g) :
    standardGaussianExpectation (fun z => f z * g z) ≤
      standardGaussianExpectation f * standardGaussianExpectation g := by
  -- Statement repair required.  The paper uses the opposite-monotonicity
  -- covariance inequality for a probability measure on `[0,1]`, not only for
  -- a Gaussian, and assumes the relevant integrals exist.  Add integrability
  -- hypotheses for `f`, `g`, and `f*g` (or boundedness), generalize to a
  -- probability measure `μ`, and integrate
  -- `(f x - f y) * (g x - g y) ≤ 0` over `μ.prod μ`.  Fubini and the probability
  -- normalization expand this to twice the desired covariance inequality.
  -- Without integrability assumptions, Mathlib's real integral convention
  -- makes the current unrestricted statement unsuitable.
  sorry

theorem latala_diffusion_fourthMoment_le {β h q s u : ℝ}
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hu : u ∈ Set.Icc q 1) :
    0 ≤ scalarPsi β q s u h := by
  -- For the statement currently written, `hu.1` selects
  -- `scalarPsi_eq_upper`; `log (cosh h) ≥ 0`, `0 ≤ s`, and `u ≤ 1` make both
  -- summands nonnegative.  Use `Real.one_le_cosh`, `Real.log_nonneg`, and
  -- `mul_nonneg`/`div_nonneg` to finish.
  --
  -- This is not the diffusion fourth-moment estimate used in the paper's
  -- equations (Flambda-representation)--(strictATdiffusion).  Formalizing that
  -- estimate requires separate definitions for the local-field diffusion,
  -- `S = sech^2 X_q`, and the decreasing kernel `F_lam`, followed by the
  -- opposite-monotonicity covariance lemma above.
  sorry

end SpinGlass.AT
