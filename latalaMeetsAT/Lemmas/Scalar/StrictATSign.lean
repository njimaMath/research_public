import Lemmas.Scalar.LatalaKernel

set_option autoImplicit false

namespace SpinGlass.AT

/-- $g_s(u)=\mathbb E[(\partial_x\Psi_s(u,X_u))^2]$ for the local-field
diffusion in Proposition `pathRS`.  Its construction is one of the remaining
stochastic-calculus obligations. -/
noncomputable def scalarOrderParameter (β h s u : ℝ) : ℝ :=
  by
    sorry

theorem strictAT_sign {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    (∀ u, 0 ≤ u → u < rsQ β h →
      data.gap * (rsQ β h - u) ≤ scalarOrderParameter β h s u - u) ∧
    (∀ u, rsQ β h < u → u ≤ 1 → scalarOrderParameter β h s u - u < 0) ∧
    ∃ c eps, 0 < c ∧ 0 < eps ∧ ∀ u,
      |u - rsQ β h| ≤ eps →
      c * |u - rsQ β h| ≤ |scalarOrderParameter β h s u - u| := by
  -- Paper route: Proposition (pathRS), equations (leftstrict),
  -- (strictATdiffusion), and (linearATsign).  Define `scalarOrderParameter` as
  -- `g_s(u) = E[(Psi_x(u, X_u))^2]` for the scalar PDE/local-field diffusion.
  -- Below `q`, Itô plus conditional Jensen gives
  -- `g_s'(u) ≤ s*atParameter`, hence the lower bound from `path_gap`.  Above
  -- `q`, the Latała kernel comparison gives
  -- `g_s'(u) ≤ s*atParameter < 1`, hence the negative sign.  Joint continuity
  -- and compactness give uniform `c` and `eps` near `q`.
  sorry

end SpinGlass.AT
