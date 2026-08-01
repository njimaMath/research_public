import Lemmas.Scalar.LatalaKernel

set_option autoImplicit false

namespace SpinGlass.AT

/-- $g_s(u)=\mathbb E[(\partial_x\Psi_s(u,X_u))^2]$ for the local-field
diffusion in Proposition `pathRS`.  The lower branch uses the drift-free
Gaussian representation up to `q`; the upper branch uses the Girsanov
transition formula from the diffusion-comparison lemma. -/
noncomputable def scalarOrderParameter (β h s u : ℝ) : ℝ :=
  let q := rsQ β h
  if u ≤ q then
    standardGaussianExpectation (fun z₀ =>
      standardGaussianExpectation (fun z =>
        Real.tanh
          (h + β * Real.sqrt ((1 - s) * q + s * u) * z₀ +
            β * Real.sqrt (s * (q - u)) * z)) ^ 2)
  else
    let lam := s * β ^ 2 * (u - q)
    standardGaussianExpectation (fun z₀ =>
      let x := h + β * Real.sqrt q * z₀
      Real.exp (-lam / 2) / Real.cosh x *
        standardGaussianExpectation (fun z =>
          Real.tanh (x + Real.sqrt lam * z) ^ 2 *
            Real.cosh (x + Real.sqrt lam * z)))

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
  -- NEEDED: differentiate the two explicit Gaussian formulas defining
  -- `scalarOrderParameter`, apply the diffusion comparison above `q`, and
  -- prove compact-uniform continuity of the resulting derivatives.
  -- BLUEPRINT: Lemma `diffusioncomparison` and Proposition `pathRS`.
  sorry

end SpinGlass.AT
