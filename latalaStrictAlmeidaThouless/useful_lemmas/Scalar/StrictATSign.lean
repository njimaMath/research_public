import Lemmas.Scalar.LatalaKernel

set_option autoImplicit false

namespace SpinGlass.AT

/--
The lower branch corresponding to
`g_s(u) = E[(∂ₓ Ψ_s(u,X_u))²]`.

The square must be outside the inner Gaussian expectation: below `q`,
`∂ₓ Ψ_s(u,x)` is the Gaussian average of `tanh`, not the Gaussian
average of `tanh²`.
-/
noncomputable def scalarOrderParameterCorrect (β h s u : ℝ) : ℝ :=
  let q := rsQ β h
  if u ≤ q then
    standardGaussianExpectation (fun z₀ =>
      (standardGaussianExpectation (fun z =>
        Real.tanh
          (h + β * Real.sqrt ((1 - s) * q + s * u) * z₀ +
            β * Real.sqrt (s * (q - u)) * z))) ^ 2)
  else
    let lam := s * β ^ 2 * (u - q)
    standardGaussianExpectation (fun z₀ =>
      let x := h + β * Real.sqrt q * z₀
      Real.exp (-lam / 2) / Real.cosh x *
        standardGaussianExpectation (fun z =>
          Real.tanh (x + Real.sqrt lam * z) ^ 2 *
            Real.cosh (x + Real.sqrt lam * z)))

/--
Purely order-theoretic closure of the strict-AT sign argument.

The two hypotheses are exactly the analytic estimates that the scalar
PDE/diffusion calculation must provide:

* on the left of `q`, `g(u) - u ≥ gap * (q-u)`;
* on the right of `q`, `u - g(u) ≥ gap * (u-q)`.

Once these estimates are available, the sign and the local linear
absolute-value estimate require no further analytic input.
-/
theorem strictAT_sign_of_linear_bounds
    {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (_hs : s ∈ Set.Icc (0 : ℝ) 1)
    (g : ℝ → ℝ)
    (hleft : ∀ u, 0 ≤ u → u ≤ rsQ β h →
      data.gap * (rsQ β h - u) ≤ g u - u)
    (hright : ∀ u, rsQ β h ≤ u → u ≤ 1 →
      data.gap * (u - rsQ β h) ≤ u - g u) :
    (∀ u, 0 ≤ u → u < rsQ β h →
      data.gap * (rsQ β h - u) ≤ g u - u) ∧
    (∀ u, rsQ β h < u → u ≤ 1 → g u - u < 0) ∧
    ∃ c eps, 0 < c ∧ 0 < eps ∧ ∀ u,
      |u - rsQ β h| ≤ eps →
      c * |u - rsQ β h| ≤ |g u - u| := by
  have hβ : 0 < β := data.β_pos (β, h) hp
  have hh : 0 < h := data.h_pos (β, h) hp
  have hqpos : 0 < rsQ β h := rsQ_pos hβ hh
  have hqlt : rsQ β h < 1 := rsQ_lt_one hβ hh
  have hgap : 0 < data.gap := data.gap_pos
  constructor
  · intro u hu0 huq
    exact hleft u hu0 huq.le
  constructor
  · intro u hqu hu1
    have hlin := hright u hqu.le hu1
    have hpositive : 0 < data.gap * (u - rsQ β h) :=
      mul_pos hgap (sub_pos.mpr hqu)
    linarith
  · refine ⟨data.gap, min (rsQ β h) (1 - rsQ β h), hgap, ?_, ?_⟩
    · exact lt_min hqpos (sub_pos.mpr hqlt)
    · intro u huabs
      by_cases huq : u ≤ rsQ β h
      · have habs_eq : |u - rsQ β h| = rsQ β h - u := by
          rw [abs_of_nonpos (sub_nonpos.mpr huq)]
          ring
        have hu0 : 0 ≤ u := by
          have hqu_le_q : rsQ β h - u ≤ rsQ β h := by
            calc
              rsQ β h - u = |u - rsQ β h| := habs_eq.symm
              _ ≤ min (rsQ β h) (1 - rsQ β h) := huabs
              _ ≤ rsQ β h := min_le_left _ _
          linarith
        have hlin := hleft u hu0 huq
        have hnonneg : 0 ≤ g u - u := by
          exact le_trans
            (mul_nonneg hgap.le (sub_nonneg.mpr huq)) hlin
        calc
          data.gap * |u - rsQ β h| =
              data.gap * (rsQ β h - u) := by rw [habs_eq]
          _ ≤ g u - u := hlin
          _ = |g u - u| := (abs_of_nonneg hnonneg).symm
      · have hqu : rsQ β h < u := lt_of_not_ge huq
        have habs_eq : |u - rsQ β h| = u - rsQ β h :=
          abs_of_nonneg (sub_nonneg.mpr hqu.le)
        have hu1 : u ≤ 1 := by
          have huq_le : u - rsQ β h ≤ 1 - rsQ β h := by
            calc
              u - rsQ β h = |u - rsQ β h| := habs_eq.symm
              _ ≤ min (rsQ β h) (1 - rsQ β h) := huabs
              _ ≤ 1 - rsQ β h := min_le_right _ _
          linarith
        have hlin := hright u hqu.le hu1
        have hnonneg : 0 ≤ u - g u := by
          exact le_trans
            (mul_nonneg hgap.le (sub_nonneg.mpr hqu.le)) hlin
        have hnonpos : g u - u ≤ 0 := by linarith
        calc
          data.gap * |u - rsQ β h| =
              data.gap * (u - rsQ β h) := by rw [habs_eq]
          _ ≤ u - g u := hlin
          _ = |g u - u| := by
            rw [abs_of_nonpos hnonpos]
            ring

/-- Specialization of `strictAT_sign_of_linear_bounds` to the corrected
scalar order parameter. -/
theorem strictAT_sign_of_scalar_linear_bounds
    {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hleft : ∀ u, 0 ≤ u → u ≤ rsQ β h →
      data.gap * (rsQ β h - u) ≤
        scalarOrderParameterCorrect β h s u - u)
    (hright : ∀ u, rsQ β h ≤ u → u ≤ 1 →
      data.gap * (u - rsQ β h) ≤
        u - scalarOrderParameterCorrect β h s u) :
    (∀ u, 0 ≤ u → u < rsQ β h →
      data.gap * (rsQ β h - u) ≤
        scalarOrderParameterCorrect β h s u - u) ∧
    (∀ u, rsQ β h < u → u ≤ 1 →
      scalarOrderParameterCorrect β h s u - u < 0) ∧
    ∃ c eps, 0 < c ∧ 0 < eps ∧ ∀ u,
      |u - rsQ β h| ≤ eps →
      c * |u - rsQ β h| ≤
        |scalarOrderParameterCorrect β h s u - u| := by
  exact strictAT_sign_of_linear_bounds data hp hs
    (scalarOrderParameterCorrect β h s) hleft hright

end SpinGlass.AT
