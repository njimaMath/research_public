import Lemmas.Cavity.Stability

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem cavityVector_norm_eq_A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    |A path s| ≤ ‖cavityVector path s‖ := by
  -- Proof route: `A path s` is coordinate `0` of `cavityVector path s` by
  -- definition.  Rewrite it, identify `|x|` with `‖x‖`, and apply
  -- `norm_apply_le_norm` for the finite function space `Fin 3 → ℝ`.  No cavity
  -- estimate is needed.  The theorem name should say `le`, not `eq`.
  sorry

theorem uniform_secondMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q, N * A path s ≤ M := by
  -- Paper route, equations (preabsorb)--(thirdsplit): use
  -- `cavity_system`, the uniform inverse bound, and the remainder estimate to
  -- get
  -- `A_s ≤ C/N + C*N^(-3/2) + C*thirdMoment`.  Replica Cauchy--Schwarz gives
  -- `|B_s|, |C_s| ≤ A_s`, so the norm of the cavity vector is controlled by
  -- `A_s`.  Split the cubic moment at a small fixed `eta`:
  -- `E|Q|^3 ≤ eta*A_s + 8*P(|Q|>eta)`.  Apply `fixedDeviation`, choose `eta`
  -- so that `C*eta ≤ 1/2`, move that term to the left, and absorb both
  -- `N^(-3/2)` and the exponential tail into `C/N`.  Enlarge the constant for
  -- finitely many small `N`, then multiply by `N`.
  -- This proof becomes available only after the cavity remainder is stated
  -- with a uniform `C_K` and the coupled-pressure/path model is repaired.
  sorry

end SpinGlass.AT
