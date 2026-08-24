import Lemmas.MainResult

/-!
# Quantitative strict-AT theorem

This entry module states the three quantitative conclusions explicitly. The
analytic proof is supplied by `SpinGlass.AT.quantitative_strictAT` from
`Lemmas.MainResult`. Do not change the file
-/

open MeasureTheory SpinGlass.AT

set_option autoImplicit false

universe u

structure QuantitativeAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q, N * A path s ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      0 ≤ rsFreeEnergy β h - skFreeEnergy path ∧
      rsFreeEnergy β h - skFreeEnergy path ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      |N * (A path s - 2 * B path s + C path s) -
        rsA β h / (1 - s * atParameter β h)| < eps

/-- Quantitative strict-AT theorem with all analytic inputs supplied by named
lemmas rather than project-specific typeclass assumptions. -/
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (K : Set (ℝ × ℝ))
    (data : UniformATData K) :
    QuantitativeAT (Ω := Ω) K := by
  have result : SpinGlass.AT.QuantitativeATConclusion (Ω := Ω) K :=
    SpinGlass.AT.quantitative_strictAT K data
  exact
    { secondMoment := result.secondMoment
      freeEnergy := result.freeEnergy
      replicon := result.replicon }
