import Lemmas.CompactCorollary

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

structure QuantitativeATConclusion {Ω : Type u} [MeasureSpace Ω]
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

theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K) (hβ : ∀ p ∈ K, 0 < p.1)
    (hh : ∀ p ∈ K, 0 < p.2)
    (hAT : ∀ p ∈ K, atParameter p.1 p.2 < 1) :
    QuantitativeATConclusion (Ω := Ω) K := by
  -- Proof route: compactness and strict AT produce `UniformATData`; the three
  -- fields are exactly the absorption, free-energy, and replicon theorems.
  -- This wrapper contains no further analytic argument.
  let data := Classical.choice
    (uniformATData_of_compact_strictAT K hKcompact hβ hh hAT)
  exact
    { secondMoment := uniform_secondMoment data
      freeEnergy := rs_freeEnergy_error data
      replicon := replicon_susceptibility data }

#print axioms SpinGlass.AT.quantitative_strictAT

end SpinGlass.AT
