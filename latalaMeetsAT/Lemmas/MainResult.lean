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

/-- Assembly of the three conclusions once the analytic cavity-remainder
estimate has been supplied.  This is the current compatibility endpoint;
the blueprint's final theorem must discharge `hCavity` internally. -/
theorem quantitative_strictAT_of_cavity_remainder {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ))
    (data : UniformATData K)
    (hCavity : ∃ C, HasCavityRemainderBound (Ω := Ω) data C) :
    QuantitativeATConclusion (Ω := Ω) K := by
  -- Compactness, positivity, and the uniform strict-AT gap are carried by
  -- `data`. The three fields are exactly the absorption, free-energy, and
  -- replicon theorems, so this wrapper contains no further analytic argument.
  obtain ⟨Crem, hCrem⟩ := hCavity
  exact
    { secondMoment := uniform_secondMoment data Crem hCrem
      freeEnergy := rs_freeEnergy_error data Crem hCrem
      replicon := replicon_susceptibility data Crem hCrem }

/-- Compatibility name retained for `Latala_AT.lean`. -/
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ))
    (data : UniformATData K)
    (hCavity : ∃ C, HasCavityRemainderBound (Ω := Ω) data C) :
    QuantitativeATConclusion (Ω := Ω) K :=
  quantitative_strictAT_of_cavity_remainder K data hCavity

#print axioms SpinGlass.AT.quantitative_strictAT

end SpinGlass.AT
