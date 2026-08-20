import ATDefs

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ProbabilityTheory NNReal

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/--
Finite-volume two-replica Guerra--Talagrand bound.

For every Lagrange multiplier `lam : ℝ`,
the constrained two-replica free energy satisfies

  N⁻¹ E log Z_{N,s}^v ≤ gtFunctional β h q s lam v.
-/
theorem twoReplica_GT_bound
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (lam : ℝ)
    (hN : 0 < N)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ attainableOverlaps N) :
    expectedConstrainedFreeEnergy path s v ≤
      gtFunctional β h q s lam v := by
  sorry

end SpinGlass.AT
