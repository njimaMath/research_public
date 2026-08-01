import Lemmas.CoupledPressure

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def quenchedTail {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s eps : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0)

/-- Contract for the uniform Gaussian concentration estimate used in the
fixed-deviation tail bound.  A finite Gaussian-coordinate realization of the
smart path supplies this contract. -/
class HasFixedDeviationEstimate (Ω : Type u) [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] : Prop where
  bound : ∀ {K : Set (ℝ × ℝ)} (data : UniformATData K)
      (eps : ℝ), 0 < eps →
    ∃ c C, 0 < c ∧ 0 < C ∧ ∀ {N : ℕ} {β h q s : ℝ}
      (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      quenchedTail path s eps ≤ C * Real.exp (-c * N)

theorem fixedDeviation {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    [HasFixedDeviationEstimate Ω] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (eps : ℝ) (heps : 0 < eps) :
    ∃ c C, 0 < c ∧ 0 < C ∧ ∀ {N : ℕ} {β h q s : ℝ}
      (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      quenchedTail path s eps ≤ C * Real.exp (-c * N) := by
  exact HasFixedDeviationEstimate.bound data eps heps

end SpinGlass.AT
