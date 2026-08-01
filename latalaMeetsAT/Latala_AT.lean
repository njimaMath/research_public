import Lemmas.MainResult

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT
namespace SK

universe u

variable {Ω : Type u} [MeasureSpace Ω]
  [IsProbabilityMeasure (volume : Measure Ω)]

/-!
# Strict-AT conclusions at the SK endpoint

The strict-AT development uses `RSSmartPathDisorder` as its model interface.
Its value at `s = 1` is the finite-volume SK endpoint.  Stating the result at
this interface avoids assuming a separate record of endpoint equalities.

The concrete `SpinGlass.SKDisorder` interface uses a different Hamiltonian
sign convention and does not contain the independent one-site Gaussian field
needed to construct the whole smart path on the same probability space.
-/

/-- The quenched centered-overlap second moment at the SK endpoint. -/
noncomputable def overlapSecondMoment {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : ℝ :=
  A path 1

/-- The quenched free-energy density at the SK endpoint. -/
noncomputable def quenchedFreeEnergy {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : ℝ :=
  skFreeEnergy path

/-- The replica-symmetric free energy. -/
noncomputable def replicaSymmetricFreeEnergy (β h : ℝ) : ℝ :=
  rsFreeEnergy β h

/-- The replicon combination along the smart path. -/
noncomputable def repliconObservable {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  A path s - 2 * B path s + C path s

/-- The three uniform strict-AT conclusions, including the SK endpoint. -/
structure QuantitativeSKATConclusion (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
        N * overlapSecondMoment path ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
        0 ≤ replicaSymmetricFreeEnergy β h - quenchedFreeEnergy path ∧
        replicaSymmetricFreeEnergy β h - quenchedFreeEnergy path ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
        |N * repliconObservable path s -
          rsA β h / (1 - s * atParameter β h)| < eps

/-- The quantitative strict-AT theorem at the SK endpoint. -/
theorem quantitative_strictAT_for_sk (K : Set (ℝ × ℝ))
    (data : UniformATData K) :
    QuantitativeSKATConclusion (Ω := Ω) K := by
  have hmain := quantitative_strictAT (Ω := Ω) K data
  refine {
    secondMoment := ?_
    freeEnergy := ?_
    replicon := ?_ }
  · obtain ⟨M, hM, hbound⟩ := hmain.secondMoment
    refine ⟨M, hM, ?_⟩
    intro N hN β h q hp hq path
    simpa [overlapSecondMoment] using
      hbound hN hp hq (by norm_num) path
  · obtain ⟨M, hM, hbound⟩ := hmain.freeEnergy
    refine ⟨M, hM, ?_⟩
    intro N hN β h q hp hq path
    simpa [replicaSymmetricFreeEnergy, quenchedFreeEnergy] using
      hbound hN hp hq path
  · intro eps heps
    obtain ⟨N0, hbound⟩ := hmain.replicon eps heps
    refine ⟨N0, ?_⟩
    intro N hN β h q s hp hq hs path
    simpa [repliconObservable] using
      hbound hN hp hq hs path

end SK
end SpinGlass.AT
