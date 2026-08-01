import Lemmas.MainResult
import SpinGlass.SKModel

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

namespace SpinGlass.AT
namespace SK

universe u

variable {Ω : Type u} [MeasureSpace Ω]
  [IsProbabilityMeasure (volume : Measure Ω)]

/-!
# Strict-AT conclusions for the SK model

The finite-volume model below uses `SpinGlass.SKDisorder`, `SpinGlass.Z`,
`SpinGlass.gibbs_pmf`, and `SpinGlass.overlap` from `SpinGlass.SKModel` and
`SpinGlass.Defs`.  The sign in `paperEnergy` makes the convention
`Z H = ∑ σ, exp (-H σ)` agree with the paper's convention `∑ σ, exp (H_N σ)`.

`SmartPathBridge` records the endpoint identification between that model and
the covariance-based smart path used by the strict-AT argument.  It makes the
model-conversion hypothesis explicit instead of silently identifying the two
Hamiltonian APIs.
-/

/-- The SK energy in the sign convention of `SpinGlass.Z`. -/
noncomputable def paperEnergy {N : ℕ} {β h : ℝ}
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) (ω : Ω) :
    SpinGlass.EnergySpace N :=
  sk.U ω - SpinGlass.magnetic_field_vector (N := N) h

/-- The finite-volume SK partition function. -/
noncomputable def partitionFunction {N : ℕ} {β h : ℝ}
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) (ω : Ω) : ℝ :=
  SpinGlass.Z N (paperEnergy sk ω)

/-- The Gibbs probability of a configuration at fixed disorder. -/
noncomputable def gibbsProbability {N : ℕ} {β h : ℝ}
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) (ω : Ω)
    (σ : SpinGlass.Config N) : ℝ :=
  SpinGlass.gibbs_pmf N (paperEnergy sk ω) σ

/-- The quenched free-energy density of the SK model. -/
noncomputable def quenchedFreeEnergy {N : ℕ} {β h : ℝ}
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) : ℝ :=
  ∫ ω, SpinGlass.free_energy_density (N := N) (paperEnergy sk ω)
    ∂(volume : Measure Ω)

/-- The quenched centered-overlap second moment. -/
noncomputable def overlapSecondMoment {N : ℕ} {β h : ℝ} (q : ℝ)
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) : ℝ :=
  ∫ ω, ∑ σs : Fin 2 → SpinGlass.Config N,
    (SpinGlass.overlap N (σs 0) (σs 1) - q) ^ 2 *
      ∏ a : Fin 2, gibbsProbability sk ω (σs a)
    ∂(volume : Measure Ω)

/-- The replica-symmetric free energy appearing in `paper.tex`. -/
noncomputable def replicaSymmetricFreeEnergy (β h : ℝ) : ℝ :=
  SpinGlass.AT.rsFreeEnergy β h

/-- Identification of an SK disorder with the abstract smart path.

The two equalities are precisely the endpoint facts used by the final theorem.
-/
structure SmartPathBridge {N : ℕ} {β h q : ℝ}
    (sk : SpinGlass.SKDisorder (Ω := Ω) N β h) where
  path : SpinGlass.AT.RSSmartPathDisorder Ω N β h q
  overlapSecondMoment_eq :
    overlapSecondMoment q sk = SpinGlass.AT.A path 1
  freeEnergy_eq :
    quenchedFreeEnergy sk = SpinGlass.AT.skFreeEnergy path

/-- The replicon combination along the smart path attached to an SK disorder. -/
noncomputable def repliconObservable {N : ℕ} {β h q : ℝ}
    {sk : SpinGlass.SKDisorder (Ω := Ω) N β h}
    (bridge : SmartPathBridge (q := q) sk) (s : ℝ) : ℝ :=
  SpinGlass.AT.A bridge.path s - 2 * SpinGlass.AT.B bridge.path s +
    SpinGlass.AT.C bridge.path s

/-- The three uniform conclusions of the paper, stated for `SKDisorder`. -/
structure QuantitativeSKATConclusion (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ}
      (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
      (_bridge : SmartPathBridge (q := q) sk),
      (β, h) ∈ K → q = SpinGlass.AT.rsQ β h →
      N * overlapSecondMoment q sk ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ}
      (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
      (_bridge : SmartPathBridge (q := q) sk),
      (β, h) ∈ K → q = SpinGlass.AT.rsQ β h →
      0 ≤ replicaSymmetricFreeEnergy β h - quenchedFreeEnergy sk ∧
      replicaSymmetricFreeEnergy β h - quenchedFreeEnergy sk ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ}
      (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
      (bridge : SmartPathBridge (q := q) sk),
      (β, h) ∈ K → q = SpinGlass.AT.rsQ β h →
      s ∈ Set.Icc (0 : ℝ) 1 →
      |N * repliconObservable bridge s -
        SpinGlass.AT.rsA β h /
          (1 - s * SpinGlass.AT.atParameter β h)| < eps

/-- The quantitative strict-AT theorem specialized to the SK definitions. -/
theorem quantitative_strictAT_for_sk (K : Set (ℝ × ℝ))
    (data : SpinGlass.AT.UniformATData K) :
    QuantitativeSKATConclusion (Ω := Ω) K := by
  have hmain := SpinGlass.AT.quantitative_strictAT
    (Ω := Ω) K data
  refine {
    secondMoment := ?_
    freeEnergy := ?_
    replicon := ?_ }
  · obtain ⟨M, hM, hbound⟩ := hmain.secondMoment
    refine ⟨M, hM, ?_⟩
    intro N hN β h q sk bridge hp hq
    rw [bridge.overlapSecondMoment_eq]
    exact hbound hN hp hq (by norm_num) bridge.path
  · obtain ⟨M, hM, hbound⟩ := hmain.freeEnergy
    refine ⟨M, hM, ?_⟩
    intro N hN β h q sk bridge hp hq
    rw [replicaSymmetricFreeEnergy, bridge.freeEnergy_eq]
    exact hbound hN hp hq bridge.path
  · intro eps heps
    obtain ⟨N0, hbound⟩ := hmain.replicon eps heps
    refine ⟨N0, ?_⟩
    intro N hN β h q s sk bridge hp hq hs
    simpa [repliconObservable] using
      hbound hN hp hq hs bridge.path

end SK
end SpinGlass.AT
