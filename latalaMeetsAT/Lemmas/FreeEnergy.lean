import Lemmas.Absorption
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def rsFreeEnergy (β h : ℝ) : ℝ :=
  rsPathValue β h (rsQ β h) 1

noncomputable def skFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : ℝ :=
  pathFreeEnergy path 1

/-- The endpoint/continuity input needed to pass from the open-interval
derivative identity `rsGap_deriv` to a closed-interval free-energy estimate.

For the covariance-only `RSSmartPathDisorder` API, this is the missing
Gaussian-law bridge: one must show that the quenched free energy depends
continuously on the affine covariance kernel and identify the `s = 0`
Gaussian law with the independent one-site field. -/
def HasFreeEnergyEndpointBridge {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : Prop :=
  ContinuousOn
      (fun s => rsPathValue β h q s - pathFreeEnergy path s)
      (Set.Icc (0 : ℝ) 1) ∧
    rsPathValue β h q 0 - pathFreeEnergy path 0 = 0

/-- Uniform endpoint facts from `blueprint_at.tex`, equation
`freeenergyidentity`.  The Gaussian-law continuity and endpoint-identification
proof is isolated here. -/
theorem freeEnergy_endpoint_bridge {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] :
    ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      ∀ path : RSSmartPathDisorder Ω N β h q,
        HasFreeEnergyEndpointBridge path := by
  sorry

/-- The final free-energy argument, with the currently missing endpoint bridge
made explicit.

Once `HasFreeEnergyEndpointBridge path` is proved for every positive-size
smart path, this theorem is a direct replacement for `rs_freeEnergy_error`.
The proof uses the mean-value theorem, positivity of
the overlap second moment, `uniform_secondMoment`, and the compact-set bound
on `β`. -/
theorem rs_freeEnergy_error_of_endpoint_bridge
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data C)
    (hBridge :
      ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
        ∀ path : RSSmartPathDisorder Ω N β h q,
          HasFreeEnergyEndpointBridge path) :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      0 ≤ rsFreeEnergy β h - skFreeEnergy path ∧
      rsFreeEnergy β h - skFreeEnergy path ≤ M / N := by
  obtain ⟨M₀, hM₀, hsecond⟩ :=
    uniform_secondMoment (Ω := Ω) data C hCavity
  let M : ℝ := data.βmax ^ 2 * M₀ / 4
  refine ⟨M, ?_, ?_⟩
  · dsimp [M]
    positivity
  intro N hN β h q hp hq path
  subst q

  let G : ℝ → ℝ :=
    fun s =>
      rsPathValue β h (rsQ β h) s - pathFreeEnergy path s

  have hbridge :
      ContinuousOn G (Set.Icc (0 : ℝ) 1) ∧ G 0 = 0 := by
    simpa [G, HasFreeEnergyEndpointBridge] using
      (hBridge hN path)

  have hderiv :
      ∀ s ∈ Set.Ioo (0 : ℝ) 1,
        HasDerivAt G
          (β ^ 2 / 4 * overlapSecondMoment path s) s := by
    intro s hs
    simpa [G] using rsGap_deriv path hs

  obtain ⟨s, hs, hslope⟩ :=
    exists_hasDerivAt_eq_slope
      G
      (fun t => β ^ 2 / 4 * overlapSecondMoment path t)
      (by norm_num : (0 : ℝ) < 1)
      hbridge.1
      hderiv

  have hG :
      G 1 = β ^ 2 / 4 * overlapSecondMoment path s := by
    simpa [hbridge.2] using hslope.symm

  have hsIcc : s ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨le_of_lt hs.1, le_of_lt hs.2⟩

  have hover_nonneg : 0 ≤ overlapSecondMoment path s := by
    rw [← A_eq_overlapSecondMoment path s]
    exact A_nonneg path

  have hNreal : (0 : ℝ) < (N : ℝ) := by
    exact_mod_cast hN

  have hNA :
      (N : ℝ) * A path s ≤ M₀ :=
    hsecond hN hp rfl hsIcc path

  have hover_le :
      overlapSecondMoment path s ≤ M₀ / (N : ℝ) := by
    rw [← A_eq_overlapSecondMoment path s]
    apply (le_div_iff₀ hNreal).2
    simpa [mul_comm] using hNA

  have hβpos : 0 < β := data.β_pos (β, h) hp
  have hβle : β ≤ data.βmax := data.β_bound (β, h) hp
  have hβsq : β ^ 2 ≤ data.βmax ^ 2 := by
    nlinarith [data.βmax_pos]

  have hcoef :
      β ^ 2 / 4 ≤ data.βmax ^ 2 / 4 := by
    nlinarith

  have hproduct :
      β ^ 2 / 4 * overlapSecondMoment path s ≤
        (data.βmax ^ 2 / 4) * (M₀ / (N : ℝ)) := by
    exact mul_le_mul hcoef hover_le hover_nonneg (by positivity)

  have hlower : 0 ≤ G 1 := by
    rw [hG]
    exact mul_nonneg (by positivity) hover_nonneg

  have hupper : G 1 ≤ M / (N : ℝ) := by
    rw [hG]
    calc
      β ^ 2 / 4 * overlapSecondMoment path s
          ≤ (data.βmax ^ 2 / 4) * (M₀ / (N : ℝ)) := hproduct
      _ = M / (N : ℝ) := by
        dsimp [M]
        ring

  simpa [G, rsFreeEnergy, skFreeEnergy] using And.intro hlower hupper

/-- Uniform free-energy error from equation `freeenergyidentity` in the
blueprint. -/
theorem rs_freeEnergy_error {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data C) :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      0 ≤ rsFreeEnergy β h - skFreeEnergy path ∧
      rsFreeEnergy β h - skFreeEnergy path ≤ M / N := by
  apply rs_freeEnergy_error_of_endpoint_bridge data C hCavity
  intro N hN β h q path
  exact freeEnergy_endpoint_bridge hN path

end SpinGlass.AT
