import Lemmas.Cavity.Defs

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def thirdMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s)
    (fun σs : Replicas N 4 => |centeredOverlap q σs 0 1| ^ 3)

theorem thirdMoment_le_two_A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hN : 0 < N)
    (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    thirdMoment path s ≤ 2 * A path s := by
  unfold thirdMoment A
  rw [← quenchedReplicaAverage_const_mul]
  apply quenchedReplicaAverage_mono (H := fullPathHamiltonian path s) (by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const)
  intro σs
  have habs := abs_centeredOverlap_le_two hN hq σs (0 : Fin 4) (1 : Fin 4)
  have hsq := sq_nonneg (centeredOverlap q σs (0 : Fin 4) (1 : Fin 4))
  rw [show |centeredOverlap q σs (0 : Fin 4) (1 : Fin 4)| ^ 3 =
      |centeredOverlap q σs (0 : Fin 4) (1 : Fin 4)| *
        centeredOverlap q σs (0 : Fin 4) (1 : Fin 4) ^ 2 by
    rw [pow_succ, sq_abs]
    ring]
  nlinarith

/-- Compatibility name.  The actual Gronwall comparison belongs to the
last-spin interpolation and is part of the blocked cavity expansion below. -/
theorem cavity_thirdMoment_gronwall {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hN : 0 < N)
    (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    thirdMoment path s ≤ 2 * A path s :=
  thirdMoment_le_two_A path hN hq

/-- Data supplied by an explicit last-spin Gaussian decomposition.

The covariance-only smart path does not determine coupled last-spin
coordinates across interpolation parameters.  This class records the average
and its analytic estimate without adding a logical axiom. -/
class CavityInterpolationData {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    where
  average : ℝ → ℝ → ℝ
  secondDerivative_bound : ∀ {K : Set (ℝ × ℝ)}
      (data : UniformATData K) (C_K : ℝ), 0 < C_K →
      0 < N → ∀ {s u : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      s ∈ Set.Icc (0 : ℝ) 1 → u ∈ Set.Icc (0 : ℝ) 1 →
      (∀ σs, |F σs| ≤ 1) →
      |deriv (deriv (fun v => average s v)) u| ≤
        C_K * (thirdMoment path s + (N : ℝ) ^ (-3 : ℝ))

/-- The last-spin cavity interpolation average `ν_{s,u}(F)` from the paper. -/
noncomputable def cavityInterpolatedAverage {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    [CavityInterpolationData path F] (s u : ℝ) : ℝ :=
  CavityInterpolationData.average (path := path) (F := F) s u

/-- Compact strict-AT second-derivative estimate for the last-spin cavity
interpolation. -/
theorem cavity_secondDerivative_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C_K : ℝ) (hC_K : 0 < C_K)
    {N n : ℕ} (hN : 0 < N) {β h q s u : ℝ}
    (hp : (β, h) ∈ K) (hq : q = rsQ β h)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hu : u ∈ Set.Icc (0 : ℝ) 1)
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    [CavityInterpolationData path F]
    (hF : ∀ σs, |F σs| ≤ 1) :
    |deriv (deriv (fun v => cavityInterpolatedAverage path F s v)) u| ≤
      C_K * (thirdMoment path s + (N : ℝ) ^ (-3 : ℝ)) := by
  simpa [cavityInterpolatedAverage] using
    (CavityInterpolationData.secondDerivative_bound
      (path := path) (F := F) data C_K hC_K hN hp hq hs hu hF)

end SpinGlass.AT
