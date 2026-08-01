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

theorem cavity_thirdMoment_gronwall {Ω : Type u} [MeasureSpace Ω]
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

theorem cavity_secondDerivative_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    |deriv (fun t => A path t) s| ≤ 1 + thirdMoment path s := by
  -- Statement/model repair required.  The paper's second-derivative bound
  -- (cavitysecondderivativebound) concerns the last-spin interpolation
  -- `u ↦ nu_{s,u}(F)`, not the smart-path derivative `s ↦ A_s`, and its
  -- constant depends on the compact parameter set.  Define the cavity
  -- Hamiltonian and operator `D_n` from equation (cavityderivative).  Applying
  -- `D_n` twice yields finitely many bounded spin factors times three centered
  -- cavity overlaps; Hölder, replica symmetry, and the cavity third-moment
  -- Gronwall comparison give
  -- `|d²/du² nu_{s,u}(F)| ≤ C_K*(thirdMoment path s + N⁻³)`.
  -- The current bound, uniform in arbitrary `β,q,s` and about the wrong
  -- derivative, is not supported by the paper.
  sorry

end SpinGlass.AT
