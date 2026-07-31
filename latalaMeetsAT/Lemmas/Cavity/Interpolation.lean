import Lemmas.Cavity.Defs

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def thirdMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (path.H s)
    (fun σs : Replicas N 2 => |centeredOverlap q σs 0 1| ^ 3)

theorem cavity_thirdMoment_gronwall {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    thirdMoment path s ≤ 2 * A path s := by
  -- Proof route: for full overlaps this is the pointwise bound used in the paper:
  -- `|Q12| ≤ 2` implies `|Q12|^3 ≤ 2*Q12^2`; monotonicity of the finite Gibbs
  -- sum and disorder integral then gives the result.  Use
  -- `abs_centeredOverlap_le_two` followed by `nlinarith [sq_nonneg Q12]`.
  --
  -- The present statement is missing `0 < N` and `q ∈ [0,1]`; without them
  -- `|centeredOverlap q| ≤ 2` is unavailable and the claim is false for
  -- arbitrary `q`.  Add these hypotheses, or specialize to
  -- `q = rsQ β h` with positive `β,h`.  The Gronwall comparison in equations
  -- (cavityM3derivative)--(cavityM3endpoint) is a different lemma for the
  -- last-spin interpolation and should be formalized separately.
  sorry

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
