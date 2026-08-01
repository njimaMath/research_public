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

theorem fixedDeviation {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (eps : ℝ) (heps : 0 < eps) :
    ∃ c C, 0 < c ∧ 0 < C ∧ ∀ {N : ℕ} {β h q s : ℝ}
      (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      quenchedTail path s eps ≤ C * Real.exp (-c * N) := by
  -- Paper route, Corollary (tail): for the corrected quadratic pressure set
  -- `Y = log <exp (lam0*N*Q12^2/2)>`.  The sublinear pressure bound and the
  -- Gronwall estimate give `E Y = o(N)` uniformly.  Differentiate `Y` in each
  -- Gaussian disorder coordinate; the squared gradient is `O_K(N)`, so
  -- Gaussian concentration makes
  -- `P(Y > lam0*eps^2*N/4)` exponentially small.  Off this event use the
  -- pointwise exponential Markov bound
  -- `1_{|Q12|≥eps} ≤ exp (-lam0*N*eps^2/2) * exp (lam0*N*Q12^2/2)`.
  -- Average over disorder and enlarge `C` for the finitely many small `N`.
  -- This proof depends on the corrected coupled-pressure definition and on an
  -- explicit finite Gaussian realization of `RSSmartPathDisorder` so that the
  -- Lipschitz/concentration theorem applies.
  -- BLOCKED: `RSSmartPathDisorder` supplies a Gaussian law but no coordinate
  -- realization to which the available Lipschitz concentration API applies.
  -- NEEDED: a finite Gaussian concentration theorem stated directly for this
  -- pushforward law, with the computed `O(N)` squared Lipschitz constant.
  -- BLUEPRINT: Corollary `tail`.
  sorry

end SpinGlass.AT
