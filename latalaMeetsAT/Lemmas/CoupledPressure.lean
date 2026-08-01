import Lemmas.GT.Coercivity

open MeasureTheory ProbabilityTheory Filter

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def quadraticCoupledPartition {N : ℕ}
    (H : EnergySpace N) (q lam : ℝ) : ℝ :=
  ∑ p : Config N × Config N,
    Real.exp (H p.1 + H p.2 +
      lam * (N : ℝ) / 2 * (configOverlap N p.1 p.2 - q) ^ 2)

noncomputable def quadraticCoupledPressure {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s lam : ℝ) : ℝ :=
  (1 / (2 * (N : ℝ))) * ∫ ω,
    Real.log (quadraticCoupledPartition (fullPathHamiltonian path s ω) q lam)
      ∂(volume : Measure Ω)

noncomputable def normalizedCouplingExcess {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s lam : ℝ) : ℝ :=
  quadraticCoupledPressure path s lam - pathFreeEnergy path s

noncomputable def rsFreeEnergyGap {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  rsPathValue β h q s - pathFreeEnergy path s

theorem coupledPressure_sublinear {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ epsN : ℕ → ℝ, Tendsto epsN atTop (nhds 0) ∧ ∀ {N : ℕ}
      {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      overlapSecondMoment path s ≤ epsN N := by
  -- Paper route, equations (pressureenvelope)--(preconcentration): choose
  -- `0 < lam0 < c_GT`.  Split the two-replica partition sum over the `N+1`
  -- attainable overlaps.  The GT coercivity estimate beats the quadratic
  -- coupling, while Gaussian concentration bounds the maximum of the centered
  -- constrained log partition functions by `O(sqrt (N*log (N+1)))`.  This
  -- gives `epsN = C*sqrt(log(N+1)/N) + C*log(N+1)/N`.  For
  -- `D_N(s) = rsPathValue - pathFreeEnergy`, use `rsGap_deriv`, convexity in
  -- the coupling, and Gronwall to obtain both `D_N(s) ≤ C*epsN N` and the
  -- displayed overlap bound.  Standard limit lemmas show `epsN -> 0`.
  sorry

theorem preliminary_overlap_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ epsN : ℕ → ℝ, Tendsto epsN atTop (nhds 0) ∧ ∀ {N : ℕ}
      {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      overlapSecondMoment path s ≤ epsN N := by
  -- Proof route: this is only the public name for `coupledPressure_sublinear`.
  exact coupledPressure_sublinear data

end SpinGlass.AT
