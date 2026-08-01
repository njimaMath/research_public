import Lemmas.GaussianDerivative

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def pathFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  (1 / (N : ℝ)) * ∫ ω, Real.log (partitionFunction (fullPathHamiltonian path s ω))
    ∂(volume : Measure Ω)

noncomputable def overlapSecondMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s)
    (fun σs : Replicas N 2 => centeredOverlap q σs 0 1 ^ 2)

noncomputable def rsPathValue (β h q s : ℝ) : ℝ :=
  Real.log 2 + standardGaussianExpectation
    (fun z => Real.log (Real.cosh (h + β * Real.sqrt q * z))) +
      s * β ^ 2 / 4 * (1 - q) ^ 2

theorem smartPath_freeEnergy_deriv {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hs : s ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun t => pathFreeEnergy path t)
      (β ^ 2 / 4 * ((1 - q) ^ 2 - overlapSecondMoment path s)) s := by
  -- Paper route: formalize equations (freeenergyderivative), including the
  -- finite-volume cancellation.  Differentiate the finite log partition,
  -- apply the Gaussian covariance-derivative formula to the interaction and
  -- random-field terms, and use
  -- `∑ i<j x_i*x_j = ((∑ i x_i)^2 - ∑ i x_i^2) / 2`.  Rewrite the replica
  -- sums as `overlapSecondMoment` and simplify to the displayed derivative.
  sorry

theorem rsGap_deriv {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hs : s ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun t => rsPathValue β h q t - pathFreeEnergy path t)
      (β ^ 2 / 4 * overlapSecondMoment path s) s := by
  -- Proof route: `rsPathValue` is affine in `t`, with derivative
  -- `β ^ 2 / 4 * (1 - q) ^ 2`.  Subtract
  -- `smartPath_freeEnergy_deriv path hs` using `HasDerivAt.sub`, then normalize
  -- the scalar expression with `ring`.  This is equation (DNprime) in the
  -- paper.  It becomes a short algebraic proof once the preceding derivative
  -- theorem and the centered/full-Hamiltonian repair are in place.
  have hrs : HasDerivAt (fun t => rsPathValue β h q t)
      (β ^ 2 / 4 * (1 - q) ^ 2) s := by
    have hlin : HasDerivAt
        (fun t : ℝ => t * β ^ 2 / 4 * (1 - q) ^ 2)
        (β ^ 2 / 4 * (1 - q) ^ 2) s := by
      simpa only [id_eq, one_mul] using
        (((hasDerivAt_id s).mul_const (β ^ 2)).div_const 4).mul_const
          ((1 - q) ^ 2)
    exact hlin.const_add
      (Real.log 2 + standardGaussianExpectation
        (fun z => Real.log (Real.cosh (h + β * Real.sqrt q * z))))
  have hcoef : β ^ 2 / 4 * overlapSecondMoment path s =
      β ^ 2 / 4 * (1 - q) ^ 2 -
        β ^ 2 / 4 * ((1 - q) ^ 2 - overlapSecondMoment path s) := by
    ring
  rw [hcoef]
  exact hrs.sub (smartPath_freeEnergy_deriv path hs)

end SpinGlass.AT
