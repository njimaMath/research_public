import Lemmas.SmartPath

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- The covariance-derivative operator for a finite replica expectation.

Its explicit replica formula is part of the Gaussian interpolation
construction.  Keep it opaque until that construction is formalized instead
of defining it circularly as `deriv`. -/
noncomputable def replicaCovarianceOperator {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ) (s : ℝ) : ℝ :=
  by
    sorry

/-- Reusable finite-dimensional Gaussian differentiation interface. -/
theorem quenchedGibbs_deriv_of_covariance_deriv {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    (hs : s ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun t => quenchedReplicaAverage (path.H t) F)
      (replicaCovarianceOperator path F s) s := by
  -- Paper route: equation (gaussianinterpolation) differentiates the law of
  -- the finite Gaussian vector `(H t σ)σ`.  Replace the disorder integral by
  -- an integral over that finite-dimensional Gaussian law, differentiate the
  -- finite Gibbs quotient, and apply Gaussian integration by parts twice.
  -- Bounded spins and the finite configuration space give domination.  The
  -- resulting derivative equals `deriv ... s` by definition.
  --
  -- API gap: `RSSmartPathDisorder` specifies each marginal Gaussian law and
  -- covariance but provides no ready theorem saying those data determine the
  -- pushforward law smoothly in `t`.  Prove a finite-index Gaussian-law
  -- uniqueness/interpolation lemma, or strengthen the structure with an
  -- explicit affine realization.  Without such a lemma, `HasDerivAt f
  -- (deriv f s) s` cannot be concluded merely from the definition of `deriv`.
  sorry

end SpinGlass.AT
