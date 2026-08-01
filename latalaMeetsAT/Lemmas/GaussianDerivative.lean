import Lemmas.SmartPath

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Derivative in the interpolation parameter of the smart-path covariance. -/
noncomputable def smartPathCovDerivative (N : ℕ) (β q : ℝ)
    (σ τ : Config N) : ℝ :=
  (N : ℝ) * β ^ 2 / 2 * configOverlap N σ τ ^ 2 -
    (N : ℝ) * β ^ 2 * q * configOverlap N σ τ - β ^ 2 / 2

private def replicaFinCast {n : ℕ} (a : Fin n) : Fin (n + 2) :=
  ⟨a, by omega⟩

private def penultimateReplica (n : ℕ) : Fin (n + 2) := ⟨n, by omega⟩

private def lastReplica (n : ℕ) : Fin (n + 2) := ⟨n + 1, by omega⟩

/-- The covariance-derivative operator for a finite replica expectation.

This is the standard two-extra-replica Hessian contraction. The last two
replicas encode the derivatives of the Gibbs normalizing factors. -/
noncomputable def replicaCovarianceOperator {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ) (s : ℝ) : ℝ :=
  (1 / 2 : ℝ) * quenchedReplicaAverage (path.H s) (fun σs : Replicas N (n + 2) =>
    F (fun a => σs (replicaFinCast a)) *
      ((∑ a : Fin n, ∑ b : Fin n,
          smartPathCovDerivative N β q
            (σs (replicaFinCast a)) (σs (replicaFinCast b))) -
        2 * (n : ℝ) * ∑ a : Fin n,
          smartPathCovDerivative N β q
            (σs (replicaFinCast a)) (σs (penultimateReplica n)) +
        (n : ℝ) * (n + 1 : ℝ) *
          smartPathCovDerivative N β q
            (σs (penultimateReplica n)) (σs (lastReplica n)) -
        (n : ℝ) * smartPathCovDerivative N β q
          (σs (penultimateReplica n)) (σs (penultimateReplica n))))

/-- Finite-dimensional Gaussian covariance interpolation for normalized
finite Gibbs sums.  Its analytic proof is isolated here. -/
theorem quenchedGibbs_deriv_of_covariance_deriv {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (F : Replicas N n → ℝ)
    (hs : s ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun t => quenchedReplicaAverage (path.H t) F)
      (replicaCovarianceOperator path F s) s := by
  sorry

end SpinGlass.AT
