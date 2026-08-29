import SpinGlass.Core.Defs
import SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert

open MeasureTheory ProbabilityTheory Real BigOperators Filter Topology
open PhysLean.Probability.GaussianIBP

namespace SpinGlass

universe u

/-!
# Mixed p-spin disorder structures (finite `N`)

This file defines the *random* Hamiltonians used in the convex mixed p-spin model and in the
simple reference model used for Guerra's interpolation, in a way compatible with the
Hilbert–space Gaussian IBP machinery.  The legacy name `SKDisorder` is retained for the
internal disorder interface.

We keep the disorder abstract: a disorder is a centered Gaussian random vector in
`EnergySpace N` together with a specification of its covariance kernel on the
canonical basis `std_basis`.

## References
* M. Talagrand, *Mean Field Models for Spin Glasses*, Vol. I.
* D. Panchenko, *The Sherrington–Kirkpatrick Model*.
-/

variable {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

variable (N : ℕ)

/-! ### Deterministic thermodynamic quantities (aliases) -/

/-- Partition function \(Z_N(H)\). -/
noncomputable def partition_function (H : EnergySpace N) : ℝ := Z N H

-- NOTE: the free energy density is defined in `SpinGlasses/Defs.lean` as
-- `SpinGlass.free_energy_density`.

/-- Gibbs average \(\langle f \rangle_H\) under the Gibbs weights `gibbs_pmf`. -/
noncomputable def gibbs_average (H : EnergySpace N) (f : Config N → ℝ) : ℝ :=
  ∑ σ, gibbs_pmf N H σ * f σ

/-! ### Gaussian disorder specifications -/

/-- The mixed p-spin covariance kernel `N ξ(R(σ,τ))`. -/
noncomputable def mixedCovKernel (ξ : ℝ → ℝ) (σ τ : Config N) : ℝ :=
  N * ξ (overlap N σ τ)

/-- The linear reference covariance kernel `N d R(σ,τ)`. -/
noncomputable def referenceCovKernel (d : ℝ) (σ τ : Config N) : ℝ :=
  N * d * overlap N σ τ

/--
A centered mixed p-spin Hamiltonian.  The parameter `β` is retained in the type for the
smart-path interface, but now denotes the reference variance `ξ'(q)`, not an inverse
temperature.  The actual covariance function is the field `ξ`.
-/
structure SKDisorder (β h : ℝ) where
  /-- Covariance function of the mixed p-spin Hamiltonian. -/
  ξ : ℝ → ℝ
  /-- The (random) Hamiltonian. -/
  U : Ω → EnergySpace N
  /-- Centered Gaussian structure in the Hilbert space `EnergySpace N`. -/
  hU : IsGaussianHilbert.{u, 0, 0} U
  /-- Covariance on the canonical basis. -/
  cov_eq : ∀ σ τ, inner ℝ ((covOp (g := U) hU)
    (std_basis N σ)) (std_basis N τ) = mixedCovKernel N ξ σ τ

/--
Simple (reference) disorder with covariance `N β R(σ,τ)`.

This matches the “magnetic field” comparison model used in Guerra's bound.
-/
structure SimpleDisorder (β q : ℝ) where
  /-- The (random) Hamiltonian. -/
  V : Ω → EnergySpace N
  /-- Centered Gaussian structure in the Hilbert space `EnergySpace N`. -/
  hV : IsGaussianHilbert.{u, 0, 0} V
  /-- Covariance on the canonical basis. -/
  cov_eq : ∀ σ τ, inner ℝ ((covOp (g := V) hV) (std_basis N σ))
    (std_basis N τ) = referenceCovKernel N β σ τ

end SpinGlass
