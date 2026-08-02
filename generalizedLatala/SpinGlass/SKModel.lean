import SpinGlass.Defs
import SpinGlass.Mathlib.Probability.Distributions.Gaussian_IBP_Hilbert

open MeasureTheory ProbabilityTheory Real BigOperators Filter Topology
open PhysLean.Probability.GaussianIBP

namespace SpinGlass

universe u

/-!
# The Sherrington–Kirkpatrick (SK) model: disorder structures (finite `N`)

This file defines the *random* Hamiltonians used in the SK model and in the simple
reference model used for Guerra's interpolation, in a way compatible with the
Hilbert–space Gaussian IBP machinery.

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

/-- A centered finite-dimensional Gaussian disorder with a prescribed covariance kernel.

Keeping the kernel as an index lets the interpolation/IBP infrastructure be reused for
mixed `p`-spin covariances as well as for the quadratic SK kernel. -/
structure GaussianDisorder (β h : ℝ)
    (covarianceKernel : Config N → Config N → ℝ) where
  /-- The random Hamiltonian. -/
  U : Ω → EnergySpace N
  /-- Centered Gaussian structure in the Hilbert space `EnergySpace N`. -/
  hU : IsGaussianHilbert.{u, 0, 0} U
  /-- Covariance on the canonical basis. -/
  cov_eq : ∀ σ τ, inner ℝ ((covOp (g := U) hU)
    (std_basis N σ)) (std_basis N τ) = covarianceKernel σ τ

/--
SK disorder: a centered Gaussian Hamiltonian with covariance kernel `sk_cov_kernel`.

This corresponds (up to the usual normalizations) to the classical SK Hamiltonian
\(H_N(\sigma) = \frac{\beta}{\sqrt{N}}\sum_{i < j} g_{ij}\sigma_i\sigma_j\).
-/
abbrev SKDisorder (β h : ℝ) :=
  GaussianDisorder (Ω := Ω) N β h (sk_cov_kernel N β)

/-- A centered Gaussian reference disorder with a prescribed covariance kernel. -/
structure ReferenceDisorder (β q : ℝ)
    (covarianceKernel : Config N → Config N → ℝ) where
  /-- The (random) Hamiltonian. -/
  V : Ω → EnergySpace N
  /-- Centered Gaussian structure in the Hilbert space `EnergySpace N`. -/
  hV : IsGaussianHilbert.{u, 0, 0} V
  /-- Covariance on the canonical basis. -/
  cov_eq : ∀ σ τ, inner ℝ ((covOp (g := V) hV) (std_basis N σ))
    (std_basis N τ) = covarianceKernel σ τ

/--
Simple (reference) disorder for the SK smart path.

Its covariance is `N β² q R(σ,τ)`. -/
abbrev SimpleDisorder (β q : ℝ) :=
  ReferenceDisorder (Ω := Ω) N β q (simple_cov_kernel N β (fun x => q * x))

end SpinGlass
