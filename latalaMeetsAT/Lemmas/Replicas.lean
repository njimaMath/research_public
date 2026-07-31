import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Fintype.Pi

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Spin configurations used by the strict-AT development. -/
abbrev Config (N : ℕ) := Fin N → Bool

/-- A finite-volume Hamiltonian. -/
abbrev EnergySpace (N : ℕ) := Config N → ℝ

def spin {N : ℕ} (σ : Config N) (i : Fin N) : ℝ := if σ i then 1 else -1

/-- The normalized overlap of two configurations. -/
noncomputable def configOverlap (N : ℕ) (σ τ : Config N) : ℝ :=
  (1 / (N : ℝ)) * ∑ i, spin σ i * spin τ i

noncomputable def partitionFunction {N : ℕ} (H : EnergySpace N) : ℝ :=
  ∑ σ, Real.exp (H σ)

noncomputable def gibbsWeight {N : ℕ} (H : EnergySpace N) (σ : Config N) : ℝ :=
  Real.exp (H σ) / partitionFunction H

/-- An explicitly indexed family of replicas. -/
abbrev Replicas (N n : ℕ) := Fin n → Config N

/-- Finite-product Gibbs expectation.  Keeping the product as a finite sum
makes later differentiation a finite-dimensional calculation. -/
noncomputable def replicaGibbsAverage {N n : ℕ}
    (H : EnergySpace N) (F : Replicas N n → ℝ) : ℝ :=
  ∑ σs, (∏ a, gibbsWeight H (σs a)) * F σs

/-- Disorder-averaged replica expectation. -/
noncomputable def quenchedReplicaAverage {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → EnergySpace N)
    (F : Replicas N n → ℝ) : ℝ :=
  ∫ ω, replicaGibbsAverage (H ω) F ∂(volume : Measure Ω)

/-- The overlap of two selected replicas. -/
noncomputable def overlap {N n : ℕ} (σs : Replicas N n) (a b : Fin n) : ℝ :=
  configOverlap N (σs a) (σs b)

/-- Overlap centered at the replica-symmetric parameter `q`. -/
noncomputable def centeredOverlap {N n : ℕ} (q : ℝ) (σs : Replicas N n)
    (a b : Fin n) : ℝ :=
  overlap σs a b - q

theorem overlap_mem_Icc {N n : ℕ} (hN : 0 < N) (σs : Replicas N n)
    (a b : Fin n) : overlap σs a b ∈ Set.Icc (-1 : ℝ) 1 := by
  -- Proof route: unfold `overlap`, `configOverlap`, and `spin`.  Every summand
  -- `spin (σs a) i * spin (σs b) i` is either `1` or `-1`, so bound the finite
  -- sum between `-N` and `N` with `Finset.sum_le_sum`.  Use `hN` to rewrite
  -- `1 / (N : ℝ)` as a nonnegative scalar and finish both bounds with `linarith`
  -- or `nlinarith`.  This formalizes the elementary fact recorded before the
  -- Gaussian-calculus subsection of the paper.
  sorry

theorem abs_centeredOverlap_le_two {N n : ℕ} (hN : 0 < N)
    {q : ℝ} (hq : q ∈ Set.Icc (0 : ℝ) 1) (σs : Replicas N n)
    (a b : Fin n) : |centeredOverlap q σs a b| ≤ 2 := by
  -- Proof route: obtain `-1 ≤ overlap σs a b` and `overlap σs a b ≤ 1` from
  -- `overlap_mem_Icc`.  Combine these with `0 ≤ q` and `q ≤ 1`, unfold
  -- `centeredOverlap`, prove `-2 ≤ overlap ... - q ≤ 2`, and use
  -- `abs_le.mpr`.  No probability theory is needed here.
  sorry

/-- The Cauchy--Schwarz estimate used repeatedly in the absorption argument. -/
theorem mixed_overlap_abs_le_secondMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (H : Ω → EnergySpace N) (q : ℝ)
    (a b c d : Fin 4)
    (hsame : quenchedReplicaAverage H
        (fun σs => centeredOverlap q σs a b ^ 2) =
      quenchedReplicaAverage H
        (fun σs => centeredOverlap q σs c d ^ 2)) :
    |quenchedReplicaAverage H (fun σs =>
      centeredOverlap q σs a b * centeredOverlap q σs c d)| ≤
      quenchedReplicaAverage H (fun σs => centeredOverlap q σs a b ^ 2) := by
  -- Proof route: first package `replicaGibbsAverage H` as expectation for the
  -- finite probability mass function `∏ a, gibbsWeight H (σs a)`.  Positivity
  -- of `exp` shows `partitionFunction H > 0`, and the product weights sum to
  -- one.  Apply weighted finite-sum Cauchy--Schwarz pointwise in the disorder,
  -- then Cauchy--Schwarz once more to the disorder integral.  Replica
  -- relabeling, expressed by the supplied equality `hsame`, makes the two
  -- second moments equal and turns the square-root product into the claimed
  -- right side.  This is the estimate stated immediately before the paper's
  -- Gaussian-calculus subsection.  A reusable finite Gibbs probability lemma
  -- will make this proof and the later cavity estimates short.
  sorry

end SpinGlass.AT
