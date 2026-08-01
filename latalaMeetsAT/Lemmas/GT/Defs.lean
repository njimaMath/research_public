import Lemmas.Scalar.StrictATSign

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def attainableOverlaps (N : ℕ) : Finset ℝ :=
  Finset.univ.image (fun p : Config N × Config N => configOverlap N p.1 p.2)

noncomputable def constrainedPartition {N : ℕ} (H : EnergySpace N) (v : ℝ) : ℝ :=
  ∑ p : Config N × Config N,
    if configOverlap N p.1 p.2 = v then Real.exp (H p.1 + H p.2) else 0

noncomputable def expectedConstrainedFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s v : ℝ) : ℝ :=
  (1 / (N : ℝ)) * ∫ ω,
    Real.log (constrainedPartition (fullPathHamiltonian path s ω) v)
    ∂(volume : Measure Ω)

noncomputable def signedMatrixPath (v u : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  if u ≤ |v| then
    let ι : ℝ := if 0 ≤ v then 1 else -1
    !![u, ι * u; ι * u, u]
  else
    !![u, v; v, u]

noncomputable def gtMassParameter (q v u : ℝ) : ℝ :=
  if q ≤ u then if u < |v| then 1 / 2 else 1 else 0

/-- The correction term in the specialized two-replica GT functional. -/
noncomputable def gtCorrection (β q s : ℝ) : ℝ :=
  s * β ^ 2 / 2 * (1 - q ^ 2)

/-- Terminal condition
`log (1/4 * ∑_{ε₁,ε₂=±1} exp (ε₁ x₁ + ε₂ x₂ + λ ε₁ ε₂))`. -/
noncomputable def gtTerminal (lam x₁ x₂ : ℝ) : ℝ :=
  Real.log ((Real.exp (x₁ + x₂ + lam) +
    Real.exp (x₁ - x₂ - lam) +
    Real.exp (-x₁ + x₂ - lam) +
    Real.exp (-x₁ - x₂ + lam)) / 4)

theorem overlap_mem_attainableOverlaps {N : ℕ} (σ τ : Config N) :
    configOverlap N σ τ ∈ attainableOverlaps N := by
  simp [attainableOverlaps]

theorem signedMatrixPath_endpoints (v : ℝ) (hv : |v| ≤ 1) :
    signedMatrixPath v 0 = 0 ∧
      signedMatrixPath v 1 = !![1, v; v, 1] := by
  constructor
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [signedMatrixPath]
  · by_cases hu : (1 : ℝ) ≤ |v|
    · have habs : |v| = 1 := le_antisymm hv hu
      by_cases hv0 : 0 ≤ v
      · have hv1 : v = 1 := by simpa [abs_of_nonneg hv0] using habs
        subst v
        simp [signedMatrixPath]
      · have hv1 : v = -1 := by
          have hvle : v ≤ 0 := le_of_not_ge hv0
          simpa [abs_of_nonpos hvle] using congrArg Neg.neg habs
        subst v
        simp [signedMatrixPath]
    · simp [signedMatrixPath, hu]

end SpinGlass.AT
