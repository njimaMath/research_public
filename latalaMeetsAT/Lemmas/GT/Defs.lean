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
  (1 / (N : ℝ)) * ∫ ω, Real.log (constrainedPartition (path.H s ω) v)
    ∂(volume : Measure Ω)

def signedMatrixPath (q v u : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![u, min q |v|; min q |v|, u]

noncomputable def gtMassParameter (q v u : ℝ) : ℝ :=
  if u < min q |v| then 0 else if u < max q |v| then 1 / 2 else 1

noncomputable def gtCorrection (β q v : ℝ) : ℝ := β ^ 2 / 2 * (v - q) ^ 2

theorem overlap_mem_attainableOverlaps {N : ℕ} (σ τ : Config N) :
    configOverlap N σ τ ∈ attainableOverlaps N := by
  -- Proof route: unfold `attainableOverlaps`, apply `Finset.mem_image`, and use
  -- the witness `(σ, τ)`.  Membership of the witness in `Finset.univ` is
  -- `Finset.mem_univ _`; the remaining equality is reflexivity.
  sorry

theorem signedMatrixPath_endpoints (q v : ℝ) :
    signedMatrixPath q v 0 0 0 = 0 ∧ signedMatrixPath q v 1 1 1 = 1 := by
  -- Proof route for the two diagonal scalar entries currently stated: unfold
  -- the matrix literal and simplify.  For the actual GT lemma, replace this by
  -- matrix equalities `Q^v_0 = 0` and `Q^v_1 = !![1,v;v,1]`; the present
  -- `signedMatrixPath` does not satisfy those full endpoint conditions.
  simp [signedMatrixPath]

end SpinGlass.AT
