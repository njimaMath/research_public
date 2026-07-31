import Lemmas.Cavity.System

set_option autoImplicit false

namespace SpinGlass.AT

def stabilityOperator (β q r s : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  1 - s • cavityMatrix β q r

theorem cavityMatrix_determinant {β h s : ℝ} :
    Matrix.det (stabilityOperator β (rsQ β h) (rsR β h) s) =
      (1 - s * atParameter β h) *
        (1 - s * β ^ 2 * (1 - 4 * rsQ β h + 3 * rsR β h)) ^ 2 := by
  classical
  simp [stabilityOperator, cavityMatrix, atParameter, rsA,
    Matrix.det_fin_three]
  ring

theorem one_sub_anomalous_lower_bound {K : Set (ℝ × ℝ)}
    (data : UniformATData K) {β h s : ℝ}
    (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    data.gap ≤ 1 - s * atParameter β h := by
  -- Proof route: this is the uniform path gap already proved in `UniformData`.
  exact path_gap data hp hs

theorem cavityMatrix_inverse_uniform {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ M > 0, ∀ {β h s : ℝ}, (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ i j, |(stabilityOperator β (rsQ β h) (rsR β h) s)⁻¹ i j| ≤ M := by
  -- Paper route, equations (triangular)--(inverse): after correcting the
  -- anomalous-eigenvalue lemma, obtain `β^2*gamma' ≤ atParameter < 1` from
  -- `rsR_le_rsQ`.  Both diagonal denominators in the triangular inverse are
  -- therefore at least `data.gap`.  The Jordan entry is
  -- `s*mu/(1-s*β^2*gamma')^2`; bound `mu = β^2*(2*q+q^2-3*r)` uniformly using
  -- `data.βmax` and `q,r ∈ [0,1]`.  Multiply by the fixed matrices `S⁻¹` and
  -- `S`, take a maximum over their finitely many entries, and choose a positive
  -- `M`.  This gives entrywise bounds without relying on continuity of inverse.
  sorry

theorem replicon_leftEigenvector (β h s : ℝ) :
    let ell : Fin 3 → ℝ := ![1, -2, 1]
    Matrix.vecMul ell (stabilityOperator β (rsQ β h) (rsR β h) s) =
      (1 - s * atParameter β h) • ell := by
  dsimp [stabilityOperator, cavityMatrix, atParameter, rsA]
  funext j
  fin_cases j <;>
    simp [Matrix.vecMul_eq_sum, Fin.sum_univ_three, Matrix.one_apply] <;>
    ring

end SpinGlass.AT
