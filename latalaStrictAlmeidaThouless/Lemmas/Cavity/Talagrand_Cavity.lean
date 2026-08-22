import Lemmas.ATDefs
import Mathlib.Tactic
import Lemmas.weak_concentration

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ProbabilityTheory NNReal

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace SpinGlass.AT

universe u

/-!
# Cavity approximation

This file is written against `Lemmas.ATDefs`.

The fixed change of basis in `ATDefs` has row order `(V,U,D)`:

  V = 2 B - 3 C,
  U = A - 4 B + 3 C,
  D = A - 2 B + C.

The analytic theorem `cavityModeRemainder_bound_from_lastSpin` is exactly the
last-spin Gaussian interpolation argument from the appendix.  Everything after
that theorem is deterministic finite-dimensional algebra.
-/

/-! ## Deterministic mode algebra -/

def cavityModeMatrix (β q r : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![β ^ 2 * cavityKappa q r, β ^ 2 * cavityZeta q r, 0;
     0, β ^ 2 * cavityKappa q r, 0;
     0, 0, β ^ 2 * (1 - 2 * q + r)]

def cavityModeSource (q r : ℝ) : Fin 3 → ℝ :=
  ![cavityZeta q r, cavityKappa q r, 1 - 2 * q + r]

/-- The cavity matrix is triangular in the `(V,U,D)` basis. -/
theorem cavityChangeMatrix_mul_cavityMatrix (β q r : ℝ) :
    cavityChangeMatrix * cavityMatrix β q r =
      cavityModeMatrix β q r * cavityChangeMatrix := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [cavityChangeMatrix, cavityMatrix, cavityModeMatrix,
      cavityKappa, cavityZeta, Matrix.mul_apply, Fin.sum_univ_succ] <;>
    ring

/-- The source vector in the `(V,U,D)` basis is `(ζ,κ,1-2q+r)`. -/
theorem cavityChangeMatrix_mulVec_theta (q r : ℝ) :
    cavityChangeMatrix.mulVec (theta q r) = cavityModeSource q r := by
  ext i
  fin_cases i <;>
    simp [cavityChangeMatrix, theta, cavityModeSource, cavityKappa,
      cavityZeta, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;>
    ring

/-- `cavityChangeMatrix` sends `(A,B,C)` to `(V,U,D)`. -/
theorem cavityChangeMatrix_mulVec_cavityVector
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    cavityChangeMatrix.mulVec (cavityVector path s) =
      ![cavityV path s, cavityU path s, cavityD path s] := by
  ext i
  fin_cases i <;>
    simp [cavityChangeMatrix, cavityVector, cavityV, cavityU, cavityD,
      Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;>
    ring

/-- Explicit `(V,U,D)` form of the vector remainder. -/
set_option maxHeartbeats 2000000 in
theorem cavityChangeMatrix_mulVec_cavityRemainder
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    cavityChangeMatrix.mulVec (cavityRemainder path s) =
      ![
        cavityV path s -
          s * β ^ 2 *
            (cavityZeta q (rsR β h) * cavityU path s +
              cavityKappa q (rsR β h) * cavityV path s) -
          (1 / (N : ℝ)) * cavityZeta q (rsR β h),
        cavityU path s -
          s * β ^ 2 * cavityKappa q (rsR β h) * cavityU path s -
          (1 / (N : ℝ)) * cavityKappa q (rsR β h),
        cavityD path s -
          s * β ^ 2 * (1 - 2 * q + rsR β h) * cavityD path s -
          (1 / (N : ℝ)) * (1 - 2 * q + rsR β h)
      ] := by
  have hmul :
      cavityChangeMatrix.mulVec
          ((cavityMatrix β q (rsR β h)).mulVec (cavityVector path s)) =
        (cavityModeMatrix β q (rsR β h)).mulVec
          ![cavityV path s, cavityU path s, cavityD path s] := by
    calc
      cavityChangeMatrix.mulVec
          ((cavityMatrix β q (rsR β h)).mulVec (cavityVector path s)) =
          (cavityChangeMatrix * cavityMatrix β q (rsR β h)).mulVec
            (cavityVector path s) := by
              rw [Matrix.mulVec_mulVec]
      _ = (cavityModeMatrix β q (rsR β h) * cavityChangeMatrix).mulVec
            (cavityVector path s) := by
              rw [cavityChangeMatrix_mul_cavityMatrix]
      _ = (cavityModeMatrix β q (rsR β h)).mulVec
            (cavityChangeMatrix.mulVec (cavityVector path s)) := by
              rw [Matrix.mulVec_mulVec]
      _ = (cavityModeMatrix β q (rsR β h)).mulVec
            ![cavityV path s, cavityU path s, cavityD path s] := by
              rw [cavityChangeMatrix_mulVec_cavityVector]
  rw [cavityRemainder, Matrix.mulVec_sub, Matrix.mulVec_sub,
    Matrix.mulVec_smul, Matrix.mulVec_smul, hmul,
    cavityChangeMatrix_mulVec_cavityVector,
    cavityChangeMatrix_mulVec_theta]
  ext i
  fin_cases i <;>
    simp [cavityModeMatrix, Matrix.mulVec, dotProduct,
      Fin.sum_univ_succ, smul_eq_mul] <;>
    ring_nf

/-- At the RS fixed point the replicon coefficient is the AT parameter. -/
theorem beta_sq_mul_repliconCoefficient_eq_atParameter
    {β h q : ℝ} (hq : q = rsQ β h) :
    β ^ 2 * (1 - 2 * q + rsR β h) = atParameter β h := by
  subst q
  exact (atParameter_eq_beta_sq_mul_one_sub_two_q_add_r β h).symm

/-! ## Inverting the fixed mode change -/

/-- The matrix displayed in `ATDefs` really is the inverse mode change. -/
theorem cavityChangeMatrixInv_mul_cavityChangeMatrix :
    cavityChangeMatrixInv * cavityChangeMatrix =
      (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [cavityChangeMatrixInv, cavityChangeMatrix, Matrix.mul_apply,
      Fin.sum_univ_succ]

/-- Explicit action of the inverse mode change. -/
theorem cavityChangeMatrixInv_mulVec_eq (x : Fin 3 → ℝ) :
    cavityChangeMatrixInv.mulVec x =
      ![-x 0 - 2 * x 1 + 3 * x 2,
        -x 0 - (3 / 2 : ℝ) * x 1 + (3 / 2 : ℝ) * x 2,
        -x 0 - x 1 + x 2] := by
  ext i
  fin_cases i <;>
    simp [cavityChangeMatrixInv, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;>
    ring

/-- The inverse change of basis costs at most a factor `6` in the sup norm. -/
theorem cavityChangeMatrixInv_mulVec_norm_le (x : Fin 3 → ℝ) :
    ‖cavityChangeMatrixInv.mulVec x‖ ≤ 6 * ‖x‖ := by
  sorry

/-- Recover the original remainder from the three scalar mode remainders. -/
theorem cavityRemainder_eq_inverseModeRemainder
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    cavityRemainder path s =
      cavityChangeMatrixInv.mulVec
        (cavityChangeMatrix.mulVec (cavityRemainder path s)) := by
  sorry

/-! ## The analytic last-spin estimate -/

/--
Mode version of the cavity estimate.  By
`cavityChangeMatrix_mulVec_cavityRemainder`, this is exactly the three scalar
relations

  V_s = s β² (ζ U_s + κ V_s) + ζ/N + O(η),
  U_s = s β² κ U_s + κ/N + O(η),
  D_s = s β² (1-2q+r) D_s + (1-2q+r)/N + O(η).

At `q = rsQ β h`, the third coefficient satisfies
`β² (1-2q+r) = atParameter β h`.
-/
def HasCavityModeRemainderBound
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C : ℝ) : Prop :=
  0 < C ∧
    ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
        ‖cavityChangeMatrix.mulVec (cavityRemainder path s)‖ ≤
          C * cavityErrorScale path s

/--
This is the one genuinely analytic block: the last-spin cavity interpolation.
Its proof is the appendix argument:

* write `Q = Q⁻ + ε₁ε₂/N`;
* construct `ν_{s,u}`;
* prove the Gaussian derivative identity;
* obtain the uniform cubic-moment bound by backward Gronwall;
* replace the endpoint quadratic products;
* apply the derivative identity twice to `G_A,G_B,G_C`;
* use the three-valued edge rule at `u = 0`;
* control the diagonal `E_A,E_B,E_C` terms.

The conclusion is exactly the norm form of the three scalar estimates above.
-/
theorem cavityModeRemainder_bound_from_lastSpin
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ C : ℝ, HasCavityModeRemainderBound (Ω := Ω) data C := by
  sorry

/-! ## Final cavity proposition -/

/--
The mode estimate implies the `ATDefs.HasCavityRemainderBound` estimate.
The loss of the harmless numerical factor `6` is only the norm of the fixed
inverse change of basis. assume this
-/
theorem exists_hasCavityRemainderBound
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ C : ℝ, HasCavityRemainderBound (Ω := Ω) data C := by
  sorry




end SpinGlass.AT
