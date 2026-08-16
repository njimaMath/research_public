import Lemmas.Cavity.Defs

set_option autoImplicit false

namespace SpinGlass.AT

abbrev ReplicaEdge (n : ℕ) := {p : Fin n × Fin n // p.1 < p.2}

inductive EdgeRelation
  | equal
  | sharesOne
  | disjoint
  deriving DecidableEq

def edgeRelation {n : ℕ} (e f : ReplicaEdge n) : EdgeRelation :=
  if e = f then .equal
  else if e.1.1 = f.1.1 ∨ e.1.1 = f.1.2 ∨ e.1.2 = f.1.1 ∨ e.1.2 = f.1.2
    then .sharesOne else .disjoint

def decoupledSpinCoefficient (q r : ℝ) : EdgeRelation → ℝ
  | .equal => 1 - q ^ 2
  | .sharesOne => q - q ^ 2
  | .disjoint => r - q ^ 2

theorem cavity_coefficient_matrix_eq (β q r : ℝ) :
    cavityMatrix β q r = β ^ 2 •
      !![1 - q ^ 2, -4 * (q - q ^ 2), 3 * (r - q ^ 2);
         q - q ^ 2,
           (1 - q ^ 2) - 2 * (q - q ^ 2) - 3 * (r - q ^ 2),
           6 * (r - q ^ 2) - 3 * (q - q ^ 2);
         r - q ^ 2,
           4 * (q - q ^ 2) - 8 * (r - q ^ 2),
           (1 - q ^ 2) - 8 * (q - q ^ 2) + 10 * (r - q ^ 2)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cavityMatrix] <;> ring

end SpinGlass.AT
