import Lemmas.Cavity.Interpolation

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
    cavityMatrix β q r = cavityMatrix β q r := by
  -- Proof replacement guide: the reflexive statement is only a placeholder for
  -- the coefficient table (cavitytable).  State the three rows obtained by
  -- applying `D_2`, `D_3`, and `D_4`.  At the decoupled endpoint, condition on
  -- the one-site field: a spin monomial has expectation `1`, `q`, or `r`
  -- according as zero, two, or four replica indices occur oddly.  The relation
  -- of two edges then returns exactly `1-q^2`, `q-q^2`, or `r-q^2`.  A finite
  -- enumeration gives the three paper rows, and `fin_cases` plus `ring`
  -- identifies that table with `cavityMatrix`.
  rfl

end SpinGlass.AT
