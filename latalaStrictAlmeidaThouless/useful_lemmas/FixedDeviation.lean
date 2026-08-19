import Lemmas.GT.Coercivity

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def quenchedTail {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s eps : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0)

end SpinGlass.AT
