import Lemmas.FixedDeviation
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s)
    (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2)

noncomputable def B {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    centeredOverlap q σs 0 1 * centeredOverlap q σs 0 2)

noncomputable def C {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    centeredOverlap q σs 0 1 * centeredOverlap q σs 2 3)

noncomputable def cavityVector {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : Fin 3 → ℝ :=
  ![A path s, B path s, C path s]

def theta (q r : ℝ) : Fin 3 → ℝ := ![1 - q ^ 2, q - q ^ 2, r - q ^ 2]

def cavityMatrix (β q r : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let b₂ := β ^ 2 * (1 - q ^ 2)
  let b₁ := β ^ 2 * (q - q ^ 2)
  let b₀ := β ^ 2 * (r - q ^ 2)
  !![b₂, -4 * b₁, 3 * b₀;
     b₁, b₂ - 2 * b₁ - 3 * b₀, 6 * b₀ - 3 * b₁;
     b₀, 4 * b₁ - 8 * b₀, b₂ - 8 * b₁ + 10 * b₀]

end SpinGlass.AT
