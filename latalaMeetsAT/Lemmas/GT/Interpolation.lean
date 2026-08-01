import Lemmas.GT.Defs

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def Tzero (f : ℝ → ℝ) (x : ℝ) : ℝ := f x

noncomputable def Thalf (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * Real.log (standardGaussianExpectation (fun z => Real.exp (f (x + z) / 2)))

noncomputable def Tone (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.log (standardGaussianExpectation (fun z => Real.exp (f (x + z))))

theorem Tzero_continuous {f : ℝ → ℝ} (hf : Continuous f) : Continuous (Tzero f) := by
  simpa [Tzero]

/-- The sole half-mass identity required by the specialized GT recursion. -/
theorem poissonDirichlet_half_identity (x : ℝ) : Thalf (fun _ => x) 0 = x := by
  simp [Thalf, standardGaussianExpectation]
  ring

/-- A function of the two local fields in the specialized GT recursion. -/
abbrev GTTwoField := ℝ → ℝ → ℝ

/-- Standard deviation for a covariance increment
`s * β^2 * (upper - lower)` in one active Gaussian direction. -/
noncomputable def gtIncrementScale
    (β s lower upper : ℝ) : ℝ :=
  β * Real.sqrt s * Real.sqrt (upper - lower)

/-- Sign of the rank-one direction in the signed overlap path. -/
noncomputable def gtPathSign (v : ℝ) : ℝ :=
  if 0 ≤ v then 1 else -1

/-- One recursion step for a diagonal covariance increment. -/
noncomputable def gtDiagonalStep
    (m scale : ℝ) (F : GTTwoField) : GTTwoField :=
  fun x₁ x₂ =>
    if m = 0 then
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          F (x₁ + scale * z₁) (x₂ + scale * z₂)))
    else
      (1 / m) * Real.log (standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          Real.exp (m * F (x₁ + scale * z₁) (x₂ + scale * z₂)))))

/-- One recursion step for the rank-one covariance increment below `|v|`.
The second coordinate moves in direction `sign * z`. -/
noncomputable def gtRankOneStep
    (m scale sign : ℝ) (F : GTTwoField) : GTTwoField :=
  fun x₁ x₂ =>
    if m = 0 then
      standardGaussianExpectation (fun z =>
        F (x₁ + scale * z) (x₂ + sign * scale * z))
    else
      (1 / m) * Real.log (standardGaussianExpectation (fun z =>
        Real.exp (m * F
          (x₁ + scale * z) (x₂ + sign * scale * z))))

/-- The finite two-dimensional Parisi recursion associated to
`signedMatrixPath v`, `gtMassParameter q v`, and `gtTerminal lam`.

The definition splits at the two breakpoints `q` and `|v|`.  Each branch is
a composition of at most three explicit Gaussian recursion operators. -/
noncomputable def gtSemigroupSolution
    (β q s lam v u x₁ x₂ : ℝ) : ℝ :=
  let r : ℝ := |v|
  let sign : ℝ := gtPathSign v
  let terminal : GTTwoField := gtTerminal lam
  let upper : ℝ → GTTwoField := fun lower =>
    gtDiagonalStep 1 (gtIncrementScale β s lower 1) terminal
  if q ≤ r then
    let atR : GTTwoField := upper r
    let atQ : GTTwoField :=
      gtRankOneStep (1 / 2) (gtIncrementScale β s q r) sign atR
    if r ≤ u then
      upper u x₁ x₂
    else if q ≤ u then
      gtRankOneStep (1 / 2) (gtIncrementScale β s u r) sign atR x₁ x₂
    else
      gtRankOneStep 0 (gtIncrementScale β s u q) sign atQ x₁ x₂
  else
    let atQ : GTTwoField := upper q
    let atR : GTTwoField :=
      gtDiagonalStep 0 (gtIncrementScale β s r q) atQ
    if q ≤ u then
      upper u x₁ x₂
    else if r ≤ u then
      gtDiagonalStep 0 (gtIncrementScale β s u q) atQ x₁ x₂
    else
      gtRankOneStep 0 (gtIncrementScale β s u r) sign atR x₁ x₂

/-- The specialized Guerra--Talagrand functional from the paper. -/
noncomputable def gtFunctional (β h q s lam v : ℝ) : ℝ :=
  2 * Real.log 2 + standardGaussianExpectation (fun z =>
    gtSemigroupSolution β q s lam v 0
      (h + β * Real.sqrt ((1 - s) * q) * z)
      (h + β * Real.sqrt ((1 - s) * q) * z)) -
    lam * v - gtCorrection β q s

/-- Contract for the finite-cascade interpolation theorem still needed by the
analytic development.  It records the endpoint comparison and derivative
sign as a dependency instead of asserting an unconditional result. -/
class SpecializedGTInterpolation.{v} : Prop where
  bound :
    ∀ {Ω : Type v} [MeasureSpace Ω]
      [IsProbabilityMeasure (volume : Measure Ω)]
      {N : ℕ} {β h q s lam v : ℝ}
      (path : RSSmartPathDisorder Ω N β h q),
      0 < N →
      s ∈ Set.Icc (0 : ℝ) 1 →
      v ∈ attainableOverlaps N →
      expectedConstrainedFreeEnergy path s v ≤
        gtFunctional β h q s lam v

/-- The finite-volume Guerra--Talagrand bound obtained from the explicit
finite recursion and the interpolation contract. -/
theorem twoReplica_GT_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    [SpecializedGTInterpolation.{u}]
    {N : ℕ} {β h q s lam v : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ attainableOverlaps N) :
    expectedConstrainedFreeEnergy path s v ≤ gtFunctional β h q s lam v := by
  exact SpecializedGTInterpolation.bound path hN hs hv

end SpinGlass.AT
