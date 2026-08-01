import Lemmas.Cavity.Coefficients

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def cavityRemainder {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : Fin 3 → ℝ := by
  exact cavityVector path s -
    s • (cavityMatrix β q (rsR β h)).mulVec (cavityVector path s) -
      (1 / (N : ℝ)) • theta q (rsR β h)

theorem cavity_system {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    cavityVector path s - s • (cavityMatrix β q (rsR β h)).mulVec (cavityVector path s) =
      (1 / (N : ℝ)) • theta q (rsR β h) + cavityRemainder path s := by
  unfold cavityRemainder
  abel

/-- The compact-set cavity estimate used in the paper.

The assumptions `data.β_pos`, `data.h_pos`, and `data.strictAT` supply the
positive-parameter and strict-AT restrictions.  The estimate is asserted only
at the RS fixed point and for positive system size. -/
def HasCavityRemainderBound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C : ℝ) : Prop :=
  0 < C ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
    (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
    ∀ path : RSSmartPathDisorder Ω N β h q,
      ‖cavityRemainder path s‖ ≤
        C * ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s)

theorem cavityRemainder_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    {data : UniformATData K} {C : ℝ}
    (hbound : HasCavityRemainderBound (Ω := Ω) data C)
    {N : ℕ} (hN : 0 < N) {β h q s : ℝ}
    (hp : (β, h) ∈ K) (hq : q = rsQ β h)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (path : RSSmartPathDisorder Ω N β h q) :
    ‖cavityRemainder path s‖ ≤
      C * ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s) := by
  exact hbound.2 hN hp hq hs path

end SpinGlass.AT
