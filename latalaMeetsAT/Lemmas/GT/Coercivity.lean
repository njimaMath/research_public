import Lemmas.GT.Interpolation

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem gt_local_quadratic_gap {K : Set (ℝ × ℝ)} (data : UniformATData K) :
    ∃ c > 0, c ≤ data.gap := by
  -- Proof route: take `c = data.gap`; positivity is a structure field.
  exact ⟨data.gap, data.gap_pos, le_rfl⟩

theorem gt_quadratic_coercivity {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {N : ℕ} {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      v ∈ attainableOverlaps N →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      expectedConstrainedFreeEnergy path s v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  -- Paper route: apply the exact GT functional bound, identify its multiplier
  -- derivative at zero with `g_s(v) - v`, and use `strictAT_sign`.  Near `q`,
  -- the uniform second-multiplier-derivative bound and Taylor's theorem with
  -- the opposite-sign multiplier give equation (localGTgap).  On compact sets
  -- away from `q`, continuity and the strict sign give a fixed gap.  The signed
  -- ranges `[-q,q)` and `v < -q` require equations (signedslope) and
  -- (GTuniformnegativegap).  Since `|v-q| ≤ 2`, one uniform fixed gap away from
  -- `q` can be weakened to the claimed quadratic loss.
  sorry

end SpinGlass.AT
