import Lemmas.GT.Interpolation

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem gt_local_quadratic_gap {K : Set (ℝ × ℝ)} (data : UniformATData K) :
    ∃ c > 0, c ≤ data.gap := by
  -- Proof route: take `c = data.gap`; positivity is a structure field.
  exact ⟨data.gap, data.gap_pos, le_rfl⟩

/-- Deterministic local Taylor step used by GT coercivity.  The analytic GT
development only has to supply the displayed Taylor majorant and derivative
gap. -/
theorem taylor_quadratic_loss (H : ℝ → ℝ) (d M c lambda0 delta : ℝ)
    (hM : 0 < M) (hc : 0 < c) (hlambda0 : 0 ≤ lambda0)
    (hzero : H 0 ≤ 0)
    (htaylor : ∀ lam, |lam| ≤ lambda0 →
      H lam ≤ H 0 + d * lam + M / 2 * lam ^ 2)
    (hd_upper : |d| ≤ M * lambda0)
    (hd_lower : c * |delta| ≤ |d|) :
    ∃ lam, |lam| ≤ lambda0 ∧
      H lam ≤ -(c ^ 2 / (2 * M)) * delta ^ 2 := by
  let lam := -d / M
  have hlam : |lam| ≤ lambda0 := by
    dsimp [lam]
    rw [abs_div, abs_neg, abs_of_pos hM]
    exact (div_le_iff₀ hM).2 (by simpa [mul_comm] using hd_upper)
  have ht := htaylor lam hlam
  have hlocal : H lam ≤ -(d ^ 2) / (2 * M) := by
    calc
      H lam ≤ H 0 + d * lam + M / 2 * lam ^ 2 := ht
      _ ≤ 0 + d * lam + M / 2 * lam ^ 2 := by gcongr
      _ = -(d ^ 2) / (2 * M) := by
        dsimp [lam]
        field_simp [ne_of_gt hM]
        ring
  have hsq : c ^ 2 * delta ^ 2 ≤ d ^ 2 := by
    have hmul := mul_self_le_mul_self
      (mul_nonneg hc.le (abs_nonneg delta)) hd_lower
    calc
      c ^ 2 * delta ^ 2 = (c * |delta|) * (c * |delta|) := by
        nlinarith [sq_abs delta]
      _ ≤ |d| * |d| := hmul
      _ = d ^ 2 := by nlinarith [sq_abs d]
  refine ⟨lam, hlam, hlocal.trans ?_⟩
  have hden : 0 < 2 * M := mul_pos (by norm_num) hM
  calc
    -(d ^ 2) / (2 * M) ≤ -(c ^ 2 * delta ^ 2) / (2 * M) := by
      exact (div_le_div_iff_of_pos_right hden).2 (by linarith)
    _ = -(c ^ 2 / (2 * M)) * delta ^ 2 := by ring

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
  -- BLOCKED: coercivity needs the unfinished finite GT recursion, its arbitrary
  -- multiplier bound, and the unfinished strict signed AT gap.
  -- NEEDED: zero-source flatness, the multiplier derivative, a uniform second
  -- derivative bound, and the signed far-gap cases.
  -- BLUEPRINT: Lemma `Taylorcoercivity` and Proposition `GTcoercivity`.
  sorry

end SpinGlass.AT
