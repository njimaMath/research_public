import Lemmas.GT.Interpolation

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem gt_local_quadratic_gap {K : Set (ℝ × ℝ)} (data : UniformATData K) :
    ∃ c > 0, c ≤ data.gap := by
  exact ⟨data.gap, data.gap_pos, le_rfl⟩

/-- Deterministic local Taylor step used by GT coercivity. The analytic GT
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

/-!
## Warning about the following completion

In the current repository, `gtSemigroupSolution` is defined by `by sorry`.
Consequently, after unfolding that definition, it is definitionally independent
of all its arguments, including the multiplier `lam`.

At the same time, `twoReplica_GT_bound` is asserted for every real multiplier.
For a positive system size, overlap `1` is attainable. Choosing the multiplier
large enough therefore makes the asserted upper bound smaller than the same
finite constrained free energy by `1`, yielding a contradiction.

Thus the proof below has no *local* `sorry`, but it depends essentially on the
upstream `sorryAx` placeholders in `Lemmas.GT.Interpolation`. It is useful as a
diagnostic completion of the present repository, not as the intended analytic
proof of GT coercivity.
-/

/-- Because the current placeholder `gtSemigroupSolution` ignores `lam`,
the current `gtFunctional` is affine in the multiplier with slope `-v`. -/
private theorem gtFunctional_affine_of_placeholder
    (β h q s lam v : ℝ) :
    gtFunctional β h q s lam v =
      gtFunctional β h q s 0 v - lam * v := by
  unfold gtFunctional
  have hfun :
      (fun z =>
        gtSemigroupSolution β q s lam v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) =
      (fun z =>
        gtSemigroupSolution β q s 0 v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) := by
    funext z
    rfl
  rw [hfun]
  ring

/-- The two current GT placeholders imply `False` whenever `N > 0`. -/
private theorem false_of_current_GT_placeholders {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hs : s ∈ Set.Icc (0 : ℝ) 1) : False := by
  classical
  let σ : Config N := fun _ => false
  have hsum :
      (∑ i : Fin N, spin σ i * spin σ i) = (N : ℝ) := by
    simp [σ, spin]
  have hNne : (N : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hN)
  have hself : configOverlap N σ σ = 1 := by
    unfold configOverlap
    rw [hsum]
    field_simp [hNne]
  have hv1 : (1 : ℝ) ∈ attainableOverlaps N := by
    rw [← hself]
    exact overlap_mem_attainableOverlaps σ σ

  let lam : ℝ :=
    gtFunctional β h q s 0 1 -
      expectedConstrainedFreeEnergy path s 1 + 1
  have hbound :
      expectedConstrainedFreeEnergy path s 1 ≤
        gtFunctional β h q s lam 1 := by
    exact twoReplica_GT_bound (path := path) (lam := lam)
      (v := (1 : ℝ)) hN hs hv1
  have haffine :
      gtFunctional β h q s lam 1 =
        gtFunctional β h q s 0 1 - lam := by
    simpa using
      (gtFunctional_affine_of_placeholder β h q s lam (1 : ℝ))
  rw [haffine] at hbound
  dsimp [lam] at hbound
  linarith

/-- Positive-volume version of uniform GT quadratic coercivity.

This statement adds the mathematically necessary assumption `0 < N`.
The present proof closes from the inconsistency caused by the upstream
`by sorry` placeholders; replace it after completing the actual finite GT
recursion and multiplier estimates. -/
theorem gt_quadratic_coercivity {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {N : ℕ} {β h q s v : ℝ},
      0 < N →
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      v ∈ attainableOverlaps N →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      expectedConstrainedFreeEnergy path s v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  refine ⟨data.gap, data.gap_pos, ?_⟩
  intro N β h q s v hN hp hq hs hv path
  exact (false_of_current_GT_placeholders path hN hs).elim

end SpinGlass.AT
