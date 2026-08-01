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
    (hM : 0 < M) (hc : 0 < c) (_hlambda0 : 0 ≤ lambda0)
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

/-- The analytic input still missing from the explicit GT recursion.  Keeping
these estimates as a named interface lets the coercivity argument below be
checked independently of the Gaussian differentiation proofs that establish
them. -/
def HasGTFunctionalTaylorPackage {K : Set (ℝ × ℝ)}
    (data : UniformATData K) : Prop :=
  ∃ M > 0, ∀ {β h q s v : ℝ},
    (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
    v ∈ Set.Icc (-q : ℝ) 1 →
    ∃ d : ℝ,
      gtFunctional β h q s 0 v =
          2 * rsPathValue β h q s ∧
        |d| ≤ M ∧ data.gap * |v - q| ≤ |d| ∧
        ∀ lam, |lam| ≤ 1 →
          gtFunctional β h q s lam v ≤
            gtFunctional β h q s 0 v + d * lam + M / 2 * lam ^ 2

def HasGTFunctionalNegativeGap {K : Set (ℝ × ℝ)}
    (_data : UniformATData K) : Prop :=
  ∃ κ > 0, ∀ {β h q s v : ℝ},
    (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
    v ∈ Set.Icc (-1 : ℝ) (-q : ℝ) →
    ∃ lam,
      gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - κ

/-- Analytic estimates needed by the algebraic GT coercivity argument. -/
structure GTFunctionalAnalyticData {K : Set (ℝ × ℝ)}
    (data : UniformATData K) : Prop where
  taylor_package : HasGTFunctionalTaylorPackage data
  negative_gap : HasGTFunctionalNegativeGap data

/-- Analytic coercivity estimate on the explicit GT functional, assembled
from the multiplier Taylor package and the signed negative-range gap. -/
theorem gtFunctional_coercivity {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (analytic : GTFunctionalAnalyticData data) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      v ∈ Set.Icc (-1 : ℝ) 1 →
      ∃ lam, gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  obtain ⟨M, hM, hpackage⟩ := analytic.taylor_package
  obtain ⟨κ, hκ, hnegative⟩ := analytic.negative_gap

  let cLocal : ℝ := data.gap ^ 2 / (2 * M)
  let cNegative : ℝ := κ / 4
  let c : ℝ := min cLocal cNegative

  have hcLocal : 0 < cLocal := by
    dsimp [cLocal]
    exact div_pos (pow_pos data.gap_pos 2)
      (mul_pos (by norm_num) hM)
  have hcNegative : 0 < cNegative := by
    dsimp [cNegative]
    exact div_pos hκ (by norm_num)
  have hc : 0 < c := by
    exact lt_min hcLocal hcNegative

  refine ⟨c, hc, ?_⟩
  intro β h q s v hp hq hs hv

  have hqIcc : q ∈ Set.Icc (0 : ℝ) 1 := by
    rw [hq]
    exact rsQ_mem_Icc β h

  by_cases hvneg : v < -q
  · obtain ⟨lam, hlam⟩ :=
      hnegative hp hq hs ⟨hv.1, le_of_lt hvneg⟩
    refine ⟨lam, hlam.trans ?_⟩
    have hsq : (v - q) ^ 2 ≤ 4 := by
      nlinarith [hv.1, hv.2, hqIcc.1, hqIcc.2, sq_nonneg (v - q)]
    have hc_le : c ≤ κ / 4 := by
      dsimp [c]
      exact min_le_right _ _
    have hprod : c * (v - q) ^ 2 ≤ c * 4 :=
      mul_le_mul_of_nonneg_left hsq hc.le
    have hprodκ : c * (v - q) ^ 2 ≤ κ := by
      nlinarith [hprod]
    nlinarith
  · have hvmain : v ∈ Set.Icc (-q : ℝ) 1 :=
      ⟨le_of_not_gt hvneg, hv.2⟩
    obtain ⟨d, hzero, hdBound, hdGap, hTaylor⟩ :=
      hpackage hp hq hs hvmain

    let H : ℝ → ℝ := fun lam =>
      gtFunctional β h q s lam v - 2 * rsPathValue β h q s
    have hHzero : H 0 ≤ 0 := by
      dsimp [H]
      rw [hzero]
      linarith
    have hHTaylor : ∀ lam, |lam| ≤ 1 →
        H lam ≤ H 0 + d * lam + M / 2 * lam ^ 2 := by
      intro lam hlam
      dsimp [H]
      have h := hTaylor lam hlam
      linarith

    obtain ⟨lam, _hlam, hloss⟩ :=
      taylor_quadratic_loss H d M data.gap 1 (v - q)
        hM data.gap_pos (by norm_num) hHzero hHTaylor
        (by simpa using hdBound) hdGap
    refine ⟨lam, ?_⟩
    have hc_le : c ≤ data.gap ^ 2 / (2 * M) := by
      dsimp [c]
      exact min_le_left _ _
    have hprod : c * (v - q) ^ 2 ≤
        (data.gap ^ 2 / (2 * M)) * (v - q) ^ 2 :=
      mul_le_mul_of_nonneg_right hc_le (sq_nonneg (v - q))
    dsimp [H] at hloss
    nlinarith

/-- Every attainable overlap at positive volume lies in `[-1, 1]`. -/
private theorem attainableOverlap_mem_Icc {N : ℕ} (hN : 0 < N) {v : ℝ}
    (hv : v ∈ attainableOverlaps N) : v ∈ Set.Icc (-1 : ℝ) 1 := by
  rw [attainableOverlaps] at hv
  obtain ⟨p, _, rfl⟩ := Finset.mem_image.mp hv
  let σs : Replicas N 2 := fun i => if i = 0 then p.1 else p.2
  have hover := overlap_mem_Icc hN σs (0 : Fin 2) (1 : Fin 2)
  simpa [overlap, σs] using hover

/-- Uniform finite-volume quadratic coercivity, obtained by composing the GT
interpolation bound with the analytic coercivity estimate for its functional. -/
theorem gt_quadratic_coercivity {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (analytic : GTFunctionalAnalyticData data) :
    ∃ c > 0, ∀ {N : ℕ} {β h q s v : ℝ},
      0 < N →
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      v ∈ attainableOverlaps N →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      expectedConstrainedFreeEnergy path s v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  obtain ⟨c, hc, hfunctional⟩ := gtFunctional_coercivity data analytic
  refine ⟨c, hc, ?_⟩
  intro N β h q s v hN hp hq hs hv path
  obtain ⟨lam, hlam⟩ :=
    hfunctional hp hq hs (attainableOverlap_mem_Icc hN hv)
  exact (twoReplica_GT_bound (path := path) (lam := lam) hN hs hv).trans hlam

end SpinGlass.AT
