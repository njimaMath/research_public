import Lemmas.Scalar.StrictATSign

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def attainableOverlaps (N : ℕ) : Finset ℝ :=
  Finset.univ.image (fun p : Config N × Config N => configOverlap N p.1 p.2)

noncomputable def constrainedPartition {N : ℕ} (H : EnergySpace N) (v : ℝ) : ℝ :=
  ∑ p : Config N × Config N,
    if configOverlap N p.1 p.2 = v then Real.exp (H p.1 + H p.2) else 0

noncomputable def expectedConstrainedFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s v : ℝ) : ℝ :=
  (1 / (N : ℝ)) * ∫ ω,
    Real.log (constrainedPartition (fullPathHamiltonian path s ω) v)
    ∂(volume : Measure Ω)

noncomputable def signedMatrixPath (v u : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  if u ≤ |v| then
    let ι : ℝ := if 0 ≤ v then 1 else -1
    !![u, ι * u; ι * u, u]
  else
    !![u, v; v, u]

noncomputable def gtMassParameter (q v u : ℝ) : ℝ :=
  if q ≤ u then if u < |v| then 1 / 2 else 1 else 0

/-- The correction term in the specialized two-replica GT functional. -/
noncomputable def gtCorrection (β q s : ℝ) : ℝ :=
  s * β ^ 2 / 2 * (1 - q ^ 2)

/-- Terminal condition
`log (1/4 * ∑_{ε₁,ε₂=±1} exp (ε₁ x₁ + ε₂ x₂ + λ ε₁ ε₂))`. -/
noncomputable def gtTerminal (lam x₁ x₂ : ℝ) : ℝ :=
  Real.log ((Real.exp (x₁ + x₂ + lam) +
    Real.exp (x₁ - x₂ - lam) +
    Real.exp (-x₁ + x₂ - lam) +
    Real.exp (-x₁ - x₂ + lam)) / 4)

theorem overlap_mem_attainableOverlaps {N : ℕ} (σ τ : Config N) :
    configOverlap N σ τ ∈ attainableOverlaps N := by
  simp [attainableOverlaps]

theorem signedMatrixPath_endpoints (v : ℝ) (hv : |v| ≤ 1) :
    signedMatrixPath v 0 = 0 ∧
      signedMatrixPath v 1 = !![1, v; v, 1] := by
  constructor
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [signedMatrixPath]
  · by_cases hu : (1 : ℝ) ≤ |v|
    · have habs : |v| = 1 := le_antisymm hv hu
      by_cases hv0 : 0 ≤ v
      · have hv1 : v = 1 := by simpa [abs_of_nonneg hv0] using habs
        subst v
        simp [signedMatrixPath]
      · have hv1 : v = -1 := by
          have hvle : v ≤ 0 := le_of_not_ge hv0
          simpa [abs_of_nonpos hvle] using congrArg Neg.neg habs
        subst v
        simp [signedMatrixPath]
    · simp [signedMatrixPath, hu]

end SpinGlass.AT

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

end SpinGlass.AT
