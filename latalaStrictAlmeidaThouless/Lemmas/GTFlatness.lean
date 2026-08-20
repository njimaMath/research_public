import Lemmas.Psi_continuity
import Lemmas.ATDefs
import Lemmas.GTGauss
import Lemmas.interpolatedAT

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-!
## Branchwise formulas for the GT functional

These formulas mirror the four overlap regimes established in `GTGauss`.
-/

/-! ### Case `|v| = 0` -/

lemma flatness_gtFunctional_formula_abs_v_eq_zero
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hv : |v| = 0) :
    gtFunctional β h q s lam v
      =
    2 * Real.log 2
      + standardGaussianExpectation (fun z =>
        gtDiagonalStep 0 (gtIncrementScale β s 0 q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1)
            (gtTerminal lam))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
      - gtCorrection β q s := by
  have hv0 : v = 0 := abs_eq_zero.mp hv
  subst v
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  simp [gtFunctional, gtSemigroupSolution, hq0]


lemma flatness_deriv_gtFunctional_formula_abs_v_eq_zero
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hv : |v| = 0) :
    deriv (fun l => gtFunctional β h q s l v) lam
      =
    standardGaussianExpectation (fun z =>
      deriv (fun l =>
        gtDiagonalStep 0 (gtIncrementScale β s 0 q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1)
            (gtTerminal l))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) lam) := by
  have hv0 : v = 0 := abs_eq_zero.mp hv
  subst v
  rw [deriv_gtFunctional_eq]
  simp only [sub_zero]
  apply congrArg standardGaussianExpectation
  funext z
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have hfun :
      (fun l =>
        gtSemigroupSolution β q s l 0 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
        =
      (fun l =>
        gtDiagonalStep 0 (gtIncrementScale β s 0 q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1)
            (gtTerminal l))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) := by
    funext l
    simp [gtSemigroupSolution, hq0]
  rw [hfun]


/-! ### Case `0 < |v| < q` -/

lemma flatness_gtFunctional_formula_abs_v_lt_q
    (β h q s lam v : ℝ)
    (hv0 : 0 < |v|) (hvq : |v| < q) :
    gtFunctional β h q s lam v
      =
    2 * Real.log 2
      + standardGaussianExpectation (fun z =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 |v|) (gtPathSign v)
          (gtDiagonalStep 0
            (gtIncrementScale β s |v| q)
            (gtDiagonalStep 1
              (gtIncrementScale β s q 1)
              (gtTerminal lam)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
      - lam * v - gtCorrection β q s := by
  have hqr : ¬ q ≤ |v| := not_le.mpr hvq
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hv0
  have hqpos : 0 < q := lt_trans hv0 hvq
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hqpos
  simp [gtFunctional, gtSemigroupSolution, hqr, hr0, hq0]


lemma flatness_deriv_gtFunctional_formula_abs_v_lt_q
    (β h q s lam v : ℝ)
    (hv0 : 0 < |v|) (hvq : |v| < q) :
    deriv (fun l => gtFunctional β h q s l v) lam
      =
    standardGaussianExpectation (fun z =>
      deriv (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 |v|) (gtPathSign v)
          (gtDiagonalStep 0
            (gtIncrementScale β s |v| q)
            (gtDiagonalStep 1
              (gtIncrementScale β s q 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) lam) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun x : ℝ => x - v)
  apply congrArg standardGaussianExpectation
  funext z

  have hqr : ¬ q ≤ |v| := not_le.mpr hvq
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hv0
  have hqpos : 0 < q := lt_trans hv0 hvq
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hqpos

  have hfun :
      (fun l =>
        gtSemigroupSolution β q s l v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
        =
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 |v|) (gtPathSign v)
          (gtDiagonalStep 0
            (gtIncrementScale β s |v| q)
            (gtDiagonalStep 1
              (gtIncrementScale β s q 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) := by
    funext l
    simp [gtSemigroupSolution, hqr, hr0, hq0]

  rw [hfun]


/-! ### Case `q ≤ |v| < 1` -/

lemma flatness_gtFunctional_formula_q_le_abs_v_lt_one
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hqv : q ≤ |v|) (_hv1 : |v| < 1) :
    gtFunctional β h q s lam v
      =
    2 * Real.log 2
      + standardGaussianExpectation (fun z =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q |v|) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s |v| 1)
              (gtTerminal lam)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
      - lam * v - gtCorrection β q s := by
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have hrpos : 0 < |v| := lt_of_lt_of_le hq hqv
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos
  simp [gtFunctional, gtSemigroupSolution, hqv, hr0, hq0]


lemma flatness_deriv_gtFunctional_formula_q_le_abs_v_lt_one
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hqv : q ≤ |v|) (_hv1 : |v| < 1) :
    deriv (fun l => gtFunctional β h q s l v) lam
      =
    standardGaussianExpectation (fun z =>
      deriv (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q |v|) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s |v| 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) lam) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun x : ℝ => x - v)
  apply congrArg standardGaussianExpectation
  funext z

  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have hrpos : 0 < |v| := lt_of_lt_of_le hq hqv
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos

  have hfun :
      (fun l =>
        gtSemigroupSolution β q s l v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
        =
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q |v|) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s |v| 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) := by
    funext l
    simp [gtSemigroupSolution, hqv, hr0, hq0]

  rw [hfun]


/-! ### Case `|v| = 1` -/

lemma flatness_gtFunctional_formula_abs_v_eq_one
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hq1 : q ≤ 1) (hv : |v| = 1) :
    gtFunctional β h q s lam v
      =
    2 * Real.log 2
      + standardGaussianExpectation (fun z =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q 1) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s 1 1)
              (gtTerminal lam)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
      - lam * v - gtCorrection β q s := by
  have hqv : q ≤ |v| := by
    simpa [hv] using hq1
  have hrpos : 0 < |v| := by
    rw [hv]
    norm_num
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have h10 : ¬ (1 : ℝ) ≤ 0 := by norm_num
  simp [gtFunctional, gtSemigroupSolution, hv, hq1, hqv, hr0, hq0, h10]


lemma flatness_deriv_gtFunctional_formula_abs_v_eq_one
    (β h q s lam v : ℝ)
    (hq : 0 < q) (hq1 : q ≤ 1) (hv : |v| = 1) :
    deriv (fun l => gtFunctional β h q s l v) lam
      =
    standardGaussianExpectation (fun z =>
      deriv (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q 1) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s 1 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) lam) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun x : ℝ => x - v)
  apply congrArg standardGaussianExpectation
  funext z

  have hqv : q ≤ |v| := by
    simpa [hv] using hq1
  have hrpos : 0 < |v| := by
    rw [hv]
    norm_num
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq

  have hfun :
      (fun l =>
        gtSemigroupSolution β q s l v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z))
        =
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q 1) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s 1 1)
              (gtTerminal l)))
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) := by
    funext l
    simp [gtSemigroupSolution, hv, hq1, hqv, hr0, hq0]

  rw [hfun]


/-! ### Explicit formulas for ∂_λ U_s^{λ,v}|_{λ=0} -/

/-! Case `|v| = 0`. -/

/-!
## Continuity of the GT functional
-/

/-- For fixed model parameters and overlap, the GT functional is continuous
in its Lagrange multiplier. -/
lemma continuous_gtFunctional_lam (β h q s v : ℝ) :
    Continuous (fun lam : ℝ => gtFunctional β h q s lam v) := by
  rw [continuous_iff_continuousAt]
  intro lam
  exact (hasDerivAt_gtFunctional β h q s lam v).continuousAt

/-- The compact parameter set carried by `UniformATData` lies in the strict
AT region. -/
lemma UniformATData.subset_strictATRegion {K : Set (ℝ × ℝ)}
    (data : UniformATData K) : K ⊆ strictATRegion := by
  intro p hp
  refine ⟨data.β_pos p hp, data.h_pos p hp, ?_⟩
  have hAT := data.strictAT p hp
  linarith [data.gap_pos]

/-- Joint continuity of the canonical GT functional on
`K × [0,1] × [-1,1] × [0,1]`, with the last coordinate representing
the multiplier restricted to the compact interval used in the flatness
argument. -/
lemma continuousOn_gtFunctional_uniformATData {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ContinuousOn (fun w : (ℝ × ℝ) × (ℝ × (ℝ × ℝ)) =>
      gtFunctional w.1.1 w.1.2 (rsQ w.1.1 w.1.2)
        w.2.1 w.2.2.2 w.2.2.1)
      (K ×ˢ (Icc (0 : ℝ) 1 ×ˢ (Icc (-1 : ℝ) 1 ×ˢ Icc (0 : ℝ) 1))) := by
  exact continuousOn_gtFunctional K data.subset_strictATRegion

/-!
## Elementary facts about the GT envelope
-/

/--
The infimum defining `gtEnvelope` is bounded above by every value of
`gtFunctional`, provided the range is bounded below.
-/
lemma gtEnvelope_le_functional
    (β h q s v lam : ℝ)
    (hbdd :
      BddBelow
        (Set.range (fun l : ℝ =>
          gtFunctional β h q s l v))) :
    gtEnvelope β h q s v ≤
      gtFunctional β h q s lam v := by
  rw [gtEnvelope]
  exact csInf_le hbdd ⟨lam, rfl⟩


/--
If `lam₀` is a global minimizer of the GT functional, then the GT envelope
equals the value at `lam₀`.
-/
lemma gtEnvelope_eq_functional_of_global_min
    (β h q s v lam₀ : ℝ)
    (hbdd :
      BddBelow
        (Set.range (fun l : ℝ =>
          gtFunctional β h q s l v)))
    (hmin :
      ∀ lam : ℝ,
        gtFunctional β h q s lam₀ v ≤
          gtFunctional β h q s lam v) :
    gtEnvelope β h q s v =
      gtFunctional β h q s lam₀ v := by
  apply le_antisymm

  · exact
      gtEnvelope_le_functional
        β h q s v lam₀ hbdd

  · rw [gtEnvelope]
    refine le_csInf (Set.range_nonempty _) ?_
    intro y hy
    rcases hy with ⟨lam, rfl⟩
    exact hmin lam


/-!
## Local quadratic coercivity from the Taylor package
-/

/--
The Taylor package gives a uniform quadratic loss on the region
`-q ≤ v ≤ 1`.
-/
lemma gtFunctional_quadratic_gap_of_taylor
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    (hTaylor : HasGTFunctionalTaylorPackage data) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-q : ℝ) 1 →
      ∃ lam,
        gtFunctional β h q s lam v ≤
          2 * rsPathValue β h q s
            - c * (v - q) ^ 2 := by

  rcases hTaylor with ⟨M, hM, hTaylor⟩

  let c : ℝ := data.gap ^ 2 / (2 * M)

  have hc : 0 < c := by
    dsimp [c]
    exact div_pos (sq_pos_of_pos data.gap_pos) (mul_pos (by norm_num) hM)

  refine ⟨c, hc, ?_⟩

  intro β h q s v hK hq hs hv

  obtain ⟨d, hzero, hdM, hdgap, htaylor⟩ :=
    hTaylor hK hq hs hv

  /-
  Subtract the RS value so that the Taylor estimate is centered at zero.
  -/
  let H : ℝ → ℝ :=
    fun lam =>
      gtFunctional β h q s lam v
        - 2 * rsPathValue β h q s

  have hHzero : H 0 ≤ 0 := by
    dsimp [H]
    rw [hzero]
    linarith

  have hHtaylor :
      ∀ lam, |lam| ≤ 1 →
        H lam ≤
          H 0 + d * lam + M / 2 * lam ^ 2 := by
    intro lam hlam
    have h := htaylor lam hlam
    dsimp [H]
    linarith

  /-
  Apply the deterministic completion-of-the-square lemma.
  -/
  obtain ⟨lam, hlam, hloss⟩ :=
    gt_taylor_quadratic_loss
      H d M data.gap (v - q)
      hM data.gap_pos hHzero hHtaylor
      hdM hdgap

  refine ⟨lam, ?_⟩

  dsimp [H] at hloss
  dsimp [c]

  linarith


/-!
## Turning a fixed negative-overlap gap into a quadratic gap
-/

/--
On `-1 ≤ v ≤ -q`, the distance `|v-q|` is at most `2`.
-/
lemma sub_sq_le_four_of_negative_overlap
    {q v : ℝ}
    (hq0 : 0 ≤ q)
    (hv : v ∈ Icc (-1 : ℝ) (-q)) :
    (v - q) ^ 2 ≤ 4 := by

  have hv_lower : -1 ≤ v := hv.1
  have hv_upper : v ≤ -q := hv.2

  have hq1 : q ≤ 1 := by
    linarith

  have hdiff_lower : -2 ≤ v - q := by
    linarith

  have hdiff_upper : v - q ≤ 0 := by
    linarith

  have hprod :
      0 ≤ ((v - q) + 2) * (2 - (v - q)) := by
    apply mul_nonneg
    · linarith
    · linarith

  nlinarith


/--
A fixed gap `k` on the negative-overlap region implies a quadratic gap
with coefficient `k / 4`.
-/
lemma gtFunctional_quadratic_gap_of_negative_gap
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    (hNegative : HasGTFunctionalNegativeGap data) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) (-q) →
      ∃ lam,
        gtFunctional β h q s lam v ≤
          2 * rsPathValue β h q s
            - c * (v - q) ^ 2 := by

  rcases hNegative with ⟨k, hk, hNegative⟩

  let c : ℝ := k / 4

  have hc : 0 < c := by
    dsimp [c]
    positivity

  refine ⟨c, hc, ?_⟩

  intro β h q s v hK hq hs hv

  obtain ⟨lam, hlam⟩ :=
    hNegative hK hq hs hv

  /-
  First obtain `q > 0` uniformly from `UniformATData`.
  -/
  have hq_lower :
      data.qmin ≤ q := by
    have h :=
      data.q_lower (β, h) hK
    simpa [hq] using h

  have hq0 : 0 ≤ q := by
    exact (lt_of_lt_of_le data.qmin_pos hq_lower).le

  have hsq :
      (v - q) ^ 2 ≤ 4 :=
    sub_sq_le_four_of_negative_overlap hq0 hv

  have hquad :
      c * (v - q) ^ 2 ≤ k := by
    dsimp [c]
    nlinarith

  refine ⟨lam, ?_⟩
  linarith


/-!
## Combine the two overlap regions
-/

/--
Combining the Taylor region `[-q,1]` and the negative region `[-1,-q]`
gives the uniform quadratic gap for the unoptimized GT functional.
-/
theorem gtFunctional_uniform_quadratic_gap_of_packages
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    (hTaylor : HasGTFunctionalTaylorPackage data)
    (hNegative : HasGTFunctionalNegativeGap data) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam,
        gtFunctional β h q s lam v ≤
          2 * rsPathValue β h q s
            - c * (v - q) ^ 2 := by

  obtain ⟨c₁, hc₁, hlocal⟩ :=
    gtFunctional_quadratic_gap_of_taylor
      data hTaylor

  obtain ⟨c₂, hc₂, hnegative⟩ :=
    gtFunctional_quadratic_gap_of_negative_gap
      data hNegative

  let c : ℝ := min c₁ c₂

  have hc : 0 < c := by
    dsimp [c]
    exact lt_min hc₁ hc₂

  refine ⟨c, hc, ?_⟩

  intro β h q s v hK hq hs hv

  by_cases hvq : -q ≤ v

  /-
  Case `-q ≤ v`.
  -/
  · have hv_local :
        v ∈ Icc (-q : ℝ) 1 :=
      ⟨hvq, hv.2⟩

    obtain ⟨lam, hlam⟩ :=
      hlocal hK hq hs hv_local

    refine ⟨lam, hlam.trans ?_⟩

    have hc_le : c ≤ c₁ := by
      exact min_le_left _ _

    have hsq : 0 ≤ (v - q) ^ 2 :=
      sq_nonneg _

    nlinarith

  /-
  Case `v < -q`.
  -/
  · have hv_neg :
        v ∈ Icc (-1 : ℝ) (-q) := by
      constructor
      · exact hv.1
      · exact le_of_lt (lt_of_not_ge hvq)

    obtain ⟨lam, hlam⟩ :=
      hnegative hK hq hs hv_neg

    refine ⟨lam, hlam.trans ?_⟩

    have hc_le : c ≤ c₂ := by
      exact min_le_right _ _

    have hsq : 0 ≤ (v - q) ^ 2 :=
      sq_nonneg _

    nlinarith


/-!
## Pass from the unoptimized functional to the envelope
-/

/--
Any uniform quadratic estimate for one choice of `lam` gives the same
estimate for the infimum over `lam`.
-/
lemma gtEnvelope_quadratic_gap_of_functional_gap
    {K : Set (ℝ × ℝ)}
    {c : ℝ}
    (hgap :
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        ∃ lam,
          gtFunctional β h q s lam v ≤
            2 * rsPathValue β h q s
              - c * (v - q) ^ 2)
    (hbdd :
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        BddBelow
          (Set.range (fun lam : ℝ =>
            gtFunctional β h q s lam v))) :
    ∀ {β h q s v : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      gtEnvelope β h q s v ≤
        2 * rsPathValue β h q s
          - c * (v - q) ^ 2 := by

  intro β h q s v hK hq hs hv

  obtain ⟨lam, hlam⟩ :=
    hgap hK hq hs hv

  have henv :
      gtEnvelope β h q s v ≤
        gtFunctional β h q s lam v :=
    gtEnvelope_le_functional
      β h q s v lam
      (hbdd hK hq hs hv)

  exact henv.trans hlam


/-!
## Equality at the replica-symmetric overlap
-/

/--
If `lam = 0` is a global minimizer at `v = q`, then the envelope at `q`
is exactly the RS value.
-/
lemma gtEnvelope_eq_rsPathValue_at_q
    {K : Set (ℝ × ℝ)}
    (hzero :
      ∀ {β h q s : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        gtFunctional β h q s 0 q =
          2 * rsPathValue β h q s)
    (hglobal :
      ∀ {β h q s lam : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        gtFunctional β h q s 0 q ≤
          gtFunctional β h q s lam q)
    (hbdd :
      ∀ {β h q s : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        BddBelow
          (Set.range (fun lam : ℝ =>
            gtFunctional β h q s lam q))) :
    ∀ {β h q s : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      gtEnvelope β h q s q =
        2 * rsPathValue β h q s := by

  intro β h q s hK hq hs

  calc
    gtEnvelope β h q s q
        =
        gtFunctional β h q s 0 q := by
          apply gtEnvelope_eq_functional_of_global_min
          · exact hbdd hK hq hs
          · intro lam
            exact hglobal hK hq hs

    _ = 2 * rsPathValue β h q s :=
      hzero hK hq hs


/-!
## Final proposition
-/

/--
Uniform coercivity of the optimized GT functional, together with equality
at the replica-symmetric overlap.
-/
theorem gtEnvelope_coercivity_of_packages
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    (hTaylor : HasGTFunctionalTaylorPackage data)
    (hNegative : HasGTFunctionalNegativeGap data)

    (hbdd :
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        BddBelow
          (Set.range (fun lam : ℝ =>
            gtFunctional β h q s lam v)))

    (hq_mem :
      ∀ {β h : ℝ},
        (β, h) ∈ K →
        rsQ β h ∈ Icc (-1 : ℝ) 1)

    (hzero :
      ∀ {β h q s : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        gtFunctional β h q s 0 q =
          2 * rsPathValue β h q s)

    (hglobal :
      ∀ {β h q s lam : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        gtFunctional β h q s 0 q ≤
          gtFunctional β h q s lam q) :

    ∃ c > 0,
      (∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        gtEnvelope β h q s v ≤
          2 * rsPathValue β h q s
            - c * (v - q) ^ 2)
      ∧
      (∀ {β h q s : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        gtEnvelope β h q s q =
          2 * rsPathValue β h q s) := by

  obtain ⟨c, hc, hgap⟩ :=
    gtFunctional_uniform_quadratic_gap_of_packages
      data hTaylor hNegative

  refine ⟨c, hc, ?_, ?_⟩

  /-
  Coercivity of the envelope.
  -/
  · exact
      gtEnvelope_quadratic_gap_of_functional_gap
        hgap hbdd

  /-
  Equality at `v = q`.
  -/
  · apply gtEnvelope_eq_rsPathValue_at_q hzero hglobal

    intro β h q s hK hq hs

    have hqIcc :
        q ∈ Icc (-1 : ℝ) 1 := by
      rw [hq]
      exact hq_mem hK

    exact hbdd hK hq hs hqIcc


theorem gtFunctional_uniform_quadratic_gap {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam, gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  sorry

end SpinGlass.AT
