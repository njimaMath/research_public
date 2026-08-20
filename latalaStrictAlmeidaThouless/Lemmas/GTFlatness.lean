import Lemmas.Psi_continuity
import Lemmas.ATDefs
import Lemmas.GTGauss
import Lemmas.interpolatedAT
import Mathlib.MeasureTheory.Group.IntegralConvolution

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

private abbrev flatnessGauss : Measure ℝ := gaussianReal 0 1

private lemma flatness_fLbaseD_zero (x₁ x₂ : ℝ) :
    GTFrame.fLbaseD 0 (x₁, x₂) = Real.tanh x₁ * Real.tanh x₂ := by
  have h := (GTFrame.hasDerivAt_fLbase 0 (x₁, x₂)).deriv
  change deriv (fun l => gtTerminal l x₁ x₂) 0 = _ at h
  rw [deriv_gtTerminal_zero] at h
  exact h.symm

private lemma flatness_upper_goodFam (b : ℝ) :
    GTFrame.GoodFam
      (fun (_ : Unit) l (x : ℝ × ℝ) => GTFrame.fLbase l x + b ^ 2)
      (fun (_ : Unit) l (x : ℝ × ℝ) => GTFrame.fLbaseD l x) := by
  refine
    { contF := (GTFrame.continuous_fLbase.comp (by fun_prop)).add continuous_const
      contD := GTFrame.continuous_fLbaseD.comp (by fun_prop)
      hasDeriv := fun _ l x => (GTFrame.hasDerivAt_fLbase l x).add_const _
      lipx := ?_
      bddD := fun _ l x => GTFrame.fLbaseD_bdd l x }
  intro _ l x y
  simpa using GTFrame.fLbase_lipx l x y

/-- A final level-one diagonal step does not change the multiplier derivative
of the terminal condition. -/
lemma deriv_gtDiagonalStep_one_gtTerminal_zero (scale x₁ x₂ : ℝ) :
    deriv (fun l => gtDiagonalStep 1 scale (gtTerminal l) x₁ x₂) 0 =
      Real.tanh x₁ * Real.tanh x₂ := by
  have hfun :
      (fun l => gtDiagonalStep 1 scale (gtTerminal l) x₁ x₂) =
      (fun l => gtTerminal l x₁ x₂ + scale ^ 2) := by
    funext l
    exact gtDiagonalStep_one_terminal scale l x₁ x₂
  rw [hfun, ((hasDerivAt_gtTerminal 0 x₁ x₂).add_const (scale ^ 2)).deriv]
  change GTFrame.fLbaseD 0 (x₁, x₂) = _
  exact flatness_fLbaseD_zero x₁ x₂

/-- Propagation through a zero-mass independent diagonal step. -/
lemma deriv_gtDiagonalStep_zero_one_terminal_zero (a b x₁ x₂ : ℝ) :
    deriv (fun l => gtDiagonalStep 0 a
      (gtDiagonalStep 1 b (gtTerminal l)) x₁ x₂) 0 =
    standardGaussianExpectation (fun z₁ =>
      standardGaussianExpectation (fun z₂ =>
        Real.tanh (x₁ + a * z₁) * Real.tanh (x₂ + a * z₂))) := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbase l x + b ^ 2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  have hF : GTFrame.GoodFam F D := flatness_upper_goodFam b
  let F₂ := GTFrame.step0 flatnessGauss (fun _ : Unit => 0) (fun _ => a) F
  let D₂ := GTFrame.step0 flatnessGauss (fun _ : Unit => 0) (fun _ => a) D
  have hF₂ : GTFrame.GoodFam F₂ D₂ :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hF
      continuous_const continuous_const
  let F₃ := GTFrame.step0 flatnessGauss (fun _ : Unit => a) (fun _ => 0) F₂
  let D₃ := GTFrame.step0 flatnessGauss (fun _ : Unit => a) (fun _ => 0) D₂
  have hF₃ : GTFrame.GoodFam F₃ D₃ :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hF₂
      continuous_const continuous_const
  have hd := (hF₃.hasDeriv () 0 (x₁, x₂)).deriv
  change deriv (fun l => F₃ () l (x₁, x₂)) 0 = D₃ () 0 (x₁, x₂) at hd
  have hfun :
      (fun l => gtDiagonalStep 0 a
        (gtDiagonalStep 1 b (gtTerminal l)) x₁ x₂) =
      (fun l => F₃ () l (x₁, x₂)) := by
    funext l
    have hu : gtDiagonalStep 1 b (gtTerminal l) =
        fun y₁ y₂ => gtTerminal l y₁ y₂ + b ^ 2 := by
      funext y₁ y₂
      exact gtDiagonalStep_one_terminal b l y₁ y₂
    rw [hu]
    simp [F₃, F₂, F, GTFrame.step0, flatnessGauss,
      standardGaussianExpectation, gtDiagonalStep]
  rw [hfun, hd]
  simp [D₃, D₂, D, GTFrame.step0, flatnessGauss,
    standardGaussianExpectation, flatness_fLbaseD_zero]

/-- Propagation through a zero-mass rank-one step followed by a zero-mass
independent diagonal step. -/
lemma deriv_gtRankOneStep_zero_diagonal_zero_at_zero
    (r sign a b x₁ x₂ : ℝ) :
    deriv (fun l => gtRankOneStep 0 r sign
      (gtDiagonalStep 0 a (gtDiagonalStep 1 b (gtTerminal l))) x₁ x₂) 0 =
    standardGaussianExpectation (fun z₀ =>
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          Real.tanh (x₁ + r * z₀ + a * z₁) *
            Real.tanh (x₂ + sign * r * z₀ + a * z₂)))) := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbase l x + b ^ 2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  have hF : GTFrame.GoodFam F D := flatness_upper_goodFam b
  let F₂ := GTFrame.step0 flatnessGauss (fun _ : Unit => 0) (fun _ => a) F
  let D₂ := GTFrame.step0 flatnessGauss (fun _ : Unit => 0) (fun _ => a) D
  have hF₂ : GTFrame.GoodFam F₂ D₂ :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hF
      continuous_const continuous_const
  let F₃ := GTFrame.step0 flatnessGauss (fun _ : Unit => a) (fun _ => 0) F₂
  let D₃ := GTFrame.step0 flatnessGauss (fun _ : Unit => a) (fun _ => 0) D₂
  have hF₃ : GTFrame.GoodFam F₃ D₃ :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hF₂
      continuous_const continuous_const
  let F₄ := GTFrame.step0 flatnessGauss (fun _ : Unit => r) (fun _ => sign * r) F₃
  let D₄ := GTFrame.step0 flatnessGauss (fun _ : Unit => r) (fun _ => sign * r) D₃
  have hF₄ : GTFrame.GoodFam F₄ D₄ :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hF₃
      continuous_const continuous_const
  have hd := (hF₄.hasDeriv () 0 (x₁, x₂)).deriv
  change deriv (fun l => F₄ () l (x₁, x₂)) 0 = D₄ () 0 (x₁, x₂) at hd
  have hfun :
      (fun l => gtRankOneStep 0 r sign
        (gtDiagonalStep 0 a (gtDiagonalStep 1 b (gtTerminal l))) x₁ x₂) =
      (fun l => F₄ () l (x₁, x₂)) := by
    funext l
    have hu : gtDiagonalStep 1 b (gtTerminal l) =
        fun y₁ y₂ => gtTerminal l y₁ y₂ + b ^ 2 := by
      funext y₁ y₂
      exact gtDiagonalStep_one_terminal b l y₁ y₂
    rw [hu]
    simp [F₄, F₃, F₂, F, GTFrame.step0, flatnessGauss,
      standardGaussianExpectation, gtRankOneStep, gtDiagonalStep]
  rw [hfun, hd]
  simp [D₄, D₃, D₂, D, GTFrame.step0, flatnessGauss,
    standardGaussianExpectation, flatness_fLbaseD_zero]

/-- The explicit tilted quotient produced by a mass-`1/2` rank-one step at
`lam = 0`.  The `upperScale` term records the harmless constant contributed
by the final level-one diagonal step. -/
noncomputable def gtHalfStepEndpoint
    (scale upperScale sign x₁ x₂ : ℝ) : ℝ :=
  standardGaussianExpectation (fun z =>
      (Real.tanh (x₁ + scale * z) *
        Real.tanh (x₂ + sign * scale * z)) *
      Real.exp ((1 / 2) *
        (gtTerminal 0 (x₁ + scale * z) (x₂ + sign * scale * z) +
          upperScale ^ 2))) /
    standardGaussianExpectation (fun z =>
      Real.exp ((1 / 2) *
        (gtTerminal 0 (x₁ + scale * z) (x₂ + sign * scale * z) +
          upperScale ^ 2)))

/-- On the diagonal and with positive path sign, the half-step endpoint is the
usual `tanh² cosh` tilted quotient. -/
lemma gtHalfStepEndpoint_diagonal (scale upperScale x : ℝ) :
    gtHalfStepEndpoint scale upperScale 1 x x =
      standardGaussianExpectation (fun z =>
        Real.tanh (x + scale * z) ^ 2 * Real.cosh (x + scale * z)) /
      standardGaussianExpectation (fun z => Real.cosh (x + scale * z)) := by
  have hw (y : ℝ) :
      Real.exp ((1 / 2) * (gtTerminal 0 y y + upperScale ^ 2)) =
        Real.exp (upperScale ^ 2 / 2) * Real.cosh y := by
    rw [gtTerminal_zero]
    rw [show (1 / 2 : ℝ) *
      (Real.log (Real.cosh y) + Real.log (Real.cosh y) + upperScale ^ 2) =
        upperScale ^ 2 / 2 + Real.log (Real.cosh y) by ring]
    rw [Real.exp_add, Real.exp_log (Real.cosh_pos y)]
  unfold gtHalfStepEndpoint
  simp only [one_mul]
  simp_rw [hw]
  have hn : (fun z =>
      (Real.tanh (x + scale * z) * Real.tanh (x + scale * z)) *
        (Real.exp (upperScale ^ 2 / 2) * Real.cosh (x + scale * z))) =
      fun z => Real.exp (upperScale ^ 2 / 2) *
        (Real.tanh (x + scale * z) ^ 2 * Real.cosh (x + scale * z)) := by
    funext z
    ring
  rw [hn]
  unfold standardGaussianExpectation
  rw [integral_const_mul, integral_const_mul]
  field_simp [Real.exp_ne_zero (upperScale ^ 2 / 2)]

/-- Gaussian expectation of a shifted hyperbolic cosine. -/
lemma standardGaussianExpectation_cosh_shift (scale x : ℝ) :
    standardGaussianExpectation (fun z => Real.cosh (x + scale * z)) =
      Real.exp (scale ^ 2 / 2) * Real.cosh x := by
  have hmgf (t : ℝ) :
      standardGaussianExpectation (fun z => Real.exp (t * z)) =
        Real.exp (t ^ 2 / 2) := by
    have h := congrFun (mgf_id_gaussianReal (μ := 0) (v := 1)) t
    simpa [mgf, standardGaussianExpectation] using h
  have hi₁ : Integrable (fun z : ℝ => Real.exp x * Real.exp (scale * z))
      (gaussianReal 0 1) :=
    (integrable_exp_mul_gaussianReal scale).const_mul _
  have hi₂ : Integrable (fun z : ℝ => Real.exp (-x) * Real.exp ((-scale) * z))
      (gaussianReal 0 1) :=
    (integrable_exp_mul_gaussianReal (-scale)).const_mul _
  have hfun : (fun z : ℝ => Real.cosh (x + scale * z)) =
      fun z => (Real.exp x * Real.exp (scale * z) +
        Real.exp (-x) * Real.exp ((-scale) * z)) / 2 := by
    funext z
    rw [Real.cosh_eq, Real.exp_add]
    congr 1
    rw [show -(x + scale * z) = -x + (-scale) * z by ring, Real.exp_add]
  rw [hfun]
  unfold standardGaussianExpectation
  rw [integral_div, integral_add hi₁ hi₂, integral_const_mul, integral_const_mul]
  change (Real.exp x * standardGaussianExpectation (fun z => Real.exp (scale * z)) +
      Real.exp (-x) * standardGaussianExpectation (fun z => Real.exp ((-scale) * z))) / 2 = _
  rw [hmgf scale, hmgf (-scale), Real.cosh_eq]
  ring_nf

/-- The diagonal half-step quotient is exactly the tilted heat semigroup.
This includes the degenerate case `scale = 0`. -/
lemma gtHalfStepEndpoint_diagonal_eq_tilted
    (scale upperScale x : ℝ) (hscale : 0 ≤ scale) :
    gtHalfStepEndpoint scale upperScale 1 x x =
      tiltedHeatSemigroup (scale ^ 2) (fun y => Real.tanh y ^ 2) x := by
  rw [gtHalfStepEndpoint_diagonal, standardGaussianExpectation_cosh_shift]
  unfold tiltedHeatSemigroup heatSemigroup
  rw [Real.sqrt_sq_eq_abs, abs_of_nonneg hscale]
  field_simp [Real.exp_ne_zero (scale ^ 2 / 2), ne_of_gt (Real.cosh_pos x)]
  rw [mul_assoc, ← Real.exp_add]
  ring_nf
  simp

/-- Common endpoint derivative formula for a zero-mass rank-one step followed
by the mass-`1/2` step. -/
lemma deriv_gtRankOneStep_zero_half_at_zero
    (r a b sign x₁ x₂ : ℝ) :
    deriv (fun l => gtRankOneStep 0 r sign
      (gtRankOneStep (1 / 2) a sign
        (gtDiagonalStep 1 b (gtTerminal l))) x₁ x₂) 0 =
    standardGaussianExpectation (fun z₀ =>
      gtHalfStepEndpoint a b sign
        (x₁ + r * z₀) (x₂ + sign * r * z₀)) := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbase l x + b ^ 2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x
  have hF : GTFrame.GoodFam F D := flatness_upper_goodFam b
  let FH := GTFrame.stepM flatnessGauss (1 / 2)
    (fun _ : Unit => a) (fun _ => sign * a) F
  let DH := GTFrame.stepMD flatnessGauss (1 / 2)
    (fun _ : Unit => a) (fun _ => sign * a) F D
  have hFH : GTFrame.GoodFam FH DH :=
    GTFrame.stepM_good (GTFrame.expMoments_gaussianReal 0 1) hF
      (by norm_num) continuous_const continuous_const
  let FO := GTFrame.step0 flatnessGauss
    (fun _ : Unit => r) (fun _ => sign * r) FH
  let DO := GTFrame.step0 flatnessGauss
    (fun _ : Unit => r) (fun _ => sign * r) DH
  have hFO : GTFrame.GoodFam FO DO :=
    GTFrame.step0_good (GTFrame.expMoments_gaussianReal 0 1) hFH
      continuous_const continuous_const
  have hd := (hFO.hasDeriv () 0 (x₁, x₂)).deriv
  change deriv (fun l => FO () l (x₁, x₂)) 0 = DO () 0 (x₁, x₂) at hd
  have hfun :
      (fun l => gtRankOneStep 0 r sign
        (gtRankOneStep (1 / 2) a sign
          (gtDiagonalStep 1 b (gtTerminal l))) x₁ x₂) =
      (fun l => FO () l (x₁, x₂)) := by
    funext l
    have hu : gtDiagonalStep 1 b (gtTerminal l) =
        fun y₁ y₂ => gtTerminal l y₁ y₂ + b ^ 2 := by
      funext y₁ y₂
      exact gtDiagonalStep_one_terminal b l y₁ y₂
    rw [hu]
    simp [FO, FH, F, GTFrame.step0, GTFrame.stepM, flatnessGauss,
      standardGaussianExpectation, gtRankOneStep]
  rw [hfun, hd]
  simp [DO, DH, F, D, GTFrame.step0, GTFrame.stepMD, flatnessGauss,
    standardGaussianExpectation, flatness_fLbaseD_zero, gtHalfStepEndpoint]

/-! #### Branchwise endpoint formulas for `U` -/

/-- Endpoint formula in the branch `|v| = 0`; the zero-length rank-one
increment has disappeared. -/
lemma flatness_deriv_U_abs_v_eq_zero
    (β q s v x₁ x₂ : ℝ) (hq : 0 < q) (hv : |v| = 0) :
    deriv (fun lam => gtSemigroupSolution β q s lam v 0 x₁ x₂) 0 =
    standardGaussianExpectation (fun z₁ =>
      standardGaussianExpectation (fun z₂ =>
        Real.tanh (x₁ + gtIncrementScale β s 0 q * z₁) *
          Real.tanh (x₂ + gtIncrementScale β s 0 q * z₂))) := by
  rw [deriv_gtSemigroupSolution_zero_abs_v_eq_zero β q s v x₁ x₂ hq hv]
  exact deriv_gtDiagonalStep_zero_one_terminal_zero _ _ _ _

/-- Endpoint formula in the branch `0 < |v| < q`. -/
lemma flatness_deriv_U_abs_v_lt_q
    (β q s v x₁ x₂ : ℝ) (hv0 : 0 < |v|) (hvq : |v| < q) :
    deriv (fun lam => gtSemigroupSolution β q s lam v 0 x₁ x₂) 0 =
    standardGaussianExpectation (fun z₀ =>
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          Real.tanh (x₁ + gtIncrementScale β s 0 |v| * z₀ +
            gtIncrementScale β s |v| q * z₁) *
          Real.tanh (x₂ + gtPathSign v * gtIncrementScale β s 0 |v| * z₀ +
            gtIncrementScale β s |v| q * z₂)))) := by
  rw [deriv_gtSemigroupSolution_zero_abs_v_lt_q β q s v x₁ x₂ hv0 hvq]
  exact deriv_gtRankOneStep_zero_diagonal_zero_at_zero _ _ _ _ _ _

/-- General endpoint formula in the branch `q ≤ |v| < 1`.  In particular,
the signed path factor is retained for negative overlaps. -/
lemma flatness_deriv_U_q_le_abs_v_lt_one
    (β q s v x₁ x₂ : ℝ) (hq : 0 < q) (hqv : q ≤ |v|) (hv1 : |v| < 1) :
    deriv (fun lam => gtSemigroupSolution β q s lam v 0 x₁ x₂) 0 =
    standardGaussianExpectation (fun z₀ =>
      gtHalfStepEndpoint (gtIncrementScale β s q |v|)
        (gtIncrementScale β s |v| 1) (gtPathSign v)
        (x₁ + gtIncrementScale β s 0 q * z₀)
        (x₂ + gtPathSign v * gtIncrementScale β s 0 q * z₀)) := by
  rw [deriv_gtSemigroupSolution_zero_q_le_abs_v_lt_one
    β q s v x₁ x₂ hq hqv hv1]
  exact deriv_gtRankOneStep_zero_half_at_zero _ _ _ _ _ _

/-- Endpoint formula at `|v| = 1`, obtained from the same half-step lemma as
the preceding branch. -/
lemma flatness_deriv_U_abs_v_eq_one
    (β q s v x₁ x₂ : ℝ) (hq : 0 < q) (hq1 : q ≤ 1) (hv : |v| = 1) :
    deriv (fun lam => gtSemigroupSolution β q s lam v 0 x₁ x₂) 0 =
    standardGaussianExpectation (fun z₀ =>
      gtHalfStepEndpoint (gtIncrementScale β s q 1)
        (gtIncrementScale β s 1 1) (gtPathSign v)
        (x₁ + gtIncrementScale β s 0 q * z₀)
        (x₂ + gtPathSign v * gtIncrementScale β s 0 q * z₀)) := by
  rw [deriv_gtSemigroupSolution_zero_abs_v_eq_one
    β q s v x₁ x₂ hq hq1 hv]
  exact deriv_gtRankOneStep_zero_half_at_zero _ _ _ _ _ _

private lemma gtIncrementScale_sq_of_nonneg
    (β s lower upper : ℝ) (hβ : 0 ≤ β) (hs : 0 ≤ s)
    (hlu : lower ≤ upper) :
    gtIncrementScale β s lower upper ^ 2 =
      s * β ^ 2 * (upper - lower) := by
  unfold gtIncrementScale
  rw [mul_pow, mul_pow, Real.sq_sqrt hs, Real.sq_sqrt (sub_nonneg.mpr hlu)]
  ring

private lemma gtIncrementScale_nonneg
    (β s lower upper : ℝ) (hβ : 0 ≤ β) :
    0 ≤ gtIncrementScale β s lower upper := by
  unfold gtIncrementScale
  exact mul_nonneg (mul_nonneg hβ (Real.sqrt_nonneg s))
    (Real.sqrt_nonneg (upper - lower))

/-- Positive-overlap diagonal specialization for `q ≤ v < 1`.  The initial
zero-mass step is an ordinary Gaussian expectation of the existing tilted heat
semigroup. -/
lemma flatness_deriv_U_q_le_v_lt_one_diagonal
    (β q s v x : ℝ) (hβ : 0 ≤ β) (hs : 0 ≤ s)
    (hq : 0 < q) (hqv : q ≤ v) (hv1 : v < 1) :
    deriv (fun lam => gtSemigroupSolution β q s lam v 0 x x) 0 =
    standardGaussianExpectation (fun z =>
      tiltedHeatSemigroup (s * β ^ 2 * (v - q))
        (fun y => Real.tanh y ^ 2)
        (x + gtIncrementScale β s 0 q * z)) := by
  have hv0 : 0 ≤ v := le_trans hq.le hqv
  have hsign : gtPathSign v = 1 := by simp [gtPathSign, hv0]
  rw [flatness_deriv_U_q_le_abs_v_lt_one β q s v x x hq]
  · simp only [abs_of_nonneg hv0, hsign, one_mul]
    apply congrArg standardGaussianExpectation
    funext z
    rw [gtHalfStepEndpoint_diagonal_eq_tilted]
    · rw [gtIncrementScale_sq_of_nonneg β s q v hβ hs hqv]
    · exact gtIncrementScale_nonneg β s q v hβ
  · simpa [abs_of_nonneg hv0] using hqv
  · simpa [abs_of_nonneg hv0] using hv1

/-- Positive endpoint specialization at `v = 1`. -/
lemma flatness_deriv_U_v_eq_one_diagonal
    (β q s x : ℝ) (hβ : 0 ≤ β) (hs : 0 ≤ s)
    (hq : 0 < q) (hq1 : q ≤ 1) :
    deriv (fun lam => gtSemigroupSolution β q s lam 1 0 x x) 0 =
    standardGaussianExpectation (fun z =>
      tiltedHeatSemigroup (s * β ^ 2 * (1 - q))
        (fun y => Real.tanh y ^ 2)
        (x + gtIncrementScale β s 0 q * z)) := by
  rw [flatness_deriv_U_abs_v_eq_one β q s 1 x x hq hq1 (by norm_num)]
  simp only [gtPathSign, if_pos (by norm_num : (0 : ℝ) ≤ 1), one_mul]
  apply congrArg standardGaussianExpectation
  funext z
  rw [gtHalfStepEndpoint_diagonal_eq_tilted]
  · rw [gtIncrementScale_sq_of_nonneg β s q 1 hβ hs hq1]
  · exact gtIncrementScale_nonneg β s q 1 hβ

/-! #### Immediate endpoint formulas for the GT functional -/

lemma flatness_deriv_gtFunctional_zero_abs_v_eq_zero
    (β h q s v : ℝ) (hq : 0 < q) (hv : |v| = 0) :
    deriv (fun l => gtFunctional β h q s l v) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          Real.tanh (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₁) *
          Real.tanh (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₂)))) := by
  have hv0 : v = 0 := abs_eq_zero.mp hv
  subst v
  rw [deriv_gtFunctional_eq]
  rw [sub_zero]
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_abs_v_eq_zero β q s 0 _ _ hq (abs_zero)

lemma flatness_deriv_gtFunctional_zero_abs_v_lt_q
    (β h q s v : ℝ) (hv0 : 0 < |v|) (hvq : |v| < q) :
    deriv (fun l => gtFunctional β h q s l v) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₀ =>
        standardGaussianExpectation (fun z₁ =>
          standardGaussianExpectation (fun z₂ =>
            Real.tanh (h + β * Real.sqrt ((1 - s) * q) * z +
              gtIncrementScale β s 0 |v| * z₀ +
              gtIncrementScale β s |v| q * z₁) *
            Real.tanh (h + β * Real.sqrt ((1 - s) * q) * z +
              gtPathSign v * gtIncrementScale β s 0 |v| * z₀ +
              gtIncrementScale β s |v| q * z₂))))) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun y : ℝ => y - v)
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_abs_v_lt_q β q s v _ _ hv0 hvq

lemma flatness_deriv_gtFunctional_zero_q_le_abs_v_lt_one
    (β h q s v : ℝ) (hq : 0 < q) (hqv : q ≤ |v|) (hv1 : |v| < 1) :
    deriv (fun l => gtFunctional β h q s l v) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₀ =>
        gtHalfStepEndpoint (gtIncrementScale β s q |v|)
          (gtIncrementScale β s |v| 1) (gtPathSign v)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₀)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtPathSign v * gtIncrementScale β s 0 q * z₀))) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun y : ℝ => y - v)
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_q_le_abs_v_lt_one β q s v _ _ hq hqv hv1

lemma flatness_deriv_gtFunctional_zero_abs_v_eq_one
    (β h q s v : ℝ) (hq : 0 < q) (hq1 : q ≤ 1) (hv : |v| = 1) :
    deriv (fun l => gtFunctional β h q s l v) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₀ =>
        gtHalfStepEndpoint (gtIncrementScale β s q 1)
          (gtIncrementScale β s 1 1) (gtPathSign v)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₀)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtPathSign v * gtIncrementScale β s 0 q * z₀))) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun y : ℝ => y - v)
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_abs_v_eq_one β q s v _ _ hq hq1 hv

/-- Positive-overlap functional endpoint formula in tilted-semigroup form. -/
lemma flatness_deriv_gtFunctional_zero_q_le_v_lt_one
    (β h q s v : ℝ) (hβ : 0 ≤ β) (hs : 0 ≤ s)
    (hq : 0 < q) (hqv : q ≤ v) (hv1 : v < 1) :
    deriv (fun l => gtFunctional β h q s l v) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₀ =>
        tiltedHeatSemigroup (s * β ^ 2 * (v - q))
          (fun y => Real.tanh y ^ 2)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₀))) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun y : ℝ => y - v)
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_q_le_v_lt_one_diagonal β q s v _ hβ hs hq hqv hv1

/-- Functional endpoint formula at the positive endpoint `v = 1`. -/
lemma flatness_deriv_gtFunctional_zero_v_eq_one
    (β h q s : ℝ) (hβ : 0 ≤ β) (hs : 0 ≤ s)
    (hq : 0 < q) (hq1 : q ≤ 1) :
    deriv (fun l => gtFunctional β h q s l 1) 0 =
    standardGaussianExpectation (fun z =>
      standardGaussianExpectation (fun z₀ =>
        tiltedHeatSemigroup (s * β ^ 2 * (1 - q))
          (fun y => Real.tanh y ^ 2)
          (h + β * Real.sqrt ((1 - s) * q) * z +
            gtIncrementScale β s 0 q * z₀))) - 1 := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun y : ℝ => y - 1)
  apply congrArg standardGaussianExpectation
  funext z
  exact flatness_deriv_U_v_eq_one_diagonal β q s _ hβ hs hq hq1

private lemma flatness_tiltedHeatSemigroup_zero (f : ℝ → ℝ) (x : ℝ) :
    tiltedHeatSemigroup 0 f x = f x := by
  unfold tiltedHeatSemigroup heatSemigroup standardGaussianExpectation
  simp [ne_of_gt (Real.cosh_pos x)]

private lemma flatness_gaussian_convolution_tanh_sq (h a b c : ℝ)
    (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun x =>
      standardGaussianExpectation (fun y => Real.tanh (h + a * x + b * y) ^ 2)) =
    standardGaussianExpectation (fun z => Real.tanh (h + c * z) ^ 2) := by
  let va : NNReal := NNReal.mk (a ^ 2) (sq_nonneg a) * 1
  let vb : NNReal := NNReal.mk (b ^ 2) (sq_nonneg b) * 1
  let vc : NNReal := NNReal.mk (c ^ 2) (sq_nonneg c) * 1
  have htanh : Continuous (fun x : ℝ => Real.tanh x) := by
    simp_rw [Real.tanh_eq_sinh_div_cosh]
    exact Real.continuous_sinh.div₀ Real.continuous_cosh
      (fun x => (Real.cosh_pos x).ne')
  have hma : Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1) =
      gaussianReal 0 va := by
    simpa [va] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) a)
  have hmb : Measure.map (fun x : ℝ => b * x) (gaussianReal 0 1) =
      gaussianReal 0 vb := by
    simpa [vb] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) b)
  have hmc : Measure.map (fun x : ℝ => c * x) (gaussianReal 0 1) =
      gaussianReal 0 vc := by
    simpa [vc] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) c)
  have hv : va + vb = vc := by
    apply NNReal.eq
    simp [va, vb, vc, hc]
  have hf : Integrable (fun z : ℝ => Real.tanh (h + z) ^ 2)
      (gaussianReal 0 va ∗ gaussianReal 0 vb) := by
    rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    apply Integrable.of_bound (C := 1)
    · exact ((htanh.comp (by fun_prop)).pow 2).aestronglyMeasurable
    · filter_upwards [] with z
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact (Real.tanh_sq_lt_one _).le
  have hprod : Integrable (fun p : ℝ × ℝ => Real.tanh (h + (p.1 + p.2)) ^ 2)
      ((gaussianReal 0 va).prod (gaussianReal 0 vb)) := by
    rw [Measure.conv] at hf
    exact (integrable_map_measure hf.1 (by fun_prop)).mp hf
  have houter : AEStronglyMeasurable
      (fun x : ℝ => ∫ y, Real.tanh (h + (x + y)) ^ 2 ∂gaussianReal 0 vb)
      (gaussianReal 0 va) := hprod.integral_prod_left.1
  have hinner (x : ℝ) :
      (∫ y, Real.tanh (h + a * x + b * y) ^ 2 ∂gaussianReal 0 1) =
        ∫ y, Real.tanh (h + a * x + y) ^ 2 ∂gaussianReal 0 vb := by
    have hm : AEStronglyMeasurable (fun y : ℝ => Real.tanh (h + a * x + y) ^ 2)
        (Measure.map (fun y : ℝ => b * y) (gaussianReal 0 1)) :=
      ((htanh.comp (by fun_prop)).pow 2).aestronglyMeasurable
    rw [← hmb, integral_map (by fun_prop) hm]
  have houter_map :
      (∫ x, ∫ y, Real.tanh (h + a * x + y) ^ 2 ∂gaussianReal 0 vb
        ∂gaussianReal 0 1) =
        ∫ x, ∫ y, Real.tanh (h + x + y) ^ 2 ∂gaussianReal 0 vb
          ∂gaussianReal 0 va := by
    have hm : AEStronglyMeasurable
        (fun x : ℝ => ∫ y, Real.tanh (h + (x + y)) ^ 2 ∂gaussianReal 0 vb)
        (Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1)) := by
      simpa [hma] using houter
    rw [← hma]
    simpa only [add_assoc] using (integral_map (by fun_prop) hm).symm
  unfold standardGaussianExpectation
  calc
    (∫ x, ∫ y, Real.tanh (h + a * x + b * y) ^ 2 ∂gaussianReal 0 1
        ∂gaussianReal 0 1) =
        ∫ x, ∫ y, Real.tanh (h + x + y) ^ 2 ∂gaussianReal 0 vb
          ∂gaussianReal 0 va := by
            rw [integral_congr_ae (Filter.Eventually.of_forall hinner)]
            exact houter_map
    _ = ∫ z, Real.tanh (h + z) ^ 2
          ∂(gaussianReal 0 va ∗ gaussianReal 0 vb) := by
            simpa only [add_assoc] using (integral_conv hf).symm
    _ = ∫ z, Real.tanh (h + z) ^ 2 ∂gaussianReal 0 vc := by
          rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    _ = ∫ z, Real.tanh (h + c * z) ^ 2 ∂gaussianReal 0 1 := by
          rw [← hmc, integral_map (by fun_prop)]
          exact ((htanh.comp (by fun_prop)).pow 2).aestronglyMeasurable

/-- At an interior replica-symmetric fixed point, the partial derivative of
the GT functional in the multiplier vanishes when the overlap is that fixed
point. -/
lemma flatness_deriv_gtFunctional_zero_at_fixedPoint
    (β h q s : ℝ) (hβ : 0 ≤ β) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 < q) (hq1 : q < 1) (hfixed : IsRSFixedPoint β h q) :
    deriv (fun lam => gtFunctional β h q s lam q) 0 = 0 := by
  rw [flatness_deriv_gtFunctional_zero_q_le_v_lt_one
    β h q s q hβ hs.1 hq le_rfl hq1]
  simp only [sub_self, mul_zero, flatness_tiltedHeatSemigroup_zero]
  rw [sub_eq_zero]
  have hsquare :
      (β * Real.sqrt q) ^ 2 =
        (β * Real.sqrt ((1 - s) * q)) ^ 2 +
          (gtIncrementScale β s 0 q) ^ 2 := by
    have hq0 : 0 ≤ q := hq.le
    have hs0 : 0 ≤ s := hs.1
    have h1s0 : 0 ≤ 1 - s := sub_nonneg.mpr hs.2
    simp only [gtIncrementScale, sub_zero]
    rw [mul_pow, Real.sq_sqrt hq0, mul_pow,
      Real.sq_sqrt (mul_nonneg h1s0 hq0)]
    rw [mul_pow, mul_pow, Real.sq_sqrt hs0, Real.sq_sqrt hq0]
    ring
  rw [flatness_gaussian_convolution_tanh_sq h
    (β * Real.sqrt ((1 - s) * q)) (gtIncrementScale β s 0 q)
    (β * Real.sqrt q) hsquare]
  exact hfixed.symm

/-- For the canonical replica-symmetric overlap `q = rsQ β h`, the partial
derivative of the GT functional in `lam` vanishes at `(lam, v) = (0, q)`. -/
lemma flatness_deriv_gtFunctional_zero_at_rsQ
    (β h s : ℝ) (hβ : 0 < β) (hh : 0 < h)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    deriv (fun lam => gtFunctional β h (rsQ β h) s lam (rsQ β h)) 0 = 0 := by
  exact flatness_deriv_gtFunctional_zero_at_fixedPoint β h (rsQ β h) s
    hβ.le hs (rsQ_pos hβ hh) (rsQ_lt_one hβ hh) (rsQ_fixedPoint β h)

/-!
### The AT estimate after Price's identity
-/

/--
The Cauchy--Schwarz part of the Price-theorem estimate for `tilde g`.

The hypotheses `hY₁` and `hY₂` say that the two fourth-sech moments
have the RS marginal law. In the application these follow from the fact
that both coordinates of the Gaussian pair have law
`N (h, β^2 q)`.
-/
lemma flatness_tildeG_deriv_le_pathAT_of_price
    (β h s : ℝ)
    (hβ : 0 < β) (hh : 0 < h) (hs : 0 ≤ s)
    (D E₁ E₂ : ℝ)
    (hPrice :
      D =
        s * β ^ 2 *
          standardGaussianExpectation (fun z =>
            (Real.cosh (E₁ + z))⁻¹ ^ 2 *
            (Real.cosh (E₂ + z))⁻¹ ^ 2))
    (hCS :
      standardGaussianExpectation (fun z =>
          (Real.cosh (E₁ + z))⁻¹ ^ 2 *
          (Real.cosh (E₂ + z))⁻¹ ^ 2)
        ≤
      standardGaussianExpectation (fun z =>
        (Real.cosh
          (h + β * Real.sqrt (rsQ β h) * z))⁻¹ ^ 4)) :
    D ≤ s * atParameter β h := by
  rw [hPrice]
  rw [atParameter_eq_beta_sq_mul_gaussian_sech_fourth hβ hh]
  simpa only [mul_assoc] using
    (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left hCS (sq_nonneg β)) hs)

private lemma flatness_mul_sech_sq_le_average_sech_fourth
    (x y : ℝ) :
    (Real.cosh x)⁻¹ ^ 2 * (Real.cosh y)⁻¹ ^ 2
      ≤
    ((Real.cosh x)⁻¹ ^ 4 + (Real.cosh y)⁻¹ ^ 4) / 2 := by
  have hx : 0 ≤ (Real.cosh x)⁻¹ ^ 2 := sq_nonneg _
  have hy : 0 ≤ (Real.cosh y)⁻¹ ^ 2 := sq_nonneg _
  nlinarith [sq_nonneg
    ((Real.cosh x)⁻¹ ^ 2 - (Real.cosh y)⁻¹ ^ 2)]

/--
The derivative expression appearing in Price's theorem for `tilde g`.
`Y₁` and `Y₂` are parametrizations of the two coordinates of the
Gaussian pair by an auxiliary probability space.
-/
noncomputable def flatnessTildeGPriceTerm
    (β s : ℝ) (Y₁ Y₂ : ℝ → ℝ) : ℝ :=
  s * β ^ 2 *
    standardGaussianExpectation (fun z =>
      (Real.cosh (Y₁ z))⁻¹ ^ 2 *
      (Real.cosh (Y₂ z))⁻¹ ^ 2)

lemma flatness_tildeGPriceTerm_le_pathAT
    (β h s : ℝ) (Y₁ Y₂ : ℝ → ℝ)
    (hβ : 0 < β) (hh : 0 < h) (hs : 0 ≤ s)
    (hInt₁ :
      Integrable
        (fun z =>
          (Real.cosh (Y₁ z))⁻¹ ^ 4)
        (gaussianReal 0 1))
    (hInt₂ :
      Integrable
        (fun z =>
          (Real.cosh (Y₂ z))⁻¹ ^ 4)
        (gaussianReal 0 1))
    (hY₁ :
      standardGaussianExpectation (fun z =>
        (Real.cosh (Y₁ z))⁻¹ ^ 4)
        =
      standardGaussianExpectation (fun z =>
        (Real.cosh
          (h + β * Real.sqrt (rsQ β h) * z))⁻¹ ^ 4))
    (hY₂ :
      standardGaussianExpectation (fun z =>
        (Real.cosh (Y₂ z))⁻¹ ^ 4)
        =
      standardGaussianExpectation (fun z =>
        (Real.cosh
          (h + β * Real.sqrt (rsQ β h) * z))⁻¹ ^ 4)) :
    flatnessTildeGPriceTerm β s Y₁ Y₂
      ≤ s * atParameter β h := by

  let A : ℝ :=
    standardGaussianExpectation (fun z =>
      (Real.cosh
        (h + β * Real.sqrt (rsQ β h) * z))⁻¹ ^ 4)

  have hprod :
      standardGaussianExpectation (fun z =>
          (Real.cosh (Y₁ z))⁻¹ ^ 2 *
          (Real.cosh (Y₂ z))⁻¹ ^ 2)
        ≤ A := by
    unfold standardGaussianExpectation
    have haverageInt : Integrable
        (fun z =>
          ((Real.cosh (Y₁ z))⁻¹ ^ 4 +
            (Real.cosh (Y₂ z))⁻¹ ^ 4) / 2)
        (gaussianReal 0 1) :=
      hInt₁.add hInt₂ |>.div_const 2
    have hsqMeas₁ : AEStronglyMeasurable
        (fun z => (Real.cosh (Y₁ z))⁻¹ ^ 2)
        (gaussianReal 0 1) := by
      convert Real.continuous_sqrt.comp_aestronglyMeasurable
        hInt₁.aestronglyMeasurable using 1
      funext z
      rw [show (Real.cosh (Y₁ z))⁻¹ ^ 4 =
          ((Real.cosh (Y₁ z))⁻¹ ^ 2) ^ 2 by ring,
        Real.sqrt_sq_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hsqMeas₂ : AEStronglyMeasurable
        (fun z => (Real.cosh (Y₂ z))⁻¹ ^ 2)
        (gaussianReal 0 1) := by
      convert Real.continuous_sqrt.comp_aestronglyMeasurable
        hInt₂.aestronglyMeasurable using 1
      funext z
      rw [show (Real.cosh (Y₂ z))⁻¹ ^ 4 =
          ((Real.cosh (Y₂ z))⁻¹ ^ 2) ^ 2 by ring,
        Real.sqrt_sq_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hprodInt : Integrable
        (fun z =>
          (Real.cosh (Y₁ z))⁻¹ ^ 2 *
          (Real.cosh (Y₂ z))⁻¹ ^ 2)
        (gaussianReal 0 1) := by
      apply haverageInt.mono'
      · exact hsqMeas₁.mul hsqMeas₂
      · filter_upwards [] with z
        rw [Real.norm_eq_abs, abs_of_nonneg
          (mul_nonneg (sq_nonneg _) (sq_nonneg _))]
        exact flatness_mul_sech_sq_le_average_sech_fourth
          (Y₁ z) (Y₂ z)
    calc
      (∫ z,
          (Real.cosh (Y₁ z))⁻¹ ^ 2 *
          (Real.cosh (Y₂ z))⁻¹ ^ 2
          ∂gaussianReal 0 1)
          ≤
        ∫ z,
          ((Real.cosh (Y₁ z))⁻¹ ^ 4 +
            (Real.cosh (Y₂ z))⁻¹ ^ 4) / 2
          ∂gaussianReal 0 1 := by
            apply integral_mono
            · exact hprodInt
            · exact haverageInt
            · intro z
              exact flatness_mul_sech_sq_le_average_sech_fourth
                (Y₁ z) (Y₂ z)

      _ =
        ((∫ z, (Real.cosh (Y₁ z))⁻¹ ^ 4
              ∂gaussianReal 0 1) +
          (∫ z, (Real.cosh (Y₂ z))⁻¹ ^ 4
              ∂gaussianReal 0 1)) / 2 := by
            rw [integral_div]
            rw [integral_add hInt₁ hInt₂]

      _ = A := by
        change
          (standardGaussianExpectation (fun z =>
              (Real.cosh (Y₁ z))⁻¹ ^ 4) +
           standardGaussianExpectation (fun z =>
              (Real.cosh (Y₂ z))⁻¹ ^ 4)) / 2 = A
        rw [hY₁, hY₂]
        simp [A]

  unfold flatnessTildeGPriceTerm
  rw [atParameter_eq_beta_sq_mul_gaussian_sech_fourth hβ hh]

  simpa only [mul_assoc, A] using
    (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left hprod (sq_nonneg β)) hs)

/-- For fixed model parameters and overlap, the GT functional is convex in
its Lagrange multiplier. -/
lemma convexOn_gtFunctional_lam (β h q s v : ℝ) :
    ConvexOn ℝ Set.univ (fun lam => gtFunctional β h q s lam v) := by
  apply convexOn_univ_of_deriv2_nonneg
  · intro lam
    exact (hasDerivAt_gtFunctional β h q s lam v).differentiableAt
  · exact differentiable_deriv_gtFunctional β h q s v
  · intro lam
    simpa only [Function.iterate_succ_apply, Function.iterate_zero_apply] using
      (gt_lambda_derivative_bounds β h q s lam v 0 0 0).2.1

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
## Final theorem
-/


theorem gtFunctional_uniform_quadratic_gap {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam, gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  sorry

end SpinGlass.AT
