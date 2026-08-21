import Lemmas.Psi_continuity
import Lemmas.ATDefs
import Lemmas.GTGauss
import Lemmas.interpolatedAT
import Lemmas.Propertyofg
import Lemmas.Price
import Mathlib.MeasureTheory.Group.IntegralConvolution

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-!
## Branchwise formulas for the GT functional

These formulas mirror the four overlap regimes established in `GTGauss`.
-/

/-! ### The degenerate smart-path endpoint `s = 0` -/

/-- When `s = 0`, every covariance increment in the finite GT recursion
vanishes, so the semigroup solution is just the terminal function. -/
private lemma flatness_gtSemigroupSolution_s_zero
    (β q lam v u x₁ x₂ : ℝ) :
    gtSemigroupSolution β q 0 lam v u x₁ x₂ =
      gtTerminal lam x₁ x₂ := by
  simp [gtSemigroupSolution, gtIncrementScale,
    gtDiagonalStep, gtRankOneStep, standardGaussianExpectation]

/-- At `s = 0` and `lam = 0`, the GT functional is exactly twice the
replica-symmetric path value. -/
lemma flatness_gtFunctional_s_zero_lam_zero
    (β h q v : ℝ) :
    gtFunctional β h q 0 0 v =
      2 * rsPathValue β h q 0 := by
  rw [gtFunctional]
  simp only [sub_zero, one_mul, zero_mul]
  rw [show gtCorrection β q 0 = 0 by simp [gtCorrection]]
  simp only [sub_zero]

  simp_rw [flatness_gtSemigroupSolution_s_zero]
  simp_rw [gtTerminal_zero]

  rw [rsPathValue]
  simp only [zero_mul, zero_div, add_zero]

  have hE :
      standardGaussianExpectation (fun z =>
        Real.log (Real.cosh
          (h + β * Real.sqrt q * z)) +
        Real.log (Real.cosh
          (h + β * Real.sqrt q * z))) =
      2 * standardGaussianExpectation (fun z =>
        Real.log (Real.cosh
          (h + β * Real.sqrt q * z))) := by
    unfold standardGaussianExpectation
    rw [show
      (fun z : ℝ =>
        Real.log (Real.cosh
          (h + β * Real.sqrt q * z)) +
        Real.log (Real.cosh
          (h + β * Real.sqrt q * z))) =
      (fun z : ℝ =>
        2 * Real.log (Real.cosh
          (h + β * Real.sqrt q * z))) by
      funext z
      ring]
    rw [integral_const_mul]

  rw [hE]
  ring

/-- At `s = 0`, the multiplier derivative at `lam = 0` is `q - v`
whenever `q` satisfies the replica-symmetric fixed-point equation. -/
lemma flatness_deriv_gtFunctional_s_zero_lam_zero
    (β h q v : ℝ)
    (hfixed : IsRSFixedPoint β h q) :
    deriv (fun lam => gtFunctional β h q 0 lam v) 0 =
      q - v := by
  rw [deriv_gtFunctional_eq]
  simp only [sub_zero, one_mul]

  have hE :
      standardGaussianExpectation (fun z =>
        deriv (fun lam =>
          gtSemigroupSolution β q 0 lam v 0
            (h + β * Real.sqrt q * z)
            (h + β * Real.sqrt q * z)) 0) =
      q := by
    calc
      standardGaussianExpectation (fun z =>
        deriv (fun lam =>
          gtSemigroupSolution β q 0 lam v 0
            (h + β * Real.sqrt q * z)
            (h + β * Real.sqrt q * z)) 0)
          =
        standardGaussianExpectation (fun z =>
          Real.tanh (h + β * Real.sqrt q * z) ^ 2) := by
            apply congrArg standardGaussianExpectation
            funext z
            let x := h + β * Real.sqrt q * z
            have hfun :
                (fun lam =>
                  gtSemigroupSolution β q 0 lam v 0 x x) =
                (fun lam => gtTerminal lam x x) := by
              funext lam
              exact flatness_gtSemigroupSolution_s_zero
                β q lam v 0 x x
            rw [hfun, deriv_gtTerminal_zero]
            ring
      _ = q := by
        exact hfixed.symm

  rw [hE]

/-- The canonical-overlap specialization of
`flatness_deriv_gtFunctional_s_zero_lam_zero`. -/
lemma flatness_deriv_gtFunctional_s_zero_lam_zero_rsQ
    (β h v : ℝ) :
    deriv
      (fun lam =>
        gtFunctional β h (rsQ β h) 0 lam v) 0 =
      rsQ β h - v := by
  exact flatness_deriv_gtFunctional_s_zero_lam_zero
    β h (rsQ β h) v (rsQ_fixedPoint β h)

/-!
The following helper isolates the routine passage from a pointwise formula
for the GT semigroup solution to the corresponding multiplier derivative of
the GT functional.
-/

private lemma flatness_deriv_gtFunctional_of_solution
    (β h q s lam v : ℝ) (U : ℝ → GTTwoField)
    (hU : ∀ l x₁ x₂,
      gtSemigroupSolution β q s l v 0 x₁ x₂ = U l x₁ x₂) :
    deriv (fun l => gtFunctional β h q s l v) lam =
      standardGaussianExpectation (fun z =>
        deriv (fun l => U l
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)) lam) - v := by
  rw [deriv_gtFunctional_eq]
  apply congrArg (fun x : ℝ => x - v)
  apply congrArg standardGaussianExpectation
  funext z
  congr 1
  funext l
  exact hU l _ _

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
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  simpa using
    flatness_deriv_gtFunctional_of_solution β h q s lam 0
      (fun l =>
        gtDiagonalStep 0 (gtIncrementScale β s 0 q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1)
            (gtTerminal l)))
      (by
        intro l x₁ x₂
        simp [gtSemigroupSolution, hq0])


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
  have hqr : ¬ q ≤ |v| := not_le.mpr hvq
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hv0
  have hqpos : 0 < q := lt_trans hv0 hvq
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hqpos
  exact
    flatness_deriv_gtFunctional_of_solution β h q s lam v
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 |v|) (gtPathSign v)
          (gtDiagonalStep 0
            (gtIncrementScale β s |v| q)
            (gtDiagonalStep 1
              (gtIncrementScale β s q 1)
              (gtTerminal l))))
      (by
        intro l x₁ x₂
        simp [gtSemigroupSolution, hqr, hr0, hq0])


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
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have hrpos : 0 < |v| := lt_of_lt_of_le hq hqv
  have hr0 : ¬ |v| ≤ (0 : ℝ) := not_le.mpr hrpos
  exact
    flatness_deriv_gtFunctional_of_solution β h q s lam v
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q |v|) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s |v| 1)
              (gtTerminal l))))
      (by
        intro l x₁ x₂
        simp [gtSemigroupSolution, hqv, hr0, hq0])


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
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  have h10 : ¬ (1 : ℝ) ≤ 0 := by norm_num
  simp [gtFunctional, gtSemigroupSolution, hv, hq1, hq0, h10]


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
  have hq0 : ¬ q ≤ (0 : ℝ) := not_le.mpr hq
  exact
    flatness_deriv_gtFunctional_of_solution β h q s lam v
      (fun l =>
        gtRankOneStep 0
          (gtIncrementScale β s 0 q) (gtPathSign v)
          (gtRankOneStep (1 / 2)
            (gtIncrementScale β s q 1) (gtPathSign v)
            (gtDiagonalStep 1
              (gtIncrementScale β s 1 1)
              (gtTerminal l))))
      (by
        intro l x₁ x₂
        simp [gtSemigroupSolution, hv, hq1, hq0])


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

/-! #### Identification with `tilde g_s` on negative overlaps -/

/--
For `|v| ≤ q`, the Gaussian correlation
`tilde g_s(v) = E[tanh(Y₁(v)) tanh(Y₂(v))]`.

The four independent standard Gaussians below give both coordinates
variance `β² q`; their covariance is
`β² ((1-s) q + s v)`.
-/
noncomputable def flatnessTildeG
    (β h q s v : ℝ) : ℝ :=
  standardGaussianExpectation (fun z =>
    standardGaussianExpectation (fun z₀ =>
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ =>
          Real.tanh
              (h + β * Real.sqrt ((1 - s) * q) * z
                + gtIncrementScale β s 0 |v| * z₀
                + gtIncrementScale β s |v| q * z₁) *
            Real.tanh
              (h + β * Real.sqrt ((1 - s) * q) * z
                + gtPathSign v *
                    gtIncrementScale β s 0 |v| * z₀
                + gtIncrementScale β s |v| q * z₂)))))

/--
A zero-length mass-`1/2` step does not change the endpoint
multiplier derivative.
-/
@[simp] private lemma flatness_gtHalfStepEndpoint_zero_scale
    (upperScale sign x₁ x₂ : ℝ) :
    gtHalfStepEndpoint 0 upperScale sign x₁ x₂ =
      Real.tanh x₁ * Real.tanh x₂ := by
  unfold gtHalfStepEndpoint
  simp [standardGaussianExpectation, Real.exp_ne_zero]

/--
For every negative overlap `-q ≤ v < 0`, the endpoint multiplier derivative
of the GT functional is `tilde g_s(v) - v`.
-/
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg
    (β h q s v : ℝ)
    (hq : q ∈ Set.Ioo (0 : ℝ) 1)
    (hv : v ∈ Set.Ico (-q) 0) :
    deriv (fun lam => gtFunctional β h q s lam v) 0 =
      flatnessTildeG β h q s v - v := by
  by_cases hvleft : v = -q
  · subst v
    have habs : |(-q : ℝ)| = q := by
      rw [abs_neg, abs_of_pos hq.1]
    have hqabs : q ≤ |(-q : ℝ)| := by
      rw [habs]
    have habs1 : |(-q : ℝ)| < 1 := by
      rw [habs]
      exact hq.2
    rw [flatness_deriv_gtFunctional_zero_q_le_abs_v_lt_one
      β h q s (-q) hq.1 hqabs habs1]
    apply congrArg (fun y : ℝ => y - (-q))
    unfold flatnessTildeG
    rw [habs]
    have hzero : gtIncrementScale β s q q = 0 := by
      simp [gtIncrementScale]
    rw [hzero]
    simp [standardGaussianExpectation]
  · have hvneg : v < 0 := hv.2
    have hvne : v ≠ 0 := ne_of_lt hvneg
    have hv0 : 0 < |v| := by
      exact abs_pos.mpr hvne
    have hminusqv : -q < v := by
      exact lt_of_le_of_ne hv.1 (Ne.symm hvleft)
    have hvq : |v| < q := by
      rw [abs_of_neg hvneg]
      linarith
    simpa [flatnessTildeG] using
      (flatness_deriv_gtFunctional_zero_abs_v_lt_q
        β h q s v hv0 hvq)

/-- Canonical version with `q = rsQ β h`. -/
lemma flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg_rsQ
    (β h s v : ℝ)
    (hβ : 0 < β) (hh : 0 < h)
    (hv : v ∈ Set.Ico (-(rsQ β h)) 0) :
    deriv
        (fun lam =>
          gtFunctional β h (rsQ β h) s lam v) 0 =
      flatnessTildeG β h (rsQ β h) s v - v := by
  exact
    flatness_deriv_gtFunctional_zero_eq_tildeG_sub_neg
      β h (rsQ β h) s v
      ⟨rsQ_pos hβ hh, rsQ_lt_one hβ hh⟩
      hv

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
  let va : NNReal := ⟨a ^ 2, sq_nonneg a⟩ * 1
  let vb : NNReal := ⟨b ^ 2, sq_nonneg b⟩ * 1
  let vc : NNReal := ⟨c ^ 2, sq_nonneg c⟩ * 1
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

/-! #### Identification of the endpoint derivative with `g_s(v) - v` -/

/--
A bounded continuous test function only sees the total variance of two
independent centered Gaussian increments.
-/
private lemma flatness_gaussian_convolution_bounded
    (f : ℝ → ℝ) (hfcont : Continuous f)
    (hfbound : ∀ x, ‖f x‖ ≤ 1)
    (h a b c : ℝ)
    (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun x =>
      standardGaussianExpectation (fun y =>
        f (h + a * x + b * y))) =
    standardGaussianExpectation (fun z =>
      f (h + c * z)) := by
  let va : NNReal := ⟨a ^ 2, sq_nonneg a⟩ * 1
  let vb : NNReal := ⟨b ^ 2, sq_nonneg b⟩ * 1
  let vc : NNReal := ⟨c ^ 2, sq_nonneg c⟩ * 1

  have hma :
      Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1) =
        gaussianReal 0 va := by
    simpa [va] using
      (gaussianReal_map_const_mul
        (μ := 0) (v := (1 : NNReal)) a)

  have hmb :
      Measure.map (fun x : ℝ => b * x) (gaussianReal 0 1) =
        gaussianReal 0 vb := by
    simpa [vb] using
      (gaussianReal_map_const_mul
        (μ := 0) (v := (1 : NNReal)) b)

  have hmc :
      Measure.map (fun x : ℝ => c * x) (gaussianReal 0 1) =
        gaussianReal 0 vc := by
    simpa [vc] using
      (gaussianReal_map_const_mul
        (μ := 0) (v := (1 : NNReal)) c)

  have hv : va + vb = vc := by
    apply NNReal.eq
    simp [va, vb, vc, hc]

  have hfint :
      Integrable (fun z : ℝ => f (h + z))
        (gaussianReal 0 va ∗ gaussianReal 0 vb) := by
    rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    apply Integrable.of_bound (C := 1)
    · exact (hfcont.comp (by fun_prop)).aestronglyMeasurable
    · filter_upwards [] with z
      exact hfbound (h + z)

  have hprod :
      Integrable
        (fun p : ℝ × ℝ => f (h + (p.1 + p.2)))
        ((gaussianReal 0 va).prod (gaussianReal 0 vb)) := by
    rw [Measure.conv] at hfint
    exact (integrable_map_measure hfint.1 (by fun_prop)).mp hfint

  have houter :
      AEStronglyMeasurable
        (fun x : ℝ =>
          ∫ y, f (h + (x + y)) ∂gaussianReal 0 vb)
        (gaussianReal 0 va) :=
    hprod.integral_prod_left.1

  have hinner (x : ℝ) :
      (∫ y, f (h + a * x + b * y) ∂gaussianReal 0 1) =
        ∫ y, f (h + a * x + y) ∂gaussianReal 0 vb := by
    have hm :
        AEStronglyMeasurable
          (fun y : ℝ => f (h + a * x + y))
          (Measure.map (fun y : ℝ => b * y)
            (gaussianReal 0 1)) :=
      (hfcont.comp (by fun_prop)).aestronglyMeasurable
    rw [← hmb, integral_map (by fun_prop) hm]

  have houter_map :
      (∫ x,
          ∫ y, f (h + a * x + y) ∂gaussianReal 0 vb
        ∂gaussianReal 0 1) =
      ∫ x,
          ∫ y, f (h + x + y) ∂gaussianReal 0 vb
        ∂gaussianReal 0 va := by
    have hm :
        AEStronglyMeasurable
          (fun x : ℝ =>
            ∫ y, f (h + (x + y)) ∂gaussianReal 0 vb)
          (Measure.map (fun x : ℝ => a * x)
            (gaussianReal 0 1)) := by
      simpa [hma] using houter
    rw [← hma]
    simpa only [add_assoc] using
      (integral_map (by fun_prop) hm).symm

  unfold standardGaussianExpectation
  calc
    (∫ x,
        ∫ y, f (h + a * x + b * y) ∂gaussianReal 0 1
      ∂gaussianReal 0 1)
        =
      ∫ x,
        ∫ y, f (h + x + y) ∂gaussianReal 0 vb
      ∂gaussianReal 0 va := by
        rw [integral_congr_ae
          (Filter.Eventually.of_forall hinner)]
        exact houter_map

    _ = ∫ z, f (h + z)
          ∂(gaussianReal 0 va ∗ gaussianReal 0 vb) := by
        simpa only [add_assoc] using (integral_conv hfint).symm

    _ = ∫ z, f (h + z) ∂gaussianReal 0 vc := by
        rw [gaussianReal_conv_gaussianReal, hv, zero_add]

    _ = ∫ z, f (h + c * z) ∂gaussianReal 0 1 := by
        rw [← hmc, integral_map (by fun_prop)]
        exact (hfcont.comp (by fun_prop)).aestronglyMeasurable


/--
The lower-branch endpoint profile.  This is
`(H_{a^2} tanh)^2`, written in the same two-Gaussian form
which occurs naturally in the GT derivative.
-/
private noncomputable def flatnessLowerProfile
    (a x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z₁ =>
    standardGaussianExpectation (fun z₂ =>
      Real.tanh (x + a * z₁) *
        Real.tanh (x + a * z₂)))


private lemma flatnessLowerProfile_eq_sq (a x : ℝ) :
    flatnessLowerProfile a x =
      (standardGaussianExpectation (fun z =>
        Real.tanh (x + a * z))) ^ 2 := by
  unfold flatnessLowerProfile standardGaussianExpectation
  calc
    (∫ z₁,
        ∫ z₂,
          Real.tanh (x + a * z₁) *
            Real.tanh (x + a * z₂)
        ∂gaussianReal 0 1
      ∂gaussianReal 0 1)
        =
      ∫ z₁,
        Real.tanh (x + a * z₁) *
          (∫ z₂,
            Real.tanh (x + a * z₂)
            ∂gaussianReal 0 1)
      ∂gaussianReal 0 1 := by
        apply integral_congr_ae
        filter_upwards [] with z₁
        rw [integral_const_mul]

    _ =
      (∫ z, Real.tanh (x + a * z)
        ∂gaussianReal 0 1) ^ 2 := by
        rw [integral_mul_const]
        ring


private lemma flatnessLowerProfile_zero (x : ℝ) :
    flatnessLowerProfile 0 x = Real.tanh x ^ 2 := by
  rw [flatnessLowerProfile_eq_sq]
  simp [standardGaussianExpectation]


/--
Continuity and the bound `|flatnessLowerProfile| ≤ 1` follow directly
from the `GoodFam` package already used for the endpoint computation.
-/
private lemma flatnessLowerProfile_good (a : ℝ) :
    Continuous (flatnessLowerProfile a) ∧
      ∀ x, ‖flatnessLowerProfile a x‖ ≤ 1 := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbase l x + 0 ^ 2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x

  have hF : GTFrame.GoodFam F D :=
    flatness_upper_goodFam 0

  let F₂ := GTFrame.step0 flatnessGauss
    (fun _ : Unit => 0) (fun _ => a) F
  let D₂ := GTFrame.step0 flatnessGauss
    (fun _ : Unit => 0) (fun _ => a) D

  have hF₂ : GTFrame.GoodFam F₂ D₂ :=
    GTFrame.step0_good
      (GTFrame.expMoments_gaussianReal 0 1)
      hF continuous_const continuous_const

  let F₃ := GTFrame.step0 flatnessGauss
    (fun _ : Unit => a) (fun _ => 0) F₂
  let D₃ := GTFrame.step0 flatnessGauss
    (fun _ : Unit => a) (fun _ => 0) D₂

  have hF₃ : GTFrame.GoodFam F₃ D₃ :=
    GTFrame.step0_good
      (GTFrame.expMoments_gaussianReal 0 1)
      hF₂ continuous_const continuous_const

  have hdiag : Continuous (fun x : ℝ => (x, x)) := by
    fun_prop

  have heq (x : ℝ) :
      D₃ () 0 (x, x) = flatnessLowerProfile a x := by
    simp [D₃, D₂, D, GTFrame.step0, flatnessGauss,
      flatnessLowerProfile, standardGaussianExpectation,
      flatness_fLbaseD_zero]

  constructor
  · have hc := (hF₃.contD_pt () 0).comp hdiag
    have hfun :
        flatnessLowerProfile a =
          fun x => D₃ () 0 (x, x) := by
      funext x
      exact (heq x).symm
    rw [hfun]
    exact hc

  · intro x
    have hb := hF₃.bddD () 0 (x, x)
    rw [Real.norm_eq_abs, ← heq x]
    exact hb


/--
The upper-branch tilted profile written as the endpoint derivative
of the half-mass GT step.
-/
private noncomputable def flatnessUpperProfile
    (a b x : ℝ) : ℝ :=
  gtHalfStepEndpoint a b 1 x x


private lemma flatnessUpperProfile_good (a b : ℝ) :
    Continuous (flatnessUpperProfile a b) ∧
      ∀ x, ‖flatnessUpperProfile a b x‖ ≤ 1 := by
  let F : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbase l x + b ^ 2
  let D : Unit → ℝ → ℝ × ℝ → ℝ :=
    fun _ l x => GTFrame.fLbaseD l x

  have hF : GTFrame.GoodFam F D :=
    flatness_upper_goodFam b

  let FH := GTFrame.stepM flatnessGauss (1 / 2)
    (fun _ : Unit => a) (fun _ => a) F
  let DH := GTFrame.stepMD flatnessGauss (1 / 2)
    (fun _ : Unit => a) (fun _ => a) F D

  have hFH : GTFrame.GoodFam FH DH :=
    GTFrame.stepM_good
      (GTFrame.expMoments_gaussianReal 0 1)
      hF (by norm_num) continuous_const continuous_const

  have hdiag : Continuous (fun x : ℝ => (x, x)) := by
    fun_prop

  have heq (x : ℝ) :
      DH () 0 (x, x) = flatnessUpperProfile a b x := by
    simp [DH, F, D, GTFrame.stepMD, flatnessGauss,
      standardGaussianExpectation, flatness_fLbaseD_zero,
      flatnessUpperProfile, gtHalfStepEndpoint]

  constructor
  · have hc := (hFH.contD_pt () 0).comp hdiag
    have hfun :
        flatnessUpperProfile a b =
          fun x => DH () 0 (x, x) := by
      funext x
      exact (heq x).symm
    rw [hfun]
    exact hc

  · intro x
    have hb := hFH.bddD () 0 (x, x)
    rw [Real.norm_eq_abs, ← heq x]
    exact hb


private lemma flatness_incrementScale_eq_sqrt_product
    (β s lower upper : ℝ) (hs : 0 ≤ s) :
    gtIncrementScale β s lower upper =
      β * Real.sqrt (s * (upper - lower)) := by
  unfold gtIncrementScale
  rw [Real.sqrt_mul hs]
  ring


/--
Variance identity for the two outer Gaussian increments occurring
in the GT formula.
-/
private lemma flatness_outer_variance_sq
    (β q s v : ℝ)
    (hβ : 0 ≤ β) (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hq : 0 ≤ q) (hv : 0 ≤ v) :
    (Real.sqrt
      (β ^ 2 * ((1 - s) * q + s * v))) ^ 2 =
      (β * Real.sqrt ((1 - s) * q)) ^ 2 +
        (gtIncrementScale β s 0 v) ^ 2 := by
  have h1s : 0 ≤ 1 - s := sub_nonneg.mpr hs1
  have hbase :
      0 ≤ (1 - s) * q + s * v :=
    add_nonneg (mul_nonneg h1s hq) (mul_nonneg hs0 hv)
  have hvar :
      0 ≤ β ^ 2 * ((1 - s) * q + s * v) :=
    mul_nonneg (sq_nonneg β) hbase

  rw [Real.sq_sqrt hvar]
  rw [mul_pow, Real.sq_sqrt (mul_nonneg h1s hq)]
  rw [gtIncrementScale_sq_of_nonneg
    β s 0 v hβ hs0 hv]
  ring


/--
For `0 ≤ v ≤ q`, rewrite the scalar order parameter in exactly the
Gaussian form produced by the GT endpoint derivative.
-/
private lemma flatness_scalarOrderParameter_lower_eq
    (β h q s v : ℝ)
    (hβ : 0 ≤ β)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 ≤ q)
    (hv : v ∈ Set.Icc (0 : ℝ) q) :
    scalarOrderParameter β h q s v =
      standardGaussianExpectation (fun z =>
        standardGaussianExpectation (fun z₀ =>
          flatnessLowerProfile
            (gtIncrementScale β s v q)
            (h + β * Real.sqrt ((1 - s) * q) * z +
              gtIncrementScale β s 0 v * z₀))) := by
  have hmax :
      max (q - v) 0 = q - v :=
    max_eq_left (sub_nonneg.mpr hv.2)

  have hd :
      gtIncrementScale β s v q =
        β * Real.sqrt (s * (q - v)) :=
    flatness_incrementScale_eq_sqrt_product β s v q hs.1

  have hpsi (x : ℝ) :
      scalarPsiX β q s v x ^ 2 =
        flatnessLowerProfile
          (gtIncrementScale β s v q) x := by
    unfold scalarPsiX
    rw [hmax, hd]
    exact (flatnessLowerProfile_eq_sq _ x).symm

  have hsq :=
    flatness_outer_variance_sq
      β q s v hβ hs.1 hs.2 hq hv.1

  have hgood :=
    flatnessLowerProfile_good
      (gtIncrementScale β s v q)

  have hconv :=
    flatness_gaussian_convolution_bounded
      (flatnessLowerProfile
        (gtIncrementScale β s v q))
      hgood.1 hgood.2
      h
      (β * Real.sqrt ((1 - s) * q))
      (gtIncrementScale β s 0 v)
      (Real.sqrt
        (β ^ 2 * ((1 - s) * q + s * v)))
      hsq

  unfold scalarOrderParameter localFieldExpectation
  rw [if_pos hv.2]
  unfold heatSemigroup
  simp_rw [hpsi]
  exact hconv.symm


/--
For `q < v`, rewrite the scalar order parameter in exactly the
tilted-semigroup form produced by the GT endpoint derivative.
-/
private lemma flatness_scalarOrderParameter_upper_eq
    (β h q s v : ℝ)
    (hβ : 0 ≤ β)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 ≤ q)
    (hqv : q < v) :
    scalarOrderParameter β h q s v =
      standardGaussianExpectation (fun z =>
        standardGaussianExpectation (fun z₀ =>
          tiltedHeatSemigroup
            (s * β ^ 2 * (v - q))
            (fun y => Real.tanh y ^ 2)
            (h + β * Real.sqrt ((1 - s) * q) * z +
              gtIncrementScale β s 0 q * z₀))) := by
  have hmax :
      max (q - v) 0 = 0 :=
    max_eq_right (sub_nonpos.mpr hqv.le)

  have hpsi (x : ℝ) :
      scalarPsiX β q s v x = Real.tanh x := by
    unfold scalarPsiX
    rw [hmax]
    simp [standardGaussianExpectation]

  let a : ℝ := gtIncrementScale β s q v

  have ha0 : 0 ≤ a := by
    dsimp [a]
    exact gtIncrementScale_nonneg β s q v hβ

  have hasq :
      a ^ 2 = s * β ^ 2 * (v - q) := by
    dsimp [a]
    exact gtIncrementScale_sq_of_nonneg
      β s q v hβ hs.1 hqv.le

  have hprofile (x : ℝ) :
      tiltedHeatSemigroup
        (s * β ^ 2 * (v - q))
        (fun y => Real.tanh y ^ 2) x =
      flatnessUpperProfile a 0 x := by
    symm
    unfold flatnessUpperProfile
    rw [gtHalfStepEndpoint_diagonal_eq_tilted
      a 0 x ha0, hasq]

  have hgood := flatnessUpperProfile_good a 0

  have hfcont :
      Continuous (fun x =>
        tiltedHeatSemigroup
          (s * β ^ 2 * (v - q))
          (fun y => Real.tanh y ^ 2) x) := by
    have heq :
        (fun x =>
          tiltedHeatSemigroup
            (s * β ^ 2 * (v - q))
            (fun y => Real.tanh y ^ 2) x) =
        flatnessUpperProfile a 0 := by
      funext x
      exact hprofile x
    rw [heq]
    exact hgood.1

  have hfbound :
      ∀ x,
        ‖tiltedHeatSemigroup
          (s * β ^ 2 * (v - q))
          (fun y => Real.tanh y ^ 2) x‖ ≤ 1 := by
    intro x
    rw [hprofile x]
    exact hgood.2 x

  have hsq0 :=
    flatness_outer_variance_sq
      β q s q hβ hs.1 hs.2 hq hq

  have hcollapse :
      (1 - s) * q + s * q = q := by
    ring

  have hsq :
      (Real.sqrt (β ^ 2 * q)) ^ 2 =
        (β * Real.sqrt ((1 - s) * q)) ^ 2 +
          (gtIncrementScale β s 0 q) ^ 2 := by
    simpa [hcollapse] using hsq0

  have hconv :=
    flatness_gaussian_convolution_bounded
      (fun x =>
        tiltedHeatSemigroup
          (s * β ^ 2 * (v - q))
          (fun y => Real.tanh y ^ 2) x)
      hfcont hfbound
      h
      (β * Real.sqrt ((1 - s) * q))
      (gtIncrementScale β s 0 q)
      (Real.sqrt (β ^ 2 * q))
      hsq

  unfold scalarOrderParameter localFieldExpectation
  rw [if_neg (not_le_of_gt hqv)]
  simp_rw [hpsi]
  unfold heatSemigroup
  exact hconv.symm


/--
For every positive overlap `0 ≤ v ≤ 1`, the endpoint multiplier
derivative of the GT functional is `g_s(v) - v`.
-/
lemma flatness_deriv_gtFunctional_zero_eq_scalarOrderParameter_sub
    (β h q s v : ℝ)
    (hβ : 0 ≤ β)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : q ∈ Set.Ioo (0 : ℝ) 1)
    (hv : v ∈ Set.Icc (0 : ℝ) 1) :
    deriv (fun lam => gtFunctional β h q s lam v) 0 =
      scalarOrderParameter β h q s v - v := by

  by_cases hvq : v < q

  · by_cases hvzero : v = 0

    · subst v
      rw [flatness_deriv_gtFunctional_zero_abs_v_eq_zero
        β h q s 0 hq.1 abs_zero]
      rw [flatness_scalarOrderParameter_lower_eq
        β h q s 0 hβ hs hq.1.le
        ⟨le_rfl, hq.1.le⟩]
      simp [flatnessLowerProfile, gtIncrementScale,
        standardGaussianExpectation]

    · have hvpos : 0 < v :=
        lt_of_le_of_ne hv.1 (Ne.symm hvzero)

      rw [flatness_deriv_gtFunctional_zero_abs_v_lt_q
        β h q s v
        (by simpa [abs_of_nonneg hv.1] using hvpos)
        (by simpa [abs_of_nonneg hv.1] using hvq)]

      rw [flatness_scalarOrderParameter_lower_eq
        β h q s v hβ hs hq.1.le
        ⟨hv.1, hvq.le⟩]

      simp [flatnessLowerProfile,
        abs_of_nonneg hv.1, gtPathSign, hv.1]

  · have hqv : q ≤ v := le_of_not_gt hvq

    by_cases hvqeq : v = q

    · subst v

      rw [flatness_deriv_gtFunctional_zero_q_le_v_lt_one
        β h q s q hβ hs.1 hq.1 le_rfl hq.2]

      rw [flatness_scalarOrderParameter_lower_eq
        β h q s q hβ hs hq.1.le
        ⟨hq.1.le, le_rfl⟩]

      simp [flatness_tiltedHeatSemigroup_zero,
        flatnessLowerProfile_zero, gtIncrementScale]

    · have hqvlt : q < v :=
        lt_of_le_of_ne hqv (Ne.symm hvqeq)

      by_cases hvone : v = 1

      · subst v

        rw [flatness_deriv_gtFunctional_zero_v_eq_one
          β h q s hβ hs.1 hq.1 hq.2.le]

        rw [flatness_scalarOrderParameter_upper_eq
          β h q s 1 hβ hs hq.1.le hq.2]

      · have hvlt1 : v < 1 :=
          lt_of_le_of_ne hv.2 hvone

        rw [flatness_deriv_gtFunctional_zero_q_le_v_lt_one
          β h q s v hβ hs.1 hq.1 hqv hvlt1]

        rw [flatness_scalarOrderParameter_upper_eq
          β h q s v hβ hs hq.1.le hqvlt]

/--
For the canonical replica-symmetric fixed point and every `0 ≤ v ≤ 1`,
`∂_lam GT_s(0,v) = g_s(v) - v`.
-/
lemma flatness_deriv_gtFunctional_zero_eq_g_sub
    (β h s v : ℝ)
    (hβ : 0 < β) (hh : 0 < h)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ Set.Icc (0 : ℝ) 1) :
    deriv
        (fun lam =>
          gtFunctional β h (rsQ β h) s lam v) 0 =
      scalarOrderParameterCorrect β h s v - v := by
  unfold scalarOrderParameterCorrect
  exact
    flatness_deriv_gtFunctional_zero_eq_scalarOrderParameter_sub
      β h (rsQ β h) s v
      hβ.le hs
      ⟨rsQ_pos hβ hh, rsQ_lt_one hβ hh⟩
      hv

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
  rw [flatness_deriv_gtFunctional_zero_eq_g_sub
    β h s (rsQ β h) hβ hh hs (rsQ_mem_Icc β h)]
  exact sub_eq_zero.mpr (scalarOrderParameterCorrect_at_rsQ hβ s)

/-- The sign of the endpoint multiplier derivative follows from the scalar
order-parameter crossing theorem. -/
lemma flatness_deriv_gtFunctional_zero_sign
    {K : Set (ℝ × ℝ)} (data : UniformATData K) {β h s : ℝ}
    (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    (∀ v ∈ Set.Ico (0 : ℝ) (rsQ β h),
      0 < deriv (fun lam => gtFunctional β h (rsQ β h) s lam v) 0) ∧
    deriv (fun lam => gtFunctional β h (rsQ β h) s lam (rsQ β h)) 0 = 0 ∧
    (∀ v ∈ Set.Ioc (rsQ β h) 1,
      deriv (fun lam => gtFunctional β h (rsQ β h) s lam v) 0 < 0) := by
  obtain ⟨hlower, hzero, hupper⟩ :=
    scalarOrderParameterCorrect_sign data hp hs
  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp
  have hq : rsQ β h ∈ Set.Icc (0 : ℝ) 1 := rsQ_mem_Icc β h
  refine ⟨?_, ?_, ?_⟩
  · intro v hv
    rw [flatness_deriv_gtFunctional_zero_eq_g_sub β h s v hβ hh hs
      ⟨hv.1, hv.2.le.trans hq.2⟩]
    exact hlower v hv
  · rw [flatness_deriv_gtFunctional_zero_eq_g_sub β h s (rsQ β h) hβ hh hs
      (rsQ_mem_Icc β h)]
    exact hzero
  · intro v hv
    rw [flatness_deriv_gtFunctional_zero_eq_g_sub β h s v hβ hh hs
      ⟨hq.1.trans hv.1.le, hv.2⟩]
    exact hupper v hv

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

/-- Uniform linear separation of the multiplier derivative from zero near the
replica-symmetric overlap.  Shrinking the neighborhood by `qmin / 2` ensures
that it lies in the nonnegative-overlap branch. -/
lemma flatness_local_deriv_linear_separation
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c ε : ℝ, 0 < c ∧ 0 < ε ∧
      ∀ {β h s v : ℝ},
        (β, h) ∈ K →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        |v - rsQ β h| ≤ ε →
        c * |v - rsQ β h| ≤
          |deriv
            (fun lam =>
              gtFunctional β h (rsQ β h) s lam v) 0| := by
  obtain ⟨c, ε, hc, hε, hsep⟩ :=
    scalarOrderParameterCorrect_linear_separation data

  let ε₀ : ℝ := min ε (data.qmin / 2)

  have hε₀ : 0 < ε₀ := by
    dsimp [ε₀]
    apply lt_min
    · exact hε
    · linarith [data.qmin_pos]

  refine ⟨c, ε₀, hc, hε₀, ?_⟩

  intro β h s v hp hs hv hnear

  have hβ : 0 < β := by
    simpa using data.β_pos (β, h) hp

  have hh : 0 < h := by
    simpa using data.h_pos (β, h) hp

  have hqmin : data.qmin ≤ rsQ β h := by
    simpa using data.q_lower (β, h) hp

  have hnear_qmin :
      |v - rsQ β h| ≤ data.qmin / 2 := by
    exact hnear.trans (min_le_right _ _)

  have hlo :
      -(data.qmin / 2) ≤ v - rsQ β h :=
    (abs_le.mp hnear_qmin).1

  have hv0 : 0 ≤ v := by
    nlinarith [data.qmin_pos]

  have hv01 : v ∈ Icc (0 : ℝ) 1 :=
    ⟨hv0, hv.2⟩

  rw [flatness_deriv_gtFunctional_zero_eq_g_sub
    β h s v hβ hh hs hv01]

  exact hsep hp hs hv01
    (hnear.trans (min_le_left _ _))

/-- Global second-order Taylor upper bound in the multiplier. -/
lemma flatness_gtFunctional_taylor_upper
    (β h q s v lam : ℝ) :
    gtFunctional β h q s lam v
      ≤
    gtFunctional β h q s 0 v
      + deriv (fun l => gtFunctional β h q s l v) 0 * lam
      + (5 / 4 : ℝ) * lam ^ 2 := by
  let F : ℝ → ℝ :=
    fun l => gtFunctional β h q s l v

  let G : ℝ → ℝ :=
    fun l => F l - (5 / 4 : ℝ) * l ^ 2

  have hFdiff : Differentiable ℝ F := by
    intro x
    dsimp [F]
    exact
      (hasDerivAt_gtFunctional β h q s x v).differentiableAt

  have hGdiff : Differentiable ℝ G := by
    intro x
    dsimp [G]
    exact
      (hFdiff x).sub
        (by fun_prop)

  have hGderiv (x : ℝ) :
      deriv G x =
        deriv F x - (5 / 2 : ℝ) * x := by
    have hF :
        HasDerivAt F (deriv F x) x :=
      (hFdiff x).hasDerivAt

    have hquad :
        HasDerivAt
          (fun y : ℝ => (5 / 4 : ℝ) * y ^ 2)
          ((5 / 2 : ℝ) * x) x := by
      have hsq : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
        simpa using hasDerivAt_pow 2 x
      exact (hsq.const_mul (5 / 4 : ℝ)).congr_deriv (by ring)

    exact (hF.sub hquad).deriv

  have hFderivDiff :
      Differentiable ℝ (deriv F) := by
    simpa [F] using
      differentiable_deriv_gtFunctional β h q s v

  have hGderivDiff :
      Differentiable ℝ (deriv G) := by
    have hfun :
        deriv G =
          fun x => deriv F x - (5 / 2 : ℝ) * x := by
      funext x
      exact hGderiv x

    rw [hfun]
    exact hFderivDiff.sub (by fun_prop)

  have hGsecond (x : ℝ) :
      deriv (deriv G) x ≤ 0 := by
    have hfun :
        deriv G =
          fun y => deriv F y - (5 / 2 : ℝ) * y := by
      funext y
      exact hGderiv y

    have hD :
        HasDerivAt
          (deriv F)
          (deriv (deriv F) x) x :=
      (hFderivDiff x).hasDerivAt

    have hlin :
        HasDerivAt
          (fun y : ℝ => (5 / 2 : ℝ) * y)
          (5 / 2 : ℝ) x := by
      simpa using (hasDerivAt_id x).const_mul (5 / 2 : ℝ)

    have heq :
        deriv (deriv G) x =
          deriv (deriv F) x - (5 / 2 : ℝ) := by
      rw [hfun]
      exact (hD.sub hlin).deriv

    rw [heq]

    have hbound :=
      (gt_lambda_derivative_bounds
        β h q s x v 0 0 0).2.2

    change
      deriv (deriv F) x ≤ (5 / 2 : ℝ)
      at hbound

    linarith

  have hconc :
      ConcaveOn ℝ Set.univ G := by
    apply concaveOn_univ_of_deriv2_nonpos
    · exact hGdiff
    · exact hGderivDiff
    · intro x
      simpa only
        [Function.iterate_succ_apply,
         Function.iterate_zero_apply]
        using hGsecond x

  have hGzero :
      HasDerivAt G (deriv F 0) 0 := by
    have hz : deriv G 0 = deriv F 0 := by
      simpa using hGderiv 0
    rw [← hz]
    exact (hGdiff 0).hasDerivAt

  by_cases hlam0 : lam = 0
  · subst lam
    simp

  by_cases hlam : 0 < lam
  · have hslope :=
      hconc.slope_le_of_hasDerivAt
        (Set.mem_univ 0)
        (Set.mem_univ lam)
        hlam
        hGzero

    rw [slope_def_field] at hslope
    simp only [sub_zero] at hslope

    have hmul :=
      (div_le_iff₀ hlam).mp hslope

    dsimp [G, F] at hmul ⊢
    nlinarith

  · have hlamle : lam ≤ 0 :=
      le_of_not_gt hlam

    have hlamneg : lam < 0 :=
      lt_of_le_of_ne hlamle hlam0

    have hslope :=
      hconc.le_slope_of_hasDerivAt
        (Set.mem_univ lam)
        (Set.mem_univ 0)
        hlamneg
        hGzero

    rw [slope_def_field] at hslope

    have hden : 0 < 0 - lam := by
      linarith

    have hmul :=
      (le_div_iff₀ hden).mp hslope

    dsimp [G, F] at hmul ⊢
    nlinarith

/-- A linear lower bound on the multiplier derivative yields a quadratic
improvement after optimizing a single explicit multiplier. -/
lemma flatness_quadratic_gap_of_deriv_gap
    (β h q s v c : ℝ)
    (hc : 0 < c)
    (hzero :
      gtFunctional β h q s 0 v
        ≤ 2 * rsPathValue β h q s)
    (hgap :
      c * |v - q| ≤
        |deriv
          (fun l => gtFunctional β h q s l v) 0|) :
    ∃ lam,
      gtFunctional β h q s lam v
        ≤
      2 * rsPathValue β h q s
        - (c ^ 2 / 5) * (v - q) ^ 2 := by
  let F : ℝ → ℝ :=
    fun l => gtFunctional β h q s l v

  let d : ℝ := deriv F 0
  let lam : ℝ := -(2 / 5 : ℝ) * d

  have ht :=
    flatness_gtFunctional_taylor_upper
      β h q s v lam

  change
    F lam ≤
      F 0 + d * lam + (5 / 4 : ℝ) * lam ^ 2
    at ht

  have hopt :
      d * lam + (5 / 4 : ℝ) * lam ^ 2
        = -(d ^ 2) / 5 := by
    dsimp [lam]
    ring

  have ht' : F lam ≤ F 0 - d ^ 2 / 5 := by
    nlinarith [ht, hopt]

  have hgap' :
      c * |v - q| ≤ |d| := by
    simpa [d, F] using hgap

  have hsq :
      c ^ 2 * (v - q) ^ 2 ≤ d ^ 2 := by
    have hmul :=
      mul_self_le_mul_self
        (mul_nonneg hc.le (abs_nonneg (v - q)))
        hgap'

    calc
      c ^ 2 * (v - q) ^ 2
          =
        (c * |v - q|) * (c * |v - q|) := by
          nlinarith [sq_abs (v - q)]
      _ ≤ |d| * |d| := hmul
      _ = d ^ 2 := by
          nlinarith [sq_abs d]

  have hloss :
      -(d ^ 2) / 5
        ≤
      -(c ^ 2 / 5) * (v - q) ^ 2 := by
    nlinarith

  refine ⟨lam, ?_⟩

  have hzero' :
      F 0 ≤ 2 * rsPathValue β h q s := by
    simpa [F] using hzero

  nlinarith [ht']

/-- The squared distance between any two admissible overlaps is at most four. -/
lemma sub_sq_le_four_of_overlap
    {q v : ℝ}
    (hq : q ∈ Icc (0 : ℝ) 1)
    (hv : v ∈ Icc (-1 : ℝ) 1) :
    (v - q) ^ 2 ≤ 4 := by
  have hleft : -2 ≤ v - q := by
    linarith [hv.1, hq.2]

  have hright : v - q ≤ 2 := by
    linarith [hv.2, hq.1]

  have hprod :
      0 ≤ ((v - q) + 2) * (2 - (v - q)) := by
    exact mul_nonneg
      (by linarith)
      (by linarith)

  nlinarith

/-- Deterministic assembly of a local quadratic estimate and a fixed strict
gap away from the replica-symmetric overlap. -/
lemma gtFunctional_uniform_quadratic_gap_of_local_and_away
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K)
    {c₀ ε κ : ℝ}
    (hc₀ : 0 < c₀)
    (_hε : 0 < ε)
    (hκ : 0 < κ)
    (hlocal :
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        |v - q| ≤ ε →
        ∃ lam,
          gtFunctional β h q s lam v ≤
            2 * rsPathValue β h q s
              - c₀ * (v - q) ^ 2)
    (haway :
      ∀ {β h q s v : ℝ},
        (β, h) ∈ K →
        q = rsQ β h →
        s ∈ Icc (0 : ℝ) 1 →
        v ∈ Icc (-1 : ℝ) 1 →
        ε ≤ |v - q| →
        ∃ lam,
          gtFunctional β h q s lam v ≤
            2 * rsPathValue β h q s - κ) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K →
      q = rsQ β h →
      s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam,
        gtFunctional β h q s lam v ≤
          2 * rsPathValue β h q s
            - c * (v - q) ^ 2 := by
  let c : ℝ := min c₀ (κ / 4)

  have hc : 0 < c := by
    dsimp [c]
    exact lt_min hc₀ (div_pos hκ (by norm_num))

  refine ⟨c, hc, ?_⟩

  intro β h q s v hp hq hs hv

  have hqIcc : q ∈ Icc (0 : ℝ) 1 := by
    rw [hq]
    exact rsQ_mem_Icc β h

  by_cases hnear : |v - q| ≤ ε
  · obtain ⟨lam, hlam⟩ :=
      hlocal hp hq hs hv hnear

    refine ⟨lam, hlam.trans ?_⟩

    have hc_le : c ≤ c₀ := by
      dsimp [c]
      exact min_le_left _ _

    have hmul :
        c * (v - q) ^ 2
          ≤ c₀ * (v - q) ^ 2 :=
      mul_le_mul_of_nonneg_right
        hc_le (sq_nonneg (v - q))

    linarith

  · have hfar : ε ≤ |v - q| := by
      exact (lt_of_not_ge hnear).le

    obtain ⟨lam, hlam⟩ :=
      haway hp hq hs hv hfar

    refine ⟨lam, hlam.trans ?_⟩

    have hsq :
        (v - q) ^ 2 ≤ 4 :=
      sub_sq_le_four_of_overlap hqIcc hv

    have hc_le : c ≤ κ / 4 := by
      dsimp [c]
      exact min_le_right _ _

    have hprod :
        c * (v - q) ^ 2 ≤ c * 4 :=
      mul_le_mul_of_nonneg_left hsq hc.le

    have hc4 : c * 4 ≤ κ := by
      nlinarith

    nlinarith


/-!
## The GT functional at zero multiplier

At `lam = 0` the two-replica terminal condition decouples into a sum of
one-replica terms and every step of the finite GT recursion acts on the two
fields separately.  For a nonnegative overlap the resulting value is exactly
twice the replica-symmetric path value.
-/

private lemma flatness_integrable_exp_shift (a c : ℝ) :
    Integrable (fun z : ℝ => Real.exp (a + c * z)) (gaussianReal 0 1) := by
  have hfun : (fun z : ℝ => Real.exp (a + c * z))
      = fun z : ℝ => Real.exp a * Real.exp (c * z) := by
    funext z
    rw [Real.exp_add]
  rw [hfun]
  exact (integrable_exp_mul_gaussianReal c).const_mul _

private lemma flatness_gaussianExpectation_exp (a c : ℝ) :
    standardGaussianExpectation (fun z => Real.exp (a + c * z))
      = Real.exp (a + c ^ 2 / 2) := by
  have hmgf := congrFun (mgf_id_gaussianReal (μ := 0) (v := 1)) c
  simp only [mgf, id_eq, zero_mul, NNReal.coe_one, one_mul, zero_add] at hmgf
  have hfun : (fun z : ℝ => Real.exp (a + c * z))
      = fun z : ℝ => Real.exp a * Real.exp (c * z) := by
    funext z
    rw [Real.exp_add]
  unfold standardGaussianExpectation
  rw [hfun, integral_const_mul, hmgf, ← Real.exp_add]

/-- The Gaussian expectation of a shifted hyperbolic cosine. -/
private lemma flatness_gaussianExpectation_cosh (y c : ℝ) :
    standardGaussianExpectation (fun z => Real.cosh (y + c * z))
      = Real.cosh y * Real.exp (c ^ 2 / 2) := by
  have hfun : (fun z : ℝ => Real.cosh (y + c * z))
      = fun z : ℝ => (Real.exp (y + c * z) + Real.exp (-y + (-c) * z)) / 2 := by
    funext z
    have hneg : -(y + c * z) = -y + (-c) * z := by ring
    rw [Real.cosh_eq, hneg]
  have h1 := flatness_gaussianExpectation_exp y c
  have h2 := flatness_gaussianExpectation_exp (-y) (-c)
  unfold standardGaussianExpectation at h1 h2 ⊢
  rw [hfun, integral_div,
    integral_add (flatness_integrable_exp_shift y c)
      (flatness_integrable_exp_shift (-y) (-c)), h1, h2, Real.cosh_eq, neg_sq,
    Real.exp_add, Real.exp_add]
  ring

/-- `log ∘ cosh` is continuous. -/
private lemma flatness_continuous_logCosh :
    Continuous (fun x : ℝ => Real.log (Real.cosh x)) := by
  rw [continuous_iff_continuousAt]
  intro x
  exact (Real.continuousAt_log (Real.cosh_pos x).ne').comp
    Real.continuous_cosh.continuousAt

private lemma flatness_logCosh_nonneg (x : ℝ) : 0 ≤ Real.log (Real.cosh x) :=
  Real.log_nonneg (Real.one_le_cosh x)

private lemma flatness_logCosh_le_abs (x : ℝ) : Real.log (Real.cosh x) ≤ |x| := by
  have hle : Real.cosh x ≤ Real.exp |x| := by
    rw [Real.cosh_eq]
    have h1 : Real.exp x ≤ Real.exp |x| := Real.exp_le_exp.mpr (le_abs_self x)
    have h2 : Real.exp (-x) ≤ Real.exp |x| := Real.exp_le_exp.mpr (neg_le_abs x)
    linarith
  calc Real.log (Real.cosh x)
      ≤ Real.log (Real.exp |x|) := Real.log_le_log (Real.cosh_pos x) hle
    _ = |x| := Real.log_exp _

private lemma flatness_integrable_abs_gaussian (m : ℝ) (v : NNReal) :
    Integrable (fun z : ℝ => |z|) (gaussianReal m v) := by
  have hid : Integrable (fun z : ℝ => z) (gaussianReal m v) := by
    simpa using memLp_one_iff_integrable.mp
      (memLp_id_gaussianReal (μ := m) (v := v) 1)
  exact hid.abs

/-- Integrability of an affinely shifted `log ∘ cosh` against a Gaussian. -/
private lemma flatness_integrable_logCosh (x c m : ℝ) (v : NNReal) :
    Integrable (fun z : ℝ => Real.log (Real.cosh (x + c * z))) (gaussianReal m v) := by
  refine Integrable.mono' (g := fun z : ℝ => |x| + |c| * |z|)
    ((integrable_const |x|).add
      ((flatness_integrable_abs_gaussian m v).const_mul |c|)) ?_ ?_
  · exact (flatness_continuous_logCosh.comp (by fun_prop)).aestronglyMeasurable
  · filter_upwards [] with z
    have h1 : ‖Real.log (Real.cosh (x + c * z))‖ ≤ |x + c * z| := by
      rw [Real.norm_eq_abs, abs_of_nonneg (flatness_logCosh_nonneg _)]
      exact flatness_logCosh_le_abs _
    have h2 : |x + c * z| ≤ |x| + |c| * |z| := by
      calc |x + c * z| ≤ |x| + |c * z| := abs_add_le x (c * z)
        _ = |x| + |c| * |z| := by rw [abs_mul]
    exact h1.trans h2

/-- Two independent Gaussian shifts only see the total variance. -/
private lemma flatness_gaussian_convolution_logCosh
    (h a b c : ℝ) (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun x =>
      standardGaussianExpectation (fun y =>
        Real.log (Real.cosh (h + a * x + b * y)))) =
    standardGaussianExpectation (fun z => Real.log (Real.cosh (h + c * z))) := by
  set f : ℝ → ℝ := fun t => Real.log (Real.cosh t) with hf
  have hfcont : Continuous f := flatness_continuous_logCosh
  let va : NNReal := ⟨a ^ 2, sq_nonneg a⟩ * 1
  let vb : NNReal := ⟨b ^ 2, sq_nonneg b⟩ * 1
  let vc : NNReal := ⟨c ^ 2, sq_nonneg c⟩ * 1
  have hma : Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1) = gaussianReal 0 va := by
    simpa [va] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) a)
  have hmb : Measure.map (fun x : ℝ => b * x) (gaussianReal 0 1) = gaussianReal 0 vb := by
    simpa [vb] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) b)
  have hmc : Measure.map (fun x : ℝ => c * x) (gaussianReal 0 1) = gaussianReal 0 vc := by
    simpa [vc] using (gaussianReal_map_const_mul (μ := 0) (v := (1 : NNReal)) c)
  have hv : va + vb = vc := by
    apply NNReal.eq
    simp [va, vb, vc, hc]
  have hfint : Integrable (fun z : ℝ => f (h + z))
      (gaussianReal 0 va ∗ gaussianReal 0 vb) := by
    rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    simpa [hf] using flatness_integrable_logCosh h 1 0 vc
  have hprod : Integrable (fun p : ℝ × ℝ => f (h + (p.1 + p.2)))
      ((gaussianReal 0 va).prod (gaussianReal 0 vb)) := by
    rw [Measure.conv] at hfint
    exact (integrable_map_measure hfint.1 (by fun_prop)).mp hfint
  have houter : AEStronglyMeasurable
      (fun x : ℝ => ∫ y, f (h + (x + y)) ∂gaussianReal 0 vb) (gaussianReal 0 va) :=
    hprod.integral_prod_left.1
  have hinner (x : ℝ) :
      (∫ y, f (h + a * x + b * y) ∂gaussianReal 0 1) =
        ∫ y, f (h + a * x + y) ∂gaussianReal 0 vb := by
    have hm : AEStronglyMeasurable (fun y : ℝ => f (h + a * x + y))
        (Measure.map (fun y : ℝ => b * y) (gaussianReal 0 1)) :=
      (hfcont.comp (by fun_prop)).aestronglyMeasurable
    rw [← hmb, integral_map (by fun_prop) hm]
  have houter_map :
      (∫ x, ∫ y, f (h + a * x + y) ∂gaussianReal 0 vb ∂gaussianReal 0 1) =
        ∫ x, ∫ y, f (h + x + y) ∂gaussianReal 0 vb ∂gaussianReal 0 va := by
    have hm : AEStronglyMeasurable
        (fun x : ℝ => ∫ y, f (h + (x + y)) ∂gaussianReal 0 vb)
        (Measure.map (fun x : ℝ => a * x) (gaussianReal 0 1)) := by
      simpa [hma] using houter
    rw [← hma]
    simpa only [add_assoc] using (integral_map (by fun_prop) hm).symm
  unfold standardGaussianExpectation
  calc
    (∫ x, ∫ y, f (h + a * x + b * y) ∂gaussianReal 0 1 ∂gaussianReal 0 1)
        = ∫ x, ∫ y, f (h + x + y) ∂gaussianReal 0 vb ∂gaussianReal 0 va := by
          rw [integral_congr_ae (Filter.Eventually.of_forall hinner)]
          exact houter_map
    _ = ∫ z, f (h + z) ∂(gaussianReal 0 va ∗ gaussianReal 0 vb) := by
          simpa only [add_assoc] using (integral_conv hfint).symm
    _ = ∫ z, f (h + z) ∂gaussianReal 0 vc := by
          rw [gaussianReal_conv_gaussianReal, hv, zero_add]
    _ = ∫ z, f (h + c * z) ∂gaussianReal 0 1 := by
          rw [← hmc, integral_map (by fun_prop)]
          exact (hfcont.comp (by fun_prop)).aestronglyMeasurable

/-- The heat profile of `log ∘ cosh`. -/
private noncomputable def flatnessPhi (c x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z => Real.log (Real.cosh (x + c * z)))

private lemma flatnessPhi_nonneg (c x : ℝ) : 0 ≤ flatnessPhi c x := by
  unfold flatnessPhi standardGaussianExpectation
  exact integral_nonneg fun z => flatness_logCosh_nonneg _

/-- The first absolute moment of the standard Gaussian. -/
private noncomputable def flatnessAbsMoment : ℝ :=
  ∫ z, |z| ∂(gaussianReal 0 1)

private lemma flatnessAbsMoment_nonneg : 0 ≤ flatnessAbsMoment :=
  integral_nonneg fun z => abs_nonneg z

private lemma flatnessPhi_le (c x : ℝ) :
    flatnessPhi c x ≤ |x| + |c| * flatnessAbsMoment := by
  unfold flatnessPhi standardGaussianExpectation
  have hbound : ∀ z : ℝ, Real.log (Real.cosh (x + c * z)) ≤ |x| + |c| * |z| := by
    intro z
    refine (flatness_logCosh_le_abs _).trans ?_
    calc |x + c * z| ≤ |x| + |c * z| := abs_add_le x (c * z)
      _ = |x| + |c| * |z| := by rw [abs_mul]
  have hint : Integrable (fun z : ℝ => |x| + |c| * |z|) (gaussianReal 0 1) :=
    (integrable_const |x|).add
      ((flatness_integrable_abs_gaussian 0 1).const_mul |c|)
  calc (∫ z, Real.log (Real.cosh (x + c * z)) ∂gaussianReal 0 1)
      ≤ ∫ z, (|x| + |c| * |z|) ∂gaussianReal 0 1 :=
        integral_mono (flatness_integrable_logCosh x c 0 1) hint hbound
    _ = |x| + |c| * flatnessAbsMoment := by
        rw [integral_add (integrable_const |x|)
          ((flatness_integrable_abs_gaussian 0 1).const_mul |c|),
          integral_const_mul]
        simp [flatnessAbsMoment]

private lemma flatnessPhi_abs_le (c x : ℝ) :
    |flatnessPhi c x| ≤ |x| + |c| * flatnessAbsMoment := by
  rw [abs_of_nonneg (flatnessPhi_nonneg c x)]
  exact flatnessPhi_le c x

private lemma flatness_continuous_Phi (c : ℝ) : Continuous (flatnessPhi c) := by
  rw [continuous_iff_continuousAt]
  intro x₀
  unfold flatnessPhi standardGaussianExpectation
  have hbound : Integrable
      (fun z : ℝ => (|x₀| + 1) + |c| * |z|) (gaussianReal 0 1) :=
    (integrable_const _).add
      ((flatness_integrable_abs_gaussian 0 1).const_mul |c|)
  refine continuousAt_of_dominated (F := fun x z => Real.log (Real.cosh (x + c * z)))
    (bound := fun z : ℝ => (|x₀| + 1) + |c| * |z|) ?_ ?_ hbound ?_
  · filter_upwards [] with x
    exact (flatness_continuous_logCosh.comp (by fun_prop)).aestronglyMeasurable
  · filter_upwards [Metric.ball_mem_nhds x₀ zero_lt_one] with x hx
    filter_upwards [] with z
    have hxle : |x| ≤ |x₀| + 1 := by
      have := (Real.dist_eq x x₀) ▸ (Metric.mem_ball.mp hx)
      calc |x| = |x₀ + (x - x₀)| := by ring_nf
        _ ≤ |x₀| + |x - x₀| := abs_add_le _ _
        _ ≤ |x₀| + 1 := by linarith [this.le]
    have h1 : ‖Real.log (Real.cosh (x + c * z))‖ ≤ |x| + |c| * |z| := by
      rw [Real.norm_eq_abs, abs_of_nonneg (flatness_logCosh_nonneg _)]
      refine (flatness_logCosh_le_abs _).trans ?_
      calc |x + c * z| ≤ |x| + |c * z| := abs_add_le x (c * z)
        _ = |x| + |c| * |z| := by rw [abs_mul]
    linarith
  · filter_upwards [] with z
    exact (flatness_continuous_logCosh.comp (by fun_prop)).continuousAt

private lemma flatness_integrable_Phi (b x a : ℝ) :
    Integrable (fun z : ℝ => flatnessPhi b (x + a * z)) (gaussianReal 0 1) := by
  refine Integrable.mono'
    (g := fun z : ℝ => (|x| + |b| * flatnessAbsMoment) + |a| * |z|) ?_ ?_ ?_
  · exact (integrable_const _).add
      ((flatness_integrable_abs_gaussian 0 1).const_mul |a|)
  · exact ((flatness_continuous_Phi b).comp (by fun_prop)).aestronglyMeasurable
  · filter_upwards [] with z
    refine (flatnessPhi_abs_le b (x + a * z)).trans ?_
    have h2 : |x + a * z| ≤ |x| + |a| * |z| := by
      calc |x + a * z| ≤ |x| + |a * z| := abs_add_le x (a * z)
        _ = |x| + |a| * |z| := by rw [abs_mul]
    linarith

/-- Composition of two Gaussian shifts for the `log ∘ cosh` profile. -/
private lemma flatnessPhi_compose (a b c x : ℝ) (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun z => flatnessPhi b (x + a * z))
      = flatnessPhi c x := by
  unfold flatnessPhi
  exact flatness_gaussian_convolution_logCosh x a b c hc

/-! ### Decoupled steps of the finite GT recursion -/

private lemma flatness_diagonalStep_one_terminal_zero (c x₁ x₂ : ℝ) :
    gtDiagonalStep 1 c (gtTerminal 0) x₁ x₂
      = Real.log (Real.cosh x₁) + Real.log (Real.cosh x₂) + c ^ 2 := by
  rw [gtDiagonalStep_one_terminal, gtTerminal_zero]

private lemma flatness_integral_add_const {f : ℝ → ℝ} (K : ℝ)
    (hf : Integrable f (gaussianReal 0 1)) :
    (∫ z, (f z + K) ∂gaussianReal 0 1) = (∫ z, f z ∂gaussianReal 0 1) + K := by
  rw [integral_add hf (integrable_const K)]
  simp

private lemma flatness_step_diag_zero (g : ℝ → ℝ) (b c x₁ x₂ : ℝ)
    (hg₁ : Integrable (fun z : ℝ => g (x₁ + c * z)) (gaussianReal 0 1))
    (hg₂ : Integrable (fun z : ℝ => g (x₂ + c * z)) (gaussianReal 0 1)) :
    gtDiagonalStep 0 c (fun y₁ y₂ => g y₁ + g y₂ + b) x₁ x₂
      = standardGaussianExpectation (fun z => g (x₁ + c * z))
        + standardGaussianExpectation (fun z => g (x₂ + c * z)) + b := by
  unfold gtDiagonalStep standardGaussianExpectation
  rw [if_pos rfl]
  show (∫ z₁, (∫ z₂, (g (x₁ + c * z₁) + g (x₂ + c * z₂) + b) ∂gaussianReal 0 1)
      ∂gaussianReal 0 1) = _
  have hinner : ∀ y : ℝ,
      (∫ z₂, (y + g (x₂ + c * z₂) + b) ∂gaussianReal 0 1)
        = (∫ z₂, g (x₂ + c * z₂) ∂gaussianReal 0 1) + (y + b) := by
    intro y
    rw [show (fun z₂ : ℝ => y + g (x₂ + c * z₂) + b)
        = (fun z₂ : ℝ => g (x₂ + c * z₂) + (y + b)) from funext fun z₂ => by ring]
    exact flatness_integral_add_const _ hg₂
  simp_rw [hinner]
  rw [show (fun z₁ : ℝ =>
        (∫ z₂, g (x₂ + c * z₂) ∂gaussianReal 0 1) + (g (x₁ + c * z₁) + b))
      = (fun z₁ : ℝ => g (x₁ + c * z₁) +
        ((∫ z₂, g (x₂ + c * z₂) ∂gaussianReal 0 1) + b))
      from funext fun z₁ => by ring]
  rw [flatness_integral_add_const _ hg₁]
  ring

private lemma flatness_step_rank_zero (g : ℝ → ℝ) (b c sgn x₁ x₂ : ℝ)
    (hg₁ : Integrable (fun z : ℝ => g (x₁ + c * z)) (gaussianReal 0 1))
    (hg₂ : Integrable (fun z : ℝ => g (x₂ + sgn * c * z)) (gaussianReal 0 1)) :
    gtRankOneStep 0 c sgn (fun y₁ y₂ => g y₁ + g y₂ + b) x₁ x₂
      = standardGaussianExpectation (fun z => g (x₁ + c * z))
        + standardGaussianExpectation (fun z => g (x₂ + sgn * c * z)) + b := by
  unfold gtRankOneStep standardGaussianExpectation
  rw [if_pos rfl]
  show (∫ z, (g (x₁ + c * z) + g (x₂ + sgn * c * z) + b) ∂gaussianReal 0 1) = _
  have hsum : Integrable
      (fun z : ℝ => g (x₁ + c * z) + g (x₂ + sgn * c * z)) (gaussianReal 0 1) :=
    hg₁.add hg₂
  rw [flatness_integral_add_const b hsum, integral_add hg₁ hg₂]

/-- A mass-`1/2` rank-one step with positive sign, on the diagonal. -/
private lemma flatness_step_rank_half_diag (b c x : ℝ) :
    gtRankOneStep (1 / 2) c 1
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂) + b) x x
      = 2 * Real.log (Real.cosh x) + c ^ 2 + b := by
  unfold gtRankOneStep
  rw [if_neg (by norm_num : (1 / 2 : ℝ) ≠ 0)]
  have hfun : (fun z : ℝ => Real.exp ((1 / 2 : ℝ) *
      (Real.log (Real.cosh (x + c * z)) + Real.log (Real.cosh (x + 1 * c * z)) + b)))
      = fun z : ℝ => Real.exp (b / 2) * Real.cosh (x + c * z) := by
    funext z
    rw [one_mul,
      show (1 / 2 : ℝ) * (Real.log (Real.cosh (x + c * z))
          + Real.log (Real.cosh (x + c * z)) + b)
        = Real.log (Real.cosh (x + c * z)) + b / 2 by ring,
      Real.exp_add, Real.exp_log (Real.cosh_pos _)]
    ring
  rw [hfun]
  have hconst : standardGaussianExpectation
      (fun z => Real.exp (b / 2) * Real.cosh (x + c * z))
      = Real.exp (b / 2) * (Real.cosh x * Real.exp (c ^ 2 / 2)) := by
    unfold standardGaussianExpectation
    rw [integral_const_mul]
    rw [show (∫ z, Real.cosh (x + c * z) ∂gaussianReal 0 1)
        = Real.cosh x * Real.exp (c ^ 2 / 2) from flatness_gaussianExpectation_cosh x c]
  rw [hconst,
    Real.log_mul (Real.exp_ne_zero _)
      (by positivity),
    Real.log_mul (Real.cosh_pos x).ne' (Real.exp_ne_zero _),
    Real.log_exp, Real.log_exp]
  ring

/-! ### Decoupled evaluation of the GT recursion at `lam = 0` -/

private lemma flatness_diagZero_apply (c : ℝ) (G : GTTwoField) (x₁ x₂ : ℝ) :
    gtDiagonalStep 0 c G x₁ x₂ =
      standardGaussianExpectation (fun z₁ =>
        standardGaussianExpectation (fun z₂ => G (x₁ + c * z₁) (x₂ + c * z₂))) := by
  unfold gtDiagonalStep
  rw [if_pos rfl]

private lemma flatness_rankZero_apply (c sgn : ℝ) (G : GTTwoField) (x₁ x₂ : ℝ) :
    gtRankOneStep 0 c sgn G x₁ x₂ =
      standardGaussianExpectation (fun z => G (x₁ + c * z) (x₂ + sgn * c * z)) := by
  unfold gtRankOneStep
  rw [if_pos rfl]

private lemma flatness_step_diag_zero_logCosh (b c x₁ x₂ : ℝ) :
    gtDiagonalStep 0 c
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂) + b) x₁ x₂
      = flatnessPhi c x₁ + flatnessPhi c x₂ + b :=
  flatness_step_diag_zero (fun t => Real.log (Real.cosh t)) b c x₁ x₂
    (flatness_integrable_logCosh x₁ c 0 1) (flatness_integrable_logCosh x₂ c 0 1)

private lemma flatness_expectation_affine_logCosh (x c b : ℝ) :
    standardGaussianExpectation (fun z => 2 * Real.log (Real.cosh (x + c * z)) + b)
      = 2 * flatnessPhi c x + b := by
  unfold flatnessPhi standardGaussianExpectation
  rw [integral_add ((flatness_integrable_logCosh x c 0 1).const_mul 2)
    (integrable_const b), integral_const_mul]
  simp

private lemma flatness_expectation_affine_Phi (x a c b : ℝ) :
    standardGaussianExpectation (fun z => 2 * flatnessPhi c (x + a * z) + b)
      = 2 * standardGaussianExpectation (fun z => flatnessPhi c (x + a * z)) + b := by
  unfold standardGaussianExpectation
  rw [integral_add ((flatness_integrable_Phi c x a).const_mul 2)
    (integrable_const b), integral_const_mul]
  simp

private lemma flatness_diag_one_terminal_fun (c : ℝ) :
    gtDiagonalStep 1 c (gtTerminal 0) =
      fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂) + c ^ 2 := by
  funext y₁ y₂
  exact flatness_diagonalStep_one_terminal_zero c y₁ y₂

/-- Zero-multiplier integrand in the regime `|v| = 0`. -/
private lemma flatness_integrand_zero (β s q X : ℝ) :
    gtDiagonalStep 0 (gtIncrementScale β s 0 q)
        (gtDiagonalStep 1 (gtIncrementScale β s q 1) (gtTerminal 0)) X X
      = 2 * flatnessPhi (gtIncrementScale β s 0 q) X
        + gtIncrementScale β s q 1 ^ 2 := by
  rw [flatness_diag_one_terminal_fun, flatness_step_diag_zero_logCosh]
  ring

/-- Zero-multiplier integrand in the regime `0 < |v| < q`. -/
private lemma flatness_integrand_lower (β s q r X : ℝ)
    (hβ : 0 ≤ β) (hs : 0 ≤ s) (hr : 0 ≤ r) (hrq : r ≤ q) :
    gtRankOneStep 0 (gtIncrementScale β s 0 r) 1
        (gtDiagonalStep 0 (gtIncrementScale β s r q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1) (gtTerminal 0))) X X
      = 2 * flatnessPhi (gtIncrementScale β s 0 q) X
        + gtIncrementScale β s q 1 ^ 2 := by
  have hcomp : gtIncrementScale β s 0 q ^ 2 =
      gtIncrementScale β s 0 r ^ 2 + gtIncrementScale β s r q ^ 2 := by
    rw [gtIncrementScale_sq_of_nonneg β s 0 q hβ hs (le_trans hr hrq),
      gtIncrementScale_sq_of_nonneg β s 0 r hβ hs hr,
      gtIncrementScale_sq_of_nonneg β s r q hβ hs hrq]
    ring
  rw [flatness_rankZero_apply, flatness_diag_one_terminal_fun]
  simp only [one_mul]
  have hstep : ∀ y : ℝ,
      gtDiagonalStep 0 (gtIncrementScale β s r q)
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂)
          + gtIncrementScale β s q 1 ^ 2) y y
      = 2 * flatnessPhi (gtIncrementScale β s r q) y
        + gtIncrementScale β s q 1 ^ 2 := by
    intro y
    rw [flatness_step_diag_zero_logCosh]
    ring
  simp_rw [hstep]
  rw [flatness_expectation_affine_Phi,
    flatnessPhi_compose (gtIncrementScale β s 0 r) (gtIncrementScale β s r q)
      (gtIncrementScale β s 0 q) X hcomp]

/-- Zero-multiplier integrand in the regime `q ≤ |v| ≤ 1`. -/
private lemma flatness_integrand_upper (β s q r X : ℝ) :
    gtRankOneStep 0 (gtIncrementScale β s 0 q) 1
        (gtRankOneStep (1 / 2) (gtIncrementScale β s q r) 1
          (gtDiagonalStep 1 (gtIncrementScale β s r 1) (gtTerminal 0))) X X
      = 2 * flatnessPhi (gtIncrementScale β s 0 q) X
        + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2) := by
  rw [flatness_rankZero_apply, flatness_diag_one_terminal_fun]
  simp only [one_mul]
  have hstep : ∀ y : ℝ,
      gtRankOneStep (1 / 2) (gtIncrementScale β s q r) 1
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂)
          + gtIncrementScale β s r 1 ^ 2) y y
      = 2 * Real.log (Real.cosh y)
        + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2) := by
    intro y
    rw [flatness_step_rank_half_diag]
    ring
  simp_rw [hstep]
  rw [flatness_expectation_affine_logCosh]

/-- Common bookkeeping: the accumulated increments recombine into the
replica-symmetric path value. -/
private lemma flatness_zero_multiplier_assemble
    (β h q s B : ℝ) (hβ : 0 ≤ β) (hs : s ∈ Set.Icc (0 : ℝ) 1) (hq : 0 ≤ q)
    (hB : B = s * β ^ 2 * (1 - q)) :
    2 * Real.log 2 +
        standardGaussianExpectation (fun z =>
          2 * flatnessPhi (gtIncrementScale β s 0 q)
            (h + β * Real.sqrt ((1 - s) * q) * z) + B)
        - gtCorrection β q s
      = 2 * rsPathValue β h q s := by
  have h1s : 0 ≤ 1 - s := sub_nonneg.mpr hs.2
  have hvar : (β * Real.sqrt q) ^ 2 =
      (β * Real.sqrt ((1 - s) * q)) ^ 2 + gtIncrementScale β s 0 q ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hq, mul_pow, Real.sq_sqrt (mul_nonneg h1s hq),
      gtIncrementScale_sq_of_nonneg β s 0 q hβ hs.1 hq]
    ring
  rw [flatness_expectation_affine_Phi,
    flatnessPhi_compose (β * Real.sqrt ((1 - s) * q)) (gtIncrementScale β s 0 q)
      (β * Real.sqrt q) h hvar, hB]
  unfold rsPathValue gtCorrection flatnessPhi
  ring

/-- **At zero multiplier and nonnegative overlap the GT functional is exactly
twice the replica-symmetric path value.** -/
lemma flatness_gtFunctional_zero_multiplier
    (β h q s v : ℝ) (hβ : 0 ≤ β) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 < q) (hq1 : q < 1) (hv : v ∈ Set.Icc (0 : ℝ) 1) :
    gtFunctional β h q s 0 v = 2 * rsPathValue β h q s := by
  have hvabs : |v| = v := abs_of_nonneg hv.1
  have hsign : gtPathSign v = 1 := by
    unfold gtPathSign
    rw [if_pos hv.1]
  rcases eq_or_lt_of_le hv.1 with hv0 | hvpos
  · have habs : |v| = 0 := by rw [hvabs, ← hv0]
    rw [flatness_gtFunctional_formula_abs_v_eq_zero β h q s 0 v hq habs]
    simp_rw [flatness_integrand_zero]
    exact flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le
      (gtIncrementScale_sq_of_nonneg β s q 1 hβ hs.1 hq1.le)
  · by_cases hvq : v < q
    · rw [flatness_gtFunctional_formula_abs_v_lt_q β h q s 0 v
        (by rw [hvabs]; exact hvpos) (by rw [hvabs]; exact hvq)]
      simp only [hsign, hvabs, zero_mul, sub_zero]
      simp_rw [flatness_integrand_lower β s q v _ hβ hs.1 hv.1 hvq.le]
      exact flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le
        (gtIncrementScale_sq_of_nonneg β s q 1 hβ hs.1 hq1.le)
    · have hqv : q ≤ v := le_of_not_gt hvq
      rcases eq_or_lt_of_le hv.2 with hv1 | hv1
      · have habs : |v| = 1 := by rw [hvabs, hv1]
        rw [flatness_gtFunctional_formula_abs_v_eq_one β h q s 0 v hq hq1.le habs]
        simp only [hsign, zero_mul, sub_zero]
        simp_rw [flatness_integrand_upper β s q 1]
        refine flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le ?_
        rw [gtIncrementScale_sq_of_nonneg β s q 1 hβ hs.1 hq1.le,
          gtIncrementScale_sq_of_nonneg β s 1 1 hβ hs.1 le_rfl]
        ring
      · rw [flatness_gtFunctional_formula_q_le_abs_v_lt_one β h q s 0 v hq
          (by rw [hvabs]; exact hqv) (by rw [hvabs]; exact hv1)]
        simp only [hsign, hvabs, zero_mul, sub_zero]
        simp_rw [flatness_integrand_upper β s q v]
        refine flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le ?_
        rw [gtIncrementScale_sq_of_nonneg β s q v hβ hs.1 hqv,
          gtIncrementScale_sq_of_nonneg β s v 1 hβ hs.1 hv.2]
        ring

/-! ### A quantitative multiplier for the quadratic gap -/

/-- The endpoint multiplier derivative is bounded by `2`. -/
lemma flatness_abs_deriv_gtFunctional_zero_le_two
    (β h q s v : ℝ) (hv : |v| ≤ 1) :
    |deriv (fun l => gtFunctional β h q s l v) 0| ≤ 2 := by
  rw [deriv_gtFunctional_eq]
  have hb : |standardGaussianExpectation (fun z =>
      deriv (fun l => gtSemigroupSolution β q s l v 0
        (h + β * Real.sqrt ((1 - s) * q) * z)
        (h + β * Real.sqrt ((1 - s) * q) * z)) 0)| ≤ 1 := by
    unfold standardGaussianExpectation
    have hbound := norm_integral_le_of_norm_le_const
      (μ := gaussianReal 0 1) (C := 1)
      (f := fun z : ℝ => deriv (fun l => gtSemigroupSolution β q s l v 0
        (h + β * Real.sqrt ((1 - s) * q) * z)
        (h + β * Real.sqrt ((1 - s) * q) * z)) 0)
      (Filter.Eventually.of_forall fun z => by
        simpa using (gt_lambda_derivative_bounds β h q s 0 v 0
          (h + β * Real.sqrt ((1 - s) * q) * z)
          (h + β * Real.sqrt ((1 - s) * q) * z)).1.1)
    simpa using hbound
  have h1 := abs_le.mp hb
  have h2 := abs_le.mp hv
  rw [abs_le]
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

/-- Optimizing the explicit multiplier `-(2/5) ∂_λ` turns a linear lower bound
on the multiplier derivative into a quadratic gap, with a multiplier in
`[-1,1]`. -/
lemma flatness_quadratic_gap_of_deriv_gap_mem
    (β h q s v c : ℝ) (hc : 0 ≤ c)
    (hzero : gtFunctional β h q s 0 v ≤ 2 * rsPathValue β h q s)
    (hd : |deriv (fun l => gtFunctional β h q s l v) 0| ≤ 2)
    (hgap : c * |v - q| ≤ |deriv (fun l => gtFunctional β h q s l v) 0|) :
    ∃ lam ∈ Set.Icc (-1 : ℝ) 1,
      gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - (c ^ 2 / 5) * (v - q) ^ 2 := by
  set d : ℝ := deriv (fun l => gtFunctional β h q s l v) 0 with hdef
  refine ⟨-(2 / 5 : ℝ) * d, ?_, ?_⟩
  · have h2 := abs_le.mp hd
    rw [Set.mem_Icc]
    constructor <;> linarith [h2.1, h2.2]
  · have ht := flatness_gtFunctional_taylor_upper β h q s v (-(2 / 5 : ℝ) * d)
    rw [← hdef] at ht
    have hopt : d * (-(2 / 5 : ℝ) * d) + (5 / 4 : ℝ) * (-(2 / 5 : ℝ) * d) ^ 2
        = -(d ^ 2) / 5 := by ring
    have ht' : gtFunctional β h q s (-(2 / 5 : ℝ) * d) v
        ≤ gtFunctional β h q s 0 v - d ^ 2 / 5 := by linarith [ht, hopt.ge, hopt.le]
    have habs : c * |v - q| ≤ |d| := hgap
    have hsq : c ^ 2 * (v - q) ^ 2 ≤ d ^ 2 := by
      have hmul := mul_self_le_mul_self
        (mul_nonneg hc (abs_nonneg (v - q))) habs
      nlinarith [sq_abs (v - q), sq_abs d]
    linarith [ht', hzero]

/-! ### The GT functional at zero multiplier and negative overlap

For a negative overlap the two replicas are coupled with the opposite sign.
Below the replica-symmetric breakpoint the value is still *exactly* twice the
replica-symmetric path value (a reflected Gaussian shift has the same
variance), while above the breakpoint the mass-`1/2` step is estimated by a
Cauchy--Schwarz inequality, which again produces the separable bound. -/

private lemma flatness_integrable_cosh (y c : ℝ) :
    Integrable (fun z : ℝ => Real.cosh (y + c * z)) (gaussianReal 0 1) := by
  have hfun : (fun z : ℝ => Real.cosh (y + c * z))
      = fun z : ℝ => (Real.exp (y + c * z) + Real.exp (-y + (-c) * z)) / 2 := by
    funext z
    have hneg : -(y + c * z) = -y + (-c) * z := by ring
    rw [Real.cosh_eq, hneg]
  rw [hfun]
  have hsum : Integrable
      (fun z : ℝ => Real.exp (y + c * z) + Real.exp (-y + (-c) * z))
      (gaussianReal 0 1) :=
    (flatness_integrable_exp_shift y c).add (flatness_integrable_exp_shift (-y) (-c))
  exact hsum.div_const 2

private lemma flatnessPhi_zero (x : ℝ) :
    flatnessPhi 0 x = Real.log (Real.cosh x) := by
  unfold flatnessPhi standardGaussianExpectation
  simp

private lemma flatness_integrable_Phi_reflect (b x a : ℝ) :
    Integrable (fun z : ℝ => flatnessPhi b (x + -1 * a * z)) (gaussianReal 0 1) := by
  simpa only [neg_one_mul] using flatness_integrable_Phi b x (-a)

/-- Composition of two Gaussian shifts, the outer one reflected. -/
private lemma flatnessPhi_compose_reflect (a b c x : ℝ) (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun z => flatnessPhi b (x + -1 * a * z))
      = flatnessPhi c x := by
  have h := flatnessPhi_compose (-a) b c x (by rw [hc]; ring)
  simpa only [neg_one_mul] using h

/-- A reflected pair of Gaussian shifts recombines into a single profile. -/
private lemma flatness_two_Phi_shift (a b c X K : ℝ) (hc : c ^ 2 = a ^ 2 + b ^ 2) :
    standardGaussianExpectation (fun z =>
        flatnessPhi b (X + a * z) + flatnessPhi b (X + -1 * a * z) + K)
      = 2 * flatnessPhi c X + K := by
  have h₁ : Integrable (fun z : ℝ => flatnessPhi b (X + a * z)) (gaussianReal 0 1) :=
    flatness_integrable_Phi b X a
  have h₂ : Integrable (fun z : ℝ => flatnessPhi b (X + -1 * a * z))
      (gaussianReal 0 1) := flatness_integrable_Phi_reflect b X a
  have hsum : Integrable (fun z : ℝ =>
      flatnessPhi b (X + a * z) + flatnessPhi b (X + -1 * a * z))
      (gaussianReal 0 1) := h₁.add h₂
  have e₁ : (∫ z, flatnessPhi b (X + a * z) ∂gaussianReal 0 1) = flatnessPhi c X :=
    flatnessPhi_compose a b c X hc
  have e₂ : (∫ z, flatnessPhi b (X + -1 * a * z) ∂gaussianReal 0 1) = flatnessPhi c X :=
    flatnessPhi_compose_reflect a b c X hc
  show (∫ z, (flatnessPhi b (X + a * z) + flatnessPhi b (X + -1 * a * z) + K)
      ∂gaussianReal 0 1) = _
  rw [flatness_integral_add_const K hsum, integral_add h₁ h₂, e₁, e₂]
  ring

private lemma flatness_two_logCosh_shift (a X K : ℝ) :
    standardGaussianExpectation (fun z =>
        Real.log (Real.cosh (X + a * z))
          + Real.log (Real.cosh (X + -1 * a * z)) + K)
      = 2 * flatnessPhi a X + K := by
  have h := flatness_two_Phi_shift a 0 a X K (by ring)
  simpa only [flatnessPhi_zero] using h

/-- Zero-multiplier integrand in the regime `0 < |v| < q` and negative sign. -/
private lemma flatness_integrand_lower_neg (β s q r X : ℝ)
    (hβ : 0 ≤ β) (hs : 0 ≤ s) (hr : 0 ≤ r) (hrq : r ≤ q) :
    gtRankOneStep 0 (gtIncrementScale β s 0 r) (-1)
        (gtDiagonalStep 0 (gtIncrementScale β s r q)
          (gtDiagonalStep 1 (gtIncrementScale β s q 1) (gtTerminal 0))) X X
      = 2 * flatnessPhi (gtIncrementScale β s 0 q) X
        + gtIncrementScale β s q 1 ^ 2 := by
  have hcomp : gtIncrementScale β s 0 q ^ 2 =
      gtIncrementScale β s 0 r ^ 2 + gtIncrementScale β s r q ^ 2 := by
    rw [gtIncrementScale_sq_of_nonneg β s 0 q hβ hs (le_trans hr hrq),
      gtIncrementScale_sq_of_nonneg β s 0 r hβ hs hr,
      gtIncrementScale_sq_of_nonneg β s r q hβ hs hrq]
    ring
  rw [flatness_rankZero_apply, flatness_diag_one_terminal_fun]
  simp_rw [flatness_step_diag_zero_logCosh]
  exact flatness_two_Phi_shift (gtIncrementScale β s 0 r)
    (gtIncrementScale β s r q) (gtIncrementScale β s 0 q) X _ hcomp

/-! #### The mass-`1/2` step with negative sign -/

private lemma flatness_integrable_sqrt_cosh_prod (c y₁ y₂ : ℝ) :
    Integrable (fun z : ℝ =>
      Real.sqrt (Real.cosh (y₁ + c * z)) * Real.sqrt (Real.cosh (y₂ + -1 * c * z)))
      (gaussianReal 0 1) := by
  have hsum : Integrable (fun z : ℝ =>
      Real.cosh (y₁ + c * z) + Real.cosh (y₂ + -1 * c * z)) (gaussianReal 0 1) :=
    (flatness_integrable_cosh y₁ c).add (flatness_integrable_cosh y₂ (-1 * c))
  have hbound : Integrable (fun z : ℝ =>
      (Real.cosh (y₁ + c * z) + Real.cosh (y₂ + -1 * c * z)) / 2)
      (gaussianReal 0 1) := hsum.div_const 2
  refine Integrable.mono' hbound ?_ ?_
  · exact (((Real.continuous_sqrt.comp (Real.continuous_cosh.comp (by fun_prop))).mul
      (Real.continuous_sqrt.comp
        (Real.continuous_cosh.comp (by fun_prop))))).aestronglyMeasurable
  · filter_upwards [] with z
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    nlinarith [Real.sq_sqrt (Real.cosh_pos (y₁ + c * z)).le,
      Real.sq_sqrt (Real.cosh_pos (y₂ + -1 * c * z)).le,
      sq_nonneg (Real.sqrt (Real.cosh (y₁ + c * z))
        - Real.sqrt (Real.cosh (y₂ + -1 * c * z))),
      Real.sqrt_nonneg (Real.cosh (y₁ + c * z)),
      Real.sqrt_nonneg (Real.cosh (y₂ + -1 * c * z))]

private lemma flatness_one_le_sqrt_cosh_prod (c y₁ y₂ : ℝ) :
    1 ≤ standardGaussianExpectation (fun z =>
      Real.sqrt (Real.cosh (y₁ + c * z)) *
        Real.sqrt (Real.cosh (y₂ + -1 * c * z))) := by
  have hpt : ∀ z : ℝ, (1 : ℝ) ≤
      Real.sqrt (Real.cosh (y₁ + c * z)) *
        Real.sqrt (Real.cosh (y₂ + -1 * c * z)) := by
    intro z
    have h1 : (1 : ℝ) ≤ Real.sqrt (Real.cosh (y₁ + c * z)) := by
      rw [show (1 : ℝ) = Real.sqrt 1 by simp]
      exact Real.sqrt_le_sqrt (Real.one_le_cosh _)
    have h2 : (1 : ℝ) ≤ Real.sqrt (Real.cosh (y₂ + -1 * c * z)) := by
      rw [show (1 : ℝ) = Real.sqrt 1 by simp]
      exact Real.sqrt_le_sqrt (Real.one_le_cosh _)
    nlinarith
  have h := integral_mono (integrable_const (1 : ℝ))
    (flatness_integrable_sqrt_cosh_prod c y₁ y₂) hpt
  unfold standardGaussianExpectation
  simpa using h

private lemma flatness_amgm_balance (u w E : ℝ) (hu : 0 < u) (hw : 0 < w) :
    (u / w)⁻¹ / 2 * (u ^ 2 * E) + (u / w) / 2 * (w ^ 2 * E) = E * (u * w) := by
  field_simp
  ring

/-- Cauchy--Schwarz for the reflected mass-`1/2` weight. -/
private lemma flatness_sqrt_cosh_prod_le (c y₁ y₂ : ℝ) :
    standardGaussianExpectation (fun z =>
        Real.sqrt (Real.cosh (y₁ + c * z)) *
          Real.sqrt (Real.cosh (y₂ + -1 * c * z)))
      ≤ Real.exp (c ^ 2 / 2) *
        (Real.sqrt (Real.cosh y₁) * Real.sqrt (Real.cosh y₂)) := by
  have hs₁ : 0 < Real.sqrt (Real.cosh y₁) := Real.sqrt_pos.mpr (Real.cosh_pos y₁)
  have hs₂ : 0 < Real.sqrt (Real.cosh y₂) := Real.sqrt_pos.mpr (Real.cosh_pos y₂)
  obtain ⟨t, ht0, hfinal⟩ : ∃ t : ℝ, 0 < t ∧
      t⁻¹ / 2 * (Real.cosh y₁ * Real.exp (c ^ 2 / 2))
        + t / 2 * (Real.cosh y₂ * Real.exp (c ^ 2 / 2))
        = Real.exp (c ^ 2 / 2) *
          (Real.sqrt (Real.cosh y₁) * Real.sqrt (Real.cosh y₂)) := by
    refine ⟨Real.sqrt (Real.cosh y₁) / Real.sqrt (Real.cosh y₂), div_pos hs₁ hs₂, ?_⟩
    have h := flatness_amgm_balance (Real.sqrt (Real.cosh y₁))
      (Real.sqrt (Real.cosh y₂)) (Real.exp (c ^ 2 / 2)) hs₁ hs₂
    rwa [Real.sq_sqrt (Real.cosh_pos y₁).le,
      Real.sq_sqrt (Real.cosh_pos y₂).le] at h
  have hbound : ∀ z : ℝ,
      Real.sqrt (Real.cosh (y₁ + c * z)) *
          Real.sqrt (Real.cosh (y₂ + -1 * c * z))
        ≤ t⁻¹ / 2 * Real.cosh (y₁ + c * z)
          + t / 2 * Real.cosh (y₂ + -1 * c * z) := by
    intro z
    have hA : (0 : ℝ) ≤ Real.cosh (y₁ + c * z) := (Real.cosh_pos _).le
    have h1 : Real.sqrt (Real.cosh (y₁ + c * z)) *
        Real.sqrt (Real.cosh (y₂ + -1 * c * z))
        = Real.sqrt (Real.cosh (y₁ + c * z) / t) *
          Real.sqrt (t * Real.cosh (y₂ + -1 * c * z)) := by
      rw [← Real.sqrt_mul hA,
        ← Real.sqrt_mul (show (0 : ℝ) ≤ Real.cosh (y₁ + c * z) / t by positivity)]
      congr 1
      field_simp
    have hu := Real.sq_sqrt
      (show (0 : ℝ) ≤ Real.cosh (y₁ + c * z) / t by positivity)
    have hw := Real.sq_sqrt
      (show (0 : ℝ) ≤ t * Real.cosh (y₂ + -1 * c * z) by positivity)
    have hdiv : Real.cosh (y₁ + c * z) / t = t⁻¹ * Real.cosh (y₁ + c * z) :=
      div_eq_inv_mul _ _
    rw [h1]
    nlinarith [sq_nonneg (Real.sqrt (Real.cosh (y₁ + c * z) / t)
        - Real.sqrt (t * Real.cosh (y₂ + -1 * c * z))), hu, hw, hdiv]
  have hint : Integrable (fun z : ℝ =>
      t⁻¹ / 2 * Real.cosh (y₁ + c * z)
        + t / 2 * Real.cosh (y₂ + -1 * c * z)) (gaussianReal 0 1) :=
    ((flatness_integrable_cosh y₁ c).const_mul _).add
      ((flatness_integrable_cosh y₂ (-1 * c)).const_mul _)
  have hmono : standardGaussianExpectation (fun z =>
      Real.sqrt (Real.cosh (y₁ + c * z)) *
        Real.sqrt (Real.cosh (y₂ + -1 * c * z)))
      ≤ standardGaussianExpectation (fun z =>
        t⁻¹ / 2 * Real.cosh (y₁ + c * z)
          + t / 2 * Real.cosh (y₂ + -1 * c * z)) := by
    unfold standardGaussianExpectation
    exact integral_mono (flatness_integrable_sqrt_cosh_prod c y₁ y₂) hint hbound
  have heval : standardGaussianExpectation (fun z =>
      t⁻¹ / 2 * Real.cosh (y₁ + c * z)
        + t / 2 * Real.cosh (y₂ + -1 * c * z))
      = t⁻¹ / 2 * (Real.cosh y₁ * Real.exp (c ^ 2 / 2))
        + t / 2 * (Real.cosh y₂ * Real.exp (c ^ 2 / 2)) := by
    have e₁ := flatness_gaussianExpectation_cosh y₁ c
    have e₂ := flatness_gaussianExpectation_cosh y₂ (-1 * c)
    unfold standardGaussianExpectation at e₁ e₂ ⊢
    rw [integral_add ((flatness_integrable_cosh y₁ c).const_mul _)
      ((flatness_integrable_cosh y₂ (-1 * c)).const_mul _),
      integral_const_mul, integral_const_mul, e₁, e₂]
    ring_nf
  linarith [hmono, heval.ge, heval.le, hfinal.ge, hfinal.le]

private lemma flatness_step_rank_half_neg_eq (b c y₁ y₂ : ℝ) :
    gtRankOneStep (1 / 2) c (-1)
        (fun t₁ t₂ => Real.log (Real.cosh t₁) + Real.log (Real.cosh t₂) + b ^ 2)
        y₁ y₂
      = b ^ 2 + 2 * Real.log (standardGaussianExpectation (fun z =>
          Real.sqrt (Real.cosh (y₁ + c * z)) *
            Real.sqrt (Real.cosh (y₂ + -1 * c * z)))) := by
  have hfun : (fun z : ℝ => Real.exp ((1 / 2 : ℝ) *
      (Real.log (Real.cosh (y₁ + c * z))
        + Real.log (Real.cosh (y₂ + -1 * c * z)) + b ^ 2)))
      = fun z : ℝ => Real.exp (b ^ 2 / 2) *
        (Real.sqrt (Real.cosh (y₁ + c * z)) *
          Real.sqrt (Real.cosh (y₂ + -1 * c * z))) := by
    funext z
    rw [show (1 / 2 : ℝ) * (Real.log (Real.cosh (y₁ + c * z))
          + Real.log (Real.cosh (y₂ + -1 * c * z)) + b ^ 2)
        = b ^ 2 / 2 + (Real.log (Real.sqrt (Real.cosh (y₁ + c * z)))
          + Real.log (Real.sqrt (Real.cosh (y₂ + -1 * c * z)))) from by
      rw [Real.log_sqrt (Real.cosh_pos _).le, Real.log_sqrt (Real.cosh_pos _).le]
      ring]
    rw [Real.exp_add, Real.exp_add,
      Real.exp_log (Real.sqrt_pos.mpr (Real.cosh_pos _)),
      Real.exp_log (Real.sqrt_pos.mpr (Real.cosh_pos _))]
  have hpos := flatness_one_le_sqrt_cosh_prod c y₁ y₂
  have hconst : standardGaussianExpectation (fun z => Real.exp (b ^ 2 / 2) *
      (Real.sqrt (Real.cosh (y₁ + c * z)) *
        Real.sqrt (Real.cosh (y₂ + -1 * c * z))))
      = Real.exp (b ^ 2 / 2) * standardGaussianExpectation (fun z =>
        Real.sqrt (Real.cosh (y₁ + c * z)) *
          Real.sqrt (Real.cosh (y₂ + -1 * c * z))) := by
    unfold standardGaussianExpectation
    exact integral_const_mul _ _
  unfold gtRankOneStep
  rw [if_neg (by norm_num : (1 / 2 : ℝ) ≠ 0)]
  show (1 / (1 / 2 : ℝ)) * Real.log (standardGaussianExpectation
      (fun z => Real.exp ((1 / 2 : ℝ) *
        (Real.log (Real.cosh (y₁ + c * z))
          + Real.log (Real.cosh (y₂ + -1 * c * z)) + b ^ 2)))) = _
  rw [hfun, hconst, Real.log_mul (Real.exp_ne_zero _) (by linarith), Real.log_exp]
  ring

private lemma flatness_step_rank_half_neg_nonneg (b c y₁ y₂ : ℝ) :
    0 ≤ gtRankOneStep (1 / 2) c (-1)
      (fun t₁ t₂ => Real.log (Real.cosh t₁) + Real.log (Real.cosh t₂) + b ^ 2)
      y₁ y₂ := by
  rw [flatness_step_rank_half_neg_eq]
  have h := Real.log_nonneg (flatness_one_le_sqrt_cosh_prod c y₁ y₂)
  nlinarith [sq_nonneg b]

private lemma flatness_step_rank_half_neg_le (b c y₁ y₂ : ℝ) :
    gtRankOneStep (1 / 2) c (-1)
        (fun t₁ t₂ => Real.log (Real.cosh t₁) + Real.log (Real.cosh t₂) + b ^ 2)
        y₁ y₂
      ≤ Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂) + (c ^ 2 + b ^ 2) := by
  rw [flatness_step_rank_half_neg_eq]
  have hone := flatness_one_le_sqrt_cosh_prod c y₁ y₂
  have hle := flatness_sqrt_cosh_prod_le c y₁ y₂
  have hs₁ : 0 < Real.sqrt (Real.cosh y₁) := Real.sqrt_pos.mpr (Real.cosh_pos y₁)
  have hs₂ : 0 < Real.sqrt (Real.cosh y₂) := Real.sqrt_pos.mpr (Real.cosh_pos y₂)
  have hlog := Real.log_le_log (by linarith) hle
  rw [Real.log_mul (Real.exp_ne_zero _) (by positivity), Real.log_exp,
    Real.log_mul (ne_of_gt hs₁) (ne_of_gt hs₂),
    Real.log_sqrt (Real.cosh_pos y₁).le,
    Real.log_sqrt (Real.cosh_pos y₂).le] at hlog
  linarith

/-- Zero-multiplier integrand in the regime `q ≤ |v| ≤ 1` and negative sign;
here the mass-`1/2` step only gives an inequality. -/
private lemma flatness_integrand_upper_neg_le (β s q r X : ℝ) :
    gtRankOneStep 0 (gtIncrementScale β s 0 q) (-1)
        (gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
          (gtDiagonalStep 1 (gtIncrementScale β s r 1) (gtTerminal 0))) X X
      ≤ 2 * flatnessPhi (gtIncrementScale β s 0 q) X
        + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2) := by
  rw [flatness_rankZero_apply, flatness_diag_one_terminal_fun]
  have hnn : ∀ z : ℝ, 0 ≤
      gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂)
          + gtIncrementScale β s r 1 ^ 2)
        (X + gtIncrementScale β s 0 q * z)
        (X + -1 * gtIncrementScale β s 0 q * z) :=
    fun z => flatness_step_rank_half_neg_nonneg _ _ _ _
  have hpt : ∀ z : ℝ,
      gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂)
          + gtIncrementScale β s r 1 ^ 2)
        (X + gtIncrementScale β s 0 q * z)
        (X + -1 * gtIncrementScale β s 0 q * z)
      ≤ Real.log (Real.cosh (X + gtIncrementScale β s 0 q * z))
        + Real.log (Real.cosh (X + -1 * gtIncrementScale β s 0 q * z))
        + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2) :=
    fun z => flatness_step_rank_half_neg_le _ _ _ _
  have hi₁ : Integrable
      (fun z : ℝ => Real.log (Real.cosh (X + gtIncrementScale β s 0 q * z)))
      (gaussianReal 0 1) := flatness_integrable_logCosh X _ 0 1
  have hi₂ : Integrable
      (fun z : ℝ => Real.log (Real.cosh (X + -1 * gtIncrementScale β s 0 q * z)))
      (gaussianReal 0 1) := by
    simpa only [neg_one_mul] using
      flatness_integrable_logCosh X (-gtIncrementScale β s 0 q) 0 1
  have hi₃ : Integrable (fun z : ℝ =>
      Real.log (Real.cosh (X + gtIncrementScale β s 0 q * z))
        + Real.log (Real.cosh (X + -1 * gtIncrementScale β s 0 q * z)))
      (gaussianReal 0 1) := hi₁.add hi₂
  have hint : Integrable (fun z : ℝ =>
      Real.log (Real.cosh (X + gtIncrementScale β s 0 q * z))
        + Real.log (Real.cosh (X + -1 * gtIncrementScale β s 0 q * z))
        + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2))
      (gaussianReal 0 1) := hi₃.add (integrable_const _)
  have hmono : standardGaussianExpectation (fun z =>
      gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
        (fun y₁ y₂ => Real.log (Real.cosh y₁) + Real.log (Real.cosh y₂)
          + gtIncrementScale β s r 1 ^ 2)
        (X + gtIncrementScale β s 0 q * z)
        (X + -1 * gtIncrementScale β s 0 q * z))
      ≤ standardGaussianExpectation (fun z =>
        Real.log (Real.cosh (X + gtIncrementScale β s 0 q * z))
          + Real.log (Real.cosh (X + -1 * gtIncrementScale β s 0 q * z))
          + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2)) := by
    unfold standardGaussianExpectation
    exact integral_mono_of_nonneg (Filter.Eventually.of_forall hnn) hint
      (Filter.Eventually.of_forall hpt)
  exact hmono.trans_eq (flatness_two_logCosh_shift (gtIncrementScale β s 0 q) X _)

/-- **At zero multiplier and negative overlap the GT functional is at most
twice the replica-symmetric path value.** -/
lemma flatness_gtFunctional_zero_multiplier_neg_le
    (β h q s v : ℝ) (hβ : 0 ≤ β) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 < q) (hq1 : q < 1) (hv : v ∈ Set.Ico (-1 : ℝ) 0) :
    gtFunctional β h q s 0 v ≤ 2 * rsPathValue β h q s := by
  have hvneg : v < 0 := hv.2
  have hsign : gtPathSign v = -1 := by
    unfold gtPathSign
    rw [if_neg (not_le.mpr hvneg)]
  have hvabs : |v| = -v := abs_of_neg hvneg
  have hv0 : 0 < |v| := abs_pos.mpr (ne_of_lt hvneg)
  have hv1 : |v| ≤ 1 := by rw [hvabs]; linarith [hv.1]
  by_cases hvq : |v| < q
  · rw [flatness_gtFunctional_formula_abs_v_lt_q β h q s 0 v hv0 hvq]
    simp only [hsign, zero_mul, sub_zero]
    simp_rw [flatness_integrand_lower_neg β s q |v| _ hβ hs.1 hv0.le hvq.le]
    exact le_of_eq (flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le
      (gtIncrementScale_sq_of_nonneg β s q 1 hβ hs.1 hq1.le))
  · have hqv : q ≤ |v| := le_of_not_gt hvq
    have hassemble : ∀ r : ℝ, q ≤ r → r ≤ 1 →
        2 * Real.log 2
          + standardGaussianExpectation (fun z =>
            2 * flatnessPhi (gtIncrementScale β s 0 q)
              (h + β * Real.sqrt ((1 - s) * q) * z)
              + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2))
          - gtCorrection β q s = 2 * rsPathValue β h q s := by
      intro r hqr hr1
      refine flatness_zero_multiplier_assemble β h q s _ hβ hs hq.le ?_
      rw [gtIncrementScale_sq_of_nonneg β s q r hβ hs.1 hqr,
        gtIncrementScale_sq_of_nonneg β s r 1 hβ hs.1 hr1]
      ring
    have houter : ∀ r : ℝ,
        standardGaussianExpectation (fun z =>
          gtRankOneStep 0 (gtIncrementScale β s 0 q) (-1)
            (gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
              (gtDiagonalStep 1 (gtIncrementScale β s r 1) (gtTerminal 0)))
            (h + β * Real.sqrt ((1 - s) * q) * z)
            (h + β * Real.sqrt ((1 - s) * q) * z))
        ≤ standardGaussianExpectation (fun z =>
          2 * flatnessPhi (gtIncrementScale β s 0 q)
            (h + β * Real.sqrt ((1 - s) * q) * z)
            + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2)) := by
      intro r
      have hnn : ∀ z : ℝ, 0 ≤
          gtRankOneStep 0 (gtIncrementScale β s 0 q) (-1)
            (gtRankOneStep (1 / 2) (gtIncrementScale β s q r) (-1)
              (gtDiagonalStep 1 (gtIncrementScale β s r 1) (gtTerminal 0)))
            (h + β * Real.sqrt ((1 - s) * q) * z)
            (h + β * Real.sqrt ((1 - s) * q) * z) := by
        intro z
        rw [flatness_rankZero_apply, flatness_diag_one_terminal_fun]
        unfold standardGaussianExpectation
        exact integral_nonneg fun z₀ => flatness_step_rank_half_neg_nonneg _ _ _ _
      have hint : Integrable (fun z : ℝ =>
          2 * flatnessPhi (gtIncrementScale β s 0 q)
            (h + β * Real.sqrt ((1 - s) * q) * z)
            + (gtIncrementScale β s q r ^ 2 + gtIncrementScale β s r 1 ^ 2))
          (gaussianReal 0 1) :=
        ((flatness_integrable_Phi (gtIncrementScale β s 0 q) h
          (β * Real.sqrt ((1 - s) * q))).const_mul 2).add (integrable_const _)
      unfold standardGaussianExpectation
      exact integral_mono_of_nonneg (Filter.Eventually.of_forall hnn) hint
        (Filter.Eventually.of_forall fun z =>
          flatness_integrand_upper_neg_le β s q r _)
    rcases eq_or_lt_of_le hv1 with hone | hlt
    · rw [flatness_gtFunctional_formula_abs_v_eq_one β h q s 0 v hq hq1.le hone]
      simp only [hsign, zero_mul, sub_zero]
      have h1 := houter 1
      have h2 := hassemble 1 hq1.le le_rfl
      linarith
    · rw [flatness_gtFunctional_formula_q_le_abs_v_lt_one β h q s 0 v hq hqv hlt]
      simp only [hsign, zero_mul, sub_zero]
      have h1 := houter |v|
      have h2 := hassemble |v| hqv hv1
      linarith

/-! ### Negative overlaps -/

/--
Uniform separation of the endpoint multiplier derivative from zero on the
negative-overlap branch.

This is the negative-overlap counterpart of
`scalarOrderParameterCorrect_global_separation`.  Because the two replicas are
coupled with the opposite sign, the endpoint derivative is
`tilde g_s(v) - v`, where `tilde g_s` is the Gaussian two-point function of the
signed path; the AT condition makes `v ↦ tilde g_s(v) - v` strictly decreasing,
so on `[-1,0)` it stays above its value at `v = 0`, which is bounded below by
`c₂ * rsQ β h > 0`.  Formalizing the monotonicity step needs Price's identity
for the signed two-point function, which is not available here, so this
statement is left as the single remaining input.
-/
lemma flatness_deriv_gtFunctional_zero_neg_separation {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ κ > 0, ∀ {β h s v : ℝ},
      (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 → v ∈ Set.Ico (-1 : ℝ) 0 →
      κ ≤ |deriv (fun lam => gtFunctional β h (rsQ β h) s lam v) 0| := by
  sorry

/--
Uniform quadratic gap on the negative-overlap branch.

At `lam = 0` the value never exceeds twice the replica-symmetric path value
(`flatness_gtFunctional_zero_multiplier_neg_le`); combined with the uniform
separation of the multiplier derivative this yields a quadratic gap.
-/
lemma flatness_gtFunctional_gap_neg_overlap {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h s v : ℝ},
      (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 → v ∈ Set.Ico (-1 : ℝ) 0 →
      ∃ lam ∈ Set.Icc (-1 : ℝ) 1,
        gtFunctional β h (rsQ β h) s lam v ≤
          2 * rsPathValue β h (rsQ β h) s - c * (v - rsQ β h) ^ 2 := by
  obtain ⟨κ, hκ, hsep⟩ := flatness_deriv_gtFunctional_zero_neg_separation data
  refine ⟨(κ / 2) ^ 2 / 5, by positivity, ?_⟩
  intro β h s v hp hs hv
  have hβ : 0 < β := by simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by simpa using data.h_pos (β, h) hp
  have hq0 : 0 < rsQ β h := rsQ_pos hβ hh
  have hq1 : rsQ β h < 1 := rsQ_lt_one hβ hh
  have hzero := flatness_gtFunctional_zero_multiplier_neg_le β h (rsQ β h) s v
    hβ.le hs hq0 hq1 hv
  have habs : |v| ≤ 1 := by
    rw [abs_of_neg hv.2]
    linarith [hv.1]
  have hdist : |v - rsQ β h| ≤ 2 := by
    rw [abs_of_nonpos (by linarith [hv.2, hq0] : v - rsQ β h ≤ 0)]
    linarith [hv.1, hq1]
  have hgap : κ / 2 * |v - rsQ β h| ≤
      |deriv (fun lam => gtFunctional β h (rsQ β h) s lam v) 0| := by
    have h1 : κ / 2 * |v - rsQ β h| ≤ κ / 2 * 2 :=
      mul_le_mul_of_nonneg_left hdist (by positivity)
    have h2 := hsep hp hs hv
    linarith
  exact flatness_quadratic_gap_of_deriv_gap_mem β h (rsQ β h) s v (κ / 2)
    (by positivity) hzero
    (flatness_abs_deriv_gtFunctional_zero_le_two β h (rsQ β h) s v habs) hgap


theorem gtFunctional_uniform_quadratic_gap {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ c > 0, ∀ {β h q s v : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Icc (0 : ℝ) 1 →
      v ∈ Icc (-1 : ℝ) 1 →
      ∃ lam ∈ Icc (-1 : ℝ) 1, gtFunctional β h q s lam v ≤
        2 * rsPathValue β h q s - c * (v - q) ^ 2 := by
  obtain ⟨c₂, hc₂, hsep⟩ := scalarOrderParameterCorrect_global_separation data
  obtain ⟨cneg, hcneg, hneg⟩ := flatness_gtFunctional_gap_neg_overlap data
  refine ⟨min (c₂ ^ 2 / 5) cneg, lt_min (by positivity) hcneg, ?_⟩
  intro β h q s v hp hq hs hv
  subst hq
  have hβ : 0 < β := by simpa using data.β_pos (β, h) hp
  have hh : 0 < h := by simpa using data.h_pos (β, h) hp
  have hsqnn : (0 : ℝ) ≤ (v - rsQ β h) ^ 2 := sq_nonneg _
  rcases lt_or_ge v 0 with hvneg | hv0
  · obtain ⟨lam, hlam, hle⟩ := hneg hp hs ⟨hv.1, hvneg⟩
    refine ⟨lam, hlam, hle.trans ?_⟩
    have hmin : min (c₂ ^ 2 / 5) cneg ≤ cneg := min_le_right _ _
    nlinarith
  · have hv01 : v ∈ Icc (0 : ℝ) 1 := ⟨hv0, hv.2⟩
    have hzero := flatness_gtFunctional_zero_multiplier β h (rsQ β h) s v hβ.le hs
      (rsQ_pos hβ hh) (rsQ_lt_one hβ hh) hv01
    have hgap : c₂ * |v - rsQ β h| ≤
        |deriv (fun l => gtFunctional β h (rsQ β h) s l v) 0| := by
      rw [flatness_deriv_gtFunctional_zero_eq_g_sub β h s v hβ hh hs hv01]
      exact hsep hp hs hv01
    have habs : |v| ≤ 1 := abs_le.mpr ⟨hv.1, hv.2⟩
    obtain ⟨lam, hlam, hle⟩ :=
      flatness_quadratic_gap_of_deriv_gap_mem β h (rsQ β h) s v c₂ hc₂.le hzero.le
        (flatness_abs_deriv_gtFunctional_zero_le_two β h (rsQ β h) s v habs) hgap
    refine ⟨lam, hlam, hle.trans ?_⟩
    have hmin : min (c₂ ^ 2 / 5) cneg ≤ c₂ ^ 2 / 5 := min_le_left _ _
    nlinarith

end SpinGlass.AT
