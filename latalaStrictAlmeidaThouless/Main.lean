import Lemmas.Price

open MeasureTheory ProbabilityTheory Set Matrix
open scoped RealInnerProductSpace
noncomputable section

private abbrev P := Fin 2

private def obs (h : ℝ) : EuclideanSpace ℝ P → ℝ :=
  fun x => Real.tanh (h + x 0) * Real.tanh (h + x 1)

example (h : ℝ) : ContDiff ℝ 2 (obs h) := by
  unfold obs
  simp_rw [Real.tanh_eq_sinh_div_cosh]
  apply ((Real.contDiff_sinh.comp (by fun_prop)).div
    (Real.contDiff_cosh.comp (by fun_prop)) (fun x => (Real.cosh_pos _).ne')).mul
  exact (Real.contDiff_sinh.comp (by fun_prop)).div
    (Real.contDiff_cosh.comp (by fun_prop)) (fun x => (Real.cosh_pos _).ne')

example (h : ℝ) (x : EuclideanSpace ℝ P) :
    fderiv ℝ (obs h) x =
      Real.tanh (h + x 1) •
          (ProbabilityTheory.PriceTanh.sechSq (h + x 0) •
            EuclideanSpace.proj (𝕜 := ℝ) (0 : P)) +
        Real.tanh (h + x 0) •
          (ProbabilityTheory.PriceTanh.sechSq (h + x 1) •
            EuclideanSpace.proj (𝕜 := ℝ) (1 : P)) := by
  have h0 : HasFDerivAt (fun y : EuclideanSpace ℝ P => h + y 0)
      (EuclideanSpace.proj (𝕜 := ℝ) (0 : P)) x := by
    simpa using (EuclideanSpace.proj (𝕜 := ℝ) (0 : P)).hasFDerivAt.const_add h
  have h1 : HasFDerivAt (fun y : EuclideanSpace ℝ P => h + y 1)
      (EuclideanSpace.proj (𝕜 := ℝ) (1 : P)) x := by
    simpa using (EuclideanSpace.proj (𝕜 := ℝ) (1 : P)).hasFDerivAt.const_add h
  have ht0 := (ProbabilityTheory.PriceTanh.tanh_hasDerivAt (h + x 0)).hasFDerivAt.comp x h0
  have ht1 := (ProbabilityTheory.PriceTanh.tanh_hasDerivAt (h + x 1)).hasFDerivAt.comp x h1
  have hp := ht0.mul ht1
  change fderiv ℝ ((Real.tanh ∘ fun y : EuclideanSpace ℝ P => h + y 0) *
    (Real.tanh ∘ fun y : EuclideanSpace ℝ P => h + y 1)) x = _
  rw [hp.fderiv]
  ext y
  simp
  ring

private lemma integral_stdGaussian_fin_two
    (f : EuclideanSpace ℝ P → ℝ) (hf : Continuous f)
    {C : ℝ} (hb : ∀ x, |f x| ≤ C) :
    ∫ w, f w ∂ProbabilityTheory.PriceGaussian.stdGaussian P =
      ∫ x : ℝ, ∫ y : ℝ, f (WithLp.toLp 2 ![x, y])
        ∂gaussianReal 0 1 ∂gaussianReal 0 1 := by
  let γ : Measure ℝ := gaussianReal 0 1
  have hpi : MeasurePreserving
      (MeasurableEquiv.finTwoArrow : (P → ℝ) ≃ᵐ (ℝ × ℝ))
      (Measure.pi fun _ : P => γ) (γ.prod γ) :=
    measurePreserving_finTwoArrow γ
  have hgcont : Continuous (fun p : ℝ × ℝ =>
      f (WithLp.toLp 2 ![p.1, p.2])) := by fun_prop
  have hgint : Integrable (fun p : ℝ × ℝ =>
      f (WithLp.toLp 2 ![p.1, p.2])) (γ.prod γ) := by
    apply Integrable.of_bound (C := C)
    · exact hgcont.aestronglyMeasurable
    · filter_upwards [] with p
      simpa [Real.norm_eq_abs] using hb (WithLp.toLp 2 ![p.1, p.2])
  rw [ProbabilityTheory.PriceGaussian.stdGaussian,
    integral_map (by fun_prop) hf.aestronglyMeasurable]
  calc
    (∫ x : P → ℝ, f (WithLp.toLp 2 x) ∂Measure.pi fun _ : P => γ) =
        ∫ p : ℝ × ℝ,
          f (WithLp.toLp 2 ![p.1, p.2]) ∂(γ.prod γ) := by
      have heq := hpi.integral_comp MeasurableEquiv.finTwoArrow.measurableEmbedding
        (fun p : ℝ × ℝ => f (WithLp.toLp 2 ![p.1, p.2]))
      convert heq using 1
      apply integral_congr_ae
      filter_upwards [] with x
      congr 2
      ext i
      fin_cases i <;> rfl
    _ = ∫ x : ℝ, ∫ y : ℝ, f (WithLp.toLp 2 ![x, y]) ∂γ ∂γ := by
      rw [integral_prod _ hgint]
