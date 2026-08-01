import Lemmas.Scalar.Semigroup

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

noncomputable def latalaH (t y : ℝ) : ℝ :=
  (1 + (4 - 3 * y) * t) / (1 + y * t) ^ 3

noncomputable def latalaF (lam y : ℝ) : ℝ :=
  Real.exp (-lam / 2) * standardGaussianExpectation (fun z =>
    Real.cosh (Real.sqrt lam * z) *
      latalaH (Real.sinh (Real.sqrt lam * z) ^ 2) y)

noncomputable def referenceDensity (y : ℝ) : ℝ :=
  1 / (2 * Real.sqrt (1 - y))

noncomputable def referenceExpectation (f : ℝ → ℝ) : ℝ :=
  ∫ y in Set.Icc (0 : ℝ) 1, f y * referenceDensity y

theorem latalaH_hasDerivAt {t y : ℝ} (hden : 1 + y * t ≠ 0) :
    HasDerivAt (latalaH t)
      (-6 * t * (1 + (2 - y) * t) / (1 + y * t) ^ 4) y := by
  sorry

theorem latalaH_deriv_nonpos {t y : ℝ} (ht : 0 ≤ t)
    (hy : y ∈ Set.Icc (0 : ℝ) 1) :
    deriv (latalaH t) y ≤ 0 := by
  sorry

theorem latalaF_antitone {lam : ℝ} (hlam : 0 ≤ lam) :
    AntitoneOn (latalaF lam) (Set.Icc (0 : ℝ) 1) := by
  sorry

theorem opposite_monotone_covariance_le
    {α : Type*} [LinearOrder α] [MeasurableSpace α] {μ : Measure α}
    [IsProbabilityMeasure μ] {f g : α → ℝ}
    (hf : Monotone f) (hg : Antitone g)
    (hfi : Integrable f μ) (hgi : Integrable g μ)
    (hfg : Integrable (fun x => f x * g x) μ) :
    (∫ x, f x * g x ∂μ) ≤ (∫ x, f x ∂μ) * ∫ x, g x ∂μ := by
  let If : ℝ := ∫ x, f x ∂μ
  let Ig : ℝ := ∫ x, g x ∂μ
  let Ifg : ℝ := ∫ x, f x * g x ∂μ
  have hpoint (x y : α) : (f x - f y) * (g x - g y) ≤ 0 := by
    rcases le_total x y with hxy | hyx
    · exact mul_nonpos_of_nonpos_of_nonneg
        (sub_nonpos.mpr (hf hxy)) (sub_nonneg.mpr (hg hxy))
    · exact mul_nonpos_of_nonneg_of_nonpos
        (sub_nonneg.mpr (hf hyx)) (sub_nonpos.mpr (hg hyx))
  have hinner (x : α) :
      f x * g x - f x * Ig - If * g x + Ifg ≤ 0 := by
    have hnonpos : (∫ y, (f x - f y) * (g x - g y) ∂μ) ≤ 0 :=
      integral_nonpos (hpoint x)
    have hconst : Integrable (fun _ : α => f x * g x) μ := integrable_const _
    have hfxg : Integrable (fun y => f x * g y) μ := hgi.const_mul (f x)
    have hfgx : Integrable (fun y => f y * g x) μ := hfi.mul_const (g x)
    have hexpand :
        (∫ y, (f x - f y) * (g x - g y) ∂μ) =
          f x * g x - f x * Ig - If * g x + Ifg := by
      calc
        _ = ∫ y, (f x * g x - f x * g y) -
            (f y * g x - f y * g y) ∂μ := by
              congr 1
              funext y
              ring
        _ = (∫ y, f x * g x - f x * g y ∂μ) -
            ∫ y, f y * g x - f y * g y ∂μ :=
              integral_sub (hconst.sub hfxg) (hfgx.sub hfg)
        _ = _ := by
          rw [integral_sub hconst hfxg, integral_sub hfgx hfg]
          rw [integral_const_mul (μ := μ) (f x) g,
            integral_mul_const (μ := μ) (g x) f]
          simp [If, Ig, Ifg]
          ring
    rw [hexpand] at hnonpos
    exact hnonpos
  have houter :
      (∫ x, f x * g x - f x * Ig - If * g x + Ifg ∂μ) ≤ 0 :=
    integral_nonpos hinner
  have hconstIfg : Integrable (fun _ : α => Ifg) μ := integrable_const _
  have hfIg : Integrable (fun x => f x * Ig) μ := hfi.mul_const Ig
  have hIfg : Integrable (fun x => If * g x) μ := hgi.const_mul If
  have hexpand :
      (∫ x, f x * g x - f x * Ig - If * g x + Ifg ∂μ) =
        2 * Ifg - 2 * If * Ig := by
    calc
      _ = (∫ x, f x * g x - f x * Ig - If * g x ∂μ) +
          ∫ _x : α, Ifg ∂μ :=
            integral_add ((hfg.sub hfIg).sub hIfg) hconstIfg
      _ = ((∫ x, f x * g x - f x * Ig ∂μ) -
          ∫ x, If * g x ∂μ) + ∫ _x : α, Ifg ∂μ := by
            exact congrArg (fun z : ℝ => z + ∫ _x : α, Ifg ∂μ)
              (integral_sub (hfg.sub hfIg) hIfg)
      _ = _ := by
        rw [integral_sub hfg hfIg]
        rw [integral_mul_const, integral_const_mul]
        simp [If, Ig, Ifg]
        ring
  rw [hexpand] at houter
  dsimp [If, Ig, Ifg] at houter ⊢
  linarith

theorem latalaF_reference_mean_one {lam : ℝ} (hlam : 0 ≤ lam) :
    referenceExpectation (latalaF lam) = 1 := by
  sorry

theorem latala_weighted_kernel_le {lam : ℝ} (hlam : 0 ≤ lam)
    {rho : ℝ → ℝ} (hrho : MonotoneOn rho (Set.Icc (0 : ℝ) 1))
    (hprob : referenceExpectation rho = 1)
    (hRhoInt : IntegrableOn (fun y => rho y * referenceDensity y)
      (Set.Icc (0 : ℝ) 1))
    (hFInt : IntegrableOn (fun y => latalaF lam y * referenceDensity y)
      (Set.Icc (0 : ℝ) 1))
    (hInt : IntegrableOn (fun y => rho y * latalaF lam y * referenceDensity y)
      (Set.Icc (0 : ℝ) 1)) :
    referenceExpectation (fun y => rho y * latalaF lam y) ≤ 1 := by
  sorry

end SpinGlass.AT
