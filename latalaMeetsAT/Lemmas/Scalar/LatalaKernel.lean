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
  have hden' : 1 + t * y ≠ 0 := by simpa [mul_comm] using hden
  unfold latalaH
  convert
    (((hasDerivAt_const y 1).add
      (((hasDerivAt_const y 4).sub ((hasDerivAt_id y).const_mul 3)).mul_const t)).div
        (((hasDerivAt_const y 1).add ((hasDerivAt_id y).mul_const t)).pow 3)
        (pow_ne_zero 3 hden)) using 1
  all_goals first | rfl | (
    simp only [Pi.add_apply, Pi.sub_apply, Pi.pow_apply, id_eq,
      one_mul, zero_add, Nat.cast_ofNat]
    field_simp [hden, hden']
    ring)

theorem latalaH_deriv_nonpos {t y : ℝ} (ht : 0 ≤ t)
    (hy : y ∈ Set.Icc (0 : ℝ) 1) :
    deriv (latalaH t) y ≤ 0 := by
  have hdenpos : 0 < 1 + y * t := by
    nlinarith [mul_nonneg hy.1 ht]
  rw [(latalaH_hasDerivAt (ne_of_gt hdenpos)).deriv]
  have hfac : 0 ≤ 1 + (2 - y) * t := by
    have : 0 ≤ 2 - y := by linarith [hy.2]
    positivity
  have hneg : -6 * t ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (by norm_num) ht
  exact div_nonpos_of_nonpos_of_nonneg
    (mul_nonpos_of_nonpos_of_nonneg hneg hfac) (by positivity)

private theorem cosh_mul_one_add_four_sinh_sq (x : ℝ) :
    Real.cosh x * (1 + 4 * Real.sinh x ^ 2) = Real.cosh (3 * x) := by
  have hsq : Real.sinh x ^ 2 = Real.cosh x ^ 2 - 1 := by
    nlinarith [Real.cosh_sq_sub_sinh_sq x]
  rw [show 3 * x = x + (x + x) by ring, Real.cosh_add, Real.cosh_add, Real.sinh_add]
  calc
    Real.cosh x * (1 + 4 * Real.sinh x ^ 2) =
        Real.cosh x * (4 * Real.cosh x ^ 2 - 3) := by rw [hsq]; ring
    _ = Real.cosh x ^ 3 + 3 * Real.cosh x * Real.sinh x ^ 2 := by rw [hsq]; ring
    _ = Real.cosh x * (Real.cosh x * Real.cosh x + Real.sinh x * Real.sinh x) +
        Real.sinh x * (Real.sinh x * Real.cosh x + Real.cosh x * Real.sinh x) := by ring

private theorem integrable_cosh_mul_gaussian (c : ℝ) :
    Integrable (fun z : ℝ => Real.cosh (c * z)) (gaussianReal 0 1) := by
  simp_rw [Real.cosh_eq]
  apply Integrable.div_const
  apply Integrable.add
  · exact integrable_exp_mul_gaussianReal c
  · simpa only [neg_mul] using integrable_exp_mul_gaussianReal (-c)

private theorem latalaH_nonneg_le {t y : ℝ} (ht : 0 ≤ t)
    (hy : y ∈ Set.Icc (0 : ℝ) 1) :
    0 ≤ latalaH t y ∧ latalaH t y ≤ 1 + 4 * t := by
  have hyt : 0 ≤ y * t := mul_nonneg hy.1 ht
  have hden : 0 < 1 + y * t := by linarith
  have hnum0 : 0 ≤ 1 + (4 - 3 * y) * t := by
    have : 1 ≤ 4 - 3 * y := by linarith [hy.2]
    nlinarith
  have hnum : 1 + (4 - 3 * y) * t ≤ 1 + 4 * t := by nlinarith
  constructor
  · exact div_nonneg hnum0 (pow_nonneg hden.le 3)
  · unfold latalaH
    rw [div_le_iff₀ (pow_pos hden 3)]
    have hp : 1 ≤ (1 + y * t) ^ 3 := one_le_pow₀ (by linarith)
    nlinarith

theorem latalaF_antitone {lam : ℝ} (hlam : 0 ≤ lam) :
    AntitoneOn (latalaF lam) (Set.Icc (0 : ℝ) 1) := by
  let a := Real.sqrt lam
  have hHanti : ∀ z : ℝ, AntitoneOn
      (fun y => latalaH (Real.sinh (a * z) ^ 2) y) (Set.Icc (0 : ℝ) 1) := by
    intro z
    apply antitoneOn_of_deriv_nonpos (convex_Icc (0 : ℝ) 1)
    · intro y hy
      exact (latalaH_hasDerivAt (by
        have : 0 < 1 + y * Real.sinh (a * z) ^ 2 := by
          nlinarith [hy.1, sq_nonneg (Real.sinh (a * z))]
        exact ne_of_gt this)).continuousAt.continuousWithinAt
    · intro y hy
      exact (latalaH_hasDerivAt (by
        have hy' : y ∈ Set.Icc (0 : ℝ) 1 := interior_subset hy
        have : 0 < 1 + y * Real.sinh (a * z) ^ 2 := by
          nlinarith [hy'.1, sq_nonneg (Real.sinh (a * z))]
        exact ne_of_gt this)).differentiableAt.differentiableWithinAt
    · intro y hy
      exact latalaH_deriv_nonpos (sq_nonneg _) (interior_subset hy)
  have hInt : ∀ y ∈ Set.Icc (0 : ℝ) 1,
      Integrable (fun z => Real.cosh (a * z) *
        latalaH (Real.sinh (a * z) ^ 2) y) (gaussianReal 0 1) := by
    intro y hy
    apply (integrable_cosh_mul_gaussian (3 * a)).mono'
    · apply Continuous.aestronglyMeasurable
      apply Continuous.mul
      · fun_prop
      · unfold latalaH
        apply Continuous.div
        · fun_prop
        · fun_prop
        · intro z
          apply pow_ne_zero
          have : 0 < 1 + y * Real.sinh (a * z) ^ 2 := by
            nlinarith [hy.1, sq_nonneg (Real.sinh (a * z))]
          exact ne_of_gt this
    · filter_upwards [] with z
      rw [Real.norm_eq_abs]
      have hb := latalaH_nonneg_le (sq_nonneg (Real.sinh (a * z))) hy
      rw [abs_of_nonneg (mul_nonneg (Real.cosh_pos _).le hb.1)]
      calc
        Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y ≤
            Real.cosh (a * z) * (1 + 4 * Real.sinh (a * z) ^ 2) :=
          mul_le_mul_of_nonneg_left hb.2 (Real.cosh_pos _).le
        _ = Real.cosh (3 * (a * z)) := cosh_mul_one_add_four_sinh_sq _
        _ = Real.cosh ((3 * a) * z) := by ring_nf
  unfold latalaF standardGaussianExpectation
  have hanti : AntitoneOn (fun y => ∫ z, Real.cosh (a * z) *
      latalaH (Real.sinh (a * z) ^ 2) y ∂gaussianReal 0 1)
      (Set.Icc (0 : ℝ) 1) :=
    integral_antitoneOn_of_integrand_ae
      (ae_of_all _ fun z => by
        intro x hx y hy hxy
        exact mul_le_mul_of_nonneg_left (hHanti z hx hy hxy) (Real.cosh_pos _).le)
      hInt
  intro x hx y hy hxy
  exact mul_le_mul_of_nonneg_left (hanti hx hy hxy) (Real.exp_pos _).le

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

private theorem latalaH_reference_integral (t : ℝ) (ht : 0 ≤ t) :
    ∫ y in Set.Icc (0 : ℝ) 1, latalaH t y * referenceDensity y = 1 := by
  let f : ℝ → ℝ := fun u => 1 - u ^ 2
  let f' : ℝ → ℝ := fun u => -2 * u
  let g : ℝ → ℝ := fun y => latalaH t y * referenceDensity y
  have hsub := intervalIntegral.integral_comp_mul_deriv_of_deriv_nonpos
    (a := (0 : ℝ)) (b := 1) (f := f) (f' := f') (g := g)
    (by fun_prop)
    (by
      intro u hu
      have hbase := (hasDerivAt_const u 1).sub ((hasDerivAt_id u).pow 2)
      have hf_eq : f = (fun x : ℝ => 1) - id ^ 2 := by
        funext x
        simp [f]
      have hfp_eq : f' u = (0 : ℝ) - 2 * id u ^ (2 - 1) * 1 := by
        norm_num [f']
      rw [hf_eq, hfp_eq]
      exact hbase)
    (by
      intro u hu
      norm_num at hu
      dsimp [f']
      linarith [hu.1])
  have hcomp :
      (∫ u in (0 : ℝ)..1, (g ∘ f) u * f' u) =
        ∫ u in (0 : ℝ)..1, -latalaH t (1 - u ^ 2) := by
    apply intervalIntegral.integral_congr_ae
    filter_upwards [] with u
    intro hu
    have huIoc : u ∈ Set.Ioc (0 : ℝ) 1 := by
      simpa [Set.uIoc_of_le zero_le_one] using hu
    have huI : u ∈ Set.Icc (0 : ℝ) 1 := ⟨huIoc.1.le, huIoc.2⟩
    have hu0 : u ≠ 0 := ne_of_gt huIoc.1
    have hsqrt : Real.sqrt (u ^ 2) = u := by
      rw [Real.sqrt_sq_eq_abs, abs_of_nonneg huI.1]
    dsimp [g, f, f', referenceDensity]
    rw [show 1 - (1 - u ^ 2) = u ^ 2 by ring, hsqrt]
    field_simp
  rw [hcomp] at hsub
  dsimp [f, g] at hsub
  norm_num at hsub
  have hrat : (∫ u in (0 : ℝ)..1, latalaH t (1 - u ^ 2)) = 1 := by
    let F : ℝ → ℝ := fun u => u / (1 + (1 - u ^ 2) * t) ^ 2
    have hderiv : ∀ u ∈ Set.uIcc (0 : ℝ) 1,
        HasDerivAt F (latalaH t (1 - u ^ 2)) u := by
      intro u hu
      have huI : u ∈ Set.Icc (0 : ℝ) 1 := by
        simpa [Set.uIcc_of_le zero_le_one] using hu
      have hdenpos : 0 < 1 + (1 - u ^ 2) * t := by
        have husq : u ^ 2 ≤ 1 := by
          nlinarith [mul_self_le_mul_self huI.1 huI.2]
        nlinarith
      have hden : 1 + (1 - u ^ 2) * t ≠ 0 := ne_of_gt hdenpos
      let D : ℝ → ℝ := fun x => 1 + (1 - x ^ 2) * t
      have hD : HasDerivAt D (-2 * u * t) u := by
        have hraw := (hasDerivAt_const u 1).add
          (((hasDerivAt_const u 1).sub ((hasDerivAt_id u).pow 2)).mul_const t)
        convert hraw using 1
        all_goals first | rfl | (
          simp only [D, Pi.add_apply, Pi.sub_apply, Pi.pow_apply, id_eq,
            zero_add, zero_sub, one_mul, pow_succ, pow_zero, mul_one]
          ring)
      have hbase : HasDerivAt (fun x => x / D x ^ 2)
          ((1 * D u ^ 2 - u * (2 * D u * (-2 * u * t))) / (D u ^ 2) ^ 2) u := by
        convert (hasDerivAt_id u).div (hD.pow 2) (pow_ne_zero 2 hden) using 1
        all_goals first | rfl | (
          simp only [Pi.div_apply, Pi.pow_apply, id_eq, one_mul, Nat.cast_ofNat,
            pow_succ, pow_zero, mul_one])
      have hvalue :
          latalaH t (1 - u ^ 2) =
            (1 * D u ^ 2 - u * (2 * D u * (-2 * u * t))) / (D u ^ 2) ^ 2 := by
        dsimp [D]
        unfold latalaH
        field_simp [hden]
        ring
      change HasDerivAt (fun x => x / D x ^ 2) (latalaH t (1 - u ^ 2)) u
      rw [hvalue]
      exact hbase
    have hcont : ContinuousOn (fun u : ℝ => latalaH t (1 - u ^ 2))
        (Set.uIcc (0 : ℝ) 1) := by
      intro u hu
      have huI : u ∈ Set.Icc (0 : ℝ) 1 := by
        simpa [Set.uIcc_of_le zero_le_one] using hu
      unfold latalaH
      apply ContinuousAt.continuousWithinAt
      apply ContinuousAt.div
      · fun_prop
      · fun_prop
      · apply pow_ne_zero
        have husq : u ^ 2 ≤ 1 := by
          nlinarith [mul_self_le_mul_self huI.1 huI.2]
        have : 0 < 1 + (1 - u ^ 2) * t := by nlinarith
        exact ne_of_gt this
    have hint : IntervalIntegrable (fun u : ℝ => latalaH t (1 - u ^ 2)) volume 0 1 :=
      ContinuousOn.intervalIntegrable hcont
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint]
    simp [F]
  rw [hrat] at hsub
  rw [intervalIntegral.integral_symm] at hsub
  rw [integral_Icc_eq_integral_Ioc,
    ← intervalIntegral.integral_of_le zero_le_one]
  linarith

private theorem referenceDensity_integrable :
    IntegrableOn referenceDensity (Set.Icc (0 : ℝ) 1) := by
  let f : ℝ → ℝ := fun u => 1 - u ^ 2
  let f' : ℝ → ℝ := fun u => -2 * u
  have hiff := intervalIntegral.integrable_comp_mul_deriv_iff_of_deriv_nonpos
    (a := (0 : ℝ)) (b := 1) (f := f) (f' := f') (g := referenceDensity)
    (by fun_prop)
    (by
      intro u hu
      have hbase := (hasDerivAt_const u 1).sub ((hasDerivAt_id u).pow 2)
      have hf_eq : f = (fun x : ℝ => 1) - id ^ 2 := by
        funext x
        simp [f]
      have hfp_eq : f' u = (0 : ℝ) - 2 * id u ^ (2 - 1) * 1 := by
        norm_num [f']
      rw [hf_eq, hfp_eq]
      exact hbase)
    (by
      intro u hu
      norm_num at hu
      dsimp [f']
      linarith [hu.1])
  have hleft : IntervalIntegrable (fun u => (referenceDensity ∘ f) u * f' u)
      volume 0 1 := by
    apply (intervalIntegrable_const (μ := volume) (a := (0 : ℝ)) (b := 1)
      (c := (-1 : ℝ))).congr_uIoo
    intro u hu
    norm_num at hu
    have huI : u ∈ Set.Icc (0 : ℝ) 1 := ⟨hu.1.le, hu.2.le⟩
    have hsqrt : Real.sqrt (u ^ 2) = u := by
      rw [Real.sqrt_sq_eq_abs, abs_of_nonneg huI.1]
    dsimp [referenceDensity, f, f']
    rw [show 1 - (1 - u ^ 2) = u ^ 2 by ring, hsqrt]
    field_simp [ne_of_gt hu.1]
  have hright : IntervalIntegrable referenceDensity volume 0 1 := by
    simpa [f] using (hiff.mp hleft).symm
  exact (intervalIntegrable_iff_integrableOn_Icc_of_le zero_le_one).mp hright

theorem latalaF_reference_mean_one {lam : ℝ} (hlam : 0 ≤ lam) :
    referenceExpectation (latalaF lam) = 1 := by
  let I : Set ℝ := Set.Icc 0 1
  let ν : Measure ℝ := volume.restrict I
  let a := Real.sqrt lam
  have hdom : Integrable (fun p : ℝ × ℝ => referenceDensity p.1 *
      Real.cosh ((3 * a) * p.2)) (ν.prod (gaussianReal 0 1)) := by
    exact (show Integrable referenceDensity ν from referenceDensity_integrable).mul_prod
      (integrable_cosh_mul_gaussian (3 * a))
  have hk : Integrable (Function.uncurry (fun y z => referenceDensity y *
      (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y)))
      (ν.prod (gaussianReal 0 1)) := by
    apply hdom.mono'
    · apply Measurable.aestronglyMeasurable
      dsimp [Function.uncurry, referenceDensity, latalaH]
      fun_prop
    · have hp : ∀ᵐ p ∂ν.prod (gaussianReal 0 1), p.1 ∈ I := by
        rw [Measure.ae_prod_iff_ae_ae
          (measurableSet_Icc.preimage measurable_fst)]
        filter_upwards [MeasureTheory.ae_restrict_mem (μ := volume) measurableSet_Icc] with y hy
        exact Filter.Eventually.of_forall fun _ => hy
      filter_upwards [hp] with p hpI
      have hb := latalaH_nonneg_le (sq_nonneg (Real.sinh (a * p.2))) hpI
      have hw : 0 ≤ referenceDensity p.1 := by
        dsimp [referenceDensity]
        positivity
      change |referenceDensity p.1 *
          (Real.cosh (a * p.2) * latalaH (Real.sinh (a * p.2) ^ 2) p.1)| ≤
        referenceDensity p.1 * Real.cosh ((3 * a) * p.2)
      rw [abs_of_nonneg (mul_nonneg hw (mul_nonneg (Real.cosh_pos _).le hb.1))]
      apply mul_le_mul_of_nonneg_left _ hw
      calc
        Real.cosh (a * p.2) * latalaH (Real.sinh (a * p.2) ^ 2) p.1 ≤
            Real.cosh (a * p.2) * (1 + 4 * Real.sinh (a * p.2) ^ 2) :=
          mul_le_mul_of_nonneg_left hb.2 (Real.cosh_pos _).le
        _ = Real.cosh (3 * (a * p.2)) := cosh_mul_one_add_four_sinh_sq _
        _ = Real.cosh ((3 * a) * p.2) := by ring_nf
  have hswap := MeasureTheory.integral_integral_swap
    (f := fun y z => referenceDensity y *
      (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y))
    hk
  dsimp [ν] at hswap
  unfold referenceExpectation latalaF standardGaussianExpectation
  calc
    (∫ y in Set.Icc (0 : ℝ) 1,
        (Real.exp (-lam / 2) *
          ∫ z, Real.cosh (Real.sqrt lam * z) *
            latalaH (Real.sinh (Real.sqrt lam * z) ^ 2) y
            ∂gaussianReal 0 1) * referenceDensity y) =
        Real.exp (-lam / 2) *
          ∫ y in I, ∫ z, referenceDensity y *
            (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y)
            ∂gaussianReal 0 1 := by
      rw [← MeasureTheory.integral_const_mul]
      apply MeasureTheory.integral_congr_ae
      filter_upwards with y
      rw [MeasureTheory.integral_const_mul]
      dsimp [a]
      ring
    _ = Real.exp (-lam / 2) *
        ∫ z, (∫ y in I, referenceDensity y *
          (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y))
          ∂gaussianReal 0 1 := by rw [hswap]
    _ = 1 := by
      have hinner : ∀ z : ℝ,
          (∫ y in I, referenceDensity y *
            (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y)) =
            Real.cosh (a * z) := by
        intro z
        calc
          (∫ y in I, referenceDensity y *
              (Real.cosh (a * z) * latalaH (Real.sinh (a * z) ^ 2) y)) =
              Real.cosh (a * z) * ∫ y in I,
                latalaH (Real.sinh (a * z) ^ 2) y * referenceDensity y := by
            rw [← MeasureTheory.integral_const_mul]
            apply MeasureTheory.integral_congr_ae
            filter_upwards with y
            ring
          _ = Real.cosh (a * z) := by
            rw [latalaH_reference_integral _ (sq_nonneg _)]
            ring
      simp_rw [hinner]
      have ha_sq : a ^ 2 = lam := by
        dsimp [a]
        exact Real.sq_sqrt hlam
      have hexp (c : ℝ) :
          (∫ z, Real.exp (c * z) ∂gaussianReal 0 1) = Real.exp (c ^ 2 / 2) := by
        change ProbabilityTheory.mgf id (gaussianReal 0 1) c = Real.exp (c ^ 2 / 2)
        rw [congrFun (mgf_id_gaussianReal (μ := 0) (v := 1)) c]
        congr 1
        norm_num
      rw [show (∫ z, Real.cosh (a * z) ∂gaussianReal 0 1) = Real.exp (lam / 2) by
        simp_rw [Real.cosh_eq]
        rw [integral_div, integral_add
          (integrable_exp_mul_gaussianReal a) (by
            simpa only [neg_mul] using integrable_exp_mul_gaussianReal (-a))]
        rw [hexp]
        rw [show (fun z : ℝ => Real.exp (-(a * z))) =
            fun z => Real.exp ((-a) * z) by
          funext z
          congr 1
          ring, hexp]
        rw [show (-a) ^ 2 = a ^ 2 by ring, ha_sq]
        ring]
      rw [← Real.exp_add]
      rw [show -lam / 2 + lam / 2 = 0 by ring, Real.exp_zero]

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
  let I : Set ℝ := Set.Icc 0 1
  let w : ℝ → ℝ := referenceDensity
  let μ : Measure ℝ :=
    (volume.restrict I).withDensity (fun y => ENNReal.ofReal (w y))
  have hw : Measurable w := by
    dsimp [w, referenceDensity]
    exact measurable_const.div
      (measurable_const.mul (Real.continuous_sqrt.measurable.comp
        (measurable_const.sub measurable_id)))
  have hw0 : ∀ y ∈ I, 0 ≤ w y := by
    intro y hy
    dsimp [w, referenceDensity]
    positivity
  have hint (f : ℝ → ℝ) :
      (∫ y, f y ∂μ) = ∫ y in I, f y * w y := by
    dsimp [μ]
    rw [integral_withDensity_eq_integral_toReal_smul hw.ennreal_ofReal
      (ae_of_all _ fun y => ENNReal.ofReal_ne_top.lt_top)]
    apply integral_congr_ae
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    rw [ENNReal.toReal_ofReal (hw0 y hy)]
    simp [smul_eq_mul, mul_comm]
  have hFzero : ∀ y, latalaF 0 y = 1 := by
    intro y
    simp [latalaF, standardGaussianExpectation, latalaH]
  have hmass : μ Set.univ = 1 := by
    have hzero := latalaF_reference_mean_one (lam := 0) (by norm_num)
    rw [show referenceExpectation (latalaF 0) =
        referenceExpectation (fun _ => 1) by
      congr 1
      funext y
      exact hFzero y] at hzero
    have hone : (∫ _y, (1 : ℝ) ∂μ) = 1 := by
      rw [hint]
      simpa [referenceExpectation, I, w, mul_comm] using hzero
    exact (ENNReal.toReal_eq_one_iff (μ Set.univ)).mp (by
      rw [← measureReal_def]
      simpa using hone)
  letI : IsProbabilityMeasure μ := ⟨hmass⟩
  let clamp : ℝ → ℝ := fun y =>
    (Set.projIcc 0 1 zero_le_one y : ℝ)
  let f : ℝ → ℝ := fun y => rho (clamp y)
  let g : ℝ → ℝ := fun y => latalaF lam (clamp y)
  have hclamp_mem (y : ℝ) : clamp y ∈ I := by
    exact (Set.projIcc 0 1 zero_le_one y).property
  have hclamp_mono : Monotone clamp := by
    exact fun _ _ hxy => Set.monotone_projIcc zero_le_one hxy
  have hf : Monotone f := fun x y hxy =>
    hrho (hclamp_mem x) (hclamp_mem y) (hclamp_mono hxy)
  have hg : Antitone g := fun x y hxy =>
    latalaF_antitone hlam (hclamp_mem x) (hclamp_mem y) (hclamp_mono hxy)
  have hclamp_eq : ∀ y ∈ I, clamp y = y := by
    intro y hy
    simp [clamp, Set.coe_projIcc, min_eq_right hy.2, max_eq_right hy.1]
  have hfi : Integrable f μ := by
    dsimp [μ]
    rw [integrable_withDensity_iff hw.ennreal_ofReal
      (ae_of_all _ fun y => ENNReal.ofReal_ne_top.lt_top)]
    apply hRhoInt.congr
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [f, w]
    rw [ENNReal.toReal_ofReal (hw0 y hy), hclamp_eq y hy]
  have hgi : Integrable g μ := by
    dsimp [μ]
    rw [integrable_withDensity_iff hw.ennreal_ofReal
      (ae_of_all _ fun y => ENNReal.ofReal_ne_top.lt_top)]
    apply hFInt.congr
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [g, w]
    rw [ENNReal.toReal_ofReal (hw0 y hy), hclamp_eq y hy]
  have hfgi : Integrable (fun y => f y * g y) μ := by
    dsimp [μ]
    rw [integrable_withDensity_iff hw.ennreal_ofReal
      (ae_of_all _ fun y => ENNReal.ofReal_ne_top.lt_top)]
    apply hInt.congr
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [f, g, w]
    rw [ENNReal.toReal_ofReal (hw0 y hy), hclamp_eq y hy]
  have hcov := opposite_monotone_covariance_le hf hg hfi hgi hfgi
  rw [hint, hint, hint] at hcov
  have hrho_eq : (∫ y in I, f y * w y) = referenceExpectation rho := by
    apply integral_congr_ae
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [f, w]
    rw [hclamp_eq y hy]
  have hF_eq : (∫ y in I, g y * w y) =
      referenceExpectation (latalaF lam) := by
    apply integral_congr_ae
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [g, w]
    rw [hclamp_eq y hy]
  have hprod_eq : (∫ y in I, (f y * g y) * w y) =
      referenceExpectation (fun y => rho y * latalaF lam y) := by
    apply integral_congr_ae
    filter_upwards [ae_restrict_mem measurableSet_Icc] with y hy
    dsimp [f, g, w]
    rw [hclamp_eq y hy]
  rw [hrho_eq, hF_eq, hprod_eq, hprob,
    latalaF_reference_mean_one hlam] at hcov
  simpa using hcov

end SpinGlass.AT
