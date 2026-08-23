import Lemmas.ATDefs
import SpinGlass.Replicas
import Lemmas.smart_path.IndependentGaussianAffineIBP

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ProbabilityTheory NNReal

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-!
# Last-site cavity interpolation

The definitions in this file split an energy into its even and odd parts under
flipping one spin.  This avoids changing the configuration type from `Fin N`
to `Fin (N - 1)` and makes the overlap-with-one-site-removed identity exact.
-/

/-- Flip one site of a spin configuration. -/
def flipSite {N : ℕ} (i : Fin N) (σ : SpinGlass.Config N) : SpinGlass.Config N :=
  Function.update σ i (!σ i)

@[simp] lemma flipSite_apply_same {N : ℕ} (i : Fin N)
    (σ : SpinGlass.Config N) : flipSite i σ i = !σ i := by
  simp [flipSite]

@[simp] lemma flipSite_apply_of_ne {N : ℕ} {i j : Fin N} (hji : j ≠ i)
    (σ : SpinGlass.Config N) : flipSite i σ j = σ j := by
  simp [flipSite, hji]

@[simp] lemma flipSite_involutive {N : ℕ} (i : Fin N)
    (σ : SpinGlass.Config N) : flipSite i (flipSite i σ) = σ := by
  funext j
  by_cases hji : j = i
  · subst j
    simp [flipSite]
  · simp [flipSite, hji]

@[simp] lemma spin_flipSite_same {N : ℕ} (i : Fin N)
    (σ : SpinGlass.Config N) :
    SpinGlass.spin N (flipSite i σ) i = -SpinGlass.spin N σ i := by
  simp [SpinGlass.spin, flipSite]
  cases h : σ i <;> simp

@[simp] lemma spin_flipSite_of_ne {N : ℕ} {i j : Fin N} (hji : j ≠ i)
    (σ : SpinGlass.Config N) :
    SpinGlass.spin N (flipSite i σ) j = SpinGlass.spin N σ j := by
  simp [SpinGlass.spin, flipSite, hji]

/-- Even part of an energy under a one-site flip. -/
noncomputable def siteEvenCLM {N : ℕ} (i : Fin N) :
    SpinGlass.EnergySpace N →L[ℝ] SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap
    { toFun := fun H => WithLp.toLp 2 (fun σ => (H σ + H (flipSite i σ)) / 2)
      map_add' := by
        intro H K
        ext σ
        simp
        ring
      map_smul' := by
        intro c H
        ext σ
        simp
        ring }

/-- Odd part of an energy under a one-site flip. -/
noncomputable def siteOddCLM {N : ℕ} (i : Fin N) :
    SpinGlass.EnergySpace N →L[ℝ] SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap
    { toFun := fun H => WithLp.toLp 2 (fun σ => (H σ - H (flipSite i σ)) / 2)
      map_add' := by
        intro H K
        ext σ
        simp
        ring
      map_smul' := by
        intro c H
        ext σ
        simp
        ring }

@[simp] lemma siteEvenCLM_apply {N : ℕ} (i : Fin N)
    (H : SpinGlass.EnergySpace N) (σ : SpinGlass.Config N) :
    siteEvenCLM i H σ = (H σ + H (flipSite i σ)) / 2 := rfl

@[simp] lemma siteOddCLM_apply {N : ℕ} (i : Fin N)
    (H : SpinGlass.EnergySpace N) (σ : SpinGlass.Config N) :
    siteOddCLM i H σ = (H σ - H (flipSite i σ)) / 2 := rfl

lemma siteEven_add_siteOdd {N : ℕ} (i : Fin N)
    (H : SpinGlass.EnergySpace N) : siteEvenCLM i H + siteOddCLM i H = H := by
  ext σ
  simp
  ring

@[simp] lemma siteEven_flip {N : ℕ} (i : Fin N)
    (H : SpinGlass.EnergySpace N) (σ : SpinGlass.Config N) :
    siteEvenCLM i H (flipSite i σ) = siteEvenCLM i H σ := by
  simp [siteEvenCLM_apply, add_comm]

@[simp] lemma siteOdd_flip {N : ℕ} (i : Fin N)
    (H : SpinGlass.EnergySpace N) (σ : SpinGlass.Config N) :
    siteOddCLM i H (flipSite i σ) = -siteOddCLM i H σ := by
  simp [siteOddCLM_apply]
  ring

/-- The overlap with the contribution of site `i` removed, retaining the
normalization `1/N` used in the cavity calculation. -/
noncomputable def cavityOverlapAt {N n : ℕ} (q : ℝ) (i : Fin N)
    (σs : Replicas N n) (a b : Fin n) : ℝ :=
  centeredOverlap q σs a b -
    (1 / (N : ℝ)) * SpinGlass.spin N (σs a) i * SpinGlass.spin N (σs b) i

lemma centeredOverlap_eq_cavityOverlapAt_add {N n : ℕ} (q : ℝ) (i : Fin N)
    (σs : Replicas N n) (a b : Fin n) :
    centeredOverlap q σs a b = cavityOverlapAt q i σs a b +
      (1 / (N : ℝ)) * SpinGlass.spin N (σs a) i * SpinGlass.spin N (σs b) i := by
  simp [cavityOverlapAt]

lemma abs_cavityOverlapAt_le_two {N n : ℕ} (hN : 0 < N)
    {q : ℝ} (hq : q ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (σs : Replicas N n) (a b : Fin n) :
    |cavityOverlapAt q i σs a b| ≤ 2 := by
  let x : Fin N → ℝ := fun j =>
    if j = i then 0 else SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j
  have hxlower : ∀ j, (-1 : ℝ) ≤ x j := by
    intro j
    simp only [x]
    split
    · norm_num
    · cases ha : σs a j <;> cases hb : σs b j <;>
        simp [SpinGlass.spin, ha, hb]
  have hxupper : ∀ j, x j ≤ (1 : ℝ) := by
    intro j
    simp only [x]
    split
    · norm_num
    · cases ha : σs a j <;> cases hb : σs b j <;>
        simp [SpinGlass.spin, ha, hb]
  have hsumlower : -(N : ℝ) ≤ ∑ j, x j := by
    calc
      -(N : ℝ) = ∑ _j : Fin N, (-1 : ℝ) := by simp
      _ ≤ ∑ j, x j := Finset.sum_le_sum fun j _ => hxlower j
  have hsumupper : ∑ j, x j ≤ (N : ℝ) := by
    calc
      ∑ j, x j ≤ ∑ _j : Fin N, (1 : ℝ) :=
        Finset.sum_le_sum fun j _ => hxupper j
      _ = (N : ℝ) := by simp
  have hsum : ∑ j, x j =
      ∑ j, SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j -
        SpinGlass.spin N (σs a) i * SpinGlass.spin N (σs b) i := by
    classical
    have hxerase : ∑ j ∈ (Finset.univ.erase i), x j =
        ∑ j ∈ (Finset.univ.erase i),
          SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j := by
      apply Finset.sum_congr rfl
      intro j hj
      have hji : j ≠ i := Finset.ne_of_mem_erase hj
      simp [x, hji]
    have hfull := Finset.sum_erase_add (Finset.univ : Finset (Fin N))
      (fun j => SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j)
      (Finset.mem_univ i)
    calc
      ∑ j, x j = ∑ j ∈ (Finset.univ.erase i), x j := by
        rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i)]
        simp [x]
      _ = ∑ j ∈ (Finset.univ.erase i),
          SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j := hxerase
      _ = ∑ j, SpinGlass.spin N (σs a) j * SpinGlass.spin N (σs b) j -
          SpinGlass.spin N (σs a) i * SpinGlass.spin N (σs b) i := by
        linarith
  have hNreal : (0 : ℝ) < N := by exact_mod_cast hN
  have hcav : cavityOverlapAt q i σs a b =
      (1 / (N : ℝ)) * (∑ j, x j) - q := by
    rw [cavityOverlapAt, centeredOverlap, replicaOverlap, SpinGlass.overlap, hsum]
    ring
  rw [hcav, abs_le]
  constructor
  · have hmul := mul_le_mul_of_nonneg_left hsumlower (by positivity : 0 ≤ (1 / (N : ℝ)))
    have hnorm : (1 / (N : ℝ)) * (-(N : ℝ)) = -1 := by field_simp
    rw [hnorm] at hmul
    linarith [hq.2]
  · have hmul := mul_le_mul_of_nonneg_left hsumupper (by positivity : 0 ≤ (1 / (N : ℝ)))
    have hnorm : (1 / (N : ℝ)) * (N : ℝ) = 1 := by field_simp
    rw [hnorm] at hmul
    linarith [hq.1]

/-- Uncentered overlap with site `i` removed and normalization `1/N`. -/
noncomputable def configCavityOverlapAt {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) : ℝ :=
  SpinGlass.overlap N σ τ -
    (1 / (N : ℝ)) * SpinGlass.spin N σ i * SpinGlass.spin N τ i

lemma cavityOverlapAt_eq_configCavityOverlapAt_sub {N n : ℕ} (q : ℝ)
    (i : Fin N) (σs : Replicas N n) (a b : Fin n) :
    cavityOverlapAt q i σs a b =
      configCavityOverlapAt i (σs a) (σs b) - q := by
  simp [cavityOverlapAt, configCavityOverlapAt, centeredOverlap, replicaOverlap]
  ring

lemma overlap_eq_configCavityOverlapAt_add {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    SpinGlass.overlap N σ τ = configCavityOverlapAt i σ τ +
      (1 / (N : ℝ)) * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  simp [configCavityOverlapAt]

private lemma sum_spin_flipSite_left {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    (∑ j, SpinGlass.spin N (flipSite i σ) j * SpinGlass.spin N τ j) =
      (∑ j, SpinGlass.spin N σ j * SpinGlass.spin N τ j) -
        2 * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  classical
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i),
    ← Finset.sum_erase_add (Finset.univ : Finset (Fin N))
      (fun j => SpinGlass.spin N σ j * SpinGlass.spin N τ j) (Finset.mem_univ i)]
  have heq : ∑ j ∈ (Finset.univ.erase i),
      SpinGlass.spin N (flipSite i σ) j * SpinGlass.spin N τ j =
      ∑ j ∈ (Finset.univ.erase i),
        SpinGlass.spin N σ j * SpinGlass.spin N τ j := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [spin_flipSite_of_ne (Finset.ne_of_mem_erase hj)]
  rw [heq, spin_flipSite_same]
  ring

lemma overlap_flipSite_left {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    SpinGlass.overlap N (flipSite i σ) τ =
      SpinGlass.overlap N σ τ -
        (2 / (N : ℝ)) * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  rw [SpinGlass.overlap, SpinGlass.overlap, sum_spin_flipSite_left]
  ring

lemma overlap_flipSite_right {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    SpinGlass.overlap N σ (flipSite i τ) =
      SpinGlass.overlap N σ τ -
        (2 / (N : ℝ)) * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  simp only [SpinGlass.overlap]
  have hcomm : (∑ j, SpinGlass.spin N σ j * SpinGlass.spin N (flipSite i τ) j) =
      ∑ j, SpinGlass.spin N (flipSite i τ) j * SpinGlass.spin N σ j := by
    apply Finset.sum_congr rfl
    intro j _
    ring
  have hcomm0 : (∑ j, SpinGlass.spin N τ j * SpinGlass.spin N σ j) =
      ∑ j, SpinGlass.spin N σ j * SpinGlass.spin N τ j := by
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hcomm, sum_spin_flipSite_left]
  have hNnat : N ≠ 0 := by
    intro h
    subst N
    exact Fin.elim0 i
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast hNnat
  field_simp [hN]
  rw [hcomm0]

lemma overlap_flipSite_both {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    SpinGlass.overlap N (flipSite i σ) (flipSite i τ) =
      SpinGlass.overlap N σ τ := by
  rw [overlap_flipSite_left, overlap_flipSite_right, spin_flipSite_same]
  ring

@[simp] lemma configCavityOverlapAt_flip_left {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    configCavityOverlapAt i (flipSite i σ) τ =
      configCavityOverlapAt i σ τ := by
  simp only [configCavityOverlapAt]
  rw [overlap_flipSite_left, spin_flipSite_same]
  have hNnat : N ≠ 0 := by
    intro h
    subst N
    exact Fin.elim0 i
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast hNnat
  field_simp [hN]
  ring

@[simp] lemma configCavityOverlapAt_flip_right {N : ℕ} (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    configCavityOverlapAt i σ (flipSite i τ) =
      configCavityOverlapAt i σ τ := by
  simp only [configCavityOverlapAt]
  rw [overlap_flipSite_right, spin_flipSite_same]
  have hNnat : N ≠ 0 := by
    intro h
    subst N
    exact Fin.elim0 i
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast hNnat
  field_simp [hN]
  ring

lemma skKernel_odd_difference {N : ℕ} (β : ℝ) (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    (SpinGlass.sk_cov_kernel N β σ τ -
        SpinGlass.sk_cov_kernel N β (flipSite i σ) τ -
        SpinGlass.sk_cov_kernel N β σ (flipSite i τ) +
        SpinGlass.sk_cov_kernel N β (flipSite i σ) (flipSite i τ)) / 4 =
      β ^ 2 * SpinGlass.spin N σ i * SpinGlass.spin N τ i *
        configCavityOverlapAt i σ τ := by
  have hsσ : SpinGlass.spin N σ i ^ 2 = 1 := by
    cases h : σ i <;> simp [SpinGlass.spin, h]
  have hsτ : SpinGlass.spin N τ i ^ 2 = 1 := by
    cases h : τ i <;> simp [SpinGlass.spin, h]
  have hNnat : N ≠ 0 := by
    intro h
    subst N
    exact Fin.elim0 i
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast hNnat
  rw [SpinGlass.sk_cov_kernel, SpinGlass.sk_cov_kernel,
    SpinGlass.sk_cov_kernel, SpinGlass.sk_cov_kernel,
    overlap_flipSite_left, overlap_flipSite_right,
    overlap_flipSite_both, configCavityOverlapAt]
  field_simp [hN]
  nlinarith

lemma simpleKernel_odd_difference {N : ℕ} (β q : ℝ) (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    (SpinGlass.simple_cov_kernel N β (fun x => q * x) σ τ -
        SpinGlass.simple_cov_kernel N β (fun x => q * x) (flipSite i σ) τ -
        SpinGlass.simple_cov_kernel N β (fun x => q * x) σ (flipSite i τ) +
        SpinGlass.simple_cov_kernel N β (fun x => q * x)
          (flipSite i σ) (flipSite i τ)) / 4 =
      β ^ 2 * q * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  rw [SpinGlass.simple_cov_kernel, SpinGlass.simple_cov_kernel,
    SpinGlass.simple_cov_kernel, SpinGlass.simple_cov_kernel,
    overlap_flipSite_left, overlap_flipSite_right, overlap_flipSite_both]
  have hNnat : N ≠ 0 := by
    intro h
    subst N
    exact Fin.elim0 i
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast hNnat
  field_simp [hN]
  ring

lemma skKernel_even_odd_difference {N : ℕ} (β : ℝ) (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    (SpinGlass.sk_cov_kernel N β σ τ -
        SpinGlass.sk_cov_kernel N β σ (flipSite i τ) +
        SpinGlass.sk_cov_kernel N β (flipSite i σ) τ -
        SpinGlass.sk_cov_kernel N β (flipSite i σ) (flipSite i τ)) / 4 = 0 := by
  rw [SpinGlass.sk_cov_kernel, SpinGlass.sk_cov_kernel,
    SpinGlass.sk_cov_kernel, SpinGlass.sk_cov_kernel,
    overlap_flipSite_left, overlap_flipSite_right, overlap_flipSite_both]
  ring

lemma simpleKernel_even_odd_difference {N : ℕ} (β q : ℝ) (i : Fin N)
    (σ τ : SpinGlass.Config N) :
    (SpinGlass.simple_cov_kernel N β (fun x => q * x) σ τ -
        SpinGlass.simple_cov_kernel N β (fun x => q * x) σ (flipSite i τ) +
        SpinGlass.simple_cov_kernel N β (fun x => q * x) (flipSite i σ) τ -
        SpinGlass.simple_cov_kernel N β (fun x => q * x)
          (flipSite i σ) (flipSite i τ)) / 4 = 0 := by
  rw [SpinGlass.simple_cov_kernel, SpinGlass.simple_cov_kernel,
    SpinGlass.simple_cov_kernel, SpinGlass.simple_cov_kernel,
    overlap_flipSite_left, overlap_flipSite_right, overlap_flipSite_both]
  ring

lemma sk_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h : ℝ} (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
    (σ τ : SpinGlass.Config N) :
    (∑ k : sk.hU.ι, (sk.hU.τ k : ℝ) * sk.hU.w k σ * sk.hU.w k τ) =
      SpinGlass.sk_cov_kernel N β σ τ := by
  have hcov := sk.cov_eq σ τ
  simp only [PhysLean.Probability.GaussianIBP.covOp_apply,
    sum_inner, inner_smul_left, SpinGlass.inner_std_basis_apply,
    real_inner_comm] at hcov
  simpa [mul_assoc, mul_left_comm, mul_comm] using hcov

lemma simple_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β q : ℝ} (sim : SpinGlass.SimpleDisorder (Ω := Ω) N β q)
    (σ τ : SpinGlass.Config N) :
    (∑ k : sim.hV.ι, (sim.hV.τ k : ℝ) * sim.hV.w k σ * sim.hV.w k τ) =
      SpinGlass.simple_cov_kernel N β (fun x => q * x) σ τ := by
  have hcov := sim.cov_eq σ τ
  simp only [PhysLean.Probability.GaussianIBP.covOp_apply,
    sum_inner, inner_smul_left, SpinGlass.inner_std_basis_apply,
    real_inner_comm] at hcov
  simpa [mul_assoc, mul_left_comm, mul_comm] using hcov

lemma sk_siteOdd_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h : ℝ} (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
    (i : Fin N) (σ τ : SpinGlass.Config N) :
    (∑ k : sk.hU.ι, (sk.hU.τ k : ℝ) *
        siteOddCLM i (sk.hU.w k) σ * siteOddCLM i (sk.hU.w k) τ) =
      β ^ 2 * SpinGlass.spin N σ i * SpinGlass.spin N τ i *
        configCavityOverlapAt i σ τ := by
  rw [← skKernel_odd_difference β i σ τ]
  simp_rw [siteOddCLM_apply]
  rw [← sk_basis_covariance_sum sk σ τ,
    ← sk_basis_covariance_sum sk (flipSite i σ) τ,
    ← sk_basis_covariance_sum sk σ (flipSite i τ),
    ← sk_basis_covariance_sum sk (flipSite i σ) (flipSite i τ)]
  simp only [div_eq_mul_inv]
  ring_nf
  simp only [Finset.sum_add_distrib, Finset.sum_mul]
  ring

lemma simple_siteOdd_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β q : ℝ} (sim : SpinGlass.SimpleDisorder (Ω := Ω) N β q)
    (i : Fin N) (σ τ : SpinGlass.Config N) :
    (∑ k : sim.hV.ι, (sim.hV.τ k : ℝ) *
        siteOddCLM i (sim.hV.w k) σ * siteOddCLM i (sim.hV.w k) τ) =
      β ^ 2 * q * SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
  rw [← simpleKernel_odd_difference β q i σ τ]
  simp_rw [siteOddCLM_apply]
  rw [← simple_basis_covariance_sum sim σ τ,
    ← simple_basis_covariance_sum sim (flipSite i σ) τ,
    ← simple_basis_covariance_sum sim σ (flipSite i τ),
    ← simple_basis_covariance_sum sim (flipSite i σ) (flipSite i τ)]
  simp only [div_eq_mul_inv]
  ring_nf
  simp only [Finset.sum_add_distrib, Finset.sum_mul]
  ring

lemma sk_siteEven_siteOdd_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h : ℝ} (sk : SpinGlass.SKDisorder (Ω := Ω) N β h)
    (i : Fin N) (σ τ : SpinGlass.Config N) :
    (∑ k : sk.hU.ι, (sk.hU.τ k : ℝ) *
        siteEvenCLM i (sk.hU.w k) σ * siteOddCLM i (sk.hU.w k) τ) = 0 := by
  rw [← skKernel_even_odd_difference β i σ τ]
  simp_rw [siteEvenCLM_apply, siteOddCLM_apply]
  rw [← sk_basis_covariance_sum sk σ τ,
    ← sk_basis_covariance_sum sk σ (flipSite i τ),
    ← sk_basis_covariance_sum sk (flipSite i σ) τ,
    ← sk_basis_covariance_sum sk (flipSite i σ) (flipSite i τ)]
  simp only [div_eq_mul_inv]
  ring_nf
  simp only [Finset.sum_add_distrib, Finset.sum_mul]

lemma simple_siteEven_siteOdd_basis_covariance_sum
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β q : ℝ} (sim : SpinGlass.SimpleDisorder (Ω := Ω) N β q)
    (i : Fin N) (σ τ : SpinGlass.Config N) :
    (∑ k : sim.hV.ι, (sim.hV.τ k : ℝ) *
        siteEvenCLM i (sim.hV.w k) σ * siteOddCLM i (sim.hV.w k) τ) = 0 := by
  rw [← simpleKernel_even_odd_difference β q i σ τ]
  simp_rw [siteEvenCLM_apply, siteOddCLM_apply]
  rw [← simple_basis_covariance_sum sim σ τ,
    ← simple_basis_covariance_sum sim σ (flipSite i τ),
    ← simple_basis_covariance_sum sim (flipSite i σ) τ,
    ← simple_basis_covariance_sum sim (flipSite i σ) (flipSite i τ)]
  simp only [div_eq_mul_inv]
  ring_nf
  simp only [Finset.sum_add_distrib, Finset.sum_mul]

/-- The last-site interpolation.  At `u=1` it is the original smart path.
At `u=0` the odd SK part is replaced by the odd part of the independent
simple field, whose covariance is exactly the RS one-site covariance. -/
noncomputable def lastSiteHamiltonian
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (u : ℝ) (ω : Ω) : SpinGlass.EnergySpace N :=
  Real.sqrt s • siteEvenCLM i (path.sk.U ω) +
    Real.sqrt (1 - s) • siteEvenCLM i (path.simple.V ω) +
    Real.sqrt u •
      (Real.sqrt s • siteOddCLM i (path.sk.U ω) +
        Real.sqrt (1 - s) • siteOddCLM i (path.simple.V ω)) +
    Real.sqrt (1 - u) • siteOddCLM i (path.simple.V ω) +
    SpinGlass.magnetic_field_vector N h

lemma lastSiteHamiltonian_one
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (ω : Ω) :
    lastSiteHamiltonian (s := s) path i 1 ω = fullPathHamiltonian path s ω := by
  rw [lastSiteHamiltonian, fullPathHamiltonian]
  simp only [Real.sqrt_one, one_smul, sub_self, Real.sqrt_zero, zero_smul,
    add_zero]
  calc
    Real.sqrt s • siteEvenCLM i (path.sk.U ω) +
          Real.sqrt (1 - s) • siteEvenCLM i (path.simple.V ω) +
          (Real.sqrt s • siteOddCLM i (path.sk.U ω) +
            Real.sqrt (1 - s) • siteOddCLM i (path.simple.V ω)) +
          SpinGlass.magnetic_field_vector N h =
        Real.sqrt s • (siteEvenCLM i (path.sk.U ω) + siteOddCLM i (path.sk.U ω)) +
          Real.sqrt (1 - s) •
            (siteEvenCLM i (path.simple.V ω) + siteOddCLM i (path.simple.V ω)) +
          SpinGlass.magnetic_field_vector N h := by
            simp only [smul_add]
            abel
    _ = Real.sqrt s • path.sk.U ω + Real.sqrt (1 - s) • path.simple.V ω +
          SpinGlass.magnetic_field_vector N h := by
      rw [siteEven_add_siteOdd, siteEven_add_siteOdd]

end SpinGlass.AT
