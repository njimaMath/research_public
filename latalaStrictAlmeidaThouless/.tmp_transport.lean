import Lemmas.Concentration_Coupled
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence

open MeasureTheory ProbabilityTheory Real BigOperators
open PhysLean.Probability.GaussianIBP

set_option autoImplicit false

namespace SpinGlass.AT

private lemma coefficient_inner
    (N : ℕ) (beta q s : ℝ) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hq : 0 ≤ q) (sigma tau : SpinGlass.Config N) :
    inner ℝ (coupledDisorderCoefficient N beta q s sigma)
      (coupledDisorderCoefficient N beta q s tau) =
      smartPathCovKernel N beta q s sigma tau := by
  classical
  by_cases hN : N = 0
  · subst N
    simp [coupledDisorderCoefficient, smartPathCovKernel,
      SpinGlass.sk_cov_kernel, SpinGlass.simple_cov_kernel,
      PiLp.inner_apply]
  · have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast hN
    have hs0 : 0 ≤ s := hs.1
    have h1s0 : 0 ≤ 1 - s := sub_nonneg.mpr hs.2
    have hsqrt2N : Real.sqrt (2 * (N : ℝ)) ^ 2 = 2 * N :=
      Real.sq_sqrt (by positivity)
    have hsqrts : Real.sqrt s ^ 2 = s := Real.sq_sqrt hs0
    have hsqrtq : Real.sqrt ((1 - s) * q) ^ 2 = (1 - s) * q :=
      Real.sq_sqrt (mul_nonneg h1s0 hq)
    rw [PiLp.inner_apply, Fintype.sum_sum_type]
    simp only [coupledDisorderCoefficient, RCLike.inner_apply, conj_trivial]
    change
      (∑ ij : Fin N × Fin N,
        (beta * Real.sqrt s / Real.sqrt (2 * N) *
            (SpinGlass.spin N tau ij.1 * SpinGlass.spin N tau ij.2)) *
          (beta * Real.sqrt s / Real.sqrt (2 * N) *
            (SpinGlass.spin N sigma ij.1 * SpinGlass.spin N sigma ij.2))) +
      ∑ i : Fin N,
        (beta * Real.sqrt ((1 - s) * q) * SpinGlass.spin N tau i) *
          (beta * Real.sqrt ((1 - s) * q) * SpinGlass.spin N sigma i) = _
    simp_rw [show ∀ ij : Fin N × Fin N,
        (beta * Real.sqrt s / Real.sqrt (2 * N) *
            (SpinGlass.spin N tau ij.1 * SpinGlass.spin N tau ij.2)) *
          (beta * Real.sqrt s / Real.sqrt (2 * N) *
            (SpinGlass.spin N sigma ij.1 * SpinGlass.spin N sigma ij.2)) =
          (beta ^ 2 * s / (2 * N)) *
            ((SpinGlass.spin N sigma ij.1 * SpinGlass.spin N tau ij.1) *
             (SpinGlass.spin N sigma ij.2 * SpinGlass.spin N tau ij.2)) by
      intro ij
      field_simp [Real.sqrt_ne_zero'.mpr (by positivity : (0 : ℝ) < 2 * N)]
      rw [hsqrt2N, hsqrts]
      ring]
    simp_rw [show ∀ i : Fin N,
        (beta * Real.sqrt ((1 - s) * q) * SpinGlass.spin N tau i) *
          (beta * Real.sqrt ((1 - s) * q) * SpinGlass.spin N sigma i) =
          beta ^ 2 * (1 - s) * q *
            (SpinGlass.spin N sigma i * SpinGlass.spin N tau i) by
      intro i
      calc
        _ = beta ^ 2 * Real.sqrt ((1 - s) * q) ^ 2 *
            (SpinGlass.spin N sigma i * SpinGlass.spin N tau i) := by ring
        _ = _ := by rw [hsqrtq]; ring]
    have hedgeSum :
        (∑ ij : Fin N × Fin N,
          (beta ^ 2 * s / (2 * N)) *
            ((SpinGlass.spin N sigma ij.1 * SpinGlass.spin N tau ij.1) *
             (SpinGlass.spin N sigma ij.2 * SpinGlass.spin N tau ij.2))) =
          (beta ^ 2 * s / (2 * N)) *
            (∑ i : Fin N, SpinGlass.spin N sigma i * SpinGlass.spin N tau i) ^ 2 := by
      rw [Fintype.sum_prod_type]
      calc
        (∑ i : Fin N, ∑ j : Fin N,
            (beta ^ 2 * s / (2 * N)) *
              ((SpinGlass.spin N sigma i * SpinGlass.spin N tau i) *
               (SpinGlass.spin N sigma j * SpinGlass.spin N tau j))) =
            ∑ i : Fin N,
              ((beta ^ 2 * s / (2 * N)) *
                (SpinGlass.spin N sigma i * SpinGlass.spin N tau i)) *
              (∑ j : Fin N,
                SpinGlass.spin N sigma j * SpinGlass.spin N tau j) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro j _
          ring
        _ = (beta ^ 2 * s / (2 * N)) *
              (∑ i : Fin N,
                SpinGlass.spin N sigma i * SpinGlass.spin N tau i) *
              (∑ j : Fin N,
                SpinGlass.spin N sigma j * SpinGlass.spin N tau j) := by
          rw [← Finset.sum_mul, ← Finset.mul_sum]
        _ = _ := by
          rw [show (∑ i : Fin N,
              SpinGlass.spin N sigma i * SpinGlass.spin N tau i) ^ 2 =
              (∑ i : Fin N,
                SpinGlass.spin N sigma i * SpinGlass.spin N tau i) *
              (∑ i : Fin N,
                SpinGlass.spin N sigma i * SpinGlass.spin N tau i) by
            exact pow_two _]
          rw [mul_assoc]
    rw [hedgeSum, ← Finset.mul_sum]
    unfold smartPathCovKernel SpinGlass.sk_cov_kernel
      SpinGlass.simple_cov_kernel SpinGlass.overlap
    simp_rw [pow_two]
    field_simp [hNr]

private noncomputable def flipConfig (N : ℕ) :
    SpinGlass.Config N ≃ SpinGlass.Config N where
  toFun sigma i := !sigma i
  invFun sigma i := !sigma i
  left_inv sigma := by ext i; simp
  right_inv sigma := by ext i; simp

private lemma spin_flip (N : ℕ) (sigma : SpinGlass.Config N) (i : Fin N) :
    SpinGlass.spin N (flipConfig N sigma) i = -SpinGlass.spin N sigma i := by
  cases h : sigma i <;> simp [flipConfig, SpinGlass.spin, h]

private lemma overlap_flip (N : ℕ) (sigma tau : SpinGlass.Config N) :
    SpinGlass.overlap N (flipConfig N sigma) (flipConfig N tau) =
      SpinGlass.overlap N sigma tau := by
  unfold SpinGlass.overlap
  apply congrArg ((1 / (N : ℝ)) * ·)
  apply Finset.sum_congr rfl
  intro i _
  rw [spin_flip, spin_flip]
  ring

private noncomputable def flipEnergyCLM (N : ℕ) :
    SpinGlass.EnergySpace N →L[ℝ] SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap {
    toFun := fun H => WithLp.toLp 2 (fun sigma => H (flipConfig N sigma))
    map_add' := by intro H G; ext sigma; rfl
    map_smul' := by intro c H; ext sigma; rfl }

private noncomputable def coordinateRandomCLM
    (N : ℕ) (beta q s : ℝ) :
    EuclideanSpace ℝ (CoupledGaussianIndex N) →L[ℝ]
      SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 (fun sigma =>
      -inner ℝ (coupledDisorderCoefficient N beta q s sigma) x)
    map_add' := by intro x y; ext sigma; simp [inner_add_right, add_comm]
    map_smul' := by intro c x; ext sigma; simp [inner_smul_right] }

private noncomputable def smartRandomPairCLM
    (N : ℕ) (s : ℝ) :
    (SpinGlass.EnergySpace N × SpinGlass.EnergySpace N) →L[ℝ]
      SpinGlass.EnergySpace N :=
  Real.sqrt s • ContinuousLinearMap.fst ℝ _ _ +
    Real.sqrt (1 - s) • ContinuousLinearMap.snd ℝ _ _

private noncomputable def flippedSmartRandomPairCLM
    (N : ℕ) (s : ℝ) :
    (SpinGlass.EnergySpace N × SpinGlass.EnergySpace N) →L[ℝ]
      SpinGlass.EnergySpace N :=
  (flipEnergyCLM N).comp (smartRandomPairCLM N s)

private lemma gaussianHilbert_hasGaussianLaw
    {Omega H : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]
    [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H]
    {g : Omega → H} (hg : IsGaussianHilbert g) :
    HasGaussianLaw g volume := by
  have hc : ∀ i, HasGaussianLaw (hg.c i) volume := by
    intro i
    exact HasLaw.hasGaussianLaw
      (HasLaw.mk (P := volume) (hg.c_meas i).aemeasurable (hg.c_gauss i))
  have hterm : ∀ i, HasGaussianLaw (fun omega => (hg.c i omega) • hg.w i) volume := by
    intro i
    exact (hc i).map_fun ((ContinuousLinearMap.id ℝ ℝ).smulRight (hg.w i))
  have hind : iIndepFun (fun i omega => (hg.c i omega) • hg.w i) volume :=
    hg.c_indep.comp (fun i x => x • hg.w i) (fun i => by fun_prop)
  have hsum : HasGaussianLaw (fun omega => ∑ i, (hg.c i omega) • hg.w i) volume :=
    hind.hasGaussianLaw_fun_sum hterm
  exact hsum.congr (Filter.Eventually.of_forall fun omega => by
    simpa using congrFun hg.repr omega |>.symm)

private lemma gaussianHilbert_map_mean_zero
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {g : Omega → SpinGlass.EnergySpace N}
    (hg : IsGaussianHilbert g) :
    ∫ x, id x ∂Measure.map g volume = 0 := by
  have hgLaw := gaussianHilbert_hasGaussianLaw hg
  have hmap : ∫ x, id x ∂Measure.map g volume = ∫ omega, g omega ∂volume := by
    exact integral_map hgLaw.aemeasurable aestronglyMeasurable_id
  rw [hmap]
  have hcMean : ∀ i, ∫ omega, hg.c i omega ∂volume = 0 := by
    intro i
    have hmapc := integral_map (μ := volume) (hg.c_meas i).aemeasurable
      aestronglyMeasurable_id
    rw [hg.c_gauss i] at hmapc
    simpa using hmapc.symm.trans integral_id_gaussianReal
  rw [hg.repr, integral_finset_sum]
  · simp [integral_smul_const, hcMean]
  · intro i _
    have hci : HasGaussianLaw (hg.c i) volume := HasLaw.hasGaussianLaw
      (HasLaw.mk (P := volume) (hg.c_meas i).aemeasurable (hg.c_gauss i))
    exact hci.integrable.smul_const _

private lemma gaussianHilbert_coord_secondMoment
    {Omega H : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]
    [MeasurableSpace H] [BorelSpace H]
    {g : Omega → H} (hg : IsGaussianHilbert g) [DecidableEq hg.ι]
    (i j : hg.ι) :
    ∫ omega, hg.c i omega * hg.c j omega ∂volume =
      if i = j then (hg.τ i : ℝ) else 0 := by
  classical
  have hc (k : hg.ι) : HasGaussianLaw (hg.c k) volume := HasLaw.hasGaussianLaw
    (HasLaw.mk (P := volume) (hg.c_meas k).aemeasurable (hg.c_gauss k))
  by_cases hij : i = j
  · subst j
    rw [if_pos rfl]
    simp only [← pow_two]
    have hmap := integral_map (μ := volume) (hg.c_meas i).aemeasurable
      (show AEStronglyMeasurable (fun x : ℝ => x ^ 2)
          (Measure.map (hg.c i) volume) by fun_prop)
    rw [hg.c_gauss i] at hmap
    simpa using hmap.symm.trans (integral_sq_gaussianReal_centered (hg.τ i))
  · rw [if_neg hij]
    have hmul := hg.c_indep.indepFun hij |>.integral_mul_eq_mul_integral
      (hg.c_meas i).aestronglyMeasurable (hg.c_meas j).aestronglyMeasurable
    have hmean (k : hg.ι) : ∫ omega, hg.c k omega ∂volume = 0 := by
      have hmap := integral_map (μ := volume) (hg.c_meas k).aemeasurable
        aestronglyMeasurable_id
      rw [hg.c_gauss k] at hmap
      simpa using hmap.symm.trans integral_id_gaussianReal
    simpa [hmean] using hmul

private lemma gaussianHilbert_energy_pairing
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {g : Omega → SpinGlass.EnergySpace N}
    (hg : IsGaussianHilbert g) (sigma tau : SpinGlass.Config N) :
    ∫ omega, g omega sigma * g omega tau ∂volume =
      inner ℝ ((covOp (g := g) hg) (SpinGlass.std_basis N sigma))
        (SpinGlass.std_basis N tau) := by
  classical
  have hpoint (omega : Omega) (rho : SpinGlass.Config N) :
      g omega rho = ∑ i, hg.c i omega * hg.w i rho := by
    have h := congrFun hg.repr omega
    rw [h]
    simp
  have hterm (i j : hg.ι) :
      Integrable (fun omega =>
        hg.c i omega * hg.w i sigma * (hg.c j omega * hg.w j tau)) volume := by
    have hc (k : hg.ι) : HasGaussianLaw (hg.c k) volume := HasLaw.hasGaussianLaw
      (HasLaw.mk (P := volume) (hg.c_meas k).aemeasurable (hg.c_gauss k))
    have hij : Integrable (fun omega => hg.c i omega * hg.c j omega) volume := by
      exact ((hc i).memLp_two.integrable_mul (hc j).memLp_two).congr
        (Filter.Eventually.of_forall fun omega => by rfl)
    convert hij.const_mul (hg.w i sigma * hg.w j tau) using 1 <;> ext omega <;> ring
  simp_rw [hpoint, Finset.sum_mul, Finset.mul_sum]
  rw [integral_finset_sum _ (fun i _ =>
    integrable_finset_sum _ (fun j _ => hterm i j))]
  simp_rw [integral_finset_sum _ (fun j _ => hterm _ j)]
  simp_rw [show ∀ i j, (fun omega =>
      hg.c i omega * hg.w i sigma * (hg.c j omega * hg.w j tau)) =
      fun omega => (hg.w i sigma * hg.w j tau) *
        (hg.c i omega * hg.c j omega) by
    intro i j; funext omega; ring,
    integral_const_mul, gaussianHilbert_coord_secondMoment hg]
  simp [covOp_apply, sum_inner, inner_smul_left, real_inner_comm,
    SpinGlass.inner_std_basis_apply]
  apply Finset.sum_congr rfl
  intro i _
  ring

private lemma gaussianHilbert_energy_cov
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {g : Omega → SpinGlass.EnergySpace N}
    (hg : IsGaussianHilbert g) (sigma tau : SpinGlass.Config N) :
    cov[fun omega => g omega sigma, fun omega => g omega tau; volume] =
      inner ℝ ((covOp (g := g) hg) (SpinGlass.std_basis N sigma))
        (SpinGlass.std_basis N tau) := by
  unfold covariance
  rw [show ∫ omega, g omega sigma ∂volume = 0 by
      have hmap := gaussianHilbert_map_mean_zero hg
      have hglaw := gaussianHilbert_hasGaussianLaw hg
      rw [integral_map hglaw.aemeasurable aestronglyMeasurable_id] at hmap
      have hvec : ∫ x, g x ∂volume = 0 := by simpa only [id_eq] using hmap
      rw [← eval_integral_piLp (fun rho => hglaw.integrable.eval_piLp rho) sigma,
        hvec]
      rfl,
    show ∫ omega, g omega tau ∂volume = 0 by
      have hmap := gaussianHilbert_map_mean_zero hg
      have hglaw := gaussianHilbert_hasGaussianLaw hg
      rw [integral_map hglaw.aemeasurable aestronglyMeasurable_id] at hmap
      have hvec : ∫ x, g x ∂volume = 0 := by simpa only [id_eq] using hmap
      rw [← eval_integral_piLp (fun rho => hglaw.integrable.eval_piLp rho) tau,
        hvec]
      rfl]
  simp only [mul_zero, sub_zero]
  exact gaussianHilbert_energy_pairing hg sigma tau

private lemma gaussianHilbert_energy_integral_eq_zero
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {g : Omega → SpinGlass.EnergySpace N}
    (hg : IsGaussianHilbert g) :
    ∫ omega, g omega ∂(volume : Measure Omega) = 0 := by
  have h := gaussianHilbert_map_mean_zero hg
  rw [integral_map (gaussianHilbert_hasGaussianLaw hg).aemeasurable
    aestronglyMeasurable_id] at h
  simpa only [id_eq] using h

private lemma flippedSmart_cov
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {beta h q : ℝ}
    (path : RSSmartPathDisorder Omega N beta h q)
    (s : Set.Icc (0 : ℝ) 1) (sigma tau : SpinGlass.Config N) :
    cov[
      fun omega => flippedSmartRandomPairCLM N s.1
        (path.sk.U omega, path.simple.V omega) sigma,
      fun omega => flippedSmartRandomPairCLM N s.1
        (path.sk.U omega, path.simple.V omega) tau;
      volume] = smartPathCovKernel N beta q s.1 sigma tau := by
  let sigma' := flipConfig N sigma
  let tau' := flipConfig N tau
  have hUlaw := gaussianHilbert_hasGaussianLaw path.sk.hU
  have hVlaw := gaussianHilbert_hasGaussianLaw path.simple.hV
  have hUs : MemLp (fun omega => path.sk.U omega sigma') 2 volume := by
    simpa using (hUlaw.map_fun (SpinGlass.evalCLM (N := N) sigma')).memLp_two
  have hUt : MemLp (fun omega => path.sk.U omega tau') 2 volume := by
    simpa using (hUlaw.map_fun (SpinGlass.evalCLM (N := N) tau')).memLp_two
  have hVs : MemLp (fun omega => path.simple.V omega sigma') 2 volume := by
    simpa using (hVlaw.map_fun (SpinGlass.evalCLM (N := N) sigma')).memLp_two
  have hVt : MemLp (fun omega => path.simple.V omega tau') 2 volume := by
    simpa using (hVlaw.map_fun (SpinGlass.evalCLM (N := N) tau')).memLp_two
  have hind (rho eta : SpinGlass.Config N) :
      (fun omega => path.sk.U omega rho) ⟂ᵢ[volume]
        (fun omega => path.simple.V omega eta) := by
    simpa [Function.comp_def] using path.independent.comp
      (SpinGlass.evalCLM (N := N) rho).measurable
      (SpinGlass.evalCLM (N := N) eta).measurable
  have hUV : cov[fun omega => path.sk.U omega sigma',
      fun omega => path.simple.V omega tau'; volume] = 0 :=
    (hind sigma' tau').covariance_eq_zero hUs hVt
  have hVU : cov[fun omega => path.simple.V omega sigma',
      fun omega => path.sk.U omega tau'; volume] = 0 := by
    rw [covariance_comm]
    exact (hind tau' sigma').covariance_eq_zero hUt hVs
  change cov[
    (fun omega => Real.sqrt s.1 * path.sk.U omega sigma') +
      (fun omega => Real.sqrt (1 - s.1) * path.simple.V omega sigma'),
    (fun omega => Real.sqrt s.1 * path.sk.U omega tau') +
      (fun omega => Real.sqrt (1 - s.1) * path.simple.V omega tau');
    volume] = _
  rw [covariance_add_left (hUs.const_mul _) (hVs.const_mul _)
      ((hUt.const_mul _).add (hVt.const_mul _)),
    covariance_add_right (hUs.const_mul _) (hUt.const_mul _) (hVt.const_mul _),
    covariance_add_right (hVs.const_mul _) (hUt.const_mul _) (hVt.const_mul _)]
  simp_rw [covariance_const_mul_left, covariance_const_mul_right]
  rw [gaussianHilbert_energy_cov path.sk.hU,
    gaussianHilbert_energy_cov path.simple.hV, path.sk.cov_eq, path.simple.cov_eq,
    hUV, hVU]
  have hs_sq : Real.sqrt s.1 * Real.sqrt s.1 = s.1 := by
    simpa [pow_two] using Real.sq_sqrt s.2.1
  have h1s_sq : Real.sqrt (1 - s.1) * Real.sqrt (1 - s.1) = 1 - s.1 := by
    simpa [pow_two] using Real.sq_sqrt (sub_nonneg.mpr s.2.2)
  simp only [mul_zero, zero_mul, add_zero, zero_add]
  rw [← mul_assoc, hs_sq, ← mul_assoc, h1s_sq]
  unfold smartPathCovKernel
  rw [overlap_flip, overlap_flip]

private lemma bilin_eq_sum_std
    (N : ℕ)
    (C : SpinGlass.EnergySpace N →L[ℝ]
      SpinGlass.EnergySpace N →L[ℝ] ℝ)
    (x y : SpinGlass.EnergySpace N) :
    C x y = ∑ sigma, ∑ tau, x sigma * y tau *
      C (SpinGlass.std_basis N sigma) (SpinGlass.std_basis N tau) := by
  classical
  have hx : x = ∑ sigma : SpinGlass.Config N,
      (x sigma) • SpinGlass.std_basis N sigma := by
    ext rho
    simp [SpinGlass.std_basis]
  have hy : y = ∑ tau : SpinGlass.Config N,
      (y tau) • SpinGlass.std_basis N tau := by
    ext rho
    simp [SpinGlass.std_basis]
  calc
    C x y = C (∑ sigma, (x sigma) • SpinGlass.std_basis N sigma)
        (∑ tau, (y tau) • SpinGlass.std_basis N tau) :=
      congrArg₂ (fun a b => C a b) hx hy
    _ = _ := by
      simp [map_sum, map_smul, Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro sigma _
      apply Finset.sum_congr rfl
      intro tau _
      ring

private theorem smartPath_coordinate_random_law
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {beta h q : ℝ}
    (path : RSSmartPathDisorder Omega N beta h q)
    (s : Set.Icc (0 : ℝ) 1) (hq : 0 ≤ q) :
    Measure.map (fun omega => flippedSmartRandomPairCLM N s.1
        (path.sk.U omega, path.simple.V omega)) volume =
      Measure.map (coordinateRandomCLM N beta q s.1)
        (SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N)) := by
  let pair : Omega → SpinGlass.EnergySpace N × SpinGlass.EnergySpace N :=
    fun omega => (path.sk.U omega, path.simple.V omega)
  have hUlaw := gaussianHilbert_hasGaussianLaw path.sk.hU
  have hVlaw := gaussianHilbert_hasGaussianLaw path.simple.hV
  have hpair : HasGaussianLaw pair volume :=
    path.independent.hasGaussianLaw hUlaw hVlaw
  have hleft : HasGaussianLaw (fun omega => flippedSmartRandomPairCLM N s.1
      (pair omega)) volume := hpair.map_fun (flippedSmartRandomPairCLM N s.1)
  have hright : HasGaussianLaw (coordinateRandomCLM N beta q s.1)
      (SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N)) :=
    IsGaussian.hasGaussianLaw_id.map_fun (coordinateRandomCLM N beta q s.1)
  let mu := Measure.map (fun omega => flippedSmartRandomPairCLM N s.1
      (pair omega)) volume
  let nu := Measure.map (coordinateRandomCLM N beta q s.1)
      (SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N))
  letI : IsGaussian mu := hleft.isGaussian_map
  letI : IsGaussian nu := hright.isGaussian_map
  apply ProbabilityTheory.IsGaussian.ext
  · have hUint := gaussianHilbert_energy_integral_eq_zero path.sk.hU
    have hVint := gaussianHilbert_energy_integral_eq_zero path.simple.hV
    have hpint : ∫ omega, pair omega ∂volume = (0, 0) := by
      apply Prod.ext
      · rw [(ContinuousLinearMap.fst ℝ _ _).integral_comp_comm hpair.integrable]
        simpa [pair] using hUint
      · rw [(ContinuousLinearMap.snd ℝ _ _).integral_comp_comm hpair.integrable]
        simpa [pair] using hVint
    rw [integral_map hleft.aemeasurable aestronglyMeasurable_id,
      integral_map hright.aemeasurable aestronglyMeasurable_id]
    simp only [id_eq]
    rw [← (flippedSmartRandomPairCLM N s.1).integral_comp_comm hpair.integrable,
      hpint]
    rw [← (coordinateRandomCLM N beta q s.1).integral_comp_comm
      IsGaussian.integrable_id,
      SYK.standardGaussianMeasureOnEuclidean_integral_id]
    simp
  · apply ContinuousLinearMap.ext
    intro x
    apply ContinuousLinearMap.ext
    intro y
    calc
      covarianceBilin mu x y =
          ∑ sigma, ∑ tau, x sigma * y tau *
            covarianceBilin mu (SpinGlass.std_basis N sigma)
              (SpinGlass.std_basis N tau) := bilin_eq_sum_std N _ x y
      _ = ∑ sigma, ∑ tau, x sigma * y tau *
            covarianceBilin nu (SpinGlass.std_basis N sigma)
              (SpinGlass.std_basis N tau) := by
        apply Finset.sum_congr rfl
        intro sigma _
        apply Finset.sum_congr rfl
        intro tau _
        congr 1
        rw [covarianceBilin_apply_eq_cov IsGaussian.memLp_two_id,
          covarianceBilin_apply_eq_cov IsGaussian.memLp_two_id,
          covariance_map, covariance_map]
        · simp only [Function.comp_apply, SpinGlass.inner_std_basis_apply]
          dsimp only [pair, coordinateRandomCLM]
          rw [flippedSmart_cov path s sigma tau]
          let gamma := SYK.standardGaussianMeasureOnEuclidean
            (CoupledGaussianIndex N)
          have hgamma : gamma = stdGaussian
              (EuclideanSpace ℝ (CoupledGaussianIndex N)) := by
            exact ProbabilityTheory.map_pi_eq_stdGaussian
          change smartPathCovKernel N beta q s.1 sigma tau =
            cov[fun z => -inner ℝ
                (coupledDisorderCoefficient N beta q s.1 sigma) z,
              fun z => -inner ℝ
                (coupledDisorderCoefficient N beta q s.1 tau) z; gamma]
          rw [covariance_neg_left, covariance_neg_right, neg_neg, hgamma,
            ← covarianceBilin_apply_eq_cov IsGaussian.memLp_two_id,
            ProbabilityTheory.covarianceBilin_stdGaussian]
          simpa [innerSL_apply_apply, real_inner_comm] using
            (coefficient_inner N beta q s.1 s.2 hq sigma tau).symm
        all_goals fun_prop
      _ = covarianceBilin nu x y := (bilin_eq_sum_std N _ x y).symm

end SpinGlass.AT
