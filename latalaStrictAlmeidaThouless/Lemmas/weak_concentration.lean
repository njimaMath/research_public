import Lemmas.Concentration_Coupled_Transport
import Lemmas.GTFlatness
import Lemmas.GTbound.GTBound

open MeasureTheory ProbabilityTheory Real BigOperators

set_option autoImplicit false

namespace SpinGlass.AT

private lemma integrable_log_sum_exp
    {Omega I : Type*} [MeasureSpace Omega] [IsFiniteMeasure (volume : Measure Omega)]
    [Fintype I] [Nonempty I]
    (F : I → Omega → ℝ) (hF : ∀ i, Integrable (F i)) :
    Integrable (fun omega => Real.log (∑ i, Real.exp (F i omega))) := by
  let A : Omega → ℝ := fun omega => ∑ i, |F i omega|
  have hA : Integrable A := integrable_finsetSum Finset.univ fun i _ => (hF i).abs
  have hmeas : AEStronglyMeasurable
      (fun omega => Real.log (∑ i, Real.exp (F i omega))) := by
    exact ((Finset.aemeasurable_fun_sum Finset.univ fun i _ =>
      (hF i).aemeasurable.exp).log).aestronglyMeasurable
  have hcard : (1 : ℝ) ≤ Fintype.card I := by exact_mod_cast Fintype.card_pos
  have hlogcard : 0 ≤ Real.log (Fintype.card I : ℝ) := Real.log_nonneg hcard
  refine (hA.add (integrable_const (Real.log (Fintype.card I : ℝ)))).mono' hmeas ?_
  filter_upwards [] with omega
  have hsumpos : 0 < ∑ i, Real.exp (F i omega) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hupper : Real.log (∑ i, Real.exp (F i omega)) ≤
      Real.log (Fintype.card I : ℝ) + A omega := by
    have hsum : (∑ i, Real.exp (F i omega)) ≤
        (Fintype.card I : ℝ) * Real.exp (A omega) := by
      calc
        (∑ i, Real.exp (F i omega)) ≤ ∑ _i : I, Real.exp (A omega) := by
          apply Finset.sum_le_sum
          intro i _
          apply Real.exp_le_exp.mpr
          exact (le_abs_self _).trans (Finset.single_le_sum
            (fun j _ => abs_nonneg (F j omega)) (Finset.mem_univ i))
        _ = _ := by simp
    have hlog := Real.log_le_log hsumpos hsum
    rw [Real.log_mul (by positivity) (Real.exp_ne_zero _), Real.log_exp] at hlog
    exact hlog
  let i0 : I := Classical.choice inferInstance
  have hlower : -(A omega) ≤ Real.log (∑ i, Real.exp (F i omega)) := by
    have hone : Real.exp (F i0 omega) ≤ ∑ i, Real.exp (F i omega) :=
      Finset.single_le_sum (fun i _ => Real.exp_nonneg (F i omega))
        (Finset.mem_univ i0)
    have hlog := Real.log_le_log (Real.exp_pos _) hone
    rw [Real.log_exp] at hlog
    have hFi : -(A omega) ≤ F i0 omega := by
      have hi := Finset.single_le_sum (fun j _ => abs_nonneg (F j omega))
        (Finset.mem_univ i0)
      exact (neg_le_neg hi).trans (neg_abs_le (F i0 omega))
    exact hFi.trans hlog
  rw [Real.norm_eq_abs]
  change |Real.log (∑ i, Real.exp (F i omega))| ≤
    A omega + Real.log (Fintype.card I : ℝ)
  apply abs_le.mpr
  constructor <;> linarith

private lemma integral_log_sum_exp_le
    {Omega I : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    [Fintype I] [Nonempty I]
    (F : I → Omega → ℝ) (a : I → ℝ) (B : ℝ)
    (hF : ∀ i, Integrable (F i))
    (hB : ∀ i, a i + ∫ omega, F i omega ∂volume ≤ B) :
    (∫ omega, Real.log (∑ i, Real.exp (a i + F i omega)) ∂volume) ≤
      B + Real.log (Fintype.card I : ℝ) +
        ∫ omega, Finset.univ.sup' Finset.univ_nonempty
          (fun i => F i omega - ∫ eta, F i eta ∂volume) ∂volume := by
  let X : I → Omega → ℝ := fun i omega =>
    F i omega - ∫ eta, F i eta ∂volume
  let M : Omega → ℝ := fun omega =>
    Finset.univ.sup' Finset.univ_nonempty fun i => X i omega
  have hX (i : I) : Integrable (X i) := (hF i).sub (integrable_const _)
  have hM : Integrable M := by
    have hsup : Integrable (Finset.univ.sup' Finset.univ_nonempty fun i => X i) := by
      exact Finset.sup'_induction Finset.univ_nonempty (fun i => X i)
        (p := fun f : Omega → ℝ => Integrable f)
        (fun _ hf _ hg => hf.sup hg) (fun i _ => hX i)
    refine hsup.congr ?_
    filter_upwards [] with omega
    exact Finset.sup'_apply Finset.univ_nonempty (fun i => X i) omega
  have hlog : Integrable (fun omega =>
      Real.log (∑ i, Real.exp (a i + F i omega))) := by
    apply integrable_log_sum_exp (fun i omega => a i + F i omega)
    intro i
    exact (integrable_const _).add (hF i)
  have hrhs : Integrable (fun omega =>
      B + Real.log (Fintype.card I : ℝ) + M omega) :=
    (integrable_const _).add hM
  have hpoint (omega : Omega) :
      Real.log (∑ i, Real.exp (a i + F i omega)) ≤
        B + Real.log (Fintype.card I : ℝ) + M omega := by
    have hterm (i : I) : a i + F i omega ≤ B + M omega := by
      have hi : X i omega ≤ M omega :=
        Finset.le_sup' (fun i => X i omega) (Finset.mem_univ i)
      dsimp [X] at hi
      linarith [hB i]
    have hsum : (∑ i, Real.exp (a i + F i omega)) ≤
        (Fintype.card I : ℝ) * Real.exp (B + M omega) := by
      calc
        _ ≤ ∑ _i : I, Real.exp (B + M omega) :=
          Finset.sum_le_sum fun i _ => Real.exp_le_exp.mpr (hterm i)
        _ = _ := by simp
    have hsumpos : 0 < ∑ i, Real.exp (a i + F i omega) :=
      Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
    have hcardpos : 0 < (Fintype.card I : ℝ) := by exact_mod_cast Fintype.card_pos
    have h := Real.log_le_log hsumpos hsum
    rw [Real.log_mul hcardpos.ne' (Real.exp_ne_zero _), Real.log_exp] at h
    linarith
  calc
    (∫ omega, Real.log (∑ i, Real.exp (a i + F i omega)) ∂volume) ≤
        ∫ omega, B + Real.log (Fintype.card I : ℝ) + M omega ∂volume :=
      integral_mono hlog hrhs hpoint
    _ = B + Real.log (Fintype.card I : ℝ) + ∫ omega, M omega ∂volume := by
      rw [integral_add (integrable_const _) hM]
      simp
    _ = _ := rfl

private lemma fullPath_eval_integrable
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {beta h q s : ℝ}
    (path : RSSmartPathDisorder Omega N beta h q)
    (sigma : SpinGlass.Config N) :
    Integrable (fun omega => fullPathHamiltonian path s omega sigma) := by
  have hU := (show HasGaussianLaw path.sk.U volume from by
      -- The public law bridge guarantees measurability and Gaussian moments.
      exact SpinGlass.AT.gaussianHilbert_hasGaussianLaw path.sk.hU)
  have hV := (show HasGaussianLaw path.simple.V volume from by
      exact SpinGlass.AT.gaussianHilbert_hasGaussianLaw path.simple.hV)
  have hUeval : Integrable (fun omega => path.sk.U omega sigma) := by
    simpa using (hU.map_fun (SpinGlass.evalCLM (N := N) sigma)).integrable
  have hVeval : Integrable (fun omega => path.simple.V omega sigma) := by
    simpa using (hV.map_fun (SpinGlass.evalCLM (N := N) sigma)).integrable
  change Integrable (fun omega =>
    Real.sqrt s * path.sk.U omega sigma +
      Real.sqrt (1 - s) * path.simple.V omega sigma +
        SpinGlass.magnetic_field_vector N h sigma)
  exact ((hUeval.const_mul _).add (hVeval.const_mul _)).add (integrable_const _)

private lemma constrained_log_integrable
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {N : ℕ} {beta h q s v : ℝ}
    (path : RSSmartPathDisorder Omega N beta h q)
    (hv : v ∈ attainableOverlaps N) :
    Integrable (fun omega => Real.log
      (constrainedPartition (fullPathHamiltonian path s omega) v)) := by
  classical
  let P := {p : SpinGlass.Config N × SpinGlass.Config N //
    SpinGlass.overlap N p.1 p.2 = v}
  rw [attainableOverlaps, Finset.mem_image] at hv
  obtain ⟨p, _hp, hpv⟩ := hv
  letI : Nonempty P := ⟨⟨p, hpv⟩⟩
  let F : P → Omega → ℝ := fun p omega =>
    -(fullPathHamiltonian path s omega p.1.1 +
      fullPathHamiltonian path s omega p.1.2)
  have hF (p : P) : Integrable (F p) :=
    ((fullPath_eval_integrable path p.1.1).add
      (fullPath_eval_integrable path p.1.2)).neg
  have hint := integrable_log_sum_exp F hF
  convert hint using 1
  funext omega
  congr 1
  unfold constrainedPartition
  rw [← Finset.sum_filter]
  rw [Finset.sum_subtype (p := fun p : SpinGlass.Config N × SpinGlass.Config N =>
    SpinGlass.overlap N p.1 p.2 = v)
    (Finset.univ.filter fun p : SpinGlass.Config N × SpinGlass.Config N =>
      SpinGlass.overlap N p.1 p.2 = v) (by simp)]

private lemma constrainedPartition_pos_of_attainable
    {N : ℕ} (H : SpinGlass.EnergySpace N) {v : ℝ}
    (hv : v ∈ attainableOverlaps N) :
    0 < constrainedPartition H v := by
  classical
  rw [attainableOverlaps, Finset.mem_image] at hv
  obtain ⟨p, _hp, hpv⟩ := hv
  unfold constrainedPartition
  exact Finset.sum_pos'
    (fun p _ => by split <;> positivity)
    ⟨p, Finset.mem_univ p, by rw [if_pos hpv]; exact Real.exp_pos _⟩

private lemma attainableOverlap_mem_Icc
    {N : ℕ} (hN : 0 < N) {v : ℝ}
    (hv : v ∈ attainableOverlaps N) : v ∈ Set.Icc (-1 : ℝ) 1 := by
  classical
  rw [attainableOverlaps, Finset.mem_image] at hv
  obtain ⟨p, _hp, rfl⟩ := hv
  have hterm (i : Fin N) :
      -1 ≤ SpinGlass.spin N p.1 i * SpinGlass.spin N p.2 i ∧
        SpinGlass.spin N p.1 i * SpinGlass.spin N p.2 i ≤ 1 := by
    cases h1 : p.1 i <;> cases h2 : p.2 i <;>
      simp [SpinGlass.spin, h1, h2]
  have hlo : -(N : ℝ) ≤
      ∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N p.2 i := by
    calc
      -(N : ℝ) = ∑ _i : Fin N, (-1 : ℝ) := by simp
      _ ≤ _ := Finset.sum_le_sum fun i _ => (hterm i).1
  have hhi : (∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N p.2 i) ≤
      (N : ℝ) := by
    calc
      _ ≤ ∑ _i : Fin N, (1 : ℝ) :=
        Finset.sum_le_sum fun i _ => (hterm i).2
      _ = (N : ℝ) := by simp
  have hNr : 0 < (N : ℝ) := by exact_mod_cast hN
  unfold SpinGlass.overlap
  have hinv : 0 < 1 / (N : ℝ) := one_div_pos.mpr hNr
  constructor
  · calc
      (-1 : ℝ) = (1 / (N : ℝ)) * (-(N : ℝ)) := by field_simp
      _ ≤ _ := mul_le_mul_of_nonneg_left hlo hinv.le
  · calc
      (1 / (N : ℝ)) *
          ∑ i, SpinGlass.spin N p.1 i * SpinGlass.spin N p.2 i ≤
          (1 / (N : ℝ)) * (N : ℝ) :=
        mul_le_mul_of_nonneg_left hhi hinv.le
      _ = 1 := by field_simp

private lemma rsPathValue_nonneg (beta h q : ℝ) (s : Set.Icc (0 : ℝ) 1) :
    0 ≤ rsPathValue beta h q s.1 := by
  have hlog2 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have hint : 0 ≤ standardGaussianExpectation
      (fun z => Real.log (Real.cosh (h + beta * Real.sqrt q * z))) := by
    unfold standardGaussianExpectation
    exact integral_nonneg fun z => Real.log_nonneg (Real.one_le_cosh _)
  have hcorr : 0 ≤ s.1 * beta ^ 2 / 4 * (1 - q) ^ 2 :=
    mul_nonneg
      (div_nonneg (mul_nonneg s.2.1 (sq_nonneg beta)) (by norm_num))
      (sq_nonneg (1 - q))
  unfold rsPathValue
  linarith

/-- A uniform sublinear bound for the quadratically coupled pressure along
the replica-symmetric smart path, including the degenerate case `N = 0`. -/
theorem quadraticCoupledPressure_sublinear
    {Omega : Type*} [MeasureSpace Omega]
    [IsProbabilityMeasure (volume : Measure Omega)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ rho0 > 0, ∃ C > 0,
      ∀ {N : ℕ},
      ∀ {beta h : ℝ}, (beta, h) ∈ K →
      ∀ s : Set.Icc (0 : ℝ) 1,
      ∀ path : RSSmartPathDisorder Omega N beta h (rsQ beta h),
        quadraticCoupledPressure path s.1 rho0 ≤
          rsPathValue beta h (rsQ beta h) s.1 +
            C * Real.sqrt
              (Real.log ((N : ℝ) + 1) / (N : ℝ)) := by
  obtain ⟨c, hc, hgap⟩ := gtFunctional_uniform_quadratic_gap data
  refine ⟨c, hc, 1 + 2 * data.βmax, by nlinarith [data.βmax_pos], ?_⟩
  intro N beta h hp s path
  by_cases hN : N = 0
  · subst N
    simp only [quadraticCoupledPressure, Nat.cast_zero, mul_zero, div_zero,
      zero_mul, zero_add, Real.log_one, Real.sqrt_zero]
    simpa using rsPathValue_nonneg beta h (rsQ beta h) s
  · have hNpos : 0 < N := Nat.pos_of_ne_zero hN
    let I := {v : ℝ // v ∈ attainableOverlaps N}
    let q := rsQ beta h
    let P := rsPathValue beta h q s.1
    let F : I → Omega → ℝ := fun v omega =>
      Real.log (constrainedPartition
        (fullPathHamiltonian path s.1 omega) v.1)
    let a : I → ℝ := fun v => c * (N : ℝ) / 2 * (v.1 - q) ^ 2
    have hF (v : I) : Integrable (F v) := constrained_log_integrable path v.2
    have hB (v : I) :
        a v + ∫ omega, F v omega ∂volume ≤ 2 * (N : ℝ) * P := by
      have hvIcc : v.1 ∈ Set.Icc (-1 : ℝ) 1 :=
        attainableOverlap_mem_Icc hNpos v.2
      obtain ⟨lam, hlam, hflat⟩ :=
        hgap hp rfl s.2 hvIcc
      have hgt := twoReplica_GT_bound path lam hNpos s.2 v.2
      have hcombined := hgt.trans hflat
      unfold expectedConstrainedFreeEnergy at hcombined
      have hNr : 0 < (N : ℝ) := by exact_mod_cast hNpos
      have hscaled := mul_le_mul_of_nonneg_left hcombined hNr.le
      have hmean :
          (∫ omega, F v omega ∂volume) ≤
            (N : ℝ) * (2 * P - c * (v.1 - q) ^ 2) := by
        calc
          (∫ omega, F v omega ∂volume) =
              (N : ℝ) * (1 / (N : ℝ) *
                ∫ omega, F v omega ∂volume) := by field_simp
          _ ≤ (N : ℝ) * (2 * P - c * (v.1 - q) ^ 2) := by
            simpa [F, P, q] using hscaled
      dsimp [a]
      nlinarith [sq_nonneg (v.1 - q), mul_pos hNr hc]
    have hsum := integral_log_sum_exp_le F a (2 * (N : ℝ) * P) hF hB
    have hpartition (omega : Omega) :
        Real.log (quadraticCoupledPartition
          (fullPathHamiltonian path s.1 omega) q c) =
          Real.log (∑ v : I, Real.exp (a v + F v omega)) := by
      congr 1
      rw [quadraticCoupledPartition_eq_sum_constrained]
      symm
      rw [Finset.sum_subtype (p := fun v : ℝ => v ∈ attainableOverlaps N)
        (attainableOverlaps N) (by simp)]
      apply Finset.sum_congr rfl
      intro v hv
      dsimp [a, F, q]
      rw [Real.exp_add, Real.exp_log
        (constrainedPartition_pos_of_attainable _ v.2)]
    have hsum' :
        (∫ omega, Real.log (quadraticCoupledPartition
          (fullPathHamiltonian path s.1 omega) q c) ∂volume) ≤
          2 * (N : ℝ) * P + Real.log (Fintype.card I : ℝ) +
            ∫ omega, Finset.univ.sup' Finset.univ_nonempty
              (fun v : I => F v omega - ∫ eta, F v eta ∂volume) ∂volume := by
      simpa only [hpartition] using hsum
    have hmax := coupled_log_partition_gaussian_max_path
      data N beta h s hp path
    have hmax' :
        (∫ omega, Finset.univ.sup' Finset.univ_nonempty
          (fun v : I => F v omega - ∫ eta, F v eta ∂volume) ∂volume) ≤
          2 * data.βmax * Real.sqrt N *
            Real.sqrt (2 * Real.log ((N : ℝ) + 1)) := by
      simpa [I, F] using hmax
    have hcard : Fintype.card I ≤ N + 1 := by
      simpa [I] using card_attainableOverlaps_le N
    have hcardpos : 0 < (Fintype.card I : ℝ) := by
      exact_mod_cast Fintype.card_pos
    have hcardreal : (Fintype.card I : ℝ) ≤ (N : ℝ) + 1 := by
      exact_mod_cast hcard
    have hlogcard : Real.log (Fintype.card I : ℝ) ≤
        Real.log ((N : ℝ) + 1) :=
      Real.log_le_log hcardpos hcardreal
    have htotal :
        (∫ omega, Real.log (quadraticCoupledPartition
          (fullPathHamiltonian path s.1 omega) q c) ∂volume) ≤
          2 * (N : ℝ) * P + Real.log ((N : ℝ) + 1) +
            2 * data.βmax * Real.sqrt N *
              Real.sqrt (2 * Real.log ((N : ℝ) + 1)) := by
      linarith
    let n : ℝ := N
    let ell : ℝ := Real.log (n + 1)
    let x : ℝ := ell / n
    let r : ℝ := Real.sqrt x
    have hn : 0 < n := by
      dsimp [n]
      exact_mod_cast hNpos
    have hn1 : 1 ≤ n + 1 := by linarith
    have hell : 0 ≤ ell := Real.log_nonneg hn1
    have hell_le : ell ≤ n := by
      have := Real.log_le_sub_one_of_pos (show 0 < n + 1 by linarith)
      dsimp [ell]
      linarith
    have hx : 0 ≤ x := div_nonneg hell hn.le
    have hx_le : x ≤ 1 := (div_le_one hn).2 hell_le
    have hr : 0 ≤ r := Real.sqrt_nonneg _
    have hr_sq : r ^ 2 = x := by
      dsimp [r]
      exact Real.sq_sqrt hx
    have hx_sqrt : x ≤ r := by
      nlinarith
    have hsqrt_two : Real.sqrt 2 ≤ 2 := by
      nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2),
        Real.sqrt_nonneg 2]
    have hsqrt_product : Real.sqrt (2 * ell) ≤ 2 * Real.sqrt ell := by
      rw [Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
      exact mul_le_mul_of_nonneg_right hsqrt_two (Real.sqrt_nonneg ell)
    have hsqrtn : 0 < Real.sqrt n := Real.sqrt_pos.2 hn
    have hsqrtn_sq : (Real.sqrt n) ^ 2 = n := Real.sq_sqrt hn.le
    have hratio : Real.sqrt n * Real.sqrt ell / n = r := by
      have hrform : r = Real.sqrt ell / Real.sqrt n := by
        dsimp [r, x]
        rw [Real.sqrt_div hell]
      rw [hrform]
      field_simp [hn.ne', hsqrtn.ne']
      rw [hsqrtn_sq]
      ring
    have hlog_error : (1 / (2 * n)) * ell ≤ r := by
      have heq : (1 / (2 * n)) * ell = x / 2 := by
        dsimp [x]
        field_simp [hn.ne']
      rw [heq]
      linarith
    have hgaussian_error :
        (1 / (2 * n)) *
            (2 * data.βmax * Real.sqrt n * Real.sqrt (2 * ell)) ≤
          2 * data.βmax * r := by
      calc
        (1 / (2 * n)) *
            (2 * data.βmax * Real.sqrt n * Real.sqrt (2 * ell)) ≤
            (1 / (2 * n)) *
              (2 * data.βmax * Real.sqrt n *
                (2 * Real.sqrt ell)) := by
          apply mul_le_mul_of_nonneg_left
          · apply mul_le_mul_of_nonneg_left hsqrt_product
            exact mul_nonneg
              (mul_nonneg (by norm_num) data.βmax_pos.le)
              (Real.sqrt_nonneg n)
          · positivity
        _ = 2 * data.βmax * r := by
          rw [← hratio]
          field_simp [hn.ne']
    have herror :
        (1 / (2 * n)) *
            (ell + 2 * data.βmax * Real.sqrt n * Real.sqrt (2 * ell)) ≤
          (1 + 2 * data.βmax) * r := by
      rw [mul_add]
      nlinarith
    unfold quadraticCoupledPressure
    change (1 / (2 * n)) *
        ∫ omega, Real.log (quadraticCoupledPartition
          (fullPathHamiltonian path s.1 omega) q c) ∂volume ≤
        P + (1 + 2 * data.βmax) * r
    calc
      (1 / (2 * n)) *
          ∫ omega, Real.log (quadraticCoupledPartition
            (fullPathHamiltonian path s.1 omega) q c) ∂volume ≤
          (1 / (2 * n)) *
            (2 * n * P + ell +
              2 * data.βmax * Real.sqrt n * Real.sqrt (2 * ell)) := by
        apply mul_le_mul_of_nonneg_left
        · simpa [n, ell, q, P] using htotal
        · positivity
      _ = P + (1 / (2 * n)) *
            (ell + 2 * data.βmax * Real.sqrt n * Real.sqrt (2 * ell)) := by
        field_simp [hn.ne']
        ring
      _ ≤ P + (1 + 2 * data.βmax) * r := by
        simpa [add_comm] using add_le_add_left herror P

end SpinGlass.AT
