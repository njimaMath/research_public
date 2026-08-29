import GeneralizedLatala.Observables
import GeneralizedLatala.Endpoint.Independent
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.Integral
import Mathlib.MeasureTheory.Integral.Prod

/-!
# Independent-endpoint estimates

The Kearns--Saul and Hubbard--Stratonovich estimates at the decoupled endpoint.

Main declarations:
- `endpoint_subGaussian`
- `endpoint_quadratic`

Dependencies:
- smart-path observables and product Gaussian endpoint identities

This file corresponds to the relevant part of `blueprint_latala.txt`.
-/

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ENNReal NNReal Topology

set_option maxHeartbeats 800000

namespace SpinGlass
namespace GeneralizedLatala

universe uΩ uι

variable {Ω : Type uΩ} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

variable (N : ℕ) [NeZero N] (β h q : ℝ)
variable (sk : SKDisorder.{uΩ} (Ω := Ω) N β h)
variable (sim : SimpleDisorder.{uΩ} (Ω := Ω) N β q)

/-! ## Independent endpoint -/

/-- The one-site moment generating function for the product of two replicas. -/
noncomputable def localPairMGF (a q c : ℝ) : ℝ :=
  ∑ bs : Fin 2 → Bool,
    Real.exp (c * (boolSpin (bs 0) * boolSpin (bs 1) - q)) *
      ∏ l : Fin 2,
        Real.exp (-(a * boolSpin (bs l))) /
          (∑ b : Bool, Real.exp (-(a * boolSpin b)))

private lemma sum_replica_site_factor (N n : ℕ)
    (G : Fin N → (Fin n → Bool) → ℝ) :
    ∑ σs : ReplicaSpace N n, ∏ i, G i (fun l => σs l i) =
      ∏ i, ∑ bs : Fin n → Bool, G i bs := by
  classical
  rw [Fintype.prod_sum]
  exact Fintype.sum_equiv (transposeReplicaEquiv N n)
    (fun σs => ∏ i, G i (fun l => σs l i))
    (fun x => ∏ i, G i (x i)) (fun _ => rfl)

private lemma gibbs_average_siteEnergy_pair_mgf
    (N : ℕ) (a : Fin N → ℝ) (q c : ℝ) :
    gibbs_average_n_det (N := N) (n := 2) (siteEnergy N a)
        (fun σs => Real.exp
          (c * ∑ i : Fin N, (spin N (σs 0) i * spin N (σs 1) i - q))) =
      ∏ i : Fin N, localPairMGF (a i) q c := by
  classical
  simp only [gibbs_average_n_det, gibbs_pmf_siteEnergy, spin_eq_boolSpin]
  rw [show (∑ σs : ReplicaSpace N 2,
      Real.exp (c * ∑ i : Fin N,
        (boolSpin (σs 0 i) * boolSpin (σs 1 i) - q)) *
        ∏ l : Fin 2,
          (∏ i : Fin N,
            Real.exp (-(a i * boolSpin (σs l i))) /
              ∑ b : Bool, Real.exp (-(a i * boolSpin b)))) =
      ∑ σs : ReplicaSpace N 2,
        ∏ i : Fin N,
          (Real.exp (c *
              (boolSpin (σs 0 i) * boolSpin (σs 1 i) - q)) *
            ∏ l : Fin 2,
              Real.exp (-(a i * boolSpin (σs l i))) /
                ∑ b : Bool, Real.exp (-(a i * boolSpin b))) by
    congr 1
    funext σs
    rw [Finset.prod_comm]
    rw [Finset.mul_sum, Real.exp_sum]
    simp only [Finset.prod_mul_distrib]]
  exact sum_replica_site_factor N 2
    (fun i bs =>
      Real.exp (c * (boolSpin (bs 0) * boolSpin (bs 1) - q)) *
        ∏ l : Fin 2,
          Real.exp (-(a i * boolSpin (bs l))) /
            ∑ b : Bool, Real.exp (-(a i * boolSpin b)))

private lemma localPairMGF_eq (a q c : ℝ) :
    localPairMGF a q c =
      ((1 + Real.tanh a ^ 2) / 2) * Real.exp (c * (1 - q)) +
      ((1 - Real.tanh a ^ 2) / 2) * Real.exp (-c * (1 + q)) := by
  let F : (Fin 2 → Bool) → ℝ := fun bs =>
    Real.exp (c * (boolSpin (bs 0) * boolSpin (bs 1) - q)) *
      ∏ l : Fin 2,
        Real.exp (-(a * boolSpin (bs l))) /
          (∑ b : Bool, Real.exp (-(a * boolSpin b)))
  rw [show localPairMGF a q c = ∑ bs, F bs by rfl]
  rw [Fintype.sum_equiv (finTwoArrowEquiv Bool) F
    (fun p => F ((finTwoArrowEquiv Bool).symm p)) (by
      intro x
      apply congrArg F
      funext i
      fin_cases i <;> rfl)]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool]
  simp [F, boolSpin, Fin.prod_univ_two, Real.tanh_eq_sinh_div_cosh,
    Real.sinh_eq, Real.cosh_eq]
  ring_nf
  simp only [Real.exp_neg]
  field_simp [Real.exp_ne_zero]
  ring

/-- Kearns--Saul at the independent endpoint, in the form needed for the smart path. -/
lemma endpoint_subGaussian
    (hN : 0 < N) (hβ0 : 0 ≤ β) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hfp : IsRSFixedPoint β h q) (u : ℝ) :
    nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
        2 0
        (fun σs => Real.exp
          ((u / Real.sqrt N) *
            ∑ i : Fin N,
              (spin N (σs 0) i * spin N (σs 1) i - q)))
      ≤ Real.exp (kappa q * u ^ 2 / 2) := by
  classical
  let c : ℝ := u / Real.sqrt N
  let f : ReplicaFun N 2 := fun σs => Real.exp
    (c * ∑ i : Fin N, (spin N (σs 0) i * spin N (σs 1) i - q))
  let F : EnergySpace N → ℝ := fun H =>
    gibbs_average_n_det (N := N) (n := 2)
      (H + H_field (N := N) (h := h)) f
  have hFcont : Continuous F := by
    simp only [F, gibbs_average_n_det]
    apply continuous_finset_sum
    intro σs _
    apply Continuous.mul continuous_const
    apply continuous_finset_prod
    intro l _
    exact (SpinGlass.contDiff_gibbs_pmf (N := N) (σ := σs l)).continuous.comp
      (continuous_id.add continuous_const)
  have hHt0 (ω : Ω) :
      H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω =
        sim.V ω + H_field (N := N) (h := h) := by
    simp [H_t, H_gauss]
  have hrefLaw := referenceField_hasGaussianLaw N β q
  have hnu :
      nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0 f =
        ∫ z, F (referenceField N β q z) ∂gaussianProduct N := by
    calc
      nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0 f =
          ∫ ω, F (sim.V ω) ∂ℙ := by
            rw [nu]
            apply integral_congr_ae
            filter_upwards with ω
            simp only [gibbs_average_n]
            rw [hHt0]
      _ = ∫ H, F H ∂Measure.map sim.V ℙ := by
            rw [integral_map sim.hV.repr_measurable.aemeasurable hFcont.aestronglyMeasurable]
      _ = ∫ H, F H ∂Measure.map (referenceField N β q) (gaussianProduct N) := by
            rw [simpleDisorder_law_eq_reference N β q sim hN hβ0]
      _ = ∫ z, F (referenceField N β q z) ∂gaussianProduct N := by
            rw [integral_map hrefLaw.aemeasurable hFcont.aestronglyMeasurable]
  let A : ℝ :=
    ((1 + q) / 2) * Real.exp (c * (1 - q)) +
      ((1 - q) / 2) * Real.exp (-c * (1 + q))
  have htanh : Integrable
      (fun z : ℝ => Real.tanh (h + Real.sqrt β * z) ^ 2)
      (gaussianReal 0 1) := by
    have htanh_cont : Continuous Real.tanh := by
      rw [show Real.tanh = fun x => Real.sinh x / Real.cosh x by
        funext x
        exact Real.tanh_eq_sinh_div_cosh x]
      exact Real.continuous_sinh.div Real.continuous_cosh
        (fun x => (Real.cosh_pos x).ne')
    apply (integrable_const (1 : ℝ)).mono
    · exact (htanh_cont.comp (by fun_prop)).pow 2 |>.aestronglyMeasurable
    · filter_upwards with z
      simp only [Real.norm_eq_abs, abs_pow, abs_one]
      rw [sq_abs]
      exact (Real.tanh_sq_lt_one _).le
  have hlocal :
      ∫ z, localPairMGF (h + Real.sqrt β * z) q c ∂gaussianReal 0 1 = A := by
    have hT :
        ∫ z, Real.tanh (h + Real.sqrt β * z) ^ 2 ∂gaussianReal 0 1 = q := by
      simpa [IsRSFixedPoint, standardGaussianExpectation] using hfp.symm
    rw [show (∫ z, localPairMGF (h + Real.sqrt β * z) q c
          ∂gaussianReal 0 1) =
        ∫ z,
          ((Real.exp (c * (1 - q)) + Real.exp (-c * (1 + q))) / 2 +
            Real.tanh (h + Real.sqrt β * z) ^ 2 *
              ((Real.exp (c * (1 - q)) - Real.exp (-c * (1 + q))) / 2))
          ∂gaussianReal 0 1 by
      apply integral_congr_ae
      filter_upwards with z
      rw [localPairMGF_eq]
      ring]
    rw [integral_add (integrable_const _)
      (htanh.mul_const ((Real.exp (c * (1 - q)) - Real.exp (-c * (1 + q))) / 2))]
    simp only [integral_const, probReal_univ, one_smul, integral_mul_const, hT]
    simp only [A]
    ring_nf
  have hfactor :
      ∫ z, F (referenceField N β q z) ∂gaussianProduct N = A ^ N := by
    rw [show (∫ z, F (referenceField N β q z) ∂gaussianProduct N) =
        ∫ z, ∏ i : Fin N,
          localPairMGF (h + Real.sqrt β * z i) q c ∂gaussianProduct N by
      apply integral_congr_ae
      filter_upwards with z
      simp only [F]
      change gibbs_average_n_det (N := N) (n := 2)
        (referenceField N β q z + magnetic_field_vector (N := N) h) f = _
      rw [reference_add_field_eq_siteEnergy,
        gibbs_average_siteEnergy_pair_mgf]]
    rw [gaussianProduct]
    calc
      (∫ z : Fin N → ℝ, ∏ i : Fin N,
          localPairMGF (h + Real.sqrt β * z i) q c
          ∂Measure.pi (fun _ : Fin N => gaussianReal 0 1)) =
          (∫ z, localPairMGF (h + Real.sqrt β * z) q c
            ∂gaussianReal 0 1) ^ Fintype.card (Fin N) :=
        MeasureTheory.integral_fintype_prod_eq_pow
          (f := fun z : ℝ => localPairMGF (h + Real.sqrt β * z) q c)
      _ = A ^ N := by simpa using congrArg (fun x => x ^ N) hlocal
  have hkappa : kappa q = ksCoefficient q := by
    simp [kappa, ksCoefficient]
  have hKS : A ≤ Real.exp (kappa q * c ^ 2 / 2) := by
    simpa only [A, hkappa] using
      (kearns_saul_inequality (u := c) hq0 hq1)
  have hA0 : 0 ≤ A := by
    simp only [A]
    apply add_nonneg
    · exact mul_nonneg (div_nonneg (by linarith) (by norm_num)) (Real.exp_nonneg _)
    · exact mul_nonneg (div_nonneg (by linarith) (by norm_num)) (Real.exp_nonneg _)
  have hpow : A ^ N ≤ (Real.exp (kappa q * c ^ 2 / 2)) ^ N :=
    pow_le_pow_left₀ hA0 hKS N
  rw [show nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      2 0 (fun σs => Real.exp
        ((u / Real.sqrt N) * ∑ i : Fin N,
          (spin N (σs 0) i * spin N (σs 1) i - q))) = A ^ N by
    change nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      2 0 f = A ^ N
    exact hnu.trans hfactor]
  calc
    A ^ N ≤ (Real.exp (kappa q * c ^ 2 / 2)) ^ N := hpow
    _ = Real.exp (kappa q * u ^ 2 / 2) := by
      rw [← Real.exp_nat_mul]
      congr 1
      have hNr : 0 < (N : ℝ) := by exact_mod_cast hN
      simp only [c]
      rw [div_pow, Real.sq_sqrt hNr.le]
      field_simp [ne_of_gt hNr]

/-- Hubbard--Stratonovich combined with `endpoint_subGaussian`. -/
lemma endpoint_quadratic
    (hN : 0 < N) (hβ0 : 0 ≤ β) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hfp : IsRSFixedPoint β h q) {Λ : ℝ}
    (hΛ0 : 0 ≤ Λ) (hΛ : kappa q * Λ < 1) :
    logQuadraticMoment
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 (Λ / 2)
      ≤ (1 / 2) * Real.log (1 / (1 - kappa q * Λ)) := by
  classical
  let F : ReplicaFun N 2 := fun σs => Real.exp
    ((Λ / 2) * (N : ℝ) * centeredOverlapSq N q σs)
  let A : Ω → ℝ := gibbs_average_n
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0 F
  let S : ReplicaSpace N 2 → ℝ := fun σs =>
    ∑ i : Fin N, (spin N (σs 0) i * spin N (σs 1) i - q)
  let B : ℝ → Ω → ℝ := fun z ω =>
    gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0
      (fun σs => Real.exp ((Real.sqrt Λ * z / Real.sqrt N) * S σs)) ω
  have hNr : 0 < (N : ℝ) := by exact_mod_cast hN
  have hsqrtN : Real.sqrt (N : ℝ) ≠ 0 := Real.sqrt_ne_zero'.mpr hNr
  have hS (σs : ReplicaSpace N 2) :
      S σs = (N : ℝ) * (overlap N (σs 0) (σs 1) - q) := by
    simp only [S, overlap, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_fin,
      nsmul_eq_mul]
    field_simp
  have hHS (σs : ReplicaSpace N 2) :
      F σs = ∫ z, Real.exp ((Real.sqrt Λ * z / Real.sqrt N) * S σs)
        ∂gaussianReal 0 1 := by
    calc
      F σs = Real.exp (Λ * (Real.sqrt N *
          (overlap N (σs 0) (σs 1) - q)) ^ 2 / 2) := by
        simp only [F, centeredOverlapSq]
        congr 1
        rw [mul_pow, Real.sq_sqrt hNr.le]
        ring
      _ = ∫ z, Real.exp (Real.sqrt Λ *
          (Real.sqrt N * (overlap N (σs 0) (σs 1) - q)) * z)
          ∂gaussianReal 0 1 := hubbard_stratonovich Λ _ hΛ0
      _ = ∫ z, Real.exp ((Real.sqrt Λ * z / Real.sqrt N) * S σs)
          ∂gaussianReal 0 1 := by
        apply integral_congr_ae
        filter_upwards with z
        rw [hS]
        congr 1
        field_simp
        rw [Real.sq_sqrt hNr.le]
        ring
  have hlin_int (σs : ReplicaSpace N 2) :
      Integrable (fun z => Real.exp ((Real.sqrt Λ * z / Real.sqrt N) * S σs))
        (gaussianReal 0 1) := by
    convert integrable_exp_mul_gaussianReal
      (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) (Real.sqrt Λ * S σs / Real.sqrt N) using 1
    funext z
    congr 1
    ring
  have hAeq (ω : Ω) : A ω = ∫ z, B z ω ∂gaussianReal 0 1 := by
    simp only [A, B, gibbs_average_n, gibbs_average_n_det]
    rw [show (∑ σs : ReplicaSpace N 2,
        F σs * ∏ l, gibbs_pmf N
          (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω)
          (σs l)) =
        ∑ σs : ReplicaSpace N 2,
          (∫ z, Real.exp ((Real.sqrt Λ * z / Real.sqrt N) * S σs)
            ∂gaussianReal 0 1) *
          ∏ l, gibbs_pmf N
            (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω)
            (σs l) by
      congr 1
      funext σs
      rw [hHS]]
    simp_rw [← integral_mul_const]
    rw [integral_finset_sum]
    intro σs _
    exact (hlin_int σs).mul_const _
  have hAint : Integrable A ℙ := by
    exact integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0 F
  have hAone (ω : Ω) : 1 ≤ A ω := by
    simp only [A, gibbs_average_n, F, centeredOverlapSq, gibbs_average_n_det]
    rw [← sum_prod_gibbs_pmf_eq_one
      (N := N) (n := 2)
      (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω)]
    apply Finset.sum_le_sum
    intro σs _
    have hexp : 1 ≤ Real.exp
        (Λ / 2 * (N : ℝ) * (overlap N (σs 0) (σs 1) - q) ^ 2) :=
      Real.one_le_exp (mul_nonneg
        (mul_nonneg (div_nonneg hΛ0 (by norm_num)) (Nat.cast_nonneg N))
        (sq_nonneg _))
    have hweight : 0 ≤ ∏ l, gibbs_pmf N
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω)
        (σs l) := by
      apply Finset.prod_nonneg
      intro l _
      exact gibbs_pmf_nonneg
        (N := N)
        (H := H_t (N := N) (β := β) (h := h) (q := q)
          (sk := sk) (sim := sim) 0 ω)
        (σ := σs l)
    simpa only [one_mul] using mul_le_mul_of_nonneg_right hexp hweight
  have hlogAint : Integrable (fun ω => Real.log (A ω)) ℙ := by
    apply hAint.mono'
    · exact (Real.measurable_log.comp_aemeasurable hAint.aemeasurable).aestronglyMeasurable
    · filter_upwards with ω
      rw [Real.norm_eq_abs, abs_of_nonneg (Real.log_nonneg (hAone ω))]
      exact (Real.log_le_sub_one_of_pos (zero_lt_one.trans_le (hAone ω))).trans
        (sub_le_self _ zero_le_one)
  have hJensen : (∫ ω, Real.log (A ω) ∂ℙ) ≤ Real.log (∫ ω, A ω ∂ℙ) := by
    have hj := (strictConcaveOn_log_Ioi.concaveOn.subset
      (Set.Ici_subset_Ioi.2 zero_lt_one) (convex_Ici (1 : ℝ))).le_map_integral
      (f := A) (μ := ℙ)
      (Real.continuousOn_log.mono (by
        intro x hx
        simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
        exact ne_of_gt (zero_lt_one.trans_le hx)))
      isClosed_Ici (ae_of_all _ hAone) hAint
      (hlogAint.congr (ae_of_all _ fun ω => by rfl))
    exact hj
  have hweight_int (σs : ReplicaSpace N 2) : Integrable (fun ω =>
      ∏ l, gibbs_pmf N
        (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 0 ω)
        (σs l)) ℙ := by
    let I : ReplicaFun N 2 := fun τs => if τs = σs then 1 else 0
    have hi := integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) 2 0 I
    convert hi using 1
    funext ω
    simp [I, gibbs_average_n, gibbs_average_n_det]
  have hBprod : Integrable (fun p : ℝ × Ω => B p.1 p.2)
      ((gaussianReal 0 1).prod ℙ) := by
    simp only [B, gibbs_average_n, gibbs_average_n_det]
    apply integrable_finset_sum
    intro σs _
    exact (hlin_int σs).mul_prod (hweight_int σs)
  have hBbound (z : ℝ) :
      (∫ ω, B z ω ∂ℙ) ≤ Real.exp (kappa q * Λ * z ^ 2 / 2) := by
    have hend := endpoint_subGaussian
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      hN hβ0 hq0 hq1 hfp (Real.sqrt Λ * z)
    simpa only [B, S, nu, mul_pow, Real.sq_sqrt hΛ0, mul_assoc] using hend
  have hquad_int : Integrable (fun z : ℝ => Real.exp (kappa q * Λ * z ^ 2 / 2))
      (gaussianReal 0 1) := by
    have hi := ProbabilityTheory.integrable_polynomial_exp_sq_gaussian_param_nondeg
      (v := (1 : ℝ≥0)) (by norm_num) 0
      (s := kappa q * Λ / 2) (by norm_num; linarith)
    convert hi using 1
    funext z
    ring
  have hquad_eq :
      (∫ z, Real.exp (kappa q * Λ * z ^ 2 / 2) ∂gaussianReal 0 1) =
        1 / Real.sqrt (1 - kappa q * Λ) := by
    rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : ℝ≥0) ≠ 0)]
    simp only [smul_eq_mul, gaussianPDFReal]
    norm_num only [NNReal.coe_one, zero_sub, sub_zero, mul_one]
    rw [show (∫ x : ℝ, (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(x ^ 2) / 2) *
        Real.exp (kappa q * Λ * x ^ 2 / 2)) =
        (Real.sqrt (2 * Real.pi))⁻¹ *
          ∫ x : ℝ, Real.exp (-((1 - kappa q * Λ) / 2) * x ^ 2) by
      rw [← integral_const_mul]
      apply integral_congr_ae
      filter_upwards with x
      rw [mul_assoc, ← Real.exp_add]
      congr 2
      ring]
    rw [integral_gaussian]
    have hgap : 0 < 1 - kappa q * Λ := sub_pos.mpr hΛ
    rw [show Real.pi / ((1 - kappa q * Λ) / 2) =
        (2 * Real.pi) / (1 - kappa q * Λ) by field_simp]
    rw [Real.sqrt_div (by positivity : 0 ≤ 2 * Real.pi)]
    rw [Real.sqrt_mul (by norm_num : 0 ≤ (2 : ℝ))]
    field_simp [Real.sqrt_ne_zero'.mpr (by positivity : 0 < 2 * Real.pi),
      Real.sqrt_ne_zero'.mpr hgap]
  have hAmean : (∫ ω, A ω ∂ℙ) ≤ 1 / Real.sqrt (1 - kappa q * Λ) := by
    calc
      (∫ ω, A ω ∂ℙ) = ∫ ω, ∫ z, B z ω ∂gaussianReal 0 1 ∂ℙ := by
        apply integral_congr_ae
        exact ae_of_all _ hAeq
      _ = ∫ z, ∫ ω, B z ω ∂ℙ ∂gaussianReal 0 1 := by
        exact (integral_integral_swap hBprod).symm
      _ ≤ ∫ z, Real.exp (kappa q * Λ * z ^ 2 / 2) ∂gaussianReal 0 1 := by
        exact integral_mono hBprod.integral_prod_left hquad_int hBbound
      _ = 1 / Real.sqrt (1 - kappa q * Λ) := hquad_eq
  have hAmean_one : 1 ≤ ∫ ω, A ω ∂ℙ := by
    simpa only [integral_const, probReal_univ, one_smul] using
      integral_mono (integrable_const (1 : ℝ)) hAint hAone
  change (∫ ω, Real.log (A ω) ∂ℙ) ≤
    (1 / 2) * Real.log (1 / (1 - kappa q * Λ))
  calc
    (∫ ω, Real.log (A ω) ∂ℙ) ≤ Real.log (∫ ω, A ω ∂ℙ) := hJensen
    _ ≤ Real.log (1 / Real.sqrt (1 - kappa q * Λ)) :=
      Real.log_le_log (zero_lt_one.trans_le hAmean_one) hAmean
    _ = (1 / 2) * Real.log (1 / (1 - kappa q * Λ)) := by
      have hgap : 0 ≤ 1 - kappa q * Λ := (sub_pos.mpr hΛ).le
      simp only [one_div, ← Real.sqrt_inv]
      rw [Real.log_sqrt (inv_nonneg.mpr hgap)]
      ring


end GeneralizedLatala
end SpinGlass
