import Lemmas.MainResult
import Lemmas.Cavity.TalagrandCavity

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Uniform AT data for a single point of the strict AT region. -/
noncomputable def singletonUniformATData {β h : ℝ}
    (hβ : 0 < β) (hh : 0 < h) (hAT : atParameter β h < 1) :
    UniformATData ({(β, h)} : Set (ℝ × ℝ)) where
  isCompact := isCompact_singleton
  βmax := β
  qmin := rsQ β h
  gap := 1 - atParameter β h
  βmax_pos := hβ
  qmin_pos := rsQ_pos hβ hh
  gap_pos := sub_pos.mpr hAT
  β_pos := by
    intro p hp
    rw [Set.mem_singleton_iff.mp hp]
    exact hβ
  h_pos := by
    intro p hp
    rw [Set.mem_singleton_iff.mp hp]
    exact hh
  β_bound := by
    intro p hp
    rw [Set.mem_singleton_iff.mp hp]
  q_lower := by
    intro p hp
    rw [Set.mem_singleton_iff.mp hp]
  strictAT := by
    intro p hp
    rw [Set.mem_singleton_iff.mp hp]
    linarith

lemma thirdMoment_eq_twoReplica_clt
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} [NeZero N] {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) :
    thirdMoment path s =
      SpinGlass.nu
        (N := N) (β := β) (h := h) (q := q)
        (sk := path.sk) (sim := path.simple) 2 s
        (fun σs => |SpinGlass.overlap N (σs 0) (σs 1) - q| ^ 3) := by
  unfold thirdMoment quenchedReplicaAverage SpinGlass.nu SpinGlass.gibbs_average_n
  apply integral_congr_ae
  filter_upwards with ω
  change replicaGibbsAverage (fullPathHamiltonian path s ω)
      (fun σs : Replicas N 4 => |centeredOverlap q σs 0 1| ^ 3) =
    SpinGlass.gibbs_average_n_det (N := N) (n := 2)
      (fullPathHamiltonian path s ω)
      (fun σs => |SpinGlass.overlap N (σs 0) (σs 1) - q| ^ 3)
  exact fourReplica_firstPair_eq_two (fullPathHamiltonian path s ω) q
    (fun x => |x| ^ 3)

lemma quenchedTail_eq_twoReplica_clt
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} [NeZero N] {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s eps : ℝ) :
    quenchedTail path s eps =
      SpinGlass.nu
        (N := N) (β := β) (h := h) (q := q)
        (sk := path.sk) (sim := path.simple) 2 s
        (fun σs => if eps ≤
          |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0) := by
  unfold quenchedTail quenchedReplicaAverage SpinGlass.nu SpinGlass.gibbs_average_n
  apply integral_congr_ae
  filter_upwards with ω
  change replicaGibbsAverage (fullPathHamiltonian path s ω)
      (fun σs : Replicas N 4 =>
        if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0) =
    SpinGlass.gibbs_average_n_det (N := N) (n := 2)
      (fullPathHamiltonian path s ω)
      (fun σs => if eps ≤
        |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0)
  exact fourReplica_firstPair_eq_two (fullPathHamiltonian path s ω) q
    (fun x => if eps ≤ |x| then 1 else 0)

lemma thirdMoment_split_clt
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} [NeZero N] {β h q s eps : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hq : q ∈ Set.Icc (0 : ℝ) 1) (heps : 0 ≤ eps) :
    thirdMoment path s ≤ eps * A path s + 8 * quenchedTail path s eps := by
  rw [thirdMoment_eq_twoReplica_clt, A_eq_overlapVariance,
    quenchedTail_eq_twoReplica_clt]
  unfold SpinGlass.GeneralizedLatala.overlapVariance SpinGlass.nu
  rw [← MeasureTheory.integral_const_mul, ← MeasureTheory.integral_const_mul]
  rw [← MeasureTheory.integral_add
    ((SpinGlass.integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q)
      (sk := path.sk) (sim := path.simple) 2 s
      (SpinGlass.GeneralizedLatala.centeredOverlapSq N q)).const_mul eps)
    ((SpinGlass.integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q)
      (sk := path.sk) (sim := path.simple) 2 s
      (fun σs => if eps ≤
        |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0)).const_mul 8)]
  apply integral_mono
  · exact SpinGlass.integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q)
      (sk := path.sk) (sim := path.simple) 2 s _
  · exact ((SpinGlass.integrable_gibbs_average_n
      (N := N) (β := β) (h := h) (q := q)
      (sk := path.sk) (sim := path.simple) 2 s
      (SpinGlass.GeneralizedLatala.centeredOverlapSq N q)).const_mul eps).add
      ((SpinGlass.integrable_gibbs_average_n
        (N := N) (β := β) (h := h) (q := q)
        (sk := path.sk) (sim := path.simple) 2 s
        (fun σs => if eps ≤
          |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0)).const_mul 8)
  · intro ω
    change SpinGlass.gibbs_average_n_det (N := N) (n := 2)
        (fullPathHamiltonian path s ω)
        (fun σs => |SpinGlass.overlap N (σs 0) (σs 1) - q| ^ 3) ≤
      eps * SpinGlass.gibbs_average_n_det (N := N) (n := 2)
        (fullPathHamiltonian path s ω)
        (SpinGlass.GeneralizedLatala.centeredOverlapSq N q) +
      8 * SpinGlass.gibbs_average_n_det (N := N) (n := 2)
        (fullPathHamiltonian path s ω)
        (fun σs => if eps ≤
          |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0)
    unfold SpinGlass.gibbs_average_n_det
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro σs _
    let x := |SpinGlass.overlap N (σs 0) (σs 1) - q|
    have hx0 : 0 ≤ x := abs_nonneg _
    have hx2 : x ≤ 2 := by
      have hover : SpinGlass.overlap N (σs 0) (σs 1) ∈ Set.Icc (-1 : ℝ) 1 := by
        apply attainableOverlap_mem_Icc hN
        simp only [attainableOverlaps, Finset.mem_image]
        exact ⟨(σs 0, σs 1), Finset.mem_univ _, rfl⟩
      rw [abs_le]
      constructor <;> linarith [hover.1, hover.2, hq.1, hq.2]
    have hw : 0 ≤ ∏ i, SpinGlass.gibbs_pmf N
        (fullPathHamiltonian path s ω) (σs i) :=
      Finset.prod_nonneg fun i _ => SpinGlass.gibbs_pmf_nonneg
        (N := N) (H := fullPathHamiltonian path s ω) (σs i)
    have hpoint :
        |SpinGlass.overlap N (σs 0) (σs 1) - q| ^ 3 ≤
          eps * (SpinGlass.overlap N (σs 0) (σs 1) - q) ^ 2 +
            8 * (if eps ≤ |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0) := by
      by_cases ht : eps ≤ x
      · rw [if_pos ht]
        have hx3 : x ^ 3 ≤ 8 := by nlinarith [sq_nonneg (x - 2)]
        dsimp [x] at hx3 ⊢
        nlinarith [mul_nonneg heps (sq_nonneg
          (SpinGlass.overlap N (σs 0) (σs 1) - q))]
      · rw [if_neg ht]
        have hxe : x ≤ eps := (lt_of_not_ge ht).le
        dsimp [x] at hxe ⊢
        have habssq : |SpinGlass.overlap N (σs 0) (σs 1) - q| ^ 2 =
            (SpinGlass.overlap N (σs 0) (σs 1) - q) ^ 2 :=
          sq_abs _
        nlinarith [abs_nonneg (SpinGlass.overlap N (σs 0) (σs 1) - q),
          sq_nonneg (SpinGlass.overlap N (σs 0) (σs 1) - q)]
    calc
      _ ≤ (eps * (SpinGlass.overlap N (σs 0) (σs 1) - q) ^ 2 +
          8 * (if eps ≤ |SpinGlass.overlap N (σs 0) (σs 1) - q| then 1 else 0)) *
          ∏ i, SpinGlass.gibbs_pmf N (fullPathHamiltonian path s ω) (σs i) :=
        mul_le_mul_of_nonneg_right hpoint hw
      _ = _ := by
        unfold SpinGlass.GeneralizedLatala.centeredOverlapSq
        ring

lemma nat_mul_exp_neg_tendsto_zero_clt (c : ℝ) (hc : 0 < c) :
    Tendsto (fun N : ℕ => (N : ℝ) * Real.exp (-c * (N : ℝ)))
      atTop (nhds 0) := by
  have hcN : Tendsto (fun N : ℕ => c * (N : ℝ)) atTop atTop :=
    tendsto_natCast_atTop_atTop.const_mul_atTop hc
  have hmain := (Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero 1).comp hcN
  have hcconst : Tendsto (fun _ : ℕ => c⁻¹) atTop (nhds c⁻¹) :=
    tendsto_const_nhds
  have hscaled := hcconst.mul hmain
  convert hscaled using 1
  · funext N
    field_simp [hc.ne']
    congr 1 <;> ring_nf
  · norm_num

/-- The cubic overlap error is negligible at the CLT scale. -/
theorem tendsto_scaled_thirdMoment_zero
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h)) :
    Tendsto (fun N : ℕ => (N.succ : ℝ) * thirdMoment (paths N) 1)
      atTop (nhds 0) := by
  let K : Set (ℝ × ℝ) := {(β, h)}
  let data : UniformATData K := singletonUniformATData hβ hh hAT
  have hp : (β, h) ∈ K := by simp [K]
  obtain ⟨M, hM, hsecond⟩ :=
    (quantitative_strictAT (Ω := Ω) K data).secondMoment
  rw [Metric.tendsto_atTop]
  intro δ hδ
  let eps : ℝ := δ / (2 * (M + 1))
  have heps : 0 < eps := div_pos hδ (by linarith)
  obtain ⟨c, hc, C, hC, htail⟩ := overlap_tail (Omega := Ω) data heps
  have hexp := nat_mul_exp_neg_tendsto_zero_clt c hc
  obtain ⟨N0, hN0⟩ := (Metric.tendsto_atTop.1 hexp) (δ / (16 * C)) (by positivity)
  use N0
  intro N hNN0
  let n := N.succ
  letI : NeZero n := ⟨Nat.succ_ne_zero N⟩
  have hn : 0 < n := Nat.succ_pos N
  have htail' : quenchedTail (paths N) 1 eps ≤
      C * Real.exp (-c * (n : ℝ)) := by
    rw [quenchedTail_eq_twoReplica_clt]
    exact htail hn hp ⟨1, by simp⟩ (paths N)
  have hthird := thirdMoment_split_clt (s := 1) (paths N) hn
    (rsQ_mem_Icc β h) heps.le
  have hA : (n : ℝ) * A (paths N) 1 ≤ M := by
    simpa [n] using hsecond hn hp rfl (by simp) (paths N)
  have hExpSmall : (n : ℝ) * Real.exp (-c * (n : ℝ)) < δ / (16 * C) := by
    have hd := hN0 n (le_trans hNN0 (Nat.le_succ N))
    simpa [Real.dist_eq, abs_of_nonneg
      (mul_nonneg (Nat.cast_nonneg n) (Real.exp_nonneg _))] using hd
  have htailScaled : 8 * (n : ℝ) * quenchedTail (paths N) 1 eps < δ / 2 := by
    have hmul := mul_le_mul_of_nonneg_left htail' (Nat.cast_nonneg n)
    have hposC : 0 < 8 * C := mul_pos (by norm_num) hC
    calc
      8 * (n : ℝ) * quenchedTail (paths N) 1 eps ≤
          8 * C * ((n : ℝ) * Real.exp (-c * (n : ℝ))) := by nlinarith
      _ < 8 * C * (δ / (16 * C)) := mul_lt_mul_of_pos_left hExpSmall hposC
      _ = δ / 2 := by field_simp [hC.ne']; ring
  have hscaled : (n : ℝ) * thirdMoment (paths N) 1 < δ := by
    calc
      (n : ℝ) * thirdMoment (paths N) 1 ≤
          (n : ℝ) * (eps * A (paths N) 1 +
            8 * quenchedTail (paths N) 1 eps) :=
        mul_le_mul_of_nonneg_left hthird (Nat.cast_nonneg n)
      _ = eps * ((n : ℝ) * A (paths N) 1) +
          8 * (n : ℝ) * quenchedTail (paths N) 1 eps := by ring
      _ ≤ eps * M + 8 * (n : ℝ) * quenchedTail (paths N) 1 eps := by
        gcongr
      _ < δ := by
        have hepsM : eps * M < δ / 2 := by
          dsimp [eps]
          rw [div_mul_eq_mul_div]
          apply (div_lt_iff₀ (by positivity : 0 < 2 * (M + 1))).2
          nlinarith
        linarith
  rw [Real.dist_eq, sub_zero, abs_of_nonneg
    (mul_nonneg (Nat.cast_nonneg n) (thirdMoment_nonneg (paths N) 1))]
  simpa [n] using hscaled

end SpinGlass.AT
