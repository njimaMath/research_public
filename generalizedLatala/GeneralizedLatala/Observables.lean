import GeneralizedLatala.Basic
import SpinGlass.Replica.Replicas
import SpinGlass.Interpolation.GuerraBound

/-!
# Smart-path observables

Centered overlaps, pressure, tilted observables, and coupled free energies used throughout the proof.

Main declarations:
- `overlapVariance`
- `logQuadraticMoment`
- `coupledFreeEnergy`

Dependencies:
- scalar data, replica calculus, and the Guerra covariance algebra

This file corresponds to the relevant part of `blueprint_latala.txt`.
-/

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ENNReal NNReal Topology

set_option maxHeartbeats 800000

namespace SpinGlass
namespace GeneralizedLatala

universe uΩ uι

variable {Ω : Type uΩ} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

/-! ## Smart-path observables -/

variable (N : ℕ) [NeZero N] (β h q : ℝ)
variable (sk : SKDisorder.{uΩ} (Ω := Ω) N β h)
variable (sim : SimpleDisorder.{uΩ} (Ω := Ω) N β q)

lemma measurable_H_t_updated (t : ℝ) :
    Measurable (H_t (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) t) := by
  have hU : Measurable (fun w => Real.sqrt t • sk.U w) :=
    sk.hU.repr_measurable.const_smul (Real.sqrt t)
  have hV : Measurable (fun w => Real.sqrt (1 - t) • sim.V w) :=
    sim.hV.repr_measurable.const_smul (Real.sqrt (1 - t))
  have hfield : Measurable (fun _w : Ω => H_field (N := N) (h := h)) := measurable_const
  change Measurable (((fun w => Real.sqrt t • sk.U w) +
    fun w => Real.sqrt (1 - t) • sim.V w) +
    fun _w : Ω => H_field (N := N) (h := h))
  exact (hU.add hV).add hfield

lemma measurable_dH_t_updated (t : ℝ) :
    Measurable (fun w => dH_t (N := N) (β := β) (h := h) (q := q)
      (sk := sk) (sim := sim) t w) := by
  have hU : Measurable (fun w => (1 / (2 * Real.sqrt t)) • sk.U w) :=
    sk.hU.repr_measurable.const_smul (1 / (2 * Real.sqrt t))
  have hV : Measurable (fun w => (1 / (2 * Real.sqrt (1 - t))) • sim.V w) :=
    sim.hV.repr_measurable.const_smul (1 / (2 * Real.sqrt (1 - t)))
  change Measurable ((fun w => (1 / (2 * Real.sqrt t)) • sk.U w) +
    -(fun w => (1 / (2 * Real.sqrt (1 - t))) • sim.V w))
  exact hU.add hV.neg

/-- Centered overlap `Q_ab = R_ab - q`. -/
noncomputable def centeredOverlap {n : ℕ} (a b : Fin n) : ReplicaFun N n :=
  fun σs => overlap N (σs a) (σs b) - q

/-- The square of the centered overlap of the first two replicas. -/
noncomputable def centeredOverlapSq : ReplicaFun N 2 :=
  fun σs => (overlap N (σs 0) (σs 1) - q) ^ 2

lemma overlap_mem_Icc (hN : 0 < N) (σ τ : Config N) :
    overlap N σ τ ∈ Set.Icc (-1 : ℝ) 1 := by
  have hterm (i : Fin N) :
      -1 ≤ spin N σ i * spin N τ i ∧ spin N σ i * spin N τ i ≤ 1 := by
    simp only [spin]
    split <;> split <;> norm_num
  have hlo : -(N : ℝ) ≤ ∑ i : Fin N, spin N σ i * spin N τ i := by
    simpa using Finset.sum_le_sum (s := Finset.univ) (fun i _ => (hterm i).1)
  have hhi : ∑ i : Fin N, spin N σ i * spin N τ i ≤ (N : ℝ) := by
    simpa using Finset.sum_le_sum (s := Finset.univ) (fun i _ => (hterm i).2)
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have hNne : (N : ℝ) ≠ 0 := ne_of_gt hNr
  constructor
  · rw [overlap]
    calc
      (-1 : ℝ) = (1 / (N : ℝ)) * (-(N : ℝ)) := by field_simp
      _ ≤ (1 / (N : ℝ)) * ∑ i, spin N σ i * spin N τ i :=
        mul_le_mul_of_nonneg_left hlo (one_div_nonneg.mpr hNr.le)
  · rw [overlap]
    calc
      (1 / (N : ℝ)) * ∑ i, spin N σ i * spin N τ i
          ≤ (1 / (N : ℝ)) * (N : ℝ) :=
        mul_le_mul_of_nonneg_left hhi (one_div_nonneg.mpr hNr.le)
      _ = 1 := by field_simp

/-- The Bregman remainder `Δq(R₁₂)` as a two-replica observable. -/
noncomputable def bregmanOverlap : ReplicaFun N 2 :=
  fun σs => bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1))

/-- Annealed second moment `ν_t[Q_12²]`. -/
noncomputable def overlapVariance (t : ℝ) : ℝ :=
  nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
    2 t (centeredOverlapSq N q)

/-- Annealed Gibbs expectation `ν_t[Δq(R₁₂)]`. -/
noncomputable def bregmanAverage (t : ℝ) : ℝ :=
  nu (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
    2 t (bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk))

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
lemma overlapVariance_nonneg (t : ℝ) :
    0 ≤ overlapVariance
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  classical
  apply integral_nonneg
  intro ω
  apply Finset.sum_nonneg
  intro σs _
  apply mul_nonneg (sq_nonneg _)
  apply Finset.prod_nonneg
  intro l _
  exact gibbs_pmf_nonneg
    (N := N)
    (H := H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    (σ := σs l)

/-- The interpolated pressure `N⁻¹ E log Z_{N,t}`. -/
noncomputable def interpolatedPressure (t : ℝ) : ℝ :=
  ∫ ω, free_energy_density (N := N)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω) ∂ℙ

/-- The logarithmic quadratic moment
`E log ⟨exp(λ N Q_12²)⟩_t`. -/
noncomputable def logQuadraticMoment (t coupling : ℝ) : ℝ :=
  ∫ ω, Real.log
    (gibbs_average_n
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      2 t (fun σs => Real.exp (coupling * (N : ℝ) * (centeredOverlapSq N q σs))) ω) ∂ℙ

/-- The logarithmic quadratic moment in the physical coupling `Λ`.

The exponential appearing in this quantity is `exp ((Λ / 2) * N * Q₁₂²)`.  Keeping this
wrapper separate from `logQuadraticMoment` prevents the physical coupling from being confused
with the coefficient appearing directly in the exponential. -/
noncomputable def physicalLogQuadraticMoment (t Λ : ℝ) : ℝ :=
  logQuadraticMoment
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t (Λ / 2)

/-! ### Positivity and explicit tilted observables -/

/-- The two-replica tilted partition function at fixed disorder `H`.

Here `coupling` is the coefficient in `exp (coupling * N * Q₁₂²)`, not the physical coupling
`Λ` used by `coupledFreeEnergy`. -/
noncomputable def tiltedReplicaPartitionDet (H : EnergySpace N) (coupling : ℝ) : ℝ :=
  gibbs_average_n_det (N := N) (n := 2) H
    (fun σs => Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs))

/-- The tilted partition function evaluated along the smart path. -/
noncomputable def tiltedReplicaPartition (t coupling : ℝ) (ω : Ω) : ℝ :=
  tiltedReplicaPartitionDet (N := N) (q := q)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- Strict positivity of the finite-disorder tilted denominator. -/
lemma tiltedReplicaPartitionDet_pos (H : EnergySpace N) (coupling : ℝ) :
    0 < tiltedReplicaPartitionDet (N := N) (q := q) H coupling := by
  classical
  unfold tiltedReplicaPartitionDet gibbs_average_n_det
  apply Finset.sum_pos
  · intro σs _
    exact mul_pos (Real.exp_pos _)
      (Finset.prod_pos fun l _ => gibbs_pmf_pos (N := N) (H := H) (σ := σs l))
  · exact Finset.univ_nonempty

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- For nonnegative exponential coupling, the tilted partition function is at least one. -/
lemma tiltedReplicaPartitionDet_one_le
    (H : EnergySpace N) {coupling : ℝ} (hcoupling : 0 ≤ coupling) :
    1 ≤ tiltedReplicaPartitionDet (N := N) (q := q) H coupling := by
  classical
  unfold tiltedReplicaPartitionDet gibbs_average_n_det
  rw [← sum_prod_gibbs_pmf_eq_one (N := N) (n := 2) (H := H)]
  apply Finset.sum_le_sum
  intro σs _
  have hexp : 1 ≤ Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) :=
    Real.one_le_exp
      (mul_nonneg (mul_nonneg hcoupling (Nat.cast_nonneg N)) (sq_nonneg _))
  have hweight : 0 ≤ ∏ l, gibbs_pmf N H (σs l) :=
    Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg (N := N) (H := H) (σ := σs l)
  simpa only [one_mul] using mul_le_mul_of_nonneg_right hexp hweight

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- Strict positivity of the tilted denominator along the smart path. -/
lemma tiltedReplicaPartition_pos (t coupling : ℝ) (ω : Ω) :
    0 < tiltedReplicaPartition
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      t coupling ω :=
  tiltedReplicaPartitionDet_pos
    (N := N) (q := q)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling

/-- The centered-overlap square under the quadratic tilt at fixed disorder. -/
noncomputable def tiltedCenteredOverlapSqDet
    (H : EnergySpace N) (coupling : ℝ) : ℝ :=
  gibbs_average_n_det (N := N) (n := 2) H
      (fun σs => centeredOverlapSq N q σs *
        Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs)) /
    tiltedReplicaPartitionDet (N := N) (q := q) H coupling

/-- Annealed centered-overlap square under the quadratic two-replica tilt. -/
noncomputable def tiltedCenteredOverlapSq (t coupling : ℝ) : ℝ :=
  ∫ ω, tiltedCenteredOverlapSqDet (N := N) (q := q)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling ∂ℙ

/-- The Bregman remainder under the quadratic two-replica tilt. -/
noncomputable def tiltedBregmanDet (H : EnergySpace N) (coupling : ℝ) : ℝ :=
  gibbs_average_n_det (N := N) (n := 2) H
      (fun σs => bregmanOverlap (N := N) (β := β) (h := h) (q := q) (sk := sk) σs *
        Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs)) /
    tiltedReplicaPartitionDet (N := N) (q := q) H coupling

noncomputable def tiltedBregman (t coupling : ℝ) : ℝ :=
  ∫ ω, tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling ∂ℙ

/-- The average of the four cross-pair centered-overlap squares for replicas grouped as
`(1,2)` and `(3,4)`. -/
noncomputable def crossPairCenteredOverlapSq : ReplicaFun N 4 :=
  fun σs =>
    ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (2 : Fin 4) σs) ^ 2 +
      (centeredOverlap (N := N) (q := q) (0 : Fin 4) (3 : Fin 4) σs) ^ 2 +
      (centeredOverlap (N := N) (q := q) (1 : Fin 4) (2 : Fin 4) σs) ^ 2 +
      (centeredOverlap (N := N) (q := q) (1 : Fin 4) (3 : Fin 4) σs) ^ 2) / 4

/-- The average Bregman remainder of the four cross pairs. -/
noncomputable def crossPairBregman : ReplicaFun N 4 :=
  fun σs =>
    (bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 2)) +
      bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 3)) +
      bregmanRemainder sk.ξ β q (overlap N (σs 1) (σs 2)) +
      bregmanRemainder sk.ξ β q (overlap N (σs 1) (σs 3))) / 4

/-- The four-replica cross moment at fixed disorder.  The pairs `(1,2)` and `(3,4)` receive
independent copies of the same quadratic tilt. -/
noncomputable def coupledCrossMomentDet (H : EnergySpace N) (coupling : ℝ) : ℝ :=
  gibbs_average_n_det (N := N) (n := 4) H
      (fun σs => crossPairCenteredOverlapSq (N := N) (q := q) σs *
        Real.exp (coupling * (N : ℝ) *
          ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (1 : Fin 4) σs) ^ 2 +
            (centeredOverlap (N := N) (q := q) (2 : Fin 4) (3 : Fin 4) σs) ^ 2))) /
    (tiltedReplicaPartitionDet (N := N) (q := q) H coupling) ^ 2

/-- Annealed four-replica cross moment generated by coupled Gaussian integration by parts. -/
noncomputable def coupledCrossMoment (t coupling : ℝ) : ℝ :=
  ∫ ω, coupledCrossMomentDet (N := N) (q := q)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling ∂ℙ

/-- The cross-pair Bregman remainder generated by coupled Gaussian integration by parts. -/
noncomputable def coupledCrossBregmanDet (H : EnergySpace N) (coupling : ℝ) : ℝ :=
  gibbs_average_n_det (N := N) (n := 4) H
      (fun σs => crossPairBregman (N := N) (β := β) (h := h) (q := q) (sk := sk) σs *
        Real.exp (coupling * (N : ℝ) *
          ((centeredOverlap (N := N) (q := q) (0 : Fin 4) (1 : Fin 4) σs) ^ 2 +
            (centeredOverlap (N := N) (q := q) (2 : Fin 4) (3 : Fin 4) σs) ^ 2))) /
    (tiltedReplicaPartitionDet (N := N) (q := q) H coupling) ^ 2

noncomputable def coupledCrossBregman (t coupling : ℝ) : ℝ :=
  ∫ ω, coupledCrossBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk)
    (H_t (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ω)
    coupling ∂ℙ

/-- The four-replica cross moment is nonnegative. -/
lemma coupledCrossMoment_nonneg (t coupling : ℝ) :
    0 ≤ coupledCrossMoment
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      t coupling := by
  classical
  apply integral_nonneg
  intro ω
  unfold coupledCrossMomentDet gibbs_average_n_det
  apply div_nonneg
  · apply Finset.sum_nonneg
    intro σs _
    apply mul_nonneg
    · apply mul_nonneg
      · unfold crossPairCenteredOverlapSq
        positivity
      · exact Real.exp_nonneg _
    · exact Finset.prod_nonneg fun l _ =>
        gibbs_pmf_nonneg
          (N := N)
          (H := H_t (N := N) (β := β) (h := h) (q := q)
            (sk := sk) (sim := sim) t ω)
          (σ := σs l)
  · positivity

lemma coupledCrossBregman_nonneg
    (hN : 0 < N) (hΔ : BregmanBounds sk.ξ β q Γ) (t coupling : ℝ) :
    0 ≤ coupledCrossBregman
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim)
      t coupling := by
  apply integral_nonneg
  intro ω
  unfold coupledCrossBregmanDet gibbs_average_n_det
  apply div_nonneg
  · apply Finset.sum_nonneg
    intro σs _
    apply mul_nonneg
    · apply mul_nonneg
      · unfold crossPairBregman
        have h02 := (hΔ _ (overlap_mem_Icc N hN (σs 0) (σs 2))).1
        have h03 := (hΔ _ (overlap_mem_Icc N hN (σs 0) (σs 3))).1
        have h12 := (hΔ _ (overlap_mem_Icc N hN (σs 1) (σs 2))).1
        have h13 := (hΔ _ (overlap_mem_Icc N hN (σs 1) (σs 3))).1
        positivity
      · exact Real.exp_nonneg _
    · exact Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg
        (N := N) (H := H_t N β h q sk sim t ω) (σ := σs l)
  · positivity

lemma bregmanAverage_le
    (hN : 0 < N) (hΔ : BregmanBounds sk.ξ β q Γ) (t : ℝ) :
    bregmanAverage (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t ≤
      (Γ / 2) * overlapVariance
        (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  rw [bregmanAverage, overlapVariance]
  unfold nu
  rw [← integral_const_mul]
  apply integral_mono
  · exact integrable_gibbs_average_n N β h q sk sim 2 t _
  · exact (integrable_gibbs_average_n N β h q sk sim 2 t _).const_mul (Γ / 2)
  · intro ω
    simp only [gibbs_average_n, gibbs_average_n_det, bregmanOverlap, centeredOverlapSq]
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro σs _
    calc
      bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) *
            ∏ l, gibbs_pmf N (H_t N β h q sk sim t ω) (σs l)
          ≤ ((Γ / 2) * (overlap N (σs 0) (σs 1) - q) ^ 2) *
              ∏ l, gibbs_pmf N (H_t N β h q sk sim t ω) (σs l) :=
        mul_le_mul_of_nonneg_right
          ((hΔ _ (overlap_mem_Icc N hN (σs 0) (σs 1))).2)
          (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg
            (N := N) (H := H_t N β h q sk sim t ω) (σ := σs l))
      _ = Γ / 2 * ((overlap N (σs 0) (σs 1) - q) ^ 2 *
            ∏ l, gibbs_pmf N (H_t N β h q sk sim t ω) (σs l)) := by ring

lemma bregmanAverage_nonneg
    (hN : 0 < N) (hΔ : BregmanBounds sk.ξ β q Γ) (t : ℝ) :
    0 ≤ bregmanAverage
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t := by
  rw [bregmanAverage]
  unfold nu
  apply integral_nonneg
  intro ω
  unfold gibbs_average_n gibbs_average_n_det bregmanOverlap
  apply Finset.sum_nonneg
  intro σs _
  exact mul_nonneg
    ((hΔ _ (overlap_mem_Icc N hN (σs 0) (σs 1))).1)
    (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg
      (N := N) (H := H_t N β h q sk sim t ω) (σ := σs l))

lemma tiltedBregman_le
    (hN : 0 < N) (hΔ : BregmanBounds sk.ξ β q Γ)
    (H : EnergySpace N) (coupling : ℝ) :
    tiltedBregmanDet (N := N) (β := β) (h := h) (q := q) (sk := sk) H coupling ≤
      (Γ / 2) * tiltedCenteredOverlapSqDet (N := N) (q := q) H coupling := by
  unfold tiltedBregmanDet tiltedCenteredOverlapSqDet bregmanOverlap gibbs_average_n_det
  change
    (∑ σs,
        bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) *
          Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) *
          ∏ l, gibbs_pmf N H (σs l)) /
        tiltedReplicaPartitionDet N q H coupling ≤
      (Γ / 2) *
        ((∑ σs,
            centeredOverlapSq N q σs *
              Real.exp (coupling * (N : ℝ) * centeredOverlapSq N q σs) *
              ∏ l, gibbs_pmf N H (σs l)) /
          tiltedReplicaPartitionDet N q H coupling)
  have hZ : 0 < tiltedReplicaPartitionDet N q H coupling :=
    tiltedReplicaPartitionDet_pos N q H coupling
  apply (div_le_iff₀ hZ).2
  simp only [mul_assoc, div_mul_cancel₀ _ hZ.ne']
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro σs _
  let w : ℝ := Real.exp (coupling * ((N : ℝ) * centeredOverlapSq N q σs)) *
    ∏ l, gibbs_pmf N H (σs l)
  have hw : 0 ≤ w := mul_nonneg (Real.exp_nonneg _)
    (Finset.prod_nonneg fun l _ => gibbs_pmf_nonneg
      (N := N) (H := H) (σ := σs l))
  change
    bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) * w ≤
      Γ / 2 * (centeredOverlapSq N q σs * w)
  calc
    bregmanRemainder sk.ξ β q (overlap N (σs 0) (σs 1)) * w
        ≤ ((Γ / 2) * centeredOverlapSq N q σs) * w :=
      mul_le_mul_of_nonneg_right
        ((hΔ _ (overlap_mem_Icc N hN (σs 0) (σs 1))).2) hw
    _ = Γ / 2 * (centeredOverlapSq N q σs * w) := by ring

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- Finite-volume Jensen inequality for an arbitrary replica observable. -/
lemma gibbs_average_n_det_exp_jensen {n : ℕ}
    (H : EnergySpace N) (f : ReplicaFun N n) :
    Real.exp (gibbs_average_n_det (N := N) (n := n) H f) ≤
      gibbs_average_n_det (N := N) (n := n) H (fun σs => Real.exp (f σs)) := by
  classical
  let weight : ReplicaSpace N n → ℝ :=
    fun σs => ∏ l, gibbs_pmf N H (σs l)
  have hweight : ∀ σs ∈ (Finset.univ : Finset (ReplicaSpace N n)), 0 ≤ weight σs := by
    intro σs _
    exact Finset.prod_nonneg fun l _ =>
      gibbs_pmf_nonneg (N := N) (H := H) (σ := σs l)
  have hsum : ∑ σs : ReplicaSpace N n, weight σs = 1 := by
    simpa [weight] using sum_prod_gibbs_pmf_eq_one (N := N) (n := n) H
  have hjensen := convexOn_exp.map_sum_le
    (t := (Finset.univ : Finset (ReplicaSpace N n)))
    (w := weight) (p := f) hweight hsum
    (fun σs _ => Set.mem_univ (f σs))
  simpa [gibbs_average_n_det, weight, smul_eq_mul, mul_comm] using hjensen

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- Logarithmic form of the finite-volume Jensen inequality. -/
lemma gibbs_average_n_det_le_log_exp {n : ℕ}
    (H : EnergySpace N) (f : ReplicaFun N n) :
    gibbs_average_n_det (N := N) (n := n) H f ≤
      Real.log
        (gibbs_average_n_det (N := N) (n := n) H (fun σs => Real.exp (f σs))) := by
  have hjensen := gibbs_average_n_det_exp_jensen (N := N) H f
  calc
    gibbs_average_n_det (N := N) (n := n) H f =
        Real.log (Real.exp (gibbs_average_n_det (N := N) (n := n) H f)) := by
          rw [Real.log_exp]
    _ ≤ Real.log
        (gibbs_average_n_det (N := N) (n := n) H (fun σs => Real.exp (f σs))) :=
      Real.log_le_log (Real.exp_pos _) hjensen

omit [IsProbabilityMeasure (ℙ : Measure Ω)] in
/-- Jensen's inequality specialized to the scaled centered-overlap square. -/
lemma scaled_centeredOverlapSq_le_log_gibbs_exp
    (H : EnergySpace N) (coupling : ℝ) :
    coupling * (N : ℝ) *
        gibbs_average_n_det (N := N) (n := 2) H (centeredOverlapSq N q) ≤
      Real.log
        (gibbs_average_n_det (N := N) (n := 2) H
          (fun σs => Real.exp
            (coupling * (N : ℝ) * centeredOverlapSq N q σs))) := by
  have hjensen := gibbs_average_n_det_le_log_exp (N := N) H
    (fun σs : ReplicaSpace N 2 =>
      coupling * (N : ℝ) * centeredOverlapSq N q σs)
  simpa only [gibbs_average_n_det, Finset.mul_sum, mul_assoc] using hjensen

/-- Excess coupled free energy with coupling `(Λ N / 2) Q_12²`.

Adding this quantity to `interpolatedPressure` gives the two-replica coupled free energy from
the blueprint, normalized by `2N`.
-/
noncomputable def coupledExcess (t Λ : ℝ) : ℝ :=
  (1 / (2 * (N : ℝ))) * physicalLogQuadraticMoment
    (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t Λ

/-- Fixed-disorder normalized two-replica free energy with physical coupling `Λ`. -/
noncomputable def coupledFreeEnergyDet (H : EnergySpace N) (Λ : ℝ) : ℝ :=
  free_energy_density (N := N) H +
    (1 / (2 * (N : ℝ))) * Real.log
      (tiltedReplicaPartitionDet (N := N) (q := q) H (Λ / 2))

/-- The normalized coupled two-replica free energy. -/
noncomputable def coupledFreeEnergy (t Λ : ℝ) : ℝ :=
  interpolatedPressure
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t +
    coupledExcess
      (N := N) (β := β) (h := h) (q := q) (sk := sk) (sim := sim) t Λ


end GeneralizedLatala
end SpinGlass
