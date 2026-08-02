import Lemmas.GT.Coercivity

open MeasureTheory ProbabilityTheory Filter

set_option autoImplicit false

namespace SpinGlass.AT

universe u

noncomputable def quadraticCoupledPartition {N : ℕ}
    (H : EnergySpace N) (q lam : ℝ) : ℝ :=
  ∑ p : Config N × Config N,
    Real.exp (H p.1 + H p.2 +
      lam * (N : ℝ) / 2 * (configOverlap N p.1 p.2 - q) ^ 2)

noncomputable def quadraticCoupledPressure {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s lam : ℝ) : ℝ :=
  (1 / (2 * (N : ℝ))) * ∫ ω,
    Real.log (quadraticCoupledPartition (fullPathHamiltonian path s ω) q lam)
      ∂(volume : Measure Ω)

noncomputable def normalizedCouplingExcess {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s lam : ℝ) : ℝ :=
  quadraticCoupledPressure path s lam - pathFreeEnergy path s

noncomputable def rsFreeEnergyGap {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) : ℝ :=
  rsPathValue β h q s - pathFreeEnergy path s

theorem quadraticCoupledPartition_pos {N : ℕ}
    (H : EnergySpace N) (q lam : ℝ) :
    0 < quadraticCoupledPartition H q lam := by
  unfold quadraticCoupledPartition
  exact Finset.sum_pos
    (fun p _ => Real.exp_pos
      (H p.1 + H p.2 + lam * (N : ℝ) / 2 *
        (configOverlap N p.1 p.2 - q) ^ 2))
    Finset.univ_nonempty

/-- The coupled partition function is the finite log-sum-exp envelope of
the constrained two-replica partition functions. -/
theorem quadraticCoupledPartition_eq_sum_constrained {N : ℕ}
    (H : EnergySpace N) (q lam : ℝ) :
    quadraticCoupledPartition H q lam =
      ∑ v ∈ attainableOverlaps N,
        Real.exp (lam * (N : ℝ) / 2 * (v - q) ^ 2) *
          constrainedPartition H v := by
  classical
  unfold quadraticCoupledPartition constrainedPartition
  symm
  calc
    (∑ v ∈ attainableOverlaps N,
        Real.exp (lam * (N : ℝ) / 2 * (v - q) ^ 2) *
          ∑ p : Config N × Config N,
            if configOverlap N p.1 p.2 = v then
              Real.exp (H p.1 + H p.2) else 0) =
        ∑ v ∈ attainableOverlaps N, ∑ p : Config N × Config N,
          Real.exp (lam * (N : ℝ) / 2 * (v - q) ^ 2) *
            (if configOverlap N p.1 p.2 = v then
              Real.exp (H p.1 + H p.2) else 0) := by
          simp_rw [Finset.mul_sum]
    _ = ∑ p : Config N × Config N, ∑ v ∈ attainableOverlaps N,
          Real.exp (lam * (N : ℝ) / 2 * (v - q) ^ 2) *
            (if configOverlap N p.1 p.2 = v then
              Real.exp (H p.1 + H p.2) else 0) := by
          rw [Finset.sum_comm]
    _ = ∑ p : Config N × Config N,
          Real.exp (H p.1 + H p.2 +
            lam * (N : ℝ) / 2 * (configOverlap N p.1 p.2 - q) ^ 2) := by
      apply Finset.sum_congr rfl
      intro p _
      rw [Finset.sum_eq_single (configOverlap N p.1 p.2)]
      · simp [overlap_mem_attainableOverlaps, ← Real.exp_add, add_comm]
      · intro v _ hne
        simp [hne.symm]
      · exact fun hnot => (hnot (overlap_mem_attainableOverlaps p.1 p.2)).elim

/-- There are at most `N + 1` attainable normalized overlaps. -/
theorem card_attainableOverlaps_le (N : ℕ) :
    (attainableOverlaps N).card ≤ N + 1 := by
  classical
  by_cases hNzero : N = 0
  · subst N
    have hzero : attainableOverlaps 0 = {0} := by
      ext v
      simp [attainableOverlaps, configOverlap, eq_comm]
    rw [hzero]
    norm_num
  let mismatchCount : Config N × Config N → ℕ := fun p =>
    (Finset.univ.filter fun i => p.1 i ≠ p.2 i).card
  have hoverlap_of_count (p : Config N × Config N) :
      configOverlap N p.1 p.2 =
        1 - 2 * (mismatchCount p : ℝ) / (N : ℝ) := by
    have hNreal : (N : ℝ) ≠ 0 := by exact_mod_cast hNzero
    have hterm (i : Fin N) :
        spin p.1 i * spin p.2 i =
          1 - 2 * (if p.1 i ≠ p.2 i then (1 : ℝ) else 0) := by
      simp only [spin]
      cases h1 : p.1 i <;> cases h2 : p.2 i <;> simp_all <;> norm_num
    unfold configOverlap
    rw [Finset.sum_congr rfl (fun i _ => hterm i)]
    simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, Nat.cast_ofNat, one_mul,
      ← Finset.mul_sum, Finset.sum_boole]
    dsimp [mismatchCount]
    field_simp [hNreal]
  let countImage : Finset ℕ := Finset.univ.image mismatchCount
  have hcardImage : countImage.card ≤ N + 1 := by
    calc
      countImage.card ≤ (Finset.range (N + 1)).card := by
        apply Finset.card_le_card
        intro k hk
        simp only [countImage, Finset.mem_image, Finset.mem_univ,
          true_and] at hk
        obtain ⟨p, rfl⟩ := hk
        simp only [Finset.mem_range]
        have := Finset.card_le_card
          (Finset.filter_subset (fun i => p.1 i ≠ p.2 i) Finset.univ)
        simpa [mismatchCount] using Nat.lt_succ_of_le this
      _ = N + 1 := by simp
  let overlapOfCount : ℕ → ℝ := fun k => 1 - 2 * (k : ℝ) / (N : ℝ)
  have hset : attainableOverlaps N = countImage.image overlapOfCount := by
    ext v
    constructor
    · intro hv
      simp only [attainableOverlaps, Finset.mem_image, Finset.mem_univ,
        true_and] at hv
      obtain ⟨p, rfl⟩ := hv
      rw [hoverlap_of_count p]
      simp only [Finset.mem_image]
      refine ⟨mismatchCount p, ?_, rfl⟩
      simp [countImage]
    · intro hv
      simp only [Finset.mem_image] at hv
      obtain ⟨k, hk, rfl⟩ := hv
      simp only [countImage, Finset.mem_image, Finset.mem_univ,
        true_and] at hk
      obtain ⟨p, rfl⟩ := hk
      change (1 - 2 * (mismatchCount p : ℝ) / (N : ℝ)) ∈
        attainableOverlaps N
      rw [← hoverlap_of_count p]
      exact overlap_mem_attainableOverlaps p.1 p.2
  rw [hset]
  exact (Finset.card_image_le.trans hcardImage)

end SpinGlass.AT
