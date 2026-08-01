import Lemmas.FixedDeviation

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

private theorem quenchedReplicaAverage_one {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ}
    (H : Ω → EnergySpace N) :
    quenchedReplicaAverage H (fun _ : Replicas N n => 1) = 1 := by
  unfold quenchedReplicaAverage
  have hpoint : ∀ ω, replicaGibbsAverage (H ω)
      (fun _ : Replicas N n => 1) = 1 := by
    intro ω
    unfold replicaGibbsAverage
    rw [show (∑ σs : Replicas N n, (∏ a, gibbsWeight (H ω) (σs a)) * 1) =
        ∑ σs : Replicas N n, ∏ a, gibbsWeight (H ω) (σs a) by simp]
    rw [← Fintype.prod_sum]
    simp only [sum_gibbsWeight, Finset.prod_const_one]
  rw [integral_congr_ae (ae_of_all _ hpoint)]
  simp

private theorem overlapSecondMoment_le_four {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    overlapSecondMoment path s ≤ 4 := by
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hpoint : ∀ σs : Replicas N 4,
      centeredOverlap q σs 0 1 ^ 2 ≤ 4 := by
    intro σs
    by_cases hN : 0 < N
    · have habs := abs_centeredOverlap_le_two hN hq σs (0 : Fin 4) 1
      have hsq := mul_self_le_mul_self
        (abs_nonneg (centeredOverlap q σs (0 : Fin 4) 1)) habs
      nlinarith [sq_abs (centeredOverlap q σs (0 : Fin 4) 1)]
    · have hNzero : N = 0 := Nat.eq_zero_of_not_pos hN
      subst N
      simp [centeredOverlap, overlap, configOverlap]
      nlinarith [hq.1, hq.2]
  unfold overlapSecondMoment
  calc
    quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2)
        ≤ quenchedReplicaAverage (fullPathHamiltonian path s)
            (fun _ : Replicas N 4 => 4) :=
      quenchedReplicaAverage_mono hfull hpoint
    _ = 4 := by
      rw [show (fun _ : Replicas N 4 => (4 : ℝ)) = fun σs => 4 * (1 : ℝ) by
        funext σs
        ring]
      rw [quenchedReplicaAverage_const_mul, quenchedReplicaAverage_one]
      ring

private theorem overlapSecondMoment_le_sq_add_tail {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s eps : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (hN : 0 < N)
    (hq : q ∈ Set.Icc (0 : ℝ) 1) (heps : 0 ≤ eps) :
    overlapSecondMoment path s ≤ eps ^ 2 + 4 * quenchedTail path s eps := by
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hpoint : ∀ σs : Replicas N 4,
      centeredOverlap q σs 0 1 ^ 2 ≤
        eps ^ 2 + 4 * (if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0) := by
    intro σs
    let x := centeredOverlap q σs (0 : Fin 4) 1
    have hx := abs_centeredOverlap_le_two hN hq σs (0 : Fin 4) 1
    change x ^ 2 ≤ eps ^ 2 + 4 * (if eps ≤ |x| then 1 else 0)
    by_cases hlarge : eps ≤ |x|
    · simp only [if_pos hlarge, mul_one]
      nlinarith [sq_abs x, sq_nonneg eps]
    · simp only [if_neg hlarge, mul_zero, add_zero]
      have hxeps : |x| ≤ eps := le_of_not_ge hlarge
      simpa [pow_two] using mul_self_le_mul_self (abs_nonneg x) hxeps
  unfold overlapSecondMoment quenchedTail
  calc
    quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2)
        ≤ quenchedReplicaAverage (fullPathHamiltonian path s)
            (fun σs : Replicas N 4 => eps ^ 2 +
              4 * (if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0)) :=
      quenchedReplicaAverage_mono hfull hpoint
    _ = quenchedReplicaAverage (fullPathHamiltonian path s)
          (fun _ : Replicas N 4 => eps ^ 2) +
        quenchedReplicaAverage (fullPathHamiltonian path s)
          (fun σs : Replicas N 4 =>
            4 * (if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0)) := by
      rw [quenchedReplicaAverage_add hfull]
    _ = eps ^ 2 + 4 * quenchedReplicaAverage (fullPathHamiltonian path s)
          (fun σs : Replicas N 4 =>
            if eps ≤ |centeredOverlap q σs 0 1| then 1 else 0) := by
      rw [show (fun _ : Replicas N 4 => eps ^ 2) =
          fun σs => eps ^ 2 * (1 : ℝ) by funext σs; ring]
      rw [quenchedReplicaAverage_const_mul, quenchedReplicaAverage_one]
      rw [quenchedReplicaAverage_const_mul]
      ring

/-- Sublinear coupled-pressure and Gronwall estimate.  Its finite-dimensional
Gaussian maximum and concentration proof is isolated here. -/
theorem coupledPressure_sublinear {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ epsN : ℕ → ℝ, Tendsto epsN atTop (nhds 0) ∧ ∀ {N : ℕ}
      {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      overlapSecondMoment path s ≤ epsN N := by
  let momentSet : ℕ → Set ℝ := fun N =>
    {x | x = 0 ∨ ∃ (β h q s : ℝ)
        (path : RSSmartPathDisorder Ω N β h q),
        (β, h) ∈ K ∧ q = rsQ β h ∧ s ∈ Set.Icc (0 : ℝ) 1 ∧
          x = overlapSecondMoment path s}
  let epsN : ℕ → ℝ := fun N => sSup (momentSet N)
  have hnonempty : ∀ N, (momentSet N).Nonempty := by
    intro N
    exact ⟨0, Or.inl rfl⟩
  have hbdd : ∀ N, BddAbove (momentSet N) := by
    intro N
    refine ⟨4, ?_⟩
    intro x hx
    rcases hx with rfl | ⟨β, h, q, s, path, hp, hq, hs, rfl⟩
    · norm_num
    · exact overlapSecondMoment_le_four path (by
        subst q
        exact rsQ_mem_Icc β h)
  have heps_nonneg : ∀ N, 0 ≤ epsN N := by
    intro N
    exact le_csSup (hbdd N) (Or.inl rfl)
  have heps_tendsto : Tendsto epsN atTop (nhds 0) := by
    apply Metric.tendsto_atTop.2
    intro ε hε
    let δ : ℝ := Real.sqrt (ε / 4)
    have hδ : 0 < δ := by
      dsimp [δ]
      positivity
    obtain ⟨c, C, hc, hC, htail⟩ := fixedDeviation (Ω := Ω) data δ hδ
    have hscale : Tendsto (fun N : ℕ => c * (N : ℝ)) atTop atTop :=
      tendsto_natCast_atTop_atTop.const_mul_atTop hc
    have hdecay : Tendsto (fun N : ℕ => C * Real.exp (-c * (N : ℝ)))
        atTop (nhds 0) := by
      have hexp := Real.tendsto_exp_neg_atTop_nhds_zero.comp hscale
      simpa using hexp.const_mul C
    obtain ⟨N0, hN0⟩ :=
      (Metric.tendsto_atTop.1 hdecay) (ε / 16) (by positivity)
    refine ⟨max 1 N0, ?_⟩
    intro N hN
    have hNpos : 0 < N := lt_of_lt_of_le Nat.zero_lt_one
      (le_trans (Nat.le_max_left 1 N0) hN)
    have htail_small : C * Real.exp (-c * (N : ℝ)) < ε / 16 := by
      have hdist := hN0 N (le_trans (Nat.le_max_right 1 N0) hN)
      rw [Real.dist_eq, sub_zero, abs_of_nonneg
        (mul_nonneg hC.le (Real.exp_pos _).le)] at hdist
      exact hdist
    have hsup : epsN N ≤ ε / 2 := by
      apply csSup_le (hnonempty N)
      intro x hx
      rcases hx with rfl | ⟨β, h, q, s, path, hp, hq, hs, rfl⟩
      · linarith
      · have hqIcc : q ∈ Set.Icc (0 : ℝ) 1 := by
          subst q
          exact rsQ_mem_Icc β h
        have hmoment := overlapSecondMoment_le_sq_add_tail
          (s := s) path hNpos hqIcc hδ.le
        have htail_bound := htail path hp hq hs
        have hδsq : δ ^ 2 = ε / 4 := by
          dsimp [δ]
          rw [Real.sq_sqrt (by positivity)]
        calc
          overlapSecondMoment path s
              ≤ δ ^ 2 + 4 * quenchedTail path s δ := hmoment
          _ ≤ δ ^ 2 + 4 * (C * Real.exp (-c * (N : ℝ))) := by gcongr
          _ ≤ ε / 2 := by rw [hδsq]; linarith
    rw [Real.dist_eq, sub_zero, abs_of_nonneg (heps_nonneg N)]
    exact lt_of_le_of_lt hsup (by linarith)
  refine ⟨epsN, heps_tendsto, ?_⟩
  intro N β h q s path hp hq hs
  apply le_csSup (hbdd N)
  exact Or.inr ⟨β, h, q, s, path, hp, hq, hs, rfl⟩

theorem preliminary_overlap_bound {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ epsN : ℕ → ℝ, Tendsto epsN atTop (nhds 0) ∧ ∀ {N : ℕ}
      {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q),
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      overlapSecondMoment path s ≤ epsN N := by
  -- Proof route: this is only the public name for `coupledPressure_sublinear`.
  exact coupledPressure_sublinear data

end SpinGlass.AT
