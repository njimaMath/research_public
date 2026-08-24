import Lemmas.MainResult

/-!
# Quantitative strict Almeida-Thouless theorem

This entry module defines every model-specific object occurring in its public
theorem.  The implementation in `Lemmas` is used only to discharge the proof.
-/

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

universe u

/-! ## Public model setup -/

/-- The canonical replica-symmetric fixed point. -/
noncomputable def canonicalOverlap (β h : ℝ) : ℝ :=
  sInf {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧
    SpinGlass.AT.IsRSFixedPoint β h q}

/-- The fourth local-magnetization moment at the canonical fixed point. -/
noncomputable def canonicalFourthMoment (β h : ℝ) : ℝ :=
  SpinGlass.AT.standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt (canonicalOverlap β h) * z) ^ 4)

/-- The fourth hyperbolic-secant moment. -/
noncomputable def canonicalSechFourthMoment (β h : ℝ) : ℝ :=
  1 - 2 * canonicalOverlap β h + canonicalFourthMoment β h

/-- The Almeida-Thouless stability parameter. -/
noncomputable def stabilityIndex (β h : ℝ) : ℝ :=
  β ^ 2 * canonicalSechFourthMoment β h

/-- Positive-temperature, positive-field parameters below the AT line. -/
def strictStabilityRegion : Set (ℝ × ℝ) :=
  {p | 0 < p.1 ∧ 0 < p.2 ∧ stabilityIndex p.1 p.2 < 1}

/-- An indexed family of spin replicas. -/
abbrev ReplicaFamily (N n : ℕ) := Fin n → SpinGlass.Config N

/-- A real-valued observable of finitely many replicas. -/
abbrev ReplicaObservable (N n : ℕ) := ReplicaFamily N n → ℝ

/-- Product Gibbs expectation of a replica observable. -/
noncomputable def productGibbsExpectation {N n : ℕ}
    (H : SpinGlass.EnergySpace N) (F : ReplicaObservable N n) : ℝ :=
  ∑ σs, (∏ a, SpinGlass.gibbs_pmf N H (σs a)) * F σs

/-- Disorder-averaged product Gibbs expectation. -/
noncomputable def disorderAveragedExpectation
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N)
    (F : ReplicaObservable N n) : ℝ :=
  ∫ ω, productGibbsExpectation (H ω) F ∂(volume : Measure Ω)

/-- The overlap between two selected replicas. -/
noncomputable def selectedReplicaOverlap {N n : ℕ}
    (σs : ReplicaFamily N n) (a b : Fin n) : ℝ :=
  SpinGlass.overlap N (σs a) (σs b)

/-- A selected replica overlap centered at `q`. -/
noncomputable def centeredReplicaOverlap {N n : ℕ} (q : ℝ)
    (σs : ReplicaFamily N n) (a b : Fin n) : ℝ :=
  selectedReplicaOverlap σs a b - q

/-- A coordinate realization of the replica-symmetric smart path.

The coordinates `g ω i j` and `z ω i` are the independent standard Gaussian
variables denoted by $g_{ij}$ and $z_i$ in the reference.  The last two fields
identify the coordinate formulas with the Gaussian energy-space objects used
by the general theorem. -/
structure SmartPath (Ω : Type u) [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (N : ℕ) (β h q : ℝ) where
  g : Ω → Fin N → Fin N → ℝ
  z : Ω → Fin N → ℝ
  g_standardGaussian : ∀ i j,
    Measure.map (fun ω => g ω i j) (volume : Measure Ω) = gaussianReal 0 1
  z_standardGaussian : ∀ i,
    Measure.map (fun ω => z ω i) (volume : Measure Ω) = gaussianReal 0 1
  g_independent : ProbabilityTheory.iIndepFun
    (fun ij : Fin N × Fin N => fun ω => g ω ij.1 ij.2)
    (volume : Measure Ω)
  z_independent : ProbabilityTheory.iIndepFun
    (fun i : Fin N => fun ω => z ω i) (volume : Measure Ω)
  coordinateFamiliesIndependent : ProbabilityTheory.IndepFun g z
    (volume : Measure Ω)
  skDisorder : SpinGlass.SKDisorder (Ω := Ω) N β h
  fieldDisorder : SpinGlass.SimpleDisorder (Ω := Ω) N β q
  independent : ProbabilityTheory.IndepFun skDisorder.U fieldDisorder.V
    (volume : Measure Ω)
  sk_realization : ∀ ω σ,
    skDisorder.U ω σ =
      β / Real.sqrt (2 * (N : ℝ)) *
        ∑ i : Fin N, ∑ j : Fin N,
          g ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j
  field_realization : ∀ ω σ,
    fieldDisorder.V ω σ =
      β * Real.sqrt q * ∑ i : Fin N, z ω i * SpinGlass.spin N σ i

namespace SmartPath

private def toLibrary {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) :
    SpinGlass.AT.RSSmartPathDisorder Ω N β h q :=
  { sk := path.skDisorder
    simple := path.fieldDisorder
    independent := path.independent }

end SmartPath

/-- The specific smart-path Hamiltonian $H_s(σ)$, including the external
field.  By `H_s_apply` below, this is exactly
$\frac{β\sqrt{s}}{\sqrt{2N}}\sum_{i,j}g_{ij}σ_iσ_j
 +\sum_i(h+β\sqrt{1-s}\sqrt q\,z_i)σ_i$. -/
noncomputable def H_s {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q)
    (s : ℝ) (ω : Ω) : SpinGlass.EnergySpace N :=
  Real.sqrt s • path.skDisorder.U ω +
    Real.sqrt (1 - s) • path.fieldDisorder.V ω +
    SpinGlass.magnetic_field_vector N h

theorem H_s_apply {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q)
    (s : ℝ) (ω : Ω) (σ : SpinGlass.Config N) :
    H_s path s ω σ =
      β * Real.sqrt s / Real.sqrt (2 * (N : ℝ)) *
          ∑ i : Fin N, ∑ j : Fin N,
            path.g ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j +
        ∑ i : Fin N,
          (h + β * Real.sqrt (1 - s) * Real.sqrt q * path.z ω i) *
            SpinGlass.spin N σ i := by
  rw [H_s, path.sk_realization, path.field_realization]
  simp [SpinGlass.magnetic_field_vector, SpinGlass.magnetization,
    Finset.mul_sum]
  ring

/-- The quenched free-energy density along the smart path. -/
noncomputable def smartPathFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) (s : ℝ) : ℝ :=
  ∫ ω, SpinGlass.free_energy_density
      (N := N) (H_s path s ω)
    ∂(volume : Measure Ω)

/-- The replica-symmetric free energy at the canonical fixed point. -/
noncomputable def replicaSymmetricFreeEnergy (β h : ℝ) : ℝ :=
  Real.log 2 + SpinGlass.AT.standardGaussianExpectation
    (fun z => Real.log
      (Real.cosh (h + β * Real.sqrt (canonicalOverlap β h) * z))) +
    β ^ 2 / 4 * (1 - canonicalOverlap β h) ^ 2

/-- The finite-volume SK free energy at the endpoint of the smart path. -/
noncomputable def finiteVolumeFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) : ℝ :=
  smartPathFreeEnergy path 1

/-- The second centered-overlap moment. -/
noncomputable def overlapVariance {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 => centeredReplicaOverlap q σs 0 1 ^ 2)

/-- The centered-overlap moment for two pairs sharing one replica. -/
noncomputable def sharedReplicaMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 =>
      centeredReplicaOverlap q σs 0 1 * centeredReplicaOverlap q σs 0 2)

/-- The centered-overlap moment for two disjoint replica pairs. -/
noncomputable def disjointReplicaMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : SmartPath Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 =>
      centeredReplicaOverlap q σs 0 1 * centeredReplicaOverlap q σs 2 3)

/-! ## Quantitative conclusion -/

structure QuantitativeAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : SmartPath Ω N β h q, N * overlapVariance path s ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h →
      ∀ path : SmartPath Ω N β h q,
      0 ≤ replicaSymmetricFreeEnergy β h - finiteVolumeFreeEnergy path ∧
      replicaSymmetricFreeEnergy β h - finiteVolumeFreeEnergy path ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : SmartPath Ω N β h q,
      |N * (overlapVariance path s - 2 * sharedReplicaMoment path s +
          disjointReplicaMoment path s) -
        canonicalSechFourthMoment β h / (1 - s * stabilityIndex β h)| < eps

/-- Quantitative strict-AT theorem on a compact subset of the positive-field
strict stability region. -/
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictStabilityRegion) :
    QuantitativeAT (Ω := Ω) K := by
  have hKsub' : K ⊆ SpinGlass.AT.strictATRegion := by
    intro p hp
    simpa [strictStabilityRegion, stabilityIndex,
      canonicalSechFourthMoment, canonicalFourthMoment, canonicalOverlap,
      SpinGlass.AT.strictATRegion, SpinGlass.AT.atParameter,
      SpinGlass.AT.rsA, SpinGlass.AT.rsR, SpinGlass.AT.rsQ] using hKsub hp
  have result : SpinGlass.AT.QuantitativeATConclusion (Ω := Ω) K := by
    by_cases hKne : K.Nonempty
    · obtain ⟨pβ, hpβ, hβmax⟩ :=
        hKcompact.exists_isMaxOn hKne
          (continuousOn_fst : ContinuousOn (fun p : ℝ × ℝ => p.1) K)
      have hqcont : ContinuousOn
          (fun p : ℝ × ℝ => SpinGlass.AT.rsQ p.1 p.2) K :=
        (SpinGlass.AT.continuousOn_rsParameters_of_subset_strictATRegion
          hKsub').1
      obtain ⟨pq, hpq, hqmin⟩ :=
        hKcompact.exists_isMinOn hKne hqcont
      obtain ⟨gap, hgap_pos, hgap_lower⟩ :=
        SpinGlass.AT.exists_uniform_at_gap_on_compact
          hKcompact hKne hKsub'
      let data : SpinGlass.AT.UniformATData K :=
        { isCompact := hKcompact
          βmax := pβ.1
          qmin := SpinGlass.AT.rsQ pq.1 pq.2
          gap := gap
          βmax_pos := (hKsub' hpβ).1
          qmin_pos := SpinGlass.AT.rsQ_pos
            (hKsub' hpq).1 (hKsub' hpq).2.1
          gap_pos := hgap_pos
          β_pos := fun p hp => (hKsub' hp).1
          h_pos := fun p hp => (hKsub' hp).2.1
          β_bound := fun p hp => hβmax hp
          q_lower := fun p hp => hqmin hp
          strictAT := by
            intro p hp
            have hgap := hgap_lower p hp
            linarith }
      exact SpinGlass.AT.quantitative_strictAT K data
    · let data : SpinGlass.AT.UniformATData K :=
        { isCompact := hKcompact
          βmax := 1
          qmin := 1
          gap := 1
          βmax_pos := by norm_num
          qmin_pos := by norm_num
          gap_pos := by norm_num
          β_pos := fun p hp => (hKsub' hp).1
          h_pos := fun p hp => (hKsub' hp).2.1
          β_bound := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim
          q_lower := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim
          strictAT := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim }
      exact SpinGlass.AT.quantitative_strictAT K data
  refine { secondMoment := ?_, freeEnergy := ?_, replicon := ?_ }
  · obtain ⟨M, hM, hbound⟩ := result.secondMoment
    refine ⟨M, hM, ?_⟩
    intro N hN β h q s hK hq hs path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' hs path.toLibrary
    simpa [overlapVariance, disorderAveragedExpectation,
      productGibbsExpectation, H_s,
      centeredReplicaOverlap, selectedReplicaOverlap, SmartPath.toLibrary,
      SpinGlass.AT.A, SpinGlass.AT.quenchedReplicaAverage,
      SpinGlass.AT.replicaGibbsAverage, SpinGlass.AT.fullPathHamiltonian,
      SpinGlass.AT.centeredOverlap, SpinGlass.AT.replicaOverlap] using h
  · obtain ⟨M, hM, hbound⟩ := result.freeEnergy
    refine ⟨M, hM, ?_⟩
    intro N hN β h q hK hq path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' path.toLibrary
    simpa [replicaSymmetricFreeEnergy, finiteVolumeFreeEnergy,
      smartPathFreeEnergy, H_s, canonicalOverlap,
      SmartPath.toLibrary, SpinGlass.AT.rsFreeEnergy,
      SpinGlass.AT.rsPathValue, SpinGlass.AT.skFreeEnergy,
      SpinGlass.AT.pathFreeEnergy, SpinGlass.AT.rsQ,
      SpinGlass.AT.fullPathHamiltonian] using h
  · intro eps heps
    obtain ⟨N0, hbound⟩ := result.replicon eps heps
    refine ⟨N0, ?_⟩
    intro N hN β h q s hK hq hs path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' hs path.toLibrary
    simpa [overlapVariance, sharedReplicaMoment, disjointReplicaMoment,
      disorderAveragedExpectation, productGibbsExpectation,
      H_s, centeredReplicaOverlap, selectedReplicaOverlap,
      canonicalSechFourthMoment, canonicalFourthMoment, canonicalOverlap,
      stabilityIndex, SmartPath.toLibrary, SpinGlass.AT.A, SpinGlass.AT.B,
      SpinGlass.AT.C, SpinGlass.AT.quenchedReplicaAverage,
      SpinGlass.AT.replicaGibbsAverage, SpinGlass.AT.fullPathHamiltonian,
      SpinGlass.AT.centeredOverlap, SpinGlass.AT.replicaOverlap,
      SpinGlass.AT.rsA, SpinGlass.AT.rsR, SpinGlass.AT.rsQ,
      SpinGlass.AT.atParameter] using h
