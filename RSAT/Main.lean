import Lemmas.MainResult

/-!
# Quantitative strict-AT theorem

This entry module states the three quantitative conclusions explicitly.
The public theorem assumes only that `K` is compact and that every point of
`K` has positive inverse temperature, positive external field, and satisfies
the strict AT inequality.  The uniform constants required by the analytic
proof are derived from compactness.
-/

open MeasureTheory SpinGlass.AT

set_option autoImplicit false

universe u

structure QuantitativeAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q, N * A path s ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = rsQ β h →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      0 ≤ rsFreeEnergy β h - skFreeEnergy path ∧
      rsFreeEnergy β h - skFreeEnergy path ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      |N * (A path s - 2 * B path s + C path s) -
        rsA β h / (1 - s * atParameter β h)| < eps

/-- Quantitative strict-AT theorem on a compact subset of the positive-field
strict AT region.  No uniform `βmax`, `qmin`, or AT gap is assumed: these are
derived from compactness and continuity. -/
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictATRegion) :
    QuantitativeAT (Ω := Ω) K := by
  by_cases hKne : K.Nonempty
  · obtain ⟨pβ, hpβ, hβmax⟩ :=
      hKcompact.exists_isMaxOn hKne
        (continuousOn_fst : ContinuousOn (fun p : ℝ × ℝ => p.1) K)

    have hqcont : ContinuousOn (fun p : ℝ × ℝ => rsQ p.1 p.2) K :=
      (continuousOn_rsParameters_of_subset_strictATRegion hKsub).1
    obtain ⟨pq, hpq, hqmin⟩ :=
      hKcompact.exists_isMinOn hKne hqcont

    obtain ⟨gap, hgap_pos, hgap_lower⟩ :=
      exists_uniform_at_gap_on_compact hKcompact hKne hKsub

    let data : UniformATData K :=
      { isCompact := hKcompact
        βmax := pβ.1
        qmin := rsQ pq.1 pq.2
        gap := gap
        βmax_pos := (hKsub hpβ).1
        qmin_pos := rsQ_pos (hKsub hpq).1 (hKsub hpq).2.1
        gap_pos := hgap_pos
        β_pos := by
          intro p hp
          exact (hKsub hp).1
        h_pos := by
          intro p hp
          exact (hKsub hp).2.1
        β_bound := by
          intro p hp
          exact hβmax hp
        q_lower := by
          intro p hp
          exact hqmin hp
        strictAT := by
          intro p hp
          have hgap := hgap_lower p hp
          linarith }

    have result : SpinGlass.AT.QuantitativeATConclusion (Ω := Ω) K :=
      SpinGlass.AT.quantitative_strictAT K data
    exact
      { secondMoment := result.secondMoment
        freeEnergy := result.freeEnergy
        replicon := result.replicon }

  · let data : UniformATData K :=
      { isCompact := hKcompact
        βmax := 1
        qmin := 1
        gap := 1
        βmax_pos := by norm_num
        qmin_pos := by norm_num
        gap_pos := by norm_num
        β_pos := by
          intro p hp
          exact (hKsub hp).1
        h_pos := by
          intro p hp
          exact (hKsub hp).2.1
        β_bound := by
          intro p hp
          exact (hKne ⟨p, hp⟩).elim
        q_lower := by
          intro p hp
          exact (hKne ⟨p, hp⟩).elim
        strictAT := by
          intro p hp
          exact (hKne ⟨p, hp⟩).elim }

    have result : SpinGlass.AT.QuantitativeATConclusion (Ω := Ω) K :=
      SpinGlass.AT.quantitative_strictAT K data
    exact
      { secondMoment := result.secondMoment
        freeEnergy := result.freeEnergy
        replicon := result.replicon }
