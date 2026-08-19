import Lemmas.Replicon

open Topology

set_option autoImplicit false

namespace SpinGlass.AT

/-- Continuity information on the compact parameter set. Global continuity is
not asserted because the canonical fixed-point selection is used only in the
positive-field stability regime. -/
structure RSParameterStability (K : Set (ℝ × ℝ)) : Prop where
  continuous_rsQ : ContinuousOn (fun p : ℝ × ℝ => rsQ p.1 p.2) K
  continuous_rsR : ContinuousOn (fun p : ℝ × ℝ => rsR p.1 p.2) K
  continuous_rsA : ContinuousOn (fun p : ℝ × ℝ => rsA p.1 p.2) K
  continuous_atParameter :
    ContinuousOn (fun p : ℝ × ℝ => atParameter p.1 p.2) K

/-- Compactness, positivity, strict AT, and continuity on `K` provide the
uniform numerical constants used by the finite-volume argument. -/
theorem uniformATData_of_compact_strictAT (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K) (hβ : ∀ p ∈ K, 0 < p.1)
    (hh : ∀ p ∈ K, 0 < p.2)
    (hstable : RSParameterStability K)
    (hAT : ∀ p ∈ K, atParameter p.1 p.2 < 1) :
    Nonempty (UniformATData K) := by
  classical
  by_cases hK : K.Nonempty
  · obtain ⟨pβ, hpβ, hβmax⟩ :=
      hKcompact.exists_isMaxOn hK continuous_fst.continuousOn
    obtain ⟨pq, hpq, hqmin⟩ :=
      hKcompact.exists_isMinOn hK hstable.continuous_rsQ
    obtain ⟨pg, hpg, hgapmin⟩ := hKcompact.exists_isMinOn hK
      ((continuousOn_const : ContinuousOn (fun _ : ℝ × ℝ => (1 : ℝ)) K).sub
        hstable.continuous_atParameter)
    refine ⟨{
      isCompact := hKcompact
      βmax := pβ.1
      qmin := rsQ pq.1 pq.2
      gap := 1 - atParameter pg.1 pg.2
      βmax_pos := hβ pβ hpβ
      qmin_pos := rsQ_pos (hβ pq hpq) (hh pq hpq)
      gap_pos := sub_pos.mpr (hAT pg hpg)
      β_pos := hβ
      h_pos := hh
      β_bound := ?_
      q_lower := ?_
      strictAT := ?_ }⟩
    · intro p hp
      exact hβmax hp
    · intro p hp
      exact hqmin hp
    · intro p hp
      have hg := hgapmin hp
      dsimp at hg ⊢
      linarith
  · refine ⟨{
      isCompact := hKcompact
      βmax := 1
      qmin := 1
      gap := 1
      βmax_pos := zero_lt_one
      qmin_pos := zero_lt_one
      gap_pos := zero_lt_one
      β_pos := ?_
      h_pos := ?_
      β_bound := ?_
      q_lower := ?_
      strictAT := ?_ }⟩ <;>
      intro p hp <;> exact (hK ⟨p, hp⟩).elim

end SpinGlass.AT
