import Lemmas.Replicon

open Topology

set_option autoImplicit false

namespace SpinGlass.AT

theorem continuous_rsQ : Continuous (fun p : ℝ × ℝ => rsQ p.1 p.2) := by
  -- Paper route after equation (q): prove continuity on the intended open set
  -- `β > 0, h > 0` by sequential compactness.  Any sequence of fixed points in
  -- `[0,1]` has a convergent subsequence; dominated convergence passes its
  -- limit through the bounded `tanh^2` integral, and fixed-point uniqueness
  -- identifies the limit.  A subsequence contradiction proves convergence of
  -- the full sequence.
  --
  -- Statement repair required: the paper only defines the canonical unique
  -- solution for positive `β,h`, while this theorem asserts global continuity
  -- on all of `ℝ × ℝ` for the `sInf` fallback definition.  Restrict the theorem
  -- to `{p | 0 < p.1 ∧ 0 < p.2}` or prove existence and uniqueness globally.
  sorry

theorem continuous_rsR : Continuous (fun p : ℝ × ℝ => rsR p.1 p.2) := by
  -- After restricting to the positive-parameter set, compose `continuous_rsQ`
  -- with continuity of `sqrt`, multiplication, addition, `tanh`, and fourth
  -- power.  The integrand is bounded by one, so a parameterized dominated
  -- convergence theorem gives continuity of the Gaussian integral defining
  -- `rsR`.  A reusable lemma for continuous parameter-dependent integrals
  -- against `gaussianReal 0 1` is the main Lean prerequisite.
  sorry

theorem continuous_rsA : Continuous (fun p : ℝ × ℝ => rsA p.1 p.2) := by
  -- Proof route: unfold `rsA` and combine the restricted versions of
  -- `continuous_rsQ` and `continuous_rsR` with `Continuous.const_sub`, scalar
  -- multiplication, and addition.  No new integration argument is needed.
  sorry

theorem continuous_atParameter :
    Continuous (fun p : ℝ × ℝ => atParameter p.1 p.2) := by
  -- Proof route: unfold `atParameter`; multiply continuity of `p.1 ^ 2` by
  -- `continuous_rsA`.  As above, the statement should use continuity within
  -- the positive-parameter domain unless the global `rsQ` issue is resolved.
  sorry

theorem uniformATData_of_compact_strictAT (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K) (hβ : ∀ p ∈ K, 0 < p.1)
    (hh : ∀ p ∈ K, 0 < p.2)
    (hAT : ∀ p ∈ K, atParameter p.1 p.2 < 1) :
    Nonempty (UniformATData K) := by
  classical
  by_cases hK : K.Nonempty
  · obtain ⟨pβ, hpβ, hβmax⟩ :=
      hKcompact.exists_isMaxOn hK continuous_fst.continuousOn
    obtain ⟨pq, hpq, hqmin⟩ :=
      hKcompact.exists_isMinOn hK continuous_rsQ.continuousOn
    obtain ⟨pg, hpg, hgapmin⟩ := hKcompact.exists_isMinOn hK
      ((continuous_const : Continuous (fun _ : ℝ × ℝ => (1 : ℝ))).sub
        continuous_atParameter).continuousOn
    refine ⟨{
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
