import Lemmas.CLT.Basic
import Lemmas.Cavity.Estimates

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false
set_option maxHeartbeats 1600000

namespace SpinGlass.AT
namespace CLT

universe u

open CavityEstimates

noncomputable def cavityScaledArgument {N n : ℕ} (q : ℝ) (i : Fin N)
    (σs : Replicas N (n + 2)) : ℝ :=
  Real.sqrt (N : ℝ) * cavityOverlapAt q i σs 0 1 +
    q / Real.sqrt (N : ℝ)

noncomputable def weightedCavityMoment6
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (e : ReplicaEdge 6) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 6 =>
    f (cavityScaledArgument q i σs) *
      cavityOverlapAt q i σs e.1.1 e.1.2)

noncomputable def weightedCavityVectorAt
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) : Fin 3 → ℝ :=
  ![weightedCavityMoment6 (s := s) path i v f ee01,
    weightedCavityMoment6 (s := s) path i v f ee02,
    weightedCavityMoment6 (s := s) path i v f ee23]

lemma cavityScaledArgument_relabel6
    {N : ℕ} (q : ℝ) (i : Fin N) (σs : Replicas N 6)
    (p : Equiv.Perm (Fin 6))
    (hp : CavityEstimates.SamePair6 (p 0) (p 1) 0 1) :
    cavityScaledArgument q i (replicaRelabelEquiv p σs) =
      cavityScaledArgument q i σs := by
  unfold cavityScaledArgument
  rw [cavityOverlapAt_relabel6_test]
  rw [cavityOverlapAt_samePair6_test q i σs hp]

lemma weightedCavityMoment6_by_relation
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (e : ReplicaEdge 6) :
    weightedCavityMoment6 (s := s) path i v f e =
      momentCoeff
        (weightedCavityMoment6 (s := s) path i v f ee01)
        (weightedCavityMoment6 (s := s) path i v f ee02)
        (weightedCavityMoment6 (s := s) path i v f ee23)
        (edgeRelation ee01 e) := by
  obtain ⟨p, hpbase, hpedge⟩ := exists_edgeMoment_perm6_test e
  let ce := canonicalEdge6 (edgeRelation ee01 e)
  let F : ReplicaFun N 6 := fun σs =>
    f (cavityScaledArgument q i σs) *
      cavityOverlapAt q i σs ce.1.1 ce.1.2
  have hr := quenchedReplicaAverage_relabel
    (lastSiteHamiltonian (s := s) path i v) F p
  have hpoint : (fun σs => F (replicaRelabelEquiv p σs)) = fun σs =>
      f (cavityScaledArgument q i σs) *
        cavityOverlapAt q i σs e.1.1 e.1.2 := by
    funext σs
    dsimp [F, replicaRelabelEquiv, ce]
    change f (Real.sqrt (N : ℝ) * cavityOverlapAt q i σs (p 0) (p 1) +
        q / Real.sqrt (N : ℝ)) *
        cavityOverlapAt q i σs (p ce.1.1) (p ce.1.2) = _
    rw [cavityOverlapAt_samePair6_test q i σs hpbase,
      cavityOverlapAt_samePair6_test q i σs hpedge]
    rfl
  rw [hpoint] at hr
  cases hrel : edgeRelation ee01 e <;>
    simp [momentCoeff, weightedCavityMoment6, F, ce, canonicalEdge6, hrel] at hr ⊢ <;>
    exact hr

noncomputable def weightedScoreTerm6 {N : ℕ} (q : ℝ) (i : Fin N)
    (f : ℝ → ℝ) (target deriv : ReplicaEdge 6) : ReplicaFun N 6 := fun σs =>
  f (cavityScaledArgument q i σs) * (edgeSpin6 i target σs - q) *
    cavityInteractionAt q i (σs deriv.1.1) (σs deriv.1.2)

lemma weighted_endpoint_score_term
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f : ℝ → ℝ) (target deriv : ReplicaEdge 6) :
    lastSiteQuenchedAverage (s := s) path i 0
        (weightedScoreTerm6 q i f target deriv) =
      decoupledSpinCoefficient q (rsR β h) (edgeRelation target deriv) *
        weightedCavityMoment6 (s := s) path i 0 f deriv := by
  let F : (Fin 6 → SiteBaseConfig N i) → ℝ := fun ρs =>
    f (Real.sqrt (N : ℝ) * bulkEdgeCavity6 q i ee01 ρs +
      q / Real.sqrt (N : ℝ)) * bulkEdgeCavity6 q i deriv ρs
  have hobs : weightedScoreTerm6 q i f target deriv = fun σs =>
      F (replicasSplitSiteEquiv i σs).1 *
        ((edgeSpin6 i target σs - q) * edgeSpin6 i deriv σs) := by
    funext σs
    unfold weightedScoreTerm6
    rw [cavityInteractionAt_edge6_test]
    simp only [F, cavityScaledArgument,
      bulkEdgeCavity6_split_test]
    simp only [ee01]
    ring
  rw [hobs, lastSite_zero_centered_edge_factor_test path hN hh hq i F target deriv]
  congr 1
  unfold weightedCavityMoment6
  congr 1
  funext σs
  simp only [F, cavityScaledArgument, bulkEdgeCavity6_split_test]
  simp only [ee01]

lemma weighted_endpoint_fullScore
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f : ℝ → ℝ) (target : ReplicaEdge 6) :
    lastSiteQuenchedAverage (s := s) path i 0 (fun σs : Replicas N 6 =>
      f (cavityScaledArgument q i σs) * (edgeSpin6 i target σs - q) *
        normalizedCavityScoreObservable (n := 4) q i σs) =
      abstractRow q (rsR β h)
        (weightedCavityMoment6 (s := s) path i 0 f ee01)
        (weightedCavityMoment6 (s := s) path i 0 f ee02)
        (weightedCavityMoment6 (s := s) path i 0 f ee23) target := by
  let L := lastSiteAverageLinearMap_test (n := 6) (s := s) path i 0
  have hobs : (fun σs : Replicas N 6 =>
      f (cavityScaledArgument q i σs) * (edgeSpin6 i target σs - q) *
        normalizedCavityScoreObservable (n := 4) q i σs) =
      weightedScoreTerm6 q i f target ee01 + weightedScoreTerm6 q i f target ee02 +
      weightedScoreTerm6 q i f target ee03 + weightedScoreTerm6 q i f target ee12 +
      weightedScoreTerm6 q i f target ee13 + weightedScoreTerm6 q i f target ee23 -
      (4 : ℝ) • (weightedScoreTerm6 q i f target ee04 +
        weightedScoreTerm6 q i f target ee14 +
        weightedScoreTerm6 q i f target ee24 +
        weightedScoreTerm6 q i f target ee34) +
      (10 : ℝ) • weightedScoreTerm6 q i f target ee45 := by
    funext σs
    rw [normalizedScore_four_explicit_test]
    simp only [Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul,
      weightedScoreTerm6]
    simp [ee01, ee02, ee03, ee12, ee13, ee23, ee04, ee14, ee24, ee34, ee45,
      edgeSpin6]
    ring
  change L _ = _
  rw [hobs]
  simp only [map_add, map_sub, map_smul, L,
    lastSiteAverageLinearMap_apply_test, smul_eq_mul]
  have ht (deriv : ReplicaEdge 6) :
      lastSiteQuenchedAverage (s := s) path i 0
          (weightedScoreTerm6 q i f target deriv) =
        abstractTerm q (rsR β h)
          (weightedCavityMoment6 (s := s) path i 0 f ee01)
          (weightedCavityMoment6 (s := s) path i 0 f ee02)
          (weightedCavityMoment6 (s := s) path i 0 f ee23) target deriv := by
    rw [weighted_endpoint_score_term path hN hh hq i f target deriv]
    unfold abstractTerm
    rw [weightedCavityMoment6_by_relation]
  rw [ht ee01, ht ee02, ht ee03, ht ee12, ht ee13, ht ee23,
    ht ee04, ht ee14, ht ee24, ht ee34, ht ee45]
  rfl

noncomputable def weightedOffdiag
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 4 =>
    (spinPairAt_test i target σs - q) * f (cavityScaledArgument q i σs))

noncomputable def weightedOffdiagDeriv
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  s * β ^ 2 * lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 6 =>
      (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
        f (cavityScaledArgument q i σs) *
        normalizedCavityScoreObservable (n := 4) q i σs)

lemma hasDerivAt_weightedOffdiag
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hv : v ∈ Set.Ioo (0 : ℝ) 1)
    (i : Fin N) (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    HasDerivAt (fun w => weightedOffdiag (s := s) path i w f target)
      (weightedOffdiagDeriv (s := s) path i v f target) v := by
  let F : ReplicaFun N 4 := fun σs =>
    (spinPairAt_test i target σs - q) * f (cavityScaledArgument q i σs)
  have hder := hasDerivAt_lastSiteQuenchedAverage_fixedScore path i hs hv F
  simpa [weightedOffdiag, weightedOffdiagDeriv, F, spinPairAt_test,
    scoreInternalEdge_test, initialReplicas, cavityScaledArgument] using hder

lemma weightedOffdiag_zero
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    weightedOffdiag (s := s) path i 0 f target = 0 := by
  let t6 : ReplicaEdge 6 := scoreInternalEdge_test target
  let Fbulk : (Fin 6 → SiteBaseConfig N i) → ℝ := fun ρs =>
    f (Real.sqrt (N : ℝ) * bulkEdgeCavity6 q i ee01 ρs +
      q / Real.sqrt (N : ℝ))
  have hspin := lastSite_zero_bulk_spin_two_test (s := s) path hN hh hq i Fbulk
    (edgeFinset6 t6) (edgeFinset6_card t6)
  unfold weightedOffdiag
  rw [← lastSiteAverage_initialReplicas_test (n := 4) (s := s) path i 0]
  have hobs : (fun σs : Replicas N 6 =>
      ((spinPairAt_test i target (initialReplicas σs) - q) *
        f (cavityScaledArgument q i (initialReplicas σs)))) =
      fun σs => Fbulk (replicasSplitSiteEquiv i σs).1 *
        (∏ a ∈ edgeFinset6 t6, SpinGlass.spin N (σs a) i) -
          q * Fbulk (replicasSplitSiteEquiv i σs).1 := by
    funext σs
    rw [← edgeSpin6_eq_prod i t6 σs]
    dsimp [Fbulk, t6, scoreInternalEdge_test, spinPairAt_test,
      initialReplicas, cavityScaledArgument]
    rw [bulkEdgeCavity6_split_test]
    simp only [ee01]
    simp [edgeSpin6, t6, scoreInternalEdge_test, spinPairAt_test, initialReplicas]
    ring
  rw [hobs, lastSiteQuenchedAverage_sub_test,
    lastSiteQuenchedAverage_const_mul_test]
  change _ - q * lastSiteQuenchedAverage (s := s) path i 0
      (fun σs => Fbulk (replicasSplitSiteEquiv i σs).1) = 0
  rw [hspin]
  ring

lemma weightedOffdiagDeriv_zero_eq_matrix
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f : ℝ → ℝ) (k : Fin 3) :
    weightedOffdiagDeriv (s := s) path i 0 f (targetEdge4_test k) =
      s * (cavityMatrix β q (rsR β h)).mulVec
        (weightedCavityVectorAt (s := s) path i 0 f) k := by
  have hend := weighted_endpoint_fullScore (s := s) path hN hh hq i f
    (scoreInternalEdge_test (targetEdge4_test k))
  unfold weightedOffdiagDeriv
  rw [show lastSiteQuenchedAverage (s := s) path i 0
      (fun σs : Replicas N 6 =>
        (spinPairAt_test i (scoreInternalEdge_test (targetEdge4_test k)) σs - q) *
          f (cavityScaledArgument q i σs) *
          normalizedCavityScoreObservable (n := 4) q i σs) =
      abstractRow q (rsR β h)
        (weightedCavityMoment6 (s := s) path i 0 f ee01)
        (weightedCavityMoment6 (s := s) path i 0 f ee02)
        (weightedCavityMoment6 (s := s) path i 0 f ee23)
        (scoreInternalEdge_test (targetEdge4_test k)) by
    simpa [spinPairAt_test, edgeSpin6, mul_assoc, mul_left_comm, mul_comm] using hend]
  unfold weightedCavityVectorAt
  calc
    s * β ^ 2 * abstractRow q (rsR β h)
        (weightedCavityMoment6 (s := s) path i 0 f ee01)
        (weightedCavityMoment6 (s := s) path i 0 f ee02)
        (weightedCavityMoment6 (s := s) path i 0 f ee23)
        (scoreInternalEdge_test (targetEdge4_test k)) =
      s * (β ^ 2 * abstractRow q (rsR β h)
        (weightedCavityMoment6 (s := s) path i 0 f ee01)
        (weightedCavityMoment6 (s := s) path i 0 f ee02)
        (weightedCavityMoment6 (s := s) path i 0 f ee23)
        (scoreInternalEdge_test (targetEdge4_test k))) := by ring
    _ = _ := by rw [beta_sq_abstractRow_eq_matrix_test]

noncomputable def cavitySquare
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 4 => cavityOverlapAt q i σs 0 1 ^ 2)

lemma cavitySquare_nonneg
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) : 0 ≤ cavitySquare (s := s) path i v := by
  unfold cavitySquare lastSiteQuenchedAverage quenchedReplicaAverage
  apply integral_nonneg
  intro ω
  apply replicaGibbsAverage_nonneg
  intro σs
  positivity

lemma lastSite_square_edge_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (e : ReplicaEdge (n + 2)) :
    lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N (n + 2) =>
      cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2) =
    lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N (n + 2) =>
      cavityOverlapAt q i σs 0 1 ^ 2) := by
  let F : ReplicaFun N (n + 2) := fun σs => cavityOverlapAt q i σs 0 1 ^ 2
  have hr := quenchedReplicaAverage_relabel
    (lastSiteHamiltonian (s := s) path i v) F (pairPerm_test e)
  have hp : (fun σs => F (replicaRelabelEquiv (pairPerm_test e) σs)) =
      fun σs => cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 := by
    funext σs
    dsimp only [F]
    change cavityOverlapAt q i σs (pairPerm_test e 0) (pairPerm_test e 1) ^ 2 = _
    rw [pairPerm_zero_test, pairPerm_one_test]
  rw [hp] at hr
  exact hr

lemma canonicalSquare6_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) :
    lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 6 =>
      cavityOverlapAt q i σs 0 1 ^ 2) = cavitySquare (s := s) path i v := by
  let F : ReplicaFun N 4 := fun σs => cavityOverlapAt q i σs 0 1 ^ 2
  rw [show (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) =
      fun σs => F (initialReplicas σs) by rfl,
    lastSiteAverage_initialReplicas_test]
  rfl

lemma canonicalSquare8_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) :
    lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 8 =>
      cavityOverlapAt q i σs 0 1 ^ 2) = cavitySquare (s := s) path i v := by
  calc
    _ = lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 6 =>
        cavityOverlapAt q i σs 0 1 ^ 2) := by
      let F : ReplicaFun N 6 := fun σs => cavityOverlapAt q i σs 0 1 ^ 2
      rw [show (fun σs : Replicas N 8 => cavityOverlapAt q i σs 0 1 ^ 2) =
          fun σs => F (initialReplicas σs) by rfl,
        lastSiteAverage_initialReplicas_test]
    _ = _ := canonicalSquare6_eq path i v

lemma abs_weighted_QQ_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} {β h q s v M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (W : ReplicaFun N (n + 2)) (hW : ∀ σs, |W σs| ≤ M)
    (hM : 0 ≤ M) (e f : ReplicaEdge (n + 2)) :
    |lastSiteQuenchedAverage (s := s) path i v (fun σs => W σs *
      cavityOverlapAt q i σs e.1.1 e.1.2 *
      cavityOverlapAt q i σs f.1.1 f.1.2)| ≤
      M * lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N (n + 2) => cavityOverlapAt q i σs 0 1 ^ 2) := by
  calc
    _ ≤ lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        M * (cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 +
          cavityOverlapAt q i σs f.1.1 f.1.2 ^ 2) / 2) := by
      apply abs_lastSiteAverage_le_test
      intro σs
      rw [abs_mul, abs_mul]
      have hxy : 2 * |cavityOverlapAt q i σs e.1.1 e.1.2| *
          |cavityOverlapAt q i σs f.1.1 f.1.2| ≤
          cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 +
            cavityOverlapAt q i σs f.1.1 f.1.2 ^ 2 := by
        nlinarith [sq_nonneg
          (|cavityOverlapAt q i σs e.1.1 e.1.2| -
            |cavityOverlapAt q i σs f.1.1 f.1.2|),
          sq_abs (cavityOverlapAt q i σs e.1.1 e.1.2),
          sq_abs (cavityOverlapAt q i σs f.1.1 f.1.2)]
      have hmul := mul_le_mul_of_nonneg_right hxy hM
      nlinarith [hW σs, abs_nonneg (W σs),
        abs_nonneg (cavityOverlapAt q i σs e.1.1 e.1.2),
        abs_nonneg (cavityOverlapAt q i σs f.1.1 f.1.2)]
    _ = M * lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N (n + 2) => cavityOverlapAt q i σs 0 1 ^ 2) := by
      let L := lastSiteAverageLinearMap_test (n := n + 2) (s := s) path i v
      change L _ = _
      rw [show (fun σs : Replicas N (n + 2) =>
          M * (cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 +
            cavityOverlapAt q i σs f.1.1 f.1.2 ^ 2) / 2) =
          (M / 2) • ((fun σs => cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2) +
            fun σs => cavityOverlapAt q i σs f.1.1 f.1.2 ^ 2) by
        funext σs; simp [smul_eq_mul]; ring]
      simp only [map_smul, map_add, smul_eq_mul, L,
        lastSiteAverageLinearMap_apply_test]
      rw [lastSite_square_edge_eq path i v e, lastSite_square_edge_eq path i v f]
      ring

noncomputable def weightedOffdiagSecondDeriv
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  (s * β ^ 2) ^ 2 * lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 8 =>
      (spinPairAt_test i (scoreInternalEdge_test target) (initialReplicas σs) - q) *
        f (cavityScaledArgument q i σs) *
        normalizedCavityScoreObservable (n := 4) q i (initialReplicas σs) *
        normalizedCavityScoreObservable (n := 6) q i σs)

lemma hasDerivAt_weightedOffdiagDeriv
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hv : v ∈ Set.Ioo (0 : ℝ) 1)
    (i : Fin N) (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    HasDerivAt (fun w => weightedOffdiagDeriv (s := s) path i w f target)
      (weightedOffdiagSecondDeriv (s := s) path i v f target) v := by
  let F : ReplicaFun N 6 := fun σs =>
    (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
      f (cavityScaledArgument q i σs) *
      normalizedCavityScoreObservable (n := 4) q i σs
  have hder := hasDerivAt_lastSiteQuenchedAverage_fixedScore path i hs hv F
  have hscaled := hder.const_mul (s * β ^ 2)
  simpa [weightedOffdiagDeriv, weightedOffdiagSecondDeriv, F,
    spinPairAt_test, scoreInternalEdge_test, initialReplicas,
    cavityScaledArgument, mul_assoc, pow_two] using hscaled

lemma abs_weightedOffdiagSecondDeriv_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (hf : ∀ x, |f x| ≤ M)
    (hM : 0 ≤ M) (target : ReplicaEdge 4) :
    |weightedOffdiagSecondDeriv (s := s) path i v f target| ≤
      4608 * β ^ 4 * M * cavitySquare (s := s) path i v := by
  let Base : ReplicaFun N 8 := fun σs =>
    (spinPairAt_test i (scoreInternalEdge_test target) (initialReplicas σs) - q) *
      f (cavityScaledArgument q i σs) *
      normalizedCavityScoreObservable (n := 4) q i (initialReplicas σs)
  have houter (d : ReplicaEdge 8) :
      |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        Base σs * cavityInteractionAt q i (σs d.1.1) (σs d.1.2))| ≤
        64 * M * cavitySquare (s := s) path i v := by
    let P : ReplicaFun N 8 := fun σs =>
      (spinPairAt_test i (scoreInternalEdge_test target) (initialReplicas σs) - q) *
        f (cavityScaledArgument q i σs) *
        cavityInteractionAt q i (σs d.1.1) (σs d.1.2)
    have hinner (e : ReplicaEdge 6) :
        |lastSiteQuenchedAverage (s := s) path i v (fun σs => P σs *
          cavityInteractionAt q i (initialReplicas σs e.1.1)
            (initialReplicas σs e.1.2))| ≤
          2 * M * cavitySquare (s := s) path i v := by
      let et : ReplicaEdge 8 := scoreInternalEdge_test (scoreInternalEdge_test target)
      let ei : ReplicaEdge 8 := scoreInternalEdge_test e
      let W : ReplicaFun N 8 := fun σs =>
        (spinPairAt_test i et σs - q) * f (cavityScaledArgument q i σs) *
          spinPairAt_test i d σs * spinPairAt_test i ei σs
      have hW : ∀ σs, |W σs| ≤ 2 * M := by
        intro σs
        rw [abs_mul, abs_mul, abs_mul, abs_spinPairAt_test, abs_spinPairAt_test,
          mul_one, mul_one]
        have ht := abs_centeredSpinPair_le_two_test hqI i et σs
        have hm := hf (cavityScaledArgument q i σs)
        nlinarith [abs_nonneg (f (cavityScaledArgument q i σs))]
      have hqq := abs_weighted_QQ_le (s := s) (v := v) path i W hW
        (mul_nonneg (by norm_num) hM) d ei
      rw [canonicalSquare8_eq] at hqq
      calc
        _ = |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
            W σs * cavityOverlapAt q i σs d.1.1 d.1.2 *
              cavityOverlapAt q i σs ei.1.1 ei.1.2)| := by
          congr 2
          funext σs
          dsimp [P, W, et, ei, scoreInternalEdge_test, initialReplicas]
          rw [cavityInteractionAt_eq_spin_mul_overlap_test,
            cavityInteractionAt_eq_spin_mul_overlap_test]
          simp [spinPairAt_test, scoreInternalEdge_test, initialReplicas]
          ring
        _ ≤ (2 * M) * cavitySquare (s := s) path i v := hqq
        _ = _ := by ring
    have hemb := abs_embeddedScore_four_le_test (M :=
        2 * M * cavitySquare (s := s) path i v) path i P
      (mul_nonneg (mul_nonneg (by norm_num) hM) (cavitySquare_nonneg path i v)) hinner
    calc
      _ = |lastSiteQuenchedAverage (s := s) path i v (fun σs => P σs *
          normalizedCavityScoreObservable (n := 4) q i (initialReplicas σs))| := by
        congr 2
        funext σs
        dsimp [Base, P]
        ring
      _ ≤ 32 * (2 * M * cavitySquare (s := s) path i v) := hemb
      _ = _ := by ring
  have hout := normalizedScore_average_bound_test (M :=
      64 * M * cavitySquare (s := s) path i v) path i Base
    (mul_nonneg (mul_nonneg (by norm_num) hM) (cavitySquare_nonneg path i v)) houter
  have hcard : Fintype.card (ReplicaEdge 6) = 15 := by native_decide
  norm_num [hcard] at hout
  unfold weightedOffdiagSecondDeriv
  rw [abs_mul, abs_pow, abs_mul, abs_pow, abs_of_nonneg hs.1, sq_abs]
  calc
    (s * β ^ 2) ^ 2 * _ ≤ (1 * β ^ 2) ^ 2 *
        (72 * (64 * M * cavitySquare (s := s) path i v)) := by
      have hb0 : 0 ≤ s * β ^ 2 := mul_nonneg hs.1 (sq_nonneg β)
      have hb : s * β ^ 2 ≤ 1 * β ^ 2 :=
        mul_le_mul_of_nonneg_right hs.2 (sq_nonneg β)
      exact mul_le_mul (pow_le_pow_left₀ hb0 hb 2) hout (abs_nonneg _) (by positivity)
    _ = _ := by ring

lemma cavitySquare_uniform_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (i : Fin N) {v : ℝ} (hv : v ∈ Set.Icc (0 : ℝ) 1) :
    cavitySquare (s := s) path i v ≤
      cavityVector path s 0 + thirdMoment path s + 3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
        32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
          (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) := by
  let F : ℝ → ℝ := fun w => cavityQuadratic4_test (s := s) path i w e4_01 e4_01
  let F' : ℝ → ℝ := fun w => cavityQuadraticDeriv4_test (s := s) path i w e4_01 e4_01
  have hFv : cavitySquare (s := s) path i v = F v := by
    simp [cavitySquare, F, cavityQuadratic4_test, e4_01, pow_two]
  have hF1 : cavitySquare (s := s) path i 1 = F 1 := by
    simp [cavitySquare, F, cavityQuadratic4_test, e4_01, pow_two]
  have hinterp : |F 1 - F v| ≤
      32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
        (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) := by
    rcases hv.2.eq_or_lt with rfl | hvlt
    · simp
      have ht := thirdMoment_nonneg path s
      positivity
    · obtain ⟨c, hc, hslope⟩ := exists_hasDerivAt_eq_slope F F' hvlt
        ((continuous_lastSiteQuenchedAverage path i
          (fun σs : Replicas N 4 => cavityOverlapAt q i σs 0 1 *
            cavityOverlapAt q i σs 0 1)).continuousOn)
        (fun w hw => hasDerivAt_cavityQuadratic4_test path hs
          ⟨lt_of_le_of_lt hv.1 hw.1, hw.2⟩ i e4_01 e4_01)
      have hcube := cavityCube4_uniform_test path hN hqI hs
        ⟨(lt_of_le_of_lt hv.1 hc.1).le, hc.2.le⟩ i
      have hd := abs_cavityQuadraticDeriv4_le_test path hs i c e4_01 e4_01
      change |F' c| ≤ 32 * β ^ 2 * cavityCube4 (s := s) path i c at hd
      have hslope' : F' c = (F 1 - F v) / (1 - v) := by simpa using hslope
      have hone : 0 < 1 - v := sub_pos.mpr hvlt
      have heq : F 1 - F v = F' c * (1 - v) := by
        rw [hslope']
        field_simp [hone.ne']
      have hdiff : |F 1 - F v| ≤ |F' c| := by
        rw [heq, abs_mul, abs_of_pos hone]
        have hfac : 1 - v ≤ 1 := by linarith [hv.1]
        exact mul_le_of_le_one_right (abs_nonneg (F' c)) hfac
      exact hdiff.trans (hd.trans (by gcongr))
  have hend := cavityQuadratic4_one_full_error_test (s := s) path hN i e4_01 e4_01
  have hfull : fullCenteredMoment4_test (s := s) path e4_01 e4_01 =
      cavityVector path s 0 := by
    simpa [targetEdge4_test, e4_01] using fullCenteredMoment4_target_eq_test path 0
  rw [hfull] at hend
  change |F 1 - cavityVector path s 0| ≤ _ at hend
  rw [hFv]
  have htri : F v ≤ cavityVector path s 0 +
      |F 1 - cavityVector path s 0| + |F 1 - F v| := by
    have h₁ := le_abs_self (F v - F 1)
    have h₂ := le_abs_self (F 1 - cavityVector path s 0)
    rw [abs_sub_comm (F v) (F 1)] at h₁
    linarith
  linarith

lemma weightedOffdiag_taylor_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f : ℝ → ℝ) (hf : ∀ x, |f x| ≤ M) (hM : 0 ≤ M)
    (target : ReplicaEdge 4) :
    |weightedOffdiag (s := s) path i 1 f target -
      weightedOffdiagDeriv (s := s) path i 0 f target| ≤
      4608 * β ^ 4 * M *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3)))) := by
  let G : ℝ → ℝ := fun v => weightedOffdiag (s := s) path i v f target
  let D : ℝ → ℝ := fun v => weightedOffdiagDeriv (s := s) path i v f target
  let D' : ℝ → ℝ := fun v => weightedOffdiagSecondDeriv (s := s) path i v f target
  obtain ⟨c, hc, hGc⟩ := exists_hasDerivAt_eq_slope G D (by norm_num)
    ((continuous_lastSiteQuenchedAverage path i
      (fun σs : Replicas N 4 =>
        (spinPairAt_test i target σs - q) * f (cavityScaledArgument q i σs))).continuousOn)
    (fun v hv => hasDerivAt_weightedOffdiag path hs hv i f target)
  have hG0 := weightedOffdiag_zero (s := s) path hN hh hq i f target
  have hGc' : G 1 = D c := by
    dsimp [G, D] at hGc ⊢
    norm_num at hGc
    linarith
  obtain ⟨d, hd, hDd⟩ := exists_hasDerivAt_eq_slope D D' hc.1
    ((continuous_const.mul (continuous_lastSiteQuenchedAverage path i
      (fun σs : Replicas N 6 =>
        (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
          f (cavityScaledArgument q i σs) *
          normalizedCavityScoreObservable (n := 4) q i σs))).continuousOn)
    (fun v hv => hasDerivAt_weightedOffdiagDeriv path hs
      ⟨hv.1, hv.2.trans hc.2⟩ i f target)
  have hsecond := abs_weightedOffdiagSecondDeriv_le path hqI hs i d f hf hM target
  have hsquare := cavitySquare_uniform_le path hN hqI hs i
    ⟨hd.1.le, hd.2.le.trans hc.2.le⟩
  have hDd' : D c - D 0 = D' d * c := by
    have heq : D' d * c = D c - D 0 :=
      (eq_div_iff hc.1.ne').mp (by simpa using hDd)
    linarith
  change |G 1 - D 0| ≤ _
  rw [hGc']
  change |D c - D 0| ≤ _
  rw [hDd', abs_mul]
  have hcabs : |c| ≤ 1 := by
    rw [abs_of_pos hc.1]
    exact hc.2.le
  calc
    |D' d| * |c| ≤ |D' d| * 1 :=
      mul_le_mul_of_nonneg_left hcabs (abs_nonneg _)
    _ = |D' d| := by ring
    _ ≤ 4608 * β ^ 4 * M * cavitySquare (s := s) path i d := hsecond
    _ ≤ _ := by
      have hpref : 0 ≤ 4608 * β ^ 4 * M := by positivity
      exact mul_le_mul_of_nonneg_left hsquare hpref

lemma abs_small_weighted_cavityInteraction_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} {β h q s v K : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (i : Fin N) (W : ReplicaFun N (n + 2))
    (hW : ∀ σs, |W σs| ≤ K / Real.sqrt (N : ℝ)) (hK : 0 ≤ K)
    (e : ReplicaEdge (n + 2)) :
    |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
      W σs * cavityInteractionAt q i (σs e.1.1) (σs e.1.2))| ≤
      (K / 2) * (lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N (n + 2) => cavityOverlapAt q i σs 0 1 ^ 2) +
          1 / (N : ℝ)) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  have hsqrt_sq : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  calc
    _ ≤ lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        (K / 2) * (cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 +
          1 / (N : ℝ))) := by
      apply abs_lastSiteAverage_le_test
      intro σs
      rw [cavityInteractionAt_eq_spin_mul_overlap_test, abs_mul, abs_mul]
      change |W σs| * (|spinPairAt_test i e σs| *
        |cavityOverlapAt q i σs e.1.1 e.1.2|) ≤ _
      rw [abs_spinPairAt_test, one_mul]
      have hyoung : 2 * |cavityOverlapAt q i σs e.1.1 e.1.2| /
          Real.sqrt (N : ℝ) ≤
          cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 + 1 / (N : ℝ) := by
        have hsq := sq_nonneg
          (|cavityOverlapAt q i σs e.1.1 e.1.2| - 1 / Real.sqrt (N : ℝ))
        have hinvSq : (1 / Real.sqrt (N : ℝ)) ^ 2 = 1 / (N : ℝ) := by
          rw [div_pow, hsqrt_sq]
          norm_num
        have hy : 2 * |cavityOverlapAt q i σs e.1.1 e.1.2| *
            (1 / Real.sqrt (N : ℝ)) ≤
            |cavityOverlapAt q i σs e.1.1 e.1.2| ^ 2 +
              (1 / Real.sqrt (N : ℝ)) ^ 2 := by
          nlinarith
        calc
          _ = 2 * |cavityOverlapAt q i σs e.1.1 e.1.2| *
              (1 / Real.sqrt (N : ℝ)) := by ring
          _ ≤ _ := hy
          _ = _ := by rw [hinvSq, sq_abs]
      have hbase := mul_le_mul_of_nonneg_right (hW σs)
        (abs_nonneg (cavityOverlapAt q i σs e.1.1 e.1.2))
      calc
        _ ≤ K / Real.sqrt (N : ℝ) *
            |cavityOverlapAt q i σs e.1.1 e.1.2| := hbase
        _ = (K / 2) * (2 * |cavityOverlapAt q i σs e.1.1 e.1.2| /
            Real.sqrt (N : ℝ)) := by ring
        _ ≤ _ := mul_le_mul_of_nonneg_left hyoung (by positivity)
    _ = (K / 2) * (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N (n + 2) => cavityOverlapAt q i σs 0 1 ^ 2) +
        1 / (N : ℝ)) := by
      let L := lastSiteAverageLinearMap_test (n := n + 2) (s := s) path i v
      change L _ = _
      rw [show (fun σs : Replicas N (n + 2) =>
          (K / 2) * (cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2 + 1 / (N : ℝ))) =
          (K / 2) • ((fun σs => cavityOverlapAt q i σs e.1.1 e.1.2 ^ 2) +
            (1 / (N : ℝ)) • (fun _ => 1)) by
        funext σs
        simp [smul_eq_mul]]
      simp only [map_smul, map_add, smul_eq_mul, L,
        lastSiteAverageLinearMap_apply_test]
      rw [lastSite_square_edge_eq path i v e, lastSiteQuenchedAverage_one_test]
      ring


end CLT
end SpinGlass.AT
