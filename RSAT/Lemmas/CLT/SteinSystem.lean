import Lemmas.CLT.SteinCavity

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace SpinGlass.AT
namespace CLT

universe u

open CavityEstimates

noncomputable def fullScaledArgument {N n : ℕ} (q : ℝ)
    (σs : Replicas N (n + 2)) : ℝ :=
  Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1

lemma fullScaledArgument_eq_cavity_add
    {N n : ℕ} (hN : 0 < N) (q : ℝ) (i : Fin N)
    (σs : Replicas N (n + 2)) :
    fullScaledArgument q σs = cavityScaledArgument q i σs +
      (spinPairAt_test i
        ⟨((0 : Fin (n + 2)), (1 : Fin (n + 2))), by simp⟩ σs - q) /
          Real.sqrt (N : ℝ) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : Real.sqrt (N : ℝ) ≠ 0 := (Real.sqrt_pos.2 hNr).ne'
  have hsquare : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  unfold fullScaledArgument cavityScaledArgument
  rw [centeredOverlap_eq_cavityOverlapAt_add q i σs 0 1]
  simp only [spinPairAt_test]
  field_simp [hsqrt, hNr.ne']
  ring_nf
  rw [hsquare]

noncomputable def weightedCavityMomentDeriv6
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (e : ReplicaEdge 6) : ℝ :=
  s * β ^ 2 * lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 8 =>
      f (cavityScaledArgument q i σs) *
        cavityOverlapAt q i (initialReplicas σs) e.1.1 e.1.2 *
        normalizedCavityScoreObservable (n := 6) q i σs)

lemma hasDerivAt_weightedCavityMoment6
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hv : v ∈ Set.Ioo (0 : ℝ) 1)
    (i : Fin N) (f : ℝ → ℝ) (e : ReplicaEdge 6) :
    HasDerivAt (fun w => weightedCavityMoment6 (s := s) path i w f e)
      (weightedCavityMomentDeriv6 (s := s) path i v f e) v := by
  let F : ReplicaFun N 6 := fun σs =>
    f (cavityScaledArgument q i σs) * cavityOverlapAt q i σs e.1.1 e.1.2
  have hder := hasDerivAt_lastSiteQuenchedAverage_fixedScore path i hs hv F
  simpa [weightedCavityMoment6, weightedCavityMomentDeriv6, F,
    cavityScaledArgument, initialReplicas] using hder

lemma abs_weightedCavityMomentDeriv6_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N) (v : ℝ)
    (f : ℝ → ℝ) (hf : ∀ x, |f x| ≤ M) (hM : 0 ≤ M)
    (e : ReplicaEdge 6) :
    |weightedCavityMomentDeriv6 (s := s) path i v f e| ≤
      72 * β ^ 2 * M * cavitySquare (s := s) path i v := by
  let P : ReplicaFun N 8 := fun σs =>
    f (cavityScaledArgument q i σs) *
      cavityOverlapAt q i (initialReplicas σs) e.1.1 e.1.2
  have hterm (d : ReplicaEdge 8) :
      |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        P σs * cavityInteractionAt q i (σs d.1.1) (σs d.1.2))| ≤
        M * cavitySquare (s := s) path i v := by
    let W : ReplicaFun N 8 := fun σs =>
      f (cavityScaledArgument q i σs) * spinPairAt_test i d σs
    have hW : ∀ σs, |W σs| ≤ M := by
      intro σs
      simp only [W, abs_mul, abs_spinPairAt_test, mul_one]
      exact hf _
    have hqq := abs_weighted_QQ_le (s := s) (v := v) path i W hW hM
      (scoreInternalEdge_test e) d
    rw [canonicalSquare8_eq] at hqq
    convert hqq using 1
    congr 2
    funext σs
    dsimp [P, W]
    rw [cavityInteractionAt_eq_spin_mul_overlap_test]
    simp [spinPairAt_test, scoreInternalEdge_test]
    ring
  have hscore := normalizedScore_average_bound_test
    (M := M * cavitySquare (s := s) path i v) path i P
    (mul_nonneg hM (cavitySquare_nonneg path i v)) hterm
  have hcard : Fintype.card (ReplicaEdge 6) = 15 := by native_decide
  norm_num [hcard] at hscore
  unfold weightedCavityMomentDeriv6
  rw [abs_mul, abs_mul, abs_of_nonneg hs.1, abs_of_nonneg (sq_nonneg β)]
  calc
    s * β ^ 2 * _ ≤ 1 * β ^ 2 * (72 * (M * cavitySquare (s := s) path i v)) := by
      gcongr
      exact hs.2
    _ = _ := by ring

noncomputable def weightedSource
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i v (fun σs : Replicas N 4 =>
    (spinPairAt_test i target σs - q) *
      (f (fullScaledArgument q σs) - f (cavityScaledArgument q i σs)))

noncomputable def weightedSourceDeriv
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  s * β ^ 2 * lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 6 =>
      (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
        (f (fullScaledArgument q (initialReplicas σs)) -
          f (cavityScaledArgument q i (initialReplicas σs))) *
        normalizedCavityScoreObservable (n := 4) q i σs)

lemma hasDerivAt_weightedSource
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hv : v ∈ Set.Ioo (0 : ℝ) 1)
    (i : Fin N) (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    HasDerivAt (fun w => weightedSource (s := s) path i w f target)
      (weightedSourceDeriv (s := s) path i v f target) v := by
  let F : ReplicaFun N 4 := fun σs =>
    (spinPairAt_test i target σs - q) *
      (f (fullScaledArgument q σs) - f (cavityScaledArgument q i σs))
  have hder := hasDerivAt_lastSiteQuenchedAverage_fixedScore path i hs hv F
  simpa [weightedSource, weightedSourceDeriv, F, fullScaledArgument,
    cavityScaledArgument, spinPairAt_test, scoreInternalEdge_test,
    initialReplicas] using hder

lemma abs_small_weighted_cavityInteraction_le_system
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

lemma abs_weightedSourceDeriv_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N) (v : ℝ)
    (f : ℝ → ℝ) (hf : ∀ x y, |f x - f y| ≤ L * |x - y|)
    (hL : 0 ≤ L) (target : ReplicaEdge 4) :
    |weightedSourceDeriv (s := s) path i v f target| ≤
      256 * β ^ 2 * L *
        (cavitySquare (s := s) path i v + 1 / (N : ℝ)) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  let P : ReplicaFun N 6 := fun σs =>
    (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
      (f (fullScaledArgument q (initialReplicas σs)) -
        f (cavityScaledArgument q i (initialReplicas σs)))
  have hP : ∀ σs, |P σs| ≤ (4 * L) / Real.sqrt (N : ℝ) := by
    intro σs
    rw [abs_mul]
    have ht := abs_centeredSpinPair_le_two_test hqI i
      (scoreInternalEdge_test target) σs
    have hxy := hf (fullScaledArgument q (initialReplicas σs))
      (cavityScaledArgument q i (initialReplicas σs))
    have hdiff : |fullScaledArgument q (initialReplicas σs) -
        cavityScaledArgument q i (initialReplicas σs)| ≤
        2 / Real.sqrt (N : ℝ) := by
      rw [fullScaledArgument_eq_cavity_add hN, add_sub_cancel_left]
      rw [abs_div]
      have he := abs_centeredSpinPair_le_two_test hqI i
        ⟨((0 : Fin 6), (1 : Fin 6)), by decide⟩ σs
      rw [abs_of_pos hsqrt]
      exact div_le_div_of_nonneg_right he hsqrt.le
    calc
      _ ≤ 2 * (L * |fullScaledArgument q (initialReplicas σs) -
          cavityScaledArgument q i (initialReplicas σs)|) :=
        mul_le_mul ht hxy (abs_nonneg _) (by norm_num)
      _ ≤ 2 * (L * (2 / Real.sqrt (N : ℝ))) := by gcongr
      _ = (4 * L) / Real.sqrt (N : ℝ) := by ring
  have hterm (d : ReplicaEdge 6) :
      |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        P σs * cavityInteractionAt q i (σs d.1.1) (σs d.1.2))| ≤
      (2 * L) * (lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
          1 / (N : ℝ)) := by
    simpa [show (4 * L) / 2 = 2 * L by ring] using
      abs_small_weighted_cavityInteraction_le_system (s := s) (v := v)
      path hN i P hP (by positivity : 0 ≤ 4 * L) d
  have hscore := normalizedScore_average_bound_test
    (M := (2 * L) * (lastSiteQuenchedAverage (s := s) path i v
      (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
        1 / (N : ℝ))) path i P
    (mul_nonneg (by positivity) (add_nonneg (by
      unfold lastSiteQuenchedAverage quenchedReplicaAverage
      apply integral_nonneg
      intro ω
      apply replicaGibbsAverage_nonneg
      intro σs
      positivity) (by positivity))) hterm
  have hcard : Fintype.card (ReplicaEdge 4) = 6 := by native_decide
  norm_num [hcard] at hscore
  have hscore' :
      |lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N 6 =>
          (spinPairAt_test i (scoreInternalEdge_test target) σs - q) *
            (f (fullScaledArgument q (initialReplicas σs)) -
              f (cavityScaledArgument q i (initialReplicas σs))) *
            normalizedCavityScoreObservable (n := 4) q i σs)| ≤
        32 * ((2 * L) * (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
            1 / (N : ℝ))) := by
    simpa [P, mul_assoc] using hscore
  unfold weightedSourceDeriv
  rw [abs_mul, abs_mul, abs_of_nonneg hs.1, abs_of_nonneg (sq_nonneg β)]
  calc
    s * β ^ 2 * _ ≤ 1 * β ^ 2 *
        (32 * ((2 * L) * (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
            1 / (N : ℝ)))) := by
      gcongr
      exact hs.2
    _ = 64 * β ^ 2 * L *
        (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
            1 / (N : ℝ)) := by ring
    _ = 64 * β ^ 2 * L *
        (cavitySquare (s := s) path i v + 1 / (N : ℝ)) := by
      rw [canonicalSquare6_eq]
    _ ≤ _ := by
      have hz : 0 ≤ β ^ 2 * L *
          (cavitySquare (s := s) path i v + 1 / (N : ℝ)) := by
        exact mul_nonneg (mul_nonneg (sq_nonneg β) hL)
          (add_nonneg (cavitySquare_nonneg path i v) (by positivity))
      nlinarith

lemma weightedSource_endpoint_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f : ℝ → ℝ) (hf : ∀ x y, |f x - f y| ≤ L * |x - y|)
    (hL : 0 ≤ L) (target : ReplicaEdge 4) :
    |weightedSource (s := s) path i 1 f target -
        weightedSource (s := s) path i 0 f target| ≤
      256 * β ^ 2 * L *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
          1 / (N : ℝ)) := by
  let G : ℝ → ℝ := fun v => weightedSource (s := s) path i v f target
  let D : ℝ → ℝ := fun v => weightedSourceDeriv (s := s) path i v f target
  obtain ⟨c, hc, hGc⟩ := exists_hasDerivAt_eq_slope G D (by norm_num)
    ((continuous_lastSiteQuenchedAverage path i
      (fun σs : Replicas N 4 =>
        (spinPairAt_test i target σs - q) *
          (f (fullScaledArgument q σs) -
            f (cavityScaledArgument q i σs)))).continuousOn)
    (fun v hv => hasDerivAt_weightedSource path hs hv i f target)
  have hd := abs_weightedSourceDeriv_le path hN hqI hs i c f hf hL target
  have hsq := cavitySquare_uniform_le path hN hqI hs i ⟨hc.1.le, hc.2.le⟩
  have hdiff : G 1 - G 0 = D c := by
    dsimp only [G, D] at hGc ⊢
    norm_num at hGc
    linarith
  change |G 1 - G 0| ≤ _
  rw [hdiff]
  apply hd.trans
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  linarith

noncomputable def cavityDerivativeAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f' : ℝ → ℝ) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 4 => f' (cavityScaledArgument q i σs))

noncomputable def weightedSourceLinear
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (f' : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  lastSiteQuenchedAverage (s := s) path i 0 (fun σs : Replicas N 4 =>
    (spinPairAt_test i target σs - q) *
      (f' (cavityScaledArgument q i σs) *
        ((spinPairAt_test i e4_01 σs - q) / Real.sqrt (N : ℝ))))

lemma weightedSource_zero_taylor_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s L₂ : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f f' : ℝ → ℝ)
    (hf : ∀ x y, |f x - f y - f' y * (x - y)| ≤ L₂ * (x - y) ^ 2)
    (hL₂ : 0 ≤ L₂) (target : ReplicaEdge 4) :
    |weightedSource (s := s) path i 0 f target -
        weightedSourceLinear (s := s) path i f' target| ≤
      8 * L₂ / (N : ℝ) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  have hsquare : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  unfold weightedSource weightedSourceLinear
  rw [← lastSiteQuenchedAverage_sub_test]
  calc
    _ ≤ lastSiteQuenchedAverage (s := s) path i 0
        (fun _ : Replicas N 4 => 8 * L₂ / (N : ℝ)) := by
      apply abs_lastSiteAverage_le_test
      intro σs
      have ht := abs_centeredSpinPair_le_two_test hqI i target σs
      have harg := fullScaledArgument_eq_cavity_add (n := 2) hN q i σs
      have hrem := hf (fullScaledArgument q σs) (cavityScaledArgument q i σs)
      have hdelta : fullScaledArgument q σs - cavityScaledArgument q i σs =
          (spinPairAt_test i e4_01 σs - q) / Real.sqrt (N : ℝ) := by
        rw [harg]
        simp [e4_01]
      rw [hdelta] at hrem
      have hedge := abs_centeredSpinPair_le_two_test hqI i e4_01 σs
      have hdeltaSq :
          ((spinPairAt_test i e4_01 σs - q) / Real.sqrt (N : ℝ)) ^ 2 ≤
            4 / (N : ℝ) := by
        rw [div_pow, hsquare, ← sq_abs]
        gcongr
        nlinarith [abs_nonneg (spinPairAt_test i e4_01 σs - q)]
      rw [← mul_sub, abs_mul]
      calc
        _ = |spinPairAt_test i target σs - q| *
            |f (fullScaledArgument q σs) - f (cavityScaledArgument q i σs) -
              f' (cavityScaledArgument q i σs) *
                ((spinPairAt_test i e4_01 σs - q) / Real.sqrt (N : ℝ))| := by
          ring
        _ ≤ 2 * (L₂ *
            ((spinPairAt_test i e4_01 σs - q) / Real.sqrt (N : ℝ)) ^ 2) :=
          mul_le_mul ht hrem (abs_nonneg _) (by norm_num)
        _ ≤ 2 * (L₂ * (4 / (N : ℝ))) := by gcongr
        _ = 8 * L₂ / (N : ℝ) := by ring
    _ = 8 * L₂ / (N : ℝ) := by
      rw [show (fun _ : Replicas N 4 => 8 * L₂ / (N : ℝ)) =
        fun _ => (8 * L₂ / (N : ℝ)) * 1 by
        funext σs; ring]
      rw [lastSiteQuenchedAverage_const_mul_test,
        lastSiteQuenchedAverage_one_test, mul_one]

lemma weightedSourceLinear_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f' : ℝ → ℝ) (target : ReplicaEdge 4) :
    weightedSourceLinear (s := s) path i f' target =
      decoupledSpinCoefficient q (rsR β h)
          (edgeRelation (scoreInternalEdge_test target) ee01) /
        Real.sqrt (N : ℝ) * cavityDerivativeAverage (s := s) path i 0 f' := by
  let t6 : ReplicaEdge 6 := scoreInternalEdge_test target
  let F : (Fin 6 → SiteBaseConfig N i) → ℝ := fun ρs =>
    f' (Real.sqrt (N : ℝ) * bulkEdgeCavity6 q i ee01 ρs +
      q / Real.sqrt (N : ℝ))
  have hfac := lastSite_zero_centered_edge_factor_test (s := s) path hN hh hq i
    F t6 ee01
  have hF : lastSiteQuenchedAverage (s := s) path i 0
      (fun σs : Replicas N 6 => F (replicasSplitSiteEquiv i σs).1) =
      cavityDerivativeAverage (s := s) path i 0 f' := by
    unfold cavityDerivativeAverage
    rw [← lastSiteAverage_initialReplicas_test (n := 4) (s := s) path i 0]
    congr 1
    funext σs
    simp only [F, cavityScaledArgument, bulkEdgeCavity6_split_test, ee01]
    rw [cavityOverlapAt_initialReplicas_test]
    congr 3 <;> apply Fin.ext <;> rfl
  rw [hF] at hfac
  have hzero := weightedOffdiag_zero (s := s) path hN hh hq i f' target
  unfold weightedSourceLinear
  rw [← lastSiteAverage_initialReplicas_test (n := 4) (s := s) path i 0]
  let L := lastSiteAverageLinearMap_test (n := 6) (s := s) path i 0
  change L _ = _
  have hobs : (fun σs : Replicas N 6 =>
      (spinPairAt_test i target (initialReplicas σs) - q) *
        (f' (cavityScaledArgument q i (initialReplicas σs)) *
          ((spinPairAt_test i e4_01 (initialReplicas σs) - q) /
            Real.sqrt (N : ℝ)))) =
      (1 / Real.sqrt (N : ℝ)) •
        ((fun σs => F (replicasSplitSiteEquiv i σs).1 *
          ((edgeSpin6 i t6 σs - q) * edgeSpin6 i ee01 σs)) -
        q • (fun σs =>
          (spinPairAt_test i t6 σs - q) * f' (cavityScaledArgument q i σs))) := by
    funext σs
    simp only [Pi.smul_apply, Pi.sub_apply, smul_eq_mul, F, cavityScaledArgument,
      bulkEdgeCavity6_split_test, ee01]
    dsimp [t6, scoreInternalEdge_test, spinPairAt_test, edgeSpin6, initialReplicas]
    simp [e4_01, ee01]
    ring
  rw [hobs, map_smul, map_sub, map_smul]
  simp only [smul_eq_mul, L, lastSiteAverageLinearMap_apply_test]
  have hzero6 : lastSiteQuenchedAverage (s := s) path i 0
      (fun σs : Replicas N 6 =>
        (spinPairAt_test i t6 σs - q) * f' (cavityScaledArgument q i σs)) = 0 := by
    let G : ReplicaFun N 4 := fun σs =>
      (spinPairAt_test i target σs - q) * f' (cavityScaledArgument q i σs)
    rw [show (fun σs : Replicas N 6 =>
        (spinPairAt_test i t6 σs - q) * f' (cavityScaledArgument q i σs)) =
        fun σs => G (initialReplicas σs) by
      funext σs
      simp only [G, t6, scoreInternalEdge_test, spinPairAt_test,
        cavityScaledArgument, initialReplicas]
      rw [cavityOverlapAt_initialReplicas_test]
      congr 3 <;> apply Fin.ext <;> rfl]
    rw [lastSiteAverage_initialReplicas_test]
    simpa [weightedOffdiag, G, mul_comm] using hzero
  rw [hfac, hzero6, mul_zero, sub_zero]
  ring

lemma weightedSourceLinear_target_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h) (i : Fin N)
    (f' : ℝ → ℝ) (k : Fin 3) :
    weightedSourceLinear (s := s) path i f' (targetEdge4_test k) =
      theta q (rsR β h) k / Real.sqrt (N : ℝ) *
        cavityDerivativeAverage (s := s) path i 0 f' := by
  rw [weightedSourceLinear_eq path hN hh hq]
  congr 2
  fin_cases k <;>
    norm_num [targetEdge4_test, scoreInternal_e4_01_test, scoreInternal_e4_02_test,
      scoreInternal_e4_23_test, ee01, ee02, ee23, edgeRelation,
      decoupledSpinCoefficient, theta] <;>
    simp (disch := decide)

noncomputable def weightedFullMoment
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    centeredOverlap q σs target.1.1 target.1.2 * f (fullScaledArgument q σs))

noncomputable def weightedFullVector
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (f : ℝ → ℝ) : Fin 3 → ℝ := fun k =>
  Real.sqrt (N : ℝ) * weightedFullMoment (s := s) path f (targetEdge4_test k)

lemma weightedCavityMoment_endpoint_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f : ℝ → ℝ) (hf : ∀ x, |f x| ≤ M) (hM : 0 ≤ M)
    (e : ReplicaEdge 6) :
    |weightedCavityMoment6 (s := s) path i 1 f e -
        weightedCavityMoment6 (s := s) path i 0 f e| ≤
      72 * β ^ 2 * M *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3)))) := by
  let G : ℝ → ℝ := fun v => weightedCavityMoment6 (s := s) path i v f e
  let D : ℝ → ℝ := fun v => weightedCavityMomentDeriv6 (s := s) path i v f e
  obtain ⟨c, hc, hGc⟩ := exists_hasDerivAt_eq_slope G D (by norm_num)
    ((continuous_lastSiteQuenchedAverage path i
      (fun σs : Replicas N 6 =>
        f (cavityScaledArgument q i σs) *
          cavityOverlapAt q i σs e.1.1 e.1.2)).continuousOn)
    (fun v hv => hasDerivAt_weightedCavityMoment6 path hs hv i f e)
  have hd := abs_weightedCavityMomentDeriv6_le path hs i c f hf hM e
  have hsq := cavitySquare_uniform_le path hN hqI hs i ⟨hc.1.le, hc.2.le⟩
  have hdiff : G 1 - G 0 = D c := by
    dsimp only [G, D] at hGc ⊢
    norm_num at hGc
    linarith
  change |G 1 - G 0| ≤ _
  rw [hdiff]
  exact hd.trans (mul_le_mul_of_nonneg_left hsq (by positivity))

lemma weightedCavityMoment_full_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f : ℝ → ℝ) (hf₀ : ∀ x, |f x| ≤ M)
    (hf₁ : ∀ x y, |f x - f y| ≤ L * |x - y|)
    (hM : 0 ≤ M) (hL : 0 ≤ L) (target : ReplicaEdge 4) :
    |weightedCavityMoment6 (s := s) path i 1 f (scoreInternalEdge_test target) -
        weightedFullMoment (s := s) path f target| ≤
      (M + L) *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
          1 / (N : ℝ)) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  have hsquare : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  have hcav : weightedCavityMoment6 (s := s) path i 1 f
      (scoreInternalEdge_test target) =
      lastSiteQuenchedAverage (s := s) path i 1 (fun σs : Replicas N 4 =>
        f (cavityScaledArgument q i σs) *
          cavityOverlapAt q i σs target.1.1 target.1.2) := by
    unfold weightedCavityMoment6
    rw [show (fun σs : Replicas N 6 =>
        f (cavityScaledArgument q i σs) *
          cavityOverlapAt q i σs (scoreInternalEdge_test target).1.1
            (scoreInternalEdge_test target).1.2) =
        fun σs => (fun τs : Replicas N 4 =>
          f (cavityScaledArgument q i τs) *
            cavityOverlapAt q i τs target.1.1 target.1.2)
              (initialReplicas σs) by
      funext σs
      simp only [scoreInternalEdge_test, cavityScaledArgument, initialReplicas,
        cavityOverlapAt_initialReplicas_test]
      congr 3 <;> apply Fin.ext <;> rfl]
    simpa using lastSiteAverage_initialReplicas_test (s := s) path i 1
      (fun τs : Replicas N 4 =>
        f (cavityScaledArgument q i τs) *
          cavityOverlapAt q i τs target.1.1 target.1.2)
  have hfull : weightedFullMoment (s := s) path f target =
      lastSiteQuenchedAverage (s := s) path i 1 (fun σs : Replicas N 4 =>
        centeredOverlap q σs target.1.1 target.1.2 * f (fullScaledArgument q σs)) := by
    unfold weightedFullMoment lastSiteQuenchedAverage
    congr 1
    funext ω
    rw [lastSiteHamiltonian_one]
  rw [hcav, hfull, ← lastSiteQuenchedAverage_sub_test]
  calc
    _ ≤ lastSiteQuenchedAverage (s := s) path i 1 (fun σs : Replicas N 4 =>
        (M + L) * (cavityOverlapAt q i σs target.1.1 target.1.2 ^ 2 +
          1 / (N : ℝ))) := by
      apply abs_lastSiteAverage_le_test
      intro σs
      let Q := cavityOverlapAt q i σs target.1.1 target.1.2
      let X := fullScaledArgument q σs
      let Y := cavityScaledArgument q i σs
      let ε := spinPairAt_test i target σs
      have hX : |X - Y| ≤ 2 / Real.sqrt (N : ℝ) := by
        dsimp [X, Y]
        rw [fullScaledArgument_eq_cavity_add hN, add_sub_cancel_left, abs_div,
          abs_of_pos hsqrt]
        exact div_le_div_of_nonneg_right
          (abs_centeredSpinPair_le_two_test hqI i e4_01 σs) hsqrt.le
      have hQL : |Q * (f Y - f X)| ≤ L * (Q ^ 2 + 1 / (N : ℝ)) := by
        rw [abs_mul]
        have hfyx := hf₁ Y X
        have hqyoung : 2 * |Q| / Real.sqrt (N : ℝ) ≤ Q ^ 2 + 1 / (N : ℝ) := by
          have hz := sq_nonneg (|Q| - 1 / Real.sqrt (N : ℝ))
          have hinv : (1 / Real.sqrt (N : ℝ)) ^ 2 = 1 / (N : ℝ) := by
            rw [div_pow, hsquare]
            norm_num
          calc
            2 * |Q| / Real.sqrt (N : ℝ) =
                2 * |Q| * (1 / Real.sqrt (N : ℝ)) := by ring
            _ ≤ |Q| ^ 2 + (1 / Real.sqrt (N : ℝ)) ^ 2 := by nlinarith
            _ = Q ^ 2 + 1 / (N : ℝ) := by rw [sq_abs, hinv]
        calc
          |Q| * |f Y - f X| ≤ |Q| * (L * |Y - X|) :=
            mul_le_mul_of_nonneg_left (hf₁ Y X) (abs_nonneg Q)
          _ ≤ |Q| * (L * (2 / Real.sqrt (N : ℝ))) := by
            gcongr
            simpa [abs_sub_comm] using hX
          _ = L * (2 * |Q| / Real.sqrt (N : ℝ)) := by ring
          _ ≤ _ := mul_le_mul_of_nonneg_left hqyoung hL
      have hsite : |(1 / (N : ℝ)) * ε * f X| ≤
          M * (Q ^ 2 + 1 / (N : ℝ)) := by
        rw [abs_mul, abs_mul, abs_spinPairAt_test,
          abs_of_nonneg (by positivity : 0 ≤ 1 / (N : ℝ))]
        norm_num
        have hfX := hf₀ X
        have hbase : (N : ℝ)⁻¹ ≤ Q ^ 2 + (N : ℝ)⁻¹ := by
          nlinarith [sq_nonneg Q]
        calc
          (N : ℝ)⁻¹ * |f X| ≤ (N : ℝ)⁻¹ * M := by gcongr
          _ ≤ (Q ^ 2 + (N : ℝ)⁻¹) * M := by
            exact mul_le_mul_of_nonneg_right hbase hM
          _ = M * (Q ^ 2 + (N : ℝ)⁻¹) := by ring
      have hdecomp : f Y * Q - centeredOverlap q σs target.1.1 target.1.2 * f X =
          Q * (f Y - f X) - (1 / (N : ℝ)) * ε * f X := by
        dsimp [Q, X, Y, ε]
        rw [centeredOverlap_eq_cavityOverlapAt_add q i]
        simp only [spinPairAt_test]
        ring
      rw [hdecomp]
      calc
        _ ≤ |Q * (f Y - f X)| + |(1 / (N : ℝ)) * ε * f X| := abs_sub _ _
        _ ≤ L * (Q ^ 2 + 1 / (N : ℝ)) +
            M * (Q ^ 2 + 1 / (N : ℝ)) := add_le_add hQL hsite
        _ = (M + L) * (Q ^ 2 + 1 / (N : ℝ)) := by ring
    _ = (M + L) * (cavitySquare (s := s) path i 1 + 1 / (N : ℝ)) := by
      let Lm := lastSiteAverageLinearMap_test (n := 4) (s := s) path i 1
      change Lm _ = _
      rw [show (fun σs : Replicas N 4 =>
          (M + L) * (cavityOverlapAt q i σs target.1.1 target.1.2 ^ 2 +
            1 / (N : ℝ))) =
          (M + L) • ((fun σs => cavityOverlapAt q i σs target.1.1 target.1.2 ^ 2) +
            (1 / (N : ℝ)) • (fun _ => 1)) by
        funext σs; simp [smul_eq_mul]]
      simp only [map_smul, map_add, smul_eq_mul, Lm,
        lastSiteAverageLinearMap_apply_test]
      rw [lastSite_square_edge_eq path i 1 target, lastSiteQuenchedAverage_one_test]
      change (M + L) * (cavitySquare (s := s) path i 1 + 1 / (N : ℝ) * 1) = _
      ring
    _ ≤ _ := by
      have hsq := cavitySquare_uniform_le path hN hqI hs i (v := 1) (by norm_num)
      apply mul_le_mul_of_nonneg_left _ (add_nonneg hM hL)
      linarith

noncomputable def cavityDerivativeAverageDeriv
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (v : ℝ) (f' : ℝ → ℝ) : ℝ :=
  s * β ^ 2 * lastSiteQuenchedAverage (s := s) path i v
    (fun σs : Replicas N 6 =>
      f' (cavityScaledArgument q i (initialReplicas σs)) *
        normalizedCavityScoreObservable (n := 4) q i σs)

lemma hasDerivAt_cavityDerivativeAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (hv : v ∈ Set.Ioo (0 : ℝ) 1)
    (i : Fin N) (f' : ℝ → ℝ) :
    HasDerivAt (fun w => cavityDerivativeAverage (s := s) path i w f')
      (cavityDerivativeAverageDeriv (s := s) path i v f') v := by
  let F : ReplicaFun N 4 := fun σs => f' (cavityScaledArgument q i σs)
  have hder := hasDerivAt_lastSiteQuenchedAverage_fixedScore path i hs hv F
  simpa [cavityDerivativeAverage, cavityDerivativeAverageDeriv, F,
    cavityScaledArgument, initialReplicas] using hder

lemma abs_cavityDerivativeAverageDeriv_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N) (v : ℝ)
    (f' : ℝ → ℝ) (hf' : ∀ x, |f' x| ≤ M) (hM : 0 ≤ M) :
    |cavityDerivativeAverageDeriv (s := s) path i v f'| ≤
      32 * β ^ 2 * M * Real.sqrt (N : ℝ) *
        (cavitySquare (s := s) path i v + 1 / (N : ℝ)) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  let P : ReplicaFun N 6 := fun σs =>
    f' (cavityScaledArgument q i (initialReplicas σs))
  have hP : ∀ σs, |P σs| ≤
      (M * Real.sqrt (N : ℝ)) / Real.sqrt (N : ℝ) := by
    intro σs
    rw [mul_div_cancel_right₀ M hsqrt.ne']
    exact hf' _
  have hterm (d : ReplicaEdge 6) :
      |lastSiteQuenchedAverage (s := s) path i v (fun σs =>
        P σs * cavityInteractionAt q i (σs d.1.1) (σs d.1.2))| ≤
      (M * Real.sqrt (N : ℝ) / 2) *
        (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
            1 / (N : ℝ)) := by
    exact abs_small_weighted_cavityInteraction_le_system (s := s) (v := v)
      path hN i P hP (mul_nonneg hM hsqrt.le) d
  have hscore := normalizedScore_average_bound_test
    (M := (M * Real.sqrt (N : ℝ) / 2) *
      (lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
          1 / (N : ℝ))) path i P
    (mul_nonneg (by positivity) (add_nonneg (by
      unfold lastSiteQuenchedAverage quenchedReplicaAverage
      apply integral_nonneg
      intro ω
      apply replicaGibbsAverage_nonneg
      intro σs
      positivity) (by positivity))) hterm
  have hcard : Fintype.card (ReplicaEdge 4) = 6 := by native_decide
  norm_num [hcard] at hscore
  change |lastSiteQuenchedAverage (s := s) path i v
      (fun σs : Replicas N 6 =>
        f' (cavityScaledArgument q i (initialReplicas σs)) *
          normalizedCavityScoreObservable (n := 4) q i σs)| ≤ _ at hscore
  have havg0 : 0 ≤ lastSiteQuenchedAverage (s := s) path i v
      (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) := by
    unfold lastSiteQuenchedAverage quenchedReplicaAverage
    apply integral_nonneg
    intro ω
    apply replicaGibbsAverage_nonneg
    intro σs
    positivity
  have hscore' :
      |lastSiteQuenchedAverage (s := s) path i v
        (fun σs : Replicas N 6 =>
          f' (cavityScaledArgument q i (initialReplicas σs)) *
            normalizedCavityScoreObservable (n := 4) q i σs)| ≤
        32 * ((M * Real.sqrt (N : ℝ) / 2) *
          (lastSiteQuenchedAverage (s := s) path i v
            (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
              1 / (N : ℝ))) := by
    simpa [one_div] using hscore
  unfold cavityDerivativeAverageDeriv
  rw [abs_mul, abs_mul, abs_of_nonneg hs.1, abs_of_nonneg (sq_nonneg β)]
  calc
    s * β ^ 2 * _ ≤ 1 * β ^ 2 *
        (32 * ((M * Real.sqrt (N : ℝ) / 2) *
          (lastSiteQuenchedAverage (s := s) path i v
            (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
              1 / (N : ℝ)))) := by
      gcongr
      exact hs.2
    _ ≤ 32 * β ^ 2 * M * Real.sqrt (N : ℝ) *
        (lastSiteQuenchedAverage (s := s) path i v
          (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
            1 / (N : ℝ)) := by
      have hz : 0 ≤ β ^ 2 * M * Real.sqrt (N : ℝ) *
          (lastSiteQuenchedAverage (s := s) path i v
            (fun σs : Replicas N 6 => cavityOverlapAt q i σs 0 1 ^ 2) +
              1 / (N : ℝ)) := by
        exact mul_nonneg (mul_nonneg (mul_nonneg (sq_nonneg β) hM) hsqrt.le)
          (add_nonneg havg0 (by positivity))
      nlinarith
    _ = _ := by rw [canonicalSquare6_eq]

noncomputable def fullDerivativeAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (f' : ℝ → ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s)
    (fun σs : Replicas N 4 => f' (fullScaledArgument q σs))

lemma cavityDerivativeAverage_full_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f' : ℝ → ℝ) (hf₀ : ∀ x, |f' x| ≤ M)
    (hf₁ : ∀ x y, |f' x - f' y| ≤ L * |x - y|)
    (hM : 0 ≤ M) (hL : 0 ≤ L) :
    |cavityDerivativeAverage (s := s) path i 0 f' -
        fullDerivativeAverage (s := s) path f'| ≤
      32 * β ^ 2 * M * Real.sqrt (N : ℝ) *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
          1 / (N : ℝ)) +
        2 * L / Real.sqrt (N : ℝ) := by
  let G : ℝ → ℝ := fun v => cavityDerivativeAverage (s := s) path i v f'
  let D : ℝ → ℝ := fun v => cavityDerivativeAverageDeriv (s := s) path i v f'
  obtain ⟨c, hc, hGc⟩ := exists_hasDerivAt_eq_slope G D (by norm_num)
    ((continuous_lastSiteQuenchedAverage path i
      (fun σs : Replicas N 4 => f' (cavityScaledArgument q i σs))).continuousOn)
    (fun v hv => hasDerivAt_cavityDerivativeAverage path hs hv i f')
  have hd := abs_cavityDerivativeAverageDeriv_le path hN hs i c f' hf₀ hM
  have hsq := cavitySquare_uniform_le path hN hqI hs i ⟨hc.1.le, hc.2.le⟩
  have hinterp : |G 1 - G 0| ≤
      32 * β ^ 2 * M * Real.sqrt (N : ℝ) *
        (cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
          1 / (N : ℝ)) := by
    have hdiff : G 1 - G 0 = D c := by
      dsimp only [G, D] at hGc ⊢
      norm_num at hGc
      linarith
    rw [hdiff]
    apply hd.trans
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    linarith
  have hfull : fullDerivativeAverage (s := s) path f' =
      lastSiteQuenchedAverage (s := s) path i 1
        (fun σs : Replicas N 4 => f' (fullScaledArgument q σs)) := by
    unfold fullDerivativeAverage lastSiteQuenchedAverage
    congr 1
    funext ω
    rw [lastSiteHamiltonian_one]
  have hpoint : |G 1 - fullDerivativeAverage (s := s) path f'| ≤
      2 * L / Real.sqrt (N : ℝ) := by
    rw [hfull]
    dsimp only [G, cavityDerivativeAverage]
    rw [← lastSiteQuenchedAverage_sub_test]
    calc
      _ ≤ lastSiteQuenchedAverage (s := s) path i 1
          (fun _ : Replicas N 4 => 2 * L / Real.sqrt (N : ℝ)) := by
        apply abs_lastSiteAverage_le_test
        intro σs
        have harg : |cavityScaledArgument q i σs - fullScaledArgument q σs| ≤
            2 / Real.sqrt (N : ℝ) := by
          have hNr : 0 < (N : ℝ) := by positivity
          have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
          rw [abs_sub_comm, fullScaledArgument_eq_cavity_add hN,
            add_sub_cancel_left, abs_div, abs_of_pos hsqrt]
          exact div_le_div_of_nonneg_right
            (abs_centeredSpinPair_le_two_test hqI i e4_01 σs) hsqrt.le
        calc
          _ ≤ L * |cavityScaledArgument q i σs - fullScaledArgument q σs| :=
            hf₁ _ _
          _ ≤ L * (2 / Real.sqrt (N : ℝ)) := by gcongr
          _ = _ := by ring
      _ = _ := by
        rw [show (fun _ : Replicas N 4 => 2 * L / Real.sqrt (N : ℝ)) =
            fun _ => (2 * L / Real.sqrt (N : ℝ)) * 1 by
          funext σs; ring,
          lastSiteQuenchedAverage_const_mul_test,
          lastSiteQuenchedAverage_one_test, mul_one]
  have htri := abs_sub_le (G 0) (G 1) (fullDerivativeAverage (s := s) path f')
  rw [abs_sub_comm (G 0) (G 1)] at htri
  exact htri.trans (add_le_add hinterp hpoint)

lemma quenchedReplicaAverage_add_clt
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (F G : ReplicaFun N n) :
    quenchedReplicaAverage H (F + G) =
      quenchedReplicaAverage H F + quenchedReplicaAverage H G := by
  unfold quenchedReplicaAverage
  rw [← MeasureTheory.integral_add
    (integrable_replicaGibbsAverage_comp H hH F)
    (integrable_replicaGibbsAverage_comp H hH G)]
  apply integral_congr_ae
  filter_upwards with ω
  unfold replicaGibbsAverage
  simp only [Pi.add_apply, mul_add, Finset.sum_add_distrib]

noncomputable def weightedFullSite
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (f : ℝ → ℝ) (target : ReplicaEdge 4) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path s) (fun σs : Replicas N 4 =>
    (spinPairAt_test i target σs - q) * f (fullScaledArgument q σs))

lemma weightedFullSite_decompose
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (i : Fin N) (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    weightedFullSite (s := s) path i f target =
      weightedOffdiag (s := s) path i 1 f target +
        weightedSource (s := s) path i 1 f target := by
  unfold weightedFullSite weightedOffdiag weightedSource lastSiteQuenchedAverage
  rw [show fullPathHamiltonian path s = lastSiteHamiltonian (s := s) path i 1 by
    funext ω; rw [lastSiteHamiltonian_one]]
  rw [← quenchedReplicaAverage_add_clt _
    (measurable_lastSiteHamiltonian path i 1)]
  congr 1
  funext σs
  simp only [Pi.add_apply]
  ring

lemma weightedFullMoment_eq_siteAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (f : ℝ → ℝ) (target : ReplicaEdge 4) :
    weightedFullMoment (s := s) path f target =
      (1 / (N : ℝ)) * ∑ i : Fin N, weightedFullSite (s := s) path i f target := by
  let i₀ : Fin N := ⟨0, hN⟩
  have hmeas : Measurable (fullPathHamiltonian path s) := by
    rw [← show lastSiteHamiltonian (s := s) path i₀ 1 = fullPathHamiltonian path s by
      funext ω; rw [lastSiteHamiltonian_one]]
    exact measurable_lastSiteHamiltonian path i₀ 1
  unfold weightedFullMoment weightedFullSite
  rw [show (fun σs : Replicas N 4 =>
      centeredOverlap q σs target.1.1 target.1.2 * f (fullScaledArgument q σs)) =
      (1 / (N : ℝ)) • (fun σs => ∑ i : Fin N,
        (spinPairAt_test i target σs - q) * f (fullScaledArgument q σs)) by
    funext σs
    rw [centeredOverlap_eq_site_sum_test hN]
    simp only [Pi.smul_apply, smul_eq_mul, spinPairAt_test]
    rw [mul_assoc, Finset.sum_mul]]
  change quenchedReplicaAverage (fullPathHamiltonian path s)
      (fun σs : Replicas N 4 => (1 / (N : ℝ)) * ∑ i : Fin N,
        (spinPairAt_test i target σs - q) * f (fullScaledArgument q σs)) = _
  rw [quenchedReplicaAverage_const_mul,
    quenchedReplicaAverage_sum_test (fullPathHamiltonian path s) hmeas]

lemma weightedFullVector_eq_siteAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (f : ℝ → ℝ) (k : Fin 3) :
    weightedFullVector (s := s) path f k =
      (1 / Real.sqrt (N : ℝ)) * ∑ i : Fin N,
        weightedFullSite (s := s) path i f (targetEdge4_test k) := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrt : Real.sqrt (N : ℝ) ≠ 0 := (Real.sqrt_pos.2 hNr).ne'
  have hsquare : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  unfold weightedFullVector
  rw [weightedFullMoment_eq_siteAverage path hN]
  have hcoef : Real.sqrt (N : ℝ) * (1 / (N : ℝ)) =
      1 / Real.sqrt (N : ℝ) := by
    field_simp [hsqrt, hNr.ne']
    rw [hsquare]
  rw [← mul_assoc, hcoef]

noncomputable def cltCavityScale
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q) : ℝ :=
  cavityVector path s 0 + thirdMoment path s +
    3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
    32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
      (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
    1 / (N : ℝ)

lemma weightedCavityMoment_zero_full_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hqI : q ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f : ℝ → ℝ) (hf₀ : ∀ x, |f x| ≤ M)
    (hf₁ : ∀ x y, |f x - f y| ≤ L * |x - y|)
    (hM : 0 ≤ M) (hL : 0 ≤ L) (target : ReplicaEdge 4) :
    |weightedCavityMoment6 (s := s) path i 0 f (scoreInternalEdge_test target) -
        weightedFullMoment (s := s) path f target| ≤
      (72 * β ^ 2 * M + M + L) * cltCavityScale (s := s) path := by
  have h₁ := weightedCavityMoment_endpoint_le path hN hqI hs i f hf₀ hM
    (scoreInternalEdge_test target)
  have h₂ := weightedCavityMoment_full_le path hN hqI hs i f hf₀ hf₁
    hM hL target
  have htri := abs_sub_le
    (weightedCavityMoment6 (s := s) path i 0 f (scoreInternalEdge_test target))
    (weightedCavityMoment6 (s := s) path i 1 f (scoreInternalEdge_test target))
    (weightedFullMoment (s := s) path f target)
  rw [abs_sub_comm] at h₁
  have h₁' :
      |weightedCavityMoment6 (s := s) path i 0 f (scoreInternalEdge_test target) -
          weightedCavityMoment6 (s := s) path i 1 f (scoreInternalEdge_test target)| ≤
        72 * β ^ 2 * M *
          (cltCavityScale (s := s) path - 1 / (N : ℝ)) := by
    simpa [cltCavityScale] using h₁
  apply htri.trans
  calc
    _ ≤ 72 * β ^ 2 * M *
          (cltCavityScale (s := s) path - 1 / (N : ℝ)) +
        (M + L) * cltCavityScale (s := s) path := add_le_add h₁' h₂
    _ ≤ (72 * β ^ 2 * M + M + L) * cltCavityScale (s := s) path := by
      have hscale : 0 ≤ cltCavityScale (s := s) path := by
        unfold cltCavityScale
        have hA : 0 ≤ cavityVector path s 0 := by
          unfold cavityVector A quenchedReplicaAverage
          apply integral_nonneg
          intro ω
          apply replicaGibbsAverage_nonneg
          intro σs
          positivity
        have hthird := thirdMoment_nonneg path s
        positivity
      have hinv : 0 ≤ 1 / (N : ℝ) := by positivity
      have hpref : 0 ≤ 72 * β ^ 2 * M := by positivity
      nlinarith

lemma abs_theta_le_two (q r : ℝ) (hq : q ∈ Set.Icc (0 : ℝ) 1)
    (hr : r ∈ Set.Icc (0 : ℝ) 1) (k : Fin 3) : |theta q r k| ≤ 2 := by
  have hq2 : 0 ≤ q ^ 2 := sq_nonneg q
  have hq2le : q ^ 2 ≤ 1 := by
    nlinarith [mul_nonneg hq.1 (sub_nonneg.mpr hq.2)]
  fin_cases k <;> simp [theta, abs_le] <;>
    constructor <;> nlinarith [hr.1, hr.2]

lemma weightedSource_full_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M₁ L₁ L₂ : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h)
    (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hrI : rsR β h ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f f' : ℝ → ℝ)
    (hf₁ : ∀ x y, |f x - f y| ≤ L₁ * |x - y|)
    (hfTaylor : ∀ x y, |f x - f y - f' y * (x - y)| ≤ L₂ * (x - y) ^ 2)
    (hf'₀ : ∀ x, |f' x| ≤ M₁)
    (hf'₁ : ∀ x y, |f' x - f' y| ≤ L₂ * |x - y|)
    (hM₁ : 0 ≤ M₁) (hL₁ : 0 ≤ L₁) (hL₂ : 0 ≤ L₂) (k : Fin 3) :
    |weightedSource (s := s) path i 1 f (targetEdge4_test k) -
        theta q (rsR β h) k / Real.sqrt (N : ℝ) *
          fullDerivativeAverage (s := s) path f'| ≤
      256 * β ^ 2 * L₁ * cltCavityScale (s := s) path +
        8 * L₂ / (N : ℝ) +
        (2 / Real.sqrt (N : ℝ)) *
          (32 * β ^ 2 * M₁ * Real.sqrt (N : ℝ) *
              cltCavityScale (s := s) path +
            2 * L₂ / Real.sqrt (N : ℝ)) := by
  have h₁ := weightedSource_endpoint_le path hN hqI hs i f hf₁ hL₁
    (targetEdge4_test k)
  have h₂ := weightedSource_zero_taylor_le (s := s) path hN hqI i f f' hfTaylor hL₂
    (targetEdge4_test k)
  have hlin := weightedSourceLinear_target_eq (s := s) path hN hh hq i f' k
  have h₃ := cavityDerivativeAverage_full_le path hN hqI hs i f' hf'₀ hf'₁
    hM₁ hL₂
  have htheta := abs_theta_le_two q (rsR β h) hqI hrI k
  have hscaleform :
      cavityVector path s 0 + thirdMoment path s +
          3 * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          32 * β ^ 2 * (Real.exp (64 * β ^ 2) *
            (4 * (thirdMoment path s + (1 / (N : ℝ)) ^ 3))) +
          1 / (N : ℝ) = cltCavityScale (s := s) path := rfl
  rw [hscaleform] at h₁ h₃
  rw [hlin] at h₂
  have hcoef :
      |theta q (rsR β h) k / Real.sqrt (N : ℝ) *
        (cavityDerivativeAverage (s := s) path i 0 f' -
          fullDerivativeAverage (s := s) path f')| ≤
        (2 / Real.sqrt (N : ℝ)) *
          (32 * β ^ 2 * M₁ * Real.sqrt (N : ℝ) *
              cltCavityScale (s := s) path +
            2 * L₂ / Real.sqrt (N : ℝ)) := by
    have hsqrt : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 (by positivity)
    rw [abs_mul, abs_div, abs_of_pos hsqrt]
    have hdiv : |theta q (rsR β h) k| / Real.sqrt (N : ℝ) ≤
        2 / Real.sqrt (N : ℝ) :=
      div_le_div_of_nonneg_right htheta hsqrt.le
    exact mul_le_mul hdiv h₃ (by positivity) (by positivity)
  have htri := abs_sub_le
    (weightedSource (s := s) path i 1 f (targetEdge4_test k))
    (weightedSource (s := s) path i 0 f (targetEdge4_test k))
    (theta q (rsR β h) k / Real.sqrt (N : ℝ) *
      fullDerivativeAverage (s := s) path f')
  have htri₂ := abs_sub_le
    (weightedSource (s := s) path i 0 f (targetEdge4_test k))
    (theta q (rsR β h) k / Real.sqrt (N : ℝ) *
      cavityDerivativeAverage (s := s) path i 0 f')
    (theta q (rsR β h) k / Real.sqrt (N : ℝ) *
      fullDerivativeAverage (s := s) path f')
  have heq :
      theta q (rsR β h) k / Real.sqrt (N : ℝ) *
          cavityDerivativeAverage (s := s) path i 0 f' -
        theta q (rsR β h) k / Real.sqrt (N : ℝ) *
          fullDerivativeAverage (s := s) path f' =
      theta q (rsR β h) k / Real.sqrt (N : ℝ) *
        (cavityDerivativeAverage (s := s) path i 0 f' -
          fullDerivativeAverage (s := s) path f') := by ring
  rw [heq] at htri₂
  linarith

lemma weightedOffdiag_full_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M L : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h)
    (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hrI : rsR β h ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f : ℝ → ℝ) (hf₀ : ∀ x, |f x| ≤ M)
    (hf₁ : ∀ x y, |f x - f y| ≤ L * |x - y|)
    (hM : 0 ≤ M) (hL : 0 ≤ L) (k : Fin 3) :
    |weightedOffdiag (s := s) path i 1 f (targetEdge4_test k) -
        s * (cavityMatrix β q (rsR β h)).mulVec
          (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ)| ≤
      4608 * β ^ 4 * M *
          (cltCavityScale (s := s) path - 1 / (N : ℝ)) +
        60 * β ^ 2 * (72 * β ^ 2 * M + M + L) *
          cltCavityScale (s := s) path := by
  have htay := weightedOffdiag_taylor_le path hN hqI hs hh hq i f hf₀ hM
    (targetEdge4_test k)
  have hder := weightedOffdiagDeriv_zero_eq_matrix (s := s) path hN hh hq i f k
  let x : Fin 3 → ℝ := fun j =>
    weightedCavityVectorAt (s := s) path i 0 f j -
      weightedFullMoment (s := s) path f (targetEdge4_test j)
  have hx (j : Fin 3) : |x j| ≤
      (72 * β ^ 2 * M + M + L) * cltCavityScale (s := s) path := by
    have hj := weightedCavityMoment_zero_full_le path hN hqI hs i f hf₀ hf₁
      hM hL (targetEdge4_test j)
    fin_cases j
    · change |weightedCavityMoment6 (s := s) path i 0 f ee01 -
          weightedFullMoment (s := s) path f (targetEdge4_test 0)| ≤ _
      rw [← scoreInternal_e4_01_test]
      exact hj
    · change |weightedCavityMoment6 (s := s) path i 0 f ee02 -
          weightedFullMoment (s := s) path f (targetEdge4_test 1)| ≤ _
      rw [← scoreInternal_e4_02_test]
      exact hj
    · change |weightedCavityMoment6 (s := s) path i 0 f ee23 -
          weightedFullMoment (s := s) path f (targetEdge4_test 2)| ≤ _
      rw [← scoreInternal_e4_23_test]
      exact hj
  have hscale : 0 ≤ cltCavityScale (s := s) path := by
    unfold cltCavityScale
    have hA : 0 ≤ cavityVector path s 0 := by
      unfold cavityVector A quenchedReplicaAverage
      apply integral_nonneg
      intro ω
      apply replicaGibbsAverage_nonneg
      intro σs
      positivity
    have hthird := thirdMoment_nonneg path s
    positivity
  have hR : 0 ≤ (72 * β ^ 2 * M + M + L) *
      cltCavityScale (s := s) path := by positivity
  have hMx := abs_cavityMatrix_mulVec_le_test β q (rsR β h)
    ((72 * β ^ 2 * M + M + L) * cltCavityScale (s := s) path)
    hqI hrI hR x hx k
  have hsqrt : Real.sqrt (N : ℝ) ≠ 0 := (Real.sqrt_pos.2 (by positivity)).ne'
  have heq : weightedOffdiagDeriv (s := s) path i 0 f (targetEdge4_test k) -
      s * (cavityMatrix β q (rsR β h)).mulVec
          (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ) =
      s * (cavityMatrix β q (rsR β h)).mulVec x k := by
    rw [hder]
    fin_cases k <;>
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, x,
        weightedCavityVectorAt, weightedFullVector] <;>
      field_simp [hsqrt] <;> ring
  have hmid :
      |weightedOffdiagDeriv (s := s) path i 0 f (targetEdge4_test k) -
        s * (cavityMatrix β q (rsR β h)).mulVec
          (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ)| ≤
        60 * β ^ 2 * ((72 * β ^ 2 * M + M + L) *
          cltCavityScale (s := s) path) := by
    rw [heq, abs_mul, abs_of_nonneg hs.1]
    calc
      s * |(cavityMatrix β q (rsR β h)).mulVec x k| ≤
          1 * |(cavityMatrix β q (rsR β h)).mulVec x k| := by
        gcongr
        exact hs.2
      _ ≤ _ := by simpa using hMx
  have htri := abs_sub_le
    (weightedOffdiag (s := s) path i 1 f (targetEdge4_test k))
    (weightedOffdiagDeriv (s := s) path i 0 f (targetEdge4_test k))
    (s * (cavityMatrix β q (rsR β h)).mulVec
      (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ))
  have htay' :
      |weightedOffdiag (s := s) path i 1 f (targetEdge4_test k) -
          weightedOffdiagDeriv (s := s) path i 0 f (targetEdge4_test k)| ≤
        4608 * β ^ 4 * M *
          (cltCavityScale (s := s) path - 1 / (N : ℝ)) := by
    simpa [cltCavityScale] using htay
  linarith

noncomputable def cltLocalErrorBound
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (M₀ L₁ M₁ L₂ : ℝ) : ℝ :=
  4608 * β ^ 4 * M₀ *
      (cltCavityScale (s := s) path - 1 / (N : ℝ)) +
    60 * β ^ 2 * (72 * β ^ 2 * M₀ + M₀ + L₁) *
      cltCavityScale (s := s) path +
    (256 * β ^ 2 * L₁ * cltCavityScale (s := s) path +
      8 * L₂ / (N : ℝ) +
      (2 / Real.sqrt (N : ℝ)) *
        (32 * β ^ 2 * M₁ * Real.sqrt (N : ℝ) *
            cltCavityScale (s := s) path +
          2 * L₂ / Real.sqrt (N : ℝ)))

lemma weightedFullSite_system_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M₀ L₁ M₁ L₂ : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h)
    (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hrI : rsR β h ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (i : Fin N)
    (f f' : ℝ → ℝ)
    (hf₀ : ∀ x, |f x| ≤ M₀)
    (hf₁ : ∀ x y, |f x - f y| ≤ L₁ * |x - y|)
    (hfTaylor : ∀ x y, |f x - f y - f' y * (x - y)| ≤ L₂ * (x - y) ^ 2)
    (hf'₀ : ∀ x, |f' x| ≤ M₁)
    (hf'₁ : ∀ x y, |f' x - f' y| ≤ L₂ * |x - y|)
    (hM₀ : 0 ≤ M₀) (hL₁ : 0 ≤ L₁)
    (hM₁ : 0 ≤ M₁) (hL₂ : 0 ≤ L₂) (k : Fin 3) :
    |weightedFullSite (s := s) path i f (targetEdge4_test k) -
        s * (cavityMatrix β q (rsR β h)).mulVec
            (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ) -
        theta q (rsR β h) k / Real.sqrt (N : ℝ) *
          fullDerivativeAverage (s := s) path f'| ≤
      cltLocalErrorBound (s := s) path M₀ L₁ M₁ L₂ := by
  have hoff := weightedOffdiag_full_le path hN hh hq hqI hrI hs i f hf₀ hf₁
    hM₀ hL₁ k
  have hsource := weightedSource_full_le path hN hh hq hqI hrI hs i f f' hf₁
    hfTaylor hf'₀ hf'₁ hM₁ hL₁ hL₂ k
  rw [weightedFullSite_decompose path i f (targetEdge4_test k)]
  have htri := abs_add_le
    (weightedOffdiag (s := s) path i 1 f (targetEdge4_test k) -
      s * (cavityMatrix β q (rsR β h)).mulVec
        (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ))
    (weightedSource (s := s) path i 1 f (targetEdge4_test k) -
      theta q (rsR β h) k / Real.sqrt (N : ℝ) *
        fullDerivativeAverage (s := s) path f')
  have heq :
      weightedOffdiag (s := s) path i 1 f (targetEdge4_test k) +
          weightedSource (s := s) path i 1 f (targetEdge4_test k) -
          s * (cavityMatrix β q (rsR β h)).mulVec
            (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ) -
          theta q (rsR β h) k / Real.sqrt (N : ℝ) *
            fullDerivativeAverage (s := s) path f' =
        (weightedOffdiag (s := s) path i 1 f (targetEdge4_test k) -
          s * (cavityMatrix β q (rsR β h)).mulVec
            (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ)) +
        (weightedSource (s := s) path i 1 f (targetEdge4_test k) -
          theta q (rsR β h) k / Real.sqrt (N : ℝ) *
            fullDerivativeAverage (s := s) path f') := by ring
  rw [heq]
  unfold cltLocalErrorBound
  exact htri.trans (add_le_add hoff hsource)

lemma weightedFullVector_system_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s M₀ L₁ M₁ L₂ : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (hh : 0 < h) (hq : q = rsQ β h)
    (hqI : q ∈ Set.Icc (0 : ℝ) 1) (hrI : rsR β h ∈ Set.Icc (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (f f' : ℝ → ℝ)
    (hf₀ : ∀ x, |f x| ≤ M₀)
    (hf₁ : ∀ x y, |f x - f y| ≤ L₁ * |x - y|)
    (hfTaylor : ∀ x y, |f x - f y - f' y * (x - y)| ≤ L₂ * (x - y) ^ 2)
    (hf'₀ : ∀ x, |f' x| ≤ M₁)
    (hf'₁ : ∀ x y, |f' x - f' y| ≤ L₂ * |x - y|)
    (hM₀ : 0 ≤ M₀) (hL₁ : 0 ≤ L₁)
    (hM₁ : 0 ≤ M₁) (hL₂ : 0 ≤ L₂) (k : Fin 3) :
    |weightedFullVector (s := s) path f k -
        s * (cavityMatrix β q (rsR β h)).mulVec
          (weightedFullVector (s := s) path f) k -
        theta q (rsR β h) k * fullDerivativeAverage (s := s) path f'| ≤
      Real.sqrt (N : ℝ) * cltLocalErrorBound (s := s) path M₀ L₁ M₁ L₂ := by
  have hNr : 0 < (N : ℝ) := by positivity
  have hsqrtpos : 0 < Real.sqrt (N : ℝ) := Real.sqrt_pos.2 hNr
  have hsqrt : Real.sqrt (N : ℝ) ≠ 0 := hsqrtpos.ne'
  have hsquare : Real.sqrt (N : ℝ) ^ 2 = (N : ℝ) := Real.sq_sqrt hNr.le
  let R : Fin N → ℝ := fun i =>
    weightedFullSite (s := s) path i f (targetEdge4_test k) -
      s * (cavityMatrix β q (rsR β h)).mulVec
          (weightedFullVector (s := s) path f) k / Real.sqrt (N : ℝ) -
      theta q (rsR β h) k / Real.sqrt (N : ℝ) *
        fullDerivativeAverage (s := s) path f'
  have hR (i : Fin N) : |R i| ≤
      cltLocalErrorBound (s := s) path M₀ L₁ M₁ L₂ :=
    weightedFullSite_system_le path hN hh hq hqI hrI hs i f f' hf₀ hf₁
      hfTaylor hf'₀ hf'₁ hM₀ hL₁ hM₁ hL₂ k
  have heq :
      weightedFullVector (s := s) path f k -
          s * (cavityMatrix β q (rsR β h)).mulVec
            (weightedFullVector (s := s) path f) k -
          theta q (rsR β h) k * fullDerivativeAverage (s := s) path f' =
        (1 / Real.sqrt (N : ℝ)) * ∑ i, R i := by
    rw [weightedFullVector_eq_siteAverage path hN]
    simp only [R, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    field_simp [hsqrt, hNr.ne']
    ring_nf
    rw [hsquare]
    ring
  rw [heq, abs_mul, abs_of_nonneg (by positivity : 0 ≤ 1 / Real.sqrt (N : ℝ))]
  calc
    (1 / Real.sqrt (N : ℝ)) * |∑ i, R i| ≤
        (1 / Real.sqrt (N : ℝ)) * ∑ i, |R i| := by
      gcongr
      exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ (1 / Real.sqrt (N : ℝ)) *
        ∑ _i : Fin N, cltLocalErrorBound (s := s) path M₀ L₁ M₁ L₂ := by
      gcongr with i
      exact hR i
    _ = Real.sqrt (N : ℝ) *
        cltLocalErrorBound (s := s) path M₀ L₁ M₁ L₂ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp [hsqrt, hNr.ne']
      ring_nf
      rw [hsquare]
      ring

end CLT
end SpinGlass.AT
