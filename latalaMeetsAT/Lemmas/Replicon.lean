import Lemmas.FreeEnergy

open MeasureTheory ProbabilityTheory Filter

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem thirdMoment_littleO {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    [HasFixedDeviationEstimate Ω] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (Crem : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data Crem) :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      N * thirdMoment path s < eps := by
  intro eps heps
  obtain ⟨M, hM, hsecond⟩ := uniform_secondMoment (Ω := Ω) data Crem hCavity
  let eta : ℝ := eps / (4 * (M + 1))
  have hM1 : 0 < M + 1 := by linarith
  have heta : 0 < eta := by
    dsimp [eta]
    positivity
  have hetaM : eta * M < eps / 4 := by
    dsimp [eta]
    rw [div_mul_eq_mul_div]
    apply (div_lt_iff₀ (by positivity : 0 < 4 * (M + 1))).2
    nlinarith
  obtain ⟨c, C, hc, hC, htail⟩ := fixedDeviation (Ω := Ω) data eta heta
  have hdecayReal : Tendsto
      (fun x : ℝ => 8 * C * (x ^ (1 : ℝ) * Real.exp (-c * x)))
      atTop (nhds 0) := by
    simpa using
      (tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero 1 c hc).const_mul (8 * C)
  have hdecay : Tendsto
      (fun N : ℕ => 8 * C * (N : ℝ) * Real.exp (-c * (N : ℝ)))
      atTop (nhds 0) := by
    simpa [Function.comp_def, Real.rpow_one, mul_assoc] using
      hdecayReal.comp tendsto_natCast_atTop_atTop
  obtain ⟨Ntail, hNtail⟩ := (Metric.tendsto_atTop.1 hdecay) (eps / 2) (by positivity)
  refine ⟨max 1 Ntail, ?_⟩
  intro N hN β h q s hp hq hs path
  have hNpos : 0 < N := lt_of_lt_of_le (by omega : 0 < 1) (le_trans (le_max_left _ _) hN)
  subst q
  have hqmem : rsQ β h ∈ Set.Icc (0 : ℝ) 1 := rsQ_mem_Icc β h
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hsplit : thirdMoment path s ≤
      eta * A path s + 8 * quenchedTail path s eta := by
    unfold thirdMoment A quenchedTail
    rw [← quenchedReplicaAverage_const_mul, ← quenchedReplicaAverage_const_mul,
      ← quenchedReplicaAverage_add hfull]
    apply quenchedReplicaAverage_mono hfull
    intro σs
    let X := centeredOverlap (rsQ β h) σs (0 : Fin 4) (1 : Fin 4)
    have hX : |X| ≤ 2 := abs_centeredOverlap_le_two hNpos hqmem σs 0 1
    change |X| ^ 3 ≤ eta * X ^ 2 + 8 * (if eta ≤ |X| then 1 else 0)
    exact abs_cube_le_epsilon_sq_add_indicator heta.le hX
  have hsecondN : (N : ℝ) * A path s ≤ M :=
    hsecond hNpos hp rfl hs path
  have htailN : quenchedTail path s eta ≤ C * Real.exp (-c * (N : ℝ)) :=
    htail path hp rfl hs
  have hsplitN := mul_le_mul_of_nonneg_left hsplit (Nat.cast_nonneg N)
  have htailN' := mul_le_mul_of_nonneg_left htailN (Nat.cast_nonneg N)
  have hdecayN := hNtail N (le_trans (le_max_right _ _) hN)
  rw [Real.dist_eq, sub_zero,
    abs_of_nonneg (by positivity : 0 ≤ 8 * C * (N : ℝ) * Real.exp (-c * (N : ℝ)))]
    at hdecayN
  nlinarith

theorem replicon_susceptibility {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    [HasFixedDeviationEstimate Ω] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (Crem : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data Crem) :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      |N * (A path s - 2 * B path s + C path s) -
        rsA β h / (1 - s * atParameter β h)| < eps := by
  -- Paper route, equations (repliconeq)--(repliconrho): take `vecMul` of the
  -- cavity system with `ell = ![1,-2,1]`.  Use
  -- `replicon_leftEigenvector` and `ell dot theta = rsA` to obtain
  -- `(1-s*atParameter)*(A-2*B+C) = rsA/N + ell dot remainder`.
  -- The remainder bound and `thirdMoment_littleO` imply that `N` times the
  -- final error tends uniformly to zero.  Divide by
  -- `1-s*atParameter`, whose lower bound is `data.gap` by `path_gap`, and
  -- rearrange.  Translate the uniform little-o estimate into the requested
  -- `eps,N0` quantifiers exactly as in the final paragraph of the paper.
  intro eps heps
  let δ : ℝ := eps * data.gap / (8 * (Crem + 1))
  have hδ : 0 < δ := by
    dsimp [δ]
    have hCrem : 0 < Crem := hCavity.1
    exact div_pos (mul_pos heps data.gap_pos)
      (mul_pos (by norm_num) (by linarith))
  obtain ⟨Nthird, hNthird⟩ :=
    thirdMoment_littleO (Ω := Ω) data Crem hCavity δ hδ
  have hpowlim : Tendsto
      (fun N : ℕ => (N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2))
      atTop (nhds 0) := by
    have hbase : Tendsto (fun N : ℕ => (N : ℝ) ^ (-(1 : ℝ) / 2))
        atTop (nhds 0) := by
      convert
        (tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 1 / 2)).comp
          tendsto_natCast_atTop_atTop using 1
      all_goals norm_num [Function.comp_def]
    refine hbase.congr' ?_
    filter_upwards [eventually_gt_atTop 0] with N hN
    calc
      (N : ℝ) ^ (-(1 : ℝ) / 2) =
          (N : ℝ) ^ ((1 : ℝ) + (-(3 : ℝ) / 2)) := by norm_num
      _ = (N : ℝ) ^ (1 : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) :=
        Real.rpow_add (Nat.cast_pos.mpr hN) _ _
      _ = (N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) := by rw [Real.rpow_one]
  obtain ⟨Npow, hNpow⟩ := (Metric.tendsto_atTop.1 hpowlim) δ hδ
  refine ⟨max 1 (max Nthird Npow), ?_⟩
  intro N hN β h q s hp hq hs path
  have hNpos : 0 < N :=
    lt_of_lt_of_le (by omega : 0 < 1) (le_trans (le_max_left _ _) hN)
  have hthird : (N : ℝ) * thirdMoment path s < δ :=
    hNthird (le_trans (le_max_left _ _) (le_trans (le_max_right _ _) hN))
      hp hq hs path
  have hpow : (N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) < δ := by
    have hdist :=
      hNpow N (le_trans (le_max_right _ _) (le_trans (le_max_right _ _) hN))
    rw [Real.dist_eq, sub_zero, abs_mul,
      abs_of_nonneg (Nat.cast_nonneg N),
      abs_of_nonneg (Real.rpow_nonneg (Nat.cast_nonneg N) _)] at hdist
    exact hdist
  subst q
  let R : ℝ := cavityRemainder path s 0 - 2 * cavityRemainder path s 1 +
    cavityRemainder path s 2
  have hidentity :
      (1 - s * atParameter β h) * (A path s - 2 * B path s + C path s) =
        (1 / (N : ℝ)) * rsA β h + R := by
    have hsys := cavity_system (s := s) path
    have h0 := congrFun hsys 0
    have h1 := congrFun hsys 1
    have h2 := congrFun hsys 2
    simp [cavityVector, cavityMatrix, theta] at h0 h1 h2
    dsimp [R]
    unfold atParameter rsA
    linear_combination h0 - 2 * h1 + h2
  have hRcoord : |R| ≤ 4 * ‖cavityRemainder path s‖ := by
    have h0 : |cavityRemainder path s 0| ≤ ‖cavityRemainder path s‖ := by
      simpa [Real.norm_eq_abs] using norm_le_pi_norm (cavityRemainder path s) 0
    have h1 : |cavityRemainder path s 1| ≤ ‖cavityRemainder path s‖ := by
      simpa [Real.norm_eq_abs] using norm_le_pi_norm (cavityRemainder path s) 1
    have h2 : |cavityRemainder path s 2| ≤ ‖cavityRemainder path s‖ := by
      simpa [Real.norm_eq_abs] using norm_le_pi_norm (cavityRemainder path s) 2
    dsimp [R]
    calc
      |cavityRemainder path s 0 - 2 * cavityRemainder path s 1 +
          cavityRemainder path s 2| ≤
          |cavityRemainder path s 0| + |2 * cavityRemainder path s 1| +
            |cavityRemainder path s 2| := by
        calc
          |_ - 2 * _ + _| ≤ |_ - 2 * _| + |_| := abs_add_le _ _
          _ ≤ (|cavityRemainder path s 0| + |2 * cavityRemainder path s 1|) +
              |cavityRemainder path s 2| := by
                gcongr
                simpa [sub_eq_add_neg] using
                  abs_add_le (cavityRemainder path s 0)
                    (-(2 * cavityRemainder path s 1))
      _ = |cavityRemainder path s 0| + 2 * |cavityRemainder path s 1| +
          |cavityRemainder path s 2| := by rw [abs_mul]; norm_num
      _ ≤ 4 * ‖cavityRemainder path s‖ := by linarith
  have hRbound : |R| ≤ 4 * Crem *
      ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s) := by
    have hr := cavityRemainder_bound hCavity hNpos hp rfl hs path
    nlinarith [norm_nonneg (cavityRemainder path s)]
  have hNR : (N : ℝ) * |R| < eps * data.gap := by
    have hmul := mul_le_mul_of_nonneg_left hRbound (Nat.cast_nonneg N)
    have hCrem : 0 < Crem := hCavity.1
    have hCrem1 : 0 < Crem + 1 := by linarith
    calc
      (N : ℝ) * |R| ≤ (N : ℝ) *
          (4 * Crem * ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s)) := hmul
      _ = 4 * Crem * ((N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          (N : ℝ) * thirdMoment path s) := by ring
      _ < 4 * Crem * (δ + δ) := by
        gcongr
      _ = eps * data.gap * (Crem / (Crem + 1)) := by
        dsimp [δ]
        field_simp [ne_of_gt hCrem1]
        ring
      _ < eps * data.gap := by
        calc
          eps * data.gap * (Crem / (Crem + 1)) <
              eps * data.gap * 1 :=
            mul_lt_mul_of_pos_left
              ((div_lt_one hCrem1).2 (by linarith))
              (mul_pos heps data.gap_pos)
          _ = eps * data.gap := by ring
  have hgap : data.gap ≤ 1 - s * atParameter β h := path_gap data hp hs
  have hden : 0 < 1 - s * atParameter β h := lt_of_lt_of_le data.gap_pos hgap
  have hNne : (N : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hNpos
  have hidN := hidentity
  field_simp [hNne] at hidN
  have herr :
      (N : ℝ) * (A path s - 2 * B path s + C path s) -
          rsA β h / (1 - s * atParameter β h) =
        (N : ℝ) * R / (1 - s * atParameter β h) := by
    field_simp [ne_of_gt hden]
    linear_combination hidN
  rw [herr, abs_div, abs_mul, abs_of_nonneg (Nat.cast_nonneg N), abs_of_pos hden]
  apply (div_lt_iff₀ hden).2
  exact hNR.trans_le (mul_le_mul_of_nonneg_left hgap (le_of_lt heps))

end SpinGlass.AT
