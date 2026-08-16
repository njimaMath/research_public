import Lemmas.Cavity.Stability
import Lemmas.Cavity.System
import Lemmas.FixedDeviation

open ProbabilityTheory MeasureTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Deterministic truncation inequality behind cubic absorption.  Probability
enters only after this pointwise estimate is averaged. -/
theorem abs_cube_le_epsilon_sq_add_indicator {x eps : ℝ}
    (heps : 0 ≤ eps) (hx : |x| ≤ 2) :
    |x| ^ 3 ≤ eps * x ^ 2 + 8 * (if eps ≤ |x| then 1 else 0) := by
  have hxsq : 0 ≤ x ^ 2 := sq_nonneg x
  by_cases hlarge : eps ≤ |x|
  · simp only [if_pos hlarge, mul_one]
    have hcub : |x| ^ 3 ≤ 8 := by
      nlinarith [abs_nonneg x, sq_nonneg |x|,
        mul_self_le_mul_self (abs_nonneg x) hx]
    nlinarith [mul_nonneg heps hxsq]
  · simp only [if_neg hlarge, mul_zero, add_zero]
    have hsmall : |x| ≤ eps := le_of_not_ge hlarge
    rw [show |x| ^ 3 = |x| * x ^ 2 by rw [pow_succ, sq_abs]; ring]
    exact mul_le_mul_of_nonneg_right hsmall hxsq

theorem cavityVector_abs_A_le_norm {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    |A path s| ≤ ‖cavityVector path s‖ := by
  rw [Pi.norm_def]
  have hsup := Finset.le_sup (s := Finset.univ)
    (f := fun b : Fin 3 => ‖cavityVector path s b‖₊) (Finset.mem_univ (0 : Fin 3))
  exact_mod_cast hsup

theorem A_nonneg {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : 0 ≤ A path s := by
  unfold A
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hmono := quenchedReplicaAverage_mono (H := fullPathHamiltonian path s) hfull
    (F := fun _ : Replicas N 4 => 0)
    (G := fun σs => centeredOverlap q σs 0 1 ^ 2)
    (fun σs => sq_nonneg _)
  have hzero : quenchedReplicaAverage (fullPathHamiltonian path s)
      (fun _ : Replicas N 4 => 0) = 0 := by
    simpa using quenchedReplicaAverage_const_mul (fullPathHamiltonian path s)
      0 (fun _ : Replicas N 4 => 1)
  simpa [hzero] using hmono

theorem abs_B_le_A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : |B path s| ≤ A path s := by
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hsame : quenchedReplicaAverage (fullPathHamiltonian path s)
      (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2) =
      quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun σs : Replicas N 4 => centeredOverlap q σs 0 2 ^ 2) := by
    have hp := quenchedReplicaAverage_perm (fullPathHamiltonian path s)
      (Equiv.swap (1 : Fin 4) 2)
      (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2)
    have hperm0 : (Equiv.swap (1 : Fin 4) 2) 0 = 0 := by decide
    have hperm1 : (Equiv.swap (1 : Fin 4) 2) 1 = 2 := by decide
    simpa [centeredOverlap, overlap, hperm0, hperm1] using hp.symm
  simpa [A, B] using mixed_overlap_abs_le_secondMoment
    (fullPathHamiltonian path s) hfull q (0 : Fin 4) 1 0 2 hsame

theorem abs_C_le_A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) : |C path s| ≤ A path s := by
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  let perm : Equiv.Perm (Fin 4) :=
    (Equiv.swap (0 : Fin 4) 2).trans (Equiv.swap (1 : Fin 4) 3)
  have hsame : quenchedReplicaAverage (fullPathHamiltonian path s)
      (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2) =
      quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun σs : Replicas N 4 => centeredOverlap q σs 2 3 ^ 2) := by
    have hp := quenchedReplicaAverage_perm (fullPathHamiltonian path s) perm
      (fun σs : Replicas N 4 => centeredOverlap q σs 0 1 ^ 2)
    have hperm0 : perm 0 = 2 := by native_decide
    have hperm1 : perm 1 = 3 := by native_decide
    simpa [centeredOverlap, overlap, hperm0, hperm1] using hp.symm
  simpa [A, C] using mixed_overlap_abs_le_secondMoment
    (fullPathHamiltonian path s) hfull q (0 : Fin 4) 1 2 3 hsame

/-- The infinity norm of the cavity moment vector is exactly its second
moment coordinate. -/
theorem cavityVector_norm_eq_A {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q s : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) :
    ‖cavityVector path s‖ = A path s := by
  apply le_antisymm
  · rw [pi_norm_le_iff_of_nonneg (A_nonneg path)]
    intro i
    fin_cases i
    · simp [cavityVector, Real.norm_eq_abs, abs_of_nonneg (A_nonneg path)]
    · simpa [cavityVector, Real.norm_eq_abs] using abs_B_le_A path
    · simpa [cavityVector, Real.norm_eq_abs] using abs_C_le_A path
  · have hcoord := cavityVector_abs_A_le_norm (s := s) path
    rw [abs_of_nonneg (A_nonneg path)] at hcoord
    exact hcoord

theorem uniform_secondMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) (C : ℝ)
    (hCavity : HasCavityRemainderBound (Ω := Ω) data C) :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q, N * A path s ≤ M := by
  obtain ⟨L, hLpos, hL⟩ := cavityMatrix_inverse_uniform data
  have hCpos : 0 < C := hCavity.1
  let eta : ℝ := 1 / (6 * L * C)
  have heta : 0 < eta := by
    dsimp [eta]
    positivity
  obtain ⟨c, D, hc, hD, htail⟩ := fixedDeviation (Ω := Ω) data eta heta
  let M : ℝ := 6 * L + 6 * L * C + 48 * L * C * D / c
  refine ⟨M, ?_, ?_⟩
  · dsimp [M]
    positivity
  intro N hN β h q s hp hq hs path
  subst q
  have hNreal : (0 : ℝ) < N := by exact_mod_cast hN
  have hqmem : rsQ β h ∈ Set.Icc (0 : ℝ) 1 := rsQ_mem_Icc β h
  have hβ : 0 < β := data.β_pos (β, h) hp
  have hh : 0 < h := data.h_pos (β, h) hp
  have hr0 : 0 ≤ rsR β h := by
    dsimp [rsR, standardGaussianExpectation]
    exact integral_nonneg fun z => by positivity
  have hrq : rsR β h ≤ rsQ β h := rsR_le_rsQ hβ hh
  let S := stabilityOperator β (rsQ β h) (rsR β h) s
  let V := cavityVector path s
  let R := cavityRemainder path s
  let b : Fin 3 → ℝ :=
    (1 / (N : ℝ)) • theta (rsQ β h) (rsR β h) + R
  have hrep : 0 < 1 - s * atParameter β h :=
    lt_of_lt_of_le data.gap_pos (path_gap data hp hs)
  have hanom : 0 < 1 - s * β ^ 2 *
      (1 - 4 * rsQ β h + 3 * rsR β h) := by
    have hle := anomalous_eigenvalue_le_replicon hβ hh
    have hgap := path_gap data hp hs
    nlinarith [mul_nonneg hs.1 (sub_nonneg.mpr hle)]
  have hdet : 0 < Matrix.det S := by
    rw [show Matrix.det S =
        (1 - s * atParameter β h) *
          (1 - s * β ^ 2 * (1 - 4 * rsQ β h + 3 * rsR β h)) ^ 2 by
      simpa [S] using cavityMatrix_determinant (β := β) (h := h) (s := s)]
    exact mul_pos hrep (sq_pos_of_pos hanom)
  have hunit : IsUnit (Matrix.det S) :=
    isUnit_iff_ne_zero.mpr (ne_of_gt hdet)
  have hsys : S.mulVec V = b := by
    dsimp [S, V, b, R]
    simpa [stabilityOperator, Matrix.sub_mulVec, Matrix.smul_mulVec] using
      (cavity_system (s := s) path)
  have hsolve : V = S⁻¹.mulVec b := by
    calc
      V = (1 : Matrix (Fin 3) (Fin 3) ℝ).mulVec V := (Matrix.one_mulVec V).symm
      _ = (S⁻¹ * S).mulVec V := by rw [Matrix.nonsing_inv_mul S hunit]
      _ = S⁻¹.mulVec (S.mulVec V) := (Matrix.mulVec_mulVec V S⁻¹ S).symm
      _ = S⁻¹.mulVec b := by rw [hsys]
  have htheta : ∀ j : Fin 3, |theta (rsQ β h) (rsR β h) j| ≤ 1 := by
    intro j
    fin_cases j <;> simp [theta, abs_le] <;>
      constructor <;> nlinarith [sq_nonneg (rsQ β h),
        mul_self_le_mul_self hqmem.1 hqmem.2]
  have hRcoord : ∀ j : Fin 3, |R j| ≤ ‖R‖ := by
    intro j
    simpa [Real.norm_eq_abs] using norm_le_pi_norm R j
  have hbcoord : ∀ j : Fin 3, |b j| ≤ 1 / (N : ℝ) + ‖R‖ := by
    intro j
    dsimp [b]
    calc
      |(1 / (N : ℝ)) * theta (rsQ β h) (rsR β h) j + R j| ≤
          |(1 / (N : ℝ)) * theta (rsQ β h) (rsR β h) j| + |R j| :=
        abs_add_le _ _
      _ = (1 / (N : ℝ)) * |theta (rsQ β h) (rsR β h) j| + |R j| := by
        rw [abs_mul, abs_of_pos (one_div_pos.mpr hNreal)]
      _ ≤ 1 / (N : ℝ) + ‖R‖ := by
        exact add_le_add
          (by simpa using
            (mul_le_mul_of_nonneg_left (htheta j) (one_div_nonneg.mpr hNreal.le)))
          (hRcoord j)
  have hAabs : |A path s| ≤ 3 * L * (1 / (N : ℝ) + ‖R‖) := by
    have hzero := congrFun hsolve (0 : Fin 3)
    change A path s = S⁻¹.mulVec b 0 at hzero
    rw [hzero]
    rw [Matrix.mulVec, dotProduct]
    calc
      |∑ j, S⁻¹ 0 j * b j| ≤ ∑ j, |S⁻¹ 0 j * b j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _j : Fin 3, L * (1 / (N : ℝ) + ‖R‖) := by
        apply Finset.sum_le_sum
        intro j _
        rw [abs_mul]
        exact mul_le_mul (hL hp hs 0 j) (hbcoord j) (abs_nonneg _) hLpos.le
      _ = 3 * L * (1 / (N : ℝ) + ‖R‖) := by simp; ring
  have hfull : Measurable (fullPathHamiltonian path s) := by
    apply measurable_pi_iff.mpr
    intro σ
    exact ((measurable_pi_iff.mp (path.measurable s)) σ).add measurable_const
  have hA0 : 0 ≤ A path s := by
    unfold A
    have hmono := quenchedReplicaAverage_mono (H := fullPathHamiltonian path s) hfull
      (F := fun _ : Replicas N 4 => 0)
      (G := fun σs => centeredOverlap (rsQ β h) σs 0 1 ^ 2)
      (fun σs => sq_nonneg _)
    have hzero : quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun _ : Replicas N 4 => 0) = 0 := by
      simpa using quenchedReplicaAverage_const_mul (fullPathHamiltonian path s)
        0 (fun _ : Replicas N 4 => 1)
    rw [hzero] at hmono
    exact hmono
  have hsplit : thirdMoment path s ≤
      eta * A path s + 8 * quenchedTail path s eta := by
    unfold thirdMoment A quenchedTail
    rw [← quenchedReplicaAverage_const_mul, ← quenchedReplicaAverage_const_mul,
      ← quenchedReplicaAverage_add hfull]
    apply quenchedReplicaAverage_mono hfull
    intro σs
    let X := centeredOverlap (rsQ β h) σs (0 : Fin 4) (1 : Fin 4)
    have hX : |X| ≤ 2 := abs_centeredOverlap_le_two hN hqmem σs 0 1
    change |X| ^ 3 ≤ eta * X ^ 2 + 8 * (if eta ≤ |X| then 1 else 0)
    exact abs_cube_le_epsilon_sq_add_indicator heta.le hX
  have hrem := cavityRemainder_bound hCavity hN hp rfl hs path
  change ‖R‖ ≤ C * ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s) at hrem
  have htail' : quenchedTail path s eta ≤ D * Real.exp (-c * (N : ℝ)) :=
    htail path hp rfl hs
  have hthird' : thirdMoment path s ≤
      eta * A path s + 8 * (D * Real.exp (-c * (N : ℝ))) := by
    exact hsplit.trans (add_le_add_right
      (mul_le_mul_of_nonneg_left htail' (by norm_num : (0 : ℝ) ≤ 8))
      (eta * A path s))
  have hpre : A path s ≤ 3 * L *
      (1 / (N : ℝ) + C * ((N : ℝ) ^ (-(3 : ℝ) / 2) +
        eta * A path s + 8 * (D * Real.exp (-c * (N : ℝ))))) := by
    calc
      A path s ≤ |A path s| := le_abs_self _
      _ ≤ 3 * L * (1 / (N : ℝ) + ‖R‖) := hAabs
      _ ≤ 3 * L * (1 / (N : ℝ) + C *
          ((N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s)) := by
        exact mul_le_mul_of_nonneg_left
          (add_le_add_right hrem (1 / (N : ℝ)))
          (mul_nonneg (by norm_num) hLpos.le)
      _ ≤ _ := by
        have hin := add_le_add_right hthird' ((N : ℝ) ^ (-(3 : ℝ) / 2))
        have hcineq := mul_le_mul_of_nonneg_left hin hCpos.le
        have hplus := add_le_add_right hcineq (1 / (N : ℝ))
        exact mul_le_mul_of_nonneg_left
          (by simpa [add_assoc] using hplus)
          (mul_nonneg (by norm_num) hLpos.le)
  have heta_id : 6 * L * C * eta = 1 := by
    dsimp [eta]
    field_simp
  have habsorb : A path s ≤
      6 * L / (N : ℝ) + 6 * L * C * (N : ℝ) ^ (-(3 : ℝ) / 2) +
        48 * L * C * D * Real.exp (-c * (N : ℝ)) := by
    have hpre' : A path s ≤
        3 * L / (N : ℝ) +
        3 * L * C * (N : ℝ) ^ (-(3 : ℝ) / 2) +
        (3 * L * C * eta) * A path s +
        24 * L * C * D * Real.exp (-c * (N : ℝ)) := by
      calc
        A path s ≤ 3 * L *
            (1 / (N : ℝ) + C * ((N : ℝ) ^ (-(3 : ℝ) / 2) +
              eta * A path s + 8 * (D * Real.exp (-c * (N : ℝ))))) := hpre
        _ = _ := by ring
    have hcoeff : 3 * L * C * eta = 1 / 2 := by nlinarith [heta_id]
    rw [hcoeff] at hpre'
    let T : ℝ := 3 * L / (N : ℝ) +
      3 * L * C * (N : ℝ) ^ (-(3 : ℝ) / 2) +
      24 * L * C * D * Real.exp (-c * (N : ℝ))
    have hpreT : A path s ≤ T + (1 / 2 : ℝ) * A path s := by
      simpa [T, add_assoc, add_left_comm, add_comm] using hpre'
    have hAT : A path s ≤ 2 * T := by linarith
    calc
      A path s ≤ 2 * T := hAT
      _ = _ := by dsimp [T]; ring
  have hpow : (N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) ≤ 1 := by
    calc
      (N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2) =
          (N : ℝ) ^ ((1 : ℝ) + (-(3 : ℝ) / 2)) := by
        rw [Real.rpow_add hNreal]
        simp
      _ = (N : ℝ) ^ (-(1 : ℝ) / 2) := by norm_num
      _ ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos (by exact_mod_cast hN) (by norm_num)
  have hexp : (N : ℝ) * Real.exp (-c * (N : ℝ)) ≤ 1 / c := by
    have hx : 0 < c * (N : ℝ) := mul_pos hc hNreal
    have hxexp : c * (N : ℝ) ≤ Real.exp (c * (N : ℝ)) :=
      le_trans (by linarith) (Real.add_one_le_exp _)
    have hratio : (c * (N : ℝ)) * Real.exp (-c * (N : ℝ)) ≤ 1 := by
      rw [show -c * (N : ℝ) = -(c * (N : ℝ)) by ring, Real.exp_neg]
      rw [mul_inv_le_iff₀ (Real.exp_pos _)]
      simpa using hxexp
    calc
      (N : ℝ) * Real.exp (-c * (N : ℝ)) =
          (1 / c) * ((c * (N : ℝ)) * Real.exp (-c * (N : ℝ))) := by
        field_simp [ne_of_gt hc]
      _ ≤ (1 / c) * 1 := mul_le_mul_of_nonneg_left hratio (by positivity)
      _ = 1 / c := mul_one _
  have hmul := mul_le_mul_of_nonneg_left habsorb (Nat.cast_nonneg N)
  calc
    (N : ℝ) * A path s ≤
        (N : ℝ) * (6 * L / (N : ℝ) +
          6 * L * C * (N : ℝ) ^ (-(3 : ℝ) / 2) +
          48 * L * C * D * Real.exp (-c * (N : ℝ))) := hmul
    _ = 6 * L + 6 * L * C *
          ((N : ℝ) * (N : ℝ) ^ (-(3 : ℝ) / 2)) +
        48 * L * C * D * ((N : ℝ) * Real.exp (-c * (N : ℝ))) := by
      field_simp [ne_of_gt hNreal]
    _ ≤ 6 * L + 6 * L * C * 1 + 48 * L * C * D * (1 / c) := by
      gcongr
    _ = M := by
      dsimp [M]
      ring

end SpinGlass.AT
