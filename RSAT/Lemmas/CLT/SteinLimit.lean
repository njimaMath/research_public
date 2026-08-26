import Lemmas.CLT.SteinSystem
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

open MeasureTheory ProbabilityTheory Real BigOperators Filter
open scoped Topology

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace SpinGlass.AT
namespace CLT

universe u

noncomputable def cltVariance (β h : ℝ) : ℝ :=
  3 * rsA β h / (1 - atParameter β h) -
    2 * cavityKappa (rsQ β h) (rsR β h) /
      (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) -
    cavityZeta (rsQ β h) (rsR β h) /
      (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) ^ 2

noncomputable def systemResidual (β q r m : ℝ) (x : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun k => x k - (cavityMatrix β q r).mulVec x k - theta q r k * m

private lemma scalar_mode_algebra
    (b k z a dK dA m V U D : ℝ) (hdK : dK ≠ 0) (hdA : dA ≠ 0)
    (hK : dK = 1 - b * k) :
    (-V - 2 * U + 3 * D) -
        (3 * a / dA - 2 * k / dK - z / dK ^ 2) * m =
      -(dK * V - b * z * U - z * m) / dK -
        (b * z / dK ^ 2 + 2 / dK) * (dK * U - k * m) +
        3 / dA * (dA * D - a * m) := by
  field_simp [hdK, hdA]
  rw [hK]
  ring

lemma systemResidual_scalar_identity
    {β h q m : ℝ} (hq : q = rsQ β h) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (x : Fin 3 → ℝ) :
    let dκ := 1 - β ^ 2 * cavityKappa q (rsR β h)
    let dA := 1 - atParameter β h
    let R := systemResidual β q (rsR β h) m x
    x 0 - cltVariance β h * m =
      -(2 * R 1 - 3 * R 2) / dκ -
        (β ^ 2 * cavityZeta q (rsR β h) / dκ ^ 2 + 2 / dκ) *
          (R 0 - 4 * R 1 + 3 * R 2) +
        3 / dA * (R 0 - 2 * R 1 + R 2) := by
  subst q
  have hκLe :
      β ^ 2 * cavityKappa (rsQ β h) (rsR β h) ≤ atParameter β h := by
    rw [atParameter_eq_beta_sq_mul_one_sub_two_q_add_r]
    simp only [cavityKappa]
    have hrq : rsR β h ≤ rsQ β h := rsR_le_rsQ hh
    exact mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg β)
  have hdκ : 1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h) ≠ 0 := by
    linarith
  have hdA : 1 - atParameter β h ≠ 0 := by linarith
  let R := systemResidual β (rsQ β h) (rsR β h) m x
  have hV :
      2 * R 1 - 3 * R 2 =
        (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) *
            (2 * x 1 - 3 * x 2) -
          β ^ 2 * cavityZeta (rsQ β h) (rsR β h) *
            (x 0 - 4 * x 1 + 3 * x 2) -
          cavityZeta (rsQ β h) (rsR β h) * m := by
    simp [R, systemResidual, cavityMatrix, theta, Matrix.mulVec, dotProduct,
      Fin.sum_univ_succ, cavityKappa, cavityZeta]
    ring
  have hU :
      R 0 - 4 * R 1 + 3 * R 2 =
        (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)) *
            (x 0 - 4 * x 1 + 3 * x 2) -
          cavityKappa (rsQ β h) (rsR β h) * m := by
    simp [R, systemResidual, cavityMatrix, theta, Matrix.mulVec, dotProduct,
      Fin.sum_univ_succ, cavityKappa]
    ring
  have hD :
      R 0 - 2 * R 1 + R 2 =
        (1 - atParameter β h) * (x 0 - 2 * x 1 + x 2) - rsA β h * m := by
    simp [R, systemResidual, cavityMatrix, theta, Matrix.mulVec, dotProduct,
      Fin.sum_univ_succ, atParameter, rsA]
    ring
  dsimp only
  change x 0 - cltVariance β h * m = _
  change _ = -(2 * R 1 - 3 * R 2) / _ - _ *
      (R 0 - 4 * R 1 + 3 * R 2) + _ * (R 0 - 2 * R 1 + R 2)
  rw [hV, hU, hD]
  unfold cltVariance
  have halg := scalar_mode_algebra
    (β ^ 2) (cavityKappa (rsQ β h) (rsR β h))
    (cavityZeta (rsQ β h) (rsR β h)) (rsA β h)
    (1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h))
    (1 - atParameter β h) m
    (2 * x 1 - 3 * x 2) (x 0 - 4 * x 1 + 3 * x 2)
    (x 0 - 2 * x 1 + x 2) hdκ hdA rfl
  linarith

lemma tendsto_sqrt_mul_cltCavityScale_zero
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h)) :
    Tendsto (fun N : ℕ =>
        Real.sqrt (N.succ : ℝ) * cltCavityScale (s := 1) (paths N))
      atTop (nhds 0) := by
  let K : Set (ℝ × ℝ) := {(β, h)}
  let data : UniformATData K := singletonUniformATData hβ hh hAT
  have hp : (β, h) ∈ K := by simp [K]
  obtain ⟨M, hM, hsecond⟩ :=
    (quantitative_strictAT (Ω := Ω) K data).secondMoment
  have hsuccTop : Tendsto (fun N : ℕ => N.succ) atTop atTop := by
    rw [tendsto_atTop]
    intro b
    exact eventually_atTop.2 ⟨b, fun a ha => le_trans ha (Nat.le_succ a)⟩
  have hnTop : Tendsto (fun N : ℕ => (N.succ : ℝ)) atTop atTop :=
    tendsto_natCast_atTop_atTop.comp hsuccTop
  have hsqrtTop : Tendsto (fun N : ℕ => Real.sqrt (N.succ : ℝ)) atTop atTop :=
    Real.tendsto_sqrt_atTop.comp hnTop
  have hinvS : Tendsto (fun N : ℕ => 1 / Real.sqrt (N.succ : ℝ)) atTop (nhds 0) := by
    convert hsqrtTop.inv_tendsto_atTop using 1
    ext N
    simp [one_div]
  have hA0 (N : ℕ) : 0 ≤ cavityVector (paths N) 1 0 := by
    unfold cavityVector A quenchedReplicaAverage
    apply integral_nonneg
    intro ω
    apply replicaGibbsAverage_nonneg
    intro σs
    positivity
  have hAbound (N : ℕ) :
      (N.succ : ℝ) * cavityVector (paths N) 1 0 ≤ M := by
    simpa [cavityVector] using
      hsecond (Nat.succ_pos N) hp rfl (by simp) (paths N)
  have hSA : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * cavityVector (paths N) 1 0) atTop (nhds 0) := by
    apply squeeze_zero
    · intro N
      exact mul_nonneg (Real.sqrt_nonneg _) (hA0 N)
    · intro N
      have hn : 0 < (N.succ : ℝ) := by positivity
      have hs : 0 < Real.sqrt (N.succ : ℝ) := Real.sqrt_pos.2 hn
      have hs2 : Real.sqrt (N.succ : ℝ) ^ 2 = (N.succ : ℝ) :=
        Real.sq_sqrt hn.le
      have hb := hAbound N
      calc
        Real.sqrt (N.succ : ℝ) * cavityVector (paths N) 1 0 =
            ((N.succ : ℝ) * cavityVector (paths N) 1 0) /
              Real.sqrt (N.succ : ℝ) := by
                field_simp [hs.ne']
                nlinarith
        _ ≤ M / Real.sqrt (N.succ : ℝ) := div_le_div_of_nonneg_right hb hs.le
    · simpa [div_eq_mul_inv, one_div] using hinvS.const_mul M
  have hthird := tendsto_scaled_thirdMoment_zero hβ hh hAT paths
  have hST : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * thirdMoment (paths N) 1) atTop (nhds 0) := by
    have hpdt := hthird.mul hinvS
    convert hpdt using 1
    · funext N
      have hn : 0 < (N.succ : ℝ) := by positivity
      have hs : 0 < Real.sqrt (N.succ : ℝ) := Real.sqrt_pos.2 hn
      have hs2 : Real.sqrt (N.succ : ℝ) ^ 2 = (N.succ : ℝ) :=
        Real.sq_sqrt hn.le
      field_simp [hs.ne']
      rw [hs2]
      ring
    · ring
  have hRP : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * (N.succ : ℝ) ^ (-(3 : ℝ) / 2))
      atTop (nhds 0) := by
    have hbase := (tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 1)).comp hnTop
    apply hbase.congr'
    filter_upwards with N
    have hn : 0 < (N.succ : ℝ) := by positivity
    change (N.succ : ℝ) ^ (-(1 : ℝ)) =
      Real.sqrt (N.succ : ℝ) * (N.succ : ℝ) ^ (-(3 : ℝ) / 2)
    symm
    rw [Real.sqrt_eq_rpow, ← Real.rpow_add hn]
    congr 1
    norm_num
  have hCube : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * (1 / (N.succ : ℝ)) ^ 3)
      atTop (nhds 0) := by
    have hbase := (tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 5 / 2)).comp hnTop
    apply hbase.congr'
    filter_upwards with N
    have hn : 0 < (N.succ : ℝ) := by positivity
    have hpow : (1 / (N.succ : ℝ)) ^ 3 =
        (N.succ : ℝ) ^ (-(3 : ℝ)) := by
      rw [Real.rpow_neg hn.le]
      field_simp [hn.ne']
      exact Real.rpow_natCast _ 3
    change (N.succ : ℝ) ^ (-(5 / 2 : ℝ)) =
      Real.sqrt (N.succ : ℝ) * (1 / (N.succ : ℝ)) ^ 3
    symm
    rw [Real.sqrt_eq_rpow, hpow, ← Real.rpow_add hn]
    congr 1
    norm_num
  have hInv : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * (1 / (N.succ : ℝ))) atTop (nhds 0) := by
    apply Tendsto.congr' _ hinvS
    filter_upwards with N
    have hn : 0 < (N.succ : ℝ) := by positivity
    have hs : 0 < Real.sqrt (N.succ : ℝ) := Real.sqrt_pos.2 hn
    have hs2 : Real.sqrt (N.succ : ℝ) ^ 2 = (N.succ : ℝ) :=
      Real.sq_sqrt hn.le
    field_simp [hs.ne']
    rw [hs2]
  have htotal := hSA.add hST |>.add (hRP.const_mul 3) |>.add
    ((hST.add hCube).const_mul
      (32 * β ^ 2 * Real.exp (64 * β ^ 2) * 4)) |>.add hInv
  convert htotal using 1
  · funext N
    unfold cltCavityScale
    ring
  · ring

lemma tendsto_sqrt_mul_cltLocalErrorBound_zero
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h))
    (M₀ L₁ M₁ L₂ : ℝ) :
    Tendsto (fun N : ℕ => Real.sqrt (N.succ : ℝ) *
        cltLocalErrorBound (s := 1) (paths N) M₀ L₁ M₁ L₂)
      atTop (nhds 0) := by
  have hscale := tendsto_sqrt_mul_cltCavityScale_zero hβ hh hAT paths
  have hsuccTop : Tendsto (fun N : ℕ => N.succ) atTop atTop := by
    rw [tendsto_atTop]
    intro b
    exact eventually_atTop.2 ⟨b, fun a ha => le_trans ha (Nat.le_succ a)⟩
  have hnTop : Tendsto (fun N : ℕ => (N.succ : ℝ)) atTop atTop :=
    tendsto_natCast_atTop_atTop.comp hsuccTop
  have hinvS : Tendsto (fun N : ℕ => 1 / Real.sqrt (N.succ : ℝ)) atTop (nhds 0) := by
    have hs := (Real.tendsto_sqrt_atTop.comp hnTop).inv_tendsto_atTop
    convert hs using 1
    ext N
    simp [one_div]
  have hInv : Tendsto (fun N : ℕ =>
      Real.sqrt (N.succ : ℝ) * (1 / (N.succ : ℝ))) atTop (nhds 0) := by
    apply hinvS.congr'
    filter_upwards with N
    have hn : 0 < (N.succ : ℝ) := by positivity
    have hs : 0 < Real.sqrt (N.succ : ℝ) := Real.sqrt_pos.2 hn
    have hs2 := Real.sq_sqrt hn.le
    field_simp [hs.ne']
    rw [hs2]
  have htotal :=
    ((hscale.sub hInv).const_mul (4608 * β ^ 4 * M₀)).add
      (hscale.const_mul (60 * β ^ 2 * (72 * β ^ 2 * M₀ + M₀ + L₁))) |>.add
      (hscale.const_mul (256 * β ^ 2 * L₁)) |>.add
      (hInv.const_mul (8 * L₂)) |>.add
      (hscale.const_mul (64 * β ^ 2 * M₁)) |>.add
      (hinvS.const_mul (4 * L₂))
  convert htotal using 1
  · funext N
    have hn : 0 < (N.succ : ℝ) := by positivity
    have hs : 0 < Real.sqrt (N.succ : ℝ) := Real.sqrt_pos.2 hn
    have hs2 : Real.sqrt (N.succ : ℝ) ^ 2 = (N.succ : ℝ) :=
      Real.sq_sqrt hn.le
    unfold cltLocalErrorBound
    field_simp [hs.ne', hn.ne']
    rw [hs2]
    ring
  · ring

private lemma abs_linear_three_le
    {a₀ a₁ a₂ r₀ r₁ r₂ δ : ℝ} (hδ : 0 ≤ δ)
    (h₀ : |r₀| ≤ δ) (h₁ : |r₁| ≤ δ) (h₂ : |r₂| ≤ δ) :
    |a₀ * r₀ + a₁ * r₁ + a₂ * r₂| ≤
      (|a₀| + |a₁| + |a₂|) * δ := by
  calc
    |a₀ * r₀ + a₁ * r₁ + a₂ * r₂| ≤
        |a₀ * r₀| + |a₁ * r₁| + |a₂ * r₂| := by
          exact (abs_add_le (a₀ * r₀ + a₁ * r₁) (a₂ * r₂)).trans
            (add_le_add (abs_add_le _ _) le_rfl)
    _ = |a₀| * |r₀| + |a₁| * |r₁| + |a₂| * |r₂| := by
      simp only [abs_mul]
    _ ≤ |a₀| * δ + |a₁| * δ + |a₂| * δ := by gcongr
    _ = (|a₀| + |a₁| + |a₂|) * δ := by ring

noncomputable def scalarResidualConstant (β h : ℝ) : ℝ :=
  let dκ := 1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)
  let dA := 1 - atParameter β h
  let c := β ^ 2 * cavityZeta (rsQ β h) (rsR β h) / dκ ^ 2 + 2 / dκ
  |-c + 3 / dA| + |-2 / dκ + 4 * c - 6 / dA| +
    |3 / dκ - 3 * c + 3 / dA|

lemma systemResidual_scalar_le
    {β h m δ : ℝ} (hh : 0 < h) (hAT : atParameter β h < 1)
    (x : Fin 3 → ℝ) (hδ : 0 ≤ δ)
    (hR : ∀ k, |systemResidual β (rsQ β h) (rsR β h) m x k| ≤ δ) :
    |x 0 - cltVariance β h * m| ≤ scalarResidualConstant β h * δ := by
  let R := systemResidual β (rsQ β h) (rsR β h) m x
  let dκ := 1 - β ^ 2 * cavityKappa (rsQ β h) (rsR β h)
  let dA := 1 - atParameter β h
  let c := β ^ 2 * cavityZeta (rsQ β h) (rsR β h) / dκ ^ 2 + 2 / dκ
  have hid := systemResidual_scalar_identity (q := rsQ β h) (m := m)
    rfl hh hAT x
  dsimp only at hid
  change |x 0 - cltVariance β h * m| ≤ _
  rw [hid]
  have heq :
      -(2 * R 1 - 3 * R 2) / dκ - c * (R 0 - 4 * R 1 + 3 * R 2) +
          3 / dA * (R 0 - 2 * R 1 + R 2) =
        (-c + 3 / dA) * R 0 +
          (-2 / dκ + 4 * c - 6 / dA) * R 1 +
          (3 / dκ - 3 * c + 3 / dA) * R 2 := by ring
  rw [heq]
  change _ ≤ (|-c + 3 / dA| + |-2 / dκ + 4 * c - 6 / dA| +
    |3 / dκ - 3 * c + 3 / dA|) * δ
  apply abs_linear_three_le hδ
  · exact hR 0
  · exact hR 1
  · exact hR 2

lemma weightedFullVector_scalar_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h M₀ L₁ M₁ L₂ : ℝ}
    (path : RSSmartPathDisorder Ω N β h (rsQ β h))
    (hN : 0 < N) (hh : 0 < h) (hAT : atParameter β h < 1)
    (f f' : ℝ → ℝ)
    (hf₀ : ∀ x, |f x| ≤ M₀)
    (hf₁ : ∀ x y, |f x - f y| ≤ L₁ * |x - y|)
    (hfTaylor : ∀ x y, |f x - f y - f' y * (x - y)| ≤ L₂ * (x - y) ^ 2)
    (hf'₀ : ∀ x, |f' x| ≤ M₁)
    (hf'₁ : ∀ x y, |f' x - f' y| ≤ L₂ * |x - y|)
    (hM₀ : 0 ≤ M₀) (hL₁ : 0 ≤ L₁)
    (hM₁ : 0 ≤ M₁) (hL₂ : 0 ≤ L₂) :
    |weightedFullVector (s := 1) path f 0 -
        cltVariance β h * fullDerivativeAverage (s := 1) path f'| ≤
      scalarResidualConstant β h * (Real.sqrt (N : ℝ) *
        cltLocalErrorBound (s := 1) path M₀ L₁ M₁ L₂) := by
  let δ := Real.sqrt (N : ℝ) * cltLocalErrorBound (s := 1) path M₀ L₁ M₁ L₂
  let x := weightedFullVector (s := 1) path f
  let m := fullDerivativeAverage (s := 1) path f'
  have hsys (k : Fin 3) : |systemResidual β (rsQ β h) (rsR β h) m x k| ≤ δ := by
    simpa [systemResidual, x, m, δ] using weightedFullVector_system_le (s := 1) path hN hh rfl
      (rsQ_mem_Icc β h) ⟨rsR_nonneg β h, (rsR_le_rsQ hh).trans (rsQ_mem_Icc β h).2⟩
      (by simp) f f' hf₀ hf₁ hfTaylor hf'₀ hf'₁ hM₀ hL₁ hM₁ hL₂ k
  have hδ : 0 ≤ δ := (abs_nonneg _).trans (hsys 0)
  exact systemResidual_scalar_le hh hAT x hδ hsys

lemma first_order_remainder_le_of_deriv_lipschitz
    {f f' : ℝ → ℝ} {L : ℝ}
    (hf : ∀ z, HasDerivAt f (f' z) z)
    (hf' : ∀ z w, |f' z - f' w| ≤ L * |z - w|)
    (hL : 0 ≤ L)
    (x y : ℝ) :
    |f x - f y - f' y * (x - y)| ≤ L * (x - y) ^ 2 := by
  let g : ℝ → ℝ := fun z => f z - f y - (z - y) * f' y
  have hg (z : ℝ) : HasDerivAt g (f' z - f' y) z := by
    change HasDerivAt ((fun w => f w - f y) - fun w => (w - y) * f' y)
      (f' z - f' y) z
    have H := (hf z).sub_const (f y) |>.sub
      ((hasDerivAt_id z).sub_const y |>.mul_const (f' y))
    simpa only [one_mul, Function.id_def] using H
  have hbound (z : ℝ) (hz : z ∈ Set.uIcc y x) :
      ‖deriv g z‖ ≤ L * |x - y| := by
    rw [(hg z).deriv, Real.norm_eq_abs]
    exact (hf' z y).trans (mul_le_mul_of_nonneg_left
      (by simpa [abs_sub_comm] using Set.abs_sub_left_of_mem_uIcc hz) hL)
  have hmv := (convex_uIcc y x).norm_image_sub_le_of_norm_deriv_le
    (fun z _ => (hg z).differentiableAt) hbound Set.left_mem_uIcc Set.right_mem_uIcc
  dsimp [g] at hmv
  rw [show (x - y) ^ 2 = |x - y| * |x - y| by
    nlinarith [sq_abs (x - y)]]
  simpa [mul_comm, mul_left_comm, mul_assoc] using hmv

lemma sin_test_bounds (u : ℝ) :
    let f : ℝ → ℝ := fun x => Real.sin (u * x)
    let f' : ℝ → ℝ := fun x => u * Real.cos (u * x)
    (∀ x, |f x| ≤ 1) ∧
    (∀ x y, |f x - f y| ≤ |u| * |x - y|) ∧
    (∀ x y, |f x - f y - f' y * (x - y)| ≤ u ^ 2 * (x - y) ^ 2) ∧
    (∀ x, |f' x| ≤ |u|) ∧
    (∀ x y, |f' x - f' y| ≤ u ^ 2 * |x - y|) := by
  dsimp only
  have hderiv (x : ℝ) : HasDerivAt (fun z => Real.sin (u * z))
      (u * Real.cos (u * x)) x := by
    change HasDerivAt (Real.sin ∘ fun z => u * z) _ x
    have H := Real.hasDerivAt_sin (u * x) |>.comp x
      ((hasDerivAt_id x).const_mul u)
    simpa only [one_mul, mul_one, mul_comm] using H
  have hlip (x y : ℝ) :
      |u * Real.cos (u * x) - u * Real.cos (u * y)| ≤ u ^ 2 * |x - y| := by
    rw [← mul_sub, abs_mul]
    calc
      |u| * |Real.cos (u * x) - Real.cos (u * y)| ≤
          |u| * |u * x - u * y| := by
            exact mul_le_mul_of_nonneg_left (Real.abs_cos_sub_cos_le _ _) (abs_nonneg u)
      _ = u ^ 2 * |x - y| := by
        rw [← mul_sub, abs_mul]
        calc
          |u| * (|u| * |x - y|) = |u| ^ 2 * |x - y| := by ring
          _ = u ^ 2 * |x - y| := by rw [sq_abs]
  constructor
  · exact fun x => Real.abs_sin_le_one _
  constructor
  · intro x y
    calc
      |Real.sin (u * x) - Real.sin (u * y)| ≤ |u * x - u * y| :=
        Real.abs_sin_sub_sin_le _ _
      _ = |u| * |x - y| := by rw [← mul_sub, abs_mul]
  constructor
  · exact first_order_remainder_le_of_deriv_lipschitz hderiv hlip (sq_nonneg u)
  constructor
  · intro x
    rw [abs_mul]
    exact mul_le_of_le_one_right (abs_nonneg u) (Real.abs_cos_le_one _)
  · exact hlip

lemma cos_test_bounds (u : ℝ) :
    let f : ℝ → ℝ := fun x => Real.cos (u * x)
    let f' : ℝ → ℝ := fun x => -u * Real.sin (u * x)
    (∀ x, |f x| ≤ 1) ∧
    (∀ x y, |f x - f y| ≤ |u| * |x - y|) ∧
    (∀ x y, |f x - f y - f' y * (x - y)| ≤ u ^ 2 * (x - y) ^ 2) ∧
    (∀ x, |f' x| ≤ |u|) ∧
    (∀ x y, |f' x - f' y| ≤ u ^ 2 * |x - y|) := by
  dsimp only
  have hderiv (x : ℝ) : HasDerivAt (fun z => Real.cos (u * z))
      (-u * Real.sin (u * x)) x := by
    change HasDerivAt (Real.cos ∘ fun z => u * z) _ x
    have H := Real.hasDerivAt_cos (u * x) |>.comp x
      ((hasDerivAt_id x).const_mul u)
    simpa only [one_mul, mul_one, mul_comm, neg_mul, mul_neg] using H
  have hlip (x y : ℝ) :
      |-u * Real.sin (u * x) - -u * Real.sin (u * y)| ≤ u ^ 2 * |x - y| := by
    rw [← mul_sub, abs_mul]
    calc
      |-u| * |Real.sin (u * x) - Real.sin (u * y)| ≤
          |-u| * |u * x - u * y| := by
            exact mul_le_mul_of_nonneg_left (Real.abs_sin_sub_sin_le _ _) (abs_nonneg (-u))
      _ = u ^ 2 * |x - y| := by
        rw [← mul_sub, abs_mul, abs_neg]
        calc
          |u| * (|u| * |x - y|) = |u| ^ 2 * |x - y| := by ring
          _ = u ^ 2 * |x - y| := by rw [sq_abs]
  constructor
  · exact fun x => Real.abs_cos_le_one _
  constructor
  · intro x y
    calc
      |Real.cos (u * x) - Real.cos (u * y)| ≤ |u * x - u * y| :=
        Real.abs_cos_sub_cos_le _ _
      _ = |u| * |x - y| := by rw [← mul_sub, abs_mul]
  constructor
  · exact first_order_remainder_le_of_deriv_lipschitz hderiv hlip (sq_nonneg u)
  constructor
  · intro x
    rw [abs_mul, abs_neg]
    exact mul_le_of_le_one_right (abs_nonneg u) (Real.abs_sin_le_one _)
  · exact hlip

noncomputable def quenchedReplicaAverageLinearMap
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H) :
    ReplicaFun N n →ₗ[ℝ] ℝ where
  toFun F := quenchedReplicaAverage H F
  map_add' F G := quenchedReplicaAverage_add_clt H hH F G
  map_smul' c F := by
    simp only [RingHom.id_apply]
    exact quenchedReplicaAverage_const_mul H c F

@[simp] lemma quenchedReplicaAverageLinearMap_apply
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (F : ReplicaFun N n) :
    quenchedReplicaAverageLinearMap H hH F = quenchedReplicaAverage H F := rfl

lemma hasDerivAt_quenchedReplicaAverage
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (F F' : ℝ → ReplicaFun N n) (t : ℝ)
    (hF : ∀ σs, HasDerivAt (fun u => F u σs) (F' t σs) t) :
    HasDerivAt (fun u => quenchedReplicaAverage H (F u))
      (quenchedReplicaAverage H (F' t)) t := by
  let L := quenchedReplicaAverageLinearMap (n := n) H hH
  let Lc : ReplicaFun N n →L[ℝ] ℝ := LinearMap.toContinuousLinearMap L
  have hpi : HasDerivAt F (F' t) t := hasDerivAt_pi.2 hF
  have hc := Lc.hasFDerivAt.comp_hasDerivAt t hpi
  change HasDerivAt (Lc ∘ F) (Lc (F' t)) t
  exact hc

lemma hasDerivAt_quenched_cos
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (X : Replicas N n → ℝ) (t : ℝ) :
    HasDerivAt
      (fun u => quenchedReplicaAverage H (fun σs => Real.cos (u * X σs)))
      (-quenchedReplicaAverage H (fun σs => X σs * Real.sin (t * X σs))) t := by
  have hd := hasDerivAt_quenchedReplicaAverage (n := n) H hH
    (fun u σs => Real.cos (u * X σs))
    (fun u σs => X σs * (-Real.sin (u * X σs))) t (by
      intro σs
      have Hder := Real.hasDerivAt_cos (t * X σs) |>.comp t
        ((hasDerivAt_id t).mul_const (X σs))
      change HasDerivAt (Real.cos ∘ fun u => u * X σs) _ t
      simpa only [one_mul, mul_one, neg_mul, Function.id_def, mul_comm] using Hder)
  rw [show (fun σs => X σs * (-Real.sin (t * X σs))) =
      fun σs => (-1 : ℝ) * (X σs * Real.sin (t * X σs)) by
        funext σs; ring,
    quenchedReplicaAverage_const_mul] at hd
  simpa using hd

lemma hasDerivAt_quenched_sin
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (X : Replicas N n → ℝ) (t : ℝ) :
    HasDerivAt
      (fun u => quenchedReplicaAverage H (fun σs => Real.sin (u * X σs)))
      (quenchedReplicaAverage H (fun σs => X σs * Real.cos (t * X σs))) t := by
  apply hasDerivAt_quenchedReplicaAverage (n := n) H hH
    (fun u σs => Real.sin (u * X σs))
    (fun u σs => X σs * Real.cos (u * X σs)) t
  intro σs
  have Hder := Real.hasDerivAt_sin (t * X σs) |>.comp t
    ((hasDerivAt_id t).mul_const (X σs))
  change HasDerivAt (Real.sin ∘ fun u => u * X σs) _ t
  simpa only [one_mul, mul_one, mul_comm, Function.id_def] using Hder

lemma quenchedReplicaAverage_one
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) :
    quenchedReplicaAverage H (fun _ : Replicas N n => 1) = 1 := by
  unfold quenchedReplicaAverage replicaGibbsAverage
  simp only [mul_one]
  rw [show (fun ω => ∑ σs : Replicas N n,
      ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a)) = fun _ => 1 by
    funext ω
    rw [← Fintype.prod_sum]
    simp [SpinGlass.sum_gibbs_pmf]]
  simp

lemma measurable_fullPathHamiltonian_clt
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) : Measurable (fullPathHamiltonian path s) := by
  let i : Fin N := ⟨0, hN⟩
  rw [← show lastSiteHamiltonian (s := s) path i 1 = fullPathHamiltonian path s by
    funext ω
    rw [lastSiteHamiltonian_one]]
  exact measurable_lastSiteHamiltonian path i 1

lemma weightedFullVector_zero_eq
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (f : ℝ → ℝ) :
    weightedFullVector (s := s) path f 0 =
      quenchedReplicaAverage (fullPathHamiltonian path s)
        (fun σs : Replicas N 4 =>
          fullScaledArgument q σs * f (fullScaledArgument q σs)) := by
  unfold weightedFullVector weightedFullMoment fullScaledArgument
  rw [← quenchedReplicaAverage_const_mul]
  congr 1
  funext σs
  simp [CavityEstimates.targetEdge4_test, CavityEstimates.e4_01]
  ring

noncomputable def cltCos4
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q) (t : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path 1)
    (fun σs : Replicas N 4 => Real.cos (t * fullScaledArgument q σs))

noncomputable def cltSin4
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q) (t : ℝ) : ℝ :=
  quenchedReplicaAverage (fullPathHamiltonian path 1)
    (fun σs : Replicas N 4 => Real.sin (t * fullScaledArgument q σs))

noncomputable def cltSteinError
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q) (T : ℝ) : ℝ :=
  scalarResidualConstant β h * (Real.sqrt (N : ℝ) *
    cltLocalErrorBound (s := 1) path 1 T T (T ^ 2))

lemma cltCos4_hasDerivAt
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (t : ℝ) :
    HasDerivAt (cltCos4 path)
      (-quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N 4 => fullScaledArgument q σs *
          Real.sin (t * fullScaledArgument q σs))) t := by
  exact hasDerivAt_quenched_cos _ (measurable_fullPathHamiltonian_clt path hN)
    (fullScaledArgument q) t

lemma cltSin4_hasDerivAt
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q)
    (hN : 0 < N) (t : ℝ) :
    HasDerivAt (cltSin4 path)
      (quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N 4 => fullScaledArgument q σs *
          Real.cos (t * fullScaledArgument q σs))) t := by
  exact hasDerivAt_quenched_sin _ (measurable_fullPathHamiltonian_clt path hN)
    (fullScaledArgument q) t

lemma cltCos4_stein_deriv_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h T t : ℝ}
    (path : RSSmartPathDisorder Ω N β h (rsQ β h))
    (hN : 0 < N) (hh : 0 < h) (hAT : atParameter β h < 1)
    (hT : 0 ≤ T) (ht : |t| ≤ T) :
    |deriv (cltCos4 path) t + cltVariance β h * t * cltCos4 path t| ≤
      cltSteinError path T := by
  obtain ⟨hf₀, hf₁, hfTaylor, hf'₀, hf'₁⟩ := sin_test_bounds t
  have ht2 : t ^ 2 ≤ T ^ 2 := by
    have hsq := mul_self_le_mul_self (abs_nonneg t) ht
    simpa [sq_abs, abs_of_nonneg hT, pow_two] using hsq
  have hs := weightedFullVector_scalar_le path hN hh hAT
    (fun x => Real.sin (t * x)) (fun x => t * Real.cos (t * x))
    hf₀
    (fun x y => (hf₁ x y).trans (mul_le_mul_of_nonneg_right ht (abs_nonneg _)))
    (fun x y => (hfTaylor x y).trans
      (mul_le_mul_of_nonneg_right ht2 (sq_nonneg _)))
    (fun x => (hf'₀ x).trans ht)
    (fun x y => (hf'₁ x y).trans
      (mul_le_mul_of_nonneg_right ht2 (abs_nonneg _)))
    (by norm_num) hT hT (sq_nonneg T)
  rw [weightedFullVector_zero_eq] at hs
  unfold fullDerivativeAverage at hs
  rw [show (fun σs : Replicas N 4 => t * Real.cos (t * fullScaledArgument (rsQ β h) σs)) =
      fun σs => t * (fun τs : Replicas N 4 =>
        Real.cos (t * fullScaledArgument (rsQ β h) τs)) σs by rfl,
    quenchedReplicaAverage_const_mul] at hs
  rw [(cltCos4_hasDerivAt path hN t).deriv]
  unfold cltCos4 cltSteinError
  have heq :
      (-quenchedReplicaAverage (fullPathHamiltonian path 1) (fun σs : Replicas N 4 =>
          fullScaledArgument (rsQ β h) σs *
            Real.sin (t * fullScaledArgument (rsQ β h) σs))) +
          cltVariance β h * t *
            quenchedReplicaAverage (fullPathHamiltonian path 1) (fun σs : Replicas N 4 =>
              Real.cos (t * fullScaledArgument (rsQ β h) σs)) =
        -(quenchedReplicaAverage (fullPathHamiltonian path 1) (fun σs : Replicas N 4 =>
            fullScaledArgument (rsQ β h) σs *
              Real.sin (t * fullScaledArgument (rsQ β h) σs)) -
          cltVariance β h * (t *
            quenchedReplicaAverage (fullPathHamiltonian path 1) (fun σs : Replicas N 4 =>
              Real.cos (t * fullScaledArgument (rsQ β h) σs)))) := by ring
  rw [heq, abs_neg]
  exact hs

lemma cltSin4_stein_deriv_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h T t : ℝ}
    (path : RSSmartPathDisorder Ω N β h (rsQ β h))
    (hN : 0 < N) (hh : 0 < h) (hAT : atParameter β h < 1)
    (hT : 0 ≤ T) (ht : |t| ≤ T) :
    |deriv (cltSin4 path) t + cltVariance β h * t * cltSin4 path t| ≤
      cltSteinError path T := by
  obtain ⟨hf₀, hf₁, hfTaylor, hf'₀, hf'₁⟩ := cos_test_bounds t
  have ht2 : t ^ 2 ≤ T ^ 2 := by
    have hsq := mul_self_le_mul_self (abs_nonneg t) ht
    simpa [sq_abs, abs_of_nonneg hT, pow_two] using hsq
  have hs := weightedFullVector_scalar_le path hN hh hAT
    (fun x => Real.cos (t * x)) (fun x => -t * Real.sin (t * x))
    hf₀
    (fun x y => (hf₁ x y).trans (mul_le_mul_of_nonneg_right ht (abs_nonneg _)))
    (fun x y => (hfTaylor x y).trans
      (mul_le_mul_of_nonneg_right ht2 (sq_nonneg _)))
    (fun x => (hf'₀ x).trans ht)
    (fun x y => (hf'₁ x y).trans
      (mul_le_mul_of_nonneg_right ht2 (abs_nonneg _)))
    (by norm_num) hT hT (sq_nonneg T)
  rw [weightedFullVector_zero_eq] at hs
  unfold fullDerivativeAverage at hs
  rw [show (fun σs : Replicas N 4 => -t * Real.sin (t * fullScaledArgument (rsQ β h) σs)) =
      fun σs => (-t) * (fun τs : Replicas N 4 =>
        Real.sin (t * fullScaledArgument (rsQ β h) τs)) σs by rfl,
    quenchedReplicaAverage_const_mul] at hs
  rw [(cltSin4_hasDerivAt path hN t).deriv]
  unfold cltSin4 cltSteinError
  simpa [mul_comm, mul_left_comm, mul_assoc] using hs

lemma integratingFactor_mul_hasDerivAt
    {σ u F' : ℝ} {F : ℝ → ℝ} (hF : HasDerivAt F F' u) :
    HasDerivAt (fun v => Real.exp (σ * v ^ 2 / 2) * F v)
      (Real.exp (σ * u ^ 2 / 2) * (F' + σ * u * F u)) u := by
  have hi : HasDerivAt (fun v : ℝ => σ * v ^ 2 / 2) (σ * u) u := by
    have H := ((hasDerivAt_id u).mul (hasDerivAt_id u)).mul_const (σ / 2)
    have H' := H.congr_of_eventuallyEq (Filter.Eventually.of_forall fun v => by
      change σ * v ^ 2 / 2 = (id v * id v) * (σ / 2)
      simp [Function.id_def, pow_two]
      ring)
    have hd : (u + u) * (σ / 2) = σ * u := by ring
    simp only [one_mul, mul_one, Function.id_def] at H'
    rw [hd] at H'
    exact H'
  have he := Real.hasDerivAt_exp (σ * u ^ 2 / 2) |>.comp u hi
  have hm := he.mul hF
  have hm' := hm.congr_of_eventuallyEq (Filter.Eventually.of_forall fun _ => rfl)
  simp only [Function.comp_apply] at hm'
  have hd : Real.exp (σ * u ^ 2 / 2) * (σ * u) * F u +
        Real.exp (σ * u ^ 2 / 2) * F' =
      Real.exp (σ * u ^ 2 / 2) * (F' + σ * u * F u) := by ring
  rw [hd] at hm'
  change HasDerivAt ((fun v => Real.exp (σ * v ^ 2 / 2)) * F) _ u
  exact hm'

private lemma sq_le_sq_of_abs_le {x T : ℝ} (hT : 0 ≤ T) (hx : |x| ≤ T) :
    x ^ 2 ≤ T ^ 2 := by
  have hsq := mul_self_le_mul_self (abs_nonneg x) hx
  simpa [sq_abs, abs_of_nonneg hT, pow_two] using hsq

private lemma integratingFactor_bound {σ x T : ℝ} (hT : 0 ≤ T) (hx : |x| ≤ T) :
    Real.exp (σ * x ^ 2 / 2) ≤ Real.exp (|σ| * T ^ 2 / 2) := by
  apply Real.exp_le_exp.mpr
  have hx2 := sq_le_sq_of_abs_le hT hx
  have hσ : σ ≤ |σ| := le_abs_self σ
  have h₁ : σ * x ^ 2 ≤ |σ| * x ^ 2 :=
    mul_le_mul_of_nonneg_right hσ (sq_nonneg x)
  have h₂ : |σ| * x ^ 2 ≤ |σ| * T ^ 2 :=
    mul_le_mul_of_nonneg_left hx2 (abs_nonneg σ)
  linarith

lemma cltCos4_gaussian_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h t : ℝ}
    (path : RSSmartPathDisorder Ω N β h (rsQ β h))
    (hN : 0 < N) (hh : 0 < h) (hAT : atParameter β h < 1) :
    |cltCos4 path t - Real.exp (-(cltVariance β h * t ^ 2 / 2))| ≤
      Real.exp (|cltVariance β h| * |t| ^ 2 / 2) ^ 2 * |t| *
        cltSteinError path |t| := by
  let σ := cltVariance β h
  let T := |t|
  let E := cltSteinError path T
  let G : ℝ → ℝ := fun u => Real.exp (σ * u ^ 2 / 2) * cltCos4 path u
  have hT : 0 ≤ T := abs_nonneg t
  have hG (u : ℝ) : HasDerivAt G
      (Real.exp (σ * u ^ 2 / 2) *
        (deriv (cltCos4 path) u + σ * u * cltCos4 path u)) u :=
    by
      have hc := cltCos4_hasDerivAt path hN u
      have hg := integratingFactor_mul_hasDerivAt (σ := σ) hc
      rw [← hc.deriv] at hg
      exact hg
  have hE : 0 ≤ E := by
    exact (abs_nonneg (deriv (cltCos4 path) 0 + σ * 0 * cltCos4 path 0)).trans
      (cltCos4_stein_deriv_le path hN hh hAT hT (by simp [T]))
  have hbound (u : ℝ) (hu : u ∈ Set.uIcc 0 t) :
      ‖deriv G u‖ ≤ Real.exp (|σ| * T ^ 2 / 2) * E := by
    have huT : |u| ≤ T := by
      simpa [T] using Set.abs_sub_left_of_mem_uIcc hu
    rw [(hG u).deriv, Real.norm_eq_abs, abs_mul,
      abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul (integratingFactor_bound hT huT)
      (cltCos4_stein_deriv_le path hN hh hAT hT huT)
      (abs_nonneg _) (Real.exp_nonneg _)
  have hmv := (convex_uIcc 0 t).norm_image_sub_le_of_norm_deriv_le
    (fun u _ => (hG u).differentiableAt) hbound Set.left_mem_uIcc Set.right_mem_uIcc
  have hC0 : cltCos4 path 0 = 1 := by
    unfold cltCos4
    simpa using quenchedReplicaAverage_one (fullPathHamiltonian path 1)
  have hG0 : G 0 = 1 := by simp [G, hC0]
  rw [hG0, Real.norm_eq_abs, Real.norm_eq_abs] at hmv
  have hid : cltCos4 path t - Real.exp (-(σ * t ^ 2 / 2)) =
      Real.exp (-(σ * t ^ 2 / 2)) * (G t - 1) := by
    dsimp [G]
    rw [mul_sub, ← mul_assoc, ← Real.exp_add]
    have hz : -(σ * t ^ 2 / 2) + σ * t ^ 2 / 2 = 0 := by ring
    rw [hz, Real.exp_zero, one_mul]
    ring
  rw [hid, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    Real.exp (-(σ * t ^ 2 / 2)) * |G t - 1| ≤
        Real.exp (|σ| * T ^ 2 / 2) *
          (Real.exp (|σ| * T ^ 2 / 2) * E * |t|) := by
      gcongr
      · have ht2 : t ^ 2 = T ^ 2 := by simp [T, sq_abs]
        rw [ht2]
        nlinarith [le_abs_self σ, neg_le_abs σ]
      · simpa [Real.norm_eq_abs] using hmv
    _ = Real.exp (|σ| * T ^ 2 / 2) ^ 2 * |t| * E := by ring

lemma cltSin4_zero_le
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h t : ℝ}
    (path : RSSmartPathDisorder Ω N β h (rsQ β h))
    (hN : 0 < N) (hh : 0 < h) (hAT : atParameter β h < 1) :
    |cltSin4 path t| ≤
      Real.exp (|cltVariance β h| * |t| ^ 2 / 2) ^ 2 * |t| *
        cltSteinError path |t| := by
  let σ := cltVariance β h
  let T := |t|
  let E := cltSteinError path T
  let G : ℝ → ℝ := fun u => Real.exp (σ * u ^ 2 / 2) * cltSin4 path u
  have hT : 0 ≤ T := abs_nonneg t
  have hG (u : ℝ) : HasDerivAt G
      (Real.exp (σ * u ^ 2 / 2) *
        (deriv (cltSin4 path) u + σ * u * cltSin4 path u)) u := by
    have hc := cltSin4_hasDerivAt path hN u
    have hg := integratingFactor_mul_hasDerivAt (σ := σ) hc
    rw [← hc.deriv] at hg
    exact hg
  have hE : 0 ≤ E := by
    exact (abs_nonneg (deriv (cltSin4 path) 0 + σ * 0 * cltSin4 path 0)).trans
      (cltSin4_stein_deriv_le path hN hh hAT hT (by simp [T]))
  have hbound (u : ℝ) (hu : u ∈ Set.uIcc 0 t) :
      ‖deriv G u‖ ≤ Real.exp (|σ| * T ^ 2 / 2) * E := by
    have huT : |u| ≤ T := by
      simpa [T] using Set.abs_sub_left_of_mem_uIcc hu
    rw [(hG u).deriv, Real.norm_eq_abs, abs_mul,
      abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul (integratingFactor_bound hT huT)
      (cltSin4_stein_deriv_le path hN hh hAT hT huT)
      (abs_nonneg _) (Real.exp_nonneg _)
  have hmv := (convex_uIcc 0 t).norm_image_sub_le_of_norm_deriv_le
    (fun u _ => (hG u).differentiableAt) hbound Set.left_mem_uIcc Set.right_mem_uIcc
  have hS0 : cltSin4 path 0 = 0 := by
    unfold cltSin4 quenchedReplicaAverage replicaGibbsAverage
    simp
  have hG0 : G 0 = 0 := by simp [G, hS0]
  rw [hG0, sub_zero, Real.norm_eq_abs, Real.norm_eq_abs] at hmv
  have hid : cltSin4 path t = Real.exp (-(σ * t ^ 2 / 2)) * G t := by
    dsimp [G]
    rw [← mul_assoc, ← Real.exp_add]
    have hz : -(σ * t ^ 2 / 2) + σ * t ^ 2 / 2 = 0 := by ring
    rw [hz, Real.exp_zero, one_mul]
  rw [hid, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    Real.exp (-(σ * t ^ 2 / 2)) * |G t| ≤
        Real.exp (|σ| * T ^ 2 / 2) *
          (Real.exp (|σ| * T ^ 2 / 2) * E * |t|) := by
      gcongr
      · have ht2 : t ^ 2 = T ^ 2 := by simp [T, sq_abs]
        rw [ht2]
        nlinarith [le_abs_self σ, neg_le_abs σ]
      · simpa [Real.norm_eq_abs] using hmv
    _ = Real.exp (|σ| * T ^ 2 / 2) ^ 2 * |t| * E := by ring

lemma tendsto_cltSteinError_zero
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h))
    (T : ℝ) : Tendsto (fun N => cltSteinError (paths N) T) atTop (nhds 0) := by
  unfold cltSteinError
  simpa using (tendsto_sqrt_mul_cltLocalErrorBound_zero hβ hh hAT paths
    1 T T (T ^ 2)).const_mul (scalarResidualConstant β h)

theorem cltCos4_tendsto
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h))
    (t : ℝ) :
    Tendsto (fun N => cltCos4 (paths N) t) atTop
      (nhds (Real.exp (-(cltVariance β h * t ^ 2 / 2)))) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  simp only [Real.norm_eq_abs]
  apply squeeze_zero'
  · exact Filter.Eventually.of_forall fun _ => abs_nonneg _
  · exact Filter.Eventually.of_forall fun N => cltCos4_gaussian_le
      (t := t) (paths N) (Nat.succ_pos N) hh hAT
  · have he := tendsto_cltSteinError_zero hβ hh hAT paths |t|
    simpa using he.const_mul
      (Real.exp (|cltVariance β h| * t ^ 2 / 2) ^ 2 * |t|)

theorem cltSin4_tendsto
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {β h : ℝ} (hβ : 0 < β) (hh : 0 < h)
    (hAT : atParameter β h < 1)
    (paths : ∀ N : ℕ, RSSmartPathDisorder Ω N.succ β h (rsQ β h))
    (t : ℝ) : Tendsto (fun N => cltSin4 (paths N) t) atTop (nhds 0) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  simp only [sub_zero, Real.norm_eq_abs]
  apply squeeze_zero'
  · exact Filter.Eventually.of_forall fun _ => abs_nonneg _
  · exact Filter.Eventually.of_forall fun N => cltSin4_zero_le
      (t := t) (paths N) (Nat.succ_pos N) hh hAT
  · have he := tendsto_cltSteinError_zero hβ hh hAT paths |t|
    simpa using he.const_mul
      (Real.exp (|cltVariance β h| * t ^ 2 / 2) ^ 2 * |t|)

lemma cltCos4_eq_twoReplica
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q) (t : ℝ) :
    cltCos4 path t =
      quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N 2 =>
          Real.cos (t * Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1)) := by
  let F : ReplicaFun N 2 := fun σs =>
    Real.cos (t * Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1)
  calc
    cltCos4 path t = quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N (2 + 2) => F (initialReplicas σs)) := by
      unfold cltCos4 fullScaledArgument
      congr 1
      funext σs
      simp [F, centeredOverlap, replicaOverlap, initialReplicas]
      ring_nf
    _ = quenchedReplicaAverage (fullPathHamiltonian path 1) F :=
      quenchedReplicaAverage_initialReplicas _ F

lemma cltSin4_eq_twoReplica
    {Ω : Type u} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : RSSmartPathDisorder Ω N β h q) (t : ℝ) :
    cltSin4 path t =
      quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N 2 =>
          Real.sin (t * Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1)) := by
  let F : ReplicaFun N 2 := fun σs =>
    Real.sin (t * Real.sqrt (N : ℝ) * centeredOverlap q σs 0 1)
  calc
    cltSin4 path t = quenchedReplicaAverage (fullPathHamiltonian path 1)
        (fun σs : Replicas N (2 + 2) => F (initialReplicas σs)) := by
      unfold cltSin4 fullScaledArgument
      congr 1
      funext σs
      simp [F, centeredOverlap, replicaOverlap, initialReplicas]
      ring_nf
    _ = quenchedReplicaAverage (fullPathHamiltonian path 1) F :=
      quenchedReplicaAverage_initialReplicas _ F

end CLT
end SpinGlass.AT
