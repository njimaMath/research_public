import Lemmas.RSParameters

set_option autoImplicit false

namespace SpinGlass.AT

/-- Exactly the uniform numerical information consumed by the core proof. -/
structure UniformATData (K : Set (ℝ × ℝ)) where
  isCompact : IsCompact K
  βmax : ℝ
  qmin : ℝ
  gap : ℝ
  βmax_pos : 0 < βmax
  qmin_pos : 0 < qmin
  gap_pos : 0 < gap
  β_pos : ∀ p ∈ K, 0 < p.1
  h_pos : ∀ p ∈ K, 0 < p.2
  β_bound : ∀ p ∈ K, p.1 ≤ βmax
  q_lower : ∀ p ∈ K, qmin ≤ rsQ p.1 p.2
  strictAT : ∀ p ∈ K, atParameter p.1 p.2 ≤ 1 - gap

theorem path_gap {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    data.gap ≤ 1 - s * atParameter β h := by
  have ha := data.strictAT (β, h) hp
  have hat : 0 ≤ atParameter β h :=
    atParameter_nonneg (data.β_pos (β, h) hp) (data.h_pos (β, h) hp)
  have hsa : s * atParameter β h ≤ atParameter β h := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hs.2) hat]
  linarith

end SpinGlass.AT
