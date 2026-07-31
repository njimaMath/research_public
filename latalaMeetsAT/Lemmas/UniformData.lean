import Lemmas.RSParameters

set_option autoImplicit false

namespace SpinGlass.AT

/-- Exactly the uniform numerical information consumed by the core proof. -/
structure UniformATData (K : Set (ℝ × ℝ)) where
  βmax : ℝ
  qmin : ℝ
  gap : ℝ
  βmax_pos : 0 < βmax
  qmin_pos : 0 < qmin
  gap_pos : 0 < gap
  β_bound : ∀ p ∈ K, p.1 ≤ βmax
  q_lower : ∀ p ∈ K, qmin ≤ rsQ p.1 p.2
  strictAT : ∀ p ∈ K, atParameter p.1 p.2 ≤ 1 - gap

theorem path_gap {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    data.gap ≤ 1 - s * atParameter β h := by
  have ha := data.strictAT (β, h) hp
  -- Proof route for equation (alphas): it is enough to combine
  -- `atParameter β h ≤ 1 - data.gap` with `0 ≤ s ≤ 1`.  One also needs
  -- `0 ≤ atParameter β h`, obtained from the Gaussian `sech ^ 4`
  -- representation of `rsA` and `sq_nonneg β`.  Then
  -- `s * atParameter β h ≤ atParameter β h ≤ 1 - data.gap`, and `linarith`
  -- proves the goal.  Add a lemma `atParameter_nonneg` in `RSParameters` so
  -- this argument does not repeat the integral-positivity proof.
  sorry

end SpinGlass.AT
