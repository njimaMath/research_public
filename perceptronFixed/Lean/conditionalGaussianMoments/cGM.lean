import Mathlib

namespace ConditionalGaussianMoments

/-- `d(u) := μ 1 u` (matching the blueprint notation). -/
def d (μ : ℕ → ℝ → ℝ) (u : ℝ) : ℝ :=
  μ 1 u

/--
Given a recursion coming from integration by parts for conditional moments
`μ k u = 𝔼[(X - u)^k | X ≥ u]`, this derives the closed forms of `μ k` for `k ≤ 4`
in terms of `d(u) := μ 1 u`.
-/
theorem mu_0_to_4_of_rec (μ : ℕ → ℝ → ℝ) (u : ℝ)
    (hμ0 : μ 0 u = 1)
    (hrec : ∀ k : ℕ, μ (k + 2) u = ((k + 1 : ℕ) : ℝ) * μ k u - u * μ (k + 1) u) :
    μ 0 u = 1 ∧
      μ 1 u = d μ u ∧
        μ 2 u = 1 - u * d μ u ∧
          μ 3 u = (u ^ 2 + 2) * d μ u - u ∧
            μ 4 u = u ^ 2 + 3 - u * (u ^ 2 + 5) * d μ u := by
  have hμ2 : μ 2 u = 1 - u * d μ u := by
    simpa [d, hμ0] using (hrec 0)

  have hμ2' : μ 2 u = 1 - u * μ 1 u := by
    simpa [d] using hμ2

  have hμ3' : μ 3 u = (u ^ 2 + 2) * μ 1 u - u := by
    have h : μ 3 u = (2 : ℝ) * μ 1 u - u * (1 - u * μ 1 u) := by
      simpa [hμ2'] using (hrec 1)
    calc
      μ 3 u = (2 : ℝ) * μ 1 u - u * (1 - u * μ 1 u) := h
      _ = (u ^ 2 + 2) * μ 1 u - u := by ring

  have hμ3 : μ 3 u = (u ^ 2 + 2) * d μ u - u := by
    simpa [d] using hμ3'

  have hμ4' : μ 4 u = u ^ 2 + 3 - u * (u ^ 2 + 5) * μ 1 u := by
    have h : μ 4 u = (3 : ℝ) * (1 - u * μ 1 u) - u * ((u ^ 2 + 2) * μ 1 u - u) := by
      simpa [hμ2', hμ3'] using (hrec 2)
    calc
      μ 4 u = (3 : ℝ) * (1 - u * μ 1 u) - u * ((u ^ 2 + 2) * μ 1 u - u) := h
      _ = u ^ 2 + 3 - u * (u ^ 2 + 5) * μ 1 u := by ring

  have hμ4 : μ 4 u = u ^ 2 + 3 - u * (u ^ 2 + 5) * d μ u := by
    simpa [d] using hμ4'

  refine And.intro hμ0 ?_
  refine And.intro rfl ?_
  refine And.intro hμ2 ?_
  refine And.intro hμ3 ?_
  exact hμ4

end ConditionalGaussianMoments
