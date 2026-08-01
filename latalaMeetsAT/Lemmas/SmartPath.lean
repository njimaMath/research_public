import Lemmas.UniformData
import Mathlib.Probability.Distributions.Gaussian.Basic

open MeasureTheory ProbabilityTheory Real
open scoped ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Covariance kernel of the random part of the RS smart path. -/
noncomputable def smartPathCovKernel (N : ℕ) (β q s : ℝ)
    (σ τ : Config N) : ℝ :=
  s * ((N : ℝ) * β ^ 2 / 2) * configOverlap N σ τ ^ 2 +
    (1 - s) * (N : ℝ) * β ^ 2 * q * configOverlap N σ τ -
      s * β ^ 2 / 2

/-- Abstract smart-path disorder.  This removes any hidden independence
assumption from all downstream interpolation arguments. -/
structure RSSmartPathDisorder (Ω : Type u) [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (N : ℕ) (β h q : ℝ) where
  H : ℝ → Ω → EnergySpace N
  measurable : ∀ s, Measurable (H s)
  gaussian : ∀ s, ∀ _hs : s ∈ Set.Icc (0 : ℝ) 1,
    IsGaussian (Measure.map (H s) volume)
  centered : ∀ s, ∀ _hs : s ∈ Set.Icc (0 : ℝ) 1, ∀ σ,
    ∫ ω, H s ω σ ∂(volume : Measure Ω) = 0
  covariance : ∀ s, ∀ _hs : s ∈ Set.Icc (0 : ℝ) 1, ∀ σ τ,
    ∫ ω, H s ω σ * H s ω τ ∂(volume : Measure Ω) =
      smartPathCovKernel N β q s σ τ

/-- The full smart-path Hamiltonian, including the deterministic external
field.  The random field `path.H` remains centered. -/
noncomputable def fullPathHamiltonian {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N : ℕ} {β h q : ℝ}
    (path : RSSmartPathDisorder Ω N β h q) (s : ℝ) (ω : Ω) : EnergySpace N :=
  fun σ => path.H s ω σ + h * ∑ i, spin σ i

noncomputable def pathATParameter (β h s : ℝ) : ℝ := s * atParameter β h

theorem smartPath_atParameter_eq (β h s : ℝ) :
    pathATParameter β h s = s * atParameter β h := by
  -- Proof route: unfold `pathATParameter`; this is equation (alphas).
  rfl

theorem smartPath_gap {K : Set (ℝ × ℝ)} (data : UniformATData K)
    {β h s : ℝ} (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    data.gap ≤ 1 - pathATParameter β h s := by
  -- Proof route: unfold the path parameter and reuse `path_gap`.
  simpa [pathATParameter] using path_gap data hp hs

end SpinGlass.AT
