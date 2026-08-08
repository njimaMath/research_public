import Proof_of_generalized_latala.proof

open MeasureTheory ProbabilityTheory BigOperators
open PhysLean.Probability.GaussianIBP

namespace SpinGlass
namespace GeneralizedLatala

universe uΩ

variable {Ω : Type uΩ} [MeasureSpace Ω] [IsProbabilityMeasure (ℙ : Measure Ω)]

/-! ## Model -/

/-- Spin configurations on `N` sites. -/
abbrev ModelConfig (N : ℕ) := Fin N → Bool

/-- A centered Gaussian mixed p-spin disorder with covariance
`N ξ(R(σ,τ))`.  The scalar `d` records the reference variance `ξ'(q)`, and `h`
records the external field used with the disorder. -/
structure ModelMixedDisorder (N : ℕ) (ξ : ℝ → ℝ) (d h : ℝ) where
  /-- The random energy field indexed by spin configurations. -/
  U : Ω → EnergySpace N
  /-- The field is a finite-dimensional centered Gaussian random vector. -/
  hU : IsGaussianHilbert.{uΩ, 0, 0} U
  /-- Its covariance is the mixed p-spin covariance kernel. -/
  cov_eq : ∀ σ τ,
    inner ℝ ((covOp (g := U) hU) (std_basis N σ)) (std_basis N τ) =
      (N : ℝ) * ξ
        ((1 / (N : ℝ)) * ∑ i : Fin N,
          (if σ i then (1 : ℝ) else -1) *
            (if τ i then (1 : ℝ) else -1))

/-- Convert the explicit model specification to the disorder interface used by `proof.lean`. -/
noncomputable def ModelMixedDisorder.toSKDisorder
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h) :
    SKDisorder.{uΩ} (Ω := Ω) N d h where
  U := sk.U
  hU := sk.hU
  ξ := ξ
  cov_eq := by
    intro σ τ
    rw [sk.cov_eq]
    rfl

/-- The value `±1` of a Boolean spin. -/
def modelSpin {N : ℕ} (σ : ModelConfig N) (i : Fin N) : ℝ :=
  if σ i then 1 else -1

/-- The normalized overlap of two configurations. -/
noncomputable def modelOverlap (N : ℕ) (σ τ : ModelConfig N) : ℝ :=
  (1 / (N : ℝ)) * ∑ i : Fin N, modelSpin σ i * modelSpin τ i

/-- The mixed p-spin energy at fixed disorder.  The formal Gibbs convention is
`exp (-modelEnergy)`; centered Gaussian symmetry gives the blueprint's equivalent sign. -/
noncomputable def modelEnergy
    (N : ℕ) (h : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h)
    (ω : Ω) (σ : ModelConfig N) : ℝ :=
  sk.U ω σ + h * ∑ i : Fin N, modelSpin σ i

/-- The finite-volume partition function. -/
noncomputable def modelPartitionFunction
    (N : ℕ) (h : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h) (ω : Ω) : ℝ :=
  ∑ σ : ModelConfig N, Real.exp (-modelEnergy N h sk ω σ)

/-- The Gibbs probability of a configuration at fixed disorder. -/
noncomputable def modelGibbsProbability
    (N : ℕ) (h : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h)
    (ω : Ω) (σ : ModelConfig N) : ℝ :=
  Real.exp (-modelEnergy N h sk ω σ) / modelPartitionFunction N h sk ω

/-- The quenched finite-volume pressure `φ_N`. -/
noncomputable def modelPressure
    (N : ℕ) (h : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h) : ℝ :=
  ∫ ω, (1 / (N : ℝ)) * Real.log (modelPartitionFunction N h sk ω) ∂ℙ

/-- The quantity `E⟨(R₁₂ - q)²⟩` appearing in the concentration claim. -/
noncomputable def modelOverlapSecondMoment
    (N : ℕ) (h q : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h) : ℝ :=
  ∫ ω, ∑ σs : Fin 2 → ModelConfig N,
    (modelOverlap N (σs 0) (σs 1) - q) ^ 2 *
      ∏ a : Fin 2, modelGibbsProbability N h sk ω (σs a) ∂ℙ

/-! ## Scalar replica-symmetric data -/

/-- Expectation against a standard real Gaussian. -/
noncomputable def modelGaussianExpectation (f : ℝ → ℝ) : ℝ :=
  ∫ z, f z ∂gaussianReal 0 1

/-- The fixed-point equation with `d = ξ'(q)`. -/
def ModelFixedPoint (d h q : ℝ) : Prop :=
  q = modelGaussianExpectation
    (fun z => Real.tanh (h + Real.sqrt d * z) ^ 2)

/-- The Bernoulli sub-Gaussian coefficient from the blueprint. -/
noncomputable def modelKappa (q : ℝ) : ℝ :=
  if q = 0 then 1 else q / Real.artanh q

/-- The Bregman remainder of `ξ` at `q`, with prescribed derivative `d = ξ'(q)`. -/
noncomputable def modelBregmanRemainder
    (ξ : ℝ → ℝ) (d q r : ℝ) : ℝ :=
  ξ r - ξ q - d * (r - q)

/-- The convexity and global quadratic bound encoded in the form used by the proof. -/
def ModelBregmanBounds (ξ : ℝ → ℝ) (d q Γ : ℝ) : Prop :=
  ∀ r ∈ Set.Icc (-1 : ℝ) 1,
    0 ≤ modelBregmanRemainder ξ d q r ∧
      modelBregmanRemainder ξ d q r ≤ (Γ / 2) * (r - q) ^ 2

/-- The improved high-temperature parameter `ρ = Γ κ(q)`. -/
noncomputable def modelRho (Γ q : ℝ) : ℝ :=
  Γ * modelKappa q

/-- The replica-symmetric pressure for the mixed covariance `ξ`. -/
noncomputable def modelRSPressure (ξ : ℝ → ℝ) (d h q : ℝ) : ℝ :=
  Real.log 2 +
    modelGaussianExpectation
      (fun z => Real.log (Real.cosh (h + Real.sqrt d * z))) +
    (1 / 2) * modelBregmanRemainder ξ d q 1

/-! ## Claims -/

/-- The two `O(1/N)` conclusions in the blueprint, with one common constant. -/
def ModelClaims
    (N : ℕ) (ξ : ℝ → ℝ) (d h q : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧
    modelOverlapSecondMoment N h q sk ≤ C / (N : ℝ) ∧
    0 ≤ modelRSPressure ξ d h q - modelPressure N h sk ∧
    modelRSPressure ξ d h q - modelPressure N h sk ≤ C / (N : ℝ)

/-- The mixed p-spin model and claims above are verified under the global Bregman bound and
the improved-region assumption `Γ κ(q) < 1`. -/
theorem model_result
    (N : ℕ) [NeZero N] (ξ : ℝ → ℝ) (d Γ h q : ℝ)
    (sk : ModelMixedDisorder.{uΩ} (Ω := Ω) N ξ d h)
    (sim : SimpleDisorder.{uΩ} (Ω := Ω) N d q)
    (hN : 0 < N) (hd0 : 0 ≤ d) (hΓ0 : 0 ≤ Γ)
    (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hfp : ModelFixedPoint d h q)
    (hΔ : ModelBregmanBounds ξ d q Γ)
    (hρ : modelRho Γ q < 1)
    (hIndep : IndepFun sk.U sim.V (ℙ : Measure Ω)) :
    ModelClaims N ξ d h q sk := by
  let formalSK : SKDisorder.{uΩ} (Ω := Ω) N d h :=
    sk.toSKDisorder
  have henergy (ω : Ω) (σ : ModelConfig N) :
      H_t (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim) 1 ω σ =
        modelEnergy N h sk ω σ := by
    simp [formalSK, ModelMixedDisorder.toSKDisorder, H_t, H_gauss, H_field,
      modelEnergy, magnetic_field_vector, magnetization, modelSpin, spin]
  have hpressure :
      modelPressure N h sk =
        interpolatedPressure
          (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim) 1 := by
    apply integral_congr_ae
    filter_upwards with ω
    simp [free_energy_density, modelPartitionFunction, Z, henergy]
  have hoverlap :
      modelOverlapSecondMoment N h q sk =
        overlapVariance
          (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim) 1 := by
    apply integral_congr_ae
    filter_upwards with ω
    apply Finset.sum_congr rfl
    intro σs _
    simp [centeredOverlapSq, modelOverlap, overlap, modelGibbsProbability,
      gibbs_pmf, modelPartitionFunction, Z, modelSpin, spin, henergy]
  have hmain := generalized_latala
    (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim)
    hN hd0 hΓ0 hq0 hq1 hfp hΔ hρ hIndep
  have hlambda : 0 < lambdaStar Γ q :=
    lambdaStar_pos (Γ := Γ) (q := q) hq0 hq1 hρ
  have hQ : 0 < quadraticConstant Γ q :=
    quadraticConstant_pos (Γ := Γ) (q := q) hΓ0 hq0 hq1 hρ
  let A : ℝ := quadraticConstant Γ q / lambdaStar Γ q
  let B : ℝ := Γ * quadraticConstant Γ q / (4 * lambdaStar Γ q)
  have hA : 0 ≤ A := div_nonneg hQ.le hlambda.le
  have hB : 0 ≤ B := by
    dsimp only [B]
    positivity
  have hNr : (N : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt hN
  have hoverlapA : modelOverlapSecondMoment N h q sk ≤ A / (N : ℝ) := by
    rw [hoverlap]
    calc
      overlapVariance
          (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim) 1
          ≤ quadraticConstant Γ q / (lambdaStar Γ q * (N : ℝ)) := hmain.1
      _ = A / (N : ℝ) := by
        dsimp only [A]
        field_simp [ne_of_gt hlambda, hNr]
  have hpressure0 :
      0 ≤ modelRSPressure ξ d h q - modelPressure N h sk := by
    rw [hpressure]
    exact hmain.2.1
  have hpressureB :
      modelRSPressure ξ d h q - modelPressure N h sk ≤ B / (N : ℝ) := by
    rw [hpressure]
    calc
      rsPressure ξ d h q -
          interpolatedPressure
            (N := N) (β := d) (h := h) (q := q) (sk := formalSK) (sim := sim) 1
          ≤ (Γ * quadraticConstant Γ q) /
              (4 * lambdaStar Γ q * (N : ℝ)) := hmain.2.2
      _ = B / (N : ℝ) := by
        dsimp only [B]
        field_simp [ne_of_gt hlambda, hNr]
  refine ⟨A + B, add_nonneg hA hB, ?_, hpressure0, ?_⟩
  · exact hoverlapA.trans
      (div_le_div_of_nonneg_right (le_add_of_nonneg_right hB) (Nat.cast_nonneg N))
  · exact hpressureB.trans
      (div_le_div_of_nonneg_right (le_add_of_nonneg_left hA) (Nat.cast_nonneg N))

end GeneralizedLatala
end SpinGlass
