import Lemmas.Gaussian.ConcreteModel
import Lemmas.CLT.CLT_Main

/-!
# Concrete statement of the strict Almeida--Thouless results

This file introduces every model-specific object occurring in the two claims
below.  It intentionally does not use the spin-glass definitions from the
`Lemmas` directory.
-/

open MeasureTheory ProbabilityTheory BigOperators Filter
open scoped Topology

set_option autoImplicit false

namespace Main

noncomputable section

/-! ## Replica-symmetric parameters -/

/-- Expectation with respect to a standard real Gaussian random variable:
$\mathbb E[f(Z)] = \int_{\mathbb R} f(z)\,\varphi(z)\,dz$, where
$Z \sim \mathcal N(0,1)$ and $\varphi(z) = (2\pi)^{-1/2}e^{-z^2/2}$. -/
def standardGaussianExpectation (f : ℝ → ℝ) : ℝ :=
  ∫ z, f z ∂gaussianReal 0 1

/-- The replica-symmetric fixed-point equation
$q = \mathbb E[\tanh^2(h + \beta\sqrt q\,Z)]$. -/
def IsReplicaSymmetricFixedPoint (β h q : ℝ) : Prop :=
  q = standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt q * z) ^ 2)

/-- The canonical replica-symmetric overlap:
$q(\beta,h) = \inf\{x \in [0,1] :
x = \mathbb E[\tanh^2(h + \beta\sqrt x\,Z)]\}$. -/
def q (β h : ℝ) : ℝ :=
  sInf {x : ℝ | x ∈ Set.Icc (0 : ℝ) 1 ∧
    IsReplicaSymmetricFixedPoint β h x}

/-- The assertion that $q(\beta,h)$ is the unique interval-valued fixed point:
$q(\beta,h) \in [0,1]$, $q(\beta,h) = T_{\beta,h}(q(\beta,h))$, and
$x \in [0,1] \land x = T_{\beta,h}(x) \Rightarrow x = q(\beta,h)$, where
$T_{\beta,h}(x) = \mathbb E[\tanh^2(h + \beta\sqrt x\,Z)]$. -/
def ReplicaSymmetricFixedPointClaim (β h : ℝ) : Prop :=
  IsReplicaSymmetricFixedPoint β h (q β h) ∧
    q β h ∈ Set.Icc (0 : ℝ) 1 ∧
    ∀ x : ℝ, x ∈ Set.Icc (0 : ℝ) 1 →
      IsReplicaSymmetricFixedPoint β h x → x = q β h

/-- In a positive external field, the canonical infimum is the unique
replica-symmetric fixed point in `[0,1]`. -/
theorem replicaSymmetricFixedPointClaim_of_pos_field
    (β : ℝ) {h : ℝ} (hh : 0 < h) :
    ReplicaSymmetricFixedPointClaim β h := by
  have hq : q β h = SpinGlass.AT.rsQ β h := by
    rfl
  refine ⟨?_, ?_, ?_⟩
  · rw [hq]
    exact SpinGlass.AT.rsQ_fixedPoint_of_pos_field hh
  · rw [hq]
    exact SpinGlass.AT.rsQ_mem_Icc β h
  · intro x hx hfixed
    rw [hq]
    exact SpinGlass.AT.eq_rsQ_of_isRSFixedPoint hh hx hfixed

/-- The fourth local-magnetization moment:
$r(\beta,h) = \mathbb E[\tanh^4(h + \beta\sqrt{q(\beta,h)}\,Z)]$. -/
def r (β h : ℝ) : ℝ :=
  standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt (q β h) * z) ^ 4)

/-- The de Almeida--Thouless parameter:
$\alpha(\beta,h) = \beta^2(1 - 2q(\beta,h) + r(\beta,h))$. -/
def α (β h : ℝ) : ℝ :=
  β ^ 2 * (1 - 2 * q β h + r β h)

/-- The positive-temperature, positive-field strict AT region:
$\mathcal A_{\mathrm{strict}} =
\{(\beta,h) \in \mathbb R^2 : \beta > 0,\ h > 0,\ \alpha(\beta,h) < 1\}$. -/
def strictATRegion : Set (ℝ × ℝ) :=
  {p | 0 < p.1 ∧ 0 < p.2 ∧ α p.1 p.2 < 1}

/-! ## Gaussian disorder and the finite-volume model -/

/-- A spin configuration on $N$ sites, represented by a Boolean vector:
$\Sigma_N = \{-1,+1\}^N$. -/
abbrev Configuration (N : ℕ) := Fin N → Bool

/-- The real spin associated with a Boolean coordinate:
$s_i(\sigma) = 1$ if $\sigma_i$ is true, and $s_i(\sigma) = -1$ otherwise. -/
def spin {N : ℕ} (σ : Configuration N) (i : Fin N) : ℝ :=
  if σ i then 1 else -1

/-- A single probability space carrying all coordinates $g_{ij}$ and $z_i$:
$\Omega = \mathbb R^{(\mathbb N\times\mathbb N)\sqcup\mathbb N}$. -/
abbrev GaussianSpace := ((ℕ × ℕ) ⊕ ℕ) → ℝ

/-- The countable product standard Gaussian measure:
$\mathbb P = \bigotimes_{k\in(\mathbb N\times\mathbb N)\sqcup\mathbb N}
\mathcal N(0,1)$. -/
def gaussianMeasure : Measure GaussianSpace :=
  Measure.infinitePi (fun _ : (ℕ × ℕ) ⊕ ℕ => gaussianReal 0 1)

/-- The disorder coordinate $g_{ij}(\omega) = \omega(i,j)$. -/
def g (ω : GaussianSpace) {N : ℕ} (i j : Fin N) : ℝ :=
  ω (Sum.inl (i, j))

/-- The auxiliary-field coordinate $z_i(\omega) = \omega(i)$. -/
def z (ω : GaussianSpace) {N : ℕ} (i : Fin N) : ℝ :=
  ω (Sum.inr i)

/-- The endpoint Sherrington--Kirkpatrick Hamiltonian:
$H_N(\sigma) = \dfrac{\beta}{\sqrt{2N}}
\sum_{i,j=1}^N g_{ij}s_i(\sigma)s_j(\sigma)
+ h\sum_{i=1}^N s_i(\sigma)$. -/
def H_N (N : ℕ) (β h : ℝ) (ω : GaussianSpace)
    (σ : Configuration N) : ℝ :=
  β / Real.sqrt (2 * (N : ℝ)) *
      ∑ i : Fin N, ∑ j : Fin N, g ω i j * spin σ i * spin σ j
    + h * ∑ i : Fin N, spin σ i

/-- The smart-path Hamiltonian:
$H_{N,s}(\sigma) = \dfrac{\beta\sqrt s}{\sqrt{2N}}
\sum_{i,j=1}^N g_{ij}s_i(\sigma)s_j(\sigma)
+ \sum_{i=1}^N\bigl(h+\beta\sqrt{(1-s)q}\,z_i\bigr)s_i(\sigma)$. -/
def H_N_s (N : ℕ) (β h s : ℝ) (ω : GaussianSpace)
    (σ : Configuration N) : ℝ :=
  β * Real.sqrt s / Real.sqrt (2 * (N : ℝ)) *
      ∑ i : Fin N, ∑ j : Fin N, g ω i j * spin σ i * spin σ j
    + ∑ i : Fin N,
        (h + β * Real.sqrt ((1 - s) * q β h) * z ω i) * spin σ i

/-- The concrete Hamiltonian is the canonical smart path used by the proof
backend. -/
theorem H_N_s_eq_smartPath (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (ω : GaussianSpace)
    (σ : Configuration N) :
    H_N_s N β h s ω σ = H_s (canonicalSKDisorder N β h) s ω σ := by
  have hs' : 0 ≤ 1 - s := by linarith [hs.2]
  have hq : q β h = canonicalOverlap β h := by rfl
  rw [H_s_apply]
  simp only [H_N_s, hq]
  simp only [canonicalSKDisorder, GaussianDisorder.canonical, g, z, spin,
    canonicalG, canonicalZ, SpinGlass.spin]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  rw [Real.sqrt_mul hs']
  ring

/-- The partition function along the smart path:
$Z_{N,s}(\omega) = \sum_{\sigma\in\Sigma_N}e^{-H_{N,s}(\omega,\sigma)}$. -/
def partitionFunction (N : ℕ) (β h s : ℝ) (ω : GaussianSpace) : ℝ :=
  ∑ σ : Configuration N, Real.exp (-H_N_s N β h s ω σ)

/-- The Gibbs probability mass of one configuration:
$G_{N,s}^{\omega}(\sigma) =
e^{-H_{N,s}(\omega,\sigma)}/Z_{N,s}(\omega)$. -/
def gibbsWeight (N : ℕ) (β h s : ℝ) (ω : GaussianSpace)
    (σ : Configuration N) : ℝ :=
  Real.exp (-H_N_s N β h s ω σ) /
    partitionFunction N β h s ω

theorem partitionFunction_eq_Z (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (ω : GaussianSpace) :
    partitionFunction N β h s ω =
      SpinGlass.Z N (H_s (canonicalSKDisorder N β h) s ω) := by
  classical
  unfold partitionFunction SpinGlass.Z
  apply Finset.sum_congr rfl
  intro σ hσ
  rw [H_N_s_eq_smartPath N β h s hs]

theorem gibbsWeight_eq_gibbs_pmf (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) (ω : GaussianSpace)
    (σ : Configuration N) :
    gibbsWeight N β h s ω σ =
      SpinGlass.gibbs_pmf N (H_s (canonicalSKDisorder N β h) s ω) σ := by
  rw [gibbsWeight, SpinGlass.gibbs_pmf,
    partitionFunction_eq_Z N β h s hs,
    H_N_s_eq_smartPath N β h s hs]

/-- An indexed family of replicas:
$\boldsymbol\sigma=(\sigma^1,\ldots,\sigma^n)\in\Sigma_N^n$. -/
abbrev ReplicaFamily (N n : ℕ) := Fin n → Configuration N

/-- The product Gibbs bracket at fixed disorder:
$\langle F\rangle_{s,\omega} =
\sum_{\boldsymbol\sigma\in\Sigma_N^n}
F(\boldsymbol\sigma)\prod_{a=1}^nG_{N,s}^{\omega}(\sigma^a)$. -/
def gibbsBracket {N n : ℕ} (β h s : ℝ) (ω : GaussianSpace)
    (F : ReplicaFamily N n → ℝ) : ℝ :=
  ∑ σs : ReplicaFamily N n,
    (∏ a : Fin n, gibbsWeight N β h s ω (σs a)) * F σs

/-- The replica overlap:
$R_{ab} = N^{-1}\sum_{i=1}^N s_i(\sigma^a)s_i(\sigma^b)$. -/
def R_ab {N n : ℕ} (σs : ReplicaFamily N n) (a b : Fin n) : ℝ :=
  (1 / (N : ℝ)) * ∑ i : Fin N, spin (σs a) i * spin (σs b) i

theorem R_ab_eq_selectedReplicaOverlap {N n : ℕ}
    (σs : ReplicaFamily N n) (a b : Fin n) :
    R_ab σs a b = selectedReplicaOverlap σs a b := by
  rfl

/-- The disorder-averaged Gibbs expectation:
$\nu_s(F) = \mathbb E\langle F\rangle_s$. -/
def ν_s {N n : ℕ} (β h s : ℝ) (F : ReplicaFamily N n → ℝ) : ℝ :=
  ∫ ω, gibbsBracket β h s ω F ∂(volume : Measure GaussianSpace)

theorem ν_s_eq_disorderAveragedExpectation {N n : ℕ}
    (β h s : ℝ) (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (F : ReplicaFamily N n → ℝ) :
    ν_s β h s F = disorderAveragedExpectation
      (H_s (canonicalSKDisorder N β h) s) F := by
  unfold ν_s disorderAveragedExpectation gibbsBracket
    productGibbsExpectation
  apply integral_congr_ae
  filter_upwards [] with ω
  apply Finset.sum_congr rfl
  intro σs hσs
  congr 1
  apply Finset.prod_congr rfl
  intro a ha
  exact gibbsWeight_eq_gibbs_pmf N β h s hs ω (σs a)

/-- The centered overlap:
$Q_{ab} = R_{ab} - q(\beta,h)$. -/
def Q_ab {N n : ℕ} (β h : ℝ) (σs : ReplicaFamily N n)
    (a b : Fin n) : ℝ :=
  R_ab σs a b - q β h

theorem Q_ab_eq_centeredReplicaOverlap {N n : ℕ} (β h : ℝ)
    (σs : ReplicaFamily N n) (a b : Fin n) :
    Q_ab β h σs a b =
      centeredReplicaOverlap (canonicalOverlap β h) σs a b := by
  rfl

/-- The smart-path free-energy density:
$\varphi_N(s) = N^{-1}\mathbb E[\log Z_{N,s}]$. -/
def φ_N_s (N : ℕ) (β h s : ℝ) : ℝ :=
  (1 / (N : ℝ)) *
    ∫ ω, Real.log (partitionFunction N β h s ω)
      ∂(volume : Measure GaussianSpace)

/-- $A_s = \nu_s(Q_{12}^2)$. -/
def A_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s (fun σs => Q_ab β h σs 0 1 ^ 2)

/-- $B_s = \nu_s(Q_{12}Q_{13})$. -/
def B_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s
    (fun σs => Q_ab β h σs 0 1 * Q_ab β h σs 0 2)

/-- $C_s = \nu_s(Q_{12}Q_{34})$. -/
def C_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s
    (fun σs => Q_ab β h σs 0 1 * Q_ab β h σs 2 3)

theorem A_s_eq_overlapVariance (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    A_s N β h s = overlapVariance (canonicalSKDisorder N β h) s := by
  rw [A_s, ν_s_eq_disorderAveragedExpectation β h s hs]
  rfl

theorem B_s_eq_sharedReplicaMoment (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    B_s N β h s = sharedReplicaMoment (canonicalSKDisorder N β h) s := by
  rw [B_s, ν_s_eq_disorderAveragedExpectation β h s hs]
  rfl

theorem C_s_eq_disjointReplicaMoment (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    C_s N β h s = disjointReplicaMoment (canonicalSKDisorder N β h) s := by
  rw [C_s, ν_s_eq_disorderAveragedExpectation β h s hs]
  rfl

/-- The endpoint expected free-energy density:
$\varphi_N = \varphi_N(1) = N^{-1}\mathbb E[\log Z_{N,1}]$. -/
def φ_N (N : ℕ) (β h : ℝ) : ℝ :=
  φ_N_s N β h 1

theorem φ_N_eq_finiteVolumeFreeEnergy (N : ℕ) (β h : ℝ) :
    φ_N N β h = finiteVolumeFreeEnergy (canonicalSKDisorder N β h) := by
  unfold φ_N φ_N_s finiteVolumeFreeEnergy smartPathFreeEnergy
  rw [← integral_const_mul]
  apply integral_congr_ae
  filter_upwards [] with ω
  rw [partitionFunction_eq_Z N β h 1 (by simp)]
  rfl

/-- The replica-symmetric free energy:
$\operatorname{RS}(\beta,h) = \log 2
+ \mathbb E\!\left[\log\cosh\!\left(h+\beta\sqrt q\,Z\right)\right]
+ \dfrac{\beta^2}{4}(1-q)^2$. -/
def replicaSymmetricFreeEnergy (β h : ℝ) : ℝ :=
  Real.log 2 + standardGaussianExpectation
      (fun x => Real.log (Real.cosh
        (h + β * Real.sqrt (q β h) * x)))
    + β ^ 2 / 4 * (1 - q β h) ^ 2

theorem replicaSymmetricFreeEnergy_eq_backend (β h : ℝ) :
    replicaSymmetricFreeEnergy β h =
      _root_.replicaSymmetricFreeEnergy β h := by
  rfl

/-! ## Quantitative strict-AT claim -/

/-- The three conclusions of the principal theorem, in the order in which
they occur in the paper. Uniformly on $K$, some $C_K \ge 0$ satisfies
$N A_s \le C_K$ and
$0 \le \operatorname{RS}(\beta,h)-\varphi_N(\beta,h) \le C_K/N$, while
$N(A_s-2B_s+C_s) \to
\alpha/[\beta^2(1-s\alpha)]$. -/
structure MainClaim (K : Set (ℝ × ℝ)) : Prop where
  quantitativeBounds :
    ∃ C_K : ℝ, 0 ≤ C_K ∧
      (∀ {N : ℕ}, 0 < N → ∀ {β h s : ℝ},
        (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 →
        (N : ℝ) * A_s N β h s ≤ C_K) ∧
      (∀ {N : ℕ}, 0 < N → ∀ {β h : ℝ},
        (β, h) ∈ K →
        0 ≤ replicaSymmetricFreeEnergy β h - φ_N N β h ∧
        replicaSymmetricFreeEnergy β h - φ_N N β h ≤
          C_K / (N : ℝ))
  repliconSusceptibility :
    ∀ ε > 0, ∃ N₀ : ℕ, ∀ {N : ℕ}, N₀ ≤ N →
      ∀ {β h s : ℝ}, (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 →
      |(N : ℝ) * (A_s N β h s - 2 * B_s N β h s + C_s N β h s) -
        α β h / (β ^ 2 * (1 - s * α β h))| < ε

/-- The principal theorem as a proposition on every compact subset of the
strict AT region:
$\forall K\Subset\mathcal A_{\mathrm{strict}},\ \operatorname{MainClaim}(K)$. -/
def QuantitativeStrictATClaim : Prop :=
  ∀ K : Set (ℝ × ℝ), IsCompact K → K ⊆ strictATRegion → MainClaim K

/-- Quantitative concentration, free-energy control, and susceptibility on
compact subsets of the strict AT region. -/
theorem quantitativeStrictATClaim : QuantitativeStrictATClaim := by
  intro K hKcompact hKsub
  have hKsub' : K ⊆ strictStabilityRegion := by
    intro p hp
    simpa [strictATRegion, α, r, q, standardGaussianExpectation,
      IsReplicaSymmetricFixedPoint, strictStabilityRegion, stabilityIndex,
      canonicalSechFourthMoment, canonicalFourthMoment, canonicalOverlap,
      SpinGlass.AT.IsRSFixedPoint,
      SpinGlass.AT.standardGaussianExpectation] using hKsub hp
  have result := quantitative_strictAT
    (Ω := GaussianSpace) K hKcompact hKsub'
  obtain ⟨C₁, hC₁, hbound₁⟩ := result.secondMoment
  obtain ⟨C₂, hC₂, hbound₂⟩ := result.freeEnergy
  refine ⟨?_, ?_⟩
  · refine ⟨max C₁ C₂, hC₁.trans (le_max_left C₁ C₂), ?_, ?_⟩
    · intro N hN β h s hp hs
      have hresult := hbound₁ hN hp rfl hs (canonicalSKDisorder N β h)
      rw [A_s_eq_overlapVariance N β h s hs]
      exact hresult.trans (le_max_left C₁ C₂)
    · intro N hN β h hp
      have hresult := hbound₂ hN hp rfl (canonicalSKDisorder N β h)
      refine ⟨?_, le_trans ?_ (div_le_div_of_nonneg_right
        (le_max_right C₁ C₂) (Nat.cast_nonneg N))⟩
      · rw [replicaSymmetricFreeEnergy_eq_backend,
          φ_N_eq_finiteVolumeFreeEnergy]
        exact hresult.1
      · rw [replicaSymmetricFreeEnergy_eq_backend,
          φ_N_eq_finiteVolumeFreeEnergy]
        exact hresult.2
  · intro ε hε
    obtain ⟨N₀, hbound⟩ := result.replicon ε hε
    refine ⟨N₀, ?_⟩
    intro N hN β h s hp hs
    have hresult := hbound hN hp rfl hs (canonicalSKDisorder N β h)
    have hβ : β ≠ 0 := ne_of_gt (hKsub hp).1
    have hαeq : α β h = stabilityIndex β h := by rfl
    have hratio :
        α β h / (β ^ 2 * (1 - s * α β h)) =
          canonicalSechFourthMoment β h /
            (1 - s * stabilityIndex β h) := by
      rw [hαeq, stabilityIndex]
      field_simp [hβ]
    rw [hratio]
    rw [A_s_eq_overlapVariance N β h s hs,
      B_s_eq_sharedReplicaMoment N β h s hs,
      C_s_eq_disjointReplicaMoment N β h s hs]
    exact hresult

/-! ## Overlap central limit claim -/

/-- $\kappa(\beta,h) = 1 - 4q(\beta,h) + 3r(\beta,h)$. -/
def κ (β h : ℝ) : ℝ :=
  1 - 4 * q β h + 3 * r β h

/-- $\zeta(\beta,h) = 2q(\beta,h) + q(\beta,h)^2 - 3r(\beta,h)$. -/
def ζ (β h : ℝ) : ℝ :=
  2 * q β h + q β h ^ 2 - 3 * r β h

/-- The variance appearing in the overlap central limit theorem:
$v(\beta,h) = \dfrac{3(1-2q+r)}{1-\alpha}
- \dfrac{2\kappa}{1-\beta^2\kappa}
- \dfrac{\zeta}{(1-\beta^2\kappa)^2}$. -/
def overlapVariance (β h : ℝ) : ℝ :=
  3 * (1 - 2 * q β h + r β h) / (1 - α β h)
    - 2 * κ β h / (1 - β ^ 2 * κ β h)
    - ζ β h / (1 - β ^ 2 * κ β h) ^ 2

/-- Expectation of a test function of the scaled centered endpoint overlap:
$\nu_1\!\left[f(\sqrt N\,Q_{12})\right]$. -/
def scaledOverlapExpectation (N : ℕ) (β h : ℝ) (f : ℝ → ℝ) : ℝ :=
  ν_s (N := N) (n := 2) β h 1
    (fun σs => f (Real.sqrt (N : ℝ) * Q_ab β h σs 0 1))

/-- The canonical concrete disorder through the CLT smart-path interface,
with replica-symmetric overlap parameter $q=q(\beta,h)$. -/
def concreteRSSmartPathDisorder (N : ℕ) (β h : ℝ) :
    SpinGlass.AT.RSSmartPathDisorder GaussianSpace N β h
      (SpinGlass.AT.rsQ β h) := by
  simpa [canonicalOverlap, SpinGlass.AT.rsQ] using
    (canonicalSKDisorder N β h).toLibrary

theorem integral_quenchedReplicaMeasure_eq
    {Ω : Type*} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N) (hH : Measurable H)
    (F : SpinGlass.AT.ReplicaFun N n) :
    (∫ σs, F σs ∂SpinGlass.AT.quenchedReplicaMeasure H) =
      SpinGlass.AT.quenchedReplicaAverage H F := by
  classical
  letI : IsProbabilityMeasure
      (SpinGlass.AT.quenchedReplicaMeasure (n := n) H) :=
    SpinGlass.AT.quenchedReplicaMeasure_isProbabilityMeasure
      (n := n) H hH
  rw [MeasureTheory.integral_fintype]
  · unfold SpinGlass.AT.quenchedReplicaMeasure
      SpinGlass.AT.quenchedReplicaAverage SpinGlass.AT.replicaGibbsAverage
    have hterms : ∀ σs ∈
        (Finset.univ : Finset (SpinGlass.AT.Replicas N n)), Integrable
        (fun ω => (∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a)) * F σs)
        (volume : Measure Ω) := by
      intro σs _
      have hwint : Integrable
          (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a))
          (volume : Measure Ω) := by
        simpa [SpinGlass.AT.replicaGibbsAverage,
          SpinGlass.replicaGibbsWeightNNReal] using
          (SpinGlass.AT.integrable_replicaGibbsAverage_comp H hH
            (fun τ => if τ = σs then 1 else 0))
      simpa [mul_comm] using hwint.mul_const (F σs)
    rw [MeasureTheory.integral_finsetSum Finset.univ hterms]
    apply Finset.sum_congr rfl
    intro σs _
    rw [measureReal_def, Measure.bind_apply (MeasurableSet.singleton σs)
      ((SpinGlass.AT.measurable_replicaGibbsMeasure_comp H hH).aemeasurable)]
    simp only [SpinGlass.replicaGibbsMeasure, Measure.coe_finsetSum,
      Finset.sum_apply, Measure.smul_apply,
      Measure.dirac_apply' _ (MeasurableSet.singleton σs)]
    have hwint : Integrable
        (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a))
        (volume : Measure Ω) := by
      simpa [SpinGlass.AT.replicaGibbsAverage,
        SpinGlass.replicaGibbsWeightNNReal] using
        (SpinGlass.AT.integrable_replicaGibbsAverage_comp H hH
          (fun τ => if τ = σs then 1 else 0))
    have hsum (ω : Ω) :
        (∑ x, (SpinGlass.replicaGibbsWeightNNReal
            (N := N) (n := n) (H ω) x : ENNReal) •
          ({σs} : Set (SpinGlass.AT.Replicas N n)).indicator
            (1 : SpinGlass.AT.Replicas N n → ENNReal) x) =
          (SpinGlass.replicaGibbsWeightNNReal
            (N := N) (n := n) (H ω) σs : ENNReal) := by
      rw [Finset.sum_eq_single σs]
      · simp
      · intro x _ hx
        simp [Set.indicator, hx]
      · simp
    simp_rw [hsum]
    have hwint' : Integrable
        (fun ω => (SpinGlass.replicaGibbsWeightNNReal
          (N := N) (n := n) (H ω) σs : ℝ))
        (volume : Measure Ω) := by
      change Integrable
        (fun ω => ∏ a, SpinGlass.gibbs_pmf N (H ω) (σs a)) volume
      exact hwint
    rw [MeasureTheory.lintegral_coe_eq_integral _ hwint']
    have hw0 : 0 ≤ ∫ ω,
        (SpinGlass.replicaGibbsWeightNNReal
          (N := N) (n := n) (H ω) σs : ℝ) ∂volume :=
      integral_nonneg fun ω =>
        (SpinGlass.replicaGibbsWeightNNReal
          (N := N) (n := n) (H ω) σs).coe_nonneg
    simp only [ENNReal.toReal_ofReal hw0, smul_eq_mul]
    rw [MeasureTheory.integral_mul_const]
    rfl
  · exact MeasureTheory.Integrable.of_finite

theorem integral_scaledOverlapLaw_eq_scaledOverlapExpectation
    (N : ℕ) (β h : ℝ) (f : ℝ → ℝ) (hf : Continuous f) :
    (∫ x, f x ∂(SpinGlass.AT.scaledOverlapLaw
      (concreteRSSmartPathDisorder N β h) : Measure ℝ)) =
      scaledOverlapExpectation N β h f := by
  let path := concreteRSSmartPathDisorder N β h
  let H := SpinGlass.AT.fullPathHamiltonian path 1
  have hH : Measurable H := by
    exact ((path.sk.hU.repr_measurable.const_smul (Real.sqrt 1)).add
      (path.simple.hV.repr_measurable.const_smul
        (Real.sqrt (1 - 1)))).add measurable_const
  change (∫ x, f x ∂Measure.map
    (fun σs : SpinGlass.AT.Replicas N 2 =>
      Real.sqrt (N : ℝ) *
        SpinGlass.AT.centeredOverlap (SpinGlass.AT.rsQ β h) σs 0 1)
    (SpinGlass.AT.quenchedReplicaMeasure H)) = _
  rw [MeasureTheory.integral_map (by fun_prop) hf.aestronglyMeasurable]
  rw [integral_quenchedReplicaMeasure_eq H hH]
  simp only [scaledOverlapExpectation,
    ν_s_eq_disorderAveragedExpectation β h 1 (by simp)]
  simp [path, H, concreteRSSmartPathDisorder,
    SpinGlass.AT.quenchedReplicaAverage,
    SpinGlass.AT.replicaGibbsAverage,
    disorderAveragedExpectation, productGibbsExpectation,
    SpinGlass.AT.fullPathHamiltonian, GaussianDisorder.toLibrary,
    H_s, Q_ab_eq_centeredReplicaOverlap, centeredReplicaOverlap,
    selectedReplicaOverlap, SpinGlass.AT.centeredOverlap,
    SpinGlass.AT.replicaOverlap, canonicalOverlap, SpinGlass.AT.rsQ]

/-- Weak convergence of the scaled overlap, stated against bounded continuous
real-valued test functions:
$\sqrt N\,(R_{12}-q) \Rightarrow \mathcal N(0,v)$, equivalently
$\nu_1[f(\sqrt N\,Q_{12})] \to \mathbb E[f(\sqrt v\,Z)]$ for every
bounded continuous $f:\mathbb R\to\mathbb R$. -/
def OverlapCLTClaim (β h : ℝ) : Prop :=
  0 < β → 0 < h → α β h < 1 →
    0 < overlapVariance β h ∧
      ∀ f : ℝ → ℝ, Continuous f →
        (∃ M : ℝ, ∀ x : ℝ, |f x| ≤ M) →
        Tendsto
          (fun N : ℕ => scaledOverlapExpectation N β h f)
          atTop
          (nhds (standardGaussianExpectation
            (fun x => f (Real.sqrt (overlapVariance β h) * x))))

/-- The scaled endpoint overlap converges against every bounded continuous
test function to the stated centered Gaussian limit. -/
theorem overlapCLTClaim (β h : ℝ) : OverlapCLTClaim β h := by
  intro hβ hh hAT
  have hα : α β h = SpinGlass.AT.atParameter β h := by rfl
  have hAT' : SpinGlass.AT.atParameter β h < 1 := by
    rwa [← hα]
  have hclt := SpinGlass.AT.overlapCLT_weak hβ hh hAT'
    (fun N => concreteRSSmartPathDisorder N.succ β h)
  have hvariance : overlapVariance β h =
      3 * SpinGlass.AT.rsA β h /
          (1 - SpinGlass.AT.atParameter β h)
        - 2 * SpinGlass.AT.cavityKappa
            (SpinGlass.AT.rsQ β h) (SpinGlass.AT.rsR β h) /
            (1 - β ^ 2 * SpinGlass.AT.cavityKappa
              (SpinGlass.AT.rsQ β h) (SpinGlass.AT.rsR β h))
        - SpinGlass.AT.cavityZeta
            (SpinGlass.AT.rsQ β h) (SpinGlass.AT.rsR β h) /
            (1 - β ^ 2 * SpinGlass.AT.cavityKappa
              (SpinGlass.AT.rsQ β h) (SpinGlass.AT.rsR β h)) ^ 2 := by
    rfl
  rw [← hvariance] at hclt
  refine ⟨hclt.1, ?_⟩
  intro f hf hbounded
  obtain ⟨M, hM⟩ := hbounded
  let fb : BoundedContinuousFunction ℝ ℝ :=
    BoundedContinuousFunction.mkOfBound
    ⟨f, hf⟩ (2 * M) (by
      intro x y
      rw [Real.dist_eq]
      calc
        |f x - f y| ≤ |f x| + |f y| := abs_sub _ _
        _ ≤ M + M := add_le_add (hM x) (hM y)
        _ = 2 * M := by ring)
  have hint :=
    (MeasureTheory.ProbabilityMeasure.tendsto_iff_forall_integral_tendsto.mp
      hclt.2) fb
  have hmap : Measure.map
      (fun x : ℝ => Real.sqrt (overlapVariance β h) * x)
      (gaussianReal 0 1) =
      gaussianReal 0 (overlapVariance β h).toNNReal := by
    rw [ProbabilityTheory.gaussianReal_map_const_mul]
    simp [Real.sq_sqrt hclt.1.le, Real.toNNReal_of_nonneg hclt.1.le]
  have hgaussian :
      (∫ x, fb x ∂(SpinGlass.AT.centeredGaussianLaw
        (overlapVariance β h) : Measure ℝ)) =
        standardGaussianExpectation
          (fun x => f (Real.sqrt (overlapVariance β h) * x)) := by
    change (∫ x, f x ∂gaussianReal 0 (overlapVariance β h).toNNReal) = _
    rw [← hmap, MeasureTheory.integral_map
      (by fun_prop) hf.aestronglyMeasurable]
    rfl
  rw [hgaussian] at hint
  have hfbc : Continuous (fun x => fb x) := fb.continuous
  simp_rw [integral_scaledOverlapLaw_eq_scaledOverlapExpectation
    (f := fun x => fb x) (hf := hfbc)] at hint
  have hshifted : Tendsto
      (fun N : ℕ => scaledOverlapExpectation (N + 1) β h f)
      atTop
      (nhds (standardGaussianExpectation
        (fun x => f (Real.sqrt (overlapVariance β h) * x)))) := by
    simpa [Nat.succ_eq_add_one, fb] using hint
  exact (Filter.tendsto_add_atTop_iff_nat 1).mp hshifted

end

end Main

#print axioms ConcreteMain.quantitativeStrictATClaim
#print axioms ConcreteMain.overlapCLTClaim
