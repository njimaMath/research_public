import Lemmas.Gaussian.ConcreteModel
import Lemmas.CLT.CLT_Main

/-!
# Quantitative strict Almeida-Thouless theorem

This file follows the model-and-main-result portion of
`refs/latalaArgumentsStrictAlmeidaThoulessCondition.tex`.  The Gaussian
construction and the compactness bookkeeping are kept in `Lemmas`.
-/

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

/-! ## The explicit smart path -/

/-- The smart-path Hamiltonian $H_{N,s}$ from equation (path) of the paper.

Here $N\in\mathbb N$, the coordinates $(g_{ij})_{1\leq i,j\leq N}$ and
$(z_i)_{1\leq i\leq N}$ are the canonical independent standard Gaussians,
and $q=q(\beta,h)$ is `canonicalOverlap β h`:

$$
H_{N,s}(\sigma)=\frac{\beta\sqrt{s}}{\sqrt{2N}}
\sum_{i,j=1}^N g_{ij}\sigma_i\sigma_j+
\sum_{i=1}^N\bigl(h+\beta\sqrt{(1-s)q}\,z_i\bigr)\sigma_i.
$$
-/
noncomputable def H_N_s (N : ℕ) (β h s : ℝ)
    (ω : CanonicalGaussianSpace) : SpinGlass.EnergySpace N :=
  WithLp.toLp 2 (fun σ =>
    β * Real.sqrt s / Real.sqrt (2 * (N : ℝ)) *
        ∑ i : Fin N, ∑ j : Fin N,
          canonicalG ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j +
      ∑ i : Fin N,
        (h + β * Real.sqrt ((1 - s) * canonicalOverlap β h) *
          canonicalZ ω i) * SpinGlass.spin N σ i)

/-- The displayed Hamiltonian is the concrete realization used by the proof
backend. -/
theorem H_N_s_eq_smartPath (N : ℕ) (β h s : ℝ)
    (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    H_N_s N β h s = H_s (canonicalSKDisorder N β h) s := by
  have hs' : 0 ≤ 1 - s := by linarith [hs.2]
  have hsqrt :
      Real.sqrt ((1 - s) * canonicalOverlap β h) =
        Real.sqrt (1 - s) * Real.sqrt (canonicalOverlap β h) := by
    exact Real.sqrt_mul hs' _
  funext ω
  ext σ
  rw [H_s_apply]
  simp only [H_N_s, hsqrt]
  simp only [canonicalSKDisorder, GaussianDisorder.canonical]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-! ## Gibbs brackets, replicas, and overlaps -/

/-- The product Gibbs bracket $\langle F\rangle_s$ for `n` replicas at fixed
disorder. -/
noncomputable def gibbsBracket_s {N n : ℕ} (β h s : ℝ)
    (ω : CanonicalGaussianSpace) (F : ReplicaObservable N n) : ℝ :=
  ∑ σs, (∏ a, SpinGlass.gibbs_pmf N (H_N_s N β h s ω) (σs a)) * F σs

/-- The quenched Gibbs expectation $\nu_s[F]=\mathbb E\langle F\rangle_s$. -/
noncomputable def ν_s {N n : ℕ} (β h s : ℝ)
    (F : ReplicaObservable N n) : ℝ :=
  ∫ ω, gibbsBracket_s β h s ω F
    ∂(volume : Measure CanonicalGaussianSpace)

/-- The replica overlap $R_{ab}$. -/
noncomputable def R_ab {N n : ℕ} (σs : ReplicaFamily N n)
    (a b : Fin n) : ℝ :=
  SpinGlass.overlap N (σs a) (σs b)

/-- The centered overlap $Q_{ab}=R_{ab}-q$. -/
noncomputable def Q_ab {N n : ℕ} (β h : ℝ) (σs : ReplicaFamily N n)
    (a b : Fin n) : ℝ :=
  R_ab σs a b - canonicalOverlap β h

/-- $A_s=\nu_s[Q_{12}^2]$. -/
noncomputable def A_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s (fun σs => Q_ab β h σs 0 1 ^ 2)

/-- $B_s=\nu_s[Q_{12}Q_{13}]$. -/
noncomputable def B_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s
    (fun σs => Q_ab β h σs 0 1 * Q_ab β h σs 0 2)

/-- $C_s=\nu_s[Q_{12}Q_{34}]$. -/
noncomputable def C_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ν_s (N := N) (n := 4) β h s
    (fun σs => Q_ab β h σs 0 1 * Q_ab β h σs 2 3)

/-- $\phi_N(s)=N^{-1}\mathbb E\log Z_{N,s}$. -/
noncomputable def φ_N_s (N : ℕ) (β h s : ℝ) : ℝ :=
  ∫ ω, SpinGlass.free_energy_density (N := N) (H_N_s N β h s ω)
    ∂(volume : Measure CanonicalGaussianSpace)

/-- The endpoint free energy $\phi_N(\beta,h)=\phi_N(1)$. -/
noncomputable def φ_N (N : ℕ) (β h : ℝ) : ℝ :=
  φ_N_s N β h 1

/-! ## Main claim -/

/-- The three conclusions of the main theorem in the notation of the paper. -/
structure StrictATClaim (K : Set (ℝ × ℝ)) : Prop where
  overlapConcentration :
    ∃ C_K, 0 ≤ C_K ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h s : ℝ},
      (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 → N * A_s N β h s ≤ C_K
  freeEnergyCorrection :
    ∃ C_K, 0 ≤ C_K ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h : ℝ},
      (β, h) ∈ K →
      0 ≤ replicaSymmetricFreeEnergy β h - φ_N N β h ∧
      replicaSymmetricFreeEnergy β h - φ_N N β h ≤ C_K / N
  repliconSusceptibility :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h s : ℝ},
      (β, h) ∈ K → s ∈ Set.Icc (0 : ℝ) 1 →
      |N * (A_s N β h s - 2 * B_s N β h s + C_s N β h s) -
        stabilityIndex β h /
          (β ^ 2 * (1 - s * stabilityIndex β h))| < eps

/-- Quantitative overlap concentration, free-energy correction, and replicon
susceptibility on every compact subset of the strict AT region. -/
theorem strictAT_main
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictStabilityRegion) :
    StrictATClaim K := by
  have result := quantitative_strictAT
    (Ω := CanonicalGaussianSpace) K hKcompact hKsub
  refine ⟨?_, ?_, ?_⟩
  · obtain ⟨C_K, hC_K, hbound⟩ := result.secondMoment
    refine ⟨C_K, hC_K, ?_⟩
    intro N hN β h s hp hs
    have hresult := hbound hN hp rfl hs (canonicalSKDisorder N β h)
    simpa [A_s, ν_s, gibbsBracket_s, Q_ab, R_ab, overlapVariance,
      centeredReplicaOverlap, selectedReplicaOverlap,
      disorderAveragedExpectation, productGibbsExpectation,
      H_N_s_eq_smartPath N β h s hs] using hresult
  · obtain ⟨C_K, hC_K, hbound⟩ := result.freeEnergy
    refine ⟨C_K, hC_K, ?_⟩
    intro N hN β h hp
    have hresult := hbound hN hp rfl (canonicalSKDisorder N β h)
    have hpath := H_N_s_eq_smartPath N β h 1 (by simp)
    simpa [φ_N, φ_N_s, finiteVolumeFreeEnergy, smartPathFreeEnergy,
      hpath] using hresult
  · intro eps heps
    obtain ⟨N0, hbound⟩ := result.replicon eps heps
    refine ⟨N0, ?_⟩
    intro N hN β h s hp hs
    have hresult := hbound hN hp rfl hs (canonicalSKDisorder N β h)
    have hβ : β ≠ 0 := ne_of_gt (hKsub hp).1
    have hratio :
        stabilityIndex β h / (β ^ 2 * (1 - s * stabilityIndex β h)) =
          canonicalSechFourthMoment β h /
            (1 - s * stabilityIndex β h) := by
      rw [stabilityIndex]
      field_simp [hβ]
    rw [hratio]
    simpa [A_s, B_s, C_s, ν_s, gibbsBracket_s, Q_ab, R_ab,
      overlapVariance, sharedReplicaMoment, disjointReplicaMoment,
      centeredReplicaOverlap, selectedReplicaOverlap,
      disorderAveragedExpectation, productGibbsExpectation,
      H_N_s_eq_smartPath N β h s hs] using hresult

/-! ## Weak convergence of the overlap -/

/-- The canonical Gaussian disorder, viewed through the abstract smart-path
interface at the replica-symmetric fixed point. -/
noncomputable def canonicalRSSmartPathDisorder (N : ℕ) (β h : ℝ) :
    SpinGlass.AT.RSSmartPathDisorder CanonicalGaussianSpace N β h
      (SpinGlass.AT.rsQ β h) := by
  simpa [canonicalOverlap, SpinGlass.AT.rsQ] using
    (canonicalSKDisorder N β h).toLibrary

/-- The scaled overlap at the endpoint of the canonical SK smart path
converges weakly to its centered Gaussian limit in the strict AT region. -/
theorem strictAT_overlapCLT_weak
    {β h : ℝ}
    (hβ : 0 < β)
    (hh : 0 < h)
    (hAT : SpinGlass.AT.atParameter β h < 1) :
    let σ2 : ℝ :=
      3 * SpinGlass.AT.rsA β h / (1 - SpinGlass.AT.atParameter β h)
        - 2 * SpinGlass.AT.cavityKappa (SpinGlass.AT.rsQ β h)
            (SpinGlass.AT.rsR β h) /
            (1 - β ^ 2 * SpinGlass.AT.cavityKappa (SpinGlass.AT.rsQ β h)
              (SpinGlass.AT.rsR β h))
        - SpinGlass.AT.cavityZeta (SpinGlass.AT.rsQ β h)
            (SpinGlass.AT.rsR β h) /
            (1 - β ^ 2 * SpinGlass.AT.cavityKappa (SpinGlass.AT.rsQ β h)
              (SpinGlass.AT.rsR β h)) ^ 2
    0 < σ2 ∧
      Filter.Tendsto
        (fun N : ℕ => SpinGlass.AT.scaledOverlapLaw
          (canonicalRSSmartPathDisorder N.succ β h))
        Filter.atTop
        (nhds (SpinGlass.AT.centeredGaussianLaw σ2)) := by
  exact SpinGlass.AT.overlapCLT_weak hβ hh hAT
    (fun N => canonicalRSSmartPathDisorder N.succ β h)
