import Lemmas.Gaussian.CanonicalModel

/-!
# Quantitative strict Almeida-Thouless theorem

This entry module defines every model-specific object occurring in its public
theorem.  The implementation in `Lemmas` is used only to discharge the proof.
-/

open MeasureTheory ProbabilityTheory BigOperators

set_option autoImplicit false

universe u

/-! ## Public model setup -/

/-- The canonical replica-symmetric fixed point. -/
noncomputable def canonicalOverlap (β h : ℝ) : ℝ :=
  sInf {q : ℝ | q ∈ Set.Icc (0 : ℝ) 1 ∧
    SpinGlass.AT.IsRSFixedPoint β h q}

/-- The fourth local-magnetization moment at the canonical fixed point. -/
noncomputable def canonicalFourthMoment (β h : ℝ) : ℝ :=
  SpinGlass.AT.standardGaussianExpectation
    (fun z => Real.tanh (h + β * Real.sqrt (canonicalOverlap β h) * z) ^ 4)

/-- The fourth hyperbolic-secant moment. -/
noncomputable def canonicalSechFourthMoment (β h : ℝ) : ℝ :=
  1 - 2 * canonicalOverlap β h + canonicalFourthMoment β h

/-- The Almeida-Thouless stability parameter. -/
noncomputable def stabilityIndex (β h : ℝ) : ℝ :=
  β ^ 2 * canonicalSechFourthMoment β h

/-- Positive-temperature, positive-field parameters below the AT line. -/
def strictStabilityRegion : Set (ℝ × ℝ) :=
  {p | 0 < p.1 ∧ 0 < p.2 ∧ stabilityIndex p.1 p.2 < 1}

/-- An indexed family of spin replicas. -/
abbrev ReplicaFamily (N n : ℕ) := Fin n → SpinGlass.Config N

/-- A real-valued observable of finitely many replicas. -/
abbrev ReplicaObservable (N n : ℕ) := ReplicaFamily N n → ℝ

/-- Product Gibbs expectation of a replica observable. -/
noncomputable def productGibbsExpectation {N n : ℕ}
    (H : SpinGlass.EnergySpace N) (F : ReplicaObservable N n) : ℝ :=
  ∑ σs, (∏ a, SpinGlass.gibbs_pmf N H (σs a)) * F σs

/-- Disorder-averaged product Gibbs expectation. -/
noncomputable def disorderAveragedExpectation
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → SpinGlass.EnergySpace N)
    (F : ReplicaObservable N n) : ℝ :=
  ∫ ω, productGibbsExpectation (H ω) F ∂(volume : Measure Ω)

/-- The overlap between two selected replicas. -/
noncomputable def selectedReplicaOverlap {N n : ℕ}
    (σs : ReplicaFamily N n) (a b : Fin n) : ℝ :=
  SpinGlass.overlap N (σs a) (σs b)

/-- A selected replica overlap centered at `q`. -/
noncomputable def centeredReplicaOverlap {N n : ℕ} (q : ℝ)
    (σs : ReplicaFamily N n) (a b : Fin n) : ℝ :=
  selectedReplicaOverlap σs a b - q

/-- The centered SK term built from the coordinates $g_{ij}$. -/
noncomputable def coordinateSKEnergy {Ω : Type u} {N : ℕ}
    (β : ℝ) (g : Ω → Fin N → Fin N → ℝ) (ω : Ω) :
    SpinGlass.EnergySpace N :=
  WithLp.toLp 2 (fun σ =>
    β / Real.sqrt (2 * (N : ℝ)) *
      ∑ i : Fin N, ∑ j : Fin N,
        g ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j)

/-- The centered random-field term built from the coordinates $z_i$. -/
noncomputable def coordinateFieldEnergy {Ω : Type u} {N : ℕ}
    (β q : ℝ) (z : Ω → Fin N → ℝ) (ω : Ω) :
    SpinGlass.EnergySpace N :=
  WithLp.toLp 2 (fun σ =>
    β * Real.sqrt q * ∑ i : Fin N, z ω i * SpinGlass.spin N σ i)

/-! ## Canonical iid Gaussian coordinates -/

/-- One probability space carrying all SK and auxiliary-field coordinates. -/
abbrev CanonicalGaussianSpace := ((ℕ × ℕ) ⊕ ℕ) → ℝ

/-- The countable product standard Gaussian measure. -/
noncomputable def canonicalGaussianMeasure : Measure CanonicalGaussianSpace :=
  Measure.infinitePi (fun _ : (ℕ × ℕ) ⊕ ℕ => gaussianReal 0 1)

noncomputable instance : MeasureSpace CanonicalGaussianSpace :=
  ⟨canonicalGaussianMeasure⟩

noncomputable instance :
    IsProbabilityMeasure (volume : Measure CanonicalGaussianSpace) := by
  change IsProbabilityMeasure canonicalGaussianMeasure
  unfold canonicalGaussianMeasure
  infer_instance

/-- The canonical SK coordinate $g_{ij}$. -/
def canonicalG (ω : CanonicalGaussianSpace) (i j : ℕ) : ℝ :=
  ω (Sum.inl (i, j))

/-- The canonical auxiliary-field coordinate $z_i$. -/
def canonicalZ (ω : CanonicalGaussianSpace) (i : ℕ) : ℝ :=
  ω (Sum.inr i)

private noncomputable def coordinateSKLinearMap (N : ℕ) (β : ℝ) :
    (Fin N × Fin N → ℝ) →L[ℝ] SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 (fun σ =>
        β / Real.sqrt (2 * (N : ℝ)) *
          ∑ ij : Fin N × Fin N,
            x ij * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2)
      map_add' := by
        intro x y
        ext σ
        change β / Real.sqrt (2 * (N : ℝ)) *
            ∑ ij, (x ij + y ij) * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2 =
          β / Real.sqrt (2 * (N : ℝ)) *
              ∑ ij, x ij * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2 +
            β / Real.sqrt (2 * (N : ℝ)) *
              ∑ ij, y ij * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2
        simp_rw [add_mul, Finset.sum_add_distrib]
        ring
      map_smul' := by
        intro a x
        ext σ
        change β / Real.sqrt (2 * (N : ℝ)) *
            ∑ ij, (a * x ij) * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2 =
          a * (β / Real.sqrt (2 * (N : ℝ)) *
            ∑ ij, x ij * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2)
        simp_rw [mul_assoc, ← Finset.mul_sum]
        ring }

private noncomputable def coordinateFieldLinearMap (N : ℕ) (β q : ℝ) :
    (Fin N → ℝ) →L[ℝ] SpinGlass.EnergySpace N :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 (fun σ =>
        β * Real.sqrt q * ∑ i : Fin N, x i * SpinGlass.spin N σ i)
      map_add' := by
        intro x y
        ext σ
        change β * Real.sqrt q *
            ∑ i, (x i + y i) * SpinGlass.spin N σ i =
          β * Real.sqrt q * ∑ i, x i * SpinGlass.spin N σ i +
            β * Real.sqrt q * ∑ i, y i * SpinGlass.spin N σ i
        simp_rw [add_mul, Finset.sum_add_distrib]
        ring
      map_smul' := by
        intro a x
        ext σ
        change β * Real.sqrt q * ∑ i, (a * x i) * SpinGlass.spin N σ i =
          a * (β * Real.sqrt q * ∑ i, x i * SpinGlass.spin N σ i)
        simp_rw [mul_assoc, ← Finset.mul_sum]
        ring }

private theorem standardGaussianCoordinate_hasGaussianLaw
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (X : Ω → ℝ) (hXm : Measurable X)
    (hX : Measure.map X (volume : Measure Ω) = gaussianReal 0 1) :
    HasGaussianLaw X (volume : Measure Ω) := by
  exact HasLaw.hasGaussianLaw
    { aemeasurable := hXm.aemeasurable
      map_eq := hX }

private theorem standardGaussianCoordinate_integral
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (X : Ω → ℝ) (hXm : Measurable X)
    (hX : Measure.map X (volume : Measure Ω) = gaussianReal 0 1) :
    ∫ ω, X ω ∂(volume : Measure Ω) = 0 := by
  calc
    ∫ ω, X ω ∂(volume : Measure Ω) =
        ∫ x, id x ∂Measure.map X (volume : Measure Ω) := by
          symm
          exact integral_map hXm.aemeasurable measurable_id.aestronglyMeasurable
    _ = 0 := by rw [hX]; simp

private noncomputable def coordinateSKEnergy_isGaussian
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (β : ℝ) (g : Ω → Fin N → Fin N → ℝ)
    (hgm : ∀ i j, Measurable (fun ω => g ω i j))
    (hgstd : ∀ i j,
      Measure.map (fun ω => g ω i j) (volume : Measure Ω) = gaussianReal 0 1)
    (hgind : iIndepFun
      (fun ij : Fin N × Fin N => fun ω => g ω ij.1 ij.2)
      (volume : Measure Ω)) :
    PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
      (coordinateSKEnergy β g) := by
  let X : Ω → (Fin N × Fin N → ℝ) := fun ω ij => g ω ij.1 ij.2
  have hXm : Measurable X := measurable_pi_iff.mpr fun ij => hgm ij.1 ij.2
  have hXg : HasGaussianLaw X (volume : Measure Ω) := by
    exact hgind.hasGaussianLaw fun ij =>
      standardGaussianCoordinate_hasGaussianLaw
        (fun ω => g ω ij.1 ij.2) (hgm ij.1 ij.2) (hgstd ij.1 ij.2)
  have hrepr : coordinateSKEnergy β g = coordinateSKLinearMap N β ∘ X := by
    funext ω
    ext σ
    change β / Real.sqrt (2 * (N : ℝ)) *
        ∑ i : Fin N, ∑ j : Fin N,
          g ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j =
      β / Real.sqrt (2 * (N : ℝ)) *
        ∑ ij : Fin N × Fin N,
          g ω ij.1 ij.2 * SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2
    rw [← Finset.sum_product', Finset.univ_product_univ]
  have hEmeas : Measurable (coordinateSKEnergy β g) := by
    rw [hrepr]
    exact (coordinateSKLinearMap N β).measurable.comp hXm
  have hEg : HasGaussianLaw (coordinateSKEnergy β g)
      (volume : Measure Ω) := by
    rw [hrepr]
    exact hXg.map (coordinateSKLinearMap N β)
  have hX0 : ∫ ω, X ω ∂(volume : Measure Ω) = 0 := by
    ext ij
    rw [MeasureTheory.eval_integral (fun k => hXg.integrable.eval k)]
    exact standardGaussianCoordinate_integral
      (fun ω => g ω ij.1 ij.2) (hgm ij.1 ij.2) (hgstd ij.1 ij.2)
  have hE0 : ∫ ω, coordinateSKEnergy β g ω ∂(volume : Measure Ω) = 0 := by
    rw [hrepr]
    change (∫ ω, coordinateSKLinearMap N β (X ω) ∂(volume : Measure Ω)) = 0
    calc
      _ = coordinateSKLinearMap N β
          (∫ ω, X ω ∂(volume : Measure Ω)) :=
        (coordinateSKLinearMap N β).integral_comp_comm hXg.integrable
      _ = 0 := by rw [hX0]; simp
  exact PhysLean.Probability.GaussianIBP.IsGaussianHilbert.of_hasGaussianLaw
    (coordinateSKEnergy β g) hEmeas hEg hE0

private noncomputable def coordinateFieldEnergy_isGaussian
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (β q : ℝ) (z : Ω → Fin N → ℝ)
    (hzm : ∀ i, Measurable (fun ω => z ω i))
    (hzstd : ∀ i,
      Measure.map (fun ω => z ω i) (volume : Measure Ω) = gaussianReal 0 1)
    (hzind : iIndepFun (fun i : Fin N => fun ω => z ω i)
      (volume : Measure Ω)) :
    PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
      (coordinateFieldEnergy β q z) := by
  let X : Ω → (Fin N → ℝ) := fun ω i => z ω i
  have hXm : Measurable X := measurable_pi_iff.mpr hzm
  have hXg : HasGaussianLaw X (volume : Measure Ω) := by
    exact hzind.hasGaussianLaw fun i =>
      standardGaussianCoordinate_hasGaussianLaw
        (fun ω => z ω i) (hzm i) (hzstd i)
  have hrepr : coordinateFieldEnergy β q z =
      coordinateFieldLinearMap N β q ∘ X := by
    rfl
  have hEmeas : Measurable (coordinateFieldEnergy β q z) := by
    rw [hrepr]
    exact (coordinateFieldLinearMap N β q).measurable.comp hXm
  have hEg : HasGaussianLaw (coordinateFieldEnergy β q z)
      (volume : Measure Ω) := by
    rw [hrepr]
    exact hXg.map (coordinateFieldLinearMap N β q)
  have hX0 : ∫ ω, X ω ∂(volume : Measure Ω) = 0 := by
    ext i
    rw [MeasureTheory.eval_integral (fun k => hXg.integrable.eval k)]
    exact standardGaussianCoordinate_integral (fun ω => z ω i) (hzm i) (hzstd i)
  have hE0 : ∫ ω, coordinateFieldEnergy β q z ω ∂(volume : Measure Ω) = 0 := by
    rw [hrepr]
    change (∫ ω, coordinateFieldLinearMap N β q (X ω) ∂(volume : Measure Ω)) = 0
    calc
      _ = coordinateFieldLinearMap N β q
          (∫ ω, X ω ∂(volume : Measure Ω)) :=
        (coordinateFieldLinearMap N β q).integral_comp_comm hXg.integrable
      _ = 0 := by rw [hX0]; simp
  exact PhysLean.Probability.GaussianIBP.IsGaussianHilbert.of_hasGaussianLaw
    (coordinateFieldEnergy β q z) hEmeas hEg hE0

private theorem iidStandardGaussian_secondMoment
    {Ω I : Type*} [DecidableEq I] [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (X : I → Ω → ℝ) (hXm : ∀ i, Measurable (X i))
    (hXstd : ∀ i, Measure.map (X i) (volume : Measure Ω) = gaussianReal 0 1)
    (hXind : iIndepFun X (volume : Measure Ω)) (i j : I) :
    ∫ ω, X i ω * X j ω ∂(volume : Measure Ω) = if i = j then 1 else 0 := by
  classical
  by_cases hij : i = j
  · subst j
    rw [if_pos rfl]
    simp only [← pow_two]
    have hmap := integral_map (μ := (volume : Measure Ω))
      (hXm i).aemeasurable
      (show AEStronglyMeasurable (fun x : ℝ => x ^ 2)
          (Measure.map (X i) (volume : Measure Ω)) by fun_prop)
    rw [hXstd i] at hmap
    simpa using hmap.symm.trans (integral_sq_gaussianReal_centered (v := 1))
  · rw [if_neg hij]
    have hmul := (hXind.indepFun hij).integral_mul_eq_mul_integral
      (hXm i).aestronglyMeasurable (hXm j).aestronglyMeasurable
    have hmean (k : I) : ∫ ω, X k ω ∂(volume : Measure Ω) = 0 :=
      standardGaussianCoordinate_integral (X k) (hXm k) (hXstd k)
    simpa [hmean] using hmul

private theorem coordinateSKEnergy_cov_eq
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (β : ℝ) (g : Ω → Fin N → Fin N → ℝ)
    (hgm : ∀ i j, Measurable (fun ω => g ω i j))
    (hgstd : ∀ i j,
      Measure.map (fun ω => g ω i j) (volume : Measure Ω) = gaussianReal 0 1)
    (hgind : iIndepFun
      (fun ij : Fin N × Fin N => fun ω => g ω ij.1 ij.2)
      (volume : Measure Ω))
    (hG : PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
      (coordinateSKEnergy β g)) (σ τ : SpinGlass.Config N) :
    inner ℝ
        ((PhysLean.Probability.GaussianIBP.covOp hG) (SpinGlass.std_basis N σ))
        (SpinGlass.std_basis N τ) = SpinGlass.sk_cov_kernel N β σ τ := by
  classical
  rw [← SpinGlass.GeneralizedLatala.gaussianHilbert_eval_pairing
    N (coordinateSKEnergy β g) hG σ τ]
  let a : ℝ := β / Real.sqrt (2 * (N : ℝ))
  let sσ : Fin N × Fin N → ℝ := fun ij =>
    SpinGlass.spin N σ ij.1 * SpinGlass.spin N σ ij.2
  let sτ : Fin N × Fin N → ℝ := fun ij =>
    SpinGlass.spin N τ ij.1 * SpinGlass.spin N τ ij.2
  have hcoord (ij kl : Fin N × Fin N) :
      ∫ ω, g ω ij.1 ij.2 * g ω kl.1 kl.2 ∂(volume : Measure Ω) =
        if ij = kl then 1 else 0 :=
    iidStandardGaussian_secondMoment
      (fun ij : Fin N × Fin N => fun ω => g ω ij.1 ij.2)
      (fun ij => hgm ij.1 ij.2) (fun ij => hgstd ij.1 ij.2) hgind ij kl
  have hterm (ij kl : Fin N × Fin N) : Integrable
      (fun ω => a ^ 2 * sσ ij * sτ kl *
        (g ω ij.1 ij.2 * g ω kl.1 kl.2)) (volume : Measure Ω) := by
    have hi := standardGaussianCoordinate_hasGaussianLaw
      (fun ω => g ω ij.1 ij.2) (hgm ij.1 ij.2) (hgstd ij.1 ij.2)
    have hk := standardGaussianCoordinate_hasGaussianLaw
      (fun ω => g ω kl.1 kl.2) (hgm kl.1 kl.2) (hgstd kl.1 kl.2)
    exact (hi.memLp_two.integrable_mul hk.memLp_two).const_mul _
  have hprod (ω : Ω) :
      coordinateSKEnergy β g ω σ * coordinateSKEnergy β g ω τ =
        ∑ kl : Fin N × Fin N, ∑ ij : Fin N × Fin N,
          a ^ 2 * sσ ij * sτ kl *
            (g ω ij.1 ij.2 * g ω kl.1 kl.2) := by
    simp only [coordinateSKEnergy, a, sσ, sτ]
    rw [← Finset.sum_product', Finset.univ_product_univ]
    rw [← Finset.sum_product', Finset.univ_product_univ]
    simp only [Finset.sum_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro kl _
    apply Finset.sum_congr rfl
    intro ij _
    ring
  simp_rw [hprod]
  rw [integral_finset_sum _
    (fun kl _ => integrable_finset_sum _ (fun ij _ => hterm ij kl))]
  simp_rw [integral_finset_sum _ (fun ij _ => hterm ij _)]
  simp_rw [integral_const_mul, hcoord]
  simp
  by_cases hN : N = 0
  · subst N
    simp [SpinGlass.sk_cov_kernel, SpinGlass.overlap, a, sσ, sτ]
  · have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast hN
    have hsqrt : Real.sqrt (2 * (N : ℝ)) ^ 2 = 2 * (N : ℝ) := by
      rw [Real.sq_sqrt]
      positivity
    have hpair : ∑ ij : Fin N × Fin N, sσ ij * sτ ij =
        (∑ i : Fin N, SpinGlass.spin N σ i * SpinGlass.spin N τ i) ^ 2 := by
      simp only [sσ, sτ]
      rw [← Finset.univ_product_univ, Finset.sum_product, pow_two,
        Finset.sum_mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      ring
    simp_rw [mul_assoc]
    rw [← Finset.mul_sum, hpair]
    simp only [SpinGlass.sk_cov_kernel, SpinGlass.overlap, a]
    field_simp [hNr, hsqrt]
    rw [hsqrt]
    ring

private theorem coordinateFieldEnergy_cov_eq
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (β q : ℝ) (z : Ω → Fin N → ℝ)
    (hzm : ∀ i, Measurable (fun ω => z ω i))
    (hzstd : ∀ i,
      Measure.map (fun ω => z ω i) (volume : Measure Ω) = gaussianReal 0 1)
    (hzind : iIndepFun (fun i : Fin N => fun ω => z ω i)
      (volume : Measure Ω))
    (hG : PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
      (coordinateFieldEnergy β q z)) (hq : 0 ≤ q)
    (σ τ : SpinGlass.Config N) :
    inner ℝ
        ((PhysLean.Probability.GaussianIBP.covOp hG) (SpinGlass.std_basis N σ))
        (SpinGlass.std_basis N τ) =
      SpinGlass.simple_cov_kernel N β (fun x => q * x) σ τ := by
  classical
  rw [← SpinGlass.GeneralizedLatala.gaussianHilbert_eval_pairing
    N (coordinateFieldEnergy β q z) hG σ τ]
  have hcoord (i j : Fin N) :
      ∫ ω, z ω i * z ω j ∂(volume : Measure Ω) = if i = j then 1 else 0 :=
    iidStandardGaussian_secondMoment (fun i => fun ω => z ω i)
      hzm hzstd hzind i j
  have hterm (i j : Fin N) : Integrable
      (fun ω => (β * Real.sqrt q) ^ 2 *
        SpinGlass.spin N σ i * SpinGlass.spin N τ j * (z ω i * z ω j))
      (volume : Measure Ω) := by
    have hi := standardGaussianCoordinate_hasGaussianLaw
      (fun ω => z ω i) (hzm i) (hzstd i)
    have hj := standardGaussianCoordinate_hasGaussianLaw
      (fun ω => z ω j) (hzm j) (hzstd j)
    exact (hi.memLp_two.integrable_mul hj.memLp_two).const_mul _
  have hprod (ω : Ω) :
      coordinateFieldEnergy β q z ω σ * coordinateFieldEnergy β q z ω τ =
        ∑ j : Fin N, ∑ i : Fin N, (β * Real.sqrt q) ^ 2 *
          SpinGlass.spin N σ i * SpinGlass.spin N τ j * (z ω i * z ω j) := by
    simp only [coordinateFieldEnergy, Finset.sum_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro i _
    ring
  simp_rw [hprod]
  rw [integral_finset_sum _
    (fun j _ => integrable_finset_sum _ (fun i _ => hterm i j))]
  simp_rw [integral_finset_sum _ (fun i _ => hterm i _)]
  simp_rw [integral_const_mul, hcoord]
  simp
  simp only [SpinGlass.simple_cov_kernel, SpinGlass.overlap]
  rw [mul_pow, Real.sq_sqrt hq]
  by_cases hN : N = 0
  · subst N
    simp
  · have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast hN
    have hsum :
        (∑ i : Fin N, β ^ 2 * q *
          SpinGlass.spin N σ i * SpinGlass.spin N τ i) =
          β ^ 2 * q *
            ∑ i : Fin N, SpinGlass.spin N σ i * SpinGlass.spin N τ i := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    rw [hsum]
    field_simp [hNr]

/-- The concrete Gaussian coordinates used in the Hamiltonian.

The fields certify directly that the displayed coordinate energies are the
Gaussian SK and random-field disorders required by the general theorem. -/
structure GaussianDisorder (Ω : Type u) [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (N : ℕ) (β h q : ℝ) where
  g : Ω → Fin N → Fin N → ℝ
  z : Ω → Fin N → ℝ
  g_standardGaussian : ∀ i j,
    Measure.map (fun ω => g ω i j) (volume : Measure Ω) = gaussianReal 0 1
  z_standardGaussian : ∀ i,
    Measure.map (fun ω => z ω i) (volume : Measure Ω) = gaussianReal 0 1
  g_independent : ProbabilityTheory.iIndepFun
    (fun ij : Fin N × Fin N => fun ω => g ω ij.1 ij.2)
    (volume : Measure Ω)
  z_independent : ProbabilityTheory.iIndepFun
    (fun i : Fin N => fun ω => z ω i) (volume : Measure Ω)
  coordinateFamiliesIndependent : ProbabilityTheory.IndepFun g z
    (volume : Measure Ω)
  skGaussian : PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
    (coordinateSKEnergy β g)
  sk_cov_eq : ∀ σ τ,
    inner ℝ
        ((PhysLean.Probability.GaussianIBP.covOp skGaussian)
          (SpinGlass.std_basis N σ))
        (SpinGlass.std_basis N τ) = SpinGlass.sk_cov_kernel N β σ τ
  fieldGaussian : PhysLean.Probability.GaussianIBP.IsGaussianHilbert.{u, 0, 0}
    (coordinateFieldEnergy β q z)
  field_cov_eq : ∀ σ τ,
    inner ℝ
        ((PhysLean.Probability.GaussianIBP.covOp fieldGaussian)
          (SpinGlass.std_basis N σ))
        (SpinGlass.std_basis N τ) =
      SpinGlass.simple_cov_kernel N β (fun x => q * x) σ τ
  independent : ProbabilityTheory.IndepFun
    (coordinateSKEnergy β g) (coordinateFieldEnergy β q z)
    (volume : Measure Ω)

namespace GaussianDisorder

private noncomputable def toLibrary {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (disorder : GaussianDisorder Ω N β h q) :
    SpinGlass.AT.RSSmartPathDisorder Ω N β h q :=
  { sk :=
      { U := coordinateSKEnergy β disorder.g
        hU := disorder.skGaussian
        cov_eq := disorder.sk_cov_eq }
    simple :=
      { V := coordinateFieldEnergy β q disorder.z
        hV := disorder.fieldGaussian
        cov_eq := disorder.field_cov_eq }
    independent := disorder.independent }

end GaussianDisorder

/-- The specific smart-path Hamiltonian $H_s(σ)$, including the external
field.  By `H_s_apply` below, this is exactly
$\frac{β\sqrt{s}}{\sqrt{2N}}\sum_{i,j}g_{ij}σ_iσ_j
 +\sum_i(h+β\sqrt{1-s}\sqrt q\,z_i)σ_i$. -/
noncomputable def H_s {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (disorder : GaussianDisorder Ω N β h q)
    (s : ℝ) (ω : Ω) : SpinGlass.EnergySpace N :=
  Real.sqrt s • coordinateSKEnergy β disorder.g ω +
    Real.sqrt (1 - s) • coordinateFieldEnergy β q disorder.z ω +
    SpinGlass.magnetic_field_vector N h

theorem H_s_apply {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (disorder : GaussianDisorder Ω N β h q)
    (s : ℝ) (ω : Ω) (σ : SpinGlass.Config N) :
    H_s disorder s ω σ =
      β * Real.sqrt s / Real.sqrt (2 * (N : ℝ)) *
          ∑ i : Fin N, ∑ j : Fin N,
            disorder.g ω i j * SpinGlass.spin N σ i * SpinGlass.spin N σ j +
        ∑ i : Fin N,
          (h + β * Real.sqrt (1 - s) * Real.sqrt q * disorder.z ω i) *
            SpinGlass.spin N σ i := by
  rw [H_s]
  simp [SpinGlass.magnetic_field_vector, SpinGlass.magnetization,
    coordinateSKEnergy, coordinateFieldEnergy, Finset.mul_sum]
  simp_rw [add_mul, Finset.sum_add_distrib]
  ring

/-- The quenched free-energy density along the smart path. -/
noncomputable def smartPathFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : GaussianDisorder Ω N β h q) (s : ℝ) : ℝ :=
  ∫ ω, SpinGlass.free_energy_density
      (N := N) (H_s path s ω)
    ∂(volume : Measure Ω)

/-- The replica-symmetric free energy at the canonical fixed point. -/
noncomputable def replicaSymmetricFreeEnergy (β h : ℝ) : ℝ :=
  Real.log 2 + SpinGlass.AT.standardGaussianExpectation
    (fun z => Real.log
      (Real.cosh (h + β * Real.sqrt (canonicalOverlap β h) * z))) +
    β ^ 2 / 4 * (1 - canonicalOverlap β h) ^ 2

/-- The finite-volume SK free energy at the endpoint of the smart path. -/
noncomputable def finiteVolumeFreeEnergy {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : GaussianDisorder Ω N β h q) : ℝ :=
  smartPathFreeEnergy path 1

/-- The second centered-overlap moment. -/
noncomputable def overlapVariance {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : GaussianDisorder Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 => centeredReplicaOverlap q σs 0 1 ^ 2)

/-- The centered-overlap moment for two pairs sharing one replica. -/
noncomputable def sharedReplicaMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : GaussianDisorder Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 =>
      centeredReplicaOverlap q σs 0 1 * centeredReplicaOverlap q σs 0 2)

/-- The centered-overlap moment for two disjoint replica pairs. -/
noncomputable def disjointReplicaMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q : ℝ} (path : GaussianDisorder Ω N β h q) (s : ℝ) : ℝ :=
  disorderAveragedExpectation (H_s path s)
    (fun σs : ReplicaFamily N 4 =>
      centeredReplicaOverlap q σs 0 1 * centeredReplicaOverlap q σs 2 3)

/-! ## Quantitative conclusion -/

structure QuantitativeAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] (K : Set (ℝ × ℝ)) : Prop where
  secondMoment :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : GaussianDisorder Ω N β h q, N * overlapVariance path s ≤ M
  freeEnergy :
    ∃ M, 0 ≤ M ∧ ∀ {N : ℕ}, 0 < N → ∀ {β h q : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h →
      ∀ path : GaussianDisorder Ω N β h q,
      0 ≤ replicaSymmetricFreeEnergy β h - finiteVolumeFreeEnergy path ∧
      replicaSymmetricFreeEnergy β h - finiteVolumeFreeEnergy path ≤ M / N
  replicon :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = canonicalOverlap β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : GaussianDisorder Ω N β h q,
      |N * (overlapVariance path s - 2 * sharedReplicaMoment path s +
          disjointReplicaMoment path s) -
        canonicalSechFourthMoment β h / (1 - s * stabilityIndex β h)| < eps

/-- Quantitative strict-AT theorem on a compact subset of the positive-field
strict stability region. -/
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictStabilityRegion) :
    QuantitativeAT (Ω := Ω) K := by
  have hKsub' : K ⊆ SpinGlass.AT.strictATRegion := by
    intro p hp
    simpa [strictStabilityRegion, stabilityIndex,
      canonicalSechFourthMoment, canonicalFourthMoment, canonicalOverlap,
      SpinGlass.AT.strictATRegion, SpinGlass.AT.atParameter,
      SpinGlass.AT.rsA, SpinGlass.AT.rsR, SpinGlass.AT.rsQ] using hKsub hp
  have result : SpinGlass.AT.QuantitativeATConclusion (Ω := Ω) K := by
    by_cases hKne : K.Nonempty
    · obtain ⟨pβ, hpβ, hβmax⟩ :=
        hKcompact.exists_isMaxOn hKne
          (continuousOn_fst : ContinuousOn (fun p : ℝ × ℝ => p.1) K)
      have hqcont : ContinuousOn
          (fun p : ℝ × ℝ => SpinGlass.AT.rsQ p.1 p.2) K :=
        (SpinGlass.AT.continuousOn_rsParameters_of_subset_strictATRegion
          hKsub').1
      obtain ⟨pq, hpq, hqmin⟩ :=
        hKcompact.exists_isMinOn hKne hqcont
      obtain ⟨gap, hgap_pos, hgap_lower⟩ :=
        SpinGlass.AT.exists_uniform_at_gap_on_compact
          hKcompact hKne hKsub'
      let data : SpinGlass.AT.UniformATData K :=
        { isCompact := hKcompact
          βmax := pβ.1
          qmin := SpinGlass.AT.rsQ pq.1 pq.2
          gap := gap
          βmax_pos := (hKsub' hpβ).1
          qmin_pos := SpinGlass.AT.rsQ_pos
            (hKsub' hpq).1 (hKsub' hpq).2.1
          gap_pos := hgap_pos
          β_pos := fun p hp => (hKsub' hp).1
          h_pos := fun p hp => (hKsub' hp).2.1
          β_bound := fun p hp => hβmax hp
          q_lower := fun p hp => hqmin hp
          strictAT := by
            intro p hp
            have hgap := hgap_lower p hp
            linarith }
      exact SpinGlass.AT.quantitative_strictAT K data
    · let data : SpinGlass.AT.UniformATData K :=
        { isCompact := hKcompact
          βmax := 1
          qmin := 1
          gap := 1
          βmax_pos := by norm_num
          qmin_pos := by norm_num
          gap_pos := by norm_num
          β_pos := fun p hp => (hKsub' hp).1
          h_pos := fun p hp => (hKsub' hp).2.1
          β_bound := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim
          q_lower := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim
          strictAT := by
            intro p hp
            exact (hKne ⟨p, hp⟩).elim }
      exact SpinGlass.AT.quantitative_strictAT K data
  refine { secondMoment := ?_, freeEnergy := ?_, replicon := ?_ }
  · obtain ⟨M, hM, hbound⟩ := result.secondMoment
    refine ⟨M, hM, ?_⟩
    intro N hN β h q s hK hq hs path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' hs path.toLibrary
    simpa [overlapVariance, disorderAveragedExpectation,
      productGibbsExpectation, H_s,
      centeredReplicaOverlap, selectedReplicaOverlap, GaussianDisorder.toLibrary,
      SpinGlass.AT.A, SpinGlass.AT.quenchedReplicaAverage,
      SpinGlass.AT.replicaGibbsAverage, SpinGlass.AT.fullPathHamiltonian,
      SpinGlass.AT.centeredOverlap, SpinGlass.AT.replicaOverlap] using h
  · obtain ⟨M, hM, hbound⟩ := result.freeEnergy
    refine ⟨M, hM, ?_⟩
    intro N hN β h q hK hq path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' path.toLibrary
    simpa [replicaSymmetricFreeEnergy, finiteVolumeFreeEnergy,
      smartPathFreeEnergy, H_s, canonicalOverlap,
      GaussianDisorder.toLibrary, SpinGlass.AT.rsFreeEnergy,
      SpinGlass.AT.rsPathValue, SpinGlass.AT.skFreeEnergy,
      SpinGlass.AT.pathFreeEnergy, SpinGlass.AT.rsQ,
      SpinGlass.AT.fullPathHamiltonian] using h
  · intro eps heps
    obtain ⟨N0, hbound⟩ := result.replicon eps heps
    refine ⟨N0, ?_⟩
    intro N hN β h q s hK hq hs path
    have hq' : q = SpinGlass.AT.rsQ β h := by
      simpa [canonicalOverlap, SpinGlass.AT.rsQ] using hq
    have h := hbound hN hK hq' hs path.toLibrary
    simpa [overlapVariance, sharedReplicaMoment, disjointReplicaMoment,
      disorderAveragedExpectation, productGibbsExpectation,
      H_s, centeredReplicaOverlap, selectedReplicaOverlap,
      canonicalSechFourthMoment, canonicalFourthMoment, canonicalOverlap,
      stabilityIndex, GaussianDisorder.toLibrary, SpinGlass.AT.A, SpinGlass.AT.B,
      SpinGlass.AT.C, SpinGlass.AT.quenchedReplicaAverage,
      SpinGlass.AT.replicaGibbsAverage, SpinGlass.AT.fullPathHamiltonian,
      SpinGlass.AT.centeredOverlap, SpinGlass.AT.replicaOverlap,
      SpinGlass.AT.rsA, SpinGlass.AT.rsR, SpinGlass.AT.rsQ,
      SpinGlass.AT.atParameter] using h
