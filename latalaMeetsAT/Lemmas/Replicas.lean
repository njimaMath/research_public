import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Fintype.Pi

open MeasureTheory ProbabilityTheory Real BigOperators
open scoped ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

/-- Spin configurations used by the strict-AT development. -/
abbrev Config (N : ℕ) := Fin N → Bool

/-- A finite-volume Hamiltonian. -/
abbrev EnergySpace (N : ℕ) := Config N → ℝ

def spin {N : ℕ} (σ : Config N) (i : Fin N) : ℝ := if σ i then 1 else -1

/-- The normalized overlap of two configurations. -/
noncomputable def configOverlap (N : ℕ) (σ τ : Config N) : ℝ :=
  (1 / (N : ℝ)) * ∑ i, spin σ i * spin τ i

noncomputable def partitionFunction {N : ℕ} (H : EnergySpace N) : ℝ :=
  ∑ σ, Real.exp (H σ)

noncomputable def gibbsWeight {N : ℕ} (H : EnergySpace N) (σ : Config N) : ℝ :=
  Real.exp (H σ) / partitionFunction H

/-- An explicitly indexed family of replicas. -/
abbrev Replicas (N n : ℕ) := Fin n → Config N

/-- Finite-product Gibbs expectation.  Keeping the product as a finite sum
makes later differentiation a finite-dimensional calculation. -/
noncomputable def replicaGibbsAverage {N n : ℕ}
    (H : EnergySpace N) (F : Replicas N n → ℝ) : ℝ :=
  ∑ σs, (∏ a, gibbsWeight H (σs a)) * F σs

/-- Disorder-averaged replica expectation. -/
noncomputable def quenchedReplicaAverage {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N n : ℕ} (H : Ω → EnergySpace N)
    (F : Replicas N n → ℝ) : ℝ :=
  ∫ ω, replicaGibbsAverage (H ω) F ∂(volume : Measure Ω)

/-- The overlap of two selected replicas. -/
noncomputable def overlap {N n : ℕ} (σs : Replicas N n) (a b : Fin n) : ℝ :=
  configOverlap N (σs a) (σs b)

/-- Overlap centered at the replica-symmetric parameter `q`. -/
noncomputable def centeredOverlap {N n : ℕ} (q : ℝ) (σs : Replicas N n)
    (a b : Fin n) : ℝ :=
  overlap σs a b - q

theorem overlap_mem_Icc {N n : ℕ} (hN : 0 < N) (σs : Replicas N n)
    (a b : Fin n) : overlap σs a b ∈ Set.Icc (-1 : ℝ) 1 := by
  -- Proof route: unfold `overlap`, `configOverlap`, and `spin`.  Every summand
  -- `spin (σs a) i * spin (σs b) i` is either `1` or `-1`, so bound the finite
  -- sum between `-N` and `N` with `Finset.sum_le_sum`.  Use `hN` to rewrite
  -- `1 / (N : ℝ)` as a nonnegative scalar and finish both bounds with `linarith`
  -- or `nlinarith`.  This formalizes the elementary fact recorded before the
  -- Gaussian-calculus subsection of the paper.
  let x : Fin N → ℝ := fun i => spin (σs a) i * spin (σs b) i
  have hx : ∀ i, x i = 1 ∨ x i = -1 := by
    intro i
    simp only [x, spin]
    split <;> split <;> simp_all
  have hxlower : ∀ i, (-1 : ℝ) ≤ x i := fun i => by
    rcases hx i with hi | hi <;> rw [hi] <;> norm_num
  have hxupper : ∀ i, x i ≤ (1 : ℝ) := fun i => by
    rcases hx i with hi | hi <;> rw [hi] <;> norm_num
  have hsumlower : -(N : ℝ) ≤ ∑ i, x i := by
    calc
      -(N : ℝ) = ∑ _i : Fin N, (-1 : ℝ) := by simp
      _ ≤ ∑ i, x i := Finset.sum_le_sum fun i _ => hxlower i
  have hsumupper : ∑ i, x i ≤ (N : ℝ) := by
    calc
      ∑ i, x i ≤ ∑ _i : Fin N, (1 : ℝ) :=
        Finset.sum_le_sum fun i _ => hxupper i
      _ = (N : ℝ) := by simp
  have hNreal : (0 : ℝ) < N := by exact_mod_cast hN
  change (1 / (N : ℝ)) * ∑ i, x i ∈ Set.Icc (-1 : ℝ) 1
  constructor
  · calc
      (-1 : ℝ) = (1 / (N : ℝ)) * (-(N : ℝ)) := by field_simp
      _ ≤ (1 / (N : ℝ)) * ∑ i, x i :=
        mul_le_mul_of_nonneg_left hsumlower (by positivity)
  · calc
      (1 / (N : ℝ)) * ∑ i, x i ≤ (1 / (N : ℝ)) * (N : ℝ) :=
        mul_le_mul_of_nonneg_left hsumupper (by positivity)
      _ = 1 := by field_simp

theorem abs_centeredOverlap_le_two {N n : ℕ} (hN : 0 < N)
    {q : ℝ} (hq : q ∈ Set.Icc (0 : ℝ) 1) (σs : Replicas N n)
    (a b : Fin n) : |centeredOverlap q σs a b| ≤ 2 := by
  -- Proof route: obtain `-1 ≤ overlap σs a b` and `overlap σs a b ≤ 1` from
  -- `overlap_mem_Icc`.  Combine these with `0 ≤ q` and `q ≤ 1`, unfold
  -- `centeredOverlap`, prove `-2 ≤ overlap ... - q ≤ 2`, and use
  -- `abs_le.mpr`.  No probability theory is needed here.
  have hover := overlap_mem_Icc hN σs a b
  rw [abs_le]
  constructor <;> unfold centeredOverlap <;> linarith [hq.1, hq.2, hover.1, hover.2]

private theorem partitionFunction_pos {N : ℕ} (H : EnergySpace N) :
    0 < partitionFunction H := by
  unfold partitionFunction
  exact Finset.sum_pos (fun σ _ => Real.exp_pos (H σ)) Finset.univ_nonempty

private theorem gibbsWeight_nonneg {N : ℕ} (H : EnergySpace N) (σ : Config N) :
    0 ≤ gibbsWeight H σ := by
  exact div_nonneg (Real.exp_pos _).le (partitionFunction_pos H).le

private theorem gibbsWeight_le_one {N : ℕ} (H : EnergySpace N) (σ : Config N) :
    gibbsWeight H σ ≤ 1 := by
  apply (div_le_one (partitionFunction_pos H)).2
  unfold partitionFunction
  exact Finset.single_le_sum (fun τ _ => (Real.exp_pos (H τ)).le) (Finset.mem_univ σ)

private theorem measurable_replicaGibbsAverage {Ω : Type u} [MeasurableSpace Ω]
    {N n : ℕ} {H : Ω → EnergySpace N} (hH : Measurable H)
    (F : Replicas N n → ℝ) :
    Measurable (fun ω => replicaGibbsAverage (H ω) F) := by
  unfold replicaGibbsAverage
  apply Finset.measurable_sum
  intro σs _
  apply Measurable.mul
  · apply Finset.measurable_prod
    intro a _
    apply Measurable.div
    · exact Real.measurable_exp.comp ((measurable_pi_iff.mp hH) (σs a))
    · unfold partitionFunction
      apply Finset.measurable_sum
      intro σ _
      exact Real.measurable_exp.comp ((measurable_pi_iff.mp hH) σ)
  · exact measurable_const

private theorem integrable_replicaGibbsAverage {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ}
    {H : Ω → EnergySpace N} (hH : Measurable H)
    (F : Replicas N n → ℝ) :
    Integrable (fun ω => replicaGibbsAverage (H ω) F) := by
  have hmeas : AEStronglyMeasurable
      (fun ω => replicaGibbsAverage (H ω) F) (volume : Measure Ω) :=
    (measurable_replicaGibbsAverage hH F).aestronglyMeasurable
  have hbound : ∀ᵐ ω ∂(volume : Measure Ω),
      ‖replicaGibbsAverage (H ω) F‖ ≤ ∑ σs : Replicas N n, |F σs| := by
    filter_upwards [] with ω
    rw [Real.norm_eq_abs]
    unfold replicaGibbsAverage
    calc
      |∑ σs, (∏ a, gibbsWeight (H ω) (σs a)) * F σs| ≤
          ∑ σs, |(∏ a, gibbsWeight (H ω) (σs a)) * F σs| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ σs, |F σs| := by
        apply Finset.sum_le_sum
        intro σs _
        rw [abs_mul, abs_of_nonneg (Finset.prod_nonneg fun a _ =>
          gibbsWeight_nonneg (H ω) (σs a))]
        have hw : ∏ a, gibbsWeight (H ω) (σs a) ≤ 1 := by
          exact Finset.prod_le_one (fun a _ => gibbsWeight_nonneg (H ω) (σs a))
            (fun a _ => gibbsWeight_le_one (H ω) (σs a))
        exact mul_le_of_le_one_left (abs_nonneg _) hw
  have hone : Integrable (fun _ : Ω => (1 : ℝ)) (volume : Measure Ω) :=
    integrable_const 1
  simpa only [mul_one] using hone.bdd_mul hmeas hbound

/-- Monotonicity of a quenched finite-replica expectation. -/
theorem quenchedReplicaAverage_mono {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ}
    {H : Ω → EnergySpace N} (hH : Measurable H)
    {F G : Replicas N n → ℝ} (hFG : ∀ σs, F σs ≤ G σs) :
    quenchedReplicaAverage H F ≤ quenchedReplicaAverage H G := by
  unfold quenchedReplicaAverage
  apply integral_mono
  · exact integrable_replicaGibbsAverage hH F
  · exact integrable_replicaGibbsAverage hH G
  · intro ω
    unfold replicaGibbsAverage
    apply Finset.sum_le_sum
    intro σs _
    exact mul_le_mul_of_nonneg_left (hFG σs)
      (Finset.prod_nonneg fun a _ => gibbsWeight_nonneg (H ω) (σs a))

theorem quenchedReplicaAverage_const_mul {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ}
    (H : Ω → EnergySpace N) (c : ℝ) (F : Replicas N n → ℝ) :
    quenchedReplicaAverage H (fun σs => c * F σs) =
      c * quenchedReplicaAverage H F := by
  unfold quenchedReplicaAverage
  rw [← integral_const_mul]
  congr 1
  funext ω
  unfold replicaGibbsAverage
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro σs _
  ring

theorem quenchedReplicaAverage_add {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {N n : ℕ}
    {H : Ω → EnergySpace N} (hH : Measurable H)
    (F G : Replicas N n → ℝ) :
    quenchedReplicaAverage H (fun σs => F σs + G σs) =
      quenchedReplicaAverage H F + quenchedReplicaAverage H G := by
  unfold quenchedReplicaAverage
  have hpoint : (fun ω => replicaGibbsAverage (H ω) (fun σs => F σs + G σs)) =
      fun ω => replicaGibbsAverage (H ω) F + replicaGibbsAverage (H ω) G := by
    funext ω
    unfold replicaGibbsAverage
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro σs _
    ring
  rw [hpoint]
  rw [integral_add (integrable_replicaGibbsAverage hH F)
    (integrable_replicaGibbsAverage hH G)]

/-- The Cauchy--Schwarz estimate used repeatedly in the absorption argument. -/
theorem mixed_overlap_abs_le_secondMoment {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} (H : Ω → EnergySpace N) (hH : Measurable H) (q : ℝ)
    (a b c d : Fin 4)
    (hsame : quenchedReplicaAverage H
        (fun σs => centeredOverlap q σs a b ^ 2) =
      quenchedReplicaAverage H
        (fun σs => centeredOverlap q σs c d ^ 2)) :
    |quenchedReplicaAverage H (fun σs =>
      centeredOverlap q σs a b * centeredOverlap q σs c d)| ≤
      quenchedReplicaAverage H (fun σs => centeredOverlap q σs a b ^ 2) := by
  let X : Replicas N 4 → ℝ := fun σs => centeredOverlap q σs a b
  let Y : Replicas N 4 → ℝ := fun σs => centeredOverlap q σs c d
  let EXY : Ω → ℝ := fun ω => replicaGibbsAverage (H ω) (fun σs => X σs * Y σs)
  let EX2 : Ω → ℝ := fun ω => replicaGibbsAverage (H ω) (fun σs => X σs ^ 2)
  let EY2 : Ω → ℝ := fun ω => replicaGibbsAverage (H ω) (fun σs => Y σs ^ 2)
  have hpoint : ∀ ω, |EXY ω| ≤ (EX2 ω + EY2 ω) / 2 := by
    intro ω
    dsimp [EXY, EX2, EY2]
    unfold replicaGibbsAverage
    calc
      |∑ σs, (∏ a, gibbsWeight (H ω) (σs a)) * (X σs * Y σs)| ≤
          ∑ σs, |(∏ a, gibbsWeight (H ω) (σs a)) * (X σs * Y σs)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ σs, (∏ a, gibbsWeight (H ω) (σs a)) *
          ((X σs ^ 2 + Y σs ^ 2) / 2) := by
        apply Finset.sum_le_sum
        intro σs _
        rw [abs_mul, abs_of_nonneg (Finset.prod_nonneg fun i _ =>
          gibbsWeight_nonneg (H ω) (σs i))]
        apply mul_le_mul_of_nonneg_left _
          (Finset.prod_nonneg fun i _ => gibbsWeight_nonneg (H ω) (σs i))
        rw [abs_mul]
        nlinarith [sq_nonneg (|X σs| - |Y σs|), sq_abs (X σs), sq_abs (Y σs)]
      _ = (∑ σs, ((∏ a, gibbsWeight (H ω) (σs a)) * X σs ^ 2 +
          (∏ a, gibbsWeight (H ω) (σs a)) * Y σs ^ 2)) / 2 := by
        rw [Finset.sum_div]
        apply Finset.sum_congr rfl
        intro σs _
        ring
      _ = ((∑ σs, (∏ a, gibbsWeight (H ω) (σs a)) * X σs ^ 2) +
          (∑ σs, (∏ a, gibbsWeight (H ω) (σs a)) * Y σs ^ 2)) / 2 := by
        rw [Finset.sum_add_distrib]
  have hEXY : Integrable EXY :=
    integrable_replicaGibbsAverage hH (fun σs => X σs * Y σs)
  have hEX2 : Integrable EX2 :=
    integrable_replicaGibbsAverage hH (fun σs => X σs ^ 2)
  have hEY2 : Integrable EY2 :=
    integrable_replicaGibbsAverage hH (fun σs => Y σs ^ 2)
  have habs : Integrable (fun ω => |EXY ω|) := hEXY.abs
  have havg : Integrable (fun ω => (EX2 ω + EY2 ω) / 2) :=
    (hEX2.add hEY2).div_const 2
  change |∫ ω, EXY ω ∂(volume : Measure Ω)| ≤ ∫ ω, EX2 ω ∂(volume : Measure Ω)
  calc
    |∫ ω, EXY ω ∂(volume : Measure Ω)| ≤
        ∫ ω, |EXY ω| ∂(volume : Measure Ω) := by
      simpa only [Real.norm_eq_abs] using norm_integral_le_integral_norm EXY
    _ ≤ ∫ ω, (EX2 ω + EY2 ω) / 2 ∂(volume : Measure Ω) :=
      integral_mono habs havg hpoint
    _ = ((∫ ω, EX2 ω ∂(volume : Measure Ω)) +
        (∫ ω, EY2 ω ∂(volume : Measure Ω))) / 2 := by
      rw [integral_div, integral_add hEX2 hEY2]
    _ = ∫ ω, EX2 ω ∂(volume : Measure Ω) := by
      dsimp [EX2, EY2]
      unfold quenchedReplicaAverage at hsame
      rw [← hsame]
      ring

end SpinGlass.AT
