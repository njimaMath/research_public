import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.CharFun
import Mathlib.Analysis.Fourier.FourierTransform
import Mathlib.Analysis.Calculus.ContDiff.Comp
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Multidimensional Price theorem by Fourier analysis

This file targets the following statement.

Let `(X_t,Y_t)` be centered jointly Gaussian with fixed marginal covariance
matrices `A,B` and cross-covariance matrix `C(t)`.  For

  f : R^m -> R,  g : R^n -> R,

with `f,g in C_b^2`, one has

  d/dt E[f(X_t) g(Y_t)]
    = sum_i sum_j C'(t)_{ij} E[partial_i f(X_t) partial_j g(Y_t)].

The proof route is Fourier analysis:

* Fourier inversion for a smooth compactly supported approximation of
  `h(x,y) = f(x)g(y)`;
* `charFun_multivariateGaussian` for the Gaussian characteristic function;
* differentiate the characteristic function in the covariance parameter;
* identify multiplication by frequency coordinates with mixed derivatives;
* pass to `C_b^2` by cutoff + mollification and dominated convergence.

Current mathlib (August 2026) contains the Gaussian characteristic-function
and Fourier-inversion infrastructure, but not the exact `C_b^2` cutoff/mollifier
bridge needed by the last bullet as a single reusable theorem.  The only proof
hole below is isolated as `price_pair_fourier_core`; everything around it is the
intended reusable interface.
-/

open MeasureTheory ProbabilityTheory Matrix Set Filter Bornology
open scoped BigOperators RealInnerProductSpace MatrixOrder

noncomputable section

namespace PriceFourier

/-- A concrete predicate for `C_b^2` on a finite-dimensional real normed space.

We record boundedness of the function, its first Frechet derivative, and its
second Frechet derivative. -/
structure Cb2 {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f : E → ℝ) : Prop where
  contDiff : ContDiff ℝ 2 f
  bounded_zero : IsBounded (Set.range f)
  bounded_one : IsBounded (Set.range (fderiv ℝ f))
  bounded_two :
    IsBounded (Set.range (fun x => fderiv ℝ (fun y => fderiv ℝ f y) x))

variable {ι κ : Type*}
variable [Fintype ι] [Fintype κ]
variable [DecidableEq ι] [DecidableEq κ]

/-- The `ι`-coordinates of a vector indexed by `ι ⊕ κ`. -/
def leftPart (z : EuclideanSpace ℝ (ι ⊕ κ)) : EuclideanSpace ℝ ι :=
  WithLp.toLp 2 (fun i => z (Sum.inl i))

/-- The `κ`-coordinates of a vector indexed by `ι ⊕ κ`. -/
def rightPart (z : EuclideanSpace ℝ (ι ⊕ κ)) : EuclideanSpace ℝ κ :=
  WithLp.toLp 2 (fun j => z (Sum.inr j))

/-- Block covariance matrix

      [ A    C ]
      [ Cᵀ   B ]. -/
def blockCov
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ) (C : Matrix ι κ ℝ) :
    Matrix (ι ⊕ κ) (ι ⊕ κ) ℝ :=
  Matrix.fromBlocks A C C.transpose B

/-- Coordinate derivative in the orthonormal coordinate direction `i`. -/
def partialLeft (f : EuclideanSpace ℝ ι → ℝ)
    (x : EuclideanSpace ℝ ι) (i : ι) : ℝ :=
  fderiv ℝ f x (EuclideanSpace.basisFun ι ℝ i)

/-- Coordinate derivative in the orthonormal coordinate direction `j`. -/
def partialRight (g : EuclideanSpace ℝ κ → ℝ)
    (y : EuclideanSpace ℝ κ) (j : κ) : ℝ :=
  fderiv ℝ g y (EuclideanSpace.basisFun κ ℝ j)

/-- Expectation of the tensor-product observable `f(x) g(y)` under the centered
Gaussian with covariance block matrix `[A C; Cᵀ B]`. -/
def pairExpectation
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ) (C : Matrix ι κ ℝ)
    (f : EuclideanSpace ℝ ι → ℝ) (g : EuclideanSpace ℝ κ → ℝ) : ℝ :=
  ∫ z,
      f (leftPart z) * g (rightPart z)
    ∂multivariateGaussian (0 : EuclideanSpace ℝ (ι ⊕ κ)) (blockCov A B C)

/-- The right-hand side in multidimensional Price's theorem. -/
def priceRHS
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ)
    (C Cdot : Matrix ι κ ℝ)
    (f : EuclideanSpace ℝ ι → ℝ) (g : EuclideanSpace ℝ κ → ℝ) : ℝ :=
  ∑ i, ∑ j,
    Cdot i j *
      ∫ z,
        partialLeft f (leftPart z) i * partialRight g (rightPart z) j
      ∂multivariateGaussian (0 : EuclideanSpace ℝ (ι ⊕ κ)) (blockCov A B C)

/-- Fourier-analytic core of multidimensional Price's theorem.

Mathematical proof inside this lemma:

For `h(x,y)=f(x)g(y)` and a compactly supported smooth approximation `h_R`, Fourier
inversion gives

  E_t[h_R]
    = ∫ hhat_R(ξ) exp(-1/2 <ξ,S(t)ξ>) dξ

(up to mathlib's Fourier normalization).  Differentiating in `t` inserts
`-1/2 <ξ,S'(t)ξ>`.  Since only the off-diagonal block varies, the factor `1/2`
is canceled by the two symmetric cross blocks.  Fourier differentiation turns
`ξ_i η_j hhat_R` into the transform of `partial_i partial_j h_R`, hence into
`partial_i f * partial_j g`.  Finally let `R -> ∞`; bounded first and second
derivatives give the required Gaussian dominators.

The hypotheses use `HasDerivWithinAt` because the covariance block only needs to
be positive semidefinite on an open parameter set, e.g. `(-1,1)` for the usual
correlation path. -/
theorem price_pair_fourier_core
    {U : Set ℝ} (hU : IsOpen U) {t : ℝ} (ht : t ∈ U)
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ)
    (C : ℝ → Matrix ι κ ℝ) (Cdot : Matrix ι κ ℝ)
    (hPSD : ∀ s ∈ U, (blockCov A B (C s)).PosSemidef)
    (hC : ∀ i j, HasDerivWithinAt (fun s => C s i j) (Cdot i j) U t)
    (f : EuclideanSpace ℝ ι → ℝ) (g : EuclideanSpace ℝ κ → ℝ)
    (hf : Cb2 f) (hg : Cb2 g) :
    HasDerivWithinAt
      (fun s => pairExpectation A B (C s) f g)
      (priceRHS A B (C t) Cdot f g)
      U t := by
  /-
  Fourier proof implementation plan against current mathlib:

  1. Put `h z = f (leftPart z) * g (rightPart z)`.

  2. Choose compactly supported smooth `h_R` by multiplying by a smooth cutoff
     and convolving with a compactly supported `ContDiffBump` mollifier.
     The relevant mathlib entry points are
       `ContDiffBump.convolution_tendsto_right`
       `HasCompactSupport.contDiff_convolution_left`.

  3. Use Fourier inversion (`Continuous.fourierInv_fourier_eq`) and Fubini to
     rewrite the Gaussian integral of `h_R` as an integral of its Fourier
     transform against the Gaussian characteristic function.

  4. Rewrite that characteristic function with
       `charFun_multivariateGaussian (hPSD s hs)`.
     Differentiate under the frequency integral using
       `hasDerivWithinAt_integral_of_dominated_loc_of_deriv_le`.

  5. Expand
       <ξ, blockCov A B (C s) ξ>
     with `Fintype.sum_sum_type` and `Matrix.fromBlocks`.
     Only `C` varies, and the two cross terms are equal.  This cancels the
     factor `1/2` from the Gaussian exponent.

  6. Apply the Fourier derivative identity twice to replace the frequency
     multiplier by the mixed derivative of `h_R`.

  7. Pass `R -> ∞` by dominated convergence.  `hf.bounded_one`,
     `hg.bounded_one`, `hf.bounded_two`, `hg.bounded_two` supply uniform bounds;
     Gaussian measures are probability measures.

  The missing reusable mathlib bridge is Step 2 + Step 7 packaged at order two.
  Once that bridge is available this proof is routine bookkeeping around the
  existing Fourier and Gaussian APIs.
  -/
  sorry

/-- Main multidimensional Price theorem for a matrix-valued cross covariance. -/
theorem price_pair
    {U : Set ℝ} (hU : IsOpen U) {t : ℝ} (ht : t ∈ U)
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ)
    (C : ℝ → Matrix ι κ ℝ) (Cdot : Matrix ι κ ℝ)
    (hPSD : ∀ s ∈ U, (blockCov A B (C s)).PosSemidef)
    (hC : ∀ i j, HasDerivWithinAt (fun s => C s i j) (Cdot i j) U t)
    (f : EuclideanSpace ℝ ι → ℝ) (g : EuclideanSpace ℝ κ → ℝ)
    (hf : Cb2 f) (hg : Cb2 g) :
    HasDerivWithinAt
      (fun s => pairExpectation A B (C s) f g)
      (∑ i, ∑ j,
        Cdot i j *
          ∫ z,
            partialLeft f (leftPart z) i * partialRight g (rightPart z) j
          ∂multivariateGaussian (0 : EuclideanSpace ℝ (ι ⊕ κ))
            (blockCov A B (C t)))
      U t := by
  simpa [priceRHS] using
    price_pair_fourier_core hU ht A B C Cdot hPSD hC f g hf hg

/-- If the parameter set is open, the within-derivative is the ordinary derivative. -/
theorem price_pair_at
    {U : Set ℝ} (hU : IsOpen U) {t : ℝ} (ht : t ∈ U)
    (A : Matrix ι ι ℝ) (B : Matrix κ κ ℝ)
    (C : ℝ → Matrix ι κ ℝ) (Cdot : Matrix ι κ ℝ)
    (hPSD : ∀ s ∈ U, (blockCov A B (C s)).PosSemidef)
    (hC : ∀ i j, HasDerivAt (fun s => C s i j) (Cdot i j) t)
    (f : EuclideanSpace ℝ ι → ℝ) (g : EuclideanSpace ℝ κ → ℝ)
    (hf : Cb2 f) (hg : Cb2 g) :
    HasDerivAt
      (fun s => pairExpectation A B (C s) f g)
      (priceRHS A B (C t) Cdot f g)
      t := by
  have hwithin := price_pair_fourier_core hU ht A B C Cdot hPSD
    (fun i j => (hC i j).hasDerivWithinAt) f g hf hg
  exact hwithin.hasDerivAt (hU.mem_nhds ht)

end PriceFourier
