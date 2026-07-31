import Lemmas.FreeEnergy

open MeasureTheory ProbabilityTheory

set_option autoImplicit false

namespace SpinGlass.AT

universe u

theorem thirdMoment_littleO {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      N * thirdMoment path s < eps := by
  -- Paper route after equation (repliconrho): combine the optimal bound
  -- `A_s ≤ M/N` with the same fixed-deviation split used in absorption.  For a
  -- fixed cutoff `eta > 0`,
  -- `N*E|Q|^3 ≤ eta*M + 8*N*C_eta*exp(-c_eta*N)`.
  -- Given `eps`, choose `eta` so the first term is below `eps/2`, then choose
  -- `N0` so the exponentially decaying term is below `eps/2` for `N ≥ N0`.
  -- Formalize the latter with `tendsto_nat_mul_exp_neg_atTop_nhds_zero` or a
  -- small comparison lemma.  This is the quantifier form of
  -- `thirdMoment = o_K(N⁻¹)` written at the end of the paper.
  sorry

theorem replicon_susceptibility {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)] {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∀ eps > 0, ∃ N0, ∀ {N : ℕ}, N0 ≤ N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
      |N * (A path s - 2 * B path s + C path s) -
        rsA β h / (1 - s * atParameter β h)| < eps := by
  -- Paper route, equations (repliconeq)--(repliconrho): take `vecMul` of the
  -- cavity system with `ell = ![1,-2,1]`.  Use
  -- `replicon_leftEigenvector` and `ell dot theta = rsA` to obtain
  -- `(1-s*atParameter)*(A-2*B+C) = rsA/N + ell dot remainder`.
  -- The remainder bound and `thirdMoment_littleO` imply that `N` times the
  -- final error tends uniformly to zero.  Divide by
  -- `1-s*atParameter`, whose lower bound is `data.gap` by `path_gap`, and
  -- rearrange.  Translate the uniform little-o estimate into the requested
  -- `eps,N0` quantifiers exactly as in the final paragraph of the paper.
  sorry

end SpinGlass.AT
