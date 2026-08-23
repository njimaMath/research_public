# Codex Agent Instructions: complete the last-spin cavity proof

## Scope of this task

Your task is to complete the Lean proof of the analytic last-spin cavity theorem

```lean
theorem cavityModeRemainder_bound_from_lastSpin
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ C : ℝ, HasCavityModeRemainderBound (Ω := Ω) data C := by
  sorry
```

in

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\Talagrand_Cavity.lean
```

The goal is not to replace this theorem by a stronger assumption, an axiom, an
opaque wrapper, or another `sorry`.  The goal is to formalize the analytic
argument.

Complete every auxiliary proof that you introduce for this theorem.  When you
finish, there must be no `sorry`, `admit`, new `axiom`, or equivalent proof
placeholder in the dependency chain that you added for this theorem.

Do not change the public mathematical statement merely to make the proof easier.

---

## Mandatory first read: the mathematical blueprint

Before editing Lean code, read this file in full:

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\blueprint.tex
```

This is the primary mathematical blueprint for the proof.

If the current working directory is

```text
C:\Users\Public\Github\Lean\research_public\latalaStrictAlmeidaThouless
```

the same file is

```text
Lemmas\Cavity\blueprint.tex
```

or, using repository-style separators,

```text
Lemmas/Cavity/blueprint.tex
```

Do not skip the blueprint because GitHub code search does not find it.  It may
exist only in the local working tree.  Read the local file directly.

The blueprint determines the intended mathematical proof, in particular:

- the last-spin interpolation;
- the decomposition `Q = Q⁻ + ε₁ ε₂ / N`;
- the Gaussian derivative identity;
- the backward Gronwall estimate for the cubic cavity moment;
- endpoint replacement of quadratic overlap products;
- the Taylor expansion of the `G_A,G_B,G_C` terms;
- the three-valued replica-edge rule at `u = 0`;
- the explicit coefficient calculation;
- the diagonal `E_A,E_B,E_C` terms;
- the final uniform remainder estimate.

If the blueprint and an existing Lean definition appear inconsistent, do not
silently modify either one.  First identify whether the mismatch is notation,
replica ordering, normalization, or a genuine mathematical discrepancy.

---

## Repository files that must be inspected

Read the following files before designing the implementation.

### 1. Existing local agent instructions

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\AGENTS.md
```

Respect all general repository conventions in that file.

For this task, however, the explicit local goal in this file takes precedence
over any older statement saying that `Talagrand_Cavity.lean` may be treated as
an established result.  Here we are deliberately proving its analytic cavity
theorem.

### 2. Exact definitions

Read

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\ATDefs.lean
```

Search for at least:

```text
Replicas
ReplicaFun
quenchedReplicaAverage
replicaOverlap
centeredOverlap
UniformATData
RSSmartPathDisorder
fullPathHamiltonian
A
B
C
thirdMoment
cavityVector
theta
cavityMatrix
stabilityOperator
cavityChangeMatrix
cavityChangeMatrixInv
cavityU
cavityV
cavityD
cavityKappa
cavityZeta
ReplicaEdge
EdgeRelation
edgeRelation
decoupledSpinCoefficient
cavityRemainder
cavityErrorScale
HasCavityRemainderBound
```

These definitions are authoritative.  Reuse them whenever possible instead of
creating mathematically duplicate notions.

In particular, the source vector is

```lean
theta q r = ![1 - q ^ 2, q - q ^ 2, r - q ^ 2]
```

and the coefficient matrix in `(A,B,C)` coordinates is already defined as
`cavityMatrix`.

Use these as exact checks on the replica combinatorics.

### 3. Target file

Read the complete file

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\Talagrand_Cavity.lean
```

Do not inspect only the target theorem.

In particular understand and preserve the already proved deterministic
statements around it, including:

```text
cavityChangeMatrix_mul_cavityMatrix
cavityChangeMatrix_mulVec_theta
cavityChangeMatrix_mulVec_cavityVector
cavityChangeMatrix_mulVec_cavityRemainder
beta_sq_mul_repliconCoefficient_eq_atParameter
thirdMoment_nonneg
cavityChangeMatrixInv_mul_cavityChangeMatrix
cavityChangeMatrixInv_mulVec_norm_le
cavityRemainder_eq_inverseModeRemainder
exists_hasCavityRemainderBound
```

The target theorem is the analytic input.  Do not prove it by invoking
`exists_hasCavityRemainderBound` or another downstream theorem that already
depends on it.

### 4. RS fixed-point facts

Read

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\fixed_point.lean
```

Search for facts about

```text
rsQ
rsR
rsQ_fixedPoint
rsQ_eq_gaussian_tanh_sq
rsQ_pos
rsQ_lt_one
atParameter
atParameter_eq_beta_sq_mul_one_sub_two_q_add_r
```

Reuse existing fixed-point and Gaussian-moment lemmas.  Do not reprove them
inside the cavity file unless a tiny local bridge lemma is genuinely needed.

### 5. Mathematical paper source

Read the cavity proposition and its appendix proof in

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\refs\
latalaArgumentsStrictAlmeidaThoulessCondition.tex
```

Search for

```text
prop:cavity
eq:cavity-modes
eq:cavity-U
eq:cavity-V
eq:cavity-D
eq:cavity-remainders
eq:cavity-derivative-proof
eq:M3-gronwall-proof
eq:endpoint-replacement-proof
eq:cavity-Taylor-modes
eq:edge-rule-proof
eq:scalar-first-derivatives
eq:scalar-diagonal-terms
eq:scalar-diagonal-remainders
```

The paper and `blueprint.tex` give the mathematical organization.  Lean files
give the exact formal API.

### 6. Gaussian integration by parts infrastructure

Search the repository before proving Gaussian integration by parts from
scratch.  In particular inspect

```text
C:\Users\Public\Github\Lean\research_public\
generalizedLatala\SpinGlass\Mathlib\Probability\Distributions\
GaussianIntegrationByParts.lean
```

and related Gaussian/Hilbert-space files imported by the project.

Search for declarations containing terms such as

```text
gaussian_integration_by_parts
gaussianRV_integration_by_parts
stein
HasGaussianLaw
Gaussian
covariance
derivative
```

Prefer an existing general Gaussian differentiation theorem if it fits the
finite-dimensional disorder interpolation.

Do not introduce a mathematically weaker fake Gaussian structure merely to
avoid using the actual smart-path disorder.

### 7. Empty cavity helper file

Inspect

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\cavity_interpolation.lean
```

At the time these instructions were written, this file was empty.

It is the preferred location for a substantial reusable last-spin
interpolation API if putting all auxiliary lemmas directly into
`Talagrand_Cavity.lean` would make that file unmanageable.

If you use this file:

- add the necessary imports;
- import it from `Talagrand_Cavity.lean`;
- keep dependency direction acyclic;
- give helper lemmas descriptive mathematical names;
- compile the helper file independently before returning to the main theorem.

Do not move already stable deterministic cavity algebra out of
`Talagrand_Cavity.lean` unless there is a compelling reason.

---

## Exact target statement and scale

The target predicate is

```lean
HasCavityModeRemainderBound
```

and the desired scale is

```lean
cavityErrorScale path s
```

which is

```lean
(N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s.
```

The theorem must be uniform over:

```lean
N > 0,
(β,h) ∈ K,
q = rsQ β h,
s ∈ [0,1],
path : RSSmartPathDisorder Ω N β h q.
```

The constant may depend on `data : UniformATData K`, but must not depend on
`N`, `β`, `h`, `q`, `s`, or `path`.

Do not replace the error by a later concentration estimate.  The cavity theorem
is used before the final absorption argument.

---

## Mathematical decomposition to formalize

Organize the proof into small lemmas corresponding to the following blocks.

Do not try to prove the entire theorem in one tactic block.

### Last-site notation

Formalize the analogues of

```text
ε_a = σ_N^a
Q⁻_{ab} = (1/N) sum_{i<N} σ_i^a σ_i^b - q
Q_{ab} = Q⁻_{ab} + ε_a ε_b/N.
```

Be extremely careful with `Fin N`.

For `N > 0`, use a canonical last index such as `Fin.last (N-1)` only after
choosing a representation that makes the first `N-1` coordinates precise.

Before committing to a representation, search the repository and Mathlib for
existing APIs for:

```text
Fin.last
Fin.castSucc
Fin.init
Fin.sum_univ_succ
Finset.univ
```

A good formalization should make the decomposition of the overlap a simple
finite-sum identity, not a recurring source of index coercion problems.

If it is substantially easier, introduce a carefully proved equivalence
between `Fin N` and `Option (Fin (N-1))` or use a sum decomposition theorem
already in Mathlib.  Do not alter `SpinGlass.Config N` globally.

### Site exchangeability

Prove the formal versions of

```text
A_s = ν_s[Q_12 (ε_1 ε_2 - q)]
B_s = ν_s[Q_12 (ε_1 ε_3 - q)]
C_s = ν_s[Q_12 (ε_3 ε_4 - q)].
```

Do not assume site exchangeability as an axiom.

First search the existing SK model and disorder API for permutation invariance.
If a suitable theorem does not exist, isolate and prove the precise finite
permutation lemma needed here.

Because the disorder is averaged, the required statement is exchangeability
of the quenched law, not pointwise equality for a fixed disorder realization.

If proving general site exchangeability is disproportionately expensive, it
is acceptable to prove only the exact averaged identities needed for the
three observables, provided they follow from the actual Gaussian disorder law.

### Last-spin interpolation object

Construct a genuine interpolation `u ∈ [0,1]` between:

```text
u = 1: the original last-spin interaction,
u = 0: an independent scalar Gaussian field of variance s β² q.
```

The first `N-1` spin Hamiltonian must remain unchanged.

The mathematical covariance derivative must be the formal analogue of

```text
s * β^2 * ε * ε' * Q⁻(ρ,ρ').
```

Its diagonal value must be configuration independent so that it cancels in
the normalized Gibbs derivative.

Do not define an interpolation merely by postulating its covariance unless
the repository's Gaussian-law abstraction explicitly supports constructing
such a process from that covariance.

Prefer building it from independent Gaussian components already present in
the model or from an existing finite-dimensional Gaussian constructor.

### Decoupled endpoint

At `u = 0`, prove that the last spin is independent of the first `N-1` spins
and sees the field with distribution

```text
h + β * sqrt q * Z.
```

Derive the exact last-spin moments needed by the proof:

```text
E[ε_a ε_b] = q
E[ε_a ε_b ε_c ε_d] = r
```

for distinct replica indices.

Also prove the three-valued edge coefficient rule:

```text
equal edge       -> 1 - q^2
shares one index -> q - q^2
disjoint edge    -> r - q^2
```

Reuse the existing definitions

```lean
EdgeRelation
edgeRelation
decoupledSpinCoefficient
```

from `ATDefs.lean` when convenient.

The formal rule should be general enough to drive the coefficient calculation
without manually expanding dozens of spin products.

### Gaussian derivative identity

Prove a reusable derivative lemma for a bounded function `F` of finitely many
replicas, corresponding to

```text
d/du ν_{s,u}[F]
 =
 s β² {
   sum_{a<b≤n} ν[F ε_a ε_b Q⁻_{ab}]
   - n sum_{a≤n} ν[F ε_a ε_{n+1} Q⁻_{a,n+1}]
   + n(n+1)/2 ν[F ε_{n+1} ε_{n+2} Q⁻_{n+1,n+2}]
 }.
```

The coefficients

```text
1
-n
n(n+1)/2
```

must be derived, not hard-coded into separate special cases without proof.

It is acceptable to first prove a general normalized Gaussian Gibbs
differentiation theorem and then specialize it.

This derivative identity is the central analytic lemma.  Give it a stable,
reusable name.

### Uniform bound on cavity overlaps

Prove a deterministic pointwise bound such as

```text
|Q⁻_{ab}| ≤ 2.
```

Use existing overlap bounds if available.

This bound is what allows differentiation of the cubic moment without needing
a fourth-moment estimate.

### Cubic moment and backward Gronwall

Define the interpolated cubic quantity corresponding to

```text
M3(u) = ν_{s,u}[|Q⁻_12|^3].
```

Prove

```text
|M3'(u)| ≤ C * M3(u)
```

with a constant controlled only by `UniformATData`.

Prove the endpoint comparison at `u = 1` using

```text
Q = Q⁻ + εε/N
```

and then obtain a uniform-in-`u` bound of the form

```text
M3(u) ≤ C * ((N : ℝ)^(-3/2) + thirdMoment path s).
```

Use an existing Gronwall lemma from Mathlib if practical.  Search before
proving Gronwall from scratch.

If Mathlib's Gronwall API is awkward for this one-dimensional bounded interval,
a short direct proof using an exponential integrating factor is acceptable.

Do not accidentally introduce dependence on `N` into the constant.

### Quadratic endpoint replacement

For the three products

```text
(Q⁻_12)^2
Q⁻_12 Q⁻_13
Q⁻_12 Q⁻_34
```

differentiate along `u`.

Every derivative term is a product of three cavity overlaps times a bounded
spin factor.

Use Hölder to reduce those terms to the cubic moment.

Then compare `Q⁻Q⁻` at `u=1` with `QQ`.

Formalize the scalar inequality corresponding to

```text
|x| / N ≤ |x|^3 / 3 + 2 / (3 N^(3/2)).
```

You may use a slightly different universal numerical constant if it leads to a
much simpler Lean proof, as long as the final scale remains

```text
C * ((N : ℝ)^(-3/2) + thirdMoment path s).
```

Do not weaken the scale to `N^(-1)`.

### Cavity modes at `u = 0`

Define or locally abbreviate the cavity endpoint moments

```text
A⁻, B⁻, C⁻
```

and the corresponding modes

```text
U⁻ = A⁻ - 4 B⁻ + 3 C⁻
V⁻ = 2 B⁻ - 3 C⁻
D⁻ = A⁻ - 2 B⁻ + C⁻.
```

Prove that they differ from the original modes by at most the cavity error
scale.

Reuse the fixed change-of-basis matrix when this simplifies the proof.

### Off-diagonal terms `G_A,G_B,G_C`

Formalize the analogues of

```text
G_A = Q⁻_12 (ε_1 ε_2 - q)
G_B = Q⁻_12 (ε_1 ε_3 - q)
G_C = Q⁻_12 (ε_3 ε_4 - q).
```

At `u = 0`, prove their expectations vanish.

Apply the Gaussian derivative identity twice.

Prove a uniform second derivative bound by the cubic cavity moment.

Then prove the Taylor estimate

```text
g_X(1) = g_X'(0) + O(errorScale)
```

for the three transformed modes.

Do not leave the phrase "apply the derivative identity twice" unformalized.
The formal proof must account for the enlarged replica family and the Hölder
bound for each resulting cubic overlap product.

If a general Taylor theorem creates unnecessary differentiability overhead, it
is acceptable to prove the needed identity by integrating the derivative
twice, provided all integrability/continuity hypotheses are discharged.

### First derivative coefficient calculation

This calculation must match `cavityMatrix`.

A safe route is:

1. Prove the derivative vector in `(A,B,C)` coordinates is

```text
s • (cavityMatrix β q r).mulVec ![A⁻,B⁻,C⁻].
```

2. Then reuse the already proved matrix identity

```lean
cavityChangeMatrix_mul_cavityMatrix
```

to obtain the transformed coefficients.

This is strongly preferred over reproving the transformed coefficients by
large ring calculations.

The three rows in `(A,B,C)` coordinates must reduce to:

```text
(1-q^2, -4(q-q^2), 3(r-q^2))

(q-q^2,
 (1-q^2)-2(q-q^2)-3(r-q^2),
 6(r-q^2)-3(q-q^2))

(r-q^2,
 4(q-q^2)-8(r-q^2),
 (1-q^2)-8(q-q^2)+10(r-q^2)).
```

If your combinatorics gives anything else, stop and debug the replica-edge
counting.

After the change of basis, remember the Lean matrix row order is

```text
(V,U,D)
```

not `(U,V,D)`.

The expected transformed system is

```text
V -> β² (ζ U + κ V)
U -> β² κ U
D -> β² (1 - 2q + r) D.
```

### Diagonal terms `E_A,E_B,E_C`

Formalize

```text
E_A = ε_1 ε_2 (ε_1 ε_2 - q)
E_B = ε_1 ε_2 (ε_1 ε_3 - q)
E_C = ε_1 ε_2 (ε_3 ε_4 - q).
```

At `u=0`, prove their expectation vector is exactly

```text
theta q r
```

and therefore, after applying `cavityChangeMatrix`, the source is

```text
(V source, U source, D source)
=
(ζ, κ, 1-2q+r).
```

Reuse

```lean
cavityChangeMatrix_mulVec_theta
```

rather than duplicating the final linear algebra.

For the difference between `u=1` and `u=0`, differentiate once.  Every term
contains only one cavity overlap, so after multiplying by `1/N` use the same
Young/cubic-moment estimate to put the error on `cavityErrorScale`.

### Final assembly

The cleanest target is to prove directly that every coordinate of

```lean
cavityChangeMatrix.mulVec (cavityRemainder path s)
```

has absolute value at most

```text
C * cavityErrorScale path s.
```

Then deduce the norm bound required by `HasCavityModeRemainderBound`.

Use the explicit theorem

```lean
cavityChangeMatrix_mulVec_cavityRemainder
```

to check that these coordinates are exactly the scalar mode remainders.

At the RS fixed point, replace

```text
β^2 * (1 - 2*q + rsR β h)
```

by

```text
atParameter β h
```

only through the existing bridge theorem if needed.

---

## Uniform-constant discipline

Every analytic helper lemma must make clear which quantities control the
constant.

From `UniformATData`, derive rather than assume:

```text
0 < β
0 < h
β ≤ data.βmax
q = rsQ β h
0 < q
q < 1
s ∈ [0,1].
```

Use boundedness of spin variables and overlaps aggressively.

It is acceptable for the final constant to be very nonoptimal.

It is not acceptable for the final constant to depend on:

```text
N
β
h
q
s
path.
```

When a local estimate introduces a constant, either:

- give it an explicit formula in terms of `data.βmax` and universal numbers; or
- package the estimate in a way that makes the uniform dependence immediate.

Avoid existential constants nested inside the proof unless their dependencies
are transparent.

---

## Formalization strategy

### Prefer finite combinatorics over abstract machinery where appropriate

The replica number used in this proof is small and fixed in the final
applications.

For the coefficient computation, it is often safer to formalize edge
classification over finite types and use `fin_cases`, `native_decide`,
`norm_num`, `ring`, or finite enumeration after the general probabilistic
lemma is established.

Do not use brute-force enumeration as a substitute for the Gaussian
differentiation argument itself.

### Separate analytic and algebraic layers

A good architecture is:

```text
last-spin indexing and overlap decomposition
    ↓
interpolated disorder / Hamiltonian
    ↓
endpoint decoupling lemmas
    ↓
Gaussian differentiation
    ↓
moment estimates
    ↓
edge coefficient lemma
    ↓
(A,B,C) derivative vector
    ↓
existing matrix algebra
    ↓
final mode remainder bound
```

Keep matrix/ring algebra out of measure-theoretic proofs whenever possible.

### Reuse existing project abstractions

Before defining a new concept, search the repository for an existing one.

In particular, do not create duplicates of:

```text
quenchedReplicaAverage
centeredOverlap
thirdMoment
cavityMatrix
theta
EdgeRelation
decoupledSpinCoefficient.
```

### Small bridge lemmas are encouraged

If an existing general theorem almost fits, prove a local bridge lemma with a
clear name rather than unfolding many layers repeatedly.

Examples of useful helper themes:

```text
replica average invariant under replica relabeling
last-site overlap decomposition
bounded centered cavity overlap
decoupled last-spin two-replica moment
decoupled last-spin four-replica moment
decoupled edge coefficient
interpolated Gibbs derivative
cubic cavity moment derivative bound
cubic cavity moment uniform bound
quadratic cavity endpoint replacement
offDiagonal second derivative bound
diagonal endpoint replacement
```

Exact names may differ.

---

## Things you must not do

Do not solve the target theorem by:

- adding an `axiom`;
- leaving `sorry`;
- using `admit`;
- adding a typeclass assumption that contains the desired estimate;
- changing `HasCavityModeRemainderBound` to a weaker predicate;
- redefining `cavityErrorScale` to make the theorem trivial;
- assuming the last-spin derivative formula;
- assuming the cubic-moment estimate;
- invoking the target theorem through a circular alias;
- importing a file that already assumes this theorem;
- invoking a downstream theorem whose proof depends on this theorem;
- replacing the actual SK smart path by an unrelated toy model;
- silently assuming site exchangeability pointwise in the disorder;
- using the final overlap concentration theorem to prove the cavity theorem.

Do not delete already proved deterministic lemmas merely because a different
architecture feels easier.

Do not modify unrelated GT, concentration, or free-energy files unless a
genuinely reusable prerequisite is missing.  If such a modification is
necessary, keep it minimal and document why.

---

## Handling mathematical or API obstacles

If the exact blueprint proof cannot be formalized with the current project
abstractions, do not immediately weaken the theorem.

First determine precisely what infrastructure is missing.

Typical legitimate missing infrastructure could be:

- construction of the last-spin Gaussian interpolation;
- covariance computation for that interpolation;
- normalized Gibbs differentiation under a finite-dimensional Gaussian law;
- a site-permutation invariance theorem;
- a finite-replica relabeling theorem.

Implement the missing reusable lemma at the lowest sensible layer.

If a general theorem would take much more work than the cavity application,
prove the exact specialized statement needed here, but make it mathematically
honest and reusable enough to inspect.

---

## Compilation workflow

Work from

```text
C:\Users\Public\Github\Lean\research_public\latalaStrictAlmeidaThouless
```

The package is `LatalaMeetsAT`.

After every small group of lemmas, compile the narrowest relevant target.

Typical commands to try are of the form

```powershell
lake env lean Lemmas\Cavity\cavity_interpolation.lean
```

and

```powershell
lake env lean Lemmas\Cavity\Talagrand_Cavity.lean
```

Then build the project target:

```powershell
lake build LatalaMeetsAT
```

Use the actual commands supported by the local Lake setup if these need minor
adjustment.

Do not wait until the end to compile.

When Lean reports a large error downstream, reduce it to the smallest new
lemma and debug there.

---

## Axiom and placeholder audit

After the theorem compiles, search the files you edited for

```text
sorry
admit
axiom
```

and inspect the theorem's dependency chain.

The target theorem must not depend on a new project-specific axiom or proof
placeholder introduced during this task.

Existing unrelated `sorry`s elsewhere in the repository are not part of this
task unless your proof depends on them.

If an existing prerequisite you need is itself a `sorry`, do not silently use
it as if the analytic proof were complete.  Record the dependency and, when it
is part of the last-spin proof chain, fill it as well.

---

## Final verification against the mathematics

Before declaring success, compare the Lean result line by line with

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\blueprint.tex
```

Verify all of the following.

- The last-spin interpolation has the intended endpoints.
- Its covariance derivative has the correct normalization.
- The normalized derivative identity has coefficients
  `1`, `-n`, `n(n+1)/2`.
- The cubic estimate uses the boundedness of `Q⁻`.
- The endpoint replacement has scale `N^(-3/2) + thirdMoment`.
- The second derivative of each off-diagonal term is bounded by the cubic
  cavity moment.
- The edge rule has exactly the three coefficients
  `1-q^2`, `q-q^2`, `r-q^2`.
- The `(A,B,C)` coefficient vector matches `cavityMatrix`.
- The source vector matches `theta`.
- The transformed system agrees with the existing deterministic mode algebra.
- The `D` coefficient is `β²(1-2q+r)`.
- The final norm estimate is exactly a uniform
  `HasCavityModeRemainderBound`.
- No later concentration result was used.

Only after these checks should the target `sorry` be considered resolved.

---

## Definition of done

This task is complete only when:

1. `cavityModeRemainder_bound_from_lastSpin` has a genuine Lean proof.
2. Every helper theorem added for it also has a genuine proof.
3. The relevant Lean files compile.
4. `lake build LatalaMeetsAT` succeeds, except for a clearly pre-existing,
   unrelated repository failure that you can identify precisely.
5. No theorem statement was weakened.
6. No new axiom, `sorry`, `admit`, or circular assumption was introduced.
7. The proof follows the mathematics in `Lemmas/Cavity/blueprint.tex`.
8. The coefficient calculation agrees exactly with `ATDefs.cavityMatrix`,
   `theta`, and the existing cavity change-of-basis lemmas.
9. The final response reports which files were changed, what analytic helper
   lemmas were added, what commands were run, and whether any pre-existing
   unrelated build failures remain.
