# Codex instructions for the last-spin cavity theorem

## Location and scope

Place this file at

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\AGENTS.md
```

These instructions apply to the cavity formalization in this directory.

The final target is fixed. Intermediate lemmas may be changed, replaced,
strengthened, split, renamed, or reorganized if that helps complete the proof.

---

## Final goal

The public predicate to preserve is

```lean
def HasCavityModeRemainderBound
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (_data : UniformATData K) (C : ℝ) : Prop :=
  0 < C ∧
    ∀ {N : ℕ}, 0 < N → ∀ {β h q s : ℝ},
      (β, h) ∈ K → q = rsQ β h → s ∈ Set.Icc (0 : ℝ) 1 →
      ∀ path : RSSmartPathDisorder Ω N β h q,
        ‖cavityChangeMatrix.mulVec (cavityRemainder path s)‖ ≤
          C * cavityErrorScale path s
```

The theorem to prove is

```lean
theorem cavityModeRemainder_bound_from_lastSpin
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {K : Set (ℝ × ℝ)}
    (data : UniformATData K) :
    ∃ C : ℝ, HasCavityModeRemainderBound (Ω := Ω) data C := by
  ...
```

This theorem is the non-negotiable endpoint.

You may change intermediate lemmas if needed.

You may in particular:

```text
split difficult lemmas into smaller ones;
replace an awkward intermediate statement by a stronger or better one;
move cavity-analytic helpers into cavity_interpolation.lean;
add replica-relabeling lemmas;
add Gaussian differentiation lemmas;
change internal interpolation helper definitions;
replace a proof route by a mathematically equivalent route.
```

You must not weaken the final predicate or theorem.

Preserve the mathematical meaning of the public objects appearing in the final
statement:

```text
UniformATData
RSSmartPathDisorder
cavityChangeMatrix
cavityRemainder
cavityErrorScale
HasCavityModeRemainderBound.
```

---

## Mandatory blueprint

Before editing Lean code, read in full

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\blueprint.tex
```

Because `AGENTS.md` and `blueprint.tex` are siblings, you may also refer to it as

```text
blueprint.tex
```

or from the package root as

```text
Lemmas/Cavity/blueprint.tex
```

The blueprint gives the intended mathematical proof:

```text
Q = Q⁻ + ε₁ε₂/N
        ↓
construct ν_{s,u}
        ↓
prove Gaussian Gibbs derivative
        ↓
prove cubic moment bound
        ↓
replace quadratic endpoints
        ↓
differentiate G_A,G_B,G_C twice
        ↓
use the three-valued edge rule at u = 0
        ↓
control E_A,E_B,E_C
        ↓
assemble the mode remainder bound
```

Use the blueprint as the mathematical guide, with the correction below.

---

## Important correction: do not add a fresh Gaussian to Ω

The informal blueprint introduces an independent Gaussian `\widehat z`.

Do not implement that literally.

The theorem is quantified over an arbitrary fixed probability space

```lean
{Ω : Type u} [MeasureSpace Ω]
[IsProbabilityMeasure (volume : Measure Ω)]
```

and there is no assumption that `Ω` supports another independent Gaussian.

Therefore do not:

```text
add a new independent Gaussian variable to Ω;
add an assumption that such a variable exists;
replace Ω by Ω × Ω';
strengthen RSSmartPathDisorder merely to obtain extra randomness.
```

Instead, use the even/odd construction in the local file

```text
C:\Users\Public\Github\Lean\research_public\
latalaStrictAlmeidaThouless\Lemmas\Cavity\cavity_interpolation.lean
```

The intended formal replacement for `\widehat z` is the odd component of the
existing simple disorder.

The odd component should have the required covariance

```text
β^2 q * ε ε'
```

with the normalization dictated by the existing model definitions.

Its cross-covariance with the even bulk component should vanish.

The route to independence is

```text
joint Gaussianity
+
zero cross covariance
⇒
independence.
```

This must be proved, not assumed.

---

## Current assessment

There is a serious formalization gap, but there is presently no clear fatal
mathematical gap.

The major missing Lean components are:

```text
normalized Gaussian Gibbs derivative;
u = 0 even/odd independence;
u = 0 Gibbs factorization;
replica relabeling;
last-site overlap decomposition;
cubic-moment Gronwall;
quadratic endpoint replacement;
second derivative bound;
finite edge count;
uniform constant bookkeeping;
final assembly.
```

Two items are especially important and should be completed first:

```text
A. u = 0 factorization;
B. exact normalized Gaussian derivative.
```

Do not invest heavily in downstream estimates until these two points compile.

---

## Local working tree is authoritative

Before editing, run

```powershell
git status --short
```

Read the current local versions of

```text
Lemmas/Cavity/blueprint.tex
Lemmas/Cavity/cavity_interpolation.lean
Lemmas/Cavity/Talagrand_Cavity.lean
Lemmas/ATDefs.lean
Lemmas/AGENTS.md
```

Do not assume GitHub `main` contains the latest cavity work.

Do not overwrite useful local uncommitted changes.

Do not use destructive git commands on user work.

---

## Primary implementation file: cavity_interpolation.lean

Read

```text
Lemmas/Cavity/cavity_interpolation.lean
```

in full before making major changes to `Talagrand_Cavity.lean`.

This file should contain, or may be reorganized to contain, the reusable
analytic last-spin interpolation infrastructure.

The intended components are:

```text
last-site / bulk decomposition;
even component of simple disorder;
odd component of simple disorder;
covariance formulas;
zero even/odd cross covariance;
joint Gaussianity;
even/odd independence;
interpolated Hamiltonian;
endpoint u = 0 splitting;
endpoint Gibbs factorization;
Gaussian Gibbs derivative;
moment estimates.
```

You may change existing intermediate lemmas in this file if necessary.

If a current helper statement is badly shaped for the final theorem, replace it
with a better one rather than preserving it artificially.

---

## Existing parent instructions

Also read

```text
Lemmas/AGENTS.md
```

and respect its general repository conventions.

If that parent file says that `Talagrand_Cavity.lean` may be treated as already
established, that instruction does not apply to this task. Here the explicit
goal is to fill its analytic last-spin theorem.

---

## Exact definitions and consistency checks

Read

```text
Lemmas/ATDefs.lean
```

Search for:

```text
Replicas
ReplicaFun
replicaGibbsAverage
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

Reuse these definitions whenever possible.

The source vector is

```lean
theta q r = ![1 - q ^ 2, q - q ^ 2, r - q ^ 2]
```

and the `(A,B,C)` coefficient matrix is already encoded in

```lean
cavityMatrix β q r.
```

These are exact consistency checks on the combinatorics.

If a derived coefficient disagrees with `cavityMatrix`, stop and debug the
edge count.

---

## Target file and existing deterministic algebra

Read all of

```text
Lemmas/Cavity/Talagrand_Cavity.lean
```

Important existing deterministic lemmas include:

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

Reuse these where helpful.

Do not prove the analytic target by invoking a downstream theorem that depends
on it.

---

## Fixed-point facts

Read

```text
Lemmas/fixed_point.lean
```

Reuse existing facts for:

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

Do not rebuild fixed-point theory in the cavity proof.

---

## Paper source

Read

```text
refs/latalaArgumentsStrictAlmeidaThoulessCondition.tex
```

Search for:

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

Use the paper and `blueprint.tex` for the mathematics and the Lean files for
the exact formal normalization.

---

## Do not force the existing Price theorem

Inspect

```text
Lemmas/Price/Deriv.lean
```

The available covariance-path derivative theorem there assumes constant
covariance trace.

The last-spin interpolation does not directly satisfy that hypothesis in the
needed form.

Therefore do not simply apply that theorem.

Do not alter the interpolation solely to make that theorem applicable unless
you separately prove the altered interpolation has the same endpoints and the
same required derivative.

---

## Preferred Gaussian integration-by-parts route

Inspect

```text
Lemmas/smart_path/IndependentGaussianAffineIBP.lean
```

and

```text
generalizedLatala/SpinGlass/Mathlib/Probability/Distributions/
Gaussian_IBP_Hilbert.lean
```

Search for:

```text
independent_gaussian_affine_ibp
HasGaussianLaw
Gaussian integration by parts
covariance
affine
fderiv
```

The preferred architecture is:

```text
existing affine Gaussian IBP
        ↓
differentiate unnormalized replicated numerator
        ↓
differentiate normalization
        ↓
rewrite normalization terms using extra replicas
        ↓
collect coefficients
        ↓
normalized Gaussian Gibbs derivative.
```

A specialized cavity derivative theorem is acceptable and likely preferable to
proving a new general Price theorem.

---

## Gate A: u = 0 factorization

Prove the actual even/odd factorization before continuing.

Required ingredients:

```text
odd component is centered Gaussian;
even and odd components are jointly Gaussian;
their cross covariance is zero;
therefore they are independent;
at u = 0 the Hamiltonian splits into bulk plus odd last-spin field.
```

Then prove the Gibbs factorization statements actually used later.

The last-spin field must have the RS scalar Gaussian law corresponding to

```text
h + β * sqrt(q) * Z.
```

Do not introduce `Z` as a new random variable on `Ω`.

Use equality of law or the Gaussian characterization already present in the
repository.

Derive:

```text
E₀[ε_a ε_b] = q
E₀[ε_a ε_b ε_c ε_d] = r
```

for distinct replica indices.

Also prove factorization when the other factor depends only on the bulk spins.

---

## Gate B: normalized Gaussian Gibbs derivative

Prove a reusable derivative identity for a bounded observable `F` of `n`
replicas.

The target is the formal analogue of

```text
d/du ν_{s,u}[F]
 =
 s β² {
   Σ_{1≤a<b≤n} ν[F ε_a ε_b Q⁻_{ab}]
   - n Σ_{a=1}^n ν[F ε_a ε_{n+1} Q⁻_{a,n+1}]
   + n(n+1)/2 ν[F ε_{n+1} ε_{n+2} Q⁻_{n+1,n+2}]
 }.
```

The coefficients must be exactly

```text
1
-n
n(n+1)/2
```

and the global factor must be exactly

```text
s * β^2.
```

Derive these coefficients from normalized Gibbs differentiation.

Do not insert them as unexplained constants.

A good proof plan is:

```text
differentiate the n-replica numerator;
differentiate the normalization Z^{-n};
rewrite normalization derivatives with replicas n+1 and n+2;
collect all pair terms.
```

Check the diagonal covariance contribution carefully.

In the mathematical proof it cancels because the diagonal covariance
derivative is configuration independent.

The Lean proof must reproduce this cancellation or invoke a theorem where that
cancellation is explicit.

Only after Gate A and Gate B compile should you proceed.

---

## Last-site overlap decomposition

Formalize stable definitions or local abbreviations for

```text
ε_a
Q⁻_{ab}
```

and prove

```text
Q_{ab} = Q⁻_{ab} + ε_a ε_b / N.
```

Be careful with `N > 0` and `Fin N`.

Search Mathlib first for:

```text
Fin.last
Fin.castSucc
Fin.init
Fin.sum_univ_succ
```

Prefer a reusable rewrite lemma.

Also prove a uniform pointwise bound such as

```text
|Q⁻_{ab}| ≤ 2.
```

No sharper bound is needed.

---

## Replica relabeling

Prove a reusable symmetry theorem for replicated Gibbs averages under finite
permutations of replica labels.

Use it to identify quantities depending only on the intersection type of two
replica edges.

The relevant classes are:

```text
equal edge;
shares one endpoint;
disjoint edges.
```

Avoid proving many relabeling equalities separately.

---

## Cubic moment estimate

Define the interpolated quantity corresponding to

```text
M3(u) = ν_{s,u}[|Q⁻_12|^3].
```

Apply the normalized derivative identity.

Use

```text
|Q⁻| ≤ 2
```

to obtain

```text
|M3'(u)| ≤ C * M3(u).
```

The constant must be uniform over all quantifiers in the final theorem.

Compare the `u = 1` endpoint with `thirdMoment path s`.

Then prove

```text
sup_{u ∈ [0,1]} M3(u)
≤ C * cavityErrorScale path s.
```

Recall

```lean
cavityErrorScale path s
=
(N : ℝ) ^ (-(3 : ℝ) / 2) + thirdMoment path s.
```

Use an existing Gronwall theorem if convenient.

Otherwise prove the elementary backward exponential estimate directly.

Do not weaken the scale from `N^{-3/2}` to `N^{-1}`.

---

## Quadratic endpoint replacement

Treat exactly:

```text
(Q⁻_12)^2
Q⁻_12 Q⁻_13
Q⁻_12 Q⁻_34.
```

Differentiate their interpolated expectations.

Every derivative term contains three cavity overlaps times a bounded spin
factor.

Use Hölder and replica relabeling to reduce all such terms to the cubic moment.

Then compare `Q⁻Q⁻` at `u = 1` with `QQ`.

Any universal Young-type inequality is acceptable if it gives the final scale

```text
N^{-3/2} + thirdMoment.
```

The precise paper constants are not important.

---

## Endpoint modes

Define or locally abbreviate:

```text
A⁻
B⁻
C⁻
U⁻
V⁻
D⁻.
```

Prove

```text
|U⁻ - U| + |V⁻ - V| + |D⁻ - D|
≤ C * cavityErrorScale path s.
```

Use `cavityChangeMatrix` when convenient.

---

## Off-diagonal observables

Formalize:

```text
G_A = Q⁻_12 (ε_1 ε_2 - q)
G_B = Q⁻_12 (ε_1 ε_3 - q)
G_C = Q⁻_12 (ε_3 ε_4 - q).
```

At `u = 0`, use endpoint factorization to prove

```text
g_A(0) = g_B(0) = g_C(0) = 0.
```

Differentiate twice.

Each second derivative term should be controlled by a product of three cavity
overlaps.

Use Hölder and the cubic estimate to prove the analogue of

```text
g_X(1) = g_X'(0) + O(cavityErrorScale).
```

A double-integral proof is acceptable if it is easier than using a general
Taylor theorem.

---

## Three-valued edge rule

Reuse:

```lean
ReplicaEdge
EdgeRelation
edgeRelation
decoupledSpinCoefficient
```

At `u = 0`, prove:

```text
equal edge:
    1 - q^2

shares one endpoint:
    q - q^2

disjoint:
    r - q^2.
```

This should be a reusable lemma driven by endpoint factorization and the
two- and four-spin moments.

---

## Finite edge count and cavityMatrix

Prefer to prove directly that the first derivative vector in `(A,B,C)`
coordinates is

```text
s • (cavityMatrix β q r).mulVec ![A⁻, B⁻, C⁻].
```

For the `n = 4` application, the normalized derivative contains:

```text
edges among {1,2,3,4} with coefficient +1;
edges {a,5} with coefficient -4;
edge {5,6} with coefficient +10.
```

Classify each derivative edge relative to:

```text
the overlap edge {1,2};
the centered spin edge appearing in G_A, G_B, or G_C.
```

Finite enumeration is encouraged after the general probabilistic edge rule has
been proved.

Useful tactics may include:

```text
fin_cases
simp
norm_num
native_decide
ring
```

The resulting matrix must agree exactly with `cavityMatrix`.

After that, use the existing theorem

```lean
cavityChangeMatrix_mul_cavityMatrix
```

instead of redoing transformed algebra.

Remember that the Lean row order is

```text
(V,U,D)
```

and the expected transformed system is

```text
V -> β²(ζ U + κ V)
U -> β² κ U
D -> β²(1 - 2q + r) D.
```

---

## Diagonal observables

Formalize:

```text
E_A = ε_1 ε_2 (ε_1 ε_2 - q)
E_B = ε_1 ε_2 (ε_1 ε_3 - q)
E_C = ε_1 ε_2 (ε_3 ε_4 - q).
```

At `u = 0`, prove that their expectation vector is

```text
theta q r.
```

Then use

```lean
cavityChangeMatrix_mulVec_theta
```

for the transformed source.

To compare `u = 1` and `u = 0`, differentiate once.

Every derivative term contains one cavity overlap.

After multiplying by `1/N`, use Young plus the cubic estimate to put the error
on `cavityErrorScale`.

---

## Final assembly

The final theorem only requires

```lean
‖cavityChangeMatrix.mulVec (cavityRemainder path s)‖ ≤
  C * cavityErrorScale path s.
```

A clean strategy is to prove coordinate bounds for

```lean
cavityChangeMatrix.mulVec (cavityRemainder path s)
```

and then bound the finite-dimensional norm.

Use

```lean
cavityChangeMatrix_mulVec_cavityRemainder
```

to identify the coordinates with the scalar mode remainders.

Reuse existing deterministic mode algebra whenever possible.

---

## Freedom to modify intermediate lemmas

You are explicitly allowed to change intermediate lemmas.

This applies especially to helper lemmas in:

```text
Lemmas/Cavity/cavity_interpolation.lean
Lemmas/Cavity/Talagrand_Cavity.lean
```

You may alter intermediate statements if:

```text
the final theorem statement remains unchanged;
the mathematical content is not weakened;
downstream public APIs remain compatible or receive equivalent replacements;
dependencies remain acyclic;
every helper used in the final proof is genuinely proved.
```

Do not preserve a bad intermediate API simply because it already exists.

---

## Uniform constants

The final constant may depend on

```text
data : UniformATData K
```

and universal numerical constants.

It must not depend on

```text
N
β
h
q
s
path.
```

Derive local bounds from `data`.

In particular derive rather than assume:

```text
0 < β
0 < h
β ≤ data.βmax
q = rsQ β h
0 < q
q < 1
0 ≤ s
s ≤ 1.
```

Nonoptimal constants are fine.

---

## Forbidden shortcuts

Do not:

```text
add a fresh Gaussian on Ω;
replace Ω by a product space;
strengthen the theorem with extra randomness assumptions;
misapply the constant-trace Price theorem;
assume the normalized Gibbs derivative;
assume u = 0 factorization;
assume replica relabeling without proof;
weaken cavityErrorScale;
weaken HasCavityModeRemainderBound;
use the final overlap concentration theorem;
use a downstream cavity theorem circularly;
add an axiom;
add a hidden assumption;
leave a new sorry;
use admit;
discard local uncommitted work.
```

Zero covariance implies independence only after joint Gaussianity has been
proved.

---

## Recommended implementation order

Use this order unless existing local work justifies a small reordering:

```text
A. inspect git status and local cavity files

B. compile current cavity_interpolation.lean

C. finish even/odd covariance and joint Gaussianity

D. prove even/odd independence

E. prove u = 0 Gibbs factorization and last-spin moments

F. prove replica relabeling

G. prove last-site overlap decomposition and |Q⁻| ≤ 2

H. prove specialized normalized Gaussian Gibbs derivative

I. verify coefficients 1, -n, n(n+1)/2 and prefactor s β²

J. prove cubic differential inequality and backward Gronwall

K. prove quadratic endpoint replacement

L. prove G_X second derivative and Taylor remainder

M. prove three-valued edge rule

N. perform finite edge count and identify cavityMatrix

O. prove diagonal source and endpoint error

P. assemble HasCavityModeRemainderBound

Q. compile Talagrand_Cavity.lean

R. build LatalaMeetsAT
```

---

## Compilation workflow

Work from

```text
C:\Users\Public\Github\Lean\research_public\latalaStrictAlmeidaThouless
```

Compile frequently:

```powershell
lake env lean Lemmas\Cavity\cavity_interpolation.lean
```

then:

```powershell
lake env lean Lemmas\Cavity\Talagrand_Cavity.lean
```

then:

```powershell
lake build LatalaMeetsAT
```

Use the exact equivalent commands supported by the local Lake setup if needed.

---

## Handling a genuine obstruction

If a proof step cannot be completed because the project lacks infrastructure,
identify the exact missing theorem.

Likely examples include:

```text
joint Gaussianity of the even/odd decomposition;
zero cross covariance implies independence for the relevant Gaussian objects;
normalized replicated Gibbs differentiation from affine Gaussian IBP;
replica relabeling for quenchedReplicaAverage;
finite-product factorization under independence.
```

Implement the smallest mathematically correct reusable theorem.

Do not weaken the final goal.

If you discover an actual theorem-level mismatch, report:

```text
the exact conflicting formulas;
the exact Lean definitions involved;
whether the issue is covariance, normalization, endpoint law, or indexing;
the smallest failed identity or counterexample available.
```

---

## Placeholder and dependency audit

After the theorem compiles, inspect every edited file for:

```text
sorry
admit
axiom
```

The target theorem must not depend on any new project-specific placeholder.

If an existing prerequisite in the actual dependency chain is still a `sorry`,
do not claim full completion.

Fill it if it is part of this cavity proof, or report it explicitly.

---

## Final mathematical audit

Before declaring success, verify:

```text
[ ] no fresh Gaussian was added to Ω

[ ] even/odd construction realizes the desired reference field

[ ] even/odd cross covariance is zero

[ ] joint Gaussianity is proved

[ ] u = 0 independence is proved

[ ] u = 0 Gibbs factorization is proved

[ ] E₀[ε_a ε_b] = q is proved

[ ] E₀[ε_a ε_b ε_c ε_d] = r is proved

[ ] replica relabeling is proved

[ ] Q = Q⁻ + ε_a ε_b/N has the exact normalization

[ ] |Q⁻| ≤ 2 or an equivalent bound is proved

[ ] normalized derivative coefficients are exactly
    1, -n, n(n+1)/2

[ ] global derivative prefactor is exactly s β²

[ ] constant-trace Price theorem was not misapplied

[ ] cubic Gronwall estimate is uniform in u

[ ] quadratic endpoint replacement has scale
    N^(-3/2) + thirdMoment

[ ] second derivative bound is reduced to cubic moments

[ ] edge rule gives exactly
    1-q², q-q², r-q²

[ ] finite edge count agrees exactly with cavityMatrix

[ ] diagonal source agrees exactly with theta

[ ] transformed coefficients agree with cavityChangeMatrix algebra

[ ] final constant is uniform in N, β, h, q, s, path

[ ] no later concentration theorem was used
```

The two most important checks remain:

```text
u = 0 factorization;
normalized Gaussian derivative.
```

---

## Definition of done

The task is complete only when:

```text
1. cavityModeRemainder_bound_from_lastSpin has a genuine Lean proof.

2. Its conclusion is exactly
   ∃ C : ℝ, HasCavityModeRemainderBound (Ω := Ω) data C.

3. The even/odd construction is used without assuming extra randomness on Ω.

4. u = 0 independence and Gibbs factorization are proved.

5. The normalized Gaussian Gibbs derivative is proved with exact coefficients
   1, -n, n(n+1)/2.

6. Cubic Gronwall, endpoint replacement, second derivative bound, edge rule,
   finite edge count, and diagonal estimates are proved.

7. Intermediate lemmas may have changed, but every helper used in the final
   theorem has a genuine proof.

8. The edited cavity files compile.

9. lake build LatalaMeetsAT succeeds, except for a precisely identified
   pre-existing unrelated failure.

10. No public final theorem was weakened.

11. No new axiom, sorry, admit, hidden assumption, or circular dependency was
    introduced.

12. The final response reports:
    files changed;
    important intermediate lemmas changed or added;
    how u = 0 factorization was proved;
    how the normalized Gaussian derivative was proved;
    compile commands run;
    any remaining unrelated build failure.
```
