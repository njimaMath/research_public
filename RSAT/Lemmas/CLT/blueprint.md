# Blueprint for the overlap CLT in the strict AT region

## Purpose

Create a kernel-checked Lean proof of the central limit theorem for the centered
overlap in the SK model throughout the positive-field strict
Almeida-Thouless region.

The target folder is

```text
RSAT/Lemmas/CLT/
```

The fixed endpoint claim is in `Claim.lean`.

The mathematical random variable is

\[
X_N=\sqrt N\,(R_{12}-q),
\qquad
q=\operatorname{rsQ}(\beta,h).
\]

For fixed \((\beta,h)\) satisfying

\[
\beta>0,\qquad h>0,\qquad
\alpha:=\operatorname{atParameter}(\beta,h)<1,
\]

the target limit is

\[
X_N\Longrightarrow \mathcal N(0,\sigma^2).
\]

The variance is

\[
\sigma^2
=
\frac{3a}{1-\alpha}
-
\frac{2\kappa}{1-\beta^2\kappa}
-
\frac{\zeta}{(1-\beta^2\kappa)^2},
\]

where

\[
a=1-2q+r=\operatorname{rsA}(\beta,h),
\]

\[
r=\operatorname{rsR}(\beta,h),
\]

\[
\kappa=1-4q+3r
=\operatorname{cavityKappa}(q,r),
\]

and

\[
\zeta=2q+q^2-3r
=\operatorname{cavityZeta}(q,r).
\]

Since

\[
\alpha=\beta^2a,
\]

this is the same variance as

\[
\frac{3\alpha}{\beta^2(1-\alpha)}
-
\frac{2\kappa}{1-\beta^2\kappa}
-
\frac{\zeta}{(1-\beta^2\kappa)^2}.
\]

The current Lean claim uses the equivalent characteristic-function
formulation, split into cosine and sine because the existing
`quenchedReplicaAverage` interface is real-valued:

\[
\nu_N[\cos(tX_N)]
\to
\exp\!\left(-\frac{\sigma^2t^2}{2}\right),
\]

\[
\nu_N[\sin(tX_N)]
\to 0.
\]

This formulation should remain the formal endpoint unless there is a concrete
reason to introduce a separate annealed probability-measure API.

## Repository facts that should be preserved

The present RSAT project already supplies the objects needed for the CLT.

Important files:

```text
RSAT/Lemmas/AT/FixedPoint.lean
RSAT/Lemmas/AT/Definitions.lean
RSAT/Lemmas/Cavity/Interpolation.lean
RSAT/Lemmas/Cavity/Estimates.lean
RSAT/Lemmas/Cavity/TalagrandCavity.lean
RSAT/Lemmas/Concentration/Weak.lean
RSAT/Lemmas/MainResult.lean
RSAT/SpinGlass/Replicas.lean
RSAT/refs/latalaArgumentsStrictAlmeidaThoulessCondition.tex
```

Use the existing definitions:

```text
rsQ
rsR
rsA
atParameter
Replicas
replicaOverlap
centeredOverlap
replicaGibbsAverage
quenchedReplicaAverage
RSSmartPathDisorder
fullPathHamiltonian
A
B
C
thirdMoment
cavityKappa
cavityZeta
cavityMatrix
cavityChangeMatrix
cavityChangeMatrixInv
cavityU
cavityV
cavityD
```

Do not introduce parallel versions of these definitions.

The Lake library already contains `.submodules Lemmas`, so adding
`Lemmas/CLT/*.lean` does not require a `lakefile.lean` modification.

The project toolchain is Lean 4.32.1.

## Mathematical inputs already proved in RSAT

The CLT proof should be downstream of the quantitative strict-AT theorem.

The existing development proves, uniformly on compact subsets of the strict
AT region,

\[
\nu_s[(R_{12}-q)^2]=O(N^{-1}).
\]

It also proves the fixed-deviation exponential tail estimate and uses it to
obtain

\[
\nu_s[|R_{12}-q|^3]=o(N^{-1}).
\]

For the cavity modes

\[
U=A-4B+3C,
\]

\[
V=2B-3C,
\]

\[
D=A-2B+C,
\]

the existing cavity closure gives, at smart-path time \(s\),

\[
(1-s\beta^2\kappa)U_s
=
\frac{\kappa}{N}
+\mathcal R^U_{N,s},
\]

\[
(1-s\beta^2\kappa)V_s
-s\beta^2\zeta U_s
=
\frac{\zeta}{N}
+\mathcal R^V_{N,s},
\]

\[
(1-s\alpha)D_s
=
\frac{a}{N}
+\mathcal R^D_{N,s},
\]

with remainders negligible after multiplication by \(N\).

At \(s=1\),

\[
NU\to
u:=
\frac{\kappa}{1-\beta^2\kappa},
\]

\[
NV\to
v:=
\frac{\zeta}{(1-\beta^2\kappa)^2},
\]

\[
ND\to
d:=
\frac{a}{1-\alpha}.
\]

The inverse mode transform in the existing Lean code is

\[
A=-V-2U+3D,
\]

\[
B=-V-\frac32U+\frac32D,
\]

\[
C=-V-U+D.
\]

Therefore

\[
NA\to a_*:=-v-2u+3d,
\]

\[
NB\to b_*:=-v-\frac32u+\frac32d,
\]

\[
NC\to c_*:=-v-u+d.
\]

The scalar CLT variance is

\[
\sigma^2=a_*.
\]

This is an essential consistency check. Any characteristic-function or Stein
calculation that produces another variance has a replica-counting error.

## Central proof idea

The recommended proof is a cavity-Stein argument.

Do not try to infer Gaussianity from convergence of the second moment. The
second moment only identifies the candidate variance.

The new analytic statement to prove is an approximate Gaussian integration by
parts identity for the scaled overlap.

For a sufficiently smooth bounded \(f:\mathbb R\to\mathbb R\), prove

\[
\nu_N[X_{12}f(X_{12})]
=
\sigma^2\,\nu_N[f'(X_{12})]
+o(1),
\]

where

\[
X_{ab}:=\sqrt N\,(R_{ab}-q).
\]

A more robust route is to prove a three-component identity first.

Define

\[
T_A(f)
:=
\nu_N[X_{12}f(X_{12})],
\]

\[
T_B(f)
:=
\nu_N[X_{13}f(X_{12})],
\]

\[
T_C(f)
:=
\nu_N[X_{34}f(X_{12})].
\]

Set

\[
T(f)
=
\begin{pmatrix}
T_A(f)\\
T_B(f)\\
T_C(f)
\end{pmatrix}.
\]

The cavity calculation should give

\[
T(f)
=
M\,T(f)
+
\theta\,\nu_N[f'(X_{12})]
+
\operatorname{Err}_N(f),
\]

where \(M\) is exactly the same cavity matrix already used for the second
moment, and

\[
\theta
=
\begin{pmatrix}
1-q^2\\
q-q^2\\
r-q^2
\end{pmatrix}.
\]

The target error is

\[
\|\operatorname{Err}_N(f)\|
\le
C_f
\left(
N^{-1/2}
+
N\,\nu_N[|R_{12}-q|^3]
\right),
\]

or any stronger bound that tends to zero using the existing results.

Once this vector identity is proved,

\[
(I-M)T(f)
=
\theta\,\nu_N[f'(X_{12})]
+o(1).
\]

The strict AT condition gives the invertibility needed to solve this system.

Then

\[
T_A(f)
=
\left[(I-M)^{-1}\theta\right]_A
\nu_N[f'(X_{12})]
+o(1).
\]

The first coordinate must simplify to \(\sigma^2\).

## The same cavity matrix must reappear

For checking the replica combinatorics, write

\[
b_2=\beta^2(1-q^2),
\qquad
b_1=\beta^2(q-q^2),
\qquad
b_0=\beta^2(r-q^2).
\]

In the \((A,B,C)\) coordinates the existing project uses

\[
M=
\begin{pmatrix}
b_2 & -4b_1 & 3b_0\\
b_1 & b_2-2b_1-3b_0 & 6b_0-3b_1\\
b_0 & 4b_1-8b_0 & b_2-8b_1+10b_0
\end{pmatrix}.
\]

Do not create a new matrix by hand unless it is then proved equal to
`cavityMatrix`.

The existing change of basis has row order

\[
(V,U,D),
\]

not \((U,V,D)\).

Under this change of basis,

\[
M
\quad\leadsto\quad
\begin{pmatrix}
\beta^2\kappa & \beta^2\zeta & 0\\
0 & \beta^2\kappa & 0\\
0 & 0 & \alpha
\end{pmatrix}.
\]

The source transforms as

\[
\theta
\quad\leadsto\quad
\begin{pmatrix}
\zeta\\
\kappa\\
a
\end{pmatrix}.
\]

At \(s=1\), solving the triangular system gives

\[
U_f
=
\frac{\kappa}{1-\beta^2\kappa}\,
\nu[f'],
\]

\[
V_f
=
\frac{\zeta}{(1-\beta^2\kappa)^2}\,
\nu[f'],
\]

\[
D_f
=
\frac{a}{1-\alpha}\,
\nu[f'].
\]

Applying the inverse mode transform gives

\[
T_A(f)
=
\left(
-\frac{\zeta}{(1-\beta^2\kappa)^2}
-\frac{2\kappa}{1-\beta^2\kappa}
+\frac{3a}{1-\alpha}
\right)
\nu[f']
+o(1).
\]

This coefficient is the target \(\sigma^2\).

## Detailed cavity calculation

### Site exchangeability

Let \(e=(a,b)\) be one of

\[
(1,2),\qquad(1,3),\qquad(3,4).
\]

For every site-symmetric function of the overlaps,

\[
\nu_N[Q_e F]
=
\nu_N[(\varepsilon_a\varepsilon_b-q)F],
\]

where \(\varepsilon_a\) is the last spin of replica \(a\).

Thus

\[
T_e(f)
=
\sqrt N\,
\nu_N[
(\varepsilon_a\varepsilon_b-q)
f(X_{12})
].
\]

Formalize this once as a reusable exchangeability lemma. Do not reprove it
three times.

### Remove the last site

Use the current convention

\[
Q^-_{ab}
=
\frac1N\sum_{i<N}\sigma_i^a\sigma_i^b-q.
\]

Then

\[
Q_{ab}
=
Q^-_{ab}
+
\frac1N\varepsilon_a\varepsilon_b.
\]

For the distinguished edge \(12\), define mathematically

\[
Y^-_{12}
=
\sqrt N\,Q^-_{12}
+
\frac q{\sqrt N}.
\]

Then

\[
X_{12}
=
Y^-_{12}
+
\frac{\varepsilon_1\varepsilon_2-q}{\sqrt N}.
\]

This centering is very useful. It makes the last-spin perturbation mean zero
at cavity time \(u=0\).

In Lean, it may be more convenient not to define `YMinus` globally. A local
`let` is enough.

### Last-spin law at cavity time zero

At \(u=0\), the last spin is independent of the first \(N-1\) sites and its
effective field is distributed as

\[
h+\beta\sqrt q\,Z.
\]

For distinct replica indices,

\[
\mathbb E_0[\varepsilon_a\varepsilon_b]=q,
\]

\[
\mathbb E_0[
\varepsilon_a\varepsilon_b
\varepsilon_c\varepsilon_d
]=r.
\]

Consequently,

\[
\mathbb E_0[
(\varepsilon_1\varepsilon_2-q)^2
]
=
1-q^2,
\]

\[
\mathbb E_0[
(\varepsilon_1\varepsilon_3-q)
(\varepsilon_1\varepsilon_2-q)
]
=
q-q^2,
\]

\[
\mathbb E_0[
(\varepsilon_3\varepsilon_4-q)
(\varepsilon_1\varepsilon_2-q)
]
=
r-q^2.
\]

These are exactly the three entries of \(\theta\).

Reuse the already formalized last-spin moment lemmas from the cavity
development whenever possible.

### Taylor expansion at cavity time zero

For a function \(f\) with bounded second derivative,

\[
f\left(
Y^-_{12}
+
\frac{\varepsilon_1\varepsilon_2-q}{\sqrt N}
\right)
=
f(Y^-_{12})
+
\frac{\varepsilon_1\varepsilon_2-q}{\sqrt N}
f'(Y^-_{12})
+
\operatorname{Rem}.
\]

Because

\[
|\varepsilon_1\varepsilon_2-q|\le2,
\]

the remainder obeys

\[
|\operatorname{Rem}|
\le
\frac{2\|f''\|_\infty}{N}
\]

up to an inessential numerical constant.

Multiplying by

\[
\sqrt N\,(\varepsilon_a\varepsilon_b-q)
\]

gives an \(O(N^{-1/2})\) contribution.

The zero-order term vanishes because

\[
\mathbb E_0[
\varepsilon_a\varepsilon_b-q
]=0.
\]

The first-order term gives

\[
\theta_e\,
\nu_{0}[f'(Y^-_{12})].
\]

Next replace

\[
\nu_0[f'(Y^-_{12})]
\]

by

\[
\nu_1[f'(X_{12})]
\]

with an error tending to zero.

A convenient sufficient bound is \(O(N^{-1/2})\). Obtain it by applying the
last-spin interpolation derivative formula to the bounded observable
\(f'(Y^-_{12})\), using the \(O(N^{-1})\) overlap second moment and
Cauchy-Schwarz.

### First cavity derivative

For

\[
G_e
=
\sqrt N\,
(\varepsilon_a\varepsilon_b-q)
f(X_{12}),
\]

use the existing last-spin interpolation

\[
g_e(u)=\nu_u[G_e].
\]

The interpolation derivative formula is already available in the cavity
development. It has the standard form

\[
\frac{d}{du}\nu_u[F]
=
\beta^2
\left[
\sum_{a<b}\nu_u[
F\varepsilon_a\varepsilon_bQ^-_{ab}]
-
n\sum_a\nu_u[
F\varepsilon_a\varepsilon_{n+1}Q^-_{a,n+1}]
+
\frac{n(n+1)}2\nu_u[
F\varepsilon_{n+1}\varepsilon_{n+2}
Q^-_{n+1,n+2}]
\right]
\]

at smart-path endpoint \(s=1\).

Evaluate \(g'_e(0)\) using independence of the last spin from the cavity
system.

The leading terms must be exactly the rows of `cavityMatrix` acting on

\[
T_A(f),\quad T_B(f),\quad T_C(f),
\]

up to endpoint-replacement errors.

Do not perform replica counting informally. For each row, classify every
replica edge into:

```text
same edge,
shares exactly one endpoint,
disjoint edge.
```

Use the existing edge-rule calculation as the model.

### Second cavity derivative and Taylor remainder in u

Write

\[
g_e(1)
=
g_e(0)
+
g'_e(0)
+
\int_0^1(1-u)g''_e(u)\,du.
\]

After two \(u\)-derivatives, every term contains two cavity overlaps, times
bounded spin factors and a bounded test function.

The desired estimate is

\[
\sup_{u\in[0,1]}|g''_e(u)|
=
o(1)
\]

or the stronger

\[
O(N^{-1/2}).
\]

A clean way to prove this is first to establish a cavity-uniform second-moment
bound

\[
\sup_{u\in[0,1]}
\nu_u[(Q^-_{ab})^2]
\le
\frac{C}{N}.
\]

This should follow from the already proved endpoint estimate and a Gronwall
argument using the same cavity derivative formula.

Then Cauchy-Schwarz gives, for any two replica edges,

\[
\nu_u[|Q^-_{ab}Q^-_{cd}|]
\le
\frac{C}{N}.
\]

Since \(G_e\) carries the prefactor \(\sqrt N\),

\[
|g''_e(u)|
\le
\frac{C_f}{\sqrt N}.
\]

If the direct second-moment propagation is awkward in Lean, the existing
cavity-uniform third-moment machinery is an acceptable alternative. In that
case record the precise power of \(N\) and verify it tends to zero.

### Endpoint replacement

The \(u=0\) equations naturally involve \(Q^-_{ab}\) and a cavity version of
the scaled overlap.

The final vector \(T(f)\) is defined using the full overlaps at \(u=1\).

Prove a general endpoint-replacement lemma for bounded Lipschitz test
functions. A useful target is

\[
\left|
\sqrt N\,\nu_0[
Q^-_{ab}f(Y^-_{12})
]
-
\nu_1[
X_{ab}f(X_{12})
]
\right|
\le
C_f\left(
N^{-1/2}
+
N\nu_1[|Q_{12}|^3]
\right).
\]

The exact right side may differ, but it must tend to zero using results that
are already upstream of the CLT.

Avoid a proof that invokes the CLT itself or any result that depends on the
CLT.

## Linear algebra layer

This part should be made almost entirely deterministic.

Create lemmas that express the approximate Stein system first in
\((A,B,C)\) coordinates and then in \((V,U,D)\) coordinates.

Recommended future file:

```text
Lemmas/CLT/SteinLinearAlgebra.lean
```

Reuse:

```text
cavityMatrix
cavityChangeMatrix
cavityChangeMatrixInv
cavityChangeMatrix_mul_cavityMatrix
cavityChangeMatrixInv_mul_cavityChangeMatrix
cavityChangeMatrixInv_mulVec_eq
cavityKappa
cavityZeta
```

Do not create a second change-of-basis matrix.

The strict AT inequalities needed are already encoded in the cavity
development:

\[
1-\alpha>0,
\]

\[
1-\beta^2\kappa>0.
\]

The second inequality follows from

\[
\beta^2\kappa\le\alpha<1.
\]

After solving the mode system, prove the first-coordinate identity by `ring`
or `field_simp` plus the known nonzero denominators.

The formal target should simplify to the exact `σ2` expression in
`Claim.lean`.

## Characteristic-function layer

Once the scalar approximate Stein identity is available, avoid adding general
probability weak-convergence machinery.

Define, for fixed \(t\),

\[
C_N(t)
=
\nu_N[\cos(tX_N)],
\]

\[
S_N(t)
=
\nu_N[\sin(tX_N)].
\]

Use the Stein identity with

\[
f(x)=\sin(tx)
\]

to obtain

\[
\nu_N[X_N\sin(tX_N)]
=
\sigma^2t\,C_N(t)
+o(1).
\]

Since

\[
C_N'(t)
=
-\nu_N[X_N\sin(tX_N)],
\]

this gives

\[
C_N'(t)
=
-\sigma^2tC_N(t)
+o(1).
\]

Use the Stein identity with

\[
f(x)=\cos(tx)
\]

to obtain

\[
\nu_N[X_N\cos(tX_N)]
=
-\sigma^2tS_N(t)
+o(1),
\]

hence

\[
S_N'(t)
=
-\sigma^2tS_N(t)
+o(1).
\]

Initial values are

\[
C_N(0)=1,
\qquad
S_N(0)=0.
\]

There are two reasonable formal routes.

### Route A: approximate ODE

Prove uniform error on every compact \(t\)-interval:

\[
\sup_{|t|\le T}
\left|
C_N'(t)+\sigma^2tC_N(t)
\right|
\to0,
\]

\[
\sup_{|t|\le T}
\left|
S_N'(t)+\sigma^2tS_N(t)
\right|
\to0.
\]

Multiply by the integrating factor

\[
e^{\sigma^2t^2/2}.
\]

For cosine,

\[
\frac{d}{dt}
\left(
e^{\sigma^2t^2/2}C_N(t)
\right)
=
e^{\sigma^2t^2/2}\operatorname{Err}_{N,c}(t).
\]

Integrate from \(0\) to \(t\). The right side tends uniformly to zero on
compact intervals, so

\[
C_N(t)
\to
e^{-\sigma^2t^2/2}.
\]

The sine equation and \(S_N(0)=0\) give

\[
S_N(t)\to0.
\]

This route is recommended because it avoids subsequence compactness.

### Route B: subsequential ODE

Use boundedness

\[
|C_N(t)|\le1,
\qquad
|S_N(t)|\le1
\]

and derivative bounds following from the second moment:

\[
|C_N'(t)|,
|S_N'(t)|
\le
\nu_N[|X_N|]
\le
\sqrt{\nu_N[X_N^2]}
\le C.
\]

Apply Arzela-Ascoli on compact intervals, identify every subsequential limit
from the exact limiting ODE, and conclude uniqueness.

This route is mathematically clean but may require more topology in Lean.

Prefer Route A unless the needed integral ODE lemmas become harder than the
compactness lemmas.

## Test-function regularity

For the CLT endpoint only, it is enough to support

```text
f x = Real.sin (t * x)
f x = Real.cos (t * x)
```

and their derivatives.

Do not build an unnecessarily general \(C_b^3\) function space before the
core proof works.

A practical sequence is:

```text
prove cavity-Stein for a function f together with explicit constants
M0, M1, M2 satisfying pointwise bounds on f, f', f'';

instantiate with sin;

instantiate with cos.
```

If a third cavity derivative is eventually used, add a bound on the third
derivative then.

For sine and cosine, all derivative bounds reduce to polynomial expressions
in \(|t|\).

## Bias issue

The claimed limit is centered at \(q\), not at the finite-volume mean overlap.

Therefore the proof must control any possible

\[
\sqrt N\,\nu_N[Q_{12}]
\]

bias.

The cavity-Stein identity with the constant test function

\[
f\equiv1
\]

should give

\[
\nu_N[X_{12}]=o(1),
\]

because \(f'=0\).

Make this an explicit corollary and use it as a consistency check.

Do not silently assume \(\nu_N[R_{12}]=q\) at finite \(N\).

## Variance positivity

A Gaussian variance must satisfy

\[
\sigma^2\ge0.
\]

The easiest formal proof may be to use the already proved second-moment limit

\[
N A_N\to\sigma^2
\]

and nonnegativity of \(A_N\), rather than prove positivity directly from the
rational formula.

Add a theorem

```text
overlapCLTVariance_nonneg
```

only after the characteristic-function theorem is working, unless a
nonnegativity fact is required earlier by an exponential identity.

## Suggested future Lean file decomposition

The current folder intentionally contains only the final claim. During the
proof phase, use the following decomposition.

```text
Lemmas/CLT/
├── AGENTS.md
├── blueprint.md
├── Claim.lean
├── Basic.lean
├── CavityMoments.lean
├── SteinCavity.lean
├── SteinLinearAlgebra.lean
├── Characteristic.lean
└── MainResult.lean
```

Suggested responsibilities:

```text
Basic.lean
  scaled overlap notation
  cosine and sine quenched expectations
  elementary bounds
  site exchangeability wrappers

CavityMoments.lean
  cavity-uniform second moment
  endpoint replacement
  Taylor bounds at u=0

SteinCavity.lean
  three-component test-function cavity identity
  error estimate tending to zero

SteinLinearAlgebra.lean
  mode transformation
  inversion
  identification of sigma^2
  scalar approximate Stein identity

Characteristic.lean
  sine/cosine derivative formulas
  approximate ODE
  integrating-factor argument

MainResult.lean
  assemble the characteristic-function CLT
  prove the exact statement of Claim.lean
```

Do not put all analytic work into one giant file.

## Recommended theorem interfaces

The names below are suggestions. Adjust only when the existing APIs make
another signature substantially cleaner.

### Scaled overlap

```lean
noncomputable def scaledCenteredOverlap
    {N n : ℕ} (q : ℝ) (σs : Replicas N n)
    (a b : Fin n) : ℝ :=
  Real.sqrt N * centeredOverlap q σs a b
```

Because the final statement uses positive system sizes, decide once whether
the formal scale is `Real.sqrt (N : ℝ)` or a `N.succ` indexed version.

Keep that choice consistent.

### Test-function Stein vector

```lean
noncomputable def steinA ...
noncomputable def steinB ...
noncomputable def steinC ...
```

Each should use the same test function applied to the distinguished overlap
`12`.

### Approximate vector Stein identity

A good proposition shape is

```text
‖T_N(f) - cavityMatrix * T_N(f) - theta * nu[f']‖
  ≤ error_N(f)
```

with a separately proved

```text
Tendsto error_N atTop (nhds 0).
```

Avoid hiding the convergence inside an opaque existential.

### Scalar Stein identity

A good endpoint theorem is

```text
|nu[X12 * f(X12)] - sigma2 * nu[f'(X12)]|
  ≤ scalarError_N(f)
```

together with

```text
Tendsto scalarError_N atTop (nhds 0).
```

This interface makes the characteristic-function file nearly independent of
replica combinatorics.

## Dependency discipline

The CLT is downstream of the existing quantitative strict-AT result.

Allowed dependencies include:

```text
Lemmas.MainResult
Lemmas.Cavity.TalagrandCavity
Lemmas.Cavity.Estimates
Lemmas.Cavity.Interpolation
Lemmas.AT.Definitions
Lemmas.AT.FixedPoint
```

Do not make any existing concentration or cavity file import `Lemmas.CLT`.
That would create a logical cycle.

A good dependency direction is

```text
existing RSAT theorem
        |
        v
CLT/Basic
        |
        v
CLT/CavityMoments
        |
        v
CLT/SteinCavity
        |
        v
CLT/SteinLinearAlgebra
        |
        v
CLT/Characteristic
        |
        v
CLT/MainResult
```

`Claim.lean` is the statement contract. During development, either keep it
as a standalone sorry-bearing endpoint or let the final `MainResult.lean`
prove the same proposition and then replace the sorry only at the end.

## Avoiding circularity

Never use any of the following to prove the cavity-Stein identity:

```text
the overlap CLT,
Gaussianity of the overlap,
convergence of all overlap moments derived from the CLT,
the final characteristic-function limit.
```

Allowed inputs are the already proved strict-AT estimates, cavity
interpolation, Gaussian integration by parts, deterministic mode algebra,
Taylor estimates, Holder, Cauchy-Schwarz, Gronwall, and elementary calculus.

The variance should be identified from the already proved cavity second-order
system, not assumed.

## Uniformity strategy

The current claim is pointwise in \((\beta,h)\), but most RSAT estimates are
formulated uniformly over compact strict-AT sets.

For a fixed strict-AT point, use the singleton compact set

\[
K=\{(\beta,h)\}.
\]

Alternatively, create a small closed neighborhood contained in the strict AT
region if a theorem requires a positive uniform margin.

Do not duplicate the proof of compact strict-AT data extraction.

If the proof later targets a uniform CLT over a compact \(K\), first finish
the pointwise theorem. Then strengthen each error estimate to be uniform in
the parameters.

## Handling N

The paper uses \(N\ge1\).

`Claim.lean` indexes the sequence by `N : ℕ` but uses the physical size

```text
N.succ
```

everywhere.

This avoids `NeZero 0` and division-by-zero side conditions in overlap
definitions.

Do not switch back and forth between `N` and `N.succ` inside a file without
explicit conversion lemmas.

## Build strategy

From `RSAT/` compile the smallest edited file first.

Examples:

```bash
lake env lean Lemmas/CLT/Claim.lean
lake env lean Lemmas/CLT/Basic.lean
lake env lean Lemmas/CLT/CavityMoments.lean
lake env lean Lemmas/CLT/SteinCavity.lean
lake env lean Lemmas/CLT/SteinLinearAlgebra.lean
lake env lean Lemmas/CLT/Characteristic.lean
lake env lean Lemmas/CLT/MainResult.lean
```

Then run

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
```

While the CLT project is intentionally under construction, `Claim.lean`
contains `sorry`. Once the proof is finished, the project-level integrity
check should again find no placeholders.

## Debugging order

When the proof does not close, debug in this order.

1. Check the three last-spin source coefficients:
   \[
   1-q^2,\quad q-q^2,\quad r-q^2.
   \]

2. Check the row ordering of the mode basis:
   \[
   (V,U,D).
   \]

3. Check that the cavity matrix in the Stein calculation is literally the
   existing `cavityMatrix`.

4. Check every power of \(N\):
   \[
   Q=O(N^{-1/2}),\qquad X=\sqrt N Q=O(1)
   \]
   in \(L^2\).

5. Check that a second cavity derivative contains two cavity overlaps, so the
   prefactor \(\sqrt N\) still leaves an \(O(N^{-1/2})\) error.

6. Check the finite-volume bias by taking \(f=1\).

7. Check the variance against the already proved limit of \(NA\).

8. Only after these checks inspect the ODE argument.

This order usually finds a replica-counting or normalization problem before
it is buried under analytic estimates.

## Milestones

### Milestone A

`Claim.lean` parses and compiles with the single `sorry`.

### Milestone B

Formalize the scaled-overlap notation and all sine/cosine elementary bounds.

### Milestone C

Prove the cavity-uniform second moment and test-function endpoint replacement.

### Milestone D

Prove the three-component approximate Stein system.

### Milestone E

Reuse the existing mode algebra and identify the scalar coefficient with
`σ2`.

### Milestone F

Prove the approximate Stein identity for sine and cosine.

### Milestone G

Solve the approximate ODE and prove both characteristic-function limits.

### Milestone H

Replace the `sorry` in `Claim.lean`.

### Milestone I

Run the full RSAT build and placeholder scan.

## Final mathematical checks

Before declaring the proof finished, verify all of the following.

At \(\beta=0\), as a formal consistency limit,

\[
\sigma^2
=
1-q^2.
\]

At smart-path time \(s=0\), the covariance classes should reduce to

\[
A_0=1-q^2,
\]

\[
B_0=q-q^2,
\]

\[
C_0=r-q^2
\]

after scaling by \(N\).

At \(s=1\), the variance obtained from the Stein system must equal the
second-moment cavity limit.

The limiting sine characteristic component must be zero.

The limiting cosine component must equal one at \(t=0\).

The variance must be nonnegative.

The theorem must center at `rsQ β h`, not at the finite-volume mean overlap.

No proof step may assume finite-volume equality
\[
\nu_N[R_{12}]=q.
\]

## Scope of the first CLT project

The first completed theorem should be the one-dimensional overlap CLT in
`Claim.lean`.

Do not initially formalize:

```text
the full joint Gaussian array of all overlaps,
a Berry-Esseen rate,
a quenched-in-disorder CLT,
free-energy fluctuations,
critical AT-line behavior.
```

After the scalar theorem is complete, the three-component Stein calculation
can be generalized to a fixed finite replica array. That extension should be
a separate project so that it does not block the first kernel-checked CLT.