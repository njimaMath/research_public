# Generalized Latała bound for convex mixed p-spin models

This directory contains a Lean 4 formalization of a generalized Latała
interpolation argument for finite-volume convex mixed p-spin models. The
formalization builds on the
[`or4nge19/SpinGlass`](https://github.com/or4nge19/SpinGlass) library. The main
result proves overlap concentration and an $O(1/N)$ error bound for the
replica-symmetric pressure in the improved high-temperature region

$$
\rho(\Gamma,q)=\Gamma\kappa(q)<1,
\qquad
\kappa(q)=
\begin{cases}
\dfrac{q}{\operatorname{artanh}(q)}, & q\ne 0,\\
1, & q=0.
\end{cases}
$$

The public entry point is [`mainresult_latala.lean`](mainresult_latala.lean), and
the main theorem is
[`SpinGlass.GeneralizedLatala.model_result`](mainresult_latala.lean#L139).

## Browse the formalization

- [`mainresult_latala.lean`](mainresult_latala.lean) contains the public model
  definitions and main theorem.
- [`GeneralizedLatala/Proof.lean`](GeneralizedLatala/Proof.lean) is the thin
  assembly module for the internal theorem `generalized_latala`.
- [`GeneralizedLatala/Basic.lean`](GeneralizedLatala/Basic.lean) and
  [`GeneralizedLatala/Observables.lean`](GeneralizedLatala/Observables.lean)
  define the scalar replica-symmetric data and smart-path observables.
- [`GeneralizedLatala/Endpoint/`](GeneralizedLatala/Endpoint/) contains the
  independent Gaussian endpoint, affine Gaussian integration by parts, and
  the endpoint exponential-moment estimates.
- [`GeneralizedLatala/Interpolation/`](GeneralizedLatala/Interpolation/)
  contains the ordinary pressure interpolation and the quadratic-coupling
  characteristic argument.
- [`GeneralizedLatala/Coupled/`](GeneralizedLatala/Coupled/) contains the
  coupled free energy, its calculus and Gaussian trace layers, finite replica
  algebra, and integrability estimates.
- [`GeneralizedLatala/Consequences.lean`](GeneralizedLatala/Consequences.lean)
  proves overlap concentration and the replica-symmetric pressure sum rule.
- [`SpinGlass/`](SpinGlass/) is organized into reusable core, Gaussian, model,
  replica, interpolation, inequality, and local Mathlib support modules.
- [`blueprint_latala.txt`](blueprint_latala.txt) is the informal mathematical
  blueprint followed by the Lean proof.

## Mathematical statement

For $N\ge 1$, a spin configuration is a function
$\sigma:{\rm Fin}(N)\to{\rm Bool}$, interpreted as a vector of
signs in $\{-1,1\}^N$. Its normalized overlap with $\tau$ is

$$
R(\sigma,\tau)=\frac1N\sum_{i=1}^N \sigma_i\tau_i.
$$

Given a covariance function $\xi$, the disorder is represented abstractly by a
centered Gaussian random field $U(\sigma)$ with covariance

$$
{\rm Cov}(U(\sigma),U(\tau))
=N\xi\!\left(R(\sigma,\tau)\right).
$$

With the sign convention used in the Lean definitions, the energy and Gibbs
weight are

$$
E_N(\sigma)=U(\sigma)+h\sum_{i=1}^N\sigma_i,
\qquad
G_N(\sigma)=\frac{\exp(-E_N(\sigma))}{Z_N}.
$$

Let $d=\xi'(q)\ge0$, and define the Bregman remainder

$$
\Delta_q(r)=\xi(r)-\xi(q)-d(r-q).
$$

Assume that, for every $r\in[-1,1]$,

$$
0\le\Delta_q(r)\le\frac{\Gamma}{2}(r-q)^2
$$

for some $\Gamma\ge0$. Let $Z$ be a standard real Gaussian and suppose that
$q$ satisfies

$$
0\le q<1,
\qquad
q=\mathbb E\!\left[
\tanh\!\left(h+\sqrt d\,Z\right)^2
\right].
$$

If $\rho(\Gamma,q)<1$, then there is a finite common constant $C\ge0$ such
that

$$
\mathbb E\left\langle(R_{12}-q)^2\right\rangle
\le \frac{C}{N}
$$

and

$$
0\le
\phi^{\mathrm{RS}}(\xi,d,h,q)-\phi_N
\le\frac{C}{N},
$$

where

$$
\phi_N=\frac1N\mathbb E[\log Z_N]
$$

and

$$
\phi^{\mathrm{RS}}(\xi,d,h,q)
=\log 2
+\mathbb E\!\left[
\log\cosh\!\left(h+\sqrt d\,Z\right)
\right]
+\frac12\Delta_q(1).
$$

In Lean, the two conclusions are bundled as `ModelClaims`. The proof actually
constructs a common constant from

$$
\lambda_{\ast}=\frac{\kappa(q)^{-1}-\Gamma}{4}
$$

and

$$
K(\Gamma,q)=
\frac12
\exp\!\left(\frac{2\rho}{1-\rho}\right)
\log\!\left(\frac{2}{1-\rho}\right).
$$

One may take

$$
C=\frac{K(\Gamma,q)}{\lambda_{\ast}}
+\frac{\Gamma K(\Gamma,q)}{4\lambda_{\ast}}.
$$

For the Sherrington--Kirkpatrick covariance
$\xi(r)=\beta^2r^2/2$, one has $d=\beta^2q$ and $\Gamma=\beta^2$, recovering
the improved condition $\beta^2\kappa(q)<1$.

## Lean interface

The theorem takes the following data:

- a probability space `Ω`;
- a positive system size `N`;
- a covariance function `ξ` and real parameters `d`, `Γ`, `h`, and `q`;
- `sk : ModelMixedDisorder N ξ d h`, carrying the centered Gaussian mixed
  p-spin field and its covariance identity;
- `sim : SimpleDisorder N d q`, the Gaussian one-site reference field used by
  the smart-path interpolation;
- the Bregman, fixed-point, and improved-region hypotheses above;
- `IndepFun sk.U sim.V ℙ`, expressing independence of the mixed and reference
  disorders.

The reference disorder is a technical input to the interpolation proof. It
does not occur in `ModelClaims`.

The definitions intended for users are all in
`SpinGlass.GeneralizedLatala`:

- `ModelConfig`, `modelSpin`, and `modelOverlap`;
- `ModelMixedDisorder`;
- `modelEnergy`, `modelPartitionFunction`, and `modelGibbsProbability`;
- `modelPressure` and `modelOverlapSecondMoment`;
- `ModelFixedPoint`, `modelKappa`, `modelBregmanRemainder`,
  `ModelBregmanBounds`, `modelRho`, and `modelRSPressure`;
- `ModelClaims` and `model_result`.

## Proof outline

The formal proof follows [`blueprint_latala.txt`](blueprint_latala.txt).

- [`GeneralizedLatala/Basic.lean`](GeneralizedLatala/Basic.lean) records the
  scalar Gaussian identities, fixed-point data, and high-temperature
  parameters.
- [`GeneralizedLatala/Endpoint/Estimates.lean`](GeneralizedLatala/Endpoint/Estimates.lean)
  combines Kearns--Saul with Hubbard--Stratonovich at the independent endpoint.
- [`GeneralizedLatala/Interpolation/Pressure.lean`](GeneralizedLatala/Interpolation/Pressure.lean)
  proves the ordinary Guerra smart-path derivative.
- [`GeneralizedLatala/Coupled/Core.lean`](GeneralizedLatala/Coupled/Core.lean),
  [`Calculus.lean`](GeneralizedLatala/Coupled/Calculus.lean), and
  [`GaussianIBP.lean`](GeneralizedLatala/Coupled/GaussianIBP.lean) construct and
  differentiate the coupled two-replica free energy.
- [`GeneralizedLatala/Coupled/ReplicaAlgebra.lean`](GeneralizedLatala/Coupled/ReplicaAlgebra.lean)
  and [`Integrability.lean`](GeneralizedLatala/Coupled/Integrability.lean)
  evaluate the Gaussian trace and justify the normalized finite-state
  observables.
- [`GeneralizedLatala/Interpolation/QuadraticCoupling.lean`](GeneralizedLatala/Interpolation/QuadraticCoupling.lean)
  proves the differential inequality, follows its characteristic, and applies
  Gronwall to obtain the uniform quadratic estimate when $\rho<1$.
- [`GeneralizedLatala/Consequences.lean`](GeneralizedLatala/Consequences.lean)
  converts that estimate into the $O(1/N)$ overlap and pressure bounds.
- [`GeneralizedLatala/Proof.lean`](GeneralizedLatala/Proof.lean) assembles the
  final internal theorem used by `model_result`.

The proof uses interpolation and replica calculus, not the cavity method.

## Type-checking

The formalization targets Lean `v4.32.1` and mathlib `v4.32.1`. In this
checkout, the Lake project at `C:\Users\Public\Github\Lean` registers this
source tree and pins the required mathlib version. From that project root, run:

```powershell
lake lean research_public/generalizedLatala/mainresult_latala.lean
```

This builds the transitive imports and checks the public entry point. Once its
imports have been built, the direct check also succeeds:

```powershell
lake env lean research_public/generalizedLatala/mainresult_latala.lean
```

The `generalizedLatala/` directory remains a source subtree and does not carry
its own `lakefile.lean` or `lean-toolchain`.
