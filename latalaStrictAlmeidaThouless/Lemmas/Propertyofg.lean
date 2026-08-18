/-
Fix $(\beta,h,s)\in\cK\times[0,1]$. For $t\ge0$, let $\mathsf H_t$ be the heat semigroup, i.e. \[ (\mathsf H_t\varphi)(x)\coloneqq\E\varphi(x+\sqrt t\,Z) . \] Then, it holds that \begin{equation}\label{eq:genH} \partial _t (\mathsf H_t\varphi)(x) = \frac{1}{2} \mathsf H_t (\partial ^2 \varphi)(x) . \end{equation} For the upper interval, introduce the tilted heat semigroup \begin{equation} (\mathsf T_t\varphi)(x) \coloneqq e^{-t/2} \frac{\bigl(\mathsf H_t (\varphi \cdot \cosh )\bigr)(x)}{\cosh(x)}. \label{eq:tiltedsemigroup} \end{equation} Differentiation under the Gaussian integral shows that its generator is $\frac12\partial_{xx}+\tanh(x)\partial_x$. For each $u\in[0,1]$, let $X_{s,u}$ denote a random variable whose law is specified by \begin{equation} \E\varphi(X_{s,u}) \coloneqq \begin{cases} \bigl(\mathsf H_{\beta^2(1-s)q+s\beta^2u}\varphi\bigr)(h), &0\le u\le q,\\[1mm] \bigl(\mathsf H_{\beta^2q} (\mathsf T_{s\beta^2(u-q)}\varphi)\bigr)(h), &q\le u\le1. \end{cases} \label{eq:localfieldlaw} \end{equation} In particular, \begin{equation}\label{eq:Xsq} X_{s,q}\stackrel{\mathrm d}=h+\beta\sqrt q\,Z. \end{equation} For $0\le u\le1$ and $x\in\mathbb R$, define \begin{equation} \Psi(u,x) \coloneqq \E\log\cosh\left( x+\beta\sqrt{s(q-u)_+}\,Z \right) +\frac{s\beta^2}{2}\bigl(1-\max\{u,q\}\bigr), \qquad Z\sim N(0,1), \label{eq:Psi-explicit} \end{equation} where $(a)_+\coloneqq\max\{a,0\}$. Equivalently, \[ \Psi(u,x) = \begin{cases} (\mathsf H_{s\beta^2(q-u)}\log\cosh)(x) +\dfrac{s\beta^2}{2}(1-q), &0\le u\le q,\\[2mm] \log\cosh(x)+\dfrac{s\beta^2}{2}(1-u), &q\le u\le1. \end{cases} \] Differentiating \eqref{eq:Psi-explicit} in $x$ gives \begin{equation} \partial_x\Psi(u,x) = \E\tanh\left( x+\beta\sqrt{s(q-u)_+}\,Z \right) = \begin{cases} (\mathsf H_{s\beta^2(q-u)}\tanh)(x), &0\le u\le q,\\ \tanh(x),&q\le u\le1. \end{cases} \label{eq:Psi-x} \end{equation} The replica-symmetric free energy for $H_{N,s}$ along the path is \begin{align} P_s^*& \coloneqq\log2+ \E\Psi(0,h+\beta\sqrt{(1-s)q}\,Z) -\frac{s\beta^2}{2}\int_q^1u\,\dd u \label{eq:scalartrial}\\ &= \log2+\E\log\cosh(h+\beta\sqrt q\,Z) +\frac{s\beta^2}{4}(1-q)^2, \label{eq:RSpathvalue} \end{align} where we have used the semigroup property and $\beta^2(1-s)q+s\beta^2q=\beta^2q$. \begin{proposition} \label{prop:pathRS} The function $g_s$ defined by \begin{equation} g_s(u)\coloneqq\E\bigl[(\partial_x\Psi(u,X_{s,u}))^2\bigr] \label{eq:gs-def} \end{equation} satisfies \begin{equation} g_s(u)-u \begin{cases} >0,&0\le u<q,\\ =0,&u=q,\\ <0,&q<u\le1. \end{cases} \label{eq:ATsign} \end{equation} Moreover, there are constants $c_{\cK},\varepsilon_{\cK}>0$ such that \begin{equation} |g_s(u)-u|\ge c_{\cK}|u-q| \qquad\text{whenever }|u-q|\le\varepsilon_{\cK}. \label{eq:linearATsign} \end{equation} \end{proposition}
\begin{proof}
Throughout the proof, write
[
a\coloneqq s\beta^2.
]
We also recall the definitions
[
q=\E\tanh^2(h+\beta\sqrt q,Z),
\qquad
\alpha=\beta^2\E\sech^4(h+\beta\sqrt q,Z),
]
and the standing assumption on $\cK$ that
[
\alpha<1.
]

First consider $u=q$. By \eqref{eq:Psi-x},
[
\partial_x\Psi(q,x)=\tanh(x),
]
while \eqref{eq:Xsq} gives
[
X_{s,q}\stackrel{\mathrm d}=h+\beta\sqrt q,Z.
]
Consequently,
[
g_s(q)
======

# \E\tanh^2(X_{s,q})

# \E\tanh^2(h+\beta\sqrt q,Z)

q.
\tag{1}
]

We next consider the interval $0\le u<q$.

Set
[
r(u)\coloneqq a(q-u),
\qquad
\tau(u)\coloneqq\beta^2(1-s)q+a u.
]
Then
[
r'(u)=-a,
\qquad
\tau'(u)=a,
]
and, by \eqref{eq:localfieldlaw},
[
\E\varphi(X_{s,u})
==================

(\mathsf H_{\tau(u)}\varphi)(h).
\tag{2}
]
Thus, for every smooth $\varphi$ with bounded derivatives,
[
\frac{\dd}{\dd u}\E\varphi(X_{s,u})
===================================

\frac{a}{2}
(\mathsf H_{\tau(u)}\varphi'')(h)
=================================

\frac{a}{2}\E\varphi''(X_{s,u}).
\tag{3}
]
Here we used \eqref{eq:genH} and $\tau'(u)=a$.

Define
[
F_u(x)\coloneqq
(\mathsf H_{r(u)}\tanh)(x).
]
By \eqref{eq:Psi-x},
[
g_s(u)=\E[F_u(X_{s,u})^2].
\tag{4}
]
Moreover, differentiating $F_u$ with respect to $u$ and using
\eqref{eq:genH},
[
\partial_uF_u(x)
================

r'(u)\frac12
\mathsf H_{r(u)}(\partial_x^2\tanh)(x)
======================================

-\frac a2 F_u''(x).
\tag{5}
]
Since the derivatives of $\tanh$ are bounded, all differentiations above
and below are justified by differentiation under the Gaussian integral.

Differentiating \eqref{eq:4} and using \eqref{eq:3} with
$\varphi=F_u^2$, we obtain
\begin{align*}
g_s'(u)
&=
\E\bigl[2F_u(X_{s,u})\partial_uF_u(X_{s,u})\bigr]
+\frac a2\E\bigl[(F_u^2)''(X_{s,u})\bigr]\
&=
-a\E\bigl[F_u(X_{s,u})F_u''(X_{s,u})\bigr]
+\frac a2
\E\bigl[(F_u^2)''(X_{s,u})\bigr].
\end{align*}
Since
[
(F_u^2)''
=========

2(F_u')^2+2F_uF_u'',
]
the terms containing $F_uF_u''$ cancel, and hence
[
g_s'(u)
=======

a,\E\bigl[(F_u'(X_{s,u}))^2\bigr].
\tag{6}
]
The heat semigroup commutes with spatial differentiation, so
[
F_u'
====

# \mathsf H_{r(u)}(\partial_x\tanh)

\mathsf H_{r(u)}(\sech^2).
]
Therefore
[
g_s'(u)
=======

a\E\left[
\left{
\mathsf H_{r(u)}(\sech^2)(X_{s,u})
\right}^2
\right].
\tag{7}
]

For every $x\in\mathbb R$, Jensen's inequality gives
[
\left{
\mathsf H_{r(u)}(\sech^2)(x)
\right}^2
\le
\mathsf H_{r(u)}(\sech^4)(x).
]
Using \eqref{eq:2},
\begin{align*}
g_s'(u)
&\le
a,\E\left[
\mathsf H_{r(u)}(\sech^4)(X_{s,u})
\right]\
&=
a,
\bigl(
\mathsf H_{\tau(u)}
\mathsf H_{r(u)}
(\sech^4)
\bigr)(h).
\end{align*}
Now
\begin{align*}
\tau(u)+r(u)
&=
\beta^2(1-s)q
+s\beta^2u
+s\beta^2(q-u)\
&=
\beta^2q.
\end{align*}
Hence, by the semigroup property,
\begin{align*}
g_s'(u)
&\le
a,(\mathsf H_{\beta^2q}\sech^4)(h)\
&=
s\beta^2
\E\sech^4(h+\beta\sqrt q,Z)\
&=
s\alpha.
\tag{8}
\end{align*}

Integrating \eqref{eq:8} from $u$ to $q$ and using \eqref{eq:1},
[
q-g_s(u)
========

# g_s(q)-g_s(u)

\int_u^qg_s'(v),\dd v
\le
s\alpha(q-u).
]
It follows that
\begin{align}
g_s(u)-u
&=
(q-u)-(q-g_s(u))\notag\
&\ge
(1-s\alpha)(q-u).
\tag{9}\label{eq:leftstrict-detailed}
\end{align}
Since $0\le s\le1$ and $\alpha<1$,
[
1-s\alpha\ge1-\alpha>0.
]
Thus
[
g_s(u)-u>0,
\qquad 0\le u<q.
\tag{10}
]

We now turn to the interval $q\le u\le1$.

It is useful first to record the generator of the tilted heat semigroup
explicitly. Let
[
\mathcal L\varphi
\coloneqq
\frac12\varphi''+\tanh(x)\varphi'.
\tag{11}
]
From \eqref{eq:tiltedsemigroup},
[
(\mathsf T_t\varphi)(x)
=======================

e^{-t/2}
\frac{\mathsf H_t(\varphi\cosh)(x)}{\cosh(x)}.
]
Differentiating in $t$,
\begin{align*}
\partial_t(\mathsf T_t\varphi)(x)
&=
-\frac12(\mathsf T_t\varphi)(x)
+
\frac{e^{-t/2}}{2\cosh(x)}
\mathsf H_t\bigl((\varphi\cosh)''\bigr)(x).
\end{align*}
Since
[
(\varphi\cosh)''
================

\varphi''\cosh
+2\varphi'\sinh
+\varphi\cosh,
]
the last $\varphi\cosh$ term cancels the term
$-\frac12\mathsf T_t\varphi$. Therefore
\begin{align*}
\partial_t(\mathsf T_t\varphi)(x)
&=
\frac{e^{-t/2}}{\cosh(x)}
\mathsf H_t
\left(
\frac12\varphi''\cosh+\varphi'\sinh
\right)(x)\
&=
\mathsf T_t
\left(
\frac12\varphi''+\tanh,\varphi'
\right)(x)\
&=
\mathsf T_t(\mathcal L\varphi)(x).
\tag{12}
\end{align*}

For $u\ge q$, \eqref{eq:localfieldlaw} becomes
[
\E\varphi(X_{s,u})
==================

\bigl(
\mathsf H_{\beta^2q}
\mathsf T_{a(u-q)}\varphi
\bigr)(h).
\tag{13}
]
Differentiating \eqref{eq:13} in $u$ and using \eqref{eq:12},
\begin{align}
\frac{\dd}{\dd u}\E\varphi(X_{s,u})
&=
a,
\bigl(
\mathsf H_{\beta^2q}
\mathsf T_{a(u-q)}
(\mathcal L\varphi)
\bigr)(h)\notag\
&=
a,\E[(\mathcal L\varphi)(X_{s,u})].
\tag{14}
\end{align}

On this interval,
[
\partial_x\Psi(u,x)=\tanh(x),
]
and hence
[
g_s(u)=\E\tanh^2(X_{s,u}).
\tag{15}
]
Set
[
\phi(x)\coloneqq\tanh^2(x).
]
Then
[
\phi'(x)=2\tanh(x)\sech^2(x),
]
and
[
\frac12\phi''(x)
================

\sech^4(x)
-2\tanh^2(x)\sech^2(x).
]
Consequently,
\begin{align*}
(\mathcal L\phi)(x)
&=
\frac12\phi''(x)+\tanh(x)\phi'(x)\
&=
\sech^4(x)
-2\tanh^2(x)\sech^2(x)
+2\tanh^2(x)\sech^2(x)\
&=
\sech^4(x).
\tag{16}
\end{align*}
Applying \eqref{eq:14} to $\phi=\tanh^2$ yields the particularly simple
identity
\begin{equation}
g_s'(u)
=======

a,\E\sech^4(X_{s,u}),
\qquad q\le u\le1.
\tag{17}\label{eq:gs-upper-derivative-detailed}
\end{equation}
At $u=q$,
[
X_{s,q}\stackrel{\mathrm d}=h+\beta\sqrt q,Z,
]
so
\begin{align}
g_s'(q)
&=
a,\E\sech^4(h+\beta\sqrt q,Z)\notag\
&=
s\alpha
<
1.
\tag{18}\label{eq:gsprimeq-detailed}
\end{align}

Suppose first that
[
a(1-q)=s\beta^2(1-q)>1.
]
By Lemma~\ref{lem:uppercomparison},
[
g_s'(u)\le s\alpha<1,
\qquad q\le u\le1.
]
Thus, for every $u>q$,
\begin{align*}
g_s(u)-g_s(q)
&=
\int_q^u g_s'(v),\dd v\
&<
u-q.
\end{align*}
Since $g_s(q)=q$, this gives
[
g_s(u)<u,
\qquad q<u\le1.
\tag{19}
]

It remains to consider
[
a(1-q)\le1.
\tag{20}
]
Since
[
\sech^4(x)\le\sech^2(x)
=1-\tanh^2(x),
]
\eqref{eq:gs-upper-derivative-detailed} and \eqref{eq:15} imply
\begin{align}
g_s'(u)
&\le
a,\E\sech^2(X_{s,u})\notag\
&=
a\left(
1-\E\tanh^2(X_{s,u})
\right)\notag\
&=
a(1-g_s(u)).
\tag{21}\label{eq:gderivative-detailed}
\end{align}

Define
[
f(u)\coloneqq g_s(u)-u.
]
Then $f(q)=0$, and \eqref{eq:gderivative-detailed} gives
\begin{align*}
f'(u)
&=
g_s'(u)-1\
&\le
a(1-g_s(u))-1\
&=
-af(u)+a(1-u)-1.
\end{align*}
Hence
\begin{equation}
f'(u)+af(u)
\le
a(1-u)-1.
\tag{22}
\end{equation}
Multiplication by the integrating factor $e^{a(u-q)}$ gives
[
\frac{\dd}{\dd u}
\left(
e^{a(u-q)}f(u)
\right)
\le
e^{a(u-q)}\bigl(a(1-u)-1\bigr).
\tag{23}
]
For every $u>q$, \eqref{eq:20} implies
[
a(1-u)-1
<
a(1-q)-1
\le0
]
when $a>0$. If $a=0$, the right-hand side of \eqref{eq:23} equals
$-1$, so it is again strictly negative. Therefore, integrating
\eqref{eq:23} from $q$ to $u>q$ and using $f(q)=0$,
[
e^{a(u-q)}f(u)
\le
\int_q^u
e^{a(v-q)}
\bigl(a(1-v)-1\bigr),\dd v
<0.
]
Thus
[
g_s(u)-u=f(u)<0,
\qquad q<u\le1.
\tag{24}
]
Together with \eqref{eq:10}, \eqref{eq:1}, and \eqref{eq:19}, this proves
\eqref{eq:ATsign}.

It remains to establish the uniform linear estimate
\eqref{eq:linearATsign}.

By compactness of $\cK$ and the strict AT condition, define
[
\alpha_{\cK}
\coloneqq
\sup_{(\beta,h)\in\cK}\alpha(\beta,h)
<1,
\qquad
\delta_{\cK}
\coloneqq
1-\alpha_{\cK}>0.
\tag{25}
]
For $u<q$, \eqref{eq:leftstrict-detailed} immediately gives
\begin{align*}
g_s(u)-u
&\ge
(1-s\alpha)(q-u)\
&\ge
\delta_{\cK}(q-u).
\tag{26}
\end{align*}
Thus the desired linear estimate already holds uniformly on the entire
lower interval.

For completeness, we give an explicit uniform argument on the upper
side. Set
[
\psi(x)\coloneqq\sech^4(x).
]
By \eqref{eq:gs-upper-derivative-detailed},
[
g_s'(u)=a,\E\psi(X_{s,u}).
]
Applying \eqref{eq:14} once more,
[
g_s''(u)
========

a^2\E[(\mathcal L\psi)(X_{s,u})],
\qquad u>q.
\tag{27}
]
A direct computation gives
[
\psi'(x)
========

-4\sech^4(x)\tanh(x),
]
and
[
\frac12\psi''(x)
================

8\sech^4(x)\tanh^2(x)
-2\sech^6(x).
]
Therefore
\begin{align*}
(\mathcal L\psi)(x)
&=
\frac12\psi''(x)+\tanh(x)\psi'(x)\
&=
4\sech^4(x)\tanh^2(x)
-2\sech^6(x).
\tag{28}
\end{align*}
In particular, there is a universal constant $C_0<\infty$, for example
$C_0=6$, such that
[
|\mathcal L\psi|_\infty\le C_0.
\tag{29}
]

Let
[
B_{\cK}\coloneqq
\sup{|\beta|:(\beta,h)\in\cK}<\infty
]
and set
[
C_{\cK}\coloneqq C_0(1+B_{\cK}^4).
]
Since $a=s\beta^2$ and $0\le s\le1$, \eqref{eq:27} gives the uniform
bound
[
|g_s''(u)|
\le
C_{\cK},
\qquad q<u\le1.
\tag{30}
]
Consequently, for $u\ge q$,
[
|g_s'(u)-g_s'(q)|
\le
C_{\cK}(u-q).
\tag{31}
]
By \eqref{eq:gsprimeq-detailed} and \eqref{eq:25},
[
g_s'(q)
=======

s\alpha
\le
\alpha_{\cK}
============

1-\delta_{\cK}.
\tag{32}
]
Choose
[
\varepsilon_{\cK}
\coloneqq
\min\left{
1,\frac{\delta_{\cK}}{2C_{\cK}}
\right}.
\tag{33}
]
Then, whenever
[
q\le u\le1,
\qquad
u-q\le\varepsilon_{\cK},
]
\eqref{eq:31} and \eqref{eq:32} imply
[
g_s'(u)
\le
1-\frac{\delta_{\cK}}2.
\tag{34}
]
Therefore
\begin{align*}
u-g_s(u)
&=
\int_q^u(1-g_s'(v)),\dd v\
&\ge
\frac{\delta_{\cK}}2(u-q).
\tag{35}
\end{align*}

Combining \eqref{eq:26} and \eqref{eq:35}, and setting
[
c_{\cK}\coloneqq\frac{\delta_{\cK}}2>0,
]
we obtain
[
|g_s(u)-u|
\ge
c_{\cK}|u-q|
]
whenever
[
u\in[0,1],
\qquad
|u-q|\le\varepsilon_{\cK}.
]
This proves \eqref{eq:linearATsign}.
\end{proof}

-/

import Lemmas.ATDefs
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.MeasureTheory.Group.IntegralConvolution
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import SpinGlass.Mathlib.Probability.Distributions.GaussianIntegrationByParts

open MeasureTheory ProbabilityTheory Real
open scoped MeasureTheory NNReal

set_option autoImplicit false

namespace SpinGlass.AT

private noncomputable def propertySech (x : ℝ) : ℝ := (Real.cosh x)⁻¹

private lemma propertySech_pos (x : ℝ) : 0 < propertySech x := by
  exact inv_pos.mpr (Real.cosh_pos x)

private lemma propertySech_le_one (x : ℝ) : propertySech x ≤ 1 := by
  exact inv_le_one_of_one_le₀ (Real.one_le_cosh x)

private lemma property_abs_sech_le_one (x : ℝ) : |propertySech x| ≤ 1 := by
  rw [abs_of_pos (propertySech_pos x)]
  exact propertySech_le_one x

private lemma property_abs_tanh_le_one (x : ℝ) : |Real.tanh x| ≤ 1 :=
  (Real.abs_tanh_lt_one x).le

private lemma property_tanh_sq_add_sech_sq (x : ℝ) :
    Real.tanh x ^ 2 + propertySech x ^ 2 = 1 := by
  unfold propertySech
  rw [Real.tanh_eq_sinh_div_cosh]
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  simp only [div_pow, inv_pow]
  field_simp [hc]
  nlinarith [Real.cosh_sq_sub_sinh_sq x]

private lemma property_continuous_tanh : Continuous (fun x : ℝ => Real.tanh x) := by
  simp_rw [Real.tanh_eq_sinh_div_cosh]
  exact Real.continuous_sinh.div₀ Real.continuous_cosh
    (fun x => (Real.cosh_pos x).ne')

private lemma property_continuous_sech : Continuous propertySech := by
  unfold propertySech
  exact Real.continuous_cosh.inv₀ fun x => (Real.cosh_pos x).ne'

private lemma property_tanh_hasDerivAt (x : ℝ) :
    HasDerivAt (fun y : ℝ => Real.tanh y) (propertySech x ^ 2) x := by
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  rw [show (fun y : ℝ => Real.tanh y) = fun y => Real.sinh y / Real.cosh y by
    funext y
    exact Real.tanh_eq_sinh_div_cosh y]
  apply ((Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc).congr_deriv
  unfold propertySech
  simp only [Pi.inv_apply, inv_pow]
  field_simp [hc]
  nlinarith [Real.cosh_sq_sub_sinh_sq x]

private lemma property_integrable_tanh_affine (a b : ℝ) :
    Integrable (fun z : ℝ => Real.tanh (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · exact (property_continuous_tanh.comp (by fun_prop)).aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_abs_tanh_le_one (a + b * z)

private lemma property_integrable_sech_sq_affine (a b : ℝ) :
    Integrable (fun z : ℝ => propertySech (a + b * z) ^ 2)
      (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · exact ((property_continuous_sech.comp (by fun_prop)).pow 2).aestronglyMeasurable
  · filter_upwards [] with z
    rw [Real.norm_eq_abs, abs_pow]
    exact pow_le_one₀ (abs_nonneg _) (property_abs_sech_le_one _)

private lemma property_standardGaussianExpectation_const (c : ℝ) :
    standardGaussianExpectation (fun _ : ℝ => c) = c := by
  simp [standardGaussianExpectation]

/-- At the replica-symmetric breakpoint, the scalar order parameter equals
the fixed point. -/
theorem scalarOrderParameterCorrect_at_rsQ {β h : ℝ}
    (hβ : 0 < β) (hh : 0 < h) (s : ℝ) :
    scalarOrderParameterCorrect β h s (rsQ β h) = rsQ β h := by
  have hq0 : 0 ≤ rsQ β h := (rsQ_mem_Icc β h).1
  have hsqrt : Real.sqrt (β ^ 2 * ((1 - s) * rsQ β h + s * rsQ β h)) =
      β * Real.sqrt (rsQ β h) := by
    rw [show (1 - s) * rsQ β h + s * rsQ β h = rsQ β h by ring]
    rw [Real.sqrt_mul (sq_nonneg β), Real.sqrt_sq_eq_abs, abs_of_pos hβ]
  have hinner (x : ℝ) : scalarPsiX β (rsQ β h) s (rsQ β h) x =
      Real.tanh x := by
    unfold scalarPsiX
    simp [standardGaussianExpectation]
  unfold scalarOrderParameterCorrect scalarOrderParameter localFieldExpectation
  simp only [if_pos le_rfl, hinner, heatSemigroup, hsqrt]
  exact (rsQ_fixedPoint β h).symm

private noncomputable def propertySech3 (x : ℝ) : ℝ := propertySech x ^ 3

private noncomputable def propertySechSecond (x : ℝ) : ℝ :=
  propertySech x * (Real.tanh x ^ 2 - propertySech x ^ 2)

private lemma property_sech_hasDerivAt (x : ℝ) :
    HasDerivAt propertySech (-propertySech x * Real.tanh x) x := by
  unfold propertySech
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  apply ((Real.hasDerivAt_cosh x).inv hc).congr_deriv
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp [hc]

private lemma property_sechDeriv_hasDerivAt (x : ℝ) :
    HasDerivAt (fun y => -propertySech y * Real.tanh y)
      (propertySechSecond x) x := by
  have hs := property_sech_hasDerivAt x
  have ht := property_tanh_hasDerivAt x
  apply (hs.neg.mul ht).congr_deriv
  unfold propertySechSecond
  simp only [Pi.neg_apply]
  ring

private lemma property_sech3_hasDerivAt (x : ℝ) :
    HasDerivAt propertySech3 (-3 * propertySech3 x * Real.tanh x) x := by
  unfold propertySech3
  apply ((property_sech_hasDerivAt x).pow 3).congr_deriv
  norm_num
  ring

private noncomputable def propertySech3Second (x : ℝ) : ℝ :=
  9 * propertySech3 x * Real.tanh x ^ 2 -
    3 * propertySech3 x * propertySech x ^ 2

private lemma property_sech3Deriv_hasDerivAt (x : ℝ) :
    HasDerivAt (fun y => -3 * propertySech3 y * Real.tanh y)
      (propertySech3Second x) x := by
  have hs := (property_sech3_hasDerivAt x).const_mul (-3)
  have ht := property_tanh_hasDerivAt x
  apply (hs.mul ht).congr_deriv
  unfold propertySech3Second
  ring

private lemma property_abs_sech3_le_one (x : ℝ) : |propertySech3 x| ≤ 1 := by
  unfold propertySech3
  rw [abs_pow]
  exact pow_le_one₀ (abs_nonneg _) (property_abs_sech_le_one x)

private lemma property_sechDeriv_abs_le_one (x : ℝ) :
    |-propertySech x * Real.tanh x| ≤ 1 := by
  rw [abs_mul, abs_neg]
  calc
    |propertySech x| * |Real.tanh x| ≤ 1 * 1 := by
      gcongr
      · exact property_abs_sech_le_one x
      · exact property_abs_tanh_le_one x
    _ = 1 := by norm_num

private lemma property_sechSecond_abs_le_two (x : ℝ) :
    |propertySechSecond x| ≤ 2 := by
  unfold propertySechSecond
  calc
    |propertySech x * (Real.tanh x ^ 2 - propertySech x ^ 2)| ≤
        |propertySech x| * (|Real.tanh x| ^ 2 + |propertySech x| ^ 2) := by
          rw [abs_mul]
          gcongr
          calc
            |Real.tanh x ^ 2 - propertySech x ^ 2| ≤
                |Real.tanh x ^ 2| + |propertySech x ^ 2| := abs_sub _ _
            _ = |Real.tanh x| ^ 2 + |propertySech x| ^ 2 := by rw [abs_pow, abs_pow]
    _ ≤ 1 * (1 ^ 2 + 1 ^ 2) := by
      gcongr
      · exact property_abs_sech_le_one x
      · exact property_abs_tanh_le_one x
      · exact property_abs_sech_le_one x
    _ = 2 := by norm_num

private lemma property_sech3Deriv_abs_le_three (x : ℝ) :
    |-3 * propertySech3 x * Real.tanh x| ≤ 3 := by
  rw [abs_mul, abs_mul, abs_neg]
  norm_num
  calc
    3 * |propertySech3 x| * |Real.tanh x| ≤ 3 * 1 * 1 := by
      gcongr
      · exact property_abs_sech3_le_one x
      · exact property_abs_tanh_le_one x
    _ = 3 := by norm_num

private lemma property_sech3Second_abs_le_twelve (x : ℝ) :
    |propertySech3Second x| ≤ 12 := by
  unfold propertySech3Second
  calc
    |9 * propertySech3 x * Real.tanh x ^ 2 -
        3 * propertySech3 x * propertySech x ^ 2| ≤
        |9 * propertySech3 x * Real.tanh x ^ 2| +
          |3 * propertySech3 x * propertySech x ^ 2| := abs_sub _ _
    _ ≤ 9 * 1 * 1 ^ 2 + 3 * 1 * 1 ^ 2 := by
      simp only [abs_mul, abs_pow]
      gcongr <;> norm_num
      · exact property_abs_sech3_le_one x
      · exact property_abs_tanh_le_one x
      · exact property_abs_sech3_le_one x
      · exact property_abs_sech_le_one x
    _ = 12 := by norm_num

private lemma property_integrable_sech_affine (a b : ℝ) :
    Integrable (fun z : ℝ => propertySech (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · exact (property_continuous_sech.comp (by fun_prop)).aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_abs_sech_le_one (a + b * z)

private lemma property_integrable_sechDeriv_affine (a b : ℝ) :
    Integrable (fun z : ℝ => -propertySech (a + b * z) *
      Real.tanh (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · have harg : Continuous (fun z : ℝ => a + b * z) :=
      continuous_const.add (continuous_const.mul continuous_id)
    exact (((property_continuous_sech.comp harg).neg.mul
      (property_continuous_tanh.comp harg))).aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_sechDeriv_abs_le_one (a + b * z)

private lemma property_integrable_sechSecond_affine (a b : ℝ) :
    Integrable (fun z : ℝ => propertySechSecond (a + b * z))
      (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 2)
  · apply Continuous.aestronglyMeasurable
    unfold propertySechSecond
    exact (property_continuous_sech.comp (by fun_prop)).mul
      (((property_continuous_tanh.comp (by fun_prop)).pow 2).sub
        ((property_continuous_sech.comp (by fun_prop)).pow 2))
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_sechSecond_abs_le_two (a + b * z)

private lemma property_integrable_sech3_affine (a b : ℝ) :
    Integrable (fun z : ℝ => propertySech3 (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 1)
  · exact ((property_continuous_sech.comp (by fun_prop)).pow 3).aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_abs_sech3_le_one (a + b * z)

private lemma property_integrable_sech3Deriv_affine (a b : ℝ) :
    Integrable (fun z : ℝ => -3 * propertySech3 (a + b * z) *
      Real.tanh (a + b * z)) (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 3)
  · have harg : Continuous (fun z : ℝ => a + b * z) :=
      continuous_const.add (continuous_const.mul continuous_id)
    exact (((continuous_const.mul
      ((property_continuous_sech.comp harg).pow 3)).mul
      (property_continuous_tanh.comp harg))).aestronglyMeasurable
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_sech3Deriv_abs_le_three (a + b * z)

private lemma property_integrable_sech3Second_affine (a b : ℝ) :
    Integrable (fun z : ℝ => propertySech3Second (a + b * z))
      (gaussianReal 0 1) := by
  apply Integrable.of_bound (C := 12)
  · apply Continuous.aestronglyMeasurable
    unfold propertySech3Second propertySech3
    have harg : Continuous (fun z : ℝ => a + b * z) :=
      continuous_const.add (continuous_const.mul continuous_id)
    exact (((continuous_const.mul ((property_continuous_sech.comp harg).pow 3)).mul
      ((property_continuous_tanh.comp harg).pow 2)).sub
      ((continuous_const.mul ((property_continuous_sech.comp harg).pow 3)).mul
        ((property_continuous_sech.comp harg).pow 2)))
  · filter_upwards [] with z
    simpa [Real.norm_eq_abs] using property_sech3Second_abs_le_twelve (a + b * z)

private noncomputable def propertySmoothSech (r x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z => propertySech (x + Real.sqrt r * z))

private noncomputable def propertySmoothSech3 (r x : ℝ) : ℝ :=
  standardGaussianExpectation (fun z => propertySech3 (x + Real.sqrt r * z))

private lemma property_sechDeriv_comp_deriv (a b z : ℝ) :
    deriv (fun y => -propertySech (a + b * y) * Real.tanh (a + b * y)) z =
      b * propertySechSecond (a + b * z) := by
  have harg : HasDerivAt (fun y : ℝ => a + b * y) b z := by
    simpa only [id_eq, mul_one] using
      ((hasDerivAt_id z).const_mul b).const_add a
  simpa [Function.comp_def, mul_comm] using
    ((property_sechDeriv_hasDerivAt (a + b * z)).comp z harg).deriv

private lemma property_sech3Deriv_comp_deriv (a b z : ℝ) :
    deriv (fun y => -3 * propertySech3 (a + b * y) *
      Real.tanh (a + b * y)) z = b * propertySech3Second (a + b * z) := by
  have harg : HasDerivAt (fun y : ℝ => a + b * y) b z := by
    simpa only [id_eq, mul_one] using
      ((hasDerivAt_id z).const_mul b).const_add a
  simpa [Function.comp_def, mul_comm] using
    ((property_sech3Deriv_hasDerivAt (a + b * z)).comp z harg).deriv

private lemma property_contDiff_sech : ContDiff ℝ ⊤ propertySech := by
  unfold propertySech
  exact Real.contDiff_cosh.inv fun x => (Real.cosh_pos x).ne'

private lemma property_contDiff_tanh : ContDiff ℝ ⊤ (fun x : ℝ => Real.tanh x) := by
  simp_rw [Real.tanh_eq_sinh_div_cosh]
  exact Real.contDiff_sinh.div Real.contDiff_cosh
    (fun x => (Real.cosh_pos x).ne')

private lemma property_sechDeriv_comp_moderate (a b : ℝ) :
    HasModerateGrowth
      (fun z => -propertySech (a + b * z) * Real.tanh (a + b * z)) := by
  refine ⟨3 * (1 + |b|), 0, by positivity, ?_, ?_⟩
  · intro z
    simpa only [pow_zero, mul_one] using
      (show |-propertySech (a + b * z) * Real.tanh (a + b * z)| ≤
          3 * (1 + |b|) by
        have hz := property_sechDeriv_abs_le_one (a + b * z)
        have hb := abs_nonneg b
        nlinarith)
  · intro z
    rw [property_sechDeriv_comp_deriv]
    simpa only [pow_zero, mul_one] using
      (show |b * propertySechSecond (a + b * z)| ≤ 3 * (1 + |b|) by
        rw [abs_mul]
        have hs := property_sechSecond_abs_le_two (a + b * z)
        have hb := abs_nonneg b
        nlinarith [mul_le_mul_of_nonneg_left hs hb])

private lemma property_sech3Deriv_comp_moderate (a b : ℝ) :
    HasModerateGrowth
      (fun z => -3 * propertySech3 (a + b * z) * Real.tanh (a + b * z)) := by
  refine ⟨16 * (1 + |b|), 0, by positivity, ?_, ?_⟩
  · intro z
    simpa only [pow_zero, mul_one] using
      (show |-3 * propertySech3 (a + b * z) * Real.tanh (a + b * z)| ≤
          16 * (1 + |b|) by
        have hz := property_sech3Deriv_abs_le_three (a + b * z)
        have hb := abs_nonneg b
        nlinarith)
  · intro z
    rw [property_sech3Deriv_comp_deriv]
    simpa only [pow_zero, mul_one] using
      (show |b * propertySech3Second (a + b * z)| ≤ 16 * (1 + |b|) by
        rw [abs_mul]
        have hs := property_sech3Second_abs_le_twelve (a + b * z)
        have hb := abs_nonneg b
        nlinarith [mul_le_mul_of_nonneg_left hs hb])

private lemma property_smoothSech_hasDerivAt_r_raw {r x : ℝ} (hr : 0 < r) :
    HasDerivAt (fun t => propertySmoothSech t x)
      (standardGaussianExpectation (fun z =>
        (-propertySech (x + Real.sqrt r * z) *
          Real.tanh (x + Real.sqrt r * z)) *
            (1 / (2 * Real.sqrt r) * z))) r := by
  unfold propertySmoothSech standardGaussianExpectation
  let F : ℝ → ℝ → ℝ := fun t z => propertySech (x + Real.sqrt t * z)
  let F' : ℝ → ℝ → ℝ := fun t z =>
    (-propertySech (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)) *
      (1 / (2 * Real.sqrt t) * z)
  let c : ℝ := Real.sqrt (r / 2)
  have hhalf : 0 < r / 2 := by linarith
  have hc : 0 < c := Real.sqrt_pos.2 hhalf
  have hboundInt : Integrable (fun z : ℝ => c⁻¹ * |z|) (gaussianReal 0 1) := by
    have hz : Integrable (fun z : ℝ => |z|) (gaussianReal 0 1) := by
      simpa using integrable_abs_pow_gaussianReal_centered (1 : ℝ≥0) 1
    exact hz.const_mul c⁻¹
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := gaussianReal 0 1) (F := F) (F' := F') (x₀ := r)
    (s := Set.Ioi (r / 2)) (bound := fun z => c⁻¹ * |z|)
    (Ioi_mem_nhds (by linarith))
    (Filter.Eventually.of_forall fun t =>
      (property_continuous_sech.comp
        (continuous_const.add (continuous_const.mul continuous_id))).aestronglyMeasurable)
    (by simpa [F] using property_integrable_sech_affine x (Real.sqrt r))
    (by
      apply Continuous.aestronglyMeasurable
      dsimp [F']
      have harg : Continuous (fun z : ℝ => x + Real.sqrt r * z) :=
        continuous_const.add (continuous_const.mul continuous_id)
      exact (((property_continuous_sech.comp harg).neg.mul
        (property_continuous_tanh.comp harg)).mul
          ((continuous_const : Continuous (fun _ : ℝ => 1 / (2 * Real.sqrt r))).mul
            continuous_id)))
    (by
      filter_upwards [] with z
      intro t ht
      have htpos : 0 < t := lt_trans hhalf ht
      have hroot : 0 < Real.sqrt t := Real.sqrt_pos.2 htpos
      have hrootle : c ≤ Real.sqrt t := Real.sqrt_le_sqrt ht.le
      have hinv : (Real.sqrt t)⁻¹ ≤ c⁻¹ := (inv_le_inv₀ hroot hc).2 hrootle
      have hcoef : |1 / (2 * Real.sqrt t)| ≤ c⁻¹ := by
        rw [abs_of_pos (by positivity : 0 < 1 / (2 * Real.sqrt t))]
        calc
          1 / (2 * Real.sqrt t) ≤ (Real.sqrt t)⁻¹ := by
            rw [one_div]
            exact (inv_le_inv₀ (by positivity) hroot).2 (by nlinarith)
          _ ≤ c⁻¹ := hinv
      dsimp [F']
      calc
        |(-propertySech (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)) *
            (1 / (2 * Real.sqrt t) * z)| =
            |-propertySech (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)| *
              |1 / (2 * Real.sqrt t)| * |z| := by simp only [abs_mul]; ring
        _ ≤ c⁻¹ * |z| := by
          have hp :
              |-propertySech (x + Real.sqrt t * z) * Real.tanh (x + Real.sqrt t * z)| *
                  |1 / (2 * Real.sqrt t)| ≤ c⁻¹ := by
            calc
              _ ≤ 1 * c⁻¹ := mul_le_mul (property_sechDeriv_abs_le_one _) hcoef
                (abs_nonneg _) (by norm_num)
              _ = c⁻¹ := one_mul _
          exact mul_le_mul_of_nonneg_right hp (abs_nonneg z))
    hboundInt
    (by
      filter_upwards [] with z
      intro t ht
      have htpos : 0 < t := lt_trans hhalf ht
      have hsqrt := Real.hasDerivAt_sqrt htpos.ne'
      have harg : HasDerivAt (fun t => x + Real.sqrt t * z)
          (1 / (2 * Real.sqrt t) * z) t := (hsqrt.mul_const z).const_add x
      simpa [F, F', Function.comp_def] using
        (property_sech_hasDerivAt _).comp t harg)
  simpa [F, F'] using h.2

private lemma property_smoothSech_hasDerivAt_r {r x : ℝ} (hr : 0 < r) :
    HasDerivAt (fun t => propertySmoothSech t x)
      ((1 / 2) * standardGaussianExpectation (fun z =>
        propertySechSecond (x + Real.sqrt r * z))) r := by
  apply (property_smoothSech_hasDerivAt_r_raw (x := x) hr).congr_deriv
  unfold standardGaussianExpectation
  let F : ℝ → ℝ := fun z =>
    -propertySech (x + Real.sqrt r * z) * Real.tanh (x + Real.sqrt r * z)
  have hcont : ContDiff ℝ 1 F := by
    exact ((property_contDiff_sech.neg.mul property_contDiff_tanh).of_le (by norm_num)).comp
      (by fun_prop)
  have hibp := gaussianReal_integration_by_parts (v := (1 : ℝ≥0)) one_ne_zero
    hcont (property_sechDeriv_comp_moderate x (Real.sqrt r))
  have hderiv : deriv F = fun z => Real.sqrt r *
      propertySechSecond (x + Real.sqrt r * z) := by
    funext z
    exact property_sechDeriv_comp_deriv x (Real.sqrt r) z
  rw [hderiv] at hibp
  simp only [NNReal.coe_one, one_mul] at hibp
  have hsqrt : Real.sqrt r ≠ 0 := (Real.sqrt_pos.2 hr).ne'
  calc
    ∫ z, F z * (1 / (2 * Real.sqrt r) * z) ∂gaussianReal 0 1 =
        (1 / (2 * Real.sqrt r)) * ∫ z, z * F z ∂gaussianReal 0 1 := by
          rw [← integral_const_mul]
          apply integral_congr_ae
          filter_upwards [] with z
          ring
    _ = (1 / (2 * Real.sqrt r)) *
        ∫ z, Real.sqrt r * propertySechSecond (x + Real.sqrt r * z)
          ∂gaussianReal 0 1 := by rw [hibp]
    _ = (1 / 2) * ∫ z, propertySechSecond (x + Real.sqrt r * z)
          ∂gaussianReal 0 1 := by
          rw [integral_const_mul]
          field_simp [hsqrt]

end SpinGlass.AT
