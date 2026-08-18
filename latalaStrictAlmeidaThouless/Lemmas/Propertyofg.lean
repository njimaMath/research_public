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
