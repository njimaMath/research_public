/-
Fix $(\beta,h,s)\in\cK\times[0,1]$.
For $t\ge0$, let $\mathsf H_t$ be the heat semigroup, namely
\[
 (\mathsf H_t\varphi)(x)
 \coloneqq
 \E\varphi(x+\sqrt t\,Z),
 \qquad Z\sim N(0,1).
\]
For instance, if $\varphi\in C_b^2(\R)$ and $t>0$, then
\[
 \begin{aligned}
 \partial_t(\mathsf H_t\varphi)(x)
 &=
 \partial_t\int_{\R}
 \varphi(x+\sqrt t\,z)\phi(z)\,dz\\
 &=
 \frac{1}{2\sqrt t}
 \int_{\R}
 z\,\varphi'(x+\sqrt t\,z)\phi(z)\,dz.
 \end{aligned}
\]
Since $\phi'(z)=-z\phi(z)$, integration by parts in $z$ gives
\[
 \begin{aligned}
 \int_{\R}
 z\,\varphi'(x+\sqrt t\,z)\phi(z)\,dz
 &=
 -\int_{\R}
 \varphi'(x+\sqrt t\,z)\phi'(z)\,dz\\
 &=
 \sqrt t
 \int_{\R}
 \varphi''(x+\sqrt t\,z)\phi(z)\,dz.
 \end{aligned}
\]
Consequently,
\begin{equation}\label{eq:genH}
 \partial_t(\mathsf H_t\varphi)(x)
 =
 \frac12\mathsf H_t(\partial^2\varphi)(x).
\end{equation}

For the upper interval, introduce the tilted heat semigroup
\begin{equation}
 (\mathsf T_t\varphi)(x)
 \coloneqq
 e^{-t/2}
 \frac{\bigl(\mathsf H_t(\varphi\cosh)\bigr)(x)}
 {\cosh x}.
 \label{eq:tiltedsemigroup}
\end{equation}
We record explicitly its generator. Put
\[
 g_t(x)\coloneqq(\mathsf T_t\varphi)(x).
\]
Then
\[
 \mathsf H_t(\varphi\cosh)(x)
 =
 e^{t/2}\cosh(x)g_t(x).
\]
Using \eqref{eq:genH},
\[
 \begin{aligned}
 \partial_tg_t(x)
 &=
 -\frac12g_t(x)
 +
 \frac{e^{-t/2}}{2\cosh x}
 \partial_{xx}
 \mathsf H_t(\varphi\cosh)(x)\\
 &=
 -\frac12g_t(x)
 +
 \frac{1}{2\cosh x}
 \partial_{xx}\bigl(\cosh(x)g_t(x)\bigr).
 \end{aligned}
\]
Since
\[
 \begin{aligned}
 \partial_{xx}\bigl(\cosh(x)g_t(x)\bigr)
 &=
 \partial_x\bigl(
   \sinh(x)g_t(x)+\cosh(x)g_t'(x)
 \bigr)\\
 &=
 \cosh(x)g_t(x)
 +2\sinh(x)g_t'(x)
 +\cosh(x)g_t''(x),
 \end{aligned}
\]
we obtain
\[
 \partial_tg_t(x)
 =
 \frac12g_t''(x)+\tanh(x)g_t'(x).
\]
Thus the generator of $\mathsf T_t$ is
\[
 \frac12\partial_{xx}+\tanh(x)\partial_x.
\]
Equivalently,
\begin{equation}\label{eq:partialT_t}
 \partial_t(\mathsf T_t\varphi)(x)
 =
 \frac{1}{2\cosh^2x}
 \partial_x\left(
   \cosh^2x\,
   \partial_x(\mathsf T_t\varphi)(x)
 \right),
\end{equation}
because
\[
 \begin{aligned}
 \frac{1}{2\cosh^2x}
 \partial_x\left(\cosh^2x\,g_t'(x)\right)
 &=
 \frac{1}{2\cosh^2x}
 \left(
  2\cosh x\sinh x\,g_t'(x)
  +\cosh^2x\,g_t''(x)
 \right)\\
 &=
 \tanh x\,g_t'(x)+\frac12g_t''(x).
 \end{aligned}
\]

For each $u\in[0,1]$, let $X_{s,u}$ denote a random variable whose
law is specified by
\begin{equation}
 \E\varphi(X_{s,u})
 \coloneqq
 \begin{cases}
  \bigl(\mathsf H_{\beta^2(1-s)q+s\beta^2u}\varphi\bigr)(h),
      &0\le u\le q,\\[1mm]
  \bigl(\mathsf H_{\beta^2q}
      (\mathsf T_{s\beta^2(u-q)}\varphi)\bigr)(h),
      &q\le u\le1.
 \end{cases}
 \label{eq:localfieldlaw}
\end{equation}
At $u=q$, both expressions reduce to
\[
 \bigl(\mathsf H_{\beta^2q}\varphi\bigr)(h)
 =
 \E\varphi(h+\beta\sqrt q\,Z),
\]
and hence
\begin{equation}\label{eq:Xsq}
 X_{s,q}\stackrel{\mathrm d}=h+\beta\sqrt q\,Z.
\end{equation}

\begin{lemma}
\label{lem:uppercomparison}
If $s\beta^2(1-q)>1$, then
\begin{equation}
 \beta^2\E\sech^4(X_{s,u})\le\alpha,
 \qquad q\le u\le1.
 \label{eq:uppercomparison}
\end{equation}
\end{lemma}

\begin{proof}
Fix $u\in[q,1]$, and set
\[
 \sigma^2\coloneqq\beta^2q,
 \qquad
 \tau\coloneqq s\beta^2(u-q),
 \qquad
 X\coloneqq h+\sigma Z.
\]
Then $X\sim N(h,\sigma^2)$.

We first dispose of the case $q=0$. In that case the defining equation
for $q$ gives
\[
 1
 =
 \sech^2(h),
\]
and therefore $h=0$. Moreover,
\[
 \mathsf H_t(\cosh)(x)
 =
 \E\cosh(x+\sqrt t\,Z).
\]
Using
\[
 \cosh(x+y)
 =
 \cosh x\cosh y+\sinh x\sinh y,
\]
together with
\[
 \E\sinh(\sqrt t\,Z)=0,
 \qquad
 \E\cosh(\sqrt t\,Z)
 =
 \frac12\left(\E e^{\sqrt t Z}+\E e^{-\sqrt t Z}\right)
 =
 e^{t/2},
\]
we obtain
\[
 \mathsf H_t(\cosh)(x)=e^{t/2}\cosh x.
\]
Hence
\[
 \mathsf T_t1=1.
\]
Since $\mathsf T_t$ is positivity preserving and $0\le\sech^4\le1$,
\[
 0\le\mathsf T_t(\sech^4)\le1.
\]
Thus the desired comparison is immediate when $q=0$.

We may therefore assume $q>0$, so that $\sigma>0$.

By \eqref{eq:localfieldlaw},
\[
 \begin{aligned}
 \E\sech^4(X_{s,u})
 &=
 \bigl(
 \mathsf H_{\beta^2q}
   (\mathsf T_\tau\sech^4)
 \bigr)(h)\\
 &=
 \E\bigl[
   (\mathsf T_\tau\sech^4)(h+\sigma Z)
 \bigr]\\
 &=
 \E\bigl[
   (\mathsf T_\tau\sech^4)(X)
 \bigr].
 \end{aligned}
\]
Thus it is enough to prove
\begin{equation}\label{eq:upper-M-comparison}
 \E\bigl[
   (\mathsf T_\tau\sech^4)(X)
 \bigr]
 \le
 \E\sech^4(X).
\end{equation}

We next prove
\begin{equation}
 |h|<\sigma^2.
 \label{eq:h-less-sigma}
\end{equation}
Let
\[
 F(a)
 \coloneqq
 \E\sech^2(a+\sigma Z).
\]
The function $F$ is even. We also verify directly that it is
nonincreasing on $[0,\infty)$. Let
\[
 \rho_\sigma(x)
 \coloneqq
 \frac{1}{\sqrt{2\pi\sigma^2}}
 e^{-x^2/(2\sigma^2)}
\]
be the density of $N(0,\sigma^2)$ and put $f_2(x)=\sech^2x$. Then
\[
 F(a)
 =
 \int_{\R}f_2(y)\rho_\sigma(a-y)\,dy.
\]
Differentiating the convolution in the $f_2$ variable gives
\[
 F'(a)
 =
 \int_{\R}f_2'(y)\rho_\sigma(a-y)\,dy.
\]
Since $f_2$ is even, $f_2'$ is odd. Pairing $y$ and $-y$ therefore yields,
for $a\ge0$,
\[
 F'(a)
 =
 \int_0^\infty
 f_2'(y)
 \left[
   \rho_\sigma(a-y)-\rho_\sigma(a+y)
 \right]\,dy.
\]
For $a,y\ge0$,
\[
 |a-y|\le a+y,
\]
and hence, since $\rho_\sigma$ is decreasing as a function of $|x|$,
\[
 \rho_\sigma(a-y)\ge\rho_\sigma(a+y).
\]
On the other hand,
\[
 f_2'(y)
 =
 -2\sech^2(y)\tanh(y)\le0,
 \qquad y\ge0.
\]
Consequently,
\[
 F'(a)\le0,
 \qquad a\ge0.
\]

Suppose now, toward a contradiction, that $|h|\ge\sigma^2$. Since $F$
is even and nonincreasing on $[0,\infty)$,
\[
 \begin{aligned}
 1-q
 &=
 \E\sech^2(h+\sigma Z)\\
 &=
 F(|h|)\\
 &\le
 F(\sigma^2)\\
 &=
 \E\sech^2(\sigma^2+\sigma Z).
 \end{aligned}
\]
We estimate the final expectation.

Let $\varepsilon$ be a symmetric Rademacher random variable independent
of $Z$, and define
\[
 Y\coloneqq\sigma^2\varepsilon+\sigma Z.
\]
Conditional on $\varepsilon=\pm1$, the variable $Y$ has density
\[
 y\longmapsto
 \frac{1}{\sqrt{2\pi\sigma^2}}
 \exp\left(
  -\frac{(y\mp\sigma^2)^2}{2\sigma^2}
 \right).
\]
Therefore Bayes' formula gives
\[
 \frac{\P(\varepsilon=1\mid Y=y)}
 {\P(\varepsilon=-1\mid Y=y)}
 =
 \exp\left(
 -\frac{(y-\sigma^2)^2-(y+\sigma^2)^2}
 {2\sigma^2}
 \right).
\]
Since
\[
 (y-\sigma^2)^2-(y+\sigma^2)^2
 =
 -4y\sigma^2,
\]
the ratio is $e^{2y}$. Hence
\[
 \P(\varepsilon=1\mid Y=y)
 =
 \frac{e^{2y}}{1+e^{2y}},
 \qquad
 \P(\varepsilon=-1\mid Y=y)
 =
 \frac{1}{1+e^{2y}},
\]
and therefore
\[
 \E[\varepsilon\mid Y=y]
 =
 \frac{e^{2y}-1}{e^{2y}+1}
 =
 \tanh y.
\]
It follows that
\[
 \Var(\varepsilon\mid Y)
 =
 1-\bigl(\E[\varepsilon\mid Y]\bigr)^2
 =
 1-\tanh^2Y
 =
 \sech^2Y.
\]
Moreover, since $\sech^2$ is even and $Z\stackrel{\mathrm d}=-Z$,
\[
 \begin{aligned}
 \E\sech^2Y
 &=
 \frac12\E\sech^2(\sigma^2+\sigma Z)
 +
 \frac12\E\sech^2(-\sigma^2+\sigma Z)\\
 &=
 \E\sech^2(\sigma^2+\sigma Z).
 \end{aligned}
\]
Thus
\[
 \E\sech^2(\sigma^2+\sigma Z)
 =
 \E\Var(\varepsilon\mid Y).
\]

Since $\E[\varepsilon\mid Y]$ is the $L^2$-optimal estimator of
$\varepsilon$ among all measurable functions of $Y$,
\[
 \E\Var(\varepsilon\mid Y)
 =
 \E\bigl[
   (\varepsilon-\E[\varepsilon\mid Y])^2
 \bigr]
 \le
 \inf_{a\in\R}\E(\varepsilon-aY)^2.
\]
We compute this last infimum explicitly. Since
\[
 Y=\sigma^2\varepsilon+\sigma Z,
\]
independence and centering give
\[
 \E[\varepsilon Y]
 =
 \sigma^2
\]
and
\[
 \begin{aligned}
 \E Y^2
 &=
 \E(\sigma^2\varepsilon+\sigma Z)^2\\
 &=
 \sigma^4\E\varepsilon^2
 +2\sigma^3\E[\varepsilon Z]
 +\sigma^2\E Z^2\\
 &=
 \sigma^4+\sigma^2\\
 &=
 \sigma^2(1+\sigma^2).
 \end{aligned}
\]
Therefore
\[
 \begin{aligned}
 \E(\varepsilon-aY)^2
 &=
 \E\varepsilon^2
 -2a\E[\varepsilon Y]
 +a^2\E Y^2\\
 &=
 1-2a\sigma^2
 +a^2\sigma^2(1+\sigma^2).
 \end{aligned}
\]
The minimizing coefficient is
\[
 a_\star
 =
 \frac{\sigma^2}{\sigma^2(1+\sigma^2)}
 =
 \frac{1}{1+\sigma^2},
\]
and substitution gives
\[
 \begin{aligned}
 \inf_{a\in\R}\E(\varepsilon-aY)^2
 &=
 1-\frac{2\sigma^2}{1+\sigma^2}
 +\frac{\sigma^2}{1+\sigma^2}\\
 &=
 \frac{1}{1+\sigma^2}.
 \end{aligned}
\]
Consequently,
\[
 \E\sech^2(\sigma^2+\sigma Z)
 \le
 \frac{1}{1+\sigma^2}.
\]
Recalling $\sigma^2=\beta^2q$, we obtain
\[
 1-q
 \le
 \frac{1}{1+\beta^2q},
\]
or equivalently
\[
 (1-q)(1+\beta^2q)\le1.
\]
Expanding the left-hand side gives
\[
 1-q+\beta^2q(1-q)\le1,
\]
hence
\[
 q\bigl(\beta^2(1-q)-1\bigr)\le0.
\]
Since $q>0$,
\[
 \beta^2(1-q)\le1.
\]
As $s\le1$, this implies
\[
 s\beta^2(1-q)\le1,
\]
contradicting the assumption
\[
 s\beta^2(1-q)>1.
\]
This proves \eqref{eq:h-less-sigma}.

We now study the tilted semigroup. Set
\[
 f\coloneqq\sech^4,
 \qquad
 g_r\coloneqq\mathsf T_r f,
 \qquad
 M(r)\coloneqq\E g_r(X),
 \qquad 0\le r\le\tau.
\]
By \eqref{eq:tiltedsemigroup},
\[
 \begin{aligned}
 g_r(x)
 &=
 e^{-r/2}
 \frac{\mathsf H_r(f\cosh)(x)}
 {\cosh x}\\
 &=
 e^{-r/2}
 \frac{\mathsf H_r(\sech^3)(x)}
 {\cosh x}.
 \end{aligned}
\]
Let
\[
 a_r(x)\coloneqq\mathsf H_r(\sech^3)(x).
\]
Since $\sech^3$ is even and nonincreasing on $[0,\infty)$, and the
centered Gaussian density is even and nonincreasing on $[0,\infty)$,
the same convolution argument used above shows that $a_r$ is even and
nonincreasing on $[0,\infty)$. Thus
\[
 a_r'(x)\le0,
 \qquad x>0.
\]
Since
\[
 g_r(x)=e^{-r/2}a_r(x)\sech x,
\]
we can also differentiate directly:
\[
 \begin{aligned}
 g_r'(x)
 &=
 e^{-r/2}
 \left(
  a_r'(x)\sech x
  -
  a_r(x)\sech x\tanh x
 \right).
 \end{aligned}
\]
For $x>0$,
\[
 a_r'(x)\le0,
 \qquad
 a_r(x)\ge0,
 \qquad
 \sech x>0,
 \qquad
 \tanh x>0,
\]
so
\begin{equation}\label{eq:gt-decreasing}
 g_r'(x)\le0,
 \qquad x>0.
\end{equation}
Also $g_r$ is even, and therefore $g_r'$ is odd.

Let
\[
 p(x)
 \coloneqq
 \frac1\sigma
 \phi\left(\frac{x-h}{\sigma}\right)
\]
be the density of $X\sim N(h,\sigma^2)$. From
\eqref{eq:partialT_t},
\[
 \begin{aligned}
 M'(r)
 &=
 \int_{\R}
 \partial_rg_r(x)p(x)\,dx\\
 &=
 \frac12
 \int_{\R}
 \frac{1}{\cosh^2x}
 \partial_x\bigl(\cosh^2x\,g_r'(x)\bigr)
 p(x)\,dx\\
 &=
 \frac12
 \int_{\R}
 \partial_x\bigl(\cosh^2x\,g_r'(x)\bigr)
 \frac{p(x)}{\cosh^2x}\,dx.
 \end{aligned}
\]
The Gaussian decay makes the boundary term vanish, so integration by
parts gives
\[
 M'(r)
 =
 -\frac12
 \int_{\R}
 \cosh^2x\,g_r'(x)
 \partial_x\left(
  \frac{p(x)}{\cosh^2x}
 \right)\,dx.
\]
Now
\[
 p'(x)
 =
 -\frac{x-h}{\sigma^2}p(x),
\]
whereas
\[
 \partial_x(\cosh^{-2}x)
 =
 -2\tanh x\,\cosh^{-2}x.
\]
Therefore
\[
 \begin{aligned}
 \partial_x\left(
   \frac{p(x)}{\cosh^2x}
 \right)
 &=
 \frac{p'(x)}{\cosh^2x}
 -
 \frac{2\tanh x\,p(x)}{\cosh^2x}\\
 &=
 -\left(
   \frac{x-h}{\sigma^2}+2\tanh x
 \right)
 \frac{p(x)}{\cosh^2x}.
 \end{aligned}
\]
Substituting this identity into the previous display yields
\[
 \begin{aligned}
 M'(r)
 &=
 \frac12
 \int_{\R}
 g_r'(x)
 \left(
   \frac{x-h}{\sigma^2}+2\tanh x
 \right)
 p(x)\,dx\\
 &=
 \E\left[
  \left(
    \tanh X+\frac{X-h}{2\sigma^2}
  \right)
  g_r'(X)
 \right].
 \end{aligned}
\]

We now pair the contributions from $x$ and $-x$. Since $g_r'$ is odd,
\[
 M'(r)
 =
 \int_0^\infty g_r'(x)K(x)\,dx,
 \tag{\ref{eq:Mprime-detail}}\label{eq:Mprime-detail}
\]
where
\[
 \begin{aligned}
 K(x)
 &\coloneqq
 \frac1\sigma
 \phi\left(\frac{x-h}{\sigma}\right)
 \left(
   \tanh x+\frac{x-h}{2\sigma^2}
 \right)\\
 &\quad+
 \frac1\sigma
 \phi\left(\frac{-x-h}{\sigma}\right)
 \left(
   \tanh x+\frac{x+h}{2\sigma^2}
 \right).
 \end{aligned}
\]
Indeed, the contribution from the negative half-line is
\[
 \begin{aligned}
 &\int_{-\infty}^0
 g_r'(x)
 \left(
   \tanh x+\frac{x-h}{2\sigma^2}
 \right)p(x)\,dx\\
 &\qquad=
 \int_0^\infty
 g_r'(-x)
 \left(
  \tanh(-x)+\frac{-x-h}{2\sigma^2}
 \right)p(-x)\,dx\\
 &\qquad=
 \int_0^\infty
 g_r'(x)
 \left(
   \tanh x+\frac{x+h}{2\sigma^2}
 \right)p(-x)\,dx.
 \end{aligned}
\]

It remains to determine the sign of $K(x)$. Write
\[
 A
 \coloneqq
 \phi\left(\frac{x-h}{\sigma}\right),
 \qquad
 B
 \coloneqq
 \phi\left(\frac{-x-h}{\sigma}\right).
\]
Then
\[
 \begin{aligned}
 \sigma K(x)
 &=
 A\left(
   \tanh x+\frac{x}{2\sigma^2}
   -\frac{h}{2\sigma^2}
 \right)\\
 &\quad+
 B\left(
   \tanh x+\frac{x}{2\sigma^2}
   +\frac{h}{2\sigma^2}
 \right)\\
 &=
 (A+B)
 \left(
   \tanh x+\frac{x}{2\sigma^2}
 \right)
 +
 \frac{h}{2\sigma^2}(B-A).
 \end{aligned}
\]
Furthermore,
\[
 \begin{aligned}
 \frac BA
 &=
 \exp\left(
  -\frac{(-x-h)^2-(x-h)^2}{2\sigma^2}
 \right)\\
 &=
 \exp\left(-\frac{2hx}{\sigma^2}\right).
 \end{aligned}
\]
Hence
\[
 \begin{aligned}
 \frac{B-A}{A+B}
 &=
 \frac{e^{-2hx/\sigma^2}-1}
 {e^{-2hx/\sigma^2}+1}\\
 &=
 -\tanh\left(\frac{hx}{\sigma^2}\right).
 \end{aligned}
\]
Consequently,
\begin{equation}\label{eq:Kidentity}
 \frac{\sigma K(x)}{A+B}
 =
 \tanh x+\frac{x}{2\sigma^2}
 -
 \frac{h}{2\sigma^2}
 \tanh\left(\frac{hx}{\sigma^2}\right).
\end{equation}
Since
\[
 h\tanh\left(\frac{hx}{\sigma^2}\right)
 =
 |h|\tanh\left(\frac{|h|x}{\sigma^2}\right),
\]
and \eqref{eq:h-less-sigma} gives
\[
 0\le\frac{|h|}{\sigma^2}<1,
\]
we have, for $x>0$,
\[
 0\le
 \frac{|h|x}{\sigma^2}
 <x.
\]
Since $\tanh$ is increasing,
\[
 \tanh\left(\frac{|h|x}{\sigma^2}\right)
 \le
 \tanh x.
\]
Therefore \eqref{eq:Kidentity} implies
\[
 \begin{aligned}
 \frac{\sigma K(x)}{A+B}
 &\ge
 \tanh x+\frac{x}{2\sigma^2}
 -
 \frac{|h|}{2\sigma^2}\tanh x\\
 &=
 \left(
   1-\frac{|h|}{2\sigma^2}
 \right)\tanh x
 +
 \frac{x}{2\sigma^2}.
 \end{aligned}
\]
For $x>0$, both $\tanh x$ and $x$ are positive, while
\[
 1-\frac{|h|}{2\sigma^2}
 >
 \frac12.
\]
Thus
\[
 K(x)>0,
 \qquad x>0.
\]

Combining this with \eqref{eq:gt-decreasing} and
\eqref{eq:Mprime-detail}, we conclude that
\[
 M'(r)\le0,
 \qquad 0\le r\le\tau.
\]
Hence
\[
 M(\tau)\le M(0).
\]
Since
\[
 M(\tau)
 =
 \E\bigl[
   (\mathsf T_\tau\sech^4)(X)
 \bigr]
 =
 \E\sech^4(X_{s,u}),
\]
whereas
\[
 M(0)
 =
 \E\sech^4(X)
 =
 \E\sech^4(h+\beta\sqrt q\,Z),
\]
we obtain
\[
 \E\sech^4(X_{s,u})
 \le
 \E\sech^4(h+\beta\sqrt q\,Z).
\]
Multiplying by $\beta^2$ and using the definition of $\alpha$ gives
\[
 \beta^2\E\sech^4(X_{s,u})
 \le
 \beta^2\E\sech^4(h+\beta\sqrt q\,Z)
 =
 \alpha.
\]
This proves \eqref{eq:uppercomparison}.
\end{proof}
-/

import Lemmas.ATDefs
