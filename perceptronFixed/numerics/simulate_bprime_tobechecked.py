import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")

# Optional SciPy: used for deterministic, high-accuracy checks.
try:
    from scipy import integrate as sp_integrate
    from scipy import special as sp_special

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    sp_integrate = None  # type: ignore[assignment]
    sp_special = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float
    sd: float


def _mean_se(x: np.ndarray) -> Estimate:
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    if x.size <= 1:
        return Estimate(mean=mean, se=float("nan"), sd=float("nan"))
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(x.size)
    return Estimate(mean=mean, se=se, sd=sd)


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def refined_h_values(*, h_min: float = 1e-8) -> list[float]:
    """Refined sweep of finite-difference steps, from 1e-2 down to h_min."""
    h_min = float(h_min)
    if not (h_min > 0):
        raise ValueError("h_min must be positive")

    values: list[float] = []
    exp = -2
    while True:
        candidates = [10.0**exp, 5.0 * 10.0 ** (exp - 1), 2.0 * 10.0 ** (exp - 1)]
        for h in candidates:
            if h < h_min:
                return values
            values.append(h)
        exp -= 1


def _make_standard_normals(n: int, seed: int, antithetic: bool) -> np.ndarray:
    n = int(n)
    if n <= 0:
        raise ValueError("--n must be positive")
    rng = np.random.default_rng(int(seed))
    if not antithetic:
        return rng.standard_normal(n, dtype=np.float64)
    half = (n + 1) // 2
    z = rng.standard_normal(half, dtype=np.float64)
    out = np.concatenate([z, -z], axis=0)[:n]
    return out.astype(np.float64, copy=False)


def _mills_E_cf_pos(u: np.ndarray, terms: int) -> np.ndarray:
    """Vectorized continued fraction for E(u) when u>0."""
    u = np.asarray(u, dtype=np.float64)
    if np.any(u <= 0):
        raise ValueError("_mills_E_cf_pos requires u>0")
    t = np.zeros_like(u)
    for k in range(int(terms), 0, -1):
        t = k / (u + t)
    return u + t


def _mills_E_cf_pos_scalar(u: float, terms: int) -> float:
    """Scalar continued fraction for E(u) when u>0."""
    if not (u > 0):
        raise ValueError("_mills_E_cf_pos_scalar requires u>0")
    t = 0.0
    for k in range(int(terms), 0, -1):
        t = k / (u + t)
    return u + t


def _barPhi_erfc(u: np.ndarray) -> np.ndarray:
    """Gaussian upper tail barPhi(u) = 0.5 * erfc(u/sqrt(2)), using an accurate erfc."""
    u = np.asarray(u, dtype=np.float64)
    if hasattr(np, "erfc"):
        return 0.5 * np.erfc(u / SQRT2)
    # Fallback if NumPy is built without erfc ufunc.
    vec = np.vectorize(math.erfc, otypes=[np.float64])
    return 0.5 * vec(u / SQRT2)


def mills_E(
    u: np.ndarray,
    *,
    impl: str,
    cf_terms: int = 80,
    cf_min: float = 1.0,
) -> np.ndarray:
    """Inverse Mills ratio E(u) = phi(u)/barPhi(u).

    impl:
      - "erfcx": use SciPy's erfcx for a stable closed form (recommended)
      - "cf": continued fraction for u>=cf_min and direct erfc elsewhere
    """
    u = np.asarray(u, dtype=np.float64)

    if impl == "erfcx":
        if not _HAVE_SCIPY:
            raise RuntimeError("mills_E impl='erfcx' requires SciPy")
        # E(u) = sqrt(2/pi) / erfcx(u/sqrt(2))
        return (SQRT_2_OVER_PI / sp_special.erfcx(u / SQRT2)).astype(np.float64, copy=False)

    if impl != "cf":
        raise ValueError("impl must be 'erfcx' or 'cf'")

    out = np.empty_like(u)
    cf_min = float(cf_min)

    mask_cf = u >= cf_min
    if np.any(mask_cf):
        out[mask_cf] = _mills_E_cf_pos(u[mask_cf], terms=int(cf_terms))

    mask_direct = ~mask_cf
    if np.any(mask_direct):
        ud = u[mask_direct]
        phi = np.exp(-0.5 * ud * ud) / SQRT_2PI
        tail = _barPhi_erfc(ud)
        out[mask_direct] = np.divide(phi, tail, out=np.zeros_like(phi), where=tail > 0)

    return out


def mills_E_scalar(
    u: float,
    *,
    impl: str,
    cf_terms: int = 80,
    cf_min: float = 1.0,
) -> float:
    """Scalar inverse Mills ratio, for SciPy quad mode."""
    if impl == "erfcx":
        if not _HAVE_SCIPY:
            raise RuntimeError("mills_E impl='erfcx' requires SciPy")
        return float(SQRT_2_OVER_PI / sp_special.erfcx(u / SQRT2))

    if impl != "cf":
        raise ValueError("impl must be 'erfcx' or 'cf'")

    if u >= float(cf_min):
        return float(_mills_E_cf_pos_scalar(u, terms=int(cf_terms)))

    # Direct formula is safe for u below the CF threshold.
    phi = math.exp(-0.5 * u * u) / SQRT_2PI
    tail = 0.5 * math.erfc(u / SQRT2)
    return phi / tail if tail > 0.0 else 0.0


def U_t(z: np.ndarray, *, kappa: float, t: float) -> np.ndarray:
    t = float(t)
    if not (0.0 < t < 1.0):
        raise ValueError("t must be in (0,1)")
    z = np.asarray(z, dtype=np.float64)
    return (float(kappa) - math.sqrt(t) * z) / math.sqrt(1.0 - t)


def B_contrib(
    z: np.ndarray,
    *,
    kappa: float,
    t: float,
    mills_impl: str,
    cf_terms: int,
    cf_min: float,
) -> np.ndarray:
    """Per-sample contributions of B(t) := (1-t) E[ E(U_t)^2 ]."""
    u = U_t(z, kappa=kappa, t=t)
    e = mills_E(u, impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
    return (1.0 - float(t)) * (e * e)


def g_contrib(
    z: np.ndarray,
    *,
    kappa: float,
    t: float,
    mills_impl: str,
    cf_terms: int,
    cf_min: float,
) -> np.ndarray:
    r"""Per-sample contributions of g(U_t).

      E'(u) = E(u)^2 - u E(u)

      g(u) := E'(u)^2 - 2(1 - E'(u)) E(u)^2.
    """
    u = U_t(z, kappa=kappa, t=t)
    e = mills_E(u, impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
    e2 = e * e
    eprime = e2 - u * e
    return eprime * eprime - 2.0 * (1.0 - eprime) * e2


def _safe_step(t: float, h: float, *, stencil_radius: int) -> float:
    """Clamp |h| so that t +/- stencil_radius*h stays in (0,1)."""
    t = float(t)
    h = abs(float(h))
    eps = 1e-12
    r = int(stencil_radius)
    if r <= 0:
        raise ValueError("stencil_radius must be positive")

    max_h = min(0.25, (t - eps) / r, (1.0 - t - eps) / r)
    if max_h <= 0:
        raise ValueError(f"t={t} too close to boundary for finite differences")

    if not (h > 0.0):
        h = 1e-4
    return min(h, max_h)


def _fd5_from_values(fm2: float, fm1: float, fp1: float, fp2: float, h: float) -> float:
    """5-point central difference: f'(t) approx (f(t-2h)-8f(t-h)+8f(t+h)-f(t+2h))/(12h)."""
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)


def _fd5_from_arrays(fm2: np.ndarray, fm1: np.ndarray, fp1: np.ndarray, fp2: np.ndarray, h: float) -> np.ndarray:
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)


def _gh_nodes_weights(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for standard normal expectation."""
    m = int(m)
    if m <= 0:
        raise ValueError("--gh-n must be positive")
    x, w = np.polynomial.hermite.hermgauss(m)  # for \int e^{-x^2} f(x) dx
    z = SQRT2 * x
    wN = w / math.sqrt(math.pi)  # weights for E[f(Z)]
    return z.astype(np.float64, copy=False), wN.astype(np.float64, copy=False)


def _normal_expect_gh(z: np.ndarray, w: np.ndarray, fvals: np.ndarray) -> float:
    return float(np.dot(w, fvals))


def _normal_expect_quad(
    f: Callable[[float], float],
    *,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    if not _HAVE_SCIPY:
        raise RuntimeError("--method quad requires SciPy")

    def integrand(z: float) -> float:
        return f(z) * math.exp(-0.5 * z * z) / SQRT_2PI

    val, err = sp_integrate.quad(
        integrand,
        -math.inf,
        math.inf,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=int(limit),
    )
    return float(val), float(err)


def run_bprime_check_mc(
    *,
    kappa: float,
    t: float,
    h_values: Sequence[float],
    n: int,
    seed: int,
    antithetic: bool,
    mills_impl: str,
    cf_terms: int,
    cf_min: float,
    tol: float,
) -> None:
    z = _make_standard_normals(n=n, seed=seed, antithetic=antithetic)

    u0 = U_t(z, kappa=kappa, t=t)
    frac_neg = float(np.mean(u0 < 0))
    u_min = float(np.min(u0))
    u_max = float(np.max(u0))

    g0 = g_contrib(z, kappa=kappa, t=t, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
    g_est = _mean_se(g0)

    print("=== B'(t) check: eq:Bprime-TOBECHECKED ===")
    print(f"method: mc (stochastic)")
    print(f"params: kappa={kappa:.6g}, t={t:.12g}")
    print(f"mc: N={z.size}, seed={seed}, antithetic={antithetic}")
    print(f"E(u) eval: mills_impl={mills_impl}, cf_terms={cf_terms}, cf_min={cf_min}")
    print(f"U_t stats: P(U_t<0)~{frac_neg:.6f}, min~{u_min:.6g}, max~{u_max:.6g}")
    print("")
    print(f"RHS: E[g(U_t)] ~ {g_est.mean:.12g}  (se~{g_est.se:.3g}, sd~{g_est.sd:.3g})")
    print(f"sign check: E[g(U_t)] < 0 ? {'YES' if g_est.mean < 0 else 'NO'}")
    print("")

    b0 = B_contrib(z, kappa=kappa, t=t, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
    B0 = float(np.mean(b0))

    for raw_h in h_values:
        h = _safe_step(t, raw_h, stencil_radius=2)

        bm2 = B_contrib(z, kappa=kappa, t=t - 2.0 * h, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        bm1 = B_contrib(z, kappa=kappa, t=t - h, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        bp1 = B_contrib(z, kappa=kappa, t=t + h, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        bp2 = B_contrib(z, kappa=kappa, t=t + 2.0 * h, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)

        Bm2 = float(np.mean(bm2))
        Bm1 = float(np.mean(bm1))
        Bp1 = float(np.mean(bp1))
        Bp2 = float(np.mean(bp2))

        fd_samples = _fd5_from_arrays(bm2, bm1, bp1, bp2, h)
        fd_est = _mean_se(fd_samples)

        diff_samples = fd_samples - g0
        diff_est = _mean_se(diff_samples)
        z_score = diff_est.mean / diff_est.se if diff_est.se and math.isfinite(diff_est.se) else float("nan")

        print(f"--- finite diff (5-pt) step h={h:.6g} (requested {raw_h:.6g}) ---")
        print(f"B(t-2h)~{Bm2:.12g}  B(t-h)~{Bm1:.12g}  B(t)~{B0:.12g}  B(t+h)~{Bp1:.12g}  B(t+2h)~{Bp2:.12g}")
        print(f"FD5: ~ {fd_est.mean:.12g}  (se~{fd_est.se:.3g})")
        print(f"DIFF: FD5 - E[g(U_t)] ~ {diff_est.mean:.12g}  (se~{diff_est.se:.3g}, z~{z_score:.2f})")
        print(f"abs diff check (|diff| <= {tol:g}): {'YES' if abs(diff_est.mean) <= tol else 'NO'}")
        print("")


def run_bprime_check_gh(
    *,
    kappa: float,
    t: float,
    h_values: Sequence[float],
    gh_n: int,
    mills_impl: str,
    cf_terms: int,
    cf_min: float,
    tol: float,
) -> None:
    z, w = _gh_nodes_weights(gh_n)

    def B_expect(tt: float) -> float:
        vals = B_contrib(z, kappa=kappa, t=tt, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        return _normal_expect_gh(z, w, vals)

    g_vals = g_contrib(z, kappa=kappa, t=t, mills_impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
    g_rhs = _normal_expect_gh(z, w, g_vals)
    B0 = B_expect(t)

    print("=== B'(t) check: eq:Bprime-TOBECHECKED ===")
    print("method: gh (deterministic Gauss-Hermite)")
    print(f"params: kappa={kappa:.6g}, t={t:.12g}")
    print(f"quadrature: gh_n={gh_n}")
    print(f"E(u) eval: mills_impl={mills_impl}, cf_terms={cf_terms}, cf_min={cf_min}")
    print("")
    print(f"RHS: E[g(U_t)] = {g_rhs:.15g}")
    print("")

    cache: dict[float, float] = {t: B0}

    def B_cached(tt: float) -> float:
        if tt not in cache:
            cache[tt] = B_expect(tt)
        return cache[tt]

    for raw_h in h_values:
        h = _safe_step(t, raw_h, stencil_radius=2)
        Bm2 = B_cached(t - 2.0 * h)
        Bm1 = B_cached(t - h)
        Bp1 = B_cached(t + h)
        Bp2 = B_cached(t + 2.0 * h)

        fd5 = _fd5_from_values(Bm2, Bm1, Bp1, Bp2, h)
        diff = fd5 - g_rhs

        print(f"--- finite diff (5-pt) step h={h:.6g} (requested {raw_h:.6g}) ---")
        print(f"B(t-2h)={Bm2:.15g}  B(t-h)={Bm1:.15g}  B(t)={B0:.15g}  B(t+h)={Bp1:.15g}  B(t+2h)={Bp2:.15g}")
        print(f"FD5: {fd5:.15g}")
        print(f"DIFF: FD5 - E[g(U_t)] = {diff:.15g}")
        print(f"abs diff check (|diff| <= {tol:g}): {'YES' if abs(diff) <= tol else 'NO'}")
        print("")


def run_bprime_check_quad(
    *,
    kappa: float,
    t: float,
    h_values: Sequence[float],
    mills_impl: str,
    cf_terms: int,
    cf_min: float,
    quad_epsabs: float,
    quad_epsrel: float,
    quad_limit: int,
    tol: float,
) -> None:
    if not _HAVE_SCIPY:
        raise RuntimeError("--method quad requires SciPy")

    def B_scalar(z: float, tt: float) -> float:
        u = (kappa - math.sqrt(tt) * z) / math.sqrt(1.0 - tt)
        e = mills_E_scalar(u, impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        return (1.0 - tt) * (e * e)

    def g_scalar(z: float) -> float:
        u = (kappa - math.sqrt(t) * z) / math.sqrt(1.0 - t)
        e = mills_E_scalar(u, impl=mills_impl, cf_terms=cf_terms, cf_min=cf_min)
        e2 = e * e
        eprime = e2 - u * e
        return eprime * eprime - 2.0 * (1.0 - eprime) * e2

    print("=== B'(t) check: eq:Bprime-TOBECHECKED ===")
    print("method: quad (deterministic adaptive integration)")
    print(f"params: kappa={kappa:.6g}, t={t:.12g}")
    print(f"quad: epsabs={quad_epsabs:g}, epsrel={quad_epsrel:g}, limit={quad_limit}")
    print(f"E(u) eval: mills_impl={mills_impl}, cf_terms={cf_terms}, cf_min={cf_min}")
    print("")

    g_rhs, g_err = _normal_expect_quad(
        g_scalar, epsabs=quad_epsabs, epsrel=quad_epsrel, limit=quad_limit
    )

    print(f"RHS: E[g(U_t)] = {g_rhs:.15g}  (quad err~{g_err:.2g})")
    print("")

    cache: dict[float, tuple[float, float]] = {}

    def B_expect(tt: float) -> tuple[float, float]:
        if tt in cache:
            return cache[tt]
        val, err = _normal_expect_quad(
            lambda z: B_scalar(z, tt), epsabs=quad_epsabs, epsrel=quad_epsrel, limit=quad_limit
        )
        cache[tt] = (val, err)
        return val, err

    B0, B0_err = B_expect(t)

    for raw_h in h_values:
        h = _safe_step(t, raw_h, stencil_radius=2)

        Bm2, em2 = B_expect(t - 2.0 * h)
        Bm1, em1 = B_expect(t - h)
        Bp1, ep1 = B_expect(t + h)
        Bp2, ep2 = B_expect(t + 2.0 * h)

        fd5 = _fd5_from_values(Bm2, Bm1, Bp1, Bp2, h)
        diff = fd5 - g_rhs

        # A conservative propagated quadrature error scale for the FD combination.
        fd_err = (abs(em2) + 8.0 * abs(em1) + 8.0 * abs(ep1) + abs(ep2)) / (12.0 * h)

        print(f"--- finite diff (5-pt) step h={h:.6g} (requested {raw_h:.6g}) ---")
        print(
            f"B(t-2h)={Bm2:.15g}  B(t-h)={Bm1:.15g}  B(t)={B0:.15g}  B(t+h)={Bp1:.15g}  B(t+2h)={Bp2:.15g}"
        )
        print(f"FD5: {fd5:.15g}  (quad-prop err~{fd_err:.2g})")
        print(f"DIFF: FD5 - E[g(U_t)] = {diff:.15g}")
        print(f"abs diff check (|diff| <= {tol:g}): {'YES' if abs(diff) <= tol else 'NO'}")
        print("")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check eq:Bprime-TOBECHECKED by comparing B'(t) finite differences "
            "to E[g(U_t)]. For |DIFF| < 1e-7 you need a deterministic method; "
            "use --method quad (recommended) or --method gh."
        )
    )
    parser.add_argument("--kappa", type=float, default=0.0, help="margin kappa")
    parser.add_argument("--t", type=float, default=0.5, help="t in (0,1)")
    parser.add_argument("--h", type=float, default=1e-4, help="finite difference step")
    parser.add_argument(
        "--h-list",
        type=str,
        default="",
        help="comma-separated list of steps (overrides --h/--refine)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="use a built-in h sweep down to 1e-8 (use --h-list to override)",
    )
    parser.add_argument(
        "--refine-min",
        type=float,
        default=1e-8,
        help="smallest h for --refine (default: 1e-8)",
    )

    default_method = "quad" if _HAVE_SCIPY else "mc"
    parser.add_argument(
        "--method",
        choices=["mc", "gh", "quad"],
        default=default_method,
        help="estimation method: mc (Monte Carlo), gh (Gauss-Hermite), quad (adaptive integration)",
    )

    default_mills = "erfcx" if _HAVE_SCIPY else "cf"
    parser.add_argument(
        "--mills",
        choices=["erfcx", "cf"],
        default=default_mills,
        help="how to compute the inverse Mills ratio E(u)",
    )

    # Monte Carlo options
    parser.add_argument("--n", type=int, default=120_000, help="Monte Carlo sample size")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--antithetic", action="store_true", help="use Z and -Z pairs")

    # Gauss-Hermite options
    parser.add_argument("--gh-n", type=int, default=260, help="number of Gauss-Hermite nodes")

    # Mills ratio CF options (used when --mills cf)
    parser.add_argument("--cf-terms", type=int, default=120, help="continued fraction depth")
    parser.add_argument(
        "--cf-min",
        type=float,
        default=1.0,
        help="use continued fraction for u >= cf_min (else use direct erfc formula)",
    )

    # Quad options
    parser.add_argument("--quad-epsabs", type=float, default=1e-12, help="quad absolute tolerance")
    parser.add_argument("--quad-epsrel", type=float, default=1e-12, help="quad relative tolerance")
    parser.add_argument("--quad-limit", type=int, default=400, help="quad max subintervals")

    parser.add_argument("--tol", type=float, default=1e-7, help="target tolerance for |DIFF|")

    args = parser.parse_args(list(argv) if argv is not None else None)

    t = float(args.t)
    if not (0.0 < t < 1.0):
        raise SystemExit("--t must be in (0,1)")

    h_values = _parse_float_list(args.h_list)
    if not h_values:
        if bool(args.refine):
            h_values = refined_h_values(h_min=float(args.refine_min))
        else:
            h_values = [float(args.h)]

    method = str(args.method)
    mills_impl = str(args.mills)

    if mills_impl == "erfcx" and not _HAVE_SCIPY:
        raise SystemExit("--mills erfcx requires SciPy; install scipy or use --mills cf")

    if method == "mc":
        run_bprime_check_mc(
            kappa=float(args.kappa),
            t=t,
            h_values=h_values,
            n=int(args.n),
            seed=int(args.seed),
            antithetic=bool(args.antithetic),
            mills_impl=mills_impl,
            cf_terms=int(args.cf_terms),
            cf_min=float(args.cf_min),
            tol=float(args.tol),
        )
        return 0

    if method == "gh":
        run_bprime_check_gh(
            kappa=float(args.kappa),
            t=t,
            h_values=h_values,
            gh_n=int(args.gh_n),
            mills_impl=mills_impl,
            cf_terms=int(args.cf_terms),
            cf_min=float(args.cf_min),
            tol=float(args.tol),
        )
        return 0

    if method == "quad":
        run_bprime_check_quad(
            kappa=float(args.kappa),
            t=t,
            h_values=h_values,
            mills_impl=mills_impl,
            cf_terms=int(args.cf_terms),
            cf_min=float(args.cf_min),
            quad_epsabs=float(args.quad_epsabs),
            quad_epsrel=float(args.quad_epsrel),
            quad_limit=int(args.quad_limit),
            tol=float(args.tol),
        )
        return 0

    raise SystemExit(f"unknown --method: {method}")


if __name__ == "__main__":
    raise SystemExit(main())
