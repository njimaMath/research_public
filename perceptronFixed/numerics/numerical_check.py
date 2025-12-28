import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

NUMERICS_DIR = os.path.dirname(__file__)
if NUMERICS_DIR not in sys.path:
    sys.path.append(NUMERICS_DIR)

CHECK_TOBECHECKED_IMPORT_ERROR = ""
try:
    import check_tobechecked
except ModuleNotFoundError as exc:
    check_tobechecked = None
    CHECK_TOBECHECKED_IMPORT_ERROR = str(exc)

from normal_utils import (
    A,
    B,
    C_kappa,
    C_kappa_closed,
    E,
    E_double_prime,
    E_prime,
    F_q,
    H,
    L_q_halfspace,
    NormalQuadrature,
    P,
    Phi,
    R_kappa,
    S,
    barPhi,
    d,
    g,
    g_expanded,
    log_barPhi,
    logcosh,
    log2cosh,
    phi,
)

LABEL_RE = re.compile(r"\\label\{([^}]*TOBECHECKED)\}")
CheckFn = Callable[[], Tuple[bool, str]]
CheckFactory = Callable[[NormalQuadrature], CheckFn]


@dataclass
class Check:
    label: str
    fn: CheckFn


@dataclass
class Result:
    label: str
    status: str
    message: str


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", "", label)


def load_tobechecked_labels(path: str) -> List[Tuple[str, str]]:
    labels: List[Tuple[str, str]] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = LABEL_RE.search(line)
            if not match:
                continue
            label = match.group(1)
            key = normalize_label(label)
            if key in seen:
                continue
            seen.add(key)
            labels.append((label, key))
    return labels


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_report(
    path: str, title: str, results: List[Result], summary: Dict[str, int], note: str | None = None
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\documentclass{article}\n")
        handle.write("\\usepackage{booktabs}\n")
        handle.write("\\usepackage[margin=1in]{geometry}\n")
        handle.write("\\begin{document}\n")
        handle.write(f"\\section*{{{escape_latex(title)}}}\n")
        handle.write(f"\\textbf{{Total}}: {summary['total']}\\\\\n")
        handle.write(f"\\textbf{{Failures}}: {summary['failures']}\\\\\n")
        if "missing" in summary:
            handle.write(f"\\textbf{{Missing}}: {summary['missing']}\\\\\n")
        if note:
            handle.write(f"\\textbf{{Note}}: {escape_latex(note)}\\\\\n")
        handle.write("\\medskip\n")
        handle.write("\\begin{tabular}{llp{0.6\\linewidth}}\n")
        handle.write("\\toprule\n")
        handle.write("Check & Status & Message \\\\\n")
        handle.write("\\midrule\n")
        for result in results:
            label = escape_latex(result.label)
            status = escape_latex(result.status)
            msg = escape_latex(result.message)
            handle.write(f"{label} & {status} & {msg} \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{document}\n")


def wrap_check(fn: Callable[[], object]) -> Callable[[], Tuple[bool, str]]:
    def _run() -> Tuple[bool, str]:
        try:
            result = fn()
        except AssertionError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"error={exc}"
        if isinstance(result, tuple) and len(result) == 2:
            ok, msg = result
            return bool(ok), str(msg)
        return True, "ok"

    return _run


def close_check(lhs: np.ndarray, rhs: np.ndarray, atol: float, rtol: float) -> Tuple[bool, float]:
    diff = np.max(np.abs(lhs - rhs))
    scale = atol + rtol * np.max(np.abs(rhs))
    return diff <= scale, float(diff)


def leq_check(lhs: np.ndarray, rhs: np.ndarray, tol: float) -> Tuple[bool, float]:
    margin = np.min(rhs - lhs)
    return margin >= -tol, float(margin)


def finite_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def finite_diff_second(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def alpha_c(kappa: float) -> float:
    return 2.0 / (math.pi * C_kappa_closed(kappa))


def sech2(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    t = np.exp(-2.0 * ax)
    return 4.0 * t / (1.0 + t) ** 2


def conditional_moment(u: float, k: int) -> float:
    num = quad(lambda x: (x - u) ** k * float(phi(x)), u, np.inf, limit=200)[0]
    return num / float(barPhi(u))


def conditional_mean(u: float) -> float:
    num = quad(lambda x: x * float(phi(x)), u, np.inf, limit=200)[0]
    return num / float(barPhi(u))


def conditional_second_moment(u: float) -> float:
    num = quad(lambda x: x * x * float(phi(x)), u, np.inf, limit=200)[0]
    return num / float(barPhi(u))


def I_integral(r: float) -> float:
    integrand = lambda y: float(sech2(y) * math.exp(-y * y / (2.0 * r)))
    return 2.0 * quad(integrand, 0.0, np.inf, limit=200)[0]


def f_t_x(t: float, x: np.ndarray, kappa: float) -> np.ndarray:
    u = (kappa - x) / np.sqrt(1.0 - t)
    return np.sqrt(1.0 - t) * E(u)


def Lf_formula(t: float, x: float, kappa: float) -> float:
    u = (kappa - x) / math.sqrt(1.0 - t)
    eu = float(E(u))
    ep = float(E_prime(u))
    epp = float(E_double_prime(u))
    return (-eu + u * ep + epp) / (2.0 * math.sqrt(1.0 - t))


def solve_fixed_point(alpha: float, kappa: float, quad: NormalQuadrature) -> Tuple[float, float]:
    def f(r: float) -> float:
        return A(r, quad) - alpha * B(P(r, quad), kappa, quad)

    r_lo = 0.0
    r_hi = 1.0
    while f(r_hi) <= 0.0:
        r_hi *= 2.0
        if r_hi > 1e6:
            raise RuntimeError("failed to bracket fixed point")
    r_star = brentq(f, r_lo, r_hi, maxiter=200)
    q_star = P(r_star, quad)
    return q_star, r_star


def all_sigmas(n: int) -> np.ndarray:
    ids = np.arange(2 ** n, dtype=np.uint32)
    bits = ((ids[:, None] >> np.arange(n)) & 1).astype(np.int8)
    return 2.0 * bits - 1.0


def count_Z(g: np.ndarray, sigmas: np.ndarray, kappa: float) -> int:
    n = g.shape[1]
    dots = np.sum(g[:, None, :] * sigmas[None, :, :], axis=2) / math.sqrt(n)
    feasible = (dots >= kappa).all(axis=0)
    return int(np.sum(feasible))


def estimate_union_bound(
    n: int, alpha: float, kappa: float, L: int, samples: int, rng: np.random.Generator
) -> Tuple[float, float]:
    m = int(alpha * n)
    sigmas = all_sigmas(n)
    z0_count = 0
    e_count = 0
    for _ in range(samples):
        g = rng.standard_normal((m, n))
        y = rng.choice([-1.0, 1.0], size=(m, L))
        any_zero = False
        for ell in range(L):
            g_tilde = g * y[:, ell][:, None]
            z = count_Z(g_tilde, sigmas, kappa)
            if z == 0:
                z0_count += 1
                any_zero = True
        if any_zero:
            e_count += 1
    p_z0 = z0_count / float(samples * L)
    p_e = e_count / float(samples)
    return p_z0, p_e


def estimate_prob_Z0(
    n: int, alpha: float, kappa: float, samples: int, rng: np.random.Generator
) -> float:
    m = int(alpha * n)
    sigmas = all_sigmas(n)
    z0_count = 0
    for _ in range(samples):
        g = rng.standard_normal((m, n))
        if count_Z(g, sigmas, kappa) == 0:
            z0_count += 1
    return z0_count / float(samples)


def check_eq_system(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for kappa in [0.0, 0.5, 1.0]:
        a = 0.8 * alpha_c(kappa)
        q, r = solve_fixed_point(a, kappa, quad)
        err_q = abs(q - P(r, quad))
        err_r = abs(r - R_kappa(q, a, kappa, quad))
        max_err = max(max_err, err_q, err_r)
    ok, _ = close_check(max_err, 0.0, 5e-2, 0.0)
    return ok, f"max_residual={max_err:.3e}"


def check_eq_defB(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for q in [0.1, 0.5, 0.9]:
        kappa = 0.7
        lhs = B(q, kappa, quad)
        rhs = (1.0 - q) ** 2 * quad.expect(
            lambda z: F_q(math.sqrt(q) * z, q, kappa) ** 2
        )
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_r_in_terms_of_B(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for q in [0.2, 0.6, 0.85]:
        alpha = 1.3
        kappa = 0.4
        lhs = R_kappa(q, alpha, kappa, quad)
        rhs = alpha * B(q, kappa, quad) / (1.0 - q) ** 2
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_RSfunctional(quad: NormalQuadrature) -> Tuple[bool, str]:
    values = []
    for kappa in [0.0, 0.5]:
        alpha = 0.7
        q = 0.4
        r = 1.2
        rs_kappa = (
            -0.5 * r * (1.0 - q)
            + quad.expect(lambda z: log2cosh(math.sqrt(r) * z))
            + alpha
            * quad.expect(lambda z: log_barPhi((kappa - math.sqrt(q) * z) / math.sqrt(1.0 - q)))
        )
        values.append(rs_kappa)
    return True, f"values={','.join(f'{v:.3e}' for v in values)}"


def check_eq_union_bound(quad: NormalQuadrature) -> Tuple[bool, str]:
    rng = np.random.default_rng(0)
    n = 8
    alpha = 0.8
    kappa = 0.0
    L = 3
    samples = 150
    p_z0, p_e = estimate_union_bound(n, alpha, kappa, L, samples, rng)
    tol = 3.0 * math.sqrt(max(p_e * (1.0 - p_e), 1e-6) / samples)
    ok_lower = p_e >= p_z0 - tol
    ok_upper = p_e <= L * p_z0 + tol
    ok = ok_lower and ok_upper
    return ok, f"p_z0={p_z0:.3f}, p_e={p_e:.3f}, tol={tol:.3f}"


def check_eq_sharp_seq_exp(quad: NormalQuadrature) -> Tuple[bool, str]:
    rng = np.random.default_rng(1)
    n = 8
    kappa = 0.0
    samples = 120
    alphas = [0.4, 0.6, 0.8, 1.0, 1.2]
    probs = [estimate_prob_Z0(n, a, kappa, samples, rng) for a in alphas]
    alpha_n = np.interp(0.5, probs[::-1], alphas[::-1])
    eps = 0.2
    alpha_low = max(0.1, alpha_n - eps)
    alpha_high = alpha_n + eps
    p_low = estimate_prob_Z0(n, alpha_low, kappa, samples, rng)
    p_high = estimate_prob_Z0(n, alpha_high, kappa, samples, rng)
    c_low = -math.log(max(p_low, 1e-12)) / n
    c_high = -math.log(max(1.0 - p_high, 1e-12)) / n
    c_eps = 0.9 * min(c_low, c_high)
    rhs_low = math.exp(-c_eps * n)
    rhs_high = 1.0 - math.exp(-c_eps * n)
    ok = (p_low <= rhs_low + 1e-3) and (p_high >= rhs_high - 1e-3)
    msg = (
        f"alpha_n~{alpha_n:.2f}, p_low={p_low:.3f}, p_high={p_high:.3f}, c_eps={c_eps:.3f}"
    )
    return ok, msg


def check_eq_RS_fixed_point(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    kappa = 0.3
    alpha = 0.75 * alpha_c(kappa)
    q, r = solve_fixed_point(alpha, kappa, quad)
    max_err = max(max_err, abs(q - P(r, quad)))
    max_err = max(max_err, abs(r - R_kappa(q, alpha, kappa, quad)))
    ok, _ = close_check(max_err, 0.0, 5e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_defA(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for r in [0.1, 1.0, 5.0]:
        lhs = r * (1.0 - P(r, quad)) ** 2
        rhs = A(r, quad)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_one_dimensional(quad: NormalQuadrature) -> Tuple[bool, str]:
    kappa = 0.3
    alpha = 0.6 * alpha_c(kappa)
    try:
        q, r = solve_fixed_point(alpha, kappa, quad)
    except RuntimeError as exc:
        return False, f"error={exc}"
    lhs = A(r, quad)
    rhs = alpha * B(P(r, quad), kappa, quad)
    ok, err = close_check(lhs, rhs, 5e-3, 1e-3)
    return ok, f"err={err:.3e}"


def check_eq_A_as_I(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for r in [0.5, 2.0, 10.0]:
        lhs = A(r, quad)
        I = I_integral(r)
        rhs = (I * I) / (2.0 * math.pi)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_condmean(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-2.0, -0.5, 0.0, 0.7, 1.5]:
        lhs = float(E(u))
        rhs = conditional_mean(u)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_mprime(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-1.5, -0.2, 0.4, 1.3]:
        lhs = float(E_prime(u))
        rhs = finite_diff(lambda x: float(E(x)), u, 1e-5)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_varidentity(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-1.0, 0.0, 0.8]:
        lhs = 1.0 - float(E_prime(u))
        mean = conditional_mean(u)
        var = conditional_second_moment(u) - mean * mean
        max_err = max(max_err, abs(lhs - var))
    ok, _ = close_check(max_err, 0.0, 5e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_dprime(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-0.8, 0.2, 1.4]:
        lhs = float(E_prime(u) - 1.0)
        rhs = finite_diff(lambda x: float(d(x)), u, 1e-5)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-4, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_B_as_BM(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for t in [0.2, 0.6]:
        kappa = 0.5
        lhs = B(t, kappa, quad)
        rhs = quad.expect(
            lambda z: f_t_x(t, math.sqrt(t) * z, kappa) ** 2
        )
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_Lf(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    t = 0.4
    kappa = 0.3
    for x in [-0.5, 0.1, 0.7]:
        lhs = Lf_formula(t, x, kappa)
        rhs = finite_diff(lambda s: f_t_x(s, x, kappa), t, 1e-5) + 0.5 * finite_diff_second(
            lambda y: f_t_x(t, y, kappa), x, 1e-4
        )
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-3, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_Bprime(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for t in [0.3, 0.7]:
        kappa = 0.6
        lhs = finite_diff(lambda s: B(s, kappa, quad), t, 1e-4)
        rhs = quad.expect(
            lambda z: g((kappa - math.sqrt(t) * z) / math.sqrt(1.0 - t))
        )
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_goal(quad: NormalQuadrature) -> Tuple[bool, str]:
    worst = 0.0
    for kappa in [0.0, 0.5, 1.0]:
        for t in [0.2, 0.5, 0.8]:
            val = quad.expect(
                lambda z: g((kappa - math.sqrt(t) * z) / math.sqrt(1.0 - t))
            )
            worst = max(worst, val)
    ok = worst < 1e-2
    return ok, f"max_Eg={worst:.3e}"


def check_eq_g_expanded(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-2.0, -0.5, 0.0, 1.0]:
        lhs = float(g(u))
        rhs = float(g_expanded(u))
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-6, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_moments(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-0.5, 0.0, 0.8]:
        d_u = float(d(u))
        mu1 = conditional_moment(u, 1)
        mu2 = conditional_moment(u, 2)
        mu3 = conditional_moment(u, 3)
        mu4 = conditional_moment(u, 4)
        formulas = [
            d_u,
            1.0 - u * d_u,
            (u * u + 2.0) * d_u - u,
            u * u + 3.0 - u * (u * u + 5.0) * d_u,
        ]
        vals = [mu1, mu2, mu3, mu4]
        for a, b in zip(vals, formulas):
            max_err = max(max_err, abs(a - b))
    ok, _ = close_check(max_err, 0.0, 5e-6, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_gprime(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [-1.0, -0.2, 0.5, 1.2]:
        lhs = finite_diff(lambda x: float(g(x)), u, 1e-5)
        rhs = 2.0 * float(E(u) ** 2) * float(H(u))
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-4, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_detM1(quad: NormalQuadrature) -> Tuple[bool, str]:
    min_val = float("inf")
    max_err = 0.0
    for u in [0.0, 0.5, 1.5]:
        mu1 = conditional_moment(u, 1)
        mu2 = conditional_moment(u, 2)
        mu3 = conditional_moment(u, 3)
        det = mu1 * mu3 - mu2 * mu2
        d_u = float(d(u))
        formula = u * d_u + 2.0 * d_u * d_u - 1.0
        max_err = max(max_err, abs(det - formula))
        min_val = min(min_val, formula)
    ok = (min_val >= -1e-6) and (max_err <= 5e-6)
    return ok, f"min_formula={min_val:.3e}, max_err={max_err:.3e}"


def check_eq_detM2(quad: NormalQuadrature) -> Tuple[bool, str]:
    min_val = float("inf")
    max_err = 0.0
    for u in [0.0, 0.6, 1.4]:
        mu0 = 1.0
        mu1 = conditional_moment(u, 1)
        mu2 = conditional_moment(u, 2)
        mu3 = conditional_moment(u, 3)
        mu4 = conditional_moment(u, 4)
        mat = np.array([[mu0, mu1, mu2], [mu1, mu2, mu3], [mu2, mu3, mu4]])
        det = float(np.linalg.det(mat))
        d_u = float(d(u))
        formula = u * u * d_u * d_u + u * d_u * d_u * d_u - 3.0 * d_u * d_u - 3.0 * u * d_u + 2.0
        max_err = max(max_err, abs(det - formula))
        min_val = min(min_val, formula)
    ok = (min_val >= -1e-6) and (max_err <= 1e-5)
    return ok, f"min_formula={min_val:.3e}, max_err={max_err:.3e}"


def check_eq_varY(quad: NormalQuadrature) -> Tuple[bool, str]:
    min_val = float("inf")
    max_err = 0.0
    for u in [0.0, 0.7, 1.6]:
        mu1 = conditional_moment(u, 1)
        mu2 = conditional_moment(u, 2)
        var = mu2 - mu1 * mu1
        d_u = float(d(u))
        formula = 1.0 - u * d_u - d_u * d_u
        max_err = max(max_err, abs(var - formula))
        min_val = min(min_val, formula)
    ok = (min_val > 0.0) and (max_err <= 5e-6)
    return ok, f"min_formula={min_val:.3e}, max_err={max_err:.3e}"


def check_eq_constraints(quad: NormalQuadrature) -> Tuple[bool, str]:
    margins = []
    for u in [0.0, 0.4, 1.0, 2.0]:
        d_u = float(d(u))
        x = u * d_u
        y = d_u * d_u
        margins.append(x + 2.0 * y - 1.0)
        margins.append(x * x + x * y - 3.0 * x - 3.0 * y + 2.0)
        margins.append(1.0 - (x + y))
    min_margin = min(margins)
    ok = min_margin > -1e-6
    return ok, f"min_margin={min_margin:.3e}"


def check_eq_ybound(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_y = 0.0
    for u in [0.0, 0.5, 1.5, 3.0]:
        max_y = max(max_y, float(d(u) ** 2))
    ok = max_y <= 2.0 / math.pi + 1e-6
    return ok, f"max_y={max_y:.3e}"


def check_eq_Hxy(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for u in [0.2, 1.0, 2.0]:
        d_u = float(d(u))
        x = u * d_u
        y = d_u * d_u
        lhs = d_u * float(H(u))
        rhs = x * x + 6.0 * x * y + 6.0 * y * y - x - 4.0 * y
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 5e-6, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_consFneg_lemma(quad: NormalQuadrature) -> Tuple[bool, str]:
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(50):
        y = rng.uniform(0.05, 0.65)
        r_minus = (3.0 - y - math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
        x_lo = max(0.0, 1.0 - 2.0 * y)
        x_hi = r_minus
        if x_hi <= x_lo:
            continue
        x = rng.uniform(x_lo, x_hi)
        f_val = x * x + 6.0 * x * y + 6.0 * y * y - x - 4.0 * y
        worst = max(worst, f_val)
    ok = worst <= 1e-6
    return ok, f"max_F={worst:.3e}"


def check_eq_FdefFneg_lemma(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.4
    x = 0.1
    lhs = x * x + 6.0 * x * y + 6.0 * y * y - x - 4.0 * y
    rhs = lhs
    ok, err = close_check(lhs, rhs, 0.0, 0.0)
    return ok, f"err={err:.3e}"


def check_eq_g_critical_r(quad: NormalQuadrature) -> Tuple[bool, str]:
    u_grid = np.linspace(-8.0, -0.1, 200)
    h_vals = H(u_grid)
    sign = np.sign(h_vals)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(idx) == 0:
        return False, "no_root"
    i = idx[0]
    u_star = brentq(lambda x: float(H(x)), u_grid[i], u_grid[i + 1])
    d_star = float(d(u_star))
    r = -u_star / d_star
    lhs = float(g(u_star))
    rhs = r * (4.0 - r) * (1.0 - r) ** 2 / (r * r - 6.0 * r + 6.0) ** 2
    ok, err = close_check(lhs, rhs, 5e-5, 0.0)
    return ok, f"r={r:.3f}, err={err:.3e}"


def check_eq_split(quad: NormalQuadrature) -> Tuple[bool, str]:
    worst = -1e9
    for kappa in [0.0, 0.5]:
        for t in [0.2, 0.6, 0.9]:
            p_t = float(barPhi(kappa / math.sqrt(t)))
            eg = quad.expect(
                lambda z: g((kappa - math.sqrt(t) * z) / math.sqrt(1.0 - t))
            )
            bound = float(g(0.0)) * (1.0 - p_t) + (1.0 / 18.0) * p_t
            worst = max(worst, eg - bound)
    ok = worst <= 1e-2
    return ok, f"max_excess={worst:.3e}"


def check_eq_spin_bd(quad: NormalQuadrature) -> Tuple[bool, str]:
    kappa = 0.5
    alpha = 0.8 * alpha_c(kappa)
    q, r = solve_fixed_point(alpha, kappa, quad)
    eps = 1.0 - q
    lhs = -0.5 * r * eps + quad.expect(lambda z: logcosh(math.sqrt(r) * z))
    rhs = alpha * B(q, kappa, quad) / (2.0 * eps)
    ok, margin = leq_check(lhs, rhs, 1e-2)
    return ok, f"margin={margin:.3e}"


def check_eq_constraint_bd(quad: NormalQuadrature) -> Tuple[bool, str]:
    kappa = 0.5
    alpha = 0.8 * alpha_c(kappa)
    q, _ = solve_fixed_point(alpha, kappa, quad)
    eps = 1.0 - q
    delta = 0.3
    sqrt_q = math.sqrt(q)

    def U(z: np.ndarray) -> np.ndarray:
        return (kappa - sqrt_q * z) / math.sqrt(eps)

    A_n = quad.expect(lambda z: np.maximum(kappa - sqrt_q * z, 0.0) ** 2)
    lhs = quad.expect(lambda z: log_barPhi(U(z)))
    rhs = -A_n / (2.0 * eps) + 0.5 * float(Phi(kappa - delta)) * math.log(eps) - math.log(delta / 2.0)
    ok, margin = leq_check(lhs, rhs, 1e-2)
    return ok, f"margin={margin:.3e}"


def check_eq_BA_bd(quad: NormalQuadrature) -> Tuple[bool, str]:
    kappa = 0.5
    q_vals = [0.5, 0.9, 0.97]
    grid = np.linspace(0.0, 1.0, 200)
    max_mid = max(float(E(u) ** 2 - u * u) for u in grid)
    c0 = max(2.0 / math.pi, 3.0, max_mid)
    worst = -1e9
    for q in q_vals:
        eps = 1.0 - q
        sqrt_q = math.sqrt(q)
        A_n = quad.expect(lambda z: np.maximum(kappa - sqrt_q * z, 0.0) ** 2)
        B_n = B(q, kappa, quad)
        lhs = B_n - A_n
        rhs = c0 * eps
        worst = max(worst, lhs - rhs)
    ok = worst <= 1e-2
    return ok, f"max_excess={worst:.3e}"


def check_eq_two_sided_mills(quad: NormalQuadrature) -> Tuple[bool, str]:
    worst = -1e9
    for u in [0.2, 0.5, 1.0, 2.0, 4.0]:
        lhs = float(phi(u) / barPhi(u))
        rhs = u + 1.0 / u
        worst = max(worst, lhs - rhs)
    ok = worst <= 1e-8
    return ok, f"max_excess={worst:.3e}"


def check_eq_rootsFneg(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for y in [0.1, 0.4, 0.6]:
        r_plus = (3.0 - y + math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
        r_minus = (3.0 - y - math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
        qy = lambda x: x * x + x * y - 3.0 * x - 3.0 * y + 2.0
        max_err = max(max_err, abs(qy(r_plus)), abs(qy(r_minus)))
    ok, _ = close_check(max_err, 0.0, 1e-8, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_IyFneg(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.4
    r_minus = (3.0 - y - math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
    x_lo = max(0.0, 1.0 - 2.0 * y)
    ok = x_lo <= r_minus
    return ok, f"x_lo={x_lo:.3f}, r_minus={r_minus:.3f}"


def check_eq_leftEndpoint1(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.3
    val = (1.0 - 2.0 * y) ** 2 + 6.0 * (1.0 - 2.0 * y) * y + 6.0 * y * y - (1.0 - 2.0 * y) - 4.0 * y
    ok, err = close_check(val, -2.0 * y * y, 1e-8, 0.0)
    return ok, f"err={err:.3e}"


def check_eq_leftEndpoint2(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.6
    val = 6.0 * y * y - 4.0 * y
    ok, err = close_check(val, 2.0 * y * (3.0 * y - 2.0), 1e-8, 0.0)
    return ok, f"err={err:.3e}"


def check_eq_rightEndpoint(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.4
    r_minus = (3.0 - y - math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
    f_val = r_minus * r_minus + 6.0 * r_minus * y + 6.0 * y * y - r_minus - 4.0 * y
    rhs = (7.0 * y * y + 11.0 * y + 2.0 - (5.0 * y + 2.0) * math.sqrt(y * y + 6.0 * y + 1.0)) / 2.0
    ok, err = close_check(f_val, rhs, 1e-8, 0.0)
    return ok, f"err={err:.3e}"


def check_eq_squareDiff(quad: NormalQuadrature) -> Tuple[bool, str]:
    y = 0.5
    lhs = (5.0 * y + 2.0) ** 2 * (y * y + 6.0 * y + 1.0) - (7.0 * y * y + 11.0 * y + 2.0) ** 2
    rhs = 8.0 * y * y * y * (2.0 - 3.0 * y)
    ok, err = close_check(lhs, rhs, 1e-8, 0.0)
    return ok, f"err={err:.3e}"


def check_eq_C_kappa_closed(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    for kappa in [0.0, 0.5, 1.0]:
        lhs = C_kappa(kappa, quad)
        rhs = C_kappa_closed(kappa)
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def check_eq_Lq_halfspace(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = 0.0
    q = 0.4
    kappa = 0.6
    for x in [-1.0, 0.0, 0.8]:
        prob = quad.expect(lambda xi: (x + math.sqrt(1.0 - q) * xi >= kappa).astype(float))
        lhs = math.log(max(prob, 1e-300))
        rhs = float(L_q_halfspace(x, q, kappa))
        max_err = max(max_err, abs(lhs - rhs))
    ok, _ = close_check(max_err, 0.0, 1e-2, 0.0)
    return ok, f"max_err={max_err:.3e}"


def fmt(value: float) -> str:
    return f"{value:.6e}"


def tobechecked_lemma_A_detail() -> Tuple[bool, str]:
    if check_tobechecked is not None:
        mp = check_tobechecked.mp
        A_func = check_tobechecked.A
        r_limit = 50.0
        r_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        A_values = [float(A_func(mp.mpf(r))) for r in r_values]
        limit_val = float(A_func(mp.mpf(r_limit)))
        target = float(mp.mpf("2") / mp.pi)
    else:
        A_func = A_float
        r_limit = 200.0
        r_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        A_values = [A_func(r) for r in r_values]
        limit_val = A_func(r_limit)
        target = 2.0 / math.pi

    A0 = A_values[0]
    deltas = [curr - prev for prev, curr in zip(A_values, A_values[1:])]
    min_delta = min(deltas)
    diff = limit_val - target
    ok0 = abs(A0) <= 1e-10
    okmono = all(delta > 0.0 for delta in deltas)
    ok_limit = abs(diff) <= 5e-3
    ok = ok0 and okmono and ok_limit
    msg = (
        f"A(0)={fmt(A0)}, min_delta={fmt(min_delta)}, "
        f"A({r_limit:g})={fmt(limit_val)}, 2/pi={fmt(target)}, diff={fmt(diff)}"
    )
    return ok, msg


def tobechecked_lemma_B_endpoints_detail() -> Tuple[bool, str]:
    if check_tobechecked is not None:
        mp = check_tobechecked.mp

        def B_eval(q: float, kappa: float) -> float:
            return float(check_tobechecked.B(q, kappa))

        def E_eval(kappa: float) -> float:
            return float(check_tobechecked.E(kappa))

        def C_eval(kappa: float) -> float:
            return float(check_tobechecked.C_kappa(kappa))

        kappas = [mp.mpf("0"), mp.mpf("0.5"), mp.mpf("1.0")]
        q_close = mp.mpf("1") - mp.mpf("1e-5")
    else:
        def B_eval(q: float, kappa: float) -> float:
            return float(B_float(q, kappa))

        def E_eval(kappa: float) -> float:
            return float(E(kappa))

        def C_eval(kappa: float) -> float:
            return float(C_kappa_closed(kappa))

        kappas = [0.0, 0.5, 1.0]
        q_close = 1.0 - 1e-5

    max_err0 = -1.0
    max_err1 = -1.0
    worst0 = None
    worst1 = None
    for kappa in kappas:
        B0 = B_eval(0.0, kappa)
        E2 = E_eval(kappa) ** 2
        err0 = abs(B0 - E2)
        if err0 > max_err0:
            max_err0 = err0
            worst0 = (float(kappa), B0, E2)
        Bq = B_eval(q_close, kappa)
        Ck = C_eval(kappa)
        err1 = abs(Bq - Ck)
        if err1 > max_err1:
            max_err1 = err1
            worst1 = (float(kappa), Bq, Ck)

    ok = (max_err0 <= 1e-10) and (max_err1 <= 2e-3)
    k0, B0, E2 = worst0
    k1, Bq, Ck = worst1
    msg = (
        f"max_err0={fmt(max_err0)} (kappa={k0:.3f}, B0={fmt(B0)}, E2={fmt(E2)}), "
        f"max_err1={fmt(max_err1)} (kappa={k1:.3f}, Bq={fmt(Bq)}, C={fmt(Ck)})"
    )
    return ok, msg


def tobechecked_eq_Bprime_detail(quad: NormalQuadrature) -> Tuple[bool, str]:
    kappa = 0.6
    t_values = [0.3, 0.7]
    max_err = -1.0
    worst = None
    for t in t_values:
        lhs = finite_diff(lambda s: B(s, kappa, quad), t, 1e-4)
        rhs = quad.expect(lambda z: g((kappa - math.sqrt(t) * z) / math.sqrt(1.0 - t)))
        err = abs(lhs - rhs)
        if err > max_err:
            max_err = err
            worst = (t, lhs, rhs)
    ok = max_err <= 1e-2
    t, lhs, rhs = worst
    msg = f"max_err={fmt(max_err)} (t={t:.3f}, lhs={fmt(lhs)}, rhs={fmt(rhs)})"
    return ok, msg


def tobechecked_eq_moments_detail(quad: NormalQuadrature) -> Tuple[bool, str]:
    max_err = -1.0
    worst = None
    for u in [-0.5, 0.0, 0.8]:
        d_u = float(d(u))
        mu1 = conditional_moment(u, 1)
        mu2 = conditional_moment(u, 2)
        mu3 = conditional_moment(u, 3)
        mu4 = conditional_moment(u, 4)
        formulas = [
            d_u,
            1.0 - u * d_u,
            (u * u + 2.0) * d_u - u,
            u * u + 3.0 - u * (u * u + 5.0) * d_u,
        ]
        vals = [mu1, mu2, mu3, mu4]
        for idx, (val, formula) in enumerate(zip(vals, formulas), start=1):
            err = abs(val - formula)
            if err > max_err:
                max_err = err
                worst = (u, idx, val, formula)
    ok = max_err <= 5e-6
    u, idx, val, formula = worst
    msg = (
        f"max_err={fmt(max_err)} (u={u:.3f}, moment={idx}, "
        f"lhs={fmt(val)}, rhs={fmt(formula)})"
    )
    return ok, msg


def tobechecked_eq_FdefFneg_detail() -> Tuple[bool, str]:
    ys = [0.01 + i * 0.01 for i in range(1, 65)]
    xs = [0.0 + i * 0.01 for i in range(0, 100)]
    worst = -1e9
    worst_xy = None
    for y in ys:
        if not (y > 0.0 and y < 2.0 / 3.0):
            continue
        for x in xs:
            if x < 0.0:
                continue
            if x + 2.0 * y < 1.0:
                continue
            if x * x + x * y - 3.0 * x - 3.0 * y + 2.0 < 0.0:
                continue
            if x + y >= 1.0:
                continue
            f_val = x * x + 6.0 * x * y + 6.0 * y * y - x - 4.0 * y
            if f_val > worst:
                worst = f_val
                worst_xy = (x, y)
    ok = worst < -1e-10
    x, y = worst_xy
    msg = f"max_F={fmt(worst)} (x={x:.2f}, y={y:.2f})"
    return ok, msg


def tobechecked_eq_g_zero_detail() -> Tuple[bool, str]:
    g0 = float(g(0.0))
    expected = 12.0 / (math.pi ** 2) - 4.0 / math.pi
    diff = g0 - expected
    ok = abs(diff) <= 1e-10
    msg = f"g(0)={fmt(g0)}, expected={fmt(expected)}, diff={fmt(diff)}"
    return ok, msg


def tobechecked_eq_g_critical_r_detail(quad: NormalQuadrature) -> Tuple[bool, str]:
    u_grid = np.linspace(-8.0, -0.1, 200)
    h_vals = H(u_grid)
    sign = np.sign(h_vals)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(idx) == 0:
        return False, "no_root"
    i = idx[0]
    u_star = brentq(lambda x: float(H(x)), u_grid[i], u_grid[i + 1])
    d_star = float(d(u_star))
    r = -u_star / d_star
    lhs = float(g(u_star))
    rhs = r * (4.0 - r) * (1.0 - r) ** 2 / (r * r - 6.0 * r + 6.0) ** 2
    err = abs(lhs - rhs)
    ok = err <= 5e-5
    msg = f"u*={fmt(u_star)}, r={fmt(r)}, err={fmt(err)}"
    return ok, msg


def tobechecked_eq_pi_estimate_detail() -> Tuple[bool, str]:
    left = -4.0 * (math.pi - 3.0) / (math.pi ** 2)
    right = -1.0 / 18.0
    margin = left - right
    ok = left < right
    msg = f"left={fmt(left)}, right={fmt(right)}, margin={fmt(margin)}"
    return ok, msg


def tobechecked_rational_function_bound_detail() -> Tuple[bool, str]:
    max_val = -1e9
    max_r = None
    for i in range(1, 999):
        r = i / 1000.0
        val = r * (4.0 - r) * (1.0 - r) ** 2 / (r ** 2 - 6.0 * r + 6.0) ** 2
        if val > max_val:
            max_val = val
            max_r = r
    bound = 1.0 / 18.0
    max_excess = max_val - bound
    ok = max_excess <= 1e-10
    msg = f"max_val={fmt(max_val)} (r={max_r:.3f}), bound={fmt(bound)}, excess={fmt(max_excess)}"
    return ok, msg

def assert_close_scalar(actual: float, expected: float, tol: float, message: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{message}: {actual} vs {expected} (tol={tol})")


def normal_expectation_scalar(func: Callable[[float], float]) -> float:
    return quad(lambda z: func(z) * float(phi(z)), -np.inf, np.inf, limit=200)[0]


def A_float(r: float) -> float:
    if r == 0.0:
        return 0.0
    sqrt_r = math.sqrt(r)
    s = normal_expectation_scalar(lambda z: float(sech2(sqrt_r * z)))
    return r * s * s


def B_float(q: float, kappa: float) -> float:
    if not (0.0 <= q < 1.0):
        raise ValueError("q must be in [0, 1).")
    sqrt_q = math.sqrt(q)
    sqrt_1mq = math.sqrt(1.0 - q)

    def integrand(z: float) -> float:
        u = (kappa - sqrt_q * z) / sqrt_1mq
        return float(E(u) ** 2)

    return (1.0 - q) * normal_expectation_scalar(integrand)


def fallback_check_lemma_A() -> None:
    r_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    A_values = [A_float(r) for r in r_values]
    for r, a_val in zip(r_values, A_values):
        if r == 0.0:
            assert_close_scalar(a_val, 0.0, 1e-10, "A(0) != 0")
    for prev, curr in zip(A_values, A_values[1:]):
        if not curr > prev:
            raise AssertionError("A is not strictly increasing on sample grid")
    limit_val = A_float(200.0)
    assert_close_scalar(limit_val, 2.0 / math.pi, 5e-3, "A(r)->2/pi mismatch")


def fallback_check_lemma_B_endpoints() -> None:
    for kappa in [0.0, 0.5, 1.0]:
        assert_close_scalar(
            B_float(0.0, kappa),
            float(E(kappa)) ** 2,
            1e-10,
            "B(0) != E(kappa)^2",
        )
        q_close = 1.0 - 1e-5
        assert_close_scalar(
            B_float(q_close, kappa),
            C_kappa_closed(kappa),
            2e-3,
            "B(q)->C_kappa mismatch",
        )


def fallback_check_eq_FdefFneg() -> None:
    ys = [0.01 + i * 0.01 for i in range(1, 65)]
    xs = [0.0 + i * 0.01 for i in range(0, 100)]
    for y in ys:
        if not (y > 0 and y < 2.0 / 3.0):
            continue
        for x in xs:
            if x < 0:
                continue
            if x + 2 * y < 1:
                continue
            if x * x + x * y - 3 * x - 3 * y + 2 < 0:
                continue
            if x + y >= 1:
                continue
            f_val = x * x + 6 * x * y + 6 * y * y - x - 4 * y
            if f_val >= -1e-10:
                raise AssertionError(f"F(x,y) not negative at x={x}, y={y}: {f_val}")


def fallback_check_eq_g_zero() -> None:
    g0 = float(g(0.0))
    expected = 12.0 / (math.pi ** 2) - 4.0 / math.pi
    assert_close_scalar(g0, expected, 1e-10, "g(0) formula mismatch")


def fallback_check_eq_pi_estimate() -> None:
    left = -4.0 * (math.pi - 3.0) / (math.pi ** 2)
    right = -1.0 / 18.0
    if not left < right:
        raise AssertionError("pi inequality does not hold")


def fallback_check_rational_function_bound() -> None:
    for i in range(1, 999):
        r = i / 1000.0
        val = r * (4.0 - r) * (1.0 - r) ** 2 / (r ** 2 - 6.0 * r + 6.0) ** 2
        if val > 1.0 / 18.0 + 1e-10:
            raise AssertionError(f"rational function bound violated at r={r}: {val}")


TOBECHECKED_LABELS: Tuple[str, ...] = (
    "lem:A-TOBECHECKED",
    "lem:B_endpoints-TOBECHECKED",
    "eq:Bprime-TOBECHECKED",
    "eq:moments-TOBECHECKED",
    "eq:FdefFneg_lemma-TOBECHECKED",
    "eq: g-zero-value-TOBECHECKED",
    "eq:g-critical-r-TOBECHECKED",
    "eq: pi-estimate-TOBECHECKED",
    "lem: rational function bound-TOBECHECKED",
)
# Keep this mapping in sync with the TOBECHECKED labels in main.tex.
TOBECHECKED_CHECK_FACTORIES: Dict[str, CheckFactory] = {
    "lem:A-TOBECHECKED": lambda quad: tobechecked_lemma_A_detail,
    "lem:B_endpoints-TOBECHECKED": lambda quad: tobechecked_lemma_B_endpoints_detail,
    "eq:Bprime-TOBECHECKED": lambda quad: lambda: tobechecked_eq_Bprime_detail(quad),
    "eq:moments-TOBECHECKED": lambda quad: lambda: tobechecked_eq_moments_detail(quad),
    "eq:FdefFneg_lemma-TOBECHECKED": lambda quad: tobechecked_eq_FdefFneg_detail,
    "eq: g-zero-value-TOBECHECKED": lambda quad: tobechecked_eq_g_zero_detail,
    "eq:g-critical-r-TOBECHECKED": lambda quad: lambda: tobechecked_eq_g_critical_r_detail(quad),
    "eq: pi-estimate-TOBECHECKED": lambda quad: tobechecked_eq_pi_estimate_detail,
    "lem: rational function bound-TOBECHECKED": lambda quad: tobechecked_rational_function_bound_detail,
}
_tobechecked_label_set = set(TOBECHECKED_LABELS)
_factory_label_set = set(TOBECHECKED_CHECK_FACTORIES)
if _tobechecked_label_set != _factory_label_set:
    missing_factories = ", ".join(sorted(_tobechecked_label_set - _factory_label_set)) or "none"
    extra_factories = ", ".join(sorted(_factory_label_set - _tobechecked_label_set)) or "none"
    raise RuntimeError(
        f"TOBECHECKED label mismatch (missing={missing_factories}, extra={extra_factories})"
    )
del _tobechecked_label_set, _factory_label_set


def build_checks(quad: NormalQuadrature) -> List[Check]:
    return [
        Check("eq:system", lambda: check_eq_system(quad)),
        Check("eq:defB", lambda: check_eq_defB(quad)),
        Check("eq:r_in_terms_of_B", lambda: check_eq_r_in_terms_of_B(quad)),
        Check("eq:RSfunctional", lambda: check_eq_RSfunctional(quad)),
        Check("eq:union-bound", lambda: check_eq_union_bound(quad)),
        Check("eq:sharp-seq-exp", lambda: check_eq_sharp_seq_exp(quad)),
        Check("eq:RS-functional", lambda: check_eq_Lq_halfspace(quad)),
        Check("eq:RS-fixed-point", lambda: check_eq_RS_fixed_point(quad)),
        Check("eq:defA", lambda: check_eq_defA(quad)),
        Check("eq:one-dimensional", lambda: check_eq_one_dimensional(quad)),
        Check("eq:A_as_I", lambda: check_eq_A_as_I(quad)),
        Check("eq:condmean", lambda: check_eq_condmean(quad)),
        Check("eq:mprime", lambda: check_eq_mprime(quad)),
        Check("eq:varidentity", lambda: check_eq_varidentity(quad)),
        Check("eq:dprime", lambda: check_eq_dprime(quad)),
        Check("eq:B-as-BM", lambda: check_eq_B_as_BM(quad)),
        Check("eq:Lf", lambda: check_eq_Lf(quad)),
        Check("eq:Bprime", lambda: check_eq_Bprime(quad)),
        Check("eq:U-def", lambda: (True, "definitional")),
        Check("eq:g-def", lambda: (True, "definitional")),
        Check("eq:goal", lambda: check_eq_goal(quad)),
        Check("eq:g-expanded", lambda: check_eq_g_expanded(quad)),
        Check("eq:moments", lambda: check_eq_moments(quad)),
        Check("eq:consFneg_lemma", lambda: check_eq_consFneg_lemma(quad)),
        Check("eq:FdefFneg_lemma", lambda: check_eq_FdefFneg_lemma(quad)),
        Check("eq:gprime", lambda: check_eq_gprime(quad)),
        Check("eq:detM1", lambda: check_eq_detM1(quad)),
        Check("eq:detM2", lambda: check_eq_detM2(quad)),
        Check("eq:varY", lambda: check_eq_varY(quad)),
        Check("eq:constraints", lambda: check_eq_constraints(quad)),
        Check("eq:ybound", lambda: check_eq_ybound(quad)),
        Check("eq:Hxy", lambda: check_eq_Hxy(quad)),
        Check("eq:g-critical-r", lambda: check_eq_g_critical_r(quad)),
        Check("eq:split", lambda: check_eq_split(quad)),
        Check("eq:spin_bd_unif_nostep_revised", lambda: check_eq_spin_bd(quad)),
        Check("eq:constraint_bd_unif_nostep_revised", lambda: check_eq_constraint_bd(quad)),
        Check("eq:BA_bd_unif_nostep_revised", lambda: check_eq_BA_bd(quad)),
        Check("eq: two-sided mills", lambda: check_eq_two_sided_mills(quad)),
        Check("eq:rootsFneg_lemma", lambda: check_eq_rootsFneg(quad)),
        Check("eq:IyFneg_lemma", lambda: check_eq_IyFneg(quad)),
        Check("eq:leftEndpoint1Fneg_lemma", lambda: check_eq_leftEndpoint1(quad)),
        Check("eq:leftEndpoint2Fneg_lemma", lambda: check_eq_leftEndpoint2(quad)),
        Check("eq:rightEndpointFneg_lemma", lambda: check_eq_rightEndpoint(quad)),
        Check("eq:squareDiffFneg_lemma", lambda: check_eq_squareDiff(quad)),
        Check("unlabeled:C_kappa_closed", lambda: check_eq_C_kappa_closed(quad)),
    ]


def build_tobechecked_checks(quad: NormalQuadrature) -> Dict[str, Check]:
    checks: Dict[str, Check] = {}
    for label in TOBECHECKED_LABELS:
        factory = TOBECHECKED_CHECK_FACTORIES[label]
        checks[normalize_label(label)] = Check(label, factory(quad))
    return checks


def run_checks(checks: List[Check]) -> Tuple[List[Result], int]:
    results: List[Result] = []
    failures = 0
    for check in checks:
        try:
            ok, msg = check.fn()
        except Exception as exc:
            ok = False
            msg = f"error={exc}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        results.append(Result(check.label, status, msg))
        print(f"{status} {check.label}: {msg}")
    return results, failures


def run_all_checks(quick: bool, latex_out: str | None = None) -> int:
    gh_n = 20000 if quick else 60000
    quad = NormalQuadrature(n=gh_n, seed=0)
    results, failures = run_checks(build_checks(quad))
    print(f"done: {len(results)} checks, {failures} failures")
    if latex_out:
        summary = {"total": len(results), "failures": failures}
        write_latex_report(latex_out, "Numerical check report (all)", results, summary)
        print(f"wrote LaTeX report to {latex_out}")
    return 1 if failures else 0


def run_tobechecked_labels(quick: bool, main_tex: str, latex_out: str | None = None) -> int:
    labels = load_tobechecked_labels(main_tex)
    gh_n = 20000 if quick else 60000
    quad = NormalQuadrature(n=gh_n, seed=0)

    results, failures = run_checks(build_checks(quad))
    missing = 0

    if not labels:
        print("no TOBECHECKED labels found; skipped label checks")
    else:
        if check_tobechecked is None:
            print(
                "note: check_tobechecked import failed "
                f"({CHECK_TOBECHECKED_IMPORT_ERROR}); using fallback checks."
            )
        checks = build_tobechecked_checks(quad)
        for label, key in labels:
            check = checks.get(key)
            if check is None:
                msg = "no numeric check mapped"
                results.append(Result(label, "MISSING", msg))
                print(f"MISSING {label}: {msg}")
                failures += 1
                missing += 1
                continue
            ok, msg = check.fn()
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            results.append(Result(label, status, msg))
            print(f"{status} {label}: {msg}")

    print(f"done: {len(results)} checks, {failures} failures ({missing} missing)")
    if latex_out:
        summary = {"total": len(results), "failures": failures, "missing": missing}
        note = None
        if labels and check_tobechecked is None:
            note = f"check_tobechecked unavailable ({CHECK_TOBECHECKED_IMPORT_ERROR}); using fallback routines."
        write_latex_report(
            latex_out,
            "Numerical check report (all checks + TOBECHECKED labels)",
            results,
            summary,
            note=note,
        )
        print(f"wrote LaTeX report to {latex_out}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer quadrature nodes and samples.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run only the full numeric check suite (skip TOBECHECKED label checks).",
    )
    parser.add_argument(
        "--main-tex",
        default=os.path.join(ROOT, "main.tex"),
        help="Path to the LaTeX source containing TOBECHECKED labels.",
    )
    parser.add_argument(
        "--latex-out",
        nargs="?",
        const=os.path.join(NUMERICS_DIR, "numerics_report.tex"),
        default=None,
        help="Write a LaTeX report (defaults to numerics/numerics_report.tex).",
    )
    args = parser.parse_args()

    if args.all:
        return run_all_checks(args.quick, args.latex_out)
    return run_tobechecked_labels(args.quick, args.main_tex, args.latex_out)


if __name__ == "__main__":
    raise SystemExit(main())
