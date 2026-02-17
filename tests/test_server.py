"""Comprehensive test suite for server.py LaTeX expression evaluation.

Tests server.py via its stdin/stdout JSON-line protocol.
Expressions that are known to fail with sympy's parse_latex are marked
with ``pytest.mark.xfail`` so the suite stays green while documenting gaps.
"""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SERVER_PY = str(Path(__file__).resolve().parent.parent / "server.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class ServerProcess:
    """Manages a running server.py process for testing."""

    def __init__(self):
        self._id = 0
        self._start_process()

    def _start_process(self):
        self.proc = subprocess.Popen(
            [sys.executable, SERVER_PY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ready = self._read_line(timeout=10)
        assert ready is not None, "server.py did not produce output"
        msg = json.loads(ready)
        assert msg.get("status") == "ready", f"Expected ready, got: {ready}"

    def evaluate(self, expression: str, timeout: float = 10) -> dict:
        """Send *expression* to the server and return the response dict."""
        self._id += 1
        req = json.dumps({"id": str(self._id), "expression": expression}) + "\n"
        try:
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._restart()
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
        line = self._read_line(timeout=timeout)
        if line is None:
            self._restart()
            raise TimeoutError(f"server.py timed out on: {expression!r}")
        return json.loads(line)

    def _restart(self):
        """Kill the current process and start a fresh one."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self._start_process()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    # -- helpers --

    def _read_line(self, timeout: float) -> str | None:
        result: list[str | None] = [None]

        def _target():
            result[0] = self.proc.stdout.readline()

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        val = result[0]
        return val.strip() if val else None


@pytest.fixture(scope="module")
def server():
    srv = ServerProcess()
    yield srv
    srv.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_pass(server, expression: str):
    """Assert that *expression* evaluates without error."""
    resp = server.evaluate(expression)
    assert "error" not in resp, (
        f"Expected success for {expression!r}, got error: {resp.get('error')}"
    )
    assert "result" in resp


def assert_error(server, expression: str):
    """Assert that *expression* returns an error."""
    resp = server.evaluate(expression)
    assert "error" in resp, (
        f"Expected error for {expression!r}, got result: {resp.get('result')}"
    )


# ===================================================================
# PASSING TESTS — expressions that work with parse_latex
# ===================================================================


class TestBasicArithmetic:
    def test_addition(self, server):
        assert_pass(server, "1+1")

    def test_subtraction(self, server):
        assert_pass(server, "10-3")

    def test_multiplication_times(self, server):
        assert_pass(server, "3 \\times 4")

    def test_multiplication_cdot(self, server):
        assert_pass(server, "3 \\cdot 4")

    def test_division_div(self, server):
        assert_pass(server, "10 \\div 2")

    def test_large_addition(self, server):
        assert_pass(server, "999+1")

    def test_negative_result(self, server):
        assert_pass(server, "5-10")

    def test_zero(self, server):
        assert_pass(server, "0+0")


class TestFractions:
    def test_frac(self, server):
        assert_pass(server, "\\frac{1}{2}")

    def test_dfrac(self, server):
        assert_pass(server, "\\dfrac{3}{4}")

    def test_nested_frac(self, server):
        assert_pass(server, "\\frac{\\frac{1}{2}}{\\frac{3}{4}}")

    def test_frac_addition(self, server):
        assert_pass(server, "\\frac{1}{2} + \\frac{1}{3}")

    def test_frac_subtraction(self, server):
        assert_pass(server, "\\frac{5}{6} - \\frac{1}{6}")

    def test_frac_multiplication(self, server):
        assert_pass(server, "\\frac{2}{3} \\times \\frac{3}{4}")


class TestPowers:
    def test_square(self, server):
        assert_pass(server, "3^{2}")

    def test_cube(self, server):
        assert_pass(server, "2^{3}")

    def test_negative_exponent(self, server):
        assert_pass(server, "2^{-1}")

    def test_fractional_exponent(self, server):
        assert_pass(server, "8^{\\frac{1}{3}}")

    def test_nested_powers(self, server):
        assert_pass(server, "2^{2^{3}}")

    def test_tower_of_powers(self, server):
        assert_pass(server, "x^{2^{3^{4}}}")

    def test_negative_squared(self, server):
        assert_pass(server, "(-3)^{2}")

    def test_negative_cubed(self, server):
        assert_pass(server, "(-2)^{3}")


class TestRoots:
    def test_sqrt(self, server):
        assert_pass(server, "\\sqrt{4}")

    def test_sqrt_large(self, server):
        assert_pass(server, "\\sqrt{144}")

    def test_nth_root(self, server):
        assert_pass(server, "\\sqrt[3]{27}")

    def test_4th_root(self, server):
        assert_pass(server, "\\sqrt[4]{16}")

    def test_nested_sqrt(self, server):
        assert_pass(server, "\\sqrt{\\sqrt{16}}")

    def test_triple_nested_sqrt(self, server):
        assert_pass(server, "\\sqrt{\\sqrt{\\sqrt{256}}}")

    def test_nested_nth_roots(self, server):
        assert_pass(server, "\\sqrt[3]{\\sqrt[4]{\\sqrt[5]{x}}}")


class TestTrigonometry:
    def test_sin(self, server):
        assert_pass(server, "\\sin(0)")

    def test_cos(self, server):
        assert_pass(server, "\\cos(0)")

    def test_tan(self, server):
        assert_pass(server, "\\tan(0)")

    def test_csc(self, server):
        assert_pass(server, "\\csc(\\frac{\\pi}{2})")

    def test_sec(self, server):
        assert_pass(server, "\\sec(0)")

    def test_cot(self, server):
        assert_pass(server, "\\cot(\\frac{\\pi}{4})")

    def test_sin_pi(self, server):
        assert_pass(server, "\\sin(\\pi)")

    def test_cos_pi(self, server):
        assert_pass(server, "\\cos(\\pi)")

    def test_sin_squared_plus_cos_squared(self, server):
        assert_pass(server, "\\sin^{2}(x) + \\cos^{2}(x)")

    def test_sin_times_cos(self, server):
        assert_pass(server, "\\sin(x)\\cos(x)")


class TestInverseTrig:
    def test_arcsin(self, server):
        assert_pass(server, "\\arcsin(1)")

    def test_arccos(self, server):
        assert_pass(server, "\\arccos(1)")

    def test_arctan(self, server):
        assert_pass(server, "\\arctan(1)")


class TestHyperbolic:
    def test_sinh(self, server):
        assert_pass(server, "\\sinh(0)")

    def test_cosh(self, server):
        assert_pass(server, "\\cosh(0)")

    def test_tanh(self, server):
        assert_pass(server, "\\tanh(0)")


class TestLogarithms:
    def test_ln(self, server):
        assert_pass(server, "\\ln(e)")

    def test_log(self, server):
        assert_pass(server, "\\log(10)")

    def test_log_base(self, server):
        assert_pass(server, "\\log_{2}(8)")

    def test_log_base10(self, server):
        assert_pass(server, "\\log_{10}(100)")

    def test_ln_e_to_x(self, server):
        assert_pass(server, "\\ln(e^x)")


class TestExponential:
    def test_exp(self, server):
        assert_pass(server, "\\exp(0)")

    def test_e_to_power(self, server):
        assert_pass(server, "e^{2}")

    def test_e_to_i_theta(self, server):
        assert_pass(server, "e^{i\\theta}")


class TestAbsoluteValue:
    def test_abs_positive(self, server):
        assert_pass(server, "|-5|")

    def test_abs_expression(self, server):
        assert_pass(server, "|3-7|")


class TestFactorial:
    def test_factorial(self, server):
        assert_pass(server, "5!")

    def test_factorial_large(self, server):
        assert_pass(server, "10!")

    def test_factorial_fraction(self, server):
        assert_pass(server, "\\frac{n!}{k!(n-k)!}")


class TestBinomial:
    def test_binom(self, server):
        assert_pass(server, "\\binom{5}{2}")

    def test_binom_large(self, server):
        assert_pass(server, "\\binom{10}{3}")

    def test_binom_in_expression(self, server):
        assert_pass(server, "\\binom{n}{k} p^k (1-p)^{n-k}")


class TestSummation:
    def test_sum_basic(self, server):
        assert_pass(server, "\\sum_{i=1}^{10} i")

    def test_sum_squares(self, server):
        assert_pass(server, "\\sum_{i=1}^{5} i^{2}")

    @pytest.mark.xfail(reason="symbolic infinite series causes 'Cannot convert symbols to int'")
    def test_sum_infinite_series(self, server):
        assert_pass(server, "\\sum_{n=0}^{\\infty} \\frac{x^n}{n!}")

    def test_double_sum(self, server):
        assert_pass(server, "\\sum_{i=1}^{n} \\sum_{j=1}^{m} a_{ij}")


class TestProduct:
    def test_prod_basic(self, server):
        assert_pass(server, "\\prod_{i=1}^{5} i")

    @pytest.mark.xfail(reason="symbolic upper bound causes evaluation error")
    def test_prod_fraction(self, server):
        assert_pass(server, "\\prod_{k=1}^{n} \\frac{k}{k+1}")


class TestIntegral:
    def test_integral_basic(self, server):
        assert_pass(server, "\\int_{0}^{1} x \\, dx")

    def test_integral_power(self, server):
        assert_pass(server, "\\int_{0}^{1} x^2 \\, dx")

    def test_integral_trig(self, server):
        assert_pass(server, "\\int_{0}^{\\pi} \\sin(x) \\, dx")

    def test_integral_no_bounds(self, server):
        assert_pass(server, "\\int f(x) \\, dx")


class TestLimit:
    def test_limit_basic(self, server):
        assert_pass(server, "\\lim_{x \\to \\infty} \\frac{1}{x}")

    def test_limit_sinc(self, server):
        assert_pass(server, "\\lim_{x \\to 0} \\frac{\\sin x}{x}")

    def test_limit_euler(self, server):
        assert_pass(server, "\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n")


class TestFloorCeiling:
    def test_floor(self, server):
        assert_pass(server, "\\lfloor 3.7 \\rfloor")

    def test_ceiling(self, server):
        assert_pass(server, "\\lceil 3.2 \\rceil")

    def test_floor_fraction(self, server):
        assert_pass(server, "\\lfloor \\frac{n}{2} \\rfloor")


class TestConstants:
    def test_pi(self, server):
        assert_pass(server, "\\pi")

    def test_e(self, server):
        assert_pass(server, "e")

    def test_pi_times_n(self, server):
        assert_pass(server, "2\\pi")

    def test_infty(self, server):
        assert_pass(server, "\\infty")


class TestDerivative:
    def test_derivative_basic(self, server):
        assert_pass(server, "\\frac{d}{dx} x^2")

    def test_second_derivative(self, server):
        assert_pass(server, "\\frac{d^2 y}{dx^2}")


class TestGreekLetters:
    def test_alpha_plus_beta(self, server):
        assert_pass(server, "\\alpha + \\beta")

    def test_gamma_function(self, server):
        assert_pass(server, "\\Gamma(\\alpha)")

    def test_zeta_s(self, server):
        assert_pass(server, "\\zeta(s)")

    def test_chi_squared(self, server):
        assert_pass(server, "\\chi^2")

    def test_omega_squared(self, server):
        assert_pass(server, "\\omega^2")

    def test_delta_subscript(self, server):
        assert_pass(server, "\\delta_{ij}")

    def test_theta_subscript(self, server):
        assert_pass(server, "\\theta_0")

    def test_alpha_subscript(self, server):
        assert_pass(server, "\\alpha_{n} + \\beta_{m}")

    def test_mu_pm_sigma(self, server):
        assert_pass(server, "\\mu \\pm \\sigma")


class TestImplicitMultiplication:
    def test_number_times_greek(self, server):
        assert_pass(server, "2\\pi r")

    def test_polynomial(self, server):
        assert_pass(server, "3x^2 + 2x + 1")

    def test_coeff_times_func(self, server):
        assert_pass(server, "2\\sin(x)")

    def test_multi_letter(self, server):
        assert_pass(server, "ab + cd")


class TestComparison:
    def test_leq(self, server):
        assert_pass(server, "a \\leq b")

    def test_geq(self, server):
        assert_pass(server, "a \\geq b")

    def test_neq(self, server):
        assert_pass(server, "a \\neq b")

    def test_approx(self, server):
        assert_pass(server, "a \\approx b")

    def test_equiv(self, server):
        assert_pass(server, "a \\equiv b")

    def test_sim(self, server):
        assert_pass(server, "a \\sim b")

    def test_propto(self, server):
        assert_pass(server, "a \\propto b")

    def test_ll(self, server):
        assert_pass(server, "a \\ll b")

    def test_gg(self, server):
        assert_pass(server, "a \\gg b")

    def test_prec(self, server):
        assert_pass(server, "a \\prec b")

    def test_succ(self, server):
        assert_pass(server, "a \\succ b")

    def test_simeq(self, server):
        assert_pass(server, "a \\simeq b")

    def test_cong(self, server):
        assert_pass(server, "a \\cong b")


class TestSetTheory:
    def test_cup(self, server):
        assert_pass(server, "A \\cup B")

    def test_cap(self, server):
        assert_pass(server, "A \\cap B")

    def test_subset(self, server):
        assert_pass(server, "A \\subset B")

    def test_subseteq(self, server):
        assert_pass(server, "A \\subseteq B")

    def test_supset(self, server):
        assert_pass(server, "A \\supset B")

    def test_supseteq(self, server):
        assert_pass(server, "A \\supseteq B")

    def test_in(self, server):
        assert_pass(server, "x \\in A")

    def test_notin(self, server):
        assert_pass(server, "x \\notin A")

    def test_emptyset(self, server):
        assert_pass(server, "\\emptyset")

    def test_setminus(self, server):
        assert_pass(server, "A \\setminus B")

    def test_times(self, server):
        assert_pass(server, "A \\times B")

    def test_overline(self, server):
        assert_pass(server, "\\overline{A}")

    def test_varnothing(self, server):
        assert_pass(server, "\\varnothing")


class TestLogicSymbols:
    def test_forall(self, server):
        assert_pass(server, "\\forall x")

    def test_exists(self, server):
        assert_pass(server, "\\exists x")

    def test_neg(self, server):
        assert_pass(server, "\\neg P")

    def test_land(self, server):
        assert_pass(server, "P \\land Q")

    def test_lor(self, server):
        assert_pass(server, "P \\lor Q")

    def test_implies(self, server):
        assert_pass(server, "P \\implies Q")

    def test_iff(self, server):
        assert_pass(server, "P \\iff Q")

    def test_leftrightarrow(self, server):
        assert_pass(server, "P \\Leftrightarrow Q")


class TestModular:
    def test_bmod(self, server):
        assert_pass(server, "a \\bmod b")

    def test_pmod(self, server):
        assert_pass(server, "a \\equiv b \\pmod{n}")

    def test_gcd(self, server):
        assert_pass(server, "\\gcd(a, b)")

    def test_mid(self, server):
        assert_pass(server, "a \\mid b")

    def test_nmid(self, server):
        assert_pass(server, "a \\nmid b")


class TestTensor:
    def test_contravariant(self, server):
        assert_pass(server, "T^{\\mu\\nu}")

    def test_covariant(self, server):
        assert_pass(server, "T_{\\mu\\nu}")

    def test_mixed(self, server):
        assert_pass(server, "T^{\\mu}_{\\nu}")

    def test_metric(self, server):
        assert_pass(server, "g_{\\mu\\nu}")

    def test_christoffel(self, server):
        assert_pass(server, "\\Gamma^{\\lambda}_{\\mu\\nu}")

    def test_riemann(self, server):
        assert_pass(server, "R^{\\rho}_{\\sigma\\mu\\nu}")

    def test_levi_civita(self, server):
        assert_pass(server, "\\epsilon_{ijk}")


class TestPhysics:
    def test_hbar(self, server):
        assert_pass(server, "\\hbar")

    def test_e_mc2(self, server):
        assert_pass(server, "E = mc^2")

    def test_dot(self, server):
        assert_pass(server, "\\dot{x}")

    def test_ddot(self, server):
        assert_pass(server, "\\ddot{x}")

    def test_hat(self, server):
        assert_pass(server, "\\hat{x}")

    def test_vec(self, server):
        assert_pass(server, "\\vec{F}")


class TestAccents:
    def test_bar(self, server):
        assert_pass(server, "\\bar{x}")

    def test_tilde(self, server):
        assert_pass(server, "\\tilde{x}")

    def test_check(self, server):
        assert_pass(server, "\\check{x}")

    def test_breve(self, server):
        assert_pass(server, "\\breve{x}")

    def test_acute(self, server):
        assert_pass(server, "\\acute{x}")

    def test_grave(self, server):
        assert_pass(server, "\\grave{x}")

    def test_widehat(self, server):
        assert_pass(server, "\\widehat{xyz}")

    def test_widetilde(self, server):
        assert_pass(server, "\\widetilde{abc}")

    def test_overline_expr(self, server):
        assert_pass(server, "\\overline{x+y}")

    def test_underline(self, server):
        assert_pass(server, "\\underline{x+y}")


class TestComplexNumbers:
    def test_z_complex(self, server):
        assert_pass(server, "z = a + bi")

    def test_conjugate(self, server):
        assert_pass(server, "\\bar{z}")

    def test_re(self, server):
        assert_pass(server, "\\Re(z)")

    def test_im(self, server):
        assert_pass(server, "\\Im(z)")


class TestSpecialSymbols:
    def test_pm(self, server):
        assert_pass(server, "\\pm")

    def test_mp(self, server):
        assert_pass(server, "\\mp")

    def test_circ(self, server):
        assert_pass(server, "\\circ")

    def test_oplus(self, server):
        assert_pass(server, "\\oplus")

    def test_otimes(self, server):
        assert_pass(server, "\\otimes")

    def test_wp(self, server):
        assert_pass(server, "\\wp(z)")


class TestOperators:
    def test_min(self, server):
        assert_pass(server, "\\min(a, b)")

    def test_max(self, server):
        assert_pass(server, "\\max(a, b)")

    def test_det(self, server):
        assert_pass(server, "\\det(\\mathbf{A})")

    def test_mathbf(self, server):
        assert_pass(server, "\\mathbf{A}")

    def test_mathbf_inverse(self, server):
        assert_pass(server, "\\mathbf{A}^{-1}")

    def test_mathbf_transpose(self, server):
        assert_pass(server, "\\mathbf{A}^{T}")

    def test_mathbb_r(self, server):
        assert_pass(server, "\\mathbb{R}")

    def test_mathbb_z(self, server):
        assert_pass(server, "\\mathbb{Z}")

    def test_mathbb_c(self, server):
        assert_pass(server, "\\mathbb{C}")

    def test_mathbb_n(self, server):
        assert_pass(server, "\\mathbb{N}")

    def test_mathbb_q(self, server):
        assert_pass(server, "\\mathbb{Q}")


class TestSpacing:
    def test_thin_space(self, server):
        assert_pass(server, "x\\,y")

    def test_medium_space(self, server):
        assert_pass(server, "x\\:y")

    def test_thick_space(self, server):
        assert_pass(server, "x\\;y")

    def test_neg_thin_space(self, server):
        assert_pass(server, "x\\!y")

    def test_quad_space(self, server):
        assert_pass(server, "x\\quad y")


class TestComplexExpressions:
    def test_quadratic_formula(self, server):
        assert_pass(server, "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}")

    def test_normal_pdf(self, server):
        assert_pass(server, "\\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}")

    def test_binomial_theorem(self, server):
        assert_pass(server, "\\sum_{k=0}^{n} \\binom{n}{k} x^k y^{n-k}")

    def test_continued_frac_with_frac(self, server):
        assert_pass(server, "\\frac{1}{1+\\frac{1}{1+\\frac{1}{1+x}}}")

    def test_partial_derivative(self, server):
        assert_pass(server, "\\frac{\\partial f}{\\partial x}")

    def test_second_partial(self, server):
        assert_pass(server, "\\frac{\\partial^2 f}{\\partial x^2}")

    def test_mixed_partial(self, server):
        assert_pass(server, "\\frac{\\partial^2 f}{\\partial x \\partial y}")

    def test_arrows_in_expression(self, server):
        assert_pass(server, "f: X \\to Y")

    def test_mapsto(self, server):
        assert_pass(server, "x \\mapsto x^2")


class TestAdvancedFunctions:
    def test_gamma(self, server):
        assert_pass(server, "\\Gamma(\\alpha)")

    def test_bessel_j(self, server):
        assert_pass(server, "J_\\nu(x)")

    def test_bessel_y(self, server):
        assert_pass(server, "Y_\\nu(x)")

    def test_legendre(self, server):
        assert_pass(server, "P_n(x)")

    def test_hermite(self, server):
        assert_pass(server, "H_n(x)")

    def test_laguerre(self, server):
        assert_pass(server, "L_n(x)")

    def test_sup(self, server):
        assert_pass(server, "\\sup_{x \\in S} f(x)")

    def test_inf(self, server):
        assert_pass(server, "\\inf_{x \\in S} f(x)")


class TestSeries:
    def test_leibniz_pi(self, server):
        assert_pass(server, "\\sum_{k=0}^{\\infty} \\frac{(-1)^k}{2k+1} = \\frac{\\pi}{4}")

    def test_e_series(self, server):
        assert_pass(server, "e = \\sum_{n=0}^{\\infty} \\frac{1}{n!}")

    def test_sin_taylor(self, server):
        assert_pass(server, "\\sin(x) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n x^{2n+1}}{(2n+1)!}")


# ===================================================================
# EXPECTED FAILURES — expressions that parse_latex cannot handle
# ===================================================================


class TestMatrixEnvironments:
    @pytest.mark.xfail(reason="parse_latex does not support \\begin{pmatrix}")
    def test_pmatrix(self, server):
        assert_pass(server, "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{bmatrix}")
    def test_bmatrix(self, server):
        assert_pass(server, "\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{vmatrix}")
    def test_vmatrix(self, server):
        assert_pass(server, "\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{Vmatrix}")
    def test_Vmatrix(self, server):
        assert_pass(server, "\\begin{Vmatrix} a & b \\\\ c & d \\end{Vmatrix}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{Bmatrix}")
    def test_Bmatrix(self, server):
        assert_pass(server, "\\begin{Bmatrix} a & b \\\\ c & d \\end{Bmatrix}")

    @pytest.mark.xfail(reason="parse_latex does not support column vectors")
    def test_column_vector(self, server):
        assert_pass(server, "\\begin{pmatrix} 1 \\\\ 0 \\\\ 0 \\end{pmatrix}")


class TestPiecewise:
    @pytest.mark.xfail(reason="parse_latex does not support \\begin{cases}")
    def test_cases_abs(self, server):
        assert_pass(server, "\\begin{cases} x & \\text{if } x \\geq 0 \\\\ -x & \\text{if } x < 0 \\end{cases}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{cases}")
    def test_cases_three(self, server):
        assert_pass(server, "f(x) = \\begin{cases} 0 & x < 0 \\\\ 1 & x = 0 \\\\ 2 & x > 0 \\end{cases}")


class TestBigSetOperators:
    @pytest.mark.xfail(reason="parse_latex does not support \\bigcup")
    def test_bigcup(self, server):
        assert_pass(server, "\\bigcup_{i=1}^{n} A_i")

    @pytest.mark.xfail(reason="parse_latex does not support \\bigcap")
    def test_bigcap(self, server):
        assert_pass(server, "\\bigcap_{i=1}^{n} A_i")


class TestStandaloneArrows:
    @pytest.mark.xfail(reason="standalone \\to is not an evaluable expression")
    def test_to(self, server):
        assert_pass(server, "\\to")

    @pytest.mark.xfail(reason="standalone \\rightarrow is not an evaluable expression")
    def test_rightarrow(self, server):
        assert_pass(server, "\\rightarrow")

    @pytest.mark.xfail(reason="standalone \\Rightarrow is not an evaluable expression")
    def test_Rightarrow(self, server):
        assert_pass(server, "\\Rightarrow")


class TestDelimiterFailures:
    def test_evaluated_at(self, server):
        assert_pass(server, "\\left. \\frac{d}{dx} f(x) \\right|_{x=a}")

    @pytest.mark.xfail(reason="mismatched \\left[ \\right) not supported")
    def test_mismatched(self, server):
        assert_pass(server, "\\left[ \\frac{a}{b} \\right)")

    @pytest.mark.xfail(reason="\\left\\{ with \\in and colon not supported")
    def test_set_builder(self, server):
        assert_pass(server, "\\left\\{ x \\in \\mathbb{R} : x > 0 \\right\\}")

    def test_left_right_abs(self, server):
        assert_pass(server, "\\left| x \\right|")

    def test_left_right_norm(self, server):
        assert_pass(server, "\\left\\| x \\right\\|")


class TestNormFailures:
    @pytest.mark.xfail(reason="\\| norm notation not supported")
    def test_norm_vec(self, server):
        assert_pass(server, "\\|\\vec{v}\\|")

    @pytest.mark.xfail(reason="\\| norm notation not supported")
    def test_norm_lp(self, server):
        assert_pass(server, "\\|f\\|_p")


class TestInnerProductFailure:
    @pytest.mark.xfail(reason="\\langle \\rangle not supported as delimiters")
    def test_inner_product(self, server):
        assert_pass(server, "\\langle u, v \\rangle")


class TestSubstackFailure:
    @pytest.mark.xfail(reason="\\substack not supported")
    def test_substack(self, server):
        assert_pass(server, "\\sum_{\\substack{0 \\leq i \\leq m \\\\ 0 < j < n}} P(i,j)")


class TestStackNotationFailure:
    @pytest.mark.xfail(reason="\\overset not supported")
    def test_overset(self, server):
        assert_pass(server, "\\overset{?}{=}")


class TestSpacingFixed:
    def test_backslash_space(self, server):
        assert_pass(server, "x\\ y")


class TestLeadingScriptsFailure:
    @pytest.mark.xfail(reason="leading subscript {}_n not supported")
    def test_leading_subscript(self, server):
        assert_pass(server, "{}_nC_r")

    @pytest.mark.xfail(reason="leading superscript {}^{14} not supported")
    def test_leading_superscript(self, server):
        assert_pass(server, "{}^{14}C")


class TestHypergeometricFailure:
    @pytest.mark.xfail(reason="hypergeometric notation not supported")
    def test_hypergeometric(self, server):
        assert_pass(server, "_2F_1(a, b; c; z)")


class TestEdgeCaseFailures:
    @pytest.mark.xfail(reason="empty string is not a valid expression")
    def test_empty(self, server):
        assert_pass(server, "")

    @pytest.mark.xfail(reason="whitespace is not a valid expression")
    def test_whitespace(self, server):
        assert_pass(server, " ")

    @pytest.mark.xfail(reason="empty group is not a valid expression")
    def test_empty_group(self, server):
        assert_pass(server, "{}")

    @pytest.mark.xfail(reason="double backslash is not a valid expression")
    def test_double_backslash(self, server):
        assert_pass(server, "\\\\")

    def test_display_math(self, server):
        assert_pass(server, "$$x$$")

    def test_inline_math(self, server):
        assert_pass(server, "$x$")

    def test_paren_math(self, server):
        assert_pass(server, "\\(x\\)")


class TestDoubleFactorial:
    def test_double_factorial(self, server):
        assert_pass(server, "(n-1)!!")


# ===================================================================
# NEW PASSING TESTS — Round 2/3 findings
# ===================================================================


class TestAMSMiscSymbols:
    """AMS miscellaneous symbols handled as atomic symbols."""

    def test_ell(self, server):
        assert_pass(server, "\\ell")

    def test_mho(self, server):
        assert_pass(server, "\\mho")

    def test_complement(self, server):
        assert_pass(server, "\\complement")

    def test_nexists(self, server):
        assert_pass(server, "\\nexists")

    def test_eth(self, server):
        assert_pass(server, "\\eth")

    def test_hslash(self, server):
        assert_pass(server, "\\hslash")

    def test_measuredangle(self, server):
        assert_pass(server, "\\measuredangle")

    def test_backprime(self, server):
        assert_pass(server, "\\backprime")


class TestAMSSymbOperators:
    """AMS symbolic operators treated as atomic symbols."""

    def test_blacksquare(self, server):
        assert_pass(server, "\\blacksquare")

    def test_square(self, server):
        assert_pass(server, "\\square")

    def test_lozenge(self, server):
        assert_pass(server, "\\lozenge")

    def test_bigstar(self, server):
        assert_pass(server, "\\bigstar")

    def test_dotplus(self, server):
        assert_pass(server, "\\dotplus")

    def test_ltimes(self, server):
        assert_pass(server, "\\ltimes")

    def test_rtimes(self, server):
        assert_pass(server, "\\rtimes")

    def test_barwedge(self, server):
        assert_pass(server, "\\barwedge")

    def test_intercal(self, server):
        assert_pass(server, "\\intercal")

    def test_divideontimes(self, server):
        assert_pass(server, "\\divideontimes")

    def test_therefore(self, server):
        assert_pass(server, "\\therefore")

    def test_because(self, server):
        assert_pass(server, "\\because")


class TestPrimeNotation:
    """Prime/derivative notation."""

    def test_first_derivative(self, server):
        assert_pass(server, "f'(x)")

    def test_second_derivative_prime(self, server):
        assert_pass(server, "f''(x)")

    def test_product_rule(self, server):
        assert_pass(server, "a'b+ab'")

    def test_double_prime_subscript(self, server):
        assert_pass(server, "y''_1")

    def test_prime_function_subscript(self, server):
        assert_pass(server, "f'_1(x)")


class TestLimitsPlacement:
    """\\limits and \\nolimits placement commands."""

    @pytest.mark.xfail(reason="\\limits not supported by parse_latex")
    def test_limits(self, server):
        assert_pass(server, "\\sum\\limits_{i=1}^{n} i")

    @pytest.mark.xfail(reason="\\nolimits not supported by parse_latex")
    def test_nolimits(self, server):
        assert_pass(server, "\\sum\\nolimits_{i=1}^{n} i")


class TestDaggerNotation:
    def test_dagger(self, server):
        assert_pass(server, "A^{\\dagger}")


class TestContinuedFraction:
    """\\cfrac is preprocessed to \\frac."""

    def test_cfrac_nested(self, server):
        assert_pass(server, "\\cfrac{1}{2+\\cfrac{1}{3+\\cfrac{1}{4}}}")


class TestExoticFunctionsPass:
    """Exotic functions that parse_latex handles correctly."""

    def test_digamma(self, server):
        assert_pass(server, "\\psi(x)")

    def test_beta_function_formula(self, server):
        assert_pass(server, "B(x, y) = \\frac{\\Gamma(x)\\Gamma(y)}{\\Gamma(x+y)}")


class TestQuadrupleDot:
    def test_ddddot(self, server):
        assert_pass(server, "\\ddddot{y}")


class TestManualDoubleIntegral:
    """Manual spacing for double integrals (spacing stripped in preprocessing)."""

    def test_manual_double_integral(self, server):
        assert_pass(server, "\\int\\!\\!\\!\\int f(x,y)\\,dx\\,dy")


class TestEnvironmentFailuresNew:
    """Additional environments not supported by parse_latex."""

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{array} with column spec")
    def test_array(self, server):
        assert_pass(server, "\\begin{array}{cc} 1 & 2 \\\\ 3 & 4 \\end{array}")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{aligned}")
    def test_aligned(self, server):
        assert_pass(server, "\\begin{aligned} x &= 1 \\\\ y &= 2 \\end{aligned}")


# ===================================================================
# NEW EXPECTED FAILURES — Round 2/3 error findings
# ===================================================================


class TestEnvironmentFailures:
    """Environments not supported by parse_latex."""

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{smallmatrix}")
    def test_smallmatrix(self, server):
        assert_pass(server, "\\left(\\begin{smallmatrix} a & b \\\\ c & d \\end{smallmatrix}\\right)")

    @pytest.mark.xfail(reason="parse_latex does not support \\begin{gathered}")
    def test_gathered(self, server):
        assert_pass(server, "\\begin{gathered} x + y = 1 \\\\ x - y = 0 \\end{gathered}")


class TestUnsupportedCommands:
    """Commands that parse_latex rejects with an error."""

    @pytest.mark.xfail(reason="\\wp with semicolons not supported")
    def test_weierstrass_p_with_periods(self, server):
        assert_pass(server, "\\wp(z; \\omega_1, \\omega_2)")

    @pytest.mark.xfail(reason="comma in subscript not supported")
    def test_mittag_leffler(self, server):
        assert_pass(server, "E_{\\alpha,\\beta}(z)")

    @pytest.mark.xfail(reason="\\cfrac[l] alignment option not supported")
    def test_cfrac_left_aligned(self, server):
        assert_pass(server, "\\cfrac[l]{a}{b+c}")

    @pytest.mark.xfail(reason="\\cfrac[r] alignment option not supported")
    def test_cfrac_right_aligned(self, server):
        assert_pass(server, "\\cfrac[r]{a}{b+c}")

    @pytest.mark.xfail(reason="\\colorbox not supported")
    def test_colorbox(self, server):
        assert_pass(server, "\\colorbox{yellow}{$E=mc^2$}")

    def test_text_differential(self, server):
        assert_pass(server, "\\text{d}x")

    def test_bra_notation(self, server):
        assert_pass(server, "\\left\\langle a\\right|")

    def test_triple_dot(self, server):
        assert_pass(server, "\\dddot{x}")

    def test_shorthand_frac(self, server):
        assert_pass(server, "\\frac12")

    @pytest.mark.xfail(reason="inequality chain not supported")
    def test_inequality_chain(self, server):
        assert_pass(server, "-1 \\leq x < 1")

    @pytest.mark.xfail(reason="\\sum with \\mid divisor condition errors")
    def test_sum_divisor(self, server):
        assert_pass(server, "\\sum\\limits_{d \\mid n} d^2")


class TestLargeFactorial:
    def test_factorial_product(self, server):
        assert_pass(server, "24! \\times 24!")


# ===================================================================
# SUSPECT TESTS — commands silently parsed as variable names
# These are the most dangerous failures: no error is raised but the
# result is semantically wrong.
# ===================================================================


class TestOperatornameAsVariable:
    """\\operatorname{...} is silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\operatorname parsed as variable, not operator")
    def test_operatorname_res(self, server):
        resp = server.evaluate("\\operatorname{Res}_{z=0} f(z)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname parsed as variable, not operator")
    def test_operatorname_span(self, server):
        resp = server.evaluate("\\operatorname{span}\\{v_1, v_2\\}")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname parsed as variable, not operator")
    def test_operatorname_hom(self, server):
        resp = server.evaluate("\\operatorname{Hom}(V, W)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname* parsed as variable")
    def test_operatorname_star_argmax(self, server):
        resp = server.evaluate("\\operatorname*{arg\\,max}_{x \\in S} f(x)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname{Li} parsed as variable")
    def test_operatorname_dilogarithm(self, server):
        resp = server.evaluate("\\operatorname{Li}_2(z)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname{erfc} parsed as variable")
    def test_operatorname_erfc(self, server):
        resp = server.evaluate("\\operatorname{erfc}(x)")
        assert "operatorname" not in resp.get("expression", "")


class TestMathFontAsVariable:
    """Math font commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\mathcal parsed as variable")
    def test_mathcal_laplace(self, server):
        resp = server.evaluate("\\mathcal{L}\\{f(t)\\}")
        assert "mathcal" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathcal parsed as variable")
    def test_mathcal_fourier(self, server):
        resp = server.evaluate("\\mathcal{F}\\{f(t)\\}")
        assert "mathcal" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathfrak parsed as variable")
    def test_mathfrak_su2(self, server):
        resp = server.evaluate("\\mathfrak{su}(2)")
        assert "mathfrak" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathfrak parsed as variable")
    def test_mathfrak_sl(self, server):
        resp = server.evaluate("\\mathfrak{sl}(n, \\mathbb{R})")
        assert "mathfrak" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathscr parsed as variable")
    def test_mathscr(self, server):
        resp = server.evaluate("\\mathscr{H}")
        assert "mathscr" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathrm parsed as variable")
    def test_mathrm_differential(self, server):
        resp = server.evaluate("\\mathrm{d}x")
        assert "mathrm" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\boldsymbol parsed as variable")
    def test_boldsymbol_nabla(self, server):
        resp = server.evaluate("\\boldsymbol{\\nabla} \\times \\mathbf{F}")
        assert "boldsymbol" not in resp.get("expression", "")


class TestBraceDecorAsVariable:
    """Brace decoration commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\overbrace parsed as variable")
    def test_overbrace(self, server):
        resp = server.evaluate("\\overbrace{a + b + c}^{\\text{sum}}")
        assert "overbrace" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\underbrace parsed as variable")
    def test_underbrace(self, server):
        resp = server.evaluate("\\underbrace{x_1 + x_2 + \\cdots + x_n}_{n \\text{ terms}}")
        assert "underbrace" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\stackrel parsed as variable")
    def test_stackrel(self, server):
        resp = server.evaluate("\\stackrel{\\text{def}}{=}")
        assert "stackrel" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\xleftarrow parsed as variable")
    def test_xleftarrow(self, server):
        resp = server.evaluate("\\xleftarrow{n+1}")
        assert "xleftarrow" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\xrightarrow parsed as variable")
    def test_xrightarrow(self, server):
        resp = server.evaluate("\\xrightarrow[T]{n \\pm 1}")
        assert "xrightarrow" not in resp.get("expression", "")


class TestCancelAsVariable:
    """Cancel commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\bcancel parsed as variable")
    def test_bcancel(self, server):
        resp = server.evaluate("\\bcancel{x}")
        assert "bcancel" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\xcancel parsed as variable")
    def test_xcancel(self, server):
        resp = server.evaluate("\\xcancel{x}")
        assert "xcancel" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\cancelto parsed as variable")
    def test_cancelto(self, server):
        resp = server.evaluate("\\cancelto{0}{x^2 + 1}")
        assert "cancelto" not in resp.get("expression", "")


class TestPhantomAsVariable:
    """Phantom/smash commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\vphantom parsed as variable")
    def test_vphantom(self, server):
        resp = server.evaluate("\\vphantom{\\frac{a}{b}}")
        assert "vphantom" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\hphantom parsed as variable")
    def test_hphantom(self, server):
        resp = server.evaluate("\\hphantom{xyz}")
        assert "hphantom" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\smash[b] parsed as variable")
    def test_smash_bottom(self, server):
        resp = server.evaluate("\\smash[b]{\\frac{a}{b}}")
        assert "smash" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\smash[t] parsed as variable")
    def test_smash_top(self, server):
        resp = server.evaluate("\\smash[t]{\\sum_{i=1}^{n}}")
        assert "smash" not in resp.get("expression", "")


class TestSidesetAsVariable:
    """\\sideset silently parsed as variable."""

    @pytest.mark.xfail(reason="\\sideset parsed as variable")
    def test_sideset(self, server):
        resp = server.evaluate("\\sideset{_a^b}{_c^d}\\sum")
        assert "sideset" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\sideset parsed as variable")
    def test_sideset_prime(self, server):
        resp = server.evaluate("\\sideset{}{'}\\sum_{k}")
        assert "sideset" not in resp.get("expression", "")


class TestColorAsVariable:
    """Color commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\textcolor parsed as variable")
    def test_textcolor(self, server):
        resp = server.evaluate("\\textcolor{red}{x^2}")
        assert "textcolor" not in resp.get("expression", "")


class TestEquationTagAsVariable:
    """Equation tag commands silently parsed as variable."""

    @pytest.mark.xfail(reason="\\tag parsed as variable")
    def test_tag(self, server):
        resp = server.evaluate("\\tag{*} x = 1")
        assert "tag" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\notag parsed as variable")
    def test_notag(self, server):
        resp = server.evaluate("\\notag")
        assert "notag" not in resp.get("expression", "")


class TestMultipleIntegralsAsVariable:
    """Multiple integral commands silently parsed as variable."""

    @pytest.mark.xfail(reason="\\iint parsed as variable")
    def test_double_integral(self, server):
        resp = server.evaluate("\\iint_D f(x,y) \\, dA")
        assert "iint" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\iiint parsed as variable")
    def test_triple_integral(self, server):
        resp = server.evaluate("\\iiint_V f \\, dV")
        assert "iiint" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\oint parsed as variable")
    def test_contour_integral(self, server):
        resp = server.evaluate("\\oint_C f \\, dz")
        assert "oint" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\iiiint parsed as variable")
    def test_quadruple_integral(self, server):
        resp = server.evaluate("\\iiiint f\\,dV")
        assert "iiiint" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\idotsint parsed as variable")
    def test_idotsint(self, server):
        resp = server.evaluate("\\idotsint f(x_1,\\ldots,x_n)\\,dx_1 \\cdots dx_n")
        assert "idotsint" not in resp.get("expression", "")


class TestSiunitxAsVariable:
    """siunitx commands silently parsed as variable multiplication."""

    @pytest.mark.xfail(reason="\\SI parsed as variable multiplication")
    def test_si_unit(self, server):
        resp = server.evaluate("\\SI{9.81}{m/s^2}")
        expr = resp.get("expression", "")
        assert "SI" not in expr and "S*I" not in expr

    @pytest.mark.xfail(reason="\\num parsed as variable")
    def test_num_formatting(self, server):
        resp = server.evaluate("\\num{1.23e10}")
        assert "num" not in resp.get("expression", "")
