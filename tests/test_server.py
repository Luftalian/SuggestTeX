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
    @pytest.mark.xfail(reason="\\left. \\right| evaluated-at notation not supported")
    def test_evaluated_at(self, server):
        assert_pass(server, "\\left. \\frac{d}{dx} f(x) \\right|_{x=a}")

    @pytest.mark.xfail(reason="mismatched \\left[ \\right) not supported")
    def test_mismatched(self, server):
        assert_pass(server, "\\left[ \\frac{a}{b} \\right)")

    @pytest.mark.xfail(reason="\\left\\{ with \\in and colon not supported")
    def test_set_builder(self, server):
        assert_pass(server, "\\left\\{ x \\in \\mathbb{R} : x > 0 \\right\\}")

    @pytest.mark.xfail(reason="\\left| \\right| absolute value not supported")
    def test_left_right_abs(self, server):
        assert_pass(server, "\\left| x \\right|")

    @pytest.mark.xfail(reason="\\left\\| \\right\\| norm not supported")
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


class TestSpacingFailure:
    @pytest.mark.xfail(reason="backslash-space not supported")
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

    @pytest.mark.xfail(reason="display math delimiters should be stripped")
    def test_display_math(self, server):
        assert_pass(server, "$$x$$")

    @pytest.mark.xfail(reason="inline math delimiters should be stripped")
    def test_inline_math(self, server):
        assert_pass(server, "$x$")

    @pytest.mark.xfail(reason="paren math delimiters should be stripped")
    def test_paren_math(self, server):
        assert_pass(server, "\\(x\\)")


class TestDoubleFactorial:
    def test_double_factorial(self, server):
        assert_pass(server, "(n-1)!!")
