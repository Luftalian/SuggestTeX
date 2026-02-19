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

    def test_qquad_space(self, server):
        assert_pass(server, "x\\qquad y")

    def test_sin_thin_space(self, server):
        """\\sin\\,x must not merge into \\sinx."""
        assert_pass(server, "\\sin\\,x")

    def test_integral_thin_space_dx(self, server):
        """d\\,x in integral context should parse correctly."""
        assert_pass(server, "\\int_0^1 f(x)\\,dx")

    def test_multiple_spacing_commands(self, server):
        """Multiple spacing commands in one expression."""
        assert_pass(server, "\\cos\\;x + \\sin\\!y")


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

    def test_evaluated_at_superscript(self, server):
        assert_pass(server, "\\left. x \\right| ^2")

    def test_nested_eval_at_in_frac(self, server):
        """Eval-at inside \\frac must preserve brace balance."""
        assert_pass(server, "\\left. \\frac{\\left. f \\right|_{a}}{g}\\right|_{b}")

    def test_nested_eval_at_inner_grouping(self, server):
        """Inner eval-at body must be wrapped in parens for correct scope."""
        assert_pass(server, "\\left. \\frac{\\left. f+g \\right|_{a}}{h}\\right|_{b}")

    def test_nested_abs_inside_eval_at(self, server):
        assert_pass(server, "\\left.\\left|x\\right|\\right|_{x=1}")

    def test_nested_abs_double(self, server):
        assert_pass(server, "\\left|\\left|x\\right|+1\\right|")

    def test_nested_abs_triple(self, server):
        assert_pass(server, "\\left|x+\\left|y\\right|\\right|")

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


class TestNormSemantics:
    """\\left\\|…\\right\\| is preprocessed to |…| (abs), losing norm distinction."""

    @pytest.mark.xfail(reason="\\left\\|…\\right\\| becomes |…| (abs), not norm")
    def test_norm_becomes_abs(self, server):
        resp = server.evaluate("\\left\\| x \\right\\|")
        expr = resp.get("expression", "")
        # Should ideally produce Norm(x) or similar, not Abs(x)
        assert "Abs" not in expr


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


# ===================================================================
# NEW PASSING TESTS — Round 4 agent team findings
# ===================================================================


class TestAdvancedIntegralsPass:
    """Advanced integral expressions confirmed working."""

    def test_gaussian_integral(self, server):
        assert_pass(server, "\\int_{-\\infty}^{\\infty} e^{-x^2} \\, dx = \\sqrt{\\pi}")

    def test_dirichlet_integral(self, server):
        assert_pass(server, "\\int_0^{\\infty} \\frac{\\sin x}{x} \\, dx = \\frac{\\pi}{2}")

    def test_convolution(self, server):
        assert_pass(server, "\\int_{-\\infty}^{\\infty} f(\\tau) g(t - \\tau) \\, d\\tau")

    def test_iterated_integral(self, server):
        assert_pass(server, "\\int_0^1 \\int_0^x f(x,y) \\, dy \\, dx")

    def test_polar_integral(self, server):
        assert_pass(server, "\\int_0^{2\\pi} \\int_0^R f(r,\\theta) \\, r \\, dr \\, d\\theta")

    def test_leibniz_integral_rule(self, server):
        assert_pass(server, "\\frac{d}{dx} \\int_{a(x)}^{b(x)} f(x,t) \\, dt")

    def test_dirac_delta_sifting(self, server):
        assert_pass(server, "\\int_{-\\infty}^{\\infty} f(x) \\delta(x - a) \\, dx = f(a)")

    def test_fresnel_integral(self, server):
        assert_pass(server, "S(x) = \\int_0^x \\sin(t^2) \\, dt")

    def test_arc_length(self, server):
        assert_pass(server, "L = \\int_a^b \\sqrt{1 + (\\frac{dy}{dx})^2} \\, dx")

    def test_gamma_integral(self, server):
        assert_pass(server, "\\Gamma(z) = \\int_0^{\\infty} t^{z-1} e^{-t} \\, dt")


class TestLinearAlgebraPass:
    """Linear algebra expressions confirmed working."""

    def test_characteristic_equation(self, server):
        assert_pass(server, "\\det(A - \\lambda I) = 0")

    def test_svd(self, server):
        assert_pass(server, "A = U \\Sigma V^T")

    def test_eigenvalue_max(self, server):
        assert_pass(server, "\\lambda_{\\max}(A)")

    def test_trace_via_sum(self, server):
        assert_pass(server, "\\sum_{i=1}^{n} a_{ii}")

    def test_det_product_eigenvalues(self, server):
        assert_pass(server, "\\prod_{i=1}^{n} \\lambda_i")

    def test_lu_decomposition(self, server):
        assert_pass(server, "A = LU")

    def test_qr_decomposition(self, server):
        assert_pass(server, "A = QR")

    def test_left_right_norm_to_abs(self, server):
        assert_pass(server, "\\left\\| x \\right\\|")


class TestNumberTheoryPass:
    """Number theory expressions confirmed working."""

    def test_catalan_number(self, server):
        assert_pass(server, "C_n = \\frac{1}{n+1}\\binom{2n}{n}")

    def test_binet_formula(self, server):
        assert_pass(server, "F_n = \\frac{\\varphi^n - \\psi^n}{\\sqrt{5}}")

    def test_liouville_function(self, server):
        assert_pass(server, "\\lambda(n) = (-1)^{\\Omega(n)}")

    def test_von_mangoldt(self, server):
        assert_pass(server, "\\Lambda(n)")

    def test_legendre_formula(self, server):
        assert_pass(server, "\\sum_{i=1}^{\\infty} \\lfloor \\frac{n}{p^i} \\rfloor")

    def test_nested_floor_ceiling(self, server):
        assert_pass(server, "\\lceil \\lfloor x \\rfloor + 0.5 \\rceil")

    def test_dirichlet_series(self, server):
        assert_pass(server, "\\sum_{n=1}^{\\infty} \\frac{f(n)}{n^s}")

    def test_bell_number(self, server):
        assert_pass(server, "B_n = \\sum_{k=0}^{n} S(n,k)")

    @pytest.mark.xfail(reason="symbolic binomial sum causes evaluation error")
    def test_binomial_sum_with_powers(self, server):
        assert_pass(server, "\\sum_{k=0}^{n} \\binom{n}{k} x^k")


class TestProbabilityPass:
    """Probability/statistics expressions confirmed working."""

    def test_pr_function(self, server):
        assert_pass(server, "\\Pr(A)")

    def test_cdf_integral(self, server):
        assert_pass(server, "F(x) = \\int_{-\\infty}^{x} f(u) \\, du")

    def test_pdf_derivative(self, server):
        assert_pass(server, "f(x) = \\frac{dF}{dx}")

    def test_mean_integral(self, server):
        assert_pass(server, "\\mu = \\int_{-\\infty}^{\\infty} x f(x) \\, dx")

    def test_dbinom(self, server):
        assert_pass(server, "\\dbinom{n}{k} p^k (1-p)^{n-k}")

    def test_beta_function(self, server):
        assert_pass(server, "B(\\alpha, \\beta) = \\frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha+\\beta)}")

    def test_poisson_pmf(self, server):
        assert_pass(server, "\\frac{\\lambda^k e^{-\\lambda}}{k!}")

    def test_exponential_pdf(self, server):
        assert_pass(server, "f(x) = \\lambda e^{-\\lambda x}")


class TestPhysicsEquationsPass:
    """Physics equations confirmed working."""

    def test_energy_momentum(self, server):
        assert_pass(server, "E^2 = (pc)^2 + (mc^2)^2")

    def test_de_broglie(self, server):
        assert_pass(server, "\\lambda = \\frac{h}{p}")

    def test_harmonic_oscillator(self, server):
        assert_pass(server, "E_n = \\hbar \\omega (n + \\frac{1}{2})")

    def test_coulomb_law(self, server):
        assert_pass(server, "F = \\frac{1}{4\\pi \\epsilon_0} \\frac{q_1 q_2}{r^2}")

    def test_maxwell_gauss(self, server):
        assert_pass(server, "\\nabla \\cdot E = \\frac{\\rho}{\\epsilon_0}")

    def test_schrodinger_time_indep(self, server):
        assert_pass(server, "H \\psi = E \\psi")

    def test_bohr_radius(self, server):
        assert_pass(server, "a_0 = \\frac{4\\pi \\epsilon_0 \\hbar^2}{m_e e^2}")

    def test_planck_einstein(self, server):
        assert_pass(server, "E = h \\nu")

    def test_ideal_gas(self, server):
        assert_pass(server, "PV = nRT")

    def test_fourier_transform(self, server):
        assert_pass(server, "\\hat{f}(\\xi) = \\int_{-\\infty}^{\\infty} f(x) e^{-2\\pi i x \\xi} \\, dx")


# ===================================================================
# NEW ERROR FAILURES — Round 4 agent team findings
# ===================================================================


class TestLimsupLiminfFailure:
    """\\limsup and \\liminf not recognized by parse_latex."""

    @pytest.mark.xfail(reason="\\limsup not supported by parse_latex")
    def test_limsup(self, server):
        assert_pass(server, "\\limsup_{n \\to \\infty} a_n")

    @pytest.mark.xfail(reason="\\liminf not supported by parse_latex")
    def test_liminf(self, server):
        assert_pass(server, "\\liminf_{n \\to \\infty} a_n")


class TestIntegralLowerBoundOnlyFailure:
    """Integrals with only a lower bound (no upper bound) fail."""

    @pytest.mark.xfail(reason="\\int with only lower bound not supported")
    def test_measure_integral(self, server):
        assert_pass(server, "\\int_X f \\, d\\mu")

    @pytest.mark.xfail(reason="\\int with only lower bound not supported")
    def test_contour_integral_lower(self, server):
        assert_pass(server, "\\int_C P \\, dx + Q \\, dy")

    @pytest.mark.xfail(reason="\\int with only lower bound not supported")
    def test_line_integral(self, server):
        assert_pass(server, "\\int_\\gamma f(z) \\, dz")

    @pytest.mark.xfail(reason="\\int_{\\mathbb{R}} missing upper bound")
    def test_lebesgue_integral(self, server):
        assert_pass(server, "\\int_{\\mathbb{R}} f(x) \\, d\\lambda(x)")


class TestSumProdUnboundedFailure:
    """\\sum and \\prod without explicit = bounds or with non-standard subscripts."""

    @pytest.mark.xfail(reason="\\sum without any bounds not supported")
    def test_sum_no_bounds(self, server):
        assert_pass(server, "\\sum \\frac{(A-E)^2}{E}")

    @pytest.mark.xfail(reason="\\sum with bare variable subscript not supported")
    def test_sum_bare_subscript(self, server):
        assert_pass(server, "\\sum_n a_n x^n")

    @pytest.mark.xfail(reason="\\prod without bounds not supported")
    def test_prod_no_bounds(self, server):
        assert_pass(server, "\\prod f(x_i)")

    @pytest.mark.xfail(reason="\\prod with bare variable subscript not supported")
    def test_prod_bare_subscript(self, server):
        assert_pass(server, "\\prod_p (1 - p^{-s})")

    @pytest.mark.xfail(reason="\\sum_{d \\mid n} divisor condition not supported")
    def test_sum_divisor_mid(self, server):
        assert_pass(server, "\\sum_{d \\mid n} d")

    @pytest.mark.xfail(reason="\\prod_{p \\mid n} divisor condition not supported")
    def test_prod_divisor_mid(self, server):
        assert_pass(server, "\\prod_{p \\mid n} p")

    @pytest.mark.xfail(reason="\\sum_{d \\mid n} Mobius inversion not supported")
    def test_mobius_inversion(self, server):
        assert_pass(server, "\\sum_{d \\mid n} \\mu(d) f(\\frac{n}{d})")


class TestSumInequalityBoundFailure:
    """\\sum/\\prod with inequality or set membership bounds fail."""

    @pytest.mark.xfail(reason="\\leq in sum subscript not supported")
    def test_sum_leq_bound(self, server):
        assert_pass(server, "\\sum_{p \\leq x} \\frac{1}{p}")

    @pytest.mark.xfail(reason="\\in in sum subscript not supported")
    def test_sum_set_membership(self, server):
        assert_pass(server, "\\sum_{x \\in S} f(x)")

    @pytest.mark.xfail(reason="\\in in prod subscript not supported")
    def test_prod_set_membership(self, server):
        assert_pass(server, "\\zeta(s) = \\prod_{p \\in \\mathbb{P}} \\frac{1}{1-p^{-s}}")

    @pytest.mark.xfail(reason="\\text in prod subscript not supported")
    def test_prod_text_condition(self, server):
        assert_pass(server, "\\prod_{p \\text{ prime}} \\frac{1}{1 - p^{-s}}")

    @pytest.mark.xfail(reason="\\sum_{x \\in \\mathcal{X}} not supported")
    def test_entropy_sum(self, server):
        assert_pass(server, "H(X) = -\\sum_{x \\in \\mathcal{X}} p(x) \\log p(x)")


class TestCommutatorBracketFailure:
    """Commutator brackets [A, B] with comma fail."""

    @pytest.mark.xfail(reason="comma inside square brackets not supported")
    def test_commutator_basic(self, server):
        assert_pass(server, "[\\hat{x}, \\hat{p}] = i\\hbar")

    @pytest.mark.xfail(reason="comma inside square brackets not supported")
    def test_commutator_spin(self, server):
        assert_pass(server, "[S_x, S_y] = i\\hbar S_z")

    @pytest.mark.xfail(reason="comma inside curly braces not supported")
    def test_anticommutator(self, server):
        assert_pass(server, "\\{\\hat{a}, \\hat{a}^{\\dagger}\\} = 1")

    @pytest.mark.xfail(reason="comma inside square brackets not supported")
    def test_angular_momentum_commutator(self, server):
        assert_pass(server, "[L_i, L_j] = i\\hbar\\epsilon_{ijk}L_k")


class TestSemicolonInArgsFailure:
    """Semicolons as argument separators fail."""

    @pytest.mark.xfail(reason="semicolon in function args not supported")
    def test_likelihood(self, server):
        assert_pass(server, "L(\\theta) = \\prod_{i=1}^{n} f(x_i; \\theta)")

    @pytest.mark.xfail(reason="semicolon in function args not supported")
    def test_mutual_information(self, server):
        assert_pass(server, "I(X;Y)")

    @pytest.mark.xfail(reason="semicolon in Weibull PDF not supported")
    def test_weibull_pdf(self, server):
        assert_pass(server, "f(x;\\lambda,k) = \\frac{k}{\\lambda}\\left(\\frac{x}{\\lambda}\\right)^{k-1}")


class TestBarSubscriptPass:
    """Subscript after \\bar{} works."""

    def test_bar_subscript(self, server):
        assert_pass(server, "\\bar{X}_n")


class TestBareDerivativeFailure:
    """Bare derivative operators without operand fail."""

    @pytest.mark.xfail(reason="bare derivative operator without operand fails")
    def test_momentum_operator(self, server):
        assert_pass(server, "\\hat{p} = -i\\hbar\\frac{\\partial}{\\partial x}")

    @pytest.mark.xfail(reason="bare derivative operator without operand fails")
    def test_angular_momentum_operator(self, server):
        assert_pass(server, "L_z = -i\\hbar\\frac{\\partial}{\\partial \\varphi}")


class TestMultinomialBinomFailure:
    """Multinomial \\binom with commas in second argument fails."""

    @pytest.mark.xfail(reason="comma in \\binom second argument not supported")
    def test_multinomial_binom(self, server):
        assert_pass(server, "\\binom{n}{k_1, k_2, k_3}")


class TestCommaTupleFailure:
    """Comma-separated tuples inside parentheses fail."""

    @pytest.mark.xfail(reason="multivariate limit tuple not supported")
    def test_multivariate_limit(self, server):
        assert_pass(server, "\\lim_{(x,y) \\to (0,0)} \\frac{xy}{x^2 + y^2}")

    @pytest.mark.xfail(reason="comma tuple in \\left(\\right) not supported")
    def test_confidence_interval(self, server):
        assert_pass(server, "\\left(\\bar{x} - z\\frac{s}{\\sqrt{n}}, \\bar{x} + z\\frac{s}{\\sqrt{n}}\\right)")

    @pytest.mark.xfail(reason="four-momentum tuple not supported")
    def test_four_momentum(self, server):
        assert_pass(server, "p^{\\mu} = (E/c, \\vec{p})")

    @pytest.mark.xfail(reason="\\langle with comma not supported")
    def test_inner_product_comma(self, server):
        assert_pass(server, "\\langle x, y \\rangle")


class TestPipeInFracFailure:
    """Pipe | inside \\frac{} confuses parser."""

    @pytest.mark.xfail(reason="| inside \\frac numerator confuses parser")
    def test_bayes_pipe_in_frac(self, server):
        assert_pass(server, "\\frac{P(B|A)P(A)}{P(B)}")


class TestCommaInSubscriptFailure:
    """Comma inside subscripts fails."""

    @pytest.mark.xfail(reason="comma in subscript not supported")
    def test_correlation_subscript(self, server):
        assert_pass(server, "\\rho_{X,Y}")


class TestPartitionFunctionFailure:
    """Partition function Z = \\sum_n without = bound."""

    @pytest.mark.xfail(reason="\\sum_{n} without = bound not supported")
    def test_partition_function(self, server):
        assert_pass(server, "Z = \\sum_{n} e^{-\\beta E_n}")


class TestNormBareFailure:
    """Bare \\| norm notation (without \\left/\\right)."""

    @pytest.mark.xfail(reason="\\| norm notation not recognized")
    def test_norm_bare(self, server):
        assert_pass(server, "\\|x\\|")

    @pytest.mark.xfail(reason="\\| with subscript not recognized")
    def test_norm_frobenius(self, server):
        assert_pass(server, "\\|A\\|_F")

    @pytest.mark.xfail(reason="\\| L2 norm not recognized")
    def test_norm_l2(self, server):
        assert_pass(server, "\\|x\\|_2")


class TestKLDivergenceFailure:
    """KL divergence with \\| inside parens."""

    @pytest.mark.xfail(reason="\\| inside parens fails")
    def test_kl_divergence(self, server):
        assert_pass(server, "D_{KL}(P \\| Q)")


# ===================================================================
# NEW SUSPECT TESTS — Round 4 agent team findings
# Commands silently parsed as variable names or producing wrong results.
# ===================================================================


class TestNablaAsVariable:
    """\\nabla silently parsed as variable, not differential operator."""

    @pytest.mark.xfail(reason="\\nabla parsed as variable, not gradient operator")
    def test_gradient(self, server):
        resp = server.evaluate("\\nabla f")
        assert "nabla" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\nabla parsed as variable in divergence")
    def test_divergence(self, server):
        resp = server.evaluate("\\nabla \\cdot \\vec{F}")
        assert "nabla" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\nabla^2 parsed as variable squared")
    def test_laplacian(self, server):
        resp = server.evaluate("\\nabla^2 f")
        assert "nabla" not in resp.get("expression", "")


class TestConditionalProbTruncation:
    """Pipe | in conditional probability causes expression truncation."""

    @pytest.mark.xfail(reason="P(A|B) pipe truncates to just P")
    def test_conditional_prob_pipe(self, server):
        resp = server.evaluate("P(A|B)")
        expr = resp.get("expression", "")
        assert "A" in expr and "B" in expr

    @pytest.mark.xfail(reason="P(X \\leq x) inequality inside P() loses args")
    def test_prob_inequality(self, server):
        resp = server.evaluate("P(X \\leq x)")
        expr = resp.get("expression", "")
        assert "X" in expr

    @pytest.mark.xfail(reason="H(X|Y) pipe truncates conditional entropy")
    def test_conditional_entropy(self, server):
        resp = server.evaluate("H(X|Y)")
        expr = resp.get("expression", "")
        assert "X" in expr and "Y" in expr

    @pytest.mark.xfail(reason="E[\\delta(T)|T] pipe in E[] truncates")
    def test_conditional_expectation(self, server):
        resp = server.evaluate("E[\\delta(T)|T]")
        expr = resp.get("expression", "")
        assert "delta" in expr.lower() or "Delta" in expr


class TestTextFunctionAsVariable:
    """\\text{func} decomposed into letter multiplication."""

    @pytest.mark.xfail(reason="\\text{Var} decomposed into V*a*r")
    def test_text_var(self, server):
        resp = server.evaluate("\\text{Var}(X)")
        assert "text" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\text{Cov} decomposed into letter multiplication")
    def test_text_cov(self, server):
        resp = server.evaluate("\\text{Cov}(X, Y)")
        assert "text" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\text{lcm} decomposed into letter multiplication")
    def test_text_lcm(self, server):
        resp = server.evaluate("\\text{lcm}(a,b)")
        assert "text" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mathrm{tr} decomposed into t*r")
    def test_mathrm_trace(self, server):
        resp = server.evaluate("\\mathrm{tr}(A)")
        assert "mathrm" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname{tr} decomposed")
    def test_operatorname_trace(self, server):
        resp = server.evaluate("\\operatorname{tr}(A)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\operatorname{rank} decomposed")
    def test_operatorname_rank(self, server):
        resp = server.evaluate("\\operatorname{rank}(A)")
        assert "operatorname" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\text{Res} decomposed into R*e*s")
    def test_text_residue(self, server):
        resp = server.evaluate("\\text{Res}_{z=z_0} f(z)")
        assert "text" not in resp.get("expression", "")


class TestDotNotationAsVariable:
    """\\dot{q} silently parsed as dot*q in equations."""

    @pytest.mark.xfail(reason="\\dot{q} parsed as dot*q, not time derivative")
    def test_euler_lagrange_dot(self, server):
        resp = server.evaluate(
            "\\frac{d}{dt}\\frac{\\partial L}{\\partial \\dot{q}} "
            "- \\frac{\\partial L}{\\partial q} = 0"
        )
        expr = resp.get("expression", "")
        assert "dot*q" not in expr

    @pytest.mark.xfail(reason="\\dot{q} = ... drops equation body")
    def test_hamilton_equation(self, server):
        resp = server.evaluate("\\dot{q}_i = \\frac{\\partial H}{\\partial p_i}")
        expr = resp.get("expression", "")
        assert "H" in expr or "Derivative" in expr


class TestNthDerivativeNotation:
    """f^{(n)}(x) parsed incorrectly."""

    @pytest.mark.xfail(reason="f^{(n)}(x) parsed as f**n*x, not nth derivative")
    def test_nth_derivative(self, server):
        resp = server.evaluate("f^{(n)}(x)")
        expr = resp.get("expression", "")
        assert "f(x)" in expr or "Derivative" in expr


class TestChooseAsVariable:
    """{n \\choose k} parsed as variable multiplication."""

    @pytest.mark.xfail(reason="{n \\choose k} not recognized as binomial")
    def test_choose_notation(self, server):
        resp = server.evaluate("{n \\choose k}")
        assert "choose" not in resp.get("expression", "")


class TestColonAsDivision:
    """Colon : silently parsed as division."""

    @pytest.mark.xfail(reason="colon parsed as division, not separator")
    def test_hypothesis_notation(self, server):
        resp = server.evaluate("H_0: \\mu = \\mu_0")
        expr = resp.get("expression", "")
        assert "H_{0}/mu" not in expr


class TestRelationsAsVariables:
    """Relational operators silently parsed as variable names."""

    @pytest.mark.xfail(reason="\\sim parsed as variable in distribution notation")
    def test_sim_distribution(self, server):
        resp = server.evaluate("X \\sim N(\\mu, \\sigma^2)")
        assert "sim" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\perp parsed as variable")
    def test_perp_independence(self, server):
        resp = server.evaluate("X \\perp Y")
        assert "perp" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\succeq parsed as variable")
    def test_succeq_psd(self, server):
        resp = server.evaluate("A \\succeq 0")
        assert "succeq" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\mid parsed as variable in conditional")
    def test_mid_conditional(self, server):
        resp = server.evaluate("P(A \\mid B)")
        assert "mid" not in resp.get("expression", "")


class TestSquareBracketAsMultiplication:
    """Square brackets E[X] parsed as E*X multiplication."""

    @pytest.mark.xfail(reason="E[X] square brackets become multiplication")
    def test_expectation_bracket(self, server):
        resp = server.evaluate("E[X]")
        expr = resp.get("expression", "")
        assert expr != "E*X"

    @pytest.mark.xfail(reason="\\mathbb{E}[X] parsed as mathbb*(E*X)")
    def test_mathbb_expectation(self, server):
        resp = server.evaluate("\\mathbb{E}[X]")
        assert "mathbb" not in resp.get("expression", "")


class TestDoubleFactorialSuspect:
    """n!! parsed as factorial(factorial(n)) instead of double factorial."""

    @pytest.mark.xfail(reason="n!! parsed as nested factorial, not double factorial")
    def test_double_factorial_semantics(self, server):
        resp = server.evaluate("n!!")
        expr = resp.get("expression", "")
        assert "factorial(factorial" not in expr

    @pytest.mark.xfail(reason="n!!! parsed as triple nested factorial")
    def test_triple_factorial_semantics(self, server):
        resp = server.evaluate("n!!!")
        expr = resp.get("expression", "")
        assert "factorial(factorial(factorial" not in expr


class TestModularArithmeticAsVariable:
    """Modular arithmetic operators silently parsed as variables."""

    @pytest.mark.xfail(reason="\\equiv parsed as variable")
    def test_equiv_pmod(self, server):
        resp = server.evaluate("a \\equiv b \\pmod{n}")
        assert "equiv" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\bmod parsed as variable")
    def test_bmod_as_variable(self, server):
        resp = server.evaluate("a \\bmod n")
        assert "bmod" not in resp.get("expression", "")


class TestLVertRVertAsVariable:
    """\\lVert/\\rVert parsed as variable names."""

    @pytest.mark.xfail(reason="\\lVert/\\rVert parsed as variables")
    def test_lvert_rvert_norm(self, server):
        resp = server.evaluate("\\lVert x \\rVert")
        assert "lVert" not in resp.get("expression", "")


class TestAccentDropsContext:
    """Accents lose subscripts or equation context."""

    @pytest.mark.xfail(reason="\\chi^2_k loses subscript _k")
    def test_chi_squared_subscript(self, server):
        resp = server.evaluate("\\chi^2_k")
        expr = resp.get("expression", "")
        assert "k" in expr

    @pytest.mark.xfail(reason="\\hat{\\theta} becomes hat*theta")
    def test_hat_theta(self, server):
        resp = server.evaluate("\\hat{\\theta}_{ML}")
        expr = resp.get("expression", "")
        assert "hat" not in expr or "hat(theta)" in expr.lower()

    @pytest.mark.xfail(reason="Entropy sum silently drops \\sum body")
    def test_entropy_sum_dropped(self, server):
        resp = server.evaluate("S = -k_B \\sum_i p_i \\ln p_i")
        expr = resp.get("expression", "")
        assert "ln" in expr.lower() or "log" in expr.lower() or "Sum" in expr


class TestOtimesOplusAsVariable:
    """\\otimes and \\oplus parsed as variable names in expressions."""

    @pytest.mark.xfail(reason="\\otimes parsed as variable in Kronecker product")
    def test_kronecker_product(self, server):
        resp = server.evaluate("A \\otimes B")
        assert "otimes" not in resp.get("expression", "")

    @pytest.mark.xfail(reason="\\oplus parsed as variable in direct sum")
    def test_direct_sum(self, server):
        resp = server.evaluate("A \\oplus B")
        assert "oplus" not in resp.get("expression", "")


class TestSupInfAsVariable:
    """\\sup and \\inf with \\in — \\in becomes variable 'in' inside subscript."""

    @pytest.mark.xfail(reason="\\sup_{x \\in S} — \\in parsed as variable 'in'")
    def test_sup_in(self, server):
        resp = server.evaluate("\\sup_{x \\in S} f(x)")
        expr = resp.get("expression", "")
        # The expression should not contain 'in' as an implicit variable
        assert "in" not in expr

    @pytest.mark.xfail(reason="\\inf_{x \\in S} — \\in parsed as variable 'in'")
    def test_inf_in(self, server):
        resp = server.evaluate("\\inf_{x \\in S} f(x)")
        expr = resp.get("expression", "")
        assert "in" not in expr
