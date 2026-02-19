import json
import re
import sys
from sympy.parsing.latex import parse_latex


def send_response(data):
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def preprocess_latex(expr: str) -> str:
    """Normalize LaTeX before passing to parse_latex.

    Handles common notation that parse_latex does not support natively:
    math-mode delimiters, spacing commands, ``\\left``/``\\right`` sizing,
    ``\\cfrac``, and empty leading groups.
    """
    # 0. Trim whitespace so delimiter checks work on padded input
    expr = expr.strip()

    # 1. Strip math-mode delimiters: $$...$$, $...$, \(...\)
    #    Only strip when the entire input is a single wrapped block.
    if expr.startswith("$$") and expr.endswith("$$") and "$$" not in expr[2:-2]:
        expr = expr[2:-2]
    elif expr.startswith("$") and expr.endswith("$") and "$" not in expr[1:-1]:
        expr = expr[1:-1]
    elif expr.startswith("\\(") and expr.endswith("\\)") and "\\(" not in expr[2:-2]:
        expr = expr[2:-2]

    # 2. Strip spacing commands (but preserve \\ row separators)
    # Use negative lookbehind to avoid matching \\, \\; \\: \\! (row sep + spacing)
    expr = re.sub(r"(?<!\\)\\[,;:!]", " ", expr)
    expr = re.sub(r"(?<!\\)\\quad\b", " ", expr)
    expr = re.sub(r"(?<!\\)\\qquad\b", " ", expr)
    # Replace backslash-space, but not double-backslash-space (row separator)
    expr = re.sub(r"(?<!\\)\\ ", " ", expr)

    # 3. Normalize \left/\right delimiters
    # \left\| ... \right\|  →  | ... |  (treat scalar norm as abs)
    expr = re.sub(r"\\left\\\|", "|", expr)
    expr = re.sub(r"\\right\\\|", "|", expr)
    # Normalize \left| ... \right| (absolute value) BEFORE eval-at so the
    # eval-at regex does not accidentally consume inner absolute-value bars.
    # Use a tempered greedy token to match innermost pairs first, then loop
    # outward so nested absolute values like \left|\left|x\right|+1\right|
    # resolve correctly to ||x|+1|.
    _abs_inner = re.compile(
        r"\\left\|((?:(?!\\left\||\\right\|).)+)\\right\|"
    )
    prev = None
    while prev != expr:
        prev = expr
        expr = _abs_inner.sub(r"|\1|", expr)
    # \left. ... \right|_{...}  →  wrap in parens to preserve eval-at scope
    # Allow optional whitespace before _ or ^ in the lookahead.
    # Use balanced-brace matching so \right| inside \frac{}{} is not
    # consumed prematurely (e.g. \left. \frac{\left. f \right|_a}{g}\right|_b).
    _BRACE_BAL = (
        r"(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*?"
    )
    expr = re.sub(
        r"\\left\.\s*(" + _BRACE_BAL + r")\\right\|\s*(?=[_^])",
        r"(\1)|",
        expr,
    )
    # Remaining standalone \left. or \right| (without matched pair)
    expr = re.sub(r"\\left\.", "", expr)
    expr = re.sub(r"\\right\|", "|", expr)
    # \left( → ( ,  \right) → )
    expr = expr.replace("\\left(", "(")
    expr = expr.replace("\\right)", ")")
    # \left[ → [ ,  \right] → ]
    expr = expr.replace("\\left[", "[")
    expr = expr.replace("\\right]", "]")
    # \left\{ → \{ ,  \right\} → \}
    expr = expr.replace("\\left\\{", "\\{")
    expr = expr.replace("\\right\\}", "\\}")

    # 4. Strip empty leading groups:  {}_x → _x ,  {}^x → ^x
    expr = re.sub(r"\{\}(?=[_^])", "", expr)

    # 5. \cfrac → \frac (lookahead for { or digit to avoid matching e.g. \dcfrac)
    expr = re.sub(r"\\cfrac(?=[{\d\[])", r"\\frac", expr)

    return expr.strip()


def main():
    send_response({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            send_response({"id": None, "error": f"Invalid JSON: {e}"})
            continue

        if not isinstance(data, dict):
            print(f"Expected JSON object, got {type(data).__name__}", file=sys.stderr)
            send_response({"id": None, "error": "Expected JSON object"})
            continue

        req_id = data.get("id")
        expression = data.get("expression", "")
        print(f"Request {req_id}: {expression}", file=sys.stderr)

        try:
            preprocessed = preprocess_latex(expression)
            print(f"Preprocessed: {preprocessed}", file=sys.stderr)
            expr = parse_latex(preprocessed)
            result = expr.evalf()
            print(f"Parsed: {expr} = {result}", file=sys.stderr)
            send_response({"id": req_id, "result": str(result), "expression": str(expr)})
        except Exception as e:
            print(f"Error for {req_id}: {e}", file=sys.stderr)
            send_response({"id": req_id, "error": str(e)})


if __name__ == "__main__":
    main()
