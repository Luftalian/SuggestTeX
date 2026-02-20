import json
import re
import sys
from sympy.parsing.latex import parse_latex


def send_response(data):
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def _replace_eval_at(expr: str) -> str:
    """Replace \\left. ... \\right|_{/^} with (...)| using brace-depth tracking.

    Unlike a fixed-depth regex, this handles arbitrarily nested braces.
    Also tracks ``\\left``/``\\right`` delimiter nesting so that inner
    ``\\left|…\\right|`` pairs (absolute value) are not mistaken for the
    eval-at closing bar.
    """
    LEFT_DOT = "\\left."
    RIGHT_BAR = "\\right|"
    LEFT_BAR = "\\left|"
    LEFT_CMD = "\\left"
    RIGHT_CMD = "\\right"
    result = []
    i = 0
    while i < len(expr):
        if expr[i:i + len(LEFT_DOT)] == LEFT_DOT:
            j = i + len(LEFT_DOT)
            brace_depth = 0
            delim_depth = 0
            found = -1
            while j < len(expr):
                c = expr[j]
                if c == '{':
                    brace_depth += 1
                    j += 1
                elif c == '}':
                    brace_depth -= 1
                    j += 1
                elif brace_depth == 0 and expr[j:j + len(LEFT_BAR)] == LEFT_BAR:
                    delim_depth += 1
                    j += len(LEFT_BAR)
                elif brace_depth == 0 and expr[j:j + len(RIGHT_BAR)] == RIGHT_BAR:
                    if delim_depth > 0:
                        delim_depth -= 1
                        j += len(RIGHT_BAR)
                    else:
                        after = j + len(RIGHT_BAR)
                        while after < len(expr) and expr[after] in ' \t':
                            after += 1
                        if after < len(expr) and expr[after] in '_^':
                            found = j
                            break
                        else:
                            j += len(RIGHT_BAR)
                elif brace_depth == 0 and expr[j:j + len(LEFT_CMD)] == LEFT_CMD:
                    # Other \left delimiters (e.g. \left(, \left[)
                    delim_depth += 1
                    j += len(LEFT_CMD)
                elif brace_depth == 0 and expr[j:j + len(RIGHT_CMD)] == RIGHT_CMD:
                    # Other \right delimiters (e.g. \right), \right])
                    if delim_depth > 0:
                        delim_depth -= 1
                    j += len(RIGHT_CMD)
                else:
                    j += 1
            if found >= 0:
                body = expr[i + len(LEFT_DOT):found]
                after = found + len(RIGHT_BAR)
                # Consume whitespace between \right| and _/^
                while after < len(expr) and expr[after] in ' \t':
                    after += 1
                result.append("(")
                result.append(body)
                result.append(")|")
                i = after
            else:
                result.append(expr[i])
                i += 1
        else:
            result.append(expr[i])
            i += 1
    return "".join(result)


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
    # Resolve eval-at (\left. ... \right|_{...}) BEFORE abs normalization
    # so that \right| from eval-at is not consumed by the abs regex.
    # Uses brace-depth tracking for unlimited nesting depth.
    prev = None
    while prev != expr:
        prev = expr
        expr = _replace_eval_at(expr)
    # Wrap bare eval-at scripts: )|_a → )|_{a}, )|^b → )|^{b}
    # SymPy's grammar requires braced forms |_{...} and |^{...}.
    # Handle single char: )|_a → )|_{a}
    expr = re.sub(r"\)\|_([^{\s\\])", r")|_{\1}", expr)
    expr = re.sub(r"\)\|\^([^{\s\\])", r")|^{\1}", expr)
    # Handle control sequences: )|_\alpha → )|_{\alpha}
    expr = re.sub(r"\)\|_(\\[a-zA-Z]+)", r")|_{\1}", expr)
    expr = re.sub(r"\)\|\^(\\[a-zA-Z]+)", r")|^{\1}", expr)
    # Handle second script after braced first (allow nested braces in first):
    # )|_{a}^b → )|_{a}^{b},  )|_{x_{0}}^n → )|_{x_{0}}^{n}
    _BRACE_GROUP = r"\{(?:[^{}]|\{[^{}]*\})*\}"
    expr = re.sub(r"(\)\|[_^]" + _BRACE_GROUP + r")\^([^{\s\\])", r"\1^{\2}", expr)
    expr = re.sub(r"(\)\|[_^]" + _BRACE_GROUP + r")_([^{\s\\])", r"\1_{\2}", expr)
    expr = re.sub(r"(\)\|[_^]" + _BRACE_GROUP + r")\^(\\[a-zA-Z]+)", r"\1^{\2}", expr)
    expr = re.sub(r"(\)\|[_^]" + _BRACE_GROUP + r")_(\\[a-zA-Z]+)", r"\1_{\2}", expr)
    # Reorder eval-at bounds: )|_{sub}^{sup} → )|^{sup}_{sub}
    # SymPy's grammar only accepts |^{...}_{...} (super before sub).
    expr = re.sub(
        r"\)\|(_" + _BRACE_GROUP + r")(\^" + _BRACE_GROUP + r")",
        r")|\2\1",
        expr,
    )
    # Normalize \left| ... \right| (absolute value) AFTER eval-at so the
    # abs regex does not consume \right| belonging to eval-at constructs.
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

    # 5. \cfrac → \frac (allow optional whitespace; lookahead for { or digit)
    expr = re.sub(r"\\cfrac\s*(?=[{\d\[])", r"\\frac", expr)

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
