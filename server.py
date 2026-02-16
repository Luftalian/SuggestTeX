import json
import sys
from sympy.parsing.latex import parse_latex


def send_response(data):
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


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
            expr = parse_latex(expression)
            result = expr.evalf()
            print(f"Parsed: {expr} = {result}", file=sys.stderr)
            send_response({"id": req_id, "result": str(result), "expression": str(expr)})
        except Exception as e:
            print(f"Error for {req_id}: {e}", file=sys.stderr)
            send_response({"id": req_id, "error": str(e)})


if __name__ == "__main__":
    main()
