"""A local calculator, so a comparison or total in an answer is computed, not judged by eye.

One expression is parsed against a whitelist: numbers, arithmetic, a few math functions and
``distance_km``. It runs in this process with no network, sandbox or model call. The result
names its input numbers, so the answer's evidence can quote the page each one came from.
"""

from __future__ import annotations

import ast
import math
import operator

from frontier_agent.core.tool import tool

MAX_EXPRESSION_CHARS = 400
MAX_MAGNITUDE = 1e100
MAX_EXPONENT = 64
EARTH_RADIUS_KM = 6371.0088


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in decimal degrees, by the haversine formula."""
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("latitude_out_of_range")
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("longitude_out_of_range")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    half = (math.sin((phi2 - phi1) / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(half)))


_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt,
    "sin": lambda degrees: math.sin(math.radians(degrees)),
    "cos": lambda degrees: math.cos(math.radians(degrees)),
    "tan": lambda degrees: math.tan(math.radians(degrees)),
    "distance_km": distance_km,
}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not_a_number")
    if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise ValueError("number_out_of_range")
    return value


def _evaluate(node: ast.AST, inputs: list[float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, inputs)
    if isinstance(node, ast.Constant):
        inputs.append(_number(node.value))
        return inputs[-1]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        if isinstance(node.operand, ast.Constant):
            # A signed coordinate such as -73.56 is one input, as written.
            inputs.append(sign * _number(node.operand.value))
            return inputs[-1]
        return _number(sign * _evaluate(node.operand, inputs))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left, inputs), _evaluate(node.right, inputs)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent_too_large")
        return _number(_BINARY[type(node.op)](left, right))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        if node.func.id == "round":
            # The number of decimals is a choice of display, not an input.
            args = [_evaluate(arg, inputs) for arg in node.args[:1]]
            args += [int(_evaluate(arg, [])) for arg in node.args[1:]]
        else:
            args = [_evaluate(arg, inputs) for arg in node.args]
        return _number(_FUNCTIONS[node.func.id](*args))
    raise ValueError("unsupported_expression")


def evaluate(expression: str) -> tuple[float, list[float]]:
    """The value of one expression and the numbers it was given, in order."""
    text = str(expression or "").strip()
    if not text or len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression_empty_or_too_long")
    inputs: list[float] = []
    return _evaluate(ast.parse(text, mode="eval"), inputs), inputs


def show(value: float) -> str:
    return repr(value) if isinstance(value, int) else f"{value:.10g}"


@tool
async def calculate(expression: str) -> str:
    """Compute an arithmetic result exactly instead of estimating it.

    Use it for every comparison, difference or total of numbers, dates or distances the
    answer depends on, with each input number copied from the evidence.

    Args:
        expression: One expression using numbers, + - * / // % **, parentheses and the
            functions abs, round, min, max, sqrt, sin, cos, tan (degrees) and
            distance_km(lat1, lon1, lat2, lon2) for decimal-degree coordinates, for
            example "distance_km(45.5081, -73.5686, 45.5106, -73.5564)".

    Returns:
        The expression, its input numbers and the result, or an error.
    """
    try:
        value, inputs = evaluate(expression)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError, RecursionError) as exc:
        return f"[ERROR]: calculate could not evaluate the expression: {exc}"
    return (f"CALCULATION: {str(expression).strip()}\n"
            f"INPUTS: {', '.join(show(number) for number in inputs)}\n"
            f"RESULT: {show(value)}")


__all__ = ["calculate", "distance_km", "evaluate"]
