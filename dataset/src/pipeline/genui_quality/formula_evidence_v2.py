"""Lossless supported Formula spelling normalization for the opt-in pilot.

Mirrors ``normalizeFormulaText`` in Android's FlatSpecRenderer.kt. This is
not algebraic equivalence: fractions, scripts, operators, unknown commands,
units, digits, qualifications and equation-row order are retained.
"""
from __future__ import annotations

import re

from .metrics_v5_4 import normalize_formula_wrapper

_COMMANDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ",
    "Sigma": "Σ", "sum": "∑", "rightarrow": "→", "sqrt": "√",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "infty": "∞",
    "times": "×", "cdot": "·", "div": "÷", "$": "$", "%": "%",
    " ": " ", ",": " ", ";": " ", "left": "", "right": "",
}
_TEXT_GROUP = re.compile(r"\\(?:text|mathrm)(?![A-Za-z])\s*")


def _unwrap_text_groups(text: str) -> str:
    output: list[str] = []
    cursor = 0
    for match in _TEXT_GROUP.finditer(text):
        if match.start() < cursor or match.end() >= len(text) or text[match.end()] != "{":
            continue
        depth = 0
        end = match.end()
        for end in range(match.end(), len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        if depth:
            continue
        output.extend((text[cursor:match.start()], _unwrap_text_groups(text[match.end() + 1:end])))
        cursor = end + 1
    output.append(text[cursor:])
    return "".join(output)


def _aligned_rows(value: str) -> str:
    if not (value.startswith(r"\begin{aligned}") and value.endswith(r"\end{aligned}")):
        return value
    body = value[len(r"\begin{aligned}"):-len(r"\end{aligned}")]
    if r"\begin" in body or r"\end" in body:
        return value
    output: list[str] = []
    depth = 0
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            pair = body[index:index + 2]
            output.append("\n" if pair == "\\\\" and depth == 0 else pair)
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        output.append(" " if char == "&" and depth == 0 else char)
        index += 1
    return "".join(output)


def normalize_formula_evidence_v2(raw: str) -> str:
    """Normalize renderer-supported spelling without erasing math structure.

    Numeric ``{,}`` is only a formatting group and renders as a comma. Removing
    those braces prevents one displayed number being counted as two. General
    braces remain because they delimit scripts and fraction arguments.
    """
    value = normalize_formula_wrapper(raw)
    value = _aligned_rows(value)
    # Tokenize commands before unwrapping: \times\text{weight} must not turn
    # into an unrecognized \timesweight command.
    value = re.sub(r"\\[A-Za-z]+|\\[ $%,;]", lambda m: _COMMANDS.get(m[0][1:], m[0]), value)
    value = _unwrap_text_groups(value)
    value = re.sub(r"(?<=\d)\{,\}(?=\d{3}(?:\D|$))", ",", value)
    value = value.replace("–", "-").replace("−", "-")
    return "\n".join(re.sub(r"[^\S\r\n]+", " ", line).strip() for line in value.splitlines()).strip()


__all__ = ["normalize_formula_evidence_v2"]
