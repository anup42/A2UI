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
    # Digit-delimited {,} also covers Indian grouping (17{,}20{,}000).
    # Retain the comma and every other grouping character.
    value = re.sub(r"(?<=\d)\{,\}(?=\d)", ",", value)
    value = value.replace("–", "-").replace("−", "-")
    return "\n".join(re.sub(r"[^\S\r\n]+", " ", line).strip() for line in value.splitlines()).strip()


def formula_syntax_errors_v2(raw: str) -> tuple[str, ...]:
    """Reject unsupported renderer commands, never erase or repair them."""
    value = _aligned_rows(normalize_formula_wrapper(raw))
    allowed = {*_COMMANDS, "frac", "text", "mathrm"}
    errors = {
        "unsupported_command:" + match[1]
        for match in re.finditer(r"\\([A-Za-z]+|.)", value)
        if match[1] not in allowed
    }
    depth = 0
    for char in value:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth < 0:
            errors.add("unbalanced_group")
    if depth:
        errors.add("unbalanced_group")
    return tuple(sorted(errors))


def formula_literal_key_v2(raw: str) -> str:
    """Spelling equivalence only; retain math structure, words and units."""
    label = re.fullmatch(r"([\w][\w /-]*:\s*)\$(.+)(?<!\\)\$", raw.strip())
    if label and re.search(r"[=≈×÷]|\\(?:times|div|approx)(?![A-Za-z])", label[2]):
        raw = label[1] + label[2]
    value = normalize_formula_evidence_v2(raw)
    # Match a signed scalar exponent with or without explicit script braces.
    # Grouped expressions, adjacent digits, and symbolic scripts stay exact.
    value = re.sub(r"\^\{([+-]?\d+)\}(?![\w.,])", r"^\1", value)
    # Bold numeric literals in supplied LaTeX change emphasis only. Bold
    # variables and arbitrary groups may encode distinct mathematical roles.
    value = re.sub(r"\\mathbf\{([$€£¥₹]?[+-]?\d[\d,.]*%?)\}", r"\1", value)
    # Source LaTeX floor/ceiling delimiters and literal Unicode brackets name
    # the same operation. This comparison does NOT add unsupported commands
    # to the renderer syntax allowlist: output must still pass that gate.
    value = re.sub(r"\\(lfloor|rfloor|lceil|rceil)(?![A-Za-z])",
                   lambda m: {"lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉"}[m[1]], value)
    if "=" in value:
        # A word x between numeric operands (possibly carrying unit words)
        # spells multiplication. Standalone variables, e.g. x=2, stay intact.
        value = re.sub(r"(?<!\w)(\d[\d.,]*(?:\s+[A-Za-z]+){0,3})\s+x\s+(?=[\d$€£₹])",
                       lambda m: m[1] + " × ", value)
    return " ".join(re.findall(r"\w+|[^\w\s]", value))


__all__ = ["formula_literal_key_v2", "formula_syntax_errors_v2", "normalize_formula_evidence_v2"]
