"""Offline source checks with an explicit, limited verification scope.

No network, model calls, text rewriting, or asset existence requirements. A
contract is optional trusted input supplied with the query, never a model's
self-assessment. Passing its declared checks is not arbitrary fact checking.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

VERSION = "source-quality-v1.1"
MODALITIES = {"answer", "document", "dashboard", "interactive_tool"}
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_ACTION = re.compile(r"^\s*Action:\s*\[Button:\s*([^\]\n]+)\]\s*(.*?)\s*$", re.M)
_HTTP_URL = re.compile(r"https?://[^\s<>\"\]\)]+")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric input")
    result = Decimal(str(value))
    if not result.is_finite() or abs(result) > Decimal("1e30"):
        raise ValueError("numeric input must be finite and within 1e30")
    return result


def calculate(spec: dict[str, Any]) -> Decimal:
    """Evaluate a small typed arithmetic language, never Python/eval strings."""
    if not isinstance(spec, dict):
        raise ValueError("calculation must be an object")
    op = spec.get("op")
    with localcontext() as context:
        context.prec = 40
        if op in {"sum", "mean", "product"}:
            values = spec.get("values")
            if not isinstance(values, list) or not 1 <= len(values) <= 1000:
                raise ValueError("values must contain 1..1000 numbers")
            numbers = [_decimal(value) for value in values]
            if op == "sum":
                return sum(numbers, Decimal(0))
            if op == "mean":
                return sum(numbers, Decimal(0)) / len(numbers)
            result = Decimal(1)
            for number in numbers:
                result *= number
            return result
        if op == "weighted_mean":
            values, weights = spec.get("values"), spec.get("weights")
            if not isinstance(values, list) or not isinstance(weights, list) or not 1 <= len(values) <= 1000 or len(values) != len(weights):
                raise ValueError("weighted_mean requires equally sized nonempty values/weights")
            v, w = list(map(_decimal, values)), list(map(_decimal, weights))
            if any(weight < 0 for weight in w) or sum(w) <= 0:
                raise ValueError("weights must be nonnegative with positive total")
            return sum((a * b for a, b in zip(v, w)), Decimal(0)) / sum(w)
        if op in {"difference", "ratio", "percentage"}:
            a, b = _decimal(spec["a"]), _decimal(spec["b"])
            if op == "difference":
                return a - b
            if b == 0:
                raise ValueError("zero denominator")
            return a / b * (100 if op == "percentage" else 1)
        if op in {"loan_payment", "loan_interest", "annuity_future_value"}:
            principal, rate = _decimal(spec["amount"]), _decimal(spec["period_rate"])
            periods = spec["periods"]
            if isinstance(periods, bool) or not isinstance(periods, int) or not 1 <= periods <= 12000:
                raise ValueError("periods must be an integer in 1..12000")
            if principal < 0 or not 0 <= rate <= 1:
                raise ValueError("amount must be nonnegative; period_rate must be in 0..1")
            factor = (1 + rate) ** periods
            if op == "annuity_future_value":
                timing = spec.get("payment_timing", "end")
                if timing not in {"start", "end"}:
                    raise ValueError("payment_timing must be start or end")
                result = principal * periods if rate == 0 else principal * (factor - 1) / rate
                return result * (1 + rate if timing == "start" else 1)
            payment = principal / periods if rate == 0 else principal * rate * factor / (factor - 1)
            return payment if op == "loan_payment" else payment * periods - principal
    raise ValueError(f"unsupported calculation operation: {op}")


def infer_modality(query: str) -> str:
    text = query.casefold()
    if re.search(r"\b(interface|app|scanner)\b", text) and re.search(r"\b(button|scan|copy|save|interactive)\b", text):
        return "interactive_tool"
    if re.search(r"\b(dashboard|chart|graph|visuali[sz])", text):
        return "dashboard"
    if re.search(r"\b(letter|essay|document|article|brochure|resume)\b", text):
        return "document"
    return "answer"


def parse_actions(text: str) -> list[dict[str, str]]:
    return [{"label": match.group(1).strip(), "destination": match.group(2).strip().removeprefix("<").removesuffix(">").strip()}
            for match in _ACTION.finditer(text)]


def _url_issue(value: str) -> str | None:
    if not value or any(char.isspace() for char in value):
        return "empty_or_whitespace_destination"
    # Stage2 action destinations must be executable references, not Stage3's
    # symbolic URL tokens (including malformed/truncated variants).
    if re.search(r"\[(?:URL|ACTION_URL|SOURCE_URL)_\d+", value, re.IGNORECASE):
        return "unresolved_destination_placeholder"
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"}:
            # Literal ellipses in a destination path are an explicit truncation
            # marker in generated sources. Do not decode or reject query text:
            # a search for '...' is valid and is not a truncated destination.
            if any(marker in parsed.netloc or marker in parsed.path for marker in ("...", "…")):
                return "truncated_http_destination"
            host = parsed.hostname or ""
            if not host or host.startswith(".") or ".." in host or parsed.username is not None:
                return "invalid_http_destination"
            parsed.port  # Validate a malformed port, without contacting the host.
        elif parsed.scheme not in {"app", "action", "mailto", "tel"}:
            return "unsupported_destination_scheme"
    except ValueError:
        return "invalid_destination"
    return None


def _bound_numbers(text: str, label: str) -> list[Decimal]:
    """Read explicit 'Label: number' and '| Label | number |' bindings.

    Formula prose is intentionally not interpreted. Every matched binding is
    checked so a correct table does not hide a contradictory summary line.
    """
    escaped = re.escape(label)
    pattern = re.compile(rf"(?:^|\|)\s*(?:[-*]\s+)?(?:\*\*)?{escaped}(?:\*\*)?\s*(?::|\|)\s*(?:\*\*)?[$€£₹]?\s*([-+]?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)", re.I | re.M)
    return [Decimal(match.group(1).replace(",", "")) for match in pattern.finditer(text)]


def _clock(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("schedule clock must be HH:MM on one day")
    hours, minutes = map(int, value.split(":"))
    return 60 * hours + minutes


def assess_source_quality(query: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Assess explicit facts/calculations/constraints/capabilities conservatively.

    Findings outside a reviewed contract request review rather than claiming
    factual failure. Assets and placeholder media never affect eligibility.
    """
    text = str(response_text)
    query_text = str(query.get("query_text") or "")
    contract = query.get("source_contract")
    findings: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def finding(code: str, severity: str, **details: Any) -> None:
        findings.append({"code": code, "severity": severity, **details})

    def checked(kind: str, identifier: str, passed: bool, **details: Any) -> None:
        checks.append({"kind": kind, "id": identifier, "passed": passed, **details})
        if not passed:
            finding(f"{kind}_failed", "error", id=identifier, **details)

    if not isinstance(contract, dict):
        if contract is not None:
            finding("invalid_source_contract", "error")
        contract = {}
    verification = contract.get("verification")
    reviewed = (contract.get("version") == 1 and isinstance(verification, dict)
                and verification.get("status") == "reviewed"
                and isinstance(verification.get("reference"), str) and bool(verification["reference"].strip()))
    if not reviewed:
        finding("source_not_independently_reviewed", "review")
    if contract and contract.get("version") != 1:
        finding("unsupported_source_contract_version", "error")
    modality = contract.get("modality", infer_modality(query_text))
    if not isinstance(modality, str) or modality not in MODALITIES:
        finding("invalid_modality", "error")
        modality = infer_modality(query_text)
    query_quality = query.get("query_quality") or {}
    if query.get("source") == "fallback":
        finding("fallback_query", "error")
    if isinstance(query_quality, dict) and query_quality.get("near_duplicate_candidates"):
        finding("query_near_duplicate_candidate", "review")

    def items(key: str) -> list[dict[str, Any]]:
        values = contract.get(key, [])
        if not isinstance(values, list) or len(values) > 1000 or any(not isinstance(v, dict) for v in values):
            finding("invalid_contract_collection", "error", field=key)
            return []
        return values

    for fact in items("facts"):
        identifier = str(fact.get("id", ""))
        value = fact.get("value")
        if not identifier or not isinstance(value, str) or not value:
            finding("invalid_fact", "error", id=identifier)
            continue
        provenance = fact.get("provenance", {})
        kind = provenance.get("kind") if isinstance(provenance, dict) else None
        if kind == "query":
            checked("fact_source", identifier, value in query_text, value=value)
        elif kind == "reference":
            if not reviewed or not provenance.get("reference"):
                finding("unverified_fact_reference", "review", id=identifier)
        elif kind != "hypothetical":
            finding("missing_fact_provenance", "review", id=identifier)
        if fact.get("required", True):
            checked("fact_preservation", identifier, value in text, value=value)
        # Bind a supplied value to its declared role, not just token presence.
        if fact.get("label"):
            label = str(fact["label"])
            pattern = re.compile(rf"(?:^|\|)\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*(?::|\|)\s*(?:\*\*)?{re.escape(value)}(?=\s|\*|\||$)", re.I | re.M)
            checked("fact_role", identifier, bool(pattern.search(text)), label=label, value=value)

    for calculation in items("calculations"):
        identifier = str(calculation.get("id", ""))
        label = calculation.get("result_label")
        try:
            if not identifier or not isinstance(label, str) or not label:
                raise ValueError("calculation id and result_label are required")
            expected = calculate(calculation)
            tolerance = _decimal(calculation.get("tolerance", "0.005"))
            if tolerance < 0:
                raise ValueError("tolerance must be nonnegative")
            actual = _bound_numbers(text, label)
            passed = bool(actual) and all(abs(value - expected) <= tolerance for value in actual)
            checked("calculation", identifier, passed, result_label=label,
                    expected=str(expected), observed=list(map(str, actual)), tolerance=str(tolerance))
        except (ValueError, KeyError, TypeError, InvalidOperation, ArithmeticError) as exc:
            finding("invalid_calculation_contract", "error", id=identifier, reason=str(exc))

    for constraint in items("constraints"):
        identifier = str(constraint.get("id", ""))
        kind = constraint.get("kind")
        try:
            if not identifier:
                raise ValueError("constraint id is required")
            if kind in {"required_text", "forbidden_text"}:
                value = constraint["value"]
                if not isinstance(value, str) or not value:
                    raise ValueError("constraint value must be nonempty text")
                present = value.casefold() in text.casefold()
                checked("constraint", identifier, present if kind == "required_text" else not present, constraint_kind=kind, value=value)
            elif kind == "number_range":
                actual = _bound_numbers(text, str(constraint["label"]))
                low, high = _decimal(constraint["min"]), _decimal(constraint["max"])
                if low > high:
                    raise ValueError("range min exceeds max")
                checked("constraint", identifier, bool(actual) and all(low <= value <= high for value in actual), constraint_kind=kind, observed=list(map(str, actual)))
            elif kind == "schedule":
                start, end = _clock(constraint["window_start"]), _clock(constraint["window_end"])
                if end <= start:
                    raise ValueError("schedule requires an increasing same-day window")
                intervals = []
                for line in text.splitlines():
                    match = re.search(r"\b(\d\d:\d\d)\s*[-–]\s*(\d\d:\d\d)\b", line)
                    if match:
                        intervals.append((_clock(match[1]), _clock(match[2])))
                intervals.sort()
                passed = bool(intervals) and all(start <= a < b <= end for a, b in intervals)
                passed = passed and all(previous[1] <= current[0] for previous, current in zip(intervals, intervals[1:]))
                checked("constraint", identifier, passed, constraint_kind=kind, observed_intervals=intervals,
                        total_minutes=sum(b - a for a, b in intervals))
            else:
                raise ValueError(f"unsupported constraint kind: {kind}")
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            finding("invalid_constraint_contract", "error", id=identifier, reason=str(exc))

    declared_actions = items("actions")
    by_label: dict[str, dict[str, Any]] = {}
    for action in declared_actions:
        label, destination = action.get("label"), action.get("destination")
        if not isinstance(label, str) or not label or not isinstance(destination, str) or _url_issue(destination):
            finding("invalid_action_contract", "error", label=label)
            continue
        if label.casefold() in by_label:
            finding("duplicate_action_contract_label", "error", label=label)
            continue
        if not isinstance(action.get("mode"), str) or action.get("mode") not in {"navigation", "capability", "mock"}:
            finding("invalid_action_mode", "error", label=label)
        if action.get("mode") == "capability" and (not action.get("capability_id") or not isinstance(action.get("parameters"), dict)):
            finding("unbound_action_capability", "error", label=label)
        by_label[label.casefold()] = action
    actions = parse_actions(text)
    observed_labels = set()
    query_urls = set(_HTTP_URL.findall(query_text))
    for action in actions:
        label, destination = action["label"], action["destination"]
        observed_labels.add(label.casefold())
        issue = _url_issue(destination)
        if issue:
            finding(issue, "error", label=label, destination=destination)
        declared = by_label.get(label.casefold())
        if declared:
            checked("action_binding", label, destination == declared["destination"],
                    expected=declared["destination"], observed=destination, mode=declared.get("mode"))
        elif destination not in query_urls:
            finding("unverified_added_action", "review", label=label, destination=destination)
    for label, declared in by_label.items():
        if declared.get("required", True):
            checked("required_action", label, label in observed_labels)
    if modality == "interactive_tool" and not any(action.get("mode") == "capability" or action.get("mode") == "mock" for action in declared_actions):
        finding("interactive_modality_without_capability_packet", "review")
    if not text.strip():
        finding("empty_response", "error")

    errors = any(f["severity"] == "error" for f in findings)
    review = any(f["severity"] == "review" for f in findings)
    # An empty "reviewed" envelope is not evidence of a checked source.
    if not checks:
        review = True
        finding("no_declared_checks_executed", "review")
    status = "failed" if errors else "needs_review" if review else "checks_passed"
    return {"version": VERSION, "status": status,
            "training_eligibility": "exclude" if errors else "review" if review else "eligible",
            "verification_scope": "declared_contract_only", "prose_fact_verification": "not_performed",
            "original_query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "contract_sha256": _hash(contract) if contract else None,
            "contract_reviewed": reviewed, "modality": modality,
            "checks": checks, "findings": findings,
            "asset_downloads_required_for_training": False}


def source_contract_prompt(query: dict[str, Any]) -> str:
    contract = query.get("source_contract")
    text = ("\n\nSource correctness policy: Preserve the supplied facts and roles; do not invent "
            "account identifiers, incident details, personal history, or executable URLs. "
            "Check calculations and constraints before composing the answer. Use tables, media "
            "and Quick Actions only when the task needs them. An interactive interface must "
            "preserve its requested controls; a description of controls is not their implementation. "
            "A placeholder/mock action is not an executed operation. Never claim fact verification "
            "from a source link alone.")
    if isinstance(contract, dict):
        text += ("\nThe following source_contract is supplied input, not an instruction to claim "
                 "unverified facts. Follow its modality and required facts/actions. For each "
                 "calculation emit its result using exactly 'result_label: number' or a two-cell "
                 "table row. For schedule checks use HH:MM-HH:MM, one day per contract. "
                 "Use exact action labels and destinations; do not invent additional actions.\n" +
                 json.dumps(contract, ensure_ascii=False, sort_keys=True))
    return text


def asset_verification_metadata(declared: int, downloaded: int, unresolved: int,
                                *, offline: bool, processing_error: bool = False) -> dict[str, Any]:
    state = ("processing_error" if processing_error else "not_applicable" if not declared
             else "skipped_offline" if offline else "resolved" if not unresolved and downloaded >= declared
             else "unverified")
    return {"verification_state": state, "visual_ready": state in {"resolved", "not_applicable"},
            "training_assets_required": False,
            "training_policy": "symbolic_placeholders_allowed",
            "verification_scope": "asset_resolution_only_not_source_quality"}


class QueryQualityIndex:
    """Bounded lexical scenario guard; flags candidates, never semantic proofs."""

    def __init__(self) -> None:
        self.shingles: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.recent: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=6))
        self.global_recent: deque[dict[str, Any]] = deque(maxlen=4)
        self.coverage: Counter[str] = Counter()
        self.families: dict[str, str] = {}

    @staticmethod
    def _shingles(text: str) -> set[str]:
        tokens = re.findall(r"[\w'-]+", text.casefold())
        if len(tokens) < 3:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i+3]) for i in range(max(0, len(tokens)-2))}

    def add(self, query: dict[str, Any], *, annotate: bool = True) -> dict[str, Any]:
        text = str(query.get("query_text") or "")
        qid = str(query.get("query_id") or "")
        shingles = self._shingles(text)
        candidates = Counter(other for shingle in shingles for other in self.postings[shingle])
        matches = []
        for other, overlap in candidates.most_common(100):
            union = len(shingles) + len(self.shingles[other]) - overlap
            score = overlap / union if union else 0
            if score >= .65:
                matches.append({"query_id": other, "word_trigram_jaccard": round(score, 6)})
        modality = infer_modality(text)
        family = self.families[matches[0]["query_id"]] if matches else "scenario_" + _hash(sorted(shingles))[:24]
        quality = {"version": VERSION, "modality": modality,
                   "scenario_fingerprint": _hash(sorted(shingles)),
                   "scenario_fingerprint_scope": "literal_word_trigrams_including_numbers",
                   "near_duplicate_candidates": matches[:5],
                   "duplicate_check_scope": "lexical_candidates_not_semantic_equivalence",
                   "scenario_family_scope": "lexical_candidate_group_for_split_isolation",
                   "training_eligibility": "exclude" if query.get("source") == "fallback" else "review",
                   "intent_modality_count_before": self.coverage[f"{query.get('intent')}:{modality}"]}
        self.shingles[qid] = shingles
        self.families[qid] = family
        for shingle in shingles:
            self.postings[shingle].add(qid)
        self.coverage[f"{query.get('intent')}:{modality}"] += 1
        summary = {"intent": query.get("intent"), "query": text[:180], "modality": modality}
        self.recent[str(query.get("intent"))].append(summary)
        self.global_recent.append(summary)
        if annotate:
            query = {**query, "query_quality": quality, "scenario_family_id": family}
        return query

    def prompt_context(self, intent: str) -> str:
        recent = list(self.recent[intent])
        other_recent = [row for row in self.global_recent if row["intent"] != intent][-2:]
        return ("\n\nScenario diversity guard: change the actual task, entities, constraints, "
                "data shape and interaction requirements, not only its opening phrase. "
                "Use offline supplied facts for time-sensitive scenarios. Do not fabricate "
                "verified fact/capability packets. Avoid these recently accepted scenarios:\n" +
                json.dumps(recent, ensure_ascii=False) + "\nRecent other-intent scenarios: " +
                json.dumps(other_recent, ensure_ascii=False) + "\nCurrent intent/modality counts: " +
                json.dumps({key: value for key, value in self.coverage.items() if key.startswith(intent + ":")}, sort_keys=True))


class QualityQueryWriter:
    """Annotate all Stage1 append paths, including fallback paths."""

    def __init__(self, writer: Any, index: QueryQualityIndex, prompt_path: str, prompt_text: str) -> None:
        self.writer, self.index = writer, index
        self.prompt_path = prompt_path
        self.prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        header = next((line.strip().lstrip("#").strip() for line in prompt_text.splitlines() if line.strip()), "")
        self.prompt_version = header if re.fullmatch(r"query_gen[\w.-]*", header) else prompt_path.replace("\\", "/").rsplit("/", 1)[-1]

    def append(self, record: dict[str, Any]) -> None:
        enriched = self.index.add(record)
        enriched["gen"] = {**enriched.get("gen", {}), "prompt_path": self.prompt_path,
                           "prompt_version": self.prompt_version,
                           "prompt_template_sha256": self.prompt_hash}
        self.writer.append(enriched)
