#!/usr/bin/env python3
"""Reassess diagnostic generation flags after source-proven v10 repairs.

This is an additional gate, not a replacement for wire, provenance, source,
reference, and v10 semantic checks. It never modifies source or target content.
Generic one-to-one block matching is replaced only when every inferred generic
source unit remains in renderer-visible text (allowing block regrouping and
sentence reorder), with all its numeric anchors preserved. Other flags require
a fresh complete v5.4 score and must clear. Counts, not factual certification,
are the scope of these deterministic checks.
"""
from __future__ import annotations

from functools import lru_cache
from dataclasses import replace
import json
import re
import time
import unicodedata

POLICY = "space-v11-post-v10-repair-reassessment-20260930-v1"
GENERIC_REVIEW_REASONS = frozenset({"content_unit_fidelity", "exact_numbers_dates_units_fbeta"})
_REFERENCE = re.compile(r"<?\[(?:[A-Z_]+URL|URL|[A-Z_]+ASSET)_?\d+\]>?")
_RAW_URL = re.compile(r"(?:https?|ftp)://[^\s<>\"']+")
_ORDERED_PREFIX = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_GROUPED_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)")
_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d+(?:[.,:/\-]\d+)*(?:[%℃°]+)?")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z])")


@lru_cache(maxsize=1)
def _config():
    from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
    return coerce_reward_config_v5_4(None)


def _plain(value):
    """Only remove transport/display syntax; never repair or paraphrase words."""
    text = unicodedata.normalize("NFKC", str(value)).replace("−", "-")
    text = text.replace("**", "").replace("__", "")
    text = _ORDERED_PREFIX.sub("", text)
    text = _REFERENCE.sub(" ", text)
    text = _RAW_URL.sub(" ", text)
    return _GROUPED_NUMBER.sub(lambda match: match[0].replace(",", ""), text)


def _tokens(value):
    text = _plain(value).casefold().replace(">=", "≥").replace("<=", "≤").replace("!=", "≠")
    # Retain comparator/currency/percent symbols and signed numeric facts.
    return " ".join(re.findall(r"[+\-]?\d+(?:[.,:/\-]\d+)*|\w+|[<>≤≥≠=+%$€£¥₹°]", text))


def _numbers(value):
    return set(_NUMBER.findall(_plain(value)))


def _static_card_labels(graph, reachable):
    """Native RenderCard's normal-route literal title/subtitle evidence.

    Do not assume static reachability means dynamic visibility. Conditional or
    repeated graphs, source-link cards, and email-shaped cards stay with the
    existing evaluator until their exact native special route can be proved.
    """
    def dynamic(value):
        if isinstance(value,dict):
            if any(str(key).startswith('$') or str(key).casefold() in {'if','when','visible','visibility','hidden','condition','repeat','template'} for key in value):
                return True
            return any(dynamic(item) for item in value.values())
        return isinstance(value,list) and any(dynamic(item) for item in value)
    if dynamic(graph):
        return []
    elements=graph['elements']
    labels=[]
    for element_id in sorted(reachable):
        element=elements.get(element_id,{})
        if element.get('type')!='Card':
            continue
        subtree=[];pending=[element_id];seen=set()
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key)
            child=elements.get(key,{})
            subtree.append(child)
            pending.extend(child.get('children') or [])
            literal_child=(child.get('props') or {}).get('child')
            if isinstance(literal_child,str):pending.append(literal_child)
        serialized=json.dumps(subtree,ensure_ascii=False)
        if _REFERENCE.search(serialized) or _RAW_URL.search(serialized):
            continue
        if any(child.get('type')=='EmailPreview' for child in subtree):
            continue
        # Native email interception recognizes subject/sender/recipient labels
        # in child Text. This superset avoids crediting a skipped Card title.
        text='\n'.join(str((child.get('props') or {}).get('text','')) for child in subtree)
        if re.search(r'(?im)^\s*(?:subject|from|to|sender|recipient|body)\s*:',text):
            continue
        for key in ('title','subtitle'):
            value=(element.get('props') or {}).get(key)
            if isinstance(value,str) and value.strip():labels.append(value)
    return labels


def visible_content_proof(source, graph):
    """Cheap renderer-aware evidence without cubic/fuzzy block assignment.

    Global concatenation permits adjacent Text splits and Table fields while
    per-source-unit matching retains local word order, including negations.
    Repeated source statements need only one visible copy, as in v10 dedup.
    Never use arbitrary unused state or style/property strings as evidence.
    """
    from pipeline.genui_quality.source_contract_v5_4 import extract_expected_ui_contract_v5_4
    from pipeline.genui_quality.ownership_v5_4 import generic_source_texts
    from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
    from pipeline.genui_quality.graph import audit_renderer_graph

    started = time.perf_counter()
    contract = extract_expected_ui_contract_v5_4(source)
    units = list(generic_source_texts(source, contract))
    # Default v5.4 silently clips literal visible blocks at 4096 characters.
    # Set a finite bound derived from this very source/graph; no new facts or
    # unbounded expansion, and all dynamic-certification failures still hold.
    bound=min(1_000_000,max(_config().max_string_expansion_length,len(source),len(json.dumps(graph,ensure_ascii=False))))
    config=replace(_config(),max_string_expansion_length=bound)
    output = collect_output_evidence_v5_4(graph, audit_renderer_graph(graph), config)
    visible = list(output.output.visible_blocks)
    card_labels=_static_card_labels(graph,output.output.reachable_ids)
    visible.extend(card_labels)
    represented = " " + _tokens("\n".join(visible)) + " "
    # Header labels are bound to every displayed cell in their own table row.
    # Reconstruct those exact bindings, not arbitrary key/value state strings.
    contexts=[]
    for table in output.output_tables:
        for row in table.rows:
            context=" ".join(str(header)+" "+str(row.get(key,"")) for header,key in zip(table.headers,table.keys))
            contexts.append(" "+_tokens(context)+" ")
    # Email field labels can be encoded by the component role instead of Text.
    # Values come from renderer-resolved role evidence, never unused raw props.
    for role in output.role_instances.get("email",()):
        for field,labels in (("subject",("Subject",)),("from",("From","Sender")),("to",("To","Recipient","Recipients"))):
            if role.get(field):
                contexts.extend(" "+_tokens(label+": "+str(role[field]))+" " for label in labels)
        if role.get("content"):
            contexts.extend((" body "," message body "))
    missing_units = []
    missing_numbers = set()
    matched = 0
    visible_numbers = _numbers("\n".join(visible))
    unsupported_numbers = visible_numbers - _numbers(source)
    for unit in units:
        normalized = _tokens(unit)
        fragments = [_tokens(part) for part in _SENTENCE.split(unit)]
        present = not normalized or " " + normalized + " " in represented
        if not present:
            present=any(" "+normalized+" " in context for context in contexts)
        if not present:
            present = bool(fragments) and all(not part or " " + part + " " in represented for part in fragments)
        if not present:
            present=any(all(not part or " "+part+" " in context for part in fragments) for context in contexts)
        if present:
            matched += 1
        else:
            missing_units.append(unit)
        missing_numbers.update(_numbers(unit) - visible_numbers)
    reasons = []
    if not output.dynamic_evidence_complete:
        reasons.append("dynamic_evidence_incomplete")
    if missing_units:
        reasons.append("generic_source_unit_not_exactly_represented")
    if missing_numbers:
        reasons.append("generic_numeric_anchor_not_represented")
    if unsupported_numbers:
        reasons.append("unsupported_visible_numeric_anchor")
    return {"passed": not reasons, "reasons": reasons,
            "source_unit_count": len(units), "exactly_represented_unit_count": matched,
            "visible_block_count": len(visible), "missing_unit_count":len(missing_units),
            "structured_binding_contexts":len(contexts),"evidence_string_bound":bound,
            "native_static_card_label_count":len(card_labels),
            "missing_units": missing_units[:8], "missing_numeric_anchors": sorted(missing_numbers),
            "unsupported_numeric_anchors":sorted(unsupported_numbers),
            "dynamic_evidence_complete": output.dynamic_evidence_complete,
            "elapsed_ms":round((time.perf_counter()-started)*1000,3)}


def reassess(source, target, checked, original_acceptance, url_map=None):
    """Return eligibility, residual reasons, and immutable-audit evidence.

    Call only after source/provenance checks and the v10 repair chain. Missing
    or malformed generation acceptance is not silently turned into approval.
    The optional URL map binds normalized references during full rescoring.
    """
    original = original_acceptance or {}
    if not isinstance(original, dict) or not isinstance(original.get("eligible"), bool):
        return {"eligible":False,"reasons":["missing_generator_acceptance"],"evidence":{"policy":POLICY}}
    blocking = list(original.get("blocking_reasons") or [])
    review = list(original.get("review_reasons") or [])
    if any(not isinstance(reason,str) for reason in blocking+review):
        return {"eligible":False,"reasons":["invalid_generator_acceptance"],"evidence":{"policy":POLICY}}
    evidence = {"policy":POLICY,"original_eligible":original["eligible"],"original_blocking_reasons":blocking,"original_review_reasons":review,"rescored":False}
    if not original["eligible"] and not blocking and not review:
        return {"eligible":False,"reasons":["unexplained_generator_ineligibility"],"evidence":evidence}
    residual = set(blocking+review)
    if blocking or set(review)-GENERIC_REVIEW_REASONS:
        from pipeline.genui_quality._v5_4 import render_artifact_quality_v5_4
        started=time.perf_counter()
        normalized_mapping = {key:(value.get("url","") if isinstance(value,dict) else value) for key,value in (url_map or {}).items()}
        from ir_training.data.url_preprocess import restore_url_placeholders
        # The Express normalizer resolves target tokens through reference_map;
        # bind source to those same exact saved destinations before comparison.
        score_source=restore_url_placeholders(source,normalized_mapping)
        result=render_artifact_quality_v5_4(target,score_source,reference_map=normalized_mapping)
        fresh=result.evidence["training_acceptance"]
        residual=set(fresh.get("blocking_reasons") or [])|set(fresh.get("review_reasons") or [])
        evidence.update({"rescored":True,"rescore_elapsed_ms":round((time.perf_counter()-started)*1000,3),"fresh_acceptance":fresh,"fresh_fidelity_atomics":result.atomics.get("fidelity",{})})
    # Prove both generic signals together. No threshold is lowered and no
    # unrelated action/media/role/table/renderer signal is waived.
    if residual & GENERIC_REVIEW_REASONS:
        proof=visible_content_proof(source,checked.graph)
        evidence["generic_content_proof"]=proof
        if proof["passed"]:
            residual -= GENERIC_REVIEW_REASONS
        else:
            residual.update(proof["reasons"])
    reasons=sorted(residual)
    evidence["cleared_original_reasons"]=sorted(set(blocking+review)-residual)
    return {"eligible":not reasons,"reasons":reasons,"evidence":evidence}
