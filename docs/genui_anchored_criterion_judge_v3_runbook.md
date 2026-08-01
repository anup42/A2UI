# GenUI Anchored Criterion Judge (GACJ) v3

## Purpose

GACJ v3 produces a criterion-referenced absolute UI score without asking the
LLM to invent a dimension or headline number. The source response is treated as
reference content. The judge evaluates how well the rendered UI represents it;
the judge does not evaluate source factuality or writing quality.

The protocol preserves the existing blinded two-pass boundary:

1. screenshot-only: dimensions 3-10;
2. source-conditioned: dimensions 1-2, released only after pass 1 is sealed.

## Why a new version

V2 asked the model to interpolate a 0-100 score for each dimension. Blind
repeats produced ICC(A,1)=0.749 and repeat MAE=5.33, failing the frozen
reliability gate. V3 replaces free interpolation with five frozen observable
criteria per dimension.

V2 files and labels remain immutable. V3 uses separate schemas, prompts,
filenames, fingerprints, and outputs.

## Scoring

For every criterion, the judge selects one level:

| Level | Value | Meaning |
|---:|---:|---|
| 4 | 100 | Reference-quality; no material defect |
| 3 | 75 | Production-ready; minor/localized defects |
| 2 | 50 | Usable, but material improvement required |
| 1 | 25 | Severely deficient |
| 0 | 0 | Fails or no usable evidence |

Every dimension contains five equally weighted criteria. Therefore:

\[
B_d=\frac{1}{5}\sum_{k=1}^{5}25L_{dk}
\]

where `L_dk` is the integer anchor level from 0 to 4.

The judge also records independent defects. Let `C_d` be the frozen ceiling for
the most severe defect:

| Severity | Ceiling |
|---|---:|
| none | 100 |
| minor | 95 |
| moderate | 80 |
| major | 55 |
| critical | 25 |
| broken | 0 |

The host computes:

\[
S_d=5\left\lfloor\frac{\min(B_d,C_d)}{5}\right\rfloor.
\]

The LLM must not return `S_d`.

The raw source-representation and rendered-UX scores retain the frozen v2
weights:

\[
R=\frac{0.18S_1+0.14S_2}{0.32}
\]

\[
U=\frac{0.14S_3+0.12S_4+0.10S_5+0.10S_6+0.08S_7+0.04S_8+0.05S_9+0.05S_{10}}{0.68}
\]

\[
J_{raw}=\sum_{d=1}^{10}w_dS_d.
\]

Fatal policy findings are kept separate:

\[
J_{policy}=\min(J_{raw},C_{policy}).
\]

Use `J_raw` for reliability, calibration, correlations, and regression
monitoring. Use `J_policy` only for explicitly named release policy decisions.

## Evidence completeness

`not_observable` is allowed only when authorized evidence is genuinely absent.
A not-observable criterion prevents publication of a complete absolute score.
The host never silently gives it 100 and never renormalizes it away.

## Files

- `dataset/src/pipeline/genui_judge/criterion_protocol_v3.py`
- `dataset/src/pipeline/genui_judge/criterion_judgments_v3.py`
- `dataset/src/pipeline/genui_judge/criterion_packets_v3.py`
- `dataset/src/pipeline/genui_judge/criterion_analysis_v3.py`
- `dataset/schema/genui_anchored_criterion_packet_v3.schema.json`
- `dataset/schema/genui_anchored_criterion_judgment_v3.schema.json`
- `dataset/prompts/genui_anchored_criterion_judge_v3.md`
- `dataset/prompts/genui_anchored_criterion_judge_task_v3.md`
- `dataset/scripts/run_genui_anchored_criterion_judge_v3.py`

## Commands

Print the frozen protocol:

```bash
python dataset/scripts/run_genui_anchored_criterion_judge_v3.py protocol
```

Build a fresh blinded 192-occurrence workspace from the supplied 96-repeat
bundle:

```bash
python dataset/scripts/run_genui_anchored_criterion_judge_v3.py \
  build-repeat96-workspace REVIEW_BUNDLE OUTPUT_DIR
```

Run the non-LLM replay/data-path audit:

```bash
python dataset/scripts/run_genui_anchored_criterion_judge_v3.py \
  backtest-repeat96 REVIEW_BUNDLE OUTPUT_DIR
```

Validate one LLM output:

```bash
python dataset/scripts/run_genui_anchored_criterion_judge_v3.py \
  validate-pass judgment.json
```

Append passes in protocol order:

```bash
python dataset/scripts/run_genui_anchored_criterion_judge_v3.py \
  append-pass BENCHMARK_DIR judgment.json
```

## Interpreting the 96-sample replay

The replay decomposes existing v2 dimension numbers into criterion anchor
levels solely to verify exact host formulas and data handling. It is not a new
visual judgment. Consequently, replayed ICC/MAE/bias must equal v2 and cannot be
used to claim that v3 improved reliability.

A valid v3 reliability claim requires fresh independent judgments for both
occurrences of all 96 pairs, with repeat identity hidden, paired occurrences in
different fresh tasks, and reliability calculated before adjudication.

Frozen gate:

- ICC(A,1) >= 0.85;
- MAE <= 5.0;
- absolute bias < 2.0.

All three must pass.
