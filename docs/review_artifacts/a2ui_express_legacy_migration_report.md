# Legacy dataset migration report

Command under test:

```text
python dataset/scripts/migrate_legacy_dataset_to_a2ui_express.py <input> \
  --output-dir <output> --strict
```

The tool accepts JSON, JSONL, directories, and stdin JSONL. Each eligible
record is decoded through the isolated legacy boundary, validated into the
canonical graph, encoded to Express, strictly decoded, compiled to standard
A2UI v1 wire, decoded again, and admitted only when all semantic hashes match.
The original completion is retained only as `legacy_completion` with
`legacy_source_format` and `legacy_source_hash`; `genui_json` is the Express
completion.

Evidence run: a temporary JSONL projection of the repository's
`dataset/tests/fixtures/intent_flat_specs_v2.json` (32 FlatSpec fixtures).

| run | accepted | rejected | mode | result |
| --- | ---: | ---: | --- | --- |
| strict conversion | 32 | 0 | write | PASS |
| strict resume | 0 new | 0 | `--resume` | PASS; all prior source hashes skipped deterministically |

The generated temporary manifest reported:

```json
{
  "target_format": "a2ui_express_v1",
  "wire_format": "a2ui_v1_wire",
  "accepted": 32,
  "rejected": 0,
  "rejection_policy": "strict",
  "semantic_hash_policy": "Express decode and standard A2UI wire decode must equal legacy canonical hash"
}
```

Compact IR v2 is recognized only by this explicitly isolated migration tool;
the active dataset, training, metrics, and Android modules do not import the
decoder. A rejected record is written with a machine-readable reason and strict
mode returns non-zero.
