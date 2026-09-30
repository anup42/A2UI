A2UI / GenUI-LM: v10, v11 and v11s data and changes
Release date: 30 September 2026

CONTENTS
training/data/train/v10: original comparison baseline, byte-identical to the source.
training/data/train/v11: screened/repaired v10 plus admitted new data.
training/data/train/v11s: admitted new data only, with its own train/validation splits.
training/scripts and training/tests: the 14 code additions for this dataset task.
training/reports/dataset_v11_20260930: statistics, cleaning decisions, repair evidence,
verification, original-archive provenance and exact policy/reference snapshots.
CHANGES.patch and CHANGES.json: code additions and their checksums.
PACKAGE_MANIFEST.json: byte sizes and SHA-256 checksums of every other archive entry.
package_tools/build_zip_release.py: the standard-library ZIP builder/verifier.

ROW COUNTS
Version  Training  Validation  Total
v10      91,115    1,862       92,977
v11      200,214   4,089       204,303
v11s     109,121   2,229       111,350

USAGE
Extract into a new directory. Entries use repository-relative paths so the version
folders and listed code additions can be copied into the target A2UI/GenUI-LM checkout.
Use v11/train.jsonl for merged training and v11/val.jsonl for merged validation.
Use v11s/val.jsonl to measure performance on only unseen newly generated data.
v11s training/validation files are exactly the new-data suffixes of the v11 splits;
these versions overlap and should not be concatenated again.
For code changes alone, review CHANGES.patch or the 14 files listed in CHANGES.json.
The reference files under policy_snapshot are historical audit evidence.

QUALITY AND REPRODUCTION
131,400 new Stage 3 input records yielded 111,350 retained training pairs:
109,799 KEEP, 1,551 REPAIR and 20,050 QUARANTINE decisions.
97,975 retained rows cleared an original generator review flag under the recorded
source/renderer reassessment. Ten v10 held rows were recovered, and 24 remain excluded
from v11. Original v10 is preserved separately.
The release passed 75 regression tests and the independent dataset verification.
The manifests label v11 and v11s offline_verified_candidate. Model tokenization,
training and device performance testing have not been run. Source factual accuracy
has not been independently checked; source review provenance is retained.
Open training/reports/dataset_v11_20260930/REPORT.html for the comparison and full statistics,
and REPRODUCE.txt in the same folder for dataset reconstruction instructions.
Raw download archives and extraction projections must be supplied separately to
rebuild from Stage 3. Their identities/checksums are recorded in archive_manifest.json.
This archive includes the finalized datasets and sealed audit; redundant candidate
spools, scratch environments, checkpoints and original raw download archives are
outside the package scope.

INTEGRITY
The adjacent .zip.sha256 file identifies the complete ZIP. The adjacent
.verification.json records a complete decompression/CRC and SHA-256 pass of every
entry. ZIP64 supports the v11 training file, which exceeds 4 GiB.
