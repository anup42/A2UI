# Bixby corrected-FP16 handoff - 2026-09-30

The local `C:/Users/anupk/Downloads/Bixby_18Sep` source now consumes GenUICraft
0.5.8 and can select the corrected trained rank-64 E2B model with GPU inference
and MTP. No Bixby APK was built or installed in this delivery.

## Delivered files

- Code/AAR ZIP: [Bixby-GenUICraft-FP16-0.5.8-20260930.zip](../../artifacts/Bixby-GenUICraft-FP16-0.5.8-20260930.zip)
- Model, delivered separately: [model-fp16-corrected.litertlm](../../artifacts/models/model-fp16-corrected.litertlm)
- Model sidecar for other SDK hosts: [model-fp16-corrected.litertlm.fp16.json](../../artifacts/models/model-fp16-corrected.litertlm.fp16.json)
- [Archive/file hashes](delivery-verification.json)
- [Host test details](host-validation.json)

The ZIP contains 31 files under `Bixby_18Sep/`, preserving the project structure.
Copy that folder's contents into the Bixby root and merge any newer user edits.
It includes six modified host/build/documentation files, the new trusted manifest,
precision strings, model staging helper, handoff guide and file manifest, plus the
20 files comprising the published SDK 0.5.8 Maven artifact. The large model is not
inside the code ZIP.

The code ZIP is 11,462,715 bytes. Its SHA-256 is
`1f1305fcfd831707529f683714ef1c96dc4ac6c7e4c45f4f94a67ed2551faa45`.
The model is 2,682,454,016 bytes. Its SHA-256 is
`7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373`.

## Implementation

- Bixby's importer recognizes the original FP32 package and the prepared FP16
  package by full length and SHA-256, and preserves separate model files and
  verification records. Successful import selects the matching precision unless
  a newer explicit choice was made while copying.
- Corrected import installs the trusted adjacent manifest from the Common asset.
  Users select only the model file. Missing/changed sidecars are excluded from
  cached readiness and restored after model verification.
- Settings adds FP32 / FP16 (corrected, experimental), keeps the independent MTP
  control, and warns when FP16 is selected with MTP off. Existing FP32 imports and
  MTP preferences remain valid.
- Renderer explicitly passes the selected precision into the SDK trained profile.
  Precision and MTP changes invalidate the current generation/session. The SDK
  continues to own prompting, inference, numerical correction, recovery and rendering.
- The AAR contains the native correction library and JNI consumer rules. The
  transitive LiteRT-LM runtime stays pinned to 0.16.1.

## Installation and use

Rebuild and install Bixby in its internal build environment. Stage the model in
Downloads manually or use the ZIP's `scripts/deploy_genuicraft_fp16.ps1` helper.
Open Settings -> Generative visual answers -> Trained visual model and import
`model-fp16-corrected.litertlm`. Corrected FP16 is selected after verification.
Enable Faster generation (MTP) and generative visual answers. Full instructions
are in `docs/genuicraft_fp16_handoff.md` inside the ZIP.

## Verification and limits

- Actual modified Common, Settings and Renderer integration sources compile in an
  isolated Android harness against the published SDK 0.5.8; proprietary surrounding
  types are stubbed. The compiled source bytes match the delivered files.
- 12 Common provisioning/settings tests and 2 manager session-change tests passed:
  **14 tests, zero failures, zero errors, zero skipped**.
- Tests cover pinned model recognition, FP32 migration, MTP persistence, separate
  model files, rejected imports preserving files/sidecars, trusted sidecar readiness,
  missing/invalid model disablement, and configuration invalidation. The sparse-file
  cache test does not claim to verify model contents.
- Full-size successful import and a concurrent explicit selection during a successful
  full-size import were not executed in the harness. Rebuilt-Bixby device validation
  remains required. No Bixby end-to-end claim is made.
- Model SHA-256, AAR ZIP integrity/native correction entry, XML resources, PowerShell
  helper syntax, source credential-pattern scan and ZIP exclusion checks passed.
- Every archive entry matches the staged payload and the applied Bixby checkout.

The prior test-app FP16+MTP three-case pilot remains the native execution evidence;
this delivery did not repeat that benchmark. FP16+MTP worked on those cases, while
an MTP-off case produced repetitive output. See the [pilot report](../20260930_fp16_app/REPORT.md).
