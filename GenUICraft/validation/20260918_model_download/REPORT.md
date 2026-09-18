# E2B download and APK delivery — 18 September 2026

Implemented in the GenUICraft SDK test screen and installed on both connected Samsung devices: Fold7 SM-F966B and Flip8 SM-F776U. The installed APKs and shared APK have identical SHA-256.

## User flow

Choose **Gemma 4 E2B → Download model**. The explicit Download button fetches the official 2.59 GB model using Android DownloadManager. Progress, cancel, retry, and verification states are displayed. Reopening the screen reconnects to the persisted OS download ID. The app accepts the model only after exact length and SHA-256 verification, then selects it automatically and enables Convert. The optional **Use local file · advanced** path remains available. Provider and source selection persist. GPU+MTP, thinking, metrics, prompts and AAR rendering behavior remain unchanged.

Model URL is pinned to Hugging Face repository revision `b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1`. Expected size: **2,588,147,712 bytes**. SHA-256: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`. Weights are downloaded separately from the APK and retained under app-scoped external storage.

## Verified results

- **12 focused JVM tests passed**: seven download integrity/recovery/cancellation tests, two UI readiness tests, and three token metrics tests.
- Fold metrics UI regression: **1 instrumentation test passed**.
- Flip: the normal UI downloaded the entire model over Wi-Fi. Captured progress includes 0%, 72%/1.88 GB, and Ready/verified. Independent device SHA-256 matched the pinned model. Source/provider selection persisted after app replacement/reopen.
- Fold: reused the already verified model by copying it into the managed location, then exercised in-app verification. No second 2.59 GB mobile-data download was performed.
- Captured UI hierarchies confirm Convert was disabled during the download and enabled after Ready on both devices. Renderer-only remained available.
- Normal Convert + render succeeded for BXP-001 on both devices using the managed model: Flip **17.699 seconds**, **80.43 native decode token/s**; Fold **19.805 seconds**, **76.29 native decode token/s**. Both reported 1,125 input tokens, 915 output tokens, one attempt, and rendered the weather result. These are smoke checks, not a new 50-case benchmark or controlled device comparison.
- Post-verification bookkeeping errors preserve verified bytes for recovery. Invalid size/hash cannot expose Ready. Cancellation cannot promote a partial file.

Screens: [Flip verified download](flip_ready.png), [Fold verified model](fold_ready.png), [Flip conversion](flip_conversion.png), [Fold conversion](fold_conversion.png). Raw UI hierarchies, tests and build/install logs are alongside this report.

## APK and Quick Share

- APK: `GenUICraft/artifacts/GenUICraft-TestApp-E2B-Download-20260918.apk`
- Size: **213,067,308 bytes** (213.1 MB).
- SHA-256: `03768961d0de3a937bfbfa83bd7c8dbc21a3a548e5fbdc29949619fbb8928b68`.
- Samsung Quick Share delivery succeeded. The active link is retained in local delivery evidence and omitted from version control.
- Exact staged device copy: `/sdcard/Download/GenUICraft-TestApp-E2B-Download-20260918.apk` on the Flip. Its SHA-256 matches the host and both installed copies.
- Dry-run passed; Samsung displayed the generated HTTPS URL on its QR/link screen. That screen was left open. No expiry date was shown in the captured screen.
