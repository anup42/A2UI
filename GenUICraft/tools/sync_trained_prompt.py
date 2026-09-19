"""Generate or verify the SDK's frozen E2B v10 training prompt snapshot.

Run from any directory. --check is read-only; changing this snapshot requires a
checkpoint trained with the same contract, not prompt tuning on Bixby50.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
from ir_training.data.shared_prompt import create_shared_prompt_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "GenUICraft/genuicraft/src/main/assets/genuicraft/prompts/e2b_v10_shared_prompt.json"
    expected = create_shared_prompt_contract(ordering="root-first")
    if args.check:
        actual = json.loads(destination.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("SDK snapshot differs from current training contract; verify checkpoint provenance before updating.")
    else:
        destination.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Prompt parity OK: {expected['contract_sha256']}")


if __name__ == "__main__":
    main()
