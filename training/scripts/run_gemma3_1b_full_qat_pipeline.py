#!/usr/bin/env python3
"""Gemma 3 1B full-QAT -> Golden/Bixby -> selected full checkpoint -> INT8 LiteRT-LM.

Thin entrypoint over the shared Golden deployment workflow. Defaults to plan
only; --execute must be passed on the CUDA training/export/runtime host.
"""
from __future__ import annotations

from run_golden_deployment import main

if __name__ == "__main__":
    raise SystemExit(main(required_profile="1b"))
