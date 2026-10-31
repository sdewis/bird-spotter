#!/usr/bin/env python3
"""CLI wrapper around halloween.servos.deployer to package or deploy the servo node to any Raspberry Pi."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halloween.servos.deployer import main, run_ssh_interactive as run_ssh_with_password  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
