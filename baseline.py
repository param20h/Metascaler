"""Compatibility wrapper for the required inference.py entrypoint."""
from __future__ import annotations

from inference import run_inference


if __name__ == "__main__":
    run_inference()