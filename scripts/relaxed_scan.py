"""Backward-compatible standalone script. Delegates to the installed entry point."""
from multioptpy.Entrypoints import run_relaxedscan

if __name__ == "__main__":
    run_relaxedscan()
