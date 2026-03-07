"""Backward-compatible standalone script. Delegates to the installed entry point."""
from multioptpy.entrypoints import run_optmain

if __name__ == "__main__":
    run_optmain()
