#!/usr/bin/env python3
"""AFS stub — delegates to canonical EOS version. Copy to AFS once; never edit here."""
import runpy
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root, for siteconf
import siteconf

runpy.run_path(os.path.join(siteconf.PROJECT_DIR, "sweep/generate_sweep.py"), run_name="__main__")
