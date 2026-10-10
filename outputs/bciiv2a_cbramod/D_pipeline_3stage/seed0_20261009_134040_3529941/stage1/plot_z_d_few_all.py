#!/usr/bin/env python3
"""Compatibility entry point; shared implementation lives in tools/."""
from pathlib import Path
import runpy
import sys
HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / 'run_dynamic_stage1.py').is_file())
if '--checkpoint' not in sys.argv and not any(arg.startswith('--checkpoint=') for arg in sys.argv):
    sys.argv.extend(['--checkpoint', str(HERE / 'checkpoint-best.pth')])
runpy.run_path(str(ROOT / 'tools/plot_dynamic_z_d.py'), run_name='__main__')
