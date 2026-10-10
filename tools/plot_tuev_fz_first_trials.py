#!/usr/bin/env python3
"""Compatibility entry point for shared channel feature comparison."""
from pathlib import Path
import runpy
import sys
for option,value in [('--channel', 'FZ'), ('--group-by', 'class')]:
    if not any(arg == option or arg.startswith(option+'=') for arg in sys.argv):sys.argv.extend([option,value])
runpy.run_path(str(Path(__file__).with_name('plot_channel_feature_comparison.py')),run_name='__main__')
