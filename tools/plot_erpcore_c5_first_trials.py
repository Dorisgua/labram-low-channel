#!/usr/bin/env python3
"""Compatibility entry point for shared channel feature comparison."""
from pathlib import Path
import runpy
import sys
for option,value in [('--channel', 'C5'), ('--group-by', 'subject'), ('--checkpoint', '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LaBraM-unified-AON-dynamic_general/outputs/erpcore/erp_core_D_stage1/checkpoint-best.pth')]:
    if not any(arg == option or arg.startswith(option+'=') for arg in sys.argv):sys.argv.extend([option,value])
runpy.run_path(str(Path(__file__).with_name('plot_channel_feature_comparison.py')),run_name='__main__')
