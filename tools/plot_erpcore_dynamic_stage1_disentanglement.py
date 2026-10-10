#!/usr/bin/env python3
"""ERP-Core compatibility entry point for the shared z/d plotter."""
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
def option_value(option,default):
    for i,arg in enumerate(sys.argv[1:],1):
        if arg==option:return sys.argv[i+1]
        if arg.startswith(option+'='):return arg.split('=',1)[1]
    sys.argv.extend([option,str(default)])
    return str(default)
checkpoint=option_value('--checkpoint',ROOT/'outputs/erpcore/erp_core_D_stage1/checkpoint-best.pth')
option_value('--output-dir',Path(checkpoint).parent/'disentanglement_best')
option_value('--views','few')
option_value('--sample-mode','random')
option_value('--max-samples','14485')
runpy.run_path(str(Path(__file__).with_name('plot_dynamic_z_d.py')),run_name='__main__')
