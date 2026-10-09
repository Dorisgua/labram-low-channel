#!/usr/bin/env python3
"""Require completed epoch logs, a training-end record, and readable best weights."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("epochs", type=int)
args = parser.parse_args()
try:
    rows = [json.loads(line) for line in (args.output / "log.txt").read_text().splitlines() if line.strip()]
    if not rows or int(rows[-1]["epoch"]) != args.epochs - 1:
        raise ValueError("final epoch has not completed")
    logs = list((args.output / "run_logs").glob("*.log"))
    if not logs:
        raise ValueError("no terminal log")
    latest = max(logs, key=lambda path: path.stat().st_mtime_ns)
    if not any(line.startswith("Training time ") for line in latest.read_text(errors="replace").splitlines()):
        raise ValueError("latest training process has no completion record")
    checkpoint = torch.load(args.output / "checkpoint-best.pth", map_location="cpu")
    if not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        raise ValueError("best checkpoint has no model weights")
    if int(checkpoint["args"].epochs) != args.epochs:
        raise ValueError("checkpoint epoch configuration differs")
except Exception as error:
    print(f"Stage incomplete: {error}")
    raise SystemExit(1)
print(f"Stage verified complete: {args.output}")
