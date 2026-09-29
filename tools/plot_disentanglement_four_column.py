#!/usr/bin/env python3
"""Render the same four-column d_sub/d_task view for one or more input views.

Each row directory needs ``sample_alignment.csv``, ``d_sub_tsne.npy``, and
``d_task_tsne.npy``. Example::

    python tools/plot_disentanglement_four_column.py \
      --row '12 real -> 16 missing=outputs/run/disentanglement_best' \
      --output-dir outputs/run/disentanglement_best

Repeat --row to compare input views from the same checkpoint and samples.
The script only renders cached coordinates; it never fits t-SNE or runs a model.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


TASK_NAMES = {
    0: "ERN/Incorrect", 1: "ERN/Correct",
    2: "LRP/Contralateral", 3: "LRP/Ipsilateral",
    4: "MMN/Deviants", 5: "MMN/Standards",
    6: "N2pc/Contralateral", 7: "N2pc/Ipsilateral",
    8: "N400/Unrelated", 9: "N400/Related",
    10: "P3/Rare", 11: "P3/Frequent",
}


def load_row(label, directory):
    directory = Path(directory)
    with (directory / "sample_alignment.csv").open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    subjects = np.asarray([int(row["subject"]) for row in samples])
    tasks = np.asarray([int(row["task"]) for row in samples])
    coordinates = {
        branch: np.load(directory / f"d_{branch}_tsne.npy")
        for branch in ("sub", "task")
    }
    if any(value.shape != (len(samples), 2) for value in coordinates.values()):
        raise ValueError(f"Coordinate/sample count mismatch in {directory}")
    if not all(np.isfinite(value).all() for value in coordinates.values()):
        raise ValueError(f"Non-finite coordinates in {directory}")
    return {"label": label, "directory": directory, "subjects": subjects,
            "tasks": tasks, "coordinates": coordinates}


def render_rows(row_specs, output_dir, title=None, dpi=180):
    rows = [load_row(label, directory) for label, directory in row_specs]
    if not rows:
        raise ValueError("At least one row is required")
    subject_ids = sorted(set(int(v) for row in rows for v in row["subjects"]))
    task_ids = sorted(set(int(v) for row in rows for v in row["tasks"]))
    subject_colors = dict(zip(subject_ids, plt.get_cmap("nipy_spectral")(
        np.linspace(0, 1, len(subject_ids)))))
    task_colors = dict(zip(task_ids, plt.get_cmap("tab20")(
        np.linspace(0, 1, max(len(task_ids), 12)))))
    fig, axes = plt.subplots(len(rows), 4, figsize=(24, 4.5 * len(rows) + 1),
                             squeeze=False)
    columns = (("sub", "subject"), ("sub", "task"),
               ("task", "task"), ("task", "subject"))
    for row_index, row in enumerate(rows):
        for col_index, (branch, color_by) in enumerate(columns):
            axis = axes[row_index, col_index]
            labels = row["subjects"] if color_by == "subject" else row["tasks"]
            palette = subject_colors if color_by == "subject" else task_colors
            xy = row["coordinates"][branch]
            axis.scatter(xy[:, 0], xy[:, 1],
                         c=[palette[int(value)] for value in labels],
                         s=3 if len(xy) < 5000 else 1.5, alpha=0.7,
                         linewidths=0, rasterized=True)
            axis.set_title(f"d_{branch}, by {color_by}")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(row["label"])
    subject_handles = [Line2D([], [], marker="o", linestyle="", color=subject_colors[v],
                              label=f"Subject {v}") for v in subject_ids]
    task_handles = [Line2D([], [], marker="o", linestyle="", color=task_colors[v],
                           label=TASK_NAMES.get(v, f"Task {v}")) for v in task_ids]
    fig.legend(handles=subject_handles, title="Subject", loc="upper right",
               bbox_to_anchor=(0.998, 0.87), fontsize=8)
    fig.legend(handles=task_handles, title="ERP task", loc="lower right",
               bbox_to_anchor=(0.998, 0.06), fontsize=8)
    fig.suptitle(title or f"d_sub / d_task | Test n={len(rows[0]['subjects'])}")
    fig.subplots_adjust(left=0.05, right=0.84, bottom=0.07, top=0.88,
                        wspace=0.10, hspace=0.22)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"comparison_d_tsne.{suffix}", dpi=dpi)
    plt.close(fig)
    return output_dir / "comparison_d_tsne.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", action="append", required=True,
                        help="Row label and cache directory, as LABEL=DIR; repeat for more rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    row_specs = []
    for spec in args.row:
        if "=" not in spec:
            parser.error("--row must be LABEL=DIR")
        label, directory = spec.split("=", 1)
        row_specs.append((label, directory))
    print(render_rows(row_specs, args.output_dir, args.title, args.dpi))


if __name__ == "__main__":
    main()
