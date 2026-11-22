#!/usr/bin/env python3
"""
Generate confusion matrices from Furhat evaluation results.

Modes:
  (default)   12x12 per syllable, alphabetical order, separate per stage
  organize    12x12 per syllable, ordered by place of articulation + voiced/unvoiced
  group       6x6 phoneme pair groups (B/P, F/V, SS/Z, T/D, CH/J, K/G)

Flags:
  --agglomerate   Merge A and I stages into one matrix (instead of separate per stage)

Usage:
    python confusion_matrix.py results.csv
    python confusion_matrix.py results.csv --mode organize
    python confusion_matrix.py results.csv --mode group
    python confusion_matrix.py results.csv --mode organize --agglomerate
    python confusion_matrix.py results.csv --user 525ace60 --noise noise0
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Display name overrides (backend -> user-facing)
# ---------------------------------------------------------------------------
DISPLAY_LABELS = {
    "IGI": "IGUI",
}

def display_name(syl):
    return DISPLAY_LABELS.get(syl, syl)


# ---------------------------------------------------------------------------
# Organized order: place of articulation, consistent manner at each place
# (stops/affricates first, then fricatives), voiced then unvoiced within each
# ---------------------------------------------------------------------------
ORGANIZED_ORDER_A = [
    "ABA",  "APA",    # bilabial stops
    "AVA",  "AFA",    # labiodental fricatives
    "ADA",  "ATA",    # alveolar stops
    "AZA",  "ASSA",   # alveolar fricatives
    "AJA",  "ACHA",   # postalveolar affricates
    "AGA",  "AKA",    # velar stops
]
ORGANIZED_ORDER_I = [
    "IBI",  "IPI",    # bilabial stops
    "IVI",  "IFI",    # labiodental fricatives
    "IDI",  "ITI",    # alveolar stops
    "IZI",  "ISSI",   # alveolar fricatives
    "IJI",  "ICHI",   # postalveolar affricates
    "IGI",  "IKI",    # velar stops
]
# For agglomerate mode with organize: interleave A and I by position
ORGANIZED_ORDER_AGGLOM = []
for a, i in zip(ORGANIZED_ORDER_A, ORGANIZED_ORDER_I):
    ORGANIZED_ORDER_AGGLOM.extend([a, i])


# ---------------------------------------------------------------------------
# Phoneme group definitions (for --mode group)
# Order: consistent with consonant order (by place, stops before fricatives)
# ---------------------------------------------------------------------------
PHONEME_GROUPS = {}
_GROUP_DEFS = [
    ("B/P",  ["ABA", "APA", "IBI", "IPI"]),
    ("V/F",  ["AVA", "AFA", "IVI", "IFI"]),
    ("D/T",  ["ADA", "ATA", "IDI", "ITI"]),
    ("Z/SS", ["AZA", "ASSA", "IZI", "ISSI"]),
    ("J/CH", ["AJA", "ACHA", "IJI", "ICHI"]),
    ("G/K",  ["AGA", "AKA", "IGI", "IKI"]),
]
GROUP_ORDER = [g[0] for g in _GROUP_DEFS]
for _label, _syls in _GROUP_DEFS:
    for _syl in _syls:
        PHONEME_GROUPS[_syl] = _label


# ---------------------------------------------------------------------------
# Consonant mapping (for --mode consonant): syllable -> consonant label
# Order: by place, stops before fricatives, voiced before unvoiced
# ---------------------------------------------------------------------------
CONSONANT_MAP = {
    "ABA": "B", "IBI": "B",   "APA": "P", "IPI": "P",
    "AVA": "V", "IVI": "V",   "AFA": "F", "IFI": "F",
    "ADA": "D", "IDI": "D",   "ATA": "T", "ITI": "T",
    "AZA": "Z", "IZI": "Z",   "ASSA": "SS", "ISSI": "SS",
    "AJA": "J", "IJI": "J",   "ACHA": "CH", "ICHI": "CH",
    "AGA": "G", "IGI": "G",   "AKA": "K", "IKI": "K",
}
CONSONANT_ORDER = ["B", "P", "V", "F", "D", "T", "Z", "SS", "J", "CH", "G", "K"]


# ---------------------------------------------------------------------------
# Border group definitions for visual grouping on plots
# Each entry is the number of consecutive labels that form a group.
# ---------------------------------------------------------------------------
# For 12-label modes (consonant, organize per stage):
PAIR_BORDERS_12 = [2] * 6        # 6 voiced/unvoiced pairs of 2
#   place of articulation: bilabial(2), labiodental(2), alveolar stop(2),
#   alveolar fric(2), postalveolar(2), velar(2) — all 2x2
PLACE_BORDERS_12 = [2, 2, 2, 2, 2, 2]

# Block labels for per-block accuracy (12-label modes)
BLOCK_LABELS_12 = ["Bilabial", "Labiodental", "Alv. stop", "Alv. fric", "Postalv.", "Velar"]

# For 6-label mode (group): each group is 1 cell, all places are size 1
PLACE_BORDERS_6 = [1, 1, 1, 1, 1, 1]
BLOCK_LABELS_6 = ["Bilabial", "Labiodental", "Alv. stop", "Alv. fric", "Postalv.", "Velar"]

# For 24-label agglomerate mode:
PAIR_BORDERS_24 = [2] * 12
PLACE_BORDERS_24 = [4, 4, 4, 4, 4, 4]
BLOCK_LABELS_24 = ["Bilabial", "Labiodental", "Alv. stop", "Alv. fric", "Postalv.", "Velar"]


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def _draw_group_borders(ax, group_sizes, color, linewidth):
    """Draw thick borders around groups of cells."""
    pos = 0
    for size in group_sizes:
        rect = plt.Rectangle((pos, pos), size, size,
                              fill=False, edgecolor=color, linewidth=linewidth,
                              clip_on=False)
        ax.add_patch(rect)
        pos += size


def _compute_block_accuracies(cm, block_sizes):
    """Compute accuracy for each block.
    correct = diagonal entries within the block (true positives).
    total = sum of the full rows for that block (all trials where the true
            syllable belongs to this block, regardless of what the user responded).
    """
    results = []
    pos = 0
    for size in block_sizes:
        block_diag = cm[pos:pos+size, pos:pos+size]
        correct = np.trace(block_diag)
        total = cm[pos:pos+size, :].sum()  # all responses for these rows
        acc = correct / total * 100 if total > 0 else float('nan')
        results.append({"correct": int(correct), "total": int(total), "accuracy": acc})
        pos += size
    return results


def plot_cm(cm, labels, display_labels, title, filepath, figsize=(10, 8),
            pair_borders=None, place_borders=None, block_labels=None):
    """Generic confusion matrix plot with optional group borders, bold diagonal,
    and per-block accuracy annotations."""
    correct = np.trace(cm)
    total = cm.sum()
    accuracy = correct / total * 100 if total > 0 else 0
    n = len(labels)

    # Build annotation array: bold for diagonal
    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                annot[i, j] = ""
            elif i == j:
                annot[i, j] = f"$\\mathbf{{{val}}}$"
            else:
                annot[i, j] = str(val)

    # If we have block labels, widen the figure to make room for annotations
    
    # has_blocks = block_labels and place_borders
    has_blocks = False

    if has_blocks:
        fig_w, fig_h = figsize
        fig, axes = plt.subplots(1, 2, figsize=(fig_w + 2.5, fig_h),
                                 gridspec_kw={"width_ratios": [fig_w, 2.5]})
        ax = axes[0]
        ax_side = axes[1]
    else:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels,
        #cbar_kws={"label": "Count"},
        ax=ax, linewidths=0.5, linecolor="#ddd", annot_kws={"size": 16}
    )
    
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        # Tamanho dos NÚMEROS na barra:
        cbar.ax.tick_params(labelsize=16)
        
        # Tamanho e estilo do TEXTO/RÓTULO "Count" da barra (opcional):
        cbar.ax.yaxis.label.set_size(16)
        cbar.ax.yaxis.label.set_weight("bold")

    # Draw group borders
    if pair_borders:
        _draw_group_borders(ax, pair_borders, color="#444", linewidth=2.0)
    if place_borders:
        _draw_group_borders(ax, place_borders, color="#000", linewidth=3.0)

    ax.set_xlabel("Response", fontsize=18, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=18, fontweight="bold")
    # ax.set_title(
    #     f"{title}  (accuracy: {accuracy:.1f}%, n={total})",
    #     fontsize=13, fontweight="bold", pad=16,
    # )
    # tick_size = 10 if n <= 6 else 8 if n <= 12 else 7
    tick_size = 16
    rotation = 0 if n <= 6 else 45
    ha = "center" if n <= 6 else "right"
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha=ha, fontsize=tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_size)

    # Per-block accuracy on the right side
    if has_blocks:
        block_accs = _compute_block_accuracies(cm, place_borders)
        ax_side.set_xlim(0, 1)
        ax_side.set_ylim(0, 1)
        ax_side.axis("off")
        ax_side.set_title("Block accuracy", fontsize=11, fontweight="bold", pad=16)

        # Position each label at the vertical center of its block
        pos = 0
        for i, (bl, ba) in enumerate(zip(block_labels, block_accs)):
            size = place_borders[i]
            center_y = 1.0 - (pos + size / 2) / n
            if np.isnan(ba["accuracy"]):
                text = f"{bl}\n—"
            else:
                text = f"{bl}\n{ba['accuracy']:.0f}% ({ba['correct']}/{ba['total']})"
            ax_side.text(0.15, center_y, text, fontsize=9, va="center", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                                   edgecolor="#ccc", alpha=0.9))
            pos += size

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {filepath}  ({correct}/{total} correct, {accuracy:.1f}%)")


def build_cm(df, labels, col_true="syllable", col_pred="response"):
    """Build confusion matrix from dataframe given a label order."""
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    skipped = 0
    for _, row in df.iterrows():
        ti = label_to_idx.get(row[col_true])
        pi = label_to_idx.get(row[col_pred])
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
        else:
            skipped += 1
    if skipped > 0:
        print(f"    ({skipped} trials skipped — label not in order list)")
    return cm


# ---------------------------------------------------------------------------
# Mode: default (alphabetical) or organize (by place of articulation)
# ---------------------------------------------------------------------------
def run_syllable_mode(df, output_dir, mode, agglomerate):
    """Generate per-syllable confusion matrices (12x12 or 24x24 if agglomerated)."""
    saved = []
    # Borders and block labels only for organize mode
    if mode == "organize":
        pb_12, pl_12, bl_12 = PAIR_BORDERS_12, PLACE_BORDERS_12, BLOCK_LABELS_12
        pb_24, pl_24, bl_24 = PAIR_BORDERS_24, PLACE_BORDERS_24, BLOCK_LABELS_24
    else:
        pb_12 = pl_12 = bl_12 = pb_24 = pl_24 = bl_24 = None

    if agglomerate:
        # One matrix per noise level, merging A+I
        if mode == "organize":
            labels = ORGANIZED_ORDER_A + ORGANIZED_ORDER_I
        else:
            labels = sorted(set(df["syllable"].unique()) | set(df["response"].unique()))

        all_syllables_in_data = set(df["syllable"].unique())
        labels = [l for l in labels if l in all_syllables_in_data or l in set(df["response"].unique())]
        disp = [display_name(l) for l in labels]
        figsize = (14, 12) if len(labels) > 12 else (10, 8)

        for noise in sorted(df["noise"].unique()):
            ndf = df[df["noise"] == noise]
            cm = build_cm(ndf, labels)
            title = f"All stages — {noise}"
            fp = output_dir / f"confusion_all_{noise}.png"
            plot_cm(cm, labels, disp, title, fp, figsize,
                    pair_borders=pb_24, place_borders=pl_24, block_labels=bl_24)
            saved.append(fp)

        # Aggregate
        cm = build_cm(df, labels)
        fp = output_dir / f"confusion_all_all.png"
        plot_cm(cm, labels, disp, "All stages — all noise", fp, figsize,
                pair_borders=pb_24, place_borders=pl_24, block_labels=bl_24)
        saved.append(fp)

    else:
        # Separate per stage
        for stage in sorted(df["stage"].unique()):
            sdf = df[df["stage"] == stage]
            if mode == "organize":
                order = ORGANIZED_ORDER_A if stage == "A" else ORGANIZED_ORDER_I
                labels = [l for l in order if l in set(sdf["syllable"].unique()) | set(sdf["response"].unique())]
            else:
                labels = sorted(sdf["syllable"].unique())

            disp = [display_name(l) for l in labels]
            print(f"\nStage {stage} ({len(labels)} syllables):")

            for noise in sorted(sdf["noise"].unique()):
                ndf = sdf[sdf["noise"] == noise]
                cm = build_cm(ndf, labels)
                title = f"Stage {stage} — {noise}"
                fp = output_dir / f"confusion_{stage}_{noise}.png"
                plot_cm(cm, labels, disp, title, fp,
                        pair_borders=pb_12, place_borders=pl_12, block_labels=bl_12)
                saved.append(fp)

            # Aggregate per stage
            cm = build_cm(sdf, labels)
            fp = output_dir / f"confusion_{stage}_all.png"
            plot_cm(cm, labels, disp, f"Stage {stage} — all noise", fp,
                    pair_borders=pb_12, place_borders=pl_12, block_labels=bl_12)
            saved.append(fp)

    return saved


# ---------------------------------------------------------------------------
# Mode: group (6x6 phoneme pairs)
# ---------------------------------------------------------------------------
def run_group_mode(df, output_dir):
    """Generate 6x6 grouped confusion matrices (always merges A+I)."""
    saved = []
    labels = GROUP_ORDER

    # Map syllable and response to group
    gdf = df.copy()
    gdf["syllable_g"] = gdf["syllable"].map(PHONEME_GROUPS)
    gdf["response_g"] = gdf["response"].map(PHONEME_GROUPS)
    # Drop rows where mapping failed
    before = len(gdf)
    gdf = gdf.dropna(subset=["syllable_g", "response_g"])
    if len(gdf) < before:
        print(f"  ({before - len(gdf)} trials dropped — syllable not in any group)")

    for noise in sorted(gdf["noise"].unique()):
        ndf = gdf[gdf["noise"] == noise]
        cm = build_cm(ndf, labels, col_true="syllable_g", col_pred="response_g")
        title = f"Phoneme Groups — {noise}"
        fp = output_dir / f"confusion_grouped_{noise}.png"
        plot_cm(cm, labels, labels, title, fp, figsize=(8, 6.5),
                place_borders=PLACE_BORDERS_6, block_labels=BLOCK_LABELS_6)
        saved.append(fp)

    # Aggregate
    cm = build_cm(gdf, labels, col_true="syllable_g", col_pred="response_g")
    fp = output_dir / f"confusion_grouped_all.png"
    plot_cm(cm, labels, labels, "Phoneme Groups — all noise", fp, figsize=(8, 6.5),
            place_borders=PLACE_BORDERS_6, block_labels=BLOCK_LABELS_6)
    saved.append(fp)

    return saved


# ---------------------------------------------------------------------------
# Mode: consonant (12x12 by consonant identity, merging vowels)
# ---------------------------------------------------------------------------
def run_consonant_mode(df, output_dir):
    """Generate 12x12 confusion matrices by consonant (B, P, F, V, ...).
    Merges A and I stages. Ordered by place of articulation."""
    saved = []
    labels = CONSONANT_ORDER

    cdf = df.copy()
    cdf["syllable_c"] = cdf["syllable"].map(CONSONANT_MAP)
    cdf["response_c"] = cdf["response"].map(CONSONANT_MAP)
    before = len(cdf)
    cdf = cdf.dropna(subset=["syllable_c", "response_c"])
    if len(cdf) < before:
        print(f"  ({before - len(cdf)} trials dropped — syllable not in consonant map)")

    for noise in sorted(cdf["noise"].unique()):
        ndf = cdf[cdf["noise"] == noise]
        cm = build_cm(ndf, labels, col_true="syllable_c", col_pred="response_c")
        title = f"Consonants — {noise}"
        fp = output_dir / f"confusion_consonant_{noise}.png"
        plot_cm(cm, labels, labels, title, fp,
                pair_borders=PAIR_BORDERS_12, place_borders=PLACE_BORDERS_12,
                block_labels=BLOCK_LABELS_12)
        saved.append(fp)

    # Aggregate
    cm = build_cm(cdf, labels, col_true="syllable_c", col_pred="response_c")
    fp = output_dir / f"confusion_consonant_all.png"
    plot_cm(cm, labels, labels, "Consonants — all noise", fp,
            pair_borders=PAIR_BORDERS_12, place_borders=PLACE_BORDERS_12,
            block_labels=BLOCK_LABELS_12)
    saved.append(fp)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)    12x12 per syllable, alphabetical order
  organize     12x12 per syllable, ordered by place of articulation
  group        6x6 phoneme pair groups (B/P, F/V, SS/Z, T/D, CH/J, K/G)
  consonant    12x12 by consonant (B, P, F, V, ...), merging A+I vowels

Examples:
  python confusion_matrix.py results.csv
  python confusion_matrix.py results.csv --mode organize
  python confusion_matrix.py results.csv --mode group
  python confusion_matrix.py results.csv --mode consonant
  python confusion_matrix.py results.csv --mode organize --agglomerate
  python confusion_matrix.py results.csv --user 525ace60
        """,
    )
    parser.add_argument("csv_file", help="Path to results.csv")
    parser.add_argument("--mode", choices=["default", "organize", "group", "consonant"],
                        default="default",
                        help="Matrix mode (default: per-syllable alphabetical)")
    parser.add_argument("--agglomerate", action="store_true",
                        help="Merge A and I stages into one matrix (for default/organize modes)")
    parser.add_argument("--output", "-o", default="confusion_matrices_day_2/consonant_improv", help="Output directory")
    parser.add_argument("--stage", help="Filter by stage (A or I)")
    parser.add_argument("--noise", help="Filter by noise level (e.g. noise0)")
    parser.add_argument("--user", help="Filter by user_id")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} trials from {args.csv_file}")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Stages: {sorted(df['stage'].unique())}")
    print(f"  Noise levels: {sorted(df['noise'].unique())}")
    print(f"  Mode: {args.mode}" + (" + agglomerate" if args.agglomerate else ""))

    if args.user:
        df = df[df["user_id"] == args.user]
        print(f"  Filtered to user {args.user}: {len(df)} trials")
    if args.stage:
        df = df[df["stage"] == args.stage]
        print(f"  Filtered to stage {args.stage}: {len(df)} trials")
    if args.noise:
        df = df[df["noise"] == args.noise]
        print(f"  Filtered to noise {args.noise}: {len(df)} trials")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "group":
        saved = run_group_mode(df, output_dir)
    elif args.mode == "consonant":
        saved = run_consonant_mode(df, output_dir)
    else:
        saved = run_syllable_mode(df, output_dir, args.mode, args.agglomerate)

    print(f"\nDone. {len(saved)} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()