#!/usr/bin/env python3
"""
Generate the n-shot library files and the held-out playback file from
swat_raw_labeled.csv.

Usage:
    python scripts/generate_labels.py

Output:
    - swat_normal.csv    (n-shot: 2000 rows from normal operation)
    - swat_attack.csv    (n-shot: 2000 rows from attack periods)
    - swat_playback.csv  (held-out stream the demo replays, with label column;
                          shares no row with the n-shot files)

Prerequisites:
    - Run convert_swat_data.py first
"""

import csv
import os
import random
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

N_SHOT_PER_CLASS = 2000
RANDOM_SEED = 42

# Held-out playback split (leakage-free). The single attack block is split into
# two disjoint contiguous ranges: the EARLY part — normal lead-in + the
# normal→attack transition + early attack — is reserved for the demo's PLAYBACK
# stream, while the KNN library's attack n-shot examples are drawn from a LATER
# slice of the same block, separated by a gap. The played transition is thus
# never embedded into the library. See swat_playback.csv + the leakage assert.
PLAYBACK_NORMAL_LEAD = 3000  # normal rows before the attack starts (sets up the transition)
PLAYBACK_ATTACK_ROWS = 30000  # attack rows shown during playback, after the transition
NSHOT_ATTACK_GAP = 2000  # buffer between the playback window and the attack n-shot slice

CLASS_NORMAL = "normal"
CLASS_ATTACK = "attack"


def main():
    print("=" * 70)
    print(" SWaT Dataset - Generate Labels & Splits")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)

    labeled_file = os.path.join(DATA_DIR, "swat_raw_labeled.csv")
    if not os.path.exists(labeled_file):
        print(f"  ERROR: {labeled_file} not found.")
        print("  Run 'python scripts/convert_swat_data.py' first.")
        sys.exit(1)

    # Load all rows
    print("[1/3] Loading labeled data...")
    all_rows = []
    header = None

    with open(labeled_file, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            all_rows.append(row)

    total = len(all_rows)
    normal_count = sum(1 for r in all_rows if r["label"] == CLASS_NORMAL)
    attack_count = sum(1 for r in all_rows if r["label"] == CLASS_ATTACK)

    print(f"  Total rows: {total:,}")
    print(f"  Normal:     {normal_count:,}")
    print(f"  Attack:     {attack_count:,}")
    print()

    # Sensor columns (everything except timestamp and label)
    sensor_cols = [c for c in header if c not in ("timestamp", "label")]
    inference_header = ["timestamp"] + sensor_cols

    # --- N-shot: normal (contiguous block from normal-only region) ---
    print("[2/3] Creating n-shot files...")

    # Find contiguous normal regions
    normal_nshot_rows = []
    # Look for a long run of normal rows and take a contiguous block
    run_start = None
    best_run_start = 0
    best_run_len = 0
    current_run_len = 0

    for i, row in enumerate(all_rows):
        if row["label"] == CLASS_NORMAL:
            if run_start is None:
                run_start = i
            current_run_len += 1
        else:
            if current_run_len > best_run_len:
                best_run_start = run_start
                best_run_len = current_run_len
            run_start = None
            current_run_len = 0

    if current_run_len > best_run_len:
        best_run_start = run_start
        best_run_len = current_run_len

    # Take n-shot normal from the middle of the longest normal run
    normal_start = best_run_start + (best_run_len // 2) - (N_SHOT_PER_CLASS // 2)
    normal_start = max(best_run_start, normal_start)
    normal_nshot_rows = all_rows[normal_start:normal_start + N_SHOT_PER_CLASS]
    print(f"  Normal n-shot: {len(normal_nshot_rows)} rows from index {normal_start}")

    # --- N-shot: attack (contiguous block from attack region) ---
    # Find contiguous attack regions
    attack_runs = []
    run_start = None
    current_run_len = 0

    for i, row in enumerate(all_rows):
        if row["label"] == CLASS_ATTACK:
            if run_start is None:
                run_start = i
            current_run_len += 1
        else:
            if run_start is not None and current_run_len > 0:
                attack_runs.append((run_start, current_run_len))
            run_start = None
            current_run_len = 0

    if run_start is not None and current_run_len > 0:
        attack_runs.append((run_start, current_run_len))

    # Take the longest attack run as the attack block, then carve a held-out
    # split out of it (see the constants above): the EARLY part is the playback
    # window, the attack n-shot examples come from a LATER, gapped slice.
    attack_runs.sort(key=lambda x: x[1], reverse=True)
    attack_start, attack_len = attack_runs[0]
    need = PLAYBACK_ATTACK_ROWS + NSHOT_ATTACK_GAP + N_SHOT_PER_CLASS
    if attack_len < need:
        raise SystemExit(
            f"Attack block too short ({attack_len} rows) for a held-out split (need {need})"
        )

    # Playback window: normal lead-in + the transition + early attack.
    playback_start = max(0, attack_start - PLAYBACK_NORMAL_LEAD)
    playback_end = attack_start + PLAYBACK_ATTACK_ROWS  # exclusive
    playback_rows = all_rows[playback_start:playback_end]

    # Attack n-shot: a later slice of the same attack block, after the playback
    # window plus a gap — disjoint from playback by construction.
    attack_nshot_start = playback_end + NSHOT_ATTACK_GAP
    attack_nshot_rows = all_rows[attack_nshot_start : attack_nshot_start + N_SHOT_PER_CLASS]

    print(f"  Attack block: index {attack_start}..{attack_start + attack_len} (len {attack_len})")
    print(
        f"  Attack n-shot: {len(attack_nshot_rows)} rows from index {attack_nshot_start} "
        f"(held out, after the playback window)"
    )

    # Collect n-shot timestamps for exclusion
    nshot_timestamps = set()
    for r in normal_nshot_rows + attack_nshot_rows:
        nshot_timestamps.add(int(r["timestamp"]))

    # Write n-shot files
    nshot_normal_path = os.path.join(DATA_DIR, "swat_normal.csv")
    nshot_attack_path = os.path.join(DATA_DIR, "swat_attack.csv")

    for path, rows in [
        (nshot_normal_path, normal_nshot_rows),
        (nshot_attack_path, attack_nshot_rows),
    ]:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(inference_header)
            for r in rows:
                writer.writerow([r["timestamp"]] + [r[c] for c in sensor_cols])
        print(f"  {os.path.basename(path)}: {len(rows)} rows")
    print()

    # --- Held-out playback file (contiguous, WITH label column) ---
    print("[3/3] Writing held-out playback file...")
    playback_path = os.path.join(DATA_DIR, "swat_playback.csv")
    playback_header = ["timestamp"] + sensor_cols + ["label"]
    with open(playback_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(playback_header)
        for r in playback_rows:
            writer.writerow([r["timestamp"]] + [r[c] for c in sensor_cols] + [r["label"]])
    n_pb_normal = sum(1 for r in playback_rows if r["label"] == CLASS_NORMAL)
    n_pb_attack = sum(1 for r in playback_rows if r["label"] == CLASS_ATTACK)
    print(
        f"  swat_playback.csv: {len(playback_rows)} rows "
        f"({n_pb_normal} normal + {n_pb_attack} attack)"
    )

    # Leakage check: the played rows must share no timestamp with the n-shot library.
    playback_ts = {int(r["timestamp"]) for r in playback_rows}
    leak = playback_ts & nshot_timestamps
    if leak:
        raise SystemExit(f"LEAKAGE: {len(leak)} playback rows are also in the n-shot library")
    print("  ✓ leakage check passed: playback ∩ n-shot = 0 rows")
    print()

    # Summary
    print("=" * 70)
    print(" Output Files")
    print("=" * 70)
    print()
    print(f"  {'File':<35s} {'Rows':>10} {'Description'}")
    print(f"  {'-'*70}")
    print(f"  {'swat_normal.csv':<35s} {len(normal_nshot_rows):>10,} N-shot: normal operation")
    print(f"  {'swat_attack.csv':<35s} {len(attack_nshot_rows):>10,} N-shot: attack periods")
    print(f"  {'swat_playback.csv':<35s} {len(playback_rows):>10,} Held-out playback (with label)")
    print(f"  {'swat_raw_labeled.csv':<35s} {total:>10,} Full labeled dataset")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
