"""Create a reviewable rater manifest without reading annotation contents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Detailed HumanQC folder")
    parser.add_argument("--output", default="config/human_qc_manifest.csv")
    parser.add_argument(
        "--rater-from-parent",
        action="store_true",
        help="Use each CSV's immediate parent folder as rater_id",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force after review")
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    rows = []
    for source in files:
        relative = source.relative_to(root).as_posix()
        rater_id = source.parent.name if args.rater_from_parent and source.parent != root else ""
        rows.append((relative, rater_id))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "rater_id"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    if any(not rater for _, rater in rows):
        print("Rater IDs are blank and must be completed before Goal 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
