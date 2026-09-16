"""One-shot renumbering of the diagnosed-failure channels (2026-09-11):

    old  ->  new
    R2   ->  R1   existence
    R4   ->  R2   interpenetration
    R5   ->  R3   kinematic
    R3   ->  R4   long-horizon (cumulative consistency)
    R1   ->  R5   frame invariance (P1-lite; pixel-scoped, deprioritized)

Requested directly ("the Rs are not in order, they should be"): the old
numbering was an accident of history (R3 was repurposed from "identity"
to long-horizon; R1 was built then deprioritized as output-modality-
scoped), so the four FUNDAMENTAL state-space channels read 2/4/5/3. The
new numbering puts them in 1..4 in the order they are checked, and moves
the pixel-scoped one to R5, outside the fundamental sequence.

What this script rewrites, as a SIMULTANEOUS mapping (never chained, so
R3->R4 cannot then be re-mapped R4->R2):
  * text files (py/md/yaml/toml under vitals/, scripts/, tests/, configs/,
    remote/, AGENT.md, README.md): every standalone token `R<n>` (also
    inside identifiers such as `sigma_R3`, `theta_R3`) and every lowercase
    identifier fragment `_r<n>` (`pop_r3`, `e_r3`, ...). Ranges written
    `R1-R5` / `R1..R5` / `R1–R5` mean "all five" and are preserved as-is.
    `R^2` (regression fit quality) is untouched -- the caret is not a
    word boundary the pattern accepts.
  * results JSON (results/**/*.json): every dict key exactly `R<n>` and
    every string value exactly `R<n>` -- the only places the names occur
    (surveyed before writing this: thresholds_median / termination_
    profile keys; events[].risk, gate_1b[].risk_*, l0_risk/l1_risk values).
    Originals are copied to results/archive/pre_rename_2026-09-11/ first.
  * Refuses to run a second time (marker results/.risk_names_v2).

    python3 scripts/migrate_risk_names_2026_09.py --dry-run
    python3 scripts/migrate_risk_names_2026_09.py
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKER = ROOT / "results" / ".risk_names_v2"
MAP = {"2": "1", "4": "2", "5": "3", "3": "4", "1": "5"}
TEXT_ROOTS = ["vitals", "scripts", "tests", "configs", "remote"]
TEXT_FILES = ["AGENT.md", "README.md", "pyproject.toml"]
TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".toml"}
SELF = pathlib.Path(__file__).resolve()

RANGE_RE = re.compile(r"R1(-|\.\.|–)R5")
UPPER_RE = re.compile(r"(?<![A-Za-z0-9])R([1-5])(?![0-9A-Za-z])")
LOWER_RE = re.compile(r"(?<=_)r([1-5])(?![0-9a-z])")


def rewrite_text(s):
    holes = []
    def hold(m):
        holes.append(m.group(0))
        return f"\x00{len(holes) - 1}\x00"
    s = RANGE_RE.sub(hold, s)
    s = UPPER_RE.sub(lambda m: "R" + MAP[m.group(1)], s)
    s = LOWER_RE.sub(lambda m: "r" + MAP[m.group(1)], s)
    return re.sub(r"\x00(\d+)\x00", lambda m: holes[int(m.group(1))], s)


def rewrite_json(o):
    if isinstance(o, dict):
        return {("R" + MAP[k[1]] if re.fullmatch(r"R[1-5]", k) else k): rewrite_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rewrite_json(v) for v in o]
    if isinstance(o, str) and re.fullmatch(r"R[1-5]", o):
        return "R" + MAP[o[1]]
    return o


def text_targets():
    for r in TEXT_ROOTS:
        for p in (ROOT / r).rglob("*"):
            if p.is_file() and p.suffix in TEXT_EXT and p.resolve() != SELF and "__pycache__" not in p.parts:
                yield p
    for f in TEXT_FILES:
        if (ROOT / f).exists():
            yield ROOT / f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if MARKER.exists():
        sys.exit(f"refusing to run twice: {MARKER} exists (the mapping is not idempotent)")

    n_text = n_tok = 0
    for p in text_targets():
        old = p.read_text()
        new = rewrite_text(old)
        if new != old:
            n_text += 1
            n_tok += sum(1 for _ in UPPER_RE.finditer(old)) + sum(1 for _ in LOWER_RE.finditer(old))
            if not args.dry_run:
                p.write_text(new)
            else:
                print(f"  text  {p.relative_to(ROOT)}")

    json_files = sorted((ROOT / "results").rglob("*.json"))
    backup = ROOT / "results" / "archive" / "pre_rename_2026-09-11"
    n_json = 0
    for p in json_files:
        if backup in p.parents:
            continue
        old = json.loads(p.read_text())
        new = rewrite_json(old)
        if new != old:
            n_json += 1
            if not args.dry_run:
                dst = backup / p.relative_to(ROOT / "results")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                p.write_text(json.dumps(new, indent=2))
            else:
                print(f"  json  {p.relative_to(ROOT)}")

    print(f"{'DRY RUN: ' if args.dry_run else ''}{n_text} text files ({n_tok} tokens), {n_json} json files")
    if not args.dry_run:
        MARKER.write_text("risk channels renumbered 2026-09-11: R2->R1 R4->R2 R5->R3 R3->R4 R1->R5\n")
        print(f"originals of rewritten json in {backup.relative_to(ROOT)}; marker {MARKER.relative_to(ROOT)} written")


if __name__ == "__main__":
    main()
