#!/usr/bin/env python3
"""Merge files from a source folder into a destination folder.

Features:
- Works only on file pairs that exist on both sides by basename
    (same behavior as find_same_name_files.py intersection).
- Only considers CM kernel files: .cm and .hpp.
- Ignores source-only or destination-only files.
- Skips ambiguous basenames (multiple matches on either side).
- If a destination file differs, shows conflict and asks how to resolve.
- Optional brutal-force overwrite mode.

Examples:
  # Interactive merge (default)
  python3 merge_folders_interactive.py \
    $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm \
    $HOME/OCL/aboutSHW/opencl/tests/x_attn

  # Reverse direction
  python3 merge_folders_interactive.py \
    $HOME/OCL/aboutSHW/opencl/tests/x_attn \
    $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm

  # Brutal-force overwrite destination
  python3 merge_folders_interactive.py SRC DST --force

  # Preview only
  python3 merge_folders_interactive.py SRC DST --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_SOURCE = "$HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm"
DEFAULT_DESTINATION = "$HOME/OCL/aboutSHW/opencl/tests/x_attn"
KERNEL_SUFFIXES = {".cm", ".hpp"}
KERNEL_DECL_RE = re.compile(r'(\bextern\s+"C"\s+_GENX_MAIN_\s+void\s+)([A-Za-z_]\w*|KERNEL_NAME)(\s*\()')


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def c(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{Color.RESET}"


def normalize_dir_input(path_str: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    return Path(expanded).resolve()


def build_root_labels(roots: List[Path]) -> Dict[Path, str]:
    """Create human-friendly and unique labels for roots."""
    def friendly_label(root: Path) -> str:
        text = str(root)
        if "/openvino/" in text or text.endswith("/openvino"):
            return "OpenVINO"
        if "/OCL/aboutSHW/" in text or text.endswith("/aboutSHW"):
            suffix = root.name
            if suffix == "aboutSHW":
                return "aboutSHW"
            return f"aboutSHW/{suffix}"
        return root.name

    labels: Dict[Path, str] = {}
    used: set[str] = set()
    for root in roots:
        label = friendly_label(root)
        if label in used:
            label = f"{label}/{root.name}"
        base_label = label
        idx = 2
        while label in used:
            label = f"{base_label}_{idx}"
            idx += 1
        labels[root] = label
        used.add(label)
    return labels


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass
class Stats:
    scanned: int = 0
    updated: int = 0
    skipped_same: int = 0
    kept_destination: int = 0
    with_conflict_markers: int = 0
    ambiguous_skipped: int = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def files_equal(a: Path, b: Path) -> bool:
    raw_a = a.read_bytes()
    raw_b = b.read_bytes()
    if raw_a == raw_b:
        return True

    if not (is_probably_text(a) and is_probably_text(b)):
        return False

    txt_a = raw_a.decode("utf-8", errors="replace")
    txt_b = raw_b.decode("utf-8", errors="replace")
    norm_a = KERNEL_DECL_RE.sub(r"\1__KERNEL_FN__\3", txt_a)
    norm_b = KERNEL_DECL_RE.sub(r"\1__KERNEL_FN__\3", txt_b)
    return norm_a == norm_b


def is_probably_text(path: Path, sample_bytes: int = 8192) -> bool:
    data = path.read_bytes()[:sample_bytes]
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def print_diff(src: Path, dst: Path, max_lines: int = 120) -> None:
    if not is_probably_text(src) or not is_probably_text(dst):
        print("  Binary or non-UTF8 file; textual diff is not shown.")
        print(f"  Source size: {src.stat().st_size} bytes")
        print(f"  Dest   size: {dst.stat().st_size} bytes")
        return

    src_lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    dst_lines = dst.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            dst_lines,
            src_lines,
            fromfile=str(dst),
            tofile=str(src),
            n=3,
        )
    )
    if not diff:
        print("  Files differ by hash but no textual diff was produced.")
        return

    print("  --- Diff preview (destination -> source) ---")
    for line in diff[:max_lines]:
        print(line.rstrip("\n"))
    if len(diff) > max_lines:
        print(f"  ... ({len(diff) - max_lines} more diff lines)")


def write_conflict_markers(src: Path, dst: Path, out: Path) -> None:
    src_text = src.read_text(encoding="utf-8", errors="replace")
    dst_text = dst.read_text(encoding="utf-8", errors="replace")
    merged = (
        f"<<<<<<< SOURCE: {src}\n"
        f"{src_text}"
        f"=======\n"
        f"{dst_text}"
        f">>>>>>> DESTINATION: {dst}\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(merged, encoding="utf-8")


def build_name_index(root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            idx.setdefault(p.name, []).append(p)
    return idx


def is_cm_kernel_file(path: Path) -> bool:
    return path.suffix.lower() in KERNEL_SUFFIXES


def find_common_pairs(src_root: Path, dst_root: Path) -> tuple[List[tuple[Path, Path]], int]:
    """Return one-to-one file pairs by shared basename only.

    This mirrors find_same_name_files.py behavior: only files that appear on both
    sides are considered. If a basename has multiple matches on either side,
    it is considered ambiguous and skipped.
    """
    src_idx = build_name_index(src_root)
    dst_idx = build_name_index(dst_root)

    pairs: List[tuple[Path, Path]] = []
    ambiguous = 0
    for name in sorted(set(src_idx.keys()) & set(dst_idx.keys())):
        if Path(name).suffix.lower() not in KERNEL_SUFFIXES:
            continue
        src_list = src_idx[name]
        dst_list = dst_idx[name]
        src_list = [p for p in src_list if is_cm_kernel_file(p)]
        dst_list = [p for p in dst_list if is_cm_kernel_file(p)]
        if not src_list or not dst_list:
            continue
        if len(src_list) == 1 and len(dst_list) == 1:
            pairs.append((src_list[0], dst_list[0]))
        else:
            ambiguous += 1
    return pairs, ambiguous


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and is_probably_text(src) and is_probably_text(dst):
        src_text = src.read_text(encoding="utf-8", errors="replace")
        dst_text = dst.read_text(encoding="utf-8", errors="replace")

        dst_names = [m.group(2) for m in KERNEL_DECL_RE.finditer(dst_text)]
        if dst_names:
            idx = 0

            def _replace(match: re.Match[str]) -> str:
                nonlocal idx
                replacement_name = dst_names[idx] if idx < len(dst_names) else match.group(2)
                idx += 1
                return f"{match.group(1)}{replacement_name}{match.group(3)}"

            src_text = KERNEL_DECL_RE.sub(_replace, src_text)
            dst.write_text(src_text, encoding="utf-8")
            shutil.copystat(src, dst, follow_symlinks=True)
            return

    shutil.copy2(src, dst)


def ask_user(src: Path, dst: Path) -> str:
    use_color = not os.environ.get("NO_COLOR")
    print(c("  Conflict resolution options:", Color.BOLD + Color.GREEN, use_color))
    print(c("    [o] overwrite destination with source", Color.MAGENTA, use_color))
    print(c("    [k] keep destination (skip)", Color.DIM, use_color))
    print(c("    [d] show diff", Color.CYAN, use_color))
    print(c("    [m] write conflict markers into destination", Color.MAGENTA, use_color))
    print(c("    [q] quit", Color.YELLOW, use_color))
    while True:
        choice = input("  Choose [o/k/d/m/q]: ").strip().lower()
        if choice in {"o", "k", "d", "m", "q"}:
            return choice
        print(c("  Invalid choice.", Color.YELLOW, use_color))


def merge(src_root: Path, dst_root: Path, force: bool, dry_run: bool) -> int:
    stats = Stats()
    pairs, pre_ambiguous = find_common_pairs(src_root, dst_root)
    stats.ambiguous_skipped += pre_ambiguous
    use_color = not os.environ.get("NO_COLOR")
    labels = build_root_labels([src_root, dst_root])
    src_label = labels[src_root]
    dst_label = labels[dst_root]
    same_count = 0
    differ_count = 0
    for src_file, target in pairs:
        if files_equal(src_file, target):
            same_count += 1
        else:
            differ_count += 1
    total = len(pairs)
    print(
        c(
            f"Found {total} common file pair(s) by basename. "
            f"{same_count} same, {differ_count} differ.",
            Color.BOLD + Color.GREEN,
            use_color,
        )
    )
    if pre_ambiguous:
        print(c(f"Skipped {pre_ambiguous} ambiguous basename(s) with multiple matches.", Color.YELLOW, use_color))

    for i, (src_file, target) in enumerate(pairs, start=1):
        stats.scanned += 1
        if files_equal(src_file, target):
            print(c(f"[{i}/{total}] {src_label} → {dst_label}: {rel_to_root(src_file, src_root)} -> {rel_to_root(target, dst_root)}", Color.CYAN + Color.BOLD, use_color))
            print(c("  status: same", Color.GREEN, use_color))
            stats.skipped_same += 1
            continue

        print()
        print(c(f"[{i}/{total}] {src_label} → {dst_label}: {rel_to_root(src_file, src_root)} -> {rel_to_root(target, dst_root)}", Color.CYAN + Color.BOLD, use_color))

        print(c("  conflict: destination differs from source", Color.YELLOW, use_color))

        if force:
            print(c("  action: FORCE overwrite destination", Color.MAGENTA, use_color))
            copy_file(src_file, target, dry_run)
            stats.updated += 1
            continue

        if dry_run:
            print(c("  action: DRY-RUN keep destination (use interactive mode without --dry-run to choose)", Color.DIM, use_color))
            stats.kept_destination += 1
            continue

        while True:
            choice = ask_user(src_file, target)
            if choice == "d":
                print_diff(src_file, target)
                continue
            if choice == "o":
                print(c("  action: overwrite destination", Color.MAGENTA, use_color))
                copy_file(src_file, target, dry_run)
                stats.updated += 1
                break
            if choice == "k":
                print(c("  action: keep destination", Color.DIM, use_color))
                stats.kept_destination += 1
                break
            if choice == "m":
                print(c("  action: write conflict markers into destination", Color.MAGENTA, use_color))
                if dry_run:
                    pass
                else:
                    if is_probably_text(src_file) and is_probably_text(target):
                        write_conflict_markers(src_file, target, target)
                    else:
                        print(c("  binary/non-UTF8 file; cannot write text conflict markers. keeping destination.", Color.YELLOW, use_color))
                        stats.kept_destination += 1
                        break
                stats.with_conflict_markers += 1
                break
            if choice == "q":
                print(c("Stopped by user.", Color.YELLOW, use_color))
                print_summary(stats, dry_run)
                return 1

    print_summary(stats, dry_run)
    return 0


def print_summary(stats: Stats, dry_run: bool) -> None:
    use_color = not os.environ.get("NO_COLOR")
    print(c("\n===== Summary =====", Color.BOLD + Color.GREEN, use_color))
    print(f"Scanned common pairs:       {stats.scanned}")
    print(f"Updated destination:        {stats.updated}")
    print(f"Skipped same:               {stats.skipped_same}")
    print(f"Kept destination:           {stats.kept_destination}")
    print(f"Wrote conflict markers:     {stats.with_conflict_markers}")
    print(f"Ambiguous skipped:          {stats.ambiguous_skipped}")
    if dry_run:
        print(c("(dry-run: no files were modified)", Color.DIM, use_color))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge source folder into destination folder with interactive conflict handling."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help=f"Source folder (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default=DEFAULT_DESTINATION,
        help=f"Destination folder (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Brutal-force overwrite destination whenever content differs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without changing any file.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )

    args = parser.parse_args()

    src_root = normalize_dir_input(args.source)
    dst_root = normalize_dir_input(args.destination)

    if not src_root.exists() or not src_root.is_dir():
        parser.error(f"Invalid source folder: {src_root}")
    if not dst_root.exists() or not dst_root.is_dir():
        parser.error(f"Invalid destination folder: {dst_root}")

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    return merge(src_root, dst_root, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
