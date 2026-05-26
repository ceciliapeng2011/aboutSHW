#!/usr/bin/env python3
"""Find files that share the same basename across folders.

Examples:
  # Common files between 2 folders
  python3 find_same_name_files.py \
    $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm \
    $HOME/OCL/aboutSHW/opencl/tests/x_attn

  # Common files across 3 folders (intersection of all)
  python3 find_same_name_files.py \
    $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm \
    $HOME/OCL/aboutSHW/opencl/tests/pageatten \
    $HOME/OCL/aboutSHW/opencl/tests/x_attn

  # Compare the first folder against each of the others separately
  python3 find_same_name_files.py --pairwise \
    $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm \
    $HOME/OCL/aboutSHW/opencl/tests/pageatten \
    $HOME/OCL/aboutSHW/opencl/tests/x_attn
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Set


DEFAULT_DIRS = [
    "$HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm",
    "$HOME/OCL/aboutSHW/opencl/tests/pageatten",
    "$HOME/OCL/aboutSHW/opencl/tests/x_attn",
]


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


KERNEL_DECL_RE = re.compile(r'(\bextern\s+"C"\s+_GENX_MAIN_\s+void\s+)([A-Za-z_]\w*|KERNEL_NAME)(\s*\()')


def build_root_labels(roots: List[Path]) -> Dict[Path, str]:
    """Create human-friendly and unique labels for roots."""
    def friendly_label(root: Path) -> str:
        text = str(root)
        if "/openvino/" in text or text.endswith("/openvino"):
            return "OpenVINO"
        if "/OCL/aboutSHW/" in text or text.endswith("/aboutSHW"):
            suffix = root.name
            parent = root.parent.name
            if suffix == "aboutSHW":
                return "aboutSHW"
            return f"aboutSHW/{suffix}"
        return root.name

    labels: Dict[Path, str] = {}
    used: Set[str] = set()
    for r in roots:
        label = friendly_label(r)
        if label in used:
            label = f"{label}/{r.name}"
        idx = 2
        base_label = label
        while label in used:
            label = f"{base_label}_{idx}"
            idx += 1
        labels[r] = label
        used.add(label)
    return labels


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def c(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{Color.RESET}"


def normalize_dir_input(path_str: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    return Path(expanded).resolve()


def _is_text_file(path: Path, sample_bytes: int = 8192) -> bool:
    data = path.read_bytes()[:sample_bytes]
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _normalize_kernel_decl_names(text: str) -> str:
    return KERNEL_DECL_RE.sub(r"\1__KERNEL_FN__\3", text)


def same_or_differ(a: Path, b: Path) -> bool:
    raw_a = a.read_bytes()
    raw_b = b.read_bytes()
    if raw_a == raw_b:
        return True

    if not (_is_text_file(a) and _is_text_file(b)):
        return False

    txt_a = raw_a.decode("utf-8", errors="replace")
    txt_b = raw_b.decode("utf-8", errors="replace")
    return _normalize_kernel_decl_names(txt_a) == _normalize_kernel_decl_names(txt_b)


def collect_file_map(root: Path) -> Dict[str, List[Path]]:
    """Map basename -> all matching file paths under root (recursive)."""
    file_map: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            file_map.setdefault(p.name, []).append(p)
    return file_map


def print_grouped(
    common_names: List[str],
    maps: List[Dict[str, List[Path]]],
    roots: List[Path],
    root_labels: Dict[Path, str],
    use_color: bool,
) -> None:
    if not common_names:
        print(c("No files with the same name were found.", Color.YELLOW, use_color))
        return

    same_count = 0
    differ_count = 0
    for name in common_names:
        pair_paths: List[Path] = []
        for fmap in maps:
            pair_paths.extend(fmap.get(name, []))
        if len(pair_paths) == 2 and same_or_differ(pair_paths[0], pair_paths[1]):
            same_count += 1
        elif len(pair_paths) == 2:
            differ_count += 1

    print(
        c(
            f"Found {len(common_names)} shared file name(s). "
            f"{same_count} same, {differ_count} differ.",
            Color.BOLD + Color.GREEN,
            use_color,
        )
    )
    for name in common_names:
        print()
        print(c(name, Color.CYAN + Color.BOLD, use_color))
        pair_paths: List[Path] = []
        for root, fmap in zip(roots, maps):
            for p in fmap.get(name, []):
                label = root_labels[root]
                rel = rel_to_root(p, root)
                print(f"  {c(label, Color.MAGENTA, use_color)} {c('→', Color.DIM, use_color)} {rel}")
                pair_paths.append(p)
        if len(pair_paths) == 2:
            status = "same" if same_or_differ(pair_paths[0], pair_paths[1]) else "differ"
            status_color = Color.GREEN if status == "same" else Color.YELLOW
            print(f"  {c('status:', Color.DIM, use_color)} {c(status, status_color, use_color)}")


def intersection_mode(roots: List[Path], use_color: bool) -> int:
    root_labels = build_root_labels(roots)
    maps = [collect_file_map(r) for r in roots]
    common: Set[str] = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())

    print_grouped(sorted(common), maps, roots, root_labels, use_color)
    return 0


def pairwise_mode(roots: List[Path], use_color: bool) -> int:
    root_labels = build_root_labels(roots)
    base = roots[0]
    base_map = collect_file_map(base)

    for other in roots[1:]:
        other_map = collect_file_map(other)
        common = sorted(set(base_map.keys()) & set(other_map.keys()))
        same_count = 0
        differ_count = 0
        for name in common:
            base_paths = base_map.get(name, [])
            other_paths = other_map.get(name, [])
            if len(base_paths) == 1 and len(other_paths) == 1:
                if same_or_differ(base_paths[0], other_paths[0]):
                    same_count += 1
                else:
                    differ_count += 1

        left = root_labels[base]
        right = root_labels[other]
        print()
        print(c(f"=== {left} <-> {right} ===", Color.BOLD + Color.GREEN, use_color))
        if not common:
            print(c("No shared file names.", Color.YELLOW, use_color))
            continue

        print(
            c(
                f"Found {len(common)} shared file name(s). "
                f"{same_count} same, {differ_count} differ.",
                Color.BOLD + Color.GREEN,
                use_color,
            )
        )
        for name in common:
            print()
            print(c(name, Color.CYAN + Color.BOLD, use_color))
            pair_paths: List[Path] = []
            for p in base_map.get(name, []):
                print(f"  {c(left, Color.MAGENTA, use_color)} {c('→', Color.DIM, use_color)} {rel_to_root(p, base)}")
                pair_paths.append(p)
            for p in other_map.get(name, []):
                print(f"  {c(right, Color.MAGENTA, use_color)} {c('→', Color.DIM, use_color)} {rel_to_root(p, other)}")
                pair_paths.append(p)
            if len(pair_paths) == 2:
                status = "same" if same_or_differ(pair_paths[0], pair_paths[1]) else "differ"
                status_color = Color.GREEN if status == "same" else Color.YELLOW
                print(f"  {c('status:', Color.DIM, use_color)} {c(status, status_color, use_color)}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and list files with the same basename across directories."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Two or more directories to scan recursively.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Compare first directory against each remaining directory separately.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )

    args = parser.parse_args()

    use_default_dirs = not args.dirs
    dirs = args.dirs if args.dirs else DEFAULT_DIRS

    if len(dirs) < 2:
        parser.error("Provide at least two directories.")

    roots = [normalize_dir_input(d) for d in dirs]
    for r in roots:
        if not r.exists() or not r.is_dir():
            parser.error(f"Invalid directory: {r}")

    use_color = not args.no_color

    if args.pairwise or (use_default_dirs and len(roots) > 2):
        return pairwise_mode(roots, use_color)
    return intersection_mode(roots, use_color)


if __name__ == "__main__":
    raise SystemExit(main())
