#!/usr/bin/env python3
from collections import defaultdict
import os
from pathlib import Path
import re
from typing import Dict
from typing import List
import warnings

import click
from tqdm import tqdm


# Regex to capture the `class_id` integer for a line as well as the `remainder`
# for remapping.
RE_TRUTH_LINE = re.compile(r"^(?P<class_id>\d+)\s+(?P<remainder>.*)$")


@click.command()
@click.help_option("-h", "--help")
@click.argument(
    "ROOT_DIR", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-r",
    "--remap",
    "remap_tuple",
    nargs=2,
    multiple=True,
    type=int,
    help="Remap class IDs in discovered txt files.",
)
def main(root_dir, remap_tuple):
    """
    Remap class IDs in YOLO truth text files from the given values to some new
    values.

    **All** class IDs should have a from and to value, even if it is the same
    value.
    If we encounter any truth files with a class ID that is **not** declared in
    an input mapping, we will error.

    \b
    Positional Arguments:
        ROOT_PATH:
            Directory under which we will look for .txt files to modify.
    """
    remap = {p[0]: p[1] for p in remap_tuple}

    # scan for all files to modify
    truth_path_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1] == ".txt":
                truth_path_list.append(Path(dirpath) / fname)

    new_truth_lines: Dict[Path, List[str]] = defaultdict(list)

    # Stage modified content for each file picked up.
    # We don't want to write, or overwrite any files until we know we can
    # transform everything in a valid way.
    for truth_path in tqdm(truth_path_list, desc="Parsing Truth Files", unit="files"):
        with open(truth_path) as f:
            for line in f:
                m = RE_TRUTH_LINE.match(line)
                if m is None:
                    raise RuntimeError(
                        f"Encountered issue parsing line in truth file "
                        f"{truth_path}: encountered line: {line}"
                    )
                md = m.groupdict()
                new_class_id = remap[int(md["class_id"])]
                new_truth_lines[truth_path].append(f"{new_class_id} {md['remainder']}")
            if truth_path not in new_truth_lines:
                warnings.warn(f"No lines in truth file: {truth_path}")

    # We are here if all files have successfully mapped content. Write out\
    # content to the original file locations.
    for fpath, new_lines in tqdm(
        new_truth_lines.items(), desc="Writing files back out", unit="files"
    ):
        with open(fpath, "w") as f:
            f.writelines(l + "\n" for l in new_lines)


if __name__ == "__main__":
    main()
