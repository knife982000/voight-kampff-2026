#!/usr/bin/env python
# coding: utf-8

import os
import re
import zipfile
from argparse import ArgumentParser

import pandas as pd

parser = ArgumentParser(description="Create a PAN submission zip from a generated CSV.")
parser.add_argument("--input", type=str, required=True, help="Path to the generated CSV (columns: id, text).")
parser.add_argument("--team", type=str, required=True, help="Team name (used as the directory name inside the zip).")
parser.add_argument("--output", type=str, default=None,
                    help="Output zip file path. Defaults to <team>.zip in the current directory.")
args = parser.parse_args()

output_zip = args.output or f"{args.team}.zip"

df = pd.read_csv(args.input)

if "id" not in df.columns or "text" not in df.columns:
    raise ValueError("CSV must have 'id' and 'text' columns.")


def trim_to_last_dot(text: str) -> str:
    text = text.strip()
    if text.endswith("."):
        return text
    pos = text.rfind(".")
    if pos == -1:
        return text  # no dot found — keep as-is
    return text[: pos + 1]


with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for _, row in df.iterrows():
        topic_id = str(int(row["id"])).zfill(3)
        text = trim_to_last_dot(str(row["text"]))
        arcname = f"{args.team}/{topic_id}.txt"
        zf.writestr(arcname, text)

print(f"Submission written to {output_zip} ({len(df)} files)")
print("Contents preview:")
with zipfile.ZipFile(output_zip) as zf:
    for name in sorted(zf.namelist())[:5]:
        print(f"  {name}")
    if len(df) > 5:
        print(f"  ... and {len(df) - 5} more")
