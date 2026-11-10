"""Repair UTF-8 reports that were double-encoded by a PowerShell round-trip.

Windows PowerShell 5.1 ``Get-Content -Raw`` reads a BOM-less UTF-8 file as
Windows-1252, and ``Set-Content -Encoding utf8`` then re-encodes those mis-read
characters as UTF-8.  A 3-byte character such as an em dash (E2 80 94) becomes
the six characters "\u00e2\u20ac\u201d".  This reverses that, run by run.

A whole-file reversal is wrong for a mixed file: text appended afterwards by a
UTF-8-aware editor is already correct, and would be corrupted by reversing it.
So the repair walks the text and only rewrites a run that starts on a character
which could be a UTF-8 lead byte (U+00C2..U+00F4) and whose following characters
form valid UTF-8 when mapped back through Windows-1252.  Anything else is copied
through untouched, which is why a legitimate "\u00b0", "\u00b1" or "\u2014" survives.

    python python\\repair_double_encoded.py reports\\STATUS.md reports\\model-v1.md

Originals are copied to ``<name>.bak-mojibake`` before writing.  Exit 0 when
every named file is clean or repaired.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# cp1252 code point -> raw byte.  Five bytes are undefined in cp1252; Python's
# codec refuses them, so build the table from the defined ones only.
_TO_BYTE: dict[str, int] = {}
for _b in range(0x80, 0x100):
    try:
        _TO_BYTE[bytes([_b]).decode("cp1252")] = _b
    except UnicodeDecodeError:
        pass

LEAD_LO, LEAD_HI = 0xC2, 0xF4
MARKERS = ("\u00c3", "\u00e2\u20ac", "\u00c2", "\u00c5")


def seq_len(lead: int) -> int | None:
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return None


def repair(text: str) -> tuple[str, int]:
    """Return (fixed_text, number_of_sequences_reversed)."""
    out: list[str] = []
    i, n, fixed = 0, len(text), 0
    while i < n:
        ch = text[i]
        lead = _TO_BYTE.get(ch)
        need = seq_len(lead) if lead is not None else None
        if need is not None and i + need <= n:
            chunk = text[i : i + need]
            raw = []
            for c in chunk:
                byte = _TO_BYTE.get(c)
                if byte is None:
                    raw = []
                    break
                raw.append(byte)
            if raw and raw[0] >= LEAD_LO:
                try:
                    out.append(bytes(raw).decode("utf-8"))
                    i += need
                    fixed += 1
                    continue
                except UnicodeDecodeError:
                    pass
        out.append(ch)
        i += 1
    return "".join(out), fixed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument(
        "--suffix",
        default=".bak-mojibake",
        help="backup suffix for the damaged file (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    failed = 0
    for path in args.paths:
        if not path.is_file():
            print(f"MISSING  {path}")
            failed += 1
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", "strict")
        if not any(m in text for m in MARKERS):
            print(f"clean    {path}")
            continue
        fixed, count = repair(text)
        if any(m in fixed for m in MARKERS):
            print(f"FAILED   {path}: markers remain after reversal; not written")
            failed += 1
            continue
        backup = path.with_suffix(path.suffix + args.suffix)
        shutil.copy2(path, backup)
        path.write_bytes(fixed.encode("utf-8"))
        print(
            f"repaired {path}  {count} sequences  {len(data)} -> "
            f"{len(fixed.encode('utf-8'))} bytes  backup {backup.name}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
