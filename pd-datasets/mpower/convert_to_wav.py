#!/usr/bin/env python3
"""
Convert the downloaded mPower audio files to WAV.

Synapse's bulk downloader stores each audio file in its cache under an opaque
name like

    audio_countdown.m4a-<uuid><digits>.tmp

The *content* is Apple Lossless (ALAC) inside an m4a container, 44.1 kHz mono.
This script decodes each file with ffmpeg and writes a clean, analysis-ready
WAV named after the record it belongs to, e.g.

    wav/<recordId>__audio.wav
    wav/<recordId>__countdown.wav

It reads the manifest produced by download_mpower.py (audio_file_manifest.csv)
so it knows the recordId / column for every downloaded file, and rewrites that
manifest with an extra `wavPath` column.

Design notes:
  * The original Synapse cache is left untouched, so re-downloading stays
    resumable. WAVs are additive, in a separate wav/ directory.
  * Conversion is lossless-preserving: native sample rate + channel count are
    kept; only the sample *format* changes to PCM. Default is 16-bit
    (pcm_s16le) -- the de-facto standard for speech/voice analysis (Praat,
    parselmouth, librosa, openSMILE). Override with WAV_BITS=24 or WAV_BITS=32
    if you want to retain ALAC's full declared resolution.
  * Resumable: an existing, non-empty WAV is skipped. Delete wav/ to redo.
  * Parallel across CPU cores (override with WORKERS=N).

Usage:
    python3 convert_to_wav.py                 # convert everything in the manifest
    WAV_BITS=24 python3 convert_to_wav.py     # 24-bit PCM
    WORKERS=4 python3 convert_to_wav.py       # limit parallelism
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_CSV = os.path.join(HERE, "audio_file_manifest.csv")
WAV_DIR = os.path.join(HERE, "wav")
SKIPPED_CSV = os.path.join(HERE, "audio_no_usable_signal.csv")

# Map WAV_BITS -> ffmpeg PCM codec.
_PCM_CODEC = {"16": "pcm_s16le", "24": "pcm_s24le", "32": "pcm_s32le"}


def _pcm_codec() -> str:
    bits = os.environ.get("WAV_BITS", "16")
    if bits not in _PCM_CODEC:
        sys.exit(f"ERROR: WAV_BITS must be one of {sorted(_PCM_CODEC)} (got {bits!r})")
    return _PCM_CODEC[bits]


def _stem_for_column(column: str) -> str:
    """audio_countdown.m4a -> countdown ; audio_audio.m4a -> audio."""
    stem = column.rsplit(".", 1)[0]          # drop extension
    if stem.startswith("audio_"):
        stem = stem[len("audio_"):]          # drop redundant prefix
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "audio"
    return stem


def _safe(name: str) -> str:
    """Make a manifest value safe to use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def _out_path(record_id: str, column: str, index: int) -> str:
    rid = _safe(record_id) if record_id else f"row{index}"
    return os.path.join(WAV_DIR, f"{rid}__{_stem_for_column(column)}.wav")


# A decoded WAV smaller than this is just a header with no real audio.
_MIN_WAV_BYTES = 256


def _looks_empty(src: str) -> bool:
    """True if the source has essentially no decodable audio.

    Some mPower captures were aborted after a fraction of a second and were
    written with a malformed ALAC header (ffmpeg: 'invalid samples per
    frame: 0'). These carry no usable signal; we flag rather than fail on them.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            capture_output=True, text=True,
        ).stdout.split()
    except FileNotFoundError:
        return False
    for tok in out:
        try:
            if float(tok) < 0.5:     # sub-half-second => aborted/empty capture
                return True
        except ValueError:
            continue
    return False


def _convert_one(args):
    """Worker: convert src -> dst with ffmpeg. Returns (dst, status, detail)."""
    src, dst, codec = args
    if not src or not os.path.exists(src):
        return (dst, "missing_source", src)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return (dst, "skipped", "already exists")
    tmp = dst + ".partial"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-map", "a",            # audio stream(s) only
        "-c:a", codec,          # PCM sample format; native rate/channels kept
        "-f", "wav",
        tmp,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return (dst, "error", "ffmpeg not found on PATH")
    ok = (proc.returncode == 0 and os.path.exists(tmp)
          and os.path.getsize(tmp) >= _MIN_WAV_BYTES)
    if not ok:
        if os.path.exists(tmp):
            os.remove(tmp)
        # Distinguish "raw recording has no usable audio" (expected in this
        # dataset) from a genuine conversion failure (should never happen).
        if _looks_empty(src):
            return (dst, "empty_source", "aborted/empty recording")
        return (dst, "error", (proc.stderr or "ffmpeg failed").strip()[:300])
    os.replace(tmp, dst)        # atomic: only a complete file appears at dst
    return (dst, "converted", "")


def _read_manifest():
    if not os.path.exists(MANIFEST_CSV):
        sys.exit(
            f"ERROR: {MANIFEST_CSV} not found. Run download_mpower.py first."
        )
    with open(MANIFEST_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("ERROR: manifest is empty; nothing to convert.")
    return rows


def convert_all() -> int:
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: ffmpeg not found on PATH. Install it first, e.g.\n"
            "    brew install ffmpeg"
        )

    codec = _pcm_codec()
    rows = _read_manifest()
    os.makedirs(WAV_DIR, exist_ok=True)

    # Build the work list and remember each row's destination for the manifest.
    tasks = []
    dst_for_row = []
    for i, row in enumerate(rows):
        src = row.get("path", "").strip()
        dst = _out_path(row.get("recordId", ""), row.get("column", ""), i)
        dst_for_row.append(dst)
        if src:
            tasks.append((src, dst, codec))

    workers = int(os.environ.get("WORKERS", os.cpu_count() or 4))
    print(f"Converting {len(tasks):,} file(s) -> WAV ({codec}) "
          f"using {workers} worker(s). Existing WAVs are skipped.")

    counts = {"converted": 0, "skipped": 0, "missing_source": 0,
              "empty_source": 0, "error": 0}
    errors = []
    no_signal = set()   # dst paths that had no usable audio
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_convert_one, t) for t in tasks]
        for fut in as_completed(futures):
            dst, status, detail = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                errors.append((dst, detail))
            elif status == "empty_source":
                no_signal.add(dst)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done:,}/{len(tasks):,}  "
                      f"(converted={counts['converted']:,} "
                      f"skipped={counts['skipped']:,} "
                      f"empty={counts['empty_source']:,} "
                      f"errors={counts['error']:,})", flush=True)

    # Rewrite the manifest with a wavPath column pointing at the WAV we made
    # (blank where there was no source file or no usable audio).
    fieldnames = list(rows[0].keys())
    if "wavPath" not in fieldnames:
        fieldnames.append("wavPath")
    skipped_rows = []
    with open(MANIFEST_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row, dst in zip(rows, dst_for_row):
            made_wav = row.get("path", "").strip() and os.path.exists(dst)
            row["wavPath"] = dst if made_wav else ""
            w.writerow(row)
            if row.get("path", "").strip() and not made_wav:
                skipped_rows.append(row)

    # Record which source recordings yielded no WAV, so the gap is documented.
    if skipped_rows:
        with open(SKIPPED_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for row in skipped_rows:
                w.writerow(row)

    print(f"WAV files in: {WAV_DIR}")
    print(f"Manifest updated with wavPath -> {MANIFEST_CSV}")
    print(f"Summary: converted={counts['converted']:,} "
          f"skipped={counts['skipped']:,} "
          f"empty_source={counts['empty_source']:,} "
          f"missing_source={counts['missing_source']:,} "
          f"errors={counts['error']:,}")
    if skipped_rows:
        print(f"{len(skipped_rows):,} recording(s) had no usable audio "
              f"-> logged to {SKIPPED_CSV}")
    if errors:
        print("First few real errors:")
        for dst, detail in errors[:5]:
            print(f"  {os.path.basename(dst)}: {detail}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(convert_all())
