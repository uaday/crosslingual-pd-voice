#!/usr/bin/env python3
"""
Download the mPower Voice Activity table (syn5511444) from Synapse,
including every audio file handle, into this directory.

Auth: reads a Synapse personal access token from the SYNAPSE_AUTH_TOKEN
environment variable (never hard-code the token in this file).

The download is *resumable*: files already present in the local cache are
skipped, so you can re-run this script after an interruption.

Usage:
    export SYNAPSE_AUTH_TOKEN="your-token-here"
    python3 download_mpower.py            # download everything
    LIMIT=25 python3 download_mpower.py   # dry-ish run: only first 25 rows
"""

import os
import sys
import csv

import synapseclient

TABLE_ID = "syn5511444"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "synapse_cache")   # audio files land here
META_CSV = os.path.join(HERE, "voice_activity_metadata.csv")
MANIFEST_CSV = os.path.join(HERE, "audio_file_manifest.csv")


def main() -> int:
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    has_config = os.path.exists(os.path.expanduser("~/.synapseConfig"))
    if not token and not has_config:
        sys.exit(
            "ERROR: no Synapse credentials found.\n"
            "  1. Log in at https://www.synapse.org and accept the mPower "
            "data-use terms on the table page (syn5511444).\n"
            "  2. Create a personal access token (Account Settings -> "
            "Personal Access Tokens) with 'View' and 'Download' scopes.\n"
            "  3. Provide it EITHER by:\n"
            "       export SYNAPSE_AUTH_TOKEN=\"...\"\n"
            "     OR by creating ~/.synapseConfig containing:\n"
            "       [authentication]\n"
            "       authtoken = ...\n"
            "     then re-run this script."
        )

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Keep the file cache inside this project directory (not ~/.synapseCache),
    # so all the audio stays together and disk usage is easy to see.
    syn = synapseclient.Synapse(cache_root_dir=CACHE_DIR)
    if token:
        syn.login(authToken=token)
    else:
        syn.login()  # picks up ~/.synapseConfig
    who = syn.getUserProfile()
    print(f"Logged in as: {who.get('userName')} ({who.get('ownerId')})")

    # Which columns actually hold audio file handles? Detect instead of
    # hard-coding, so this works even if the schema differs.
    columns = list(syn.getTableColumns(TABLE_ID))
    fh_cols = [c["name"] for c in columns if c["columnType"] == "FILEHANDLEID"]
    print(f"File-handle (audio) columns: {fh_cols}")
    if not fh_cols:
        sys.exit("ERROR: no FILEHANDLEID columns found on this table.")

    limit = os.environ.get("LIMIT")
    query = f"SELECT * FROM {TABLE_ID}"
    if limit:
        query += f" LIMIT {int(limit)}"
        print(f"LIMIT set -> only fetching first {int(limit)} rows (test run).")

    print(f"Querying table: {query}")
    results = syn.tableQuery(query)
    df = results.asDataFrame()
    print(f"Rows: {len(df):,}")

    # Save the metadata table (healthCode, recordId, timestamps, etc.).
    df.to_csv(META_CSV, index=False)
    print(f"Metadata written -> {META_CSV}")

    total = sum(int(df[c].notna().sum()) for c in fh_cols)
    print(f"Downloading {total:,} audio files across {len(fh_cols)} column(s). "
          "Already-cached files are skipped.")

    # Bulk-download every file handle for the audio columns. Returns a dict
    # {file_handle_id (str): local_path}. Resumable via the cache above.
    fh_to_path = syn.downloadTableColumns(results, fh_cols)
    print(f"Downloaded/verified {len(fh_to_path):,} file handles.")

    # Write a manifest linking each record + column to its local audio path.
    id_col = "recordId" if "recordId" in df.columns else df.columns[0]
    n_written = 0
    with open(MANIFEST_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recordId", "healthCode", "column", "fileHandleId", "path"])
        for _, row in df.iterrows():
            for col in fh_cols:
                val = row.get(col)
                if val is None or (isinstance(val, float) and val != val):
                    continue
                fhid = str(int(val)) if not isinstance(val, str) else val
                path = fh_to_path.get(fhid, "")
                w.writerow([
                    row.get(id_col, ""),
                    row.get("healthCode", ""),
                    col, fhid, path,
                ])
                n_written += 1
    print(f"Manifest written ({n_written:,} rows) -> {MANIFEST_CSV}")

    # Convert the downloaded m4a/ALAC files to analysis-ready WAV, unless the
    # caller opted out (SKIP_WAV=1). Conversion is resumable and skips files
    # that are already done, so re-running is cheap.
    if os.environ.get("SKIP_WAV") not in ("1", "true", "True"):
        try:
            import convert_to_wav
            print("\nConverting downloaded audio to WAV ...")
            convert_to_wav.convert_all()
        except SystemExit as e:
            # convert_to_wav uses sys.exit on hard failures (e.g. no ffmpeg);
            # keep the download itself reported as successful.
            print(f"WAV conversion did not run: {e}")
        except Exception as e:
            print(f"WAV conversion failed (download is intact): {e}")
    else:
        print("SKIP_WAV set -> leaving audio in original m4a form.")

    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
