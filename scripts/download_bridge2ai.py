"""Download the Bridge2AI-Voice adult audio release from Synapse.

Needs a Synapse personal access token (View + Download) in ~/.synapseConfig and approved
DACO access to syn72370534. The data are under a data use agreement: they stay in
pd-datasets/bridge2ai/, which git ignores, and are never committed or shared.

Files are listed from the versioned file view syn74341293 and written into the BIDS-like
layout of the release (sub-*/ses-*/audio/..., phenotype/...). Calling syn.get() per file
costs several API round trips and ran at ~1.6 files/s regardless of thread count, so
download URLs are instead requested in batches of 100 from /fileHandle/batch and fetched
directly. Every file is checked against the MD5 in the view; re-running skips files that
are already present and correct, so an interrupted run can simply be restarted.

Run:  .venv\\Scripts\\python.exe -u scripts\\download_bridge2ai.py [--only-metadata]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import synapseclient

from pdvoice.config import BRIDGE2AI_DIR

VIEW = "syn74341293"
PREFIX = "Bridge2AI-Voice Adult Cohort Audio Dataset/Dataset - Adult Data/"
ROOT = BRIDGE2AI_DIR
BATCH = 100


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def presigned(syn, rows) -> dict[str, str]:
    body = {"requestedFiles": [{"fileHandleId": str(r.dataFileHandleId), "associateObjectId": r.id,
                                "associateObjectType": "FileEntity"} for r in rows],
            "includePreSignedURLs": True, "includeFileHandles": False}
    res = syn.restPOST("/fileHandle/batch", body=json.dumps(body), endpoint=syn.fileHandleEndpoint)
    return {x["fileHandleId"]: x["preSignedURL"] for x in res["requestedFiles"] if x.get("preSignedURL")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-metadata", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    syn = synapseclient.Synapse(silent=True)
    syn.login()

    idx = syn.tableQuery(f"SELECT id, name, path, currentVersion, dataFileHandleId, "
                         f"dataFileSizeBytes, dataFileMD5Hex FROM {VIEW}").asDataFrame()
    idx.to_csv(ROOT / f"_index_{VIEW}.csv", index=False)
    idx["rel"] = idx.path.str.replace(PREFIX, "", regex=False)
    if args.only_metadata:
        idx = idx[~idx.name.str.endswith((".wav", ".json")) | (idx.rel.str.count("/") == 0)]

    todo = [r for r in idx.itertuples()
            if not ((ROOT / r.rel).exists() and (ROOT / r.rel).stat().st_size == r.dataFileSizeBytes
                    and md5(ROOT / r.rel) == r.dataFileMD5Hex)]
    print(f"{len(idx):,} files in the view; {len(idx) - len(todo):,} already present and "
          f"correct; {len(todo):,} to fetch ({sum(r.dataFileSizeBytes for r in todo) / 1e9:.2f} GB)",
          flush=True)

    http = requests.Session()
    http.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.workers))

    def fetch(row, url) -> str:
        dest = ROOT / row.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        # About one presigned URL in five is refused (403 AccessDenied) for no visible
        # reason: same bucket, file type and handle kind as the ones that work. Ask for a
        # fresh URL and retry; if that keeps failing, fall back to the slow syn.get().
        for attempt in range(4):
            with http.get(url, stream=True, timeout=120) as r:
                if r.status_code == 403 and attempt < 3:
                    url = presigned(syn, [row])[str(row.dataFileHandleId)]
                    continue
                if r.status_code == 403:
                    ent = syn.get(row.id, version=int(row.currentVersion),
                                  downloadLocation=str(dest.parent), ifcollision="overwrite.local")
                    Path(ent.path).replace(tmp)
                    break
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                break
        if md5(tmp) != row.dataFileMD5Hex:
            tmp.unlink()
            return "bad-md5"
        tmp.replace(dest)
        return "ok"

    counts: dict[str, int] = {}
    failed = []
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for b in range(0, len(todo), BATCH):
            rows = todo[b:b + BATCH]
            try:
                urls = presigned(syn, rows)
            except Exception as e:
                failed += [(r.id, r.rel, f"url: {e!r}"[:200]) for r in rows]
                continue
            futs = {ex.submit(fetch, r, urls[str(r.dataFileHandleId)]): r
                    for r in rows if str(r.dataFileHandleId) in urls}
            failed += [(r.id, r.rel, "no url") for r in rows if str(r.dataFileHandleId) not in urls]
            for f in as_completed(futs):
                try:
                    s = f.result()
                except Exception as e:
                    s = "error"
                    failed.append((futs[f].id, futs[f].rel, repr(e)[:200]))
                counts[s] = counts.get(s, 0) + 1
            done += len(rows)
            if (b // BATCH) % 10 == 9 or done == len(todo):
                print(f"  {done:,}/{len(todo):,}  {counts}", flush=True)

    if failed:
        pd.DataFrame(failed, columns=["id", "rel", "error"]).to_csv(ROOT / "_failed.csv", index=False)
        print(f"{len(failed)} failed; listed in {ROOT / '_failed.csv'}. Re-run to retry.", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
