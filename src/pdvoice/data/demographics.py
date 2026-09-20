"""Attach age and sex to the external corpora.

Each corpus stores demographics differently, and none of it is in the manifests:

    NeuroVoz    data/metadata/metadata_{hc,pd}.csv, joined on the numeric speaker ID
    IPVS        per-group .xlsx with the real header on the SECOND row, joined on the
                speaker's name -- which is also what the folder is called
    MDVR-KCL    nothing. The readme documents the room and the phone and not one word
                about the participants, and the filenames carry only ID, status, H&Y and
                UPDRS. There is no age to attach, so any age-matched analysis simply
                cannot include it. Do not fabricate a substitute.

PRIVACY. IPVS speaker ids ARE participant names (``ipvs_yhc_Alberto_R``). Anything this
module feeds into a saved artifact must be aggregated first -- counts, means, AUROCs --
never the ids themselves.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from pdvoice.config import IPVS_DIR, NEUROVOZ_DIR

_IPVS_SHEETS = [
    ("15 Young Healthy Control", "15 YHC.xlsx"),
    ("22 Elderly Healthy Control", "Tab 3.xlsx"),
    ("28 People with Parkinson's disease", "TAB 5.xlsx"),
]


def neurovoz_ages(root=NEUROVOZ_DIR) -> pd.DataFrame:
    """subject_id -> age, sex. Returns an empty frame if the metadata is absent."""
    rows = []
    for status in ("hc", "pd"):
        f = Path(root) / "data" / "metadata" / f"metadata_{status}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = [str(c).strip() for c in d.columns]
        if not {"ID", "Age"} <= set(d.columns):
            continue
        d = d.drop_duplicates("ID")
        for _, r in d.iterrows():
            rows.append({"subject_id": f"neurovoz_{status}_{int(r['ID']):04d}",
                         "age": pd.to_numeric(r["Age"], errors="coerce"),
                         "sex": str(r.get("Sex", "")).strip().upper()[:1] or None})
    return pd.DataFrame(rows).dropna(subset=["age"])


def ipvs_ages(root=IPVS_DIR) -> pd.DataFrame:
    """subject_id -> age, sex, joined on the speaker's name (which is the folder name)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for folder, fname in _IPVS_SHEETS:
            f = Path(root) / folder / fname
            if not f.exists():
                continue
            # the real column names live on the second row of these sheets
            d = pd.read_excel(f, header=1)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if not {"name", "surname", "age"} <= set(d.columns):
                continue
            tag = {"15": "yhc", "22": "ehc", "28": "pd"}[folder.split()[0]]
            for _, r in d.iterrows():
                name, sur = str(r["name"]).strip(), str(r["surname"]).strip()
                if not name or name.lower() in ("nan", "name"):
                    continue
                rows.append({"subject_id": f"ipvs_{tag}_{name}_{sur}",
                             "age": pd.to_numeric(r["age"], errors="coerce"),
                             "sex": str(r.get("sex", "")).strip().upper()[:1] or None})
    return pd.DataFrame(rows).dropna(subset=["age"])


def _name_key(subject_id: str) -> str:
    """Normalised join key for IPVS.

    The manifest builds ids from folder names and the spreadsheets from typed columns,
    so the two disagree on case, spacing and -- for patients -- an embedded batch folder
    (``ipvs_pd_1_5_Name_C``). Drop the ``ipvs_<group>`` prefix and any numeric batch
    parts, then strip everything that is not a letter or digit. Raw string equality
    matches 14 of 65 speakers; this matches 60.
    """
    parts = [x for x in subject_id.split("_")[2:] if not x.isdigit()]
    return re.sub(r"[^a-z0-9]", "", "".join(parts).lower())


def attach_age(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add ``age`` and ``sex`` columns to a manifest. NaN where unknown."""
    table = {"neurovoz": neurovoz_ages, "ipvs": ipvs_ages}.get(dataset)
    out = df.copy()
    if table is None:                       # mdvr_kcl, and anything else without demographics
        out["age"] = pd.NA
        out["sex"] = pd.NA
        return out
    t = table().drop_duplicates("subject_id")
    if dataset == "ipvs":
        t = t.assign(_k=t.subject_id.map(_name_key)).drop_duplicates("_k").set_index("_k")
        k = out["subject_id"].map(_name_key)
    else:
        t = t.set_index("subject_id")
        k = out["subject_id"]
    out["age"] = k.map(t["age"])
    out["sex"] = k.map(t["sex"])
    return out


def age_gap(df: pd.DataFrame) -> dict:
    """PD/HC mean age and gap, one row per speaker. NaN-safe."""
    s = df.dropna(subset=["age"]).drop_duplicates("subject_id")
    if s.empty:
        return {"n_with_age": 0}
    pd_a = s.loc[s.label == 1, "age"]
    hc_a = s.loc[s.label == 0, "age"]
    return {"n_with_age": int(len(s)),
            "n_pd": int(len(pd_a)), "n_hc": int(len(hc_a)),
            "age_pd": round(float(pd_a.mean()), 1) if len(pd_a) else None,
            "age_hc": round(float(hc_a.mean()), 1) if len(hc_a) else None,
            "age_gap": (round(float(pd_a.mean() - hc_a.mean()), 1)
                        if len(pd_a) and len(hc_a) else None)}
