"""Cohort construction — the ONE implementation.

Before this module, `age_match` existed in four scripts plus the report notebook.
Five copies is not just clutter: they drifted. The cap-10 and cap-ALL scale-up runs
produced age-matched cohorts with different mean ages (57 vs 60) purely because the
copies received their rows in different orders, and the selection depended on that
order. Results changed and nothing raised.

Everything that builds a cohort now imports from here.

    load_mpower_cohort()  ->  labelled + QC'd manifest with an `age` column
    cap_per_subject()     ->  at most N recordings per speaker (seeded)
    age_match_ids()       ->  frequency-match PD/HC age per bin, ORDER-INDEPENDENT
    subject_level()       ->  average recording probabilities to one row per speaker
    subject_age_stats()   ->  PD/HC mean age and gap, ONE row per speaker
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdvoice.config import MPOWER_DIR, SEED
from pdvoice.data.mpower import load_mpower
from pdvoice.preprocess import quality_control

AGE_BIN_WIDTH = 5


def attach_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``age`` column from the mPower Demographics Survey (joined on healthCode)."""
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    age = dict(zip(demo["healthCode"].astype(str), demo["age"]))
    out = df.copy()
    out["age"] = out["subject_id"].str.replace("mpower_", "", regex=False).map(age)
    return out


def cap_per_subject(df: pd.DataFrame, cap: int | None, seed: int = SEED) -> pd.DataFrame:
    """Keep at most ``cap`` recordings per speaker, chosen by a seeded shuffle.

    One mPower speaker contributed 355 recordings and another contributed 1. Without a
    cap the heavy users dominate the training distribution -- measurably so: at cap-ALL
    the within-mPower score falls from 0.649 to 0.604 and calibration degrades badly.
    ``cap=None`` or ``0`` keeps everything.

    THE INDEX IS PRESERVED ON PURPOSE. Callers routinely hold a feature matrix whose rows
    line up with the *input* frame and then select with ``X[df.index.to_numpy()]``.
    Resetting the index here silently pairs every recording with someone else's
    embedding, which does not raise -- it just returns chance performance. Reset the
    index at the end of a pipeline, never in the middle of one.
    """
    if not cap:
        return df
    shuffled = df.sample(frac=1.0, random_state=seed)
    kept = shuffled[shuffled.groupby("subject_id").cumcount().to_numpy() < cap]
    return kept.sort_index()


def age_match_ids(df: pd.DataFrame, seed: int = SEED,
                  bin_width: int = AGE_BIN_WIDTH) -> set[str]:
    """Frequency-match the PD and HC age distributions; return the speaker ids to keep.

    Within each age bin, keep ``min(n_PD, n_HC)`` speakers of each class. The result is
    a cohort where age carries no information about the label -- which is verifiable:
    a classifier given age alone scores ~0.92 before matching and ~0.49 after.

    ORDER-INDEPENDENCE is the reason this function exists. Candidate ids are **sorted**
    before sampling, so the same seed selects the same speakers no matter how the caller
    happened to order its rows. The previous copies sampled in arrival order, so capping
    (which reorders) silently changed the cohort.
    """
    subj = (df.dropna(subset=["age"])
              .drop_duplicates("subject_id")[["subject_id", "label", "age"]])
    subj = subj.assign(bin=(subj["age"] // bin_width * bin_width).astype(int))
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for _, g in subj.groupby("bin", sort=True):
        pd_ids = np.sort(g.loc[g.label == 1, "subject_id"].to_numpy())
        hc_ids = np.sort(g.loc[g.label == 0, "subject_id"].to_numpy())
        k = min(len(pd_ids), len(hc_ids))
        if k:
            keep += rng.choice(pd_ids, k, replace=False).tolist()
            keep += rng.choice(hc_ids, k, replace=False).tolist()
    return set(keep)


def attach_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``sex`` column ("Male"/"Female"/NaN) from the mPower Demographics Survey."""
    demo = pd.read_csv(MPOWER_DIR / "Demographics Survey.csv")
    g = dict(zip(demo["healthCode"].astype(str), demo["gender"]))
    out = df.copy()
    sex = out["subject_id"].str.replace("mpower_", "", regex=False).map(g)
    out["sex"] = sex.where(sex.isin(["Male", "Female"]))
    return out


def demographic_match_ids(df: pd.DataFrame, on: tuple[str, ...] = ("age",),
                          seed: int = SEED, bin_width: int = AGE_BIN_WIDTH) -> set[str]:
    """Frequency-match PD and HC jointly across one or more attributes.

    ``on=("age",)`` reproduces :func:`age_match_ids`. ``on=("age", "sex")`` matches within
    each age-bin x sex cell, which matters because **the two confounds are independent**:
    in mPower the patient group is 34.6% female against 18.6% of controls, and age matching
    only narrows that to 33.4% vs 22.3% (chi-square p = 5.6e-05, still imbalanced). Sex
    alone predicts the label at AUROC 0.58, and the embeddings encode sex at 0.97 -- so a
    model has full access to it whether we intend that or not.

    Matching on more attributes costs speakers, since every cell is capped at
    min(n_PD, n_HC). That trade is the point: a smaller honest cohort beats a larger
    confounded one.
    """
    d = df.copy()
    strata = []
    for col in on:
        if col == "age":
            d = d.dropna(subset=["age"])
            d["_age_bin"] = (d["age"] // bin_width * bin_width).astype(int)
            strata.append("_age_bin")
        else:
            d = d.dropna(subset=[col])
            strata.append(col)
    subj = d.drop_duplicates("subject_id")[["subject_id", "label", *strata]]

    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for _, g in subj.groupby(strata, sort=True):
        pd_ids = np.sort(g.loc[g.label == 1, "subject_id"].to_numpy())
        hc_ids = np.sort(g.loc[g.label == 0, "subject_id"].to_numpy())
        k = min(len(pd_ids), len(hc_ids))
        if k:
            keep += rng.choice(pd_ids, k, replace=False).tolist()
            keep += rng.choice(hc_ids, k, replace=False).tolist()
    return set(keep)


def age_match(df: pd.DataFrame, seed: int = SEED,
              bin_width: int = AGE_BIN_WIDTH) -> pd.DataFrame:
    """``age_match_ids`` applied — returns the matched rows, original order preserved."""
    keep = age_match_ids(df, seed=seed, bin_width=bin_width)
    return df[df["subject_id"].isin(keep)]


def load_mpower_cohort(cap: int | None = None, seed: int = SEED,
                       labelled_only: bool = True, with_qc: bool = True) -> pd.DataFrame:
    """mPower manifest, ready to use: labelled, quality-controlled, capped, with age.

    This is the entry point every mPower experiment should call, so that "the cohort"
    means one thing across the project.

    NOTE ON ORDER: quality control runs BEFORE capping, which is the opposite of what the
    original scripts did. Capping first means a speaker's N slots can be spent on files QC
    then deletes, so they silently contribute fewer recordings than intended -- and that
    loss was not class-neutral (patients lost ~15% of slots against ~5% for controls,
    because the unreadable rows were PD-enriched). QC first costs a full pass over the file
    headers, which is slow on 55k files but correct.
    """
    df = load_mpower()
    if labelled_only:
        df = df[df["label"].isin([0, 1])]
    df = df.reset_index(drop=True)
    if with_qc:
        df, _ = quality_control(df)
        df = df.reset_index(drop=True)
    df = cap_per_subject(df, cap, seed).reset_index(drop=True)
    return attach_age(df)


def describe_cohort(df: pd.DataFrame) -> str:
    """One-line summary for logs: speakers, class split, and the age gap."""
    s = df.drop_duplicates("subject_id")
    npd, nhc = int((s.label == 1).sum()), int((s.label == 0).sum())
    apd, ahc = s.loc[s.label == 1, "age"].mean(), s.loc[s.label == 0, "age"].mean()
    return (f"{len(df):,} recordings / {len(s):,} speakers "
            f"(PD {npd:,} age {apd:.1f} | HC {nhc:,} age {ahc:.1f} | gap {apd - ahc:+.1f} yr)")


def subject_age_stats(df: pd.DataFrame) -> dict:
    """PD/HC mean age and the gap, computed on ONE row per speaker.

    Demographics need deduplicating for the same reason metrics do (see
    ``subject_level``): recording counts are wildly uneven and not class-neutral.
    Averaging ``age`` over a recording-level frame silently weights each speaker by
    how much they recorded. On the full mPower cache that is not a rounding error --
    the top 1% of speakers hold 24% of the rows (one has 355), and the recording-level
    mean reports a PD/HC age gap of 23.6 years where the gap between speakers is 27.5.

    Capping hides the problem rather than fixing it: at cap-3 the two agree to 0.3 yr,
    so a wrong number here looks right until someone runs it uncapped.
    """
    s = df.drop_duplicates("subject_id")
    apd = float(s.loc[s.label == 1, "age"].mean())
    ahc = float(s.loc[s.label == 0, "age"].mean())
    return {"age_pd": round(apd, 2), "age_hc": round(ahc, 2),
            "age_gap": round(apd - ahc, 2), "age_level": "subject",
            "n_pd": int((s.label == 1).sum()), "n_hc": int((s.label == 0).sum())}


def subject_level(subject_ids, y, p) -> pd.DataFrame:
    """Average recording-level probabilities to ONE row per speaker.

    Every reported metric in this project is computed on the output of this function.
    Recording-level scoring would let a speaker with 355 recordings outweigh 355
    speakers with one each.
    """
    return (pd.DataFrame({"subject_id": np.asarray(subject_ids),
                          "y": np.asarray(y), "p": np.asarray(p)})
            .groupby("subject_id", as_index=False)
            .agg(y=("y", "first"), p=("p", "mean")))
