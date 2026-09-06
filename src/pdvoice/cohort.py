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
