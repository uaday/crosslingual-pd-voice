# PIPELINE — the method, step by step, as it actually runs

Every stage in order, with the code that does it, the artifact it produces, and the
findings that came out of it. **Nothing here is a proposal. Everything listed as ✅ has
already been run** — this is the map of what exists, so nothing gets repeated.

Verified against the code on 2026-09-27, not from memory.

> ⚠️ **This supersedes `RESEARCH_STATE.md` §4.2–4.3.** That section documents
> `cap → quality control`. The code does **`quality control → cap`** — the order was
> corrected on 2026-09-06 and the doc was never updated. Do not copy §4.3 into the thesis.

---

# STEP 1 — Datasets

`src/pdvoice/data/` · registry `LOADERS`

| dataset | lang | recordings | speakers | tasks | label source | role |
|---|---|---|---|---|---|---|
| **mPower** | en | **57,612** | **5,648** | vowel | self-reported | **TRAIN** |
| **NeuroVoz** | es | 2,976 | 114 | vowel, speech, ddk | clinician | test |
| **IPVS** | it | 831 | 65 | vowel, speech | clinician | test |
| **MDVR-KCL** | en | 73 | 37 | speech | clinician | test |
| Bridge2AI | en | — | — | — | clinician | 🔲 loader raises `NotImplementedError`; no data access |
| EWA-DB | — | — | — | — | — | 🔲 same |

- mPower has **2,129 unlabelled** recordings, dropped at step 3.
- Two English corpora (mPower train, MDVR-KCL test) — lets a *language* change be
  separated from a *dataset* change.

**Known defects, all measured:** F-13 (defects are class-correlated), F-17 (IPVS 44.1 kHz
batch is PD-only, encoded at 0.79 but not load-bearing), F-28 (NeuroVoz vowel task codes
leak the label — metadata alone gives **0.895**).

⚠️ **IPVS speaker IDs are real participant names.** Anything saved from them must be
aggregated first.

---

# STEP 2 — Audio processing  ✅ FINAL

**Locked. Do not revisit.** Every parameter below is decided, its justification is either
measured or explicitly marked as an untested default, and the numbers are ready to quote.

`src/pdvoice/preprocess.py` · `src/pdvoice/utils/audio.py`

---

## 2.0 The locked configuration

| parameter | value | defined in | basis |
|---|---|---|---|
| `SAMPLE_RATE` | **16 000 Hz**, mono | `config.py:36` | Nyquist — speech energy is < 8 kHz. Also the SSL encoders' required input. |
| `trim_silence(top_db=)` | **30 dB** | `utils/audio.py:78` | librosa default; vowels only, discards 2–6% |
| VAD | **Silero VAD** | `preprocess.py:27` | connected speech only; energy-trim fallback |
| `VOWEL_CORE_S` | **2.0 s**, centred | `preprocess.py:20` | ⚠️ **untested default** — see 2.5 |
| `CHUNK_S` | **5.0 s** | `preprocess.py:21` | ⚠️ **untested default** — see 2.5 |
| `CHUNK_OVERLAP` | **0.5** (50%) | `preprocess.py:22` | ⚠️ **untested default** |
| `TARGET_DBFS` | **−23.0 dBFS** | `preprocess.py:25` | EBU R128 broadcast reference |
| `peak_ceiling` | 0.98 | `utils/audio.py:38` | clipping guard |
| `MIN_DURATION_S` | **1.0 s** | `preprocess.py:19` | QC floor |

Identical for every dataset. No per-corpus tuning anywhere.

---

## 2.1 Decode and resample → 16 kHz mono

Inputs are mixed: 44.1 kHz (mPower, NeuroVoz, MDVR-KCL) and 16 kHz (most of IPVS). All
converted to 16 kHz mono float32 in [−1, 1].

**All 61,492 files are already single-channel**, so the mono conversion is a no-op — worth
stating, because it removes a class of doubt for free.

**Defensible sentence:** *"All audio was resampled to 16 kHz mono. Speech energy lies below
8 kHz, so by the Nyquist criterion this preserves all speech information; it is also the
input rate the self-supervised encoders require."*

---

## 2.2 Remove non-speech

| task | method | retains |
|---|---|---|
| sustained vowel | `trim_silence(top_db=30)` | 94–98% |
| connected speech | **Silero VAD** | 51–85% |

⚠️ **This stage is class-correlated, and the direction is not consistent.** Derived from
the F-31 per-speaker measurements:

| corpus | HC retained | PD retained | difference |
|---|---|---|---|
| IPVS | 76.1% | **59.2%** | **−16.9 pts** |
| MDVR-KCL | 59.9% | 52.3% | −7.6 pts |
| NeuroVoz | 80.9% | **88.7%** | **+7.8 pts** |

Patients pause more, so VAD deletes more of their recording — **except on NeuroVoz**,
where the "speech" task is single words and the effect reverses.

**This must be declared.** It is the same class of defect as F-13, but located in our own
processing rather than in the data.

**Why we keep it anyway** — three reasons, all real:
1. The encoder needs bounded-length input.
2. Raw silence carries **examiner speech** and room noise, which is a worse leak than the
   one it fixes.
3. The cost is measured, not guessed (F-31), and the fix is scoped: add a **timing feature
   block alongside** the embeddings rather than removing the VAD.

---

## 2.3 Windowing

| task | window |
|---|---|
| sustained vowel | central **2.0 s** (`stable_core`; returns the whole clip if shorter) |
| connected speech | **5.0 s** chunks, **50%** overlap, embeddings averaged per recording |

⚠️ **Windows per recording vary by 25× across corpora that are all labelled "connected
speech":**

| corpus / task | mean windows |
|---|---|
| NeuroVoz / speech | **1.0** — too short to chunk at all |
| IPVS / speech | 11.2 (max 47) |
| MDVR-KCL / speech | **31.1** (max 69) |

**State this.** NeuroVoz "speech" is single words averaging 4.6 s. It and MDVR-KCL are not
comparable quantities of material, and results should not be pooled across them without
saying so.

---

## 2.4 Amplitude normalisation → −23 dBFS

An explicit, named stage rather than a library side effect — this was **supervisor ask 2**.

| corpus / task | in | out |
|---|---|---|
| mPower / vowel | −27.1 | −23.0 |
| IPVS / vowel | −26.7 | −23.0 |
| IPVS / speech | −29.7 | −23.2 |
| NeuroVoz / speech | −32.2 | −23.0 |
| MDVR-KCL / speech | −35.7 | −23.9 |

**This is the one stage proven to change nothing.** `audits/amplitude_vs_harmonics.py`
attenuates real recordings by **60 dB**, renormalises, and re-measures F0, HNR, jitter and
shimmer: **all change by 0.00%** (F-14). It is a pure scalar multiply.

**Defensible sentence:** *"Amplitude normalisation is a scalar operation and provably
alters no measurement the analysis uses; we verified this by attenuating recordings across
a 60 dB range and re-measuring F0, HNR, jitter and shimmer, all of which changed by
0.00%."*

**What it discards:** absolute loudness — hypophonia is a genuine PD sign. Unusable
regardless: on an uncalibrated phone, quiet is indistinguishable from a phone held further
away.

---

## 2.5 Are 2 s and 5 s justified? — the honest position

**No citation supports them.** Checked 2026-09-27 (`docs/LITERATURE.md` §5):

- **Klempíř & Krupička 2024** (*Sensors* 24(17):5520) — **whole file, no segmentation.**
  *"we found this segmentation to be unnecessary."*
- **Klempíř et al. 2025** (*CSBJ*) — **whole file**, mean aggregation.
- **Koudounas et al. 2025** (*Interspeech*, MVP) — a fixed **5.0 s** pad-or-truncate, but
  voice pathology generally, one clip, not averaged overlapping chunks.
- **No source found for a 1–3 s vowel core.**

**They are defended by experiment instead** — F-34, the 2 × 2 × 6 grid:

| | result |
|---|---|
| within mPower | L11 0.779 → **0.816**; L15 0.763 → **0.841** (CIs disjoint) |
| 9 external comparisons | mean **−0.011**, sd 0.149, range −0.293 to +0.282 |

**Decision: keep 2 s / 5 s.** Full audio improves only the confounded home benchmark —
which is where the last layer also scores highest before collapsing to chance once
age-matched (F-32). A gain visible only there is most likely a gain in age detection.

### Write this

> *"Segment lengths were fixed at 2 s for sustained vowels and 5 s overlapping chunks for
> connected speech. Rather than assume these, we evaluated them against passing each
> recording in full — the configuration used by the studies we benchmark against. Full-length
> input improved within-corpus AUROC from 0.763 to 0.841 with non-overlapping confidence
> intervals, but changed external transfer by −0.011 on average across nine comparisons
> (range −0.293 to +0.282). We therefore retained the shorter windows, since an improvement
> confined to the demographically confounded training corpus is not evidence of
> disease-relevant signal."*

### Never write this

- ❌ *"following standard practice"* / *"the field convention"* — **not supported**
- ❌ *"we found no difference"* — say **"no difference our cohorts could detect"**; a +0.05
  effect would be invisible in ±0.29 noise
- ❌ that the windowing is tuned or optimal — two points were tested, not a range

---

## 2.6 What this step discards — declare unprompted

| stage | discards | measured cost |
|---|---|---|
| VAD | pauses, hesitation, speech rate | **F-31** 🔴 — timing alone is worth IPVS **0.878**, still **0.769** age-matched |
| 5 s chunks | long-range drift within an utterance | **F-34** — between-chunk variance is 10–13% of within-chunk |
| 2 s vowel core | **80%** of every 10 s mPower vowel | **F-34** — costs nothing that transfers |
| normalisation | absolute loudness (hypophonia) | **F-14** — 0.00% change to any used measure |

---

## 2.7 Limitations to state in the thesis

1. **VAD retention is class-correlated** — up to 16.9 points, and the sign flips between
   corpora (2.2).
2. **Window counts differ 25× across "connected speech" corpora** (2.3).
3. **Segment lengths are defaults, not cited convention** — defended by F-34, not by
   literature (2.5).
4. **The windowing test was underpowered** — ±0.29 noise at n = 37–50 means only very
   large effects were detectable (F-34).
5. **Comparisons to Klempíř are not method-matched** — their numbers come from whole-file
   embeddings, ours from averaged chunks.

---

# STEP 3 — Cohort construction  ✅ FINAL

`src/pdvoice/cohort.py`

---

## 3.0 The order — and why it is not the obvious one

```
load manifest → drop unlabelled → QUALITY CONTROL → cap N per subject (seeded) → attach age
```

⚠️ **Quality control runs BEFORE the cap.** This is the reverse of what the original
scripts did, corrected 2026-09-06.

**Why it matters — this is a methods paragraph, not a detail.** Capping first spends a
speaker's N slots on files that QC then deletes, so they silently contribute fewer
recordings than intended. That loss was **not class-neutral**: patients lost **~15%** of
their slots against **~5%** for controls, because the unreadable rows were PD-enriched. QC
first costs a full pass over 55k file headers, which is slow, and correct.

⚠️ `RESEARCH_STATE.md` §4.3 still documents the old order. **Do not copy it.**

---

## 3.1 Quality control

| rule | value |
|---|---|
| minimum duration | **1.0 s** (`MIN_DURATION_S`) |
| voicing floor | drop if voiced fraction < ~0.3 **or** HNR ≈ 0 |
| amplitude | **never discard on amplitude** |

**The amplitude rule is a deliberate reversal**, adopted after supervisor review. Quiet
recordings look like failures and are not: normalise first, *then* judge voicing.
Justification is F-14 — attenuating by 60 dB changes F0, HNR, jitter and shimmer by
**0.00%**, so a quiet file carries the same information as a loud one.

Measured effect: NeuroVoz vowel 1,066 → **1,064** recordings (2 dropped under 1 s).

---

## 3.2 Capping — `cap_per_subject`

mPower recording counts are wildly uneven: **one speaker contributed 355 recordings,
another 1**, and the top 1% of speakers hold **26%** of all recordings.

Selection is a **seeded shuffle** (`seed=42`), so it is reproducible.

**Why cap at all — measured, not assumed (F-15):** at cap-ALL the within-mPower score
*falls* from 0.649 to 0.604 and calibration degrades badly. Heavy users dominate the
training distribution and they are not representative.

> ⚠️ **`cap_per_subject` preserves the DataFrame index on purpose.** Callers hold feature
> matrices whose rows align with the input frame and select with `X[df.index.to_numpy()]`.
> Resetting the index mid-pipeline pairs every recording with someone else's embedding —
> which does not raise, it just returns chance performance. Reset at the end of a
> pipeline, never in the middle.

---

## 3.3 The cohorts

| cohort | recordings | speakers | used for |
|---|---|---|---|
| cap-3, confounded | **12,053** | **4,809** | main training cohort |
| cap-1 | 4,809 | 4,809 | classical baseline, probes |
| **age-matched, cap-1** | 1,124 | **1,124** | **the honest cohort — every headline number** |

---

## 3.4 Age matching — `age_match_ids`

**Method:** frequency matching in **5-year bins**. Within each bin keep
`min(n_PD, n_HC)` speakers of each class. Seeded, and candidate ids are **sorted before
sampling** so the same seed selects the same speakers regardless of row order — an earlier
version sampled in arrival order, so capping (which reorders) silently changed the cohort.

**Verification that it worked** — the check to quote: a classifier given **age alone**
scores **0.922** before matching and **0.531** after. Age carries no label information in
the matched cohort.

| | subjects | PD | HC | age gap |
|---|---|---|---|---|
| unmatched | 4,809 | 897 | 3,912 | **+27.5 yr** |
| age-matched | **1,124** | 562 | 562 | **+0.2 yr** |

**Cost:** 3,685 speakers, 77% of the cohort. F-16 — the honest cohort is subject-limited
at ~1,124, and that ceiling is the reason several results are underpowered.

---

## 3.5 The confound, quantified

mPower PD ≈ **60.0 yr**, HC ≈ **32.6 yr** (F-1). Age coverage **99.4%**.

**The external corpora are confounded too** — this is not an mPower-only problem:

| corpus | PD | HC | gap | age coverage |
|---|---|---|---|---|
| mPower | 60.0 | 32.6 | **+27.5** | 99.4% |
| IPVS | 67.1 | 46.6 | **+20.5** | 90.4% |
| NeuroVoz | 71.1 | 64.0 | +7.1 | 95.7% |
| MDVR-KCL | — | — | — | **0%** |

⚠️ **MDVR-KCL has no age data at all.** It can never be age-matched or age-controlled, so
every MDVR result is uncorrectable for the confound. With n = 37 it is the weakest evidence
in the project on two counts at once. **State this.**

Independent corroboration: Shukla et al. 2026 report a **27.5-year** gap on Bridge2AI — a
different corpus, a different lab, the same number (`docs/LITERATURE.md` §2).

---

## 3.6 Sex — DECIDED: not matched, declared as a limitation

**Decision (2026-09-27): keep age-only matching.** The cohort is the 1,124-speaker
age-matched one. Sex is **not** matched, and the residual imbalance is reported as a
stated limitation rather than removed.

The alternative was measured before deciding, so the choice is informed rather than
inherited. On mPower, sex coverage **99.7%**:

| cohort | subjects | PD | HC | age gap | female PD | female HC | imbalance |
|---|---|---|---|---|---|---|---|
| unmatched | 4,809 | 897 | 3,912 | +27.5 | 34.6% | 18.6% | **+16.0** |
| **age-matched — USED** | **1,124** | 562 | 562 | **+0.2** | **33.6%** | **22.6%** | **+11.0** |
| age + sex matched — not used | 1,120 | 560 | 560 | +0.2 | 25.2% | 25.2% | 0.0 |

**Age matching does not fix sex.** It narrows the imbalance from +16.0 to +11.0 points and
leaves it there. `attach_sex` and `demographic_match_ids(on=("age","sex"))` exist and work;
joint matching would cost **4 speakers** and remove the imbalance entirely. It was not
adopted because every headline result is computed on the 1,124 cohort and re-running them
was judged not worth the schedule risk this late.

### What must therefore be written

> *"Speakers were frequency-matched on age in 5-year bins. Sex was not matched. In the
> matched cohort the patient group remains 33.6% female against 22.6% of controls, and a
> linear probe recovers speaker sex from the embeddings at AUROC 0.971. Sex alone predicts
> the label at 0.58. We therefore cannot exclude a residual sex confound, and report this
> as a limitation. Joint age-and-sex matching was implemented and would have cost four
> speakers; it was not adopted because the reported results are computed on the
> age-matched cohort."*

**Do not** claim the cohort is demographically matched. It is **age**-matched.

⚠️ The externals cannot be sex-matched under any option — none of the three external
loaders returns a `sex` column.

## 3.7 Splits

| purpose | splitter | property |
|---|---|---|
| Layer-1 CV, 5 folds | `StratifiedGroupKFold` | subject-disjoint **and** label-balanced |
| train/val for locking | `GroupShuffleSplit`, `VAL_FRACTION=0.15` | subject-disjoint |

**Subject-disjoint everywhere.** No speaker appears in two partitions. Evaluation is always
at **subject level** — one mPower speaker has 355 recordings, so recording-level scoring
would let one person count 355 times. This is exactly the defect behind the published
mPower 0.94 figure (F-26).

---

## 3.8 Limitations to state

1. **The honest cohort is only 1,124 speakers** — 77% of the data is spent on matching
   (F-16). Several results are underpowered as a direct consequence.
2. **MDVR-KCL cannot be age-corrected** — 0% age coverage, n = 37.
3. **Sex is deliberately not matched** (decision, 3.6) — residual 33.6% vs 22.6%, probed at 0.971, and sex alone predicts the label at 0.58.
4. **External corpora carry their own age gaps** — IPVS +20.5 yr, NeuroVoz +7.1 yr. The
   test sets are confounded too.
5. **mPower labels are self-reported.** Tested and ruled out as the dominant problem (F-2:
   +0.011 / +0.010), but it remains a data-provenance limitation.

---

# STEP 4 — Feature extraction  ✅ FINAL

Turns a cleaned waveform into a **fixed-length row of numbers**, so recordings of different
lengths become comparable. Everything before this step is audio; everything after is a
matrix.

**Two branches run in parallel, deliberately** — they measure different physics (4.5).

`src/pdvoice/features/ssl.py` · `src/pdvoice/features/classical.py` ·
`scripts/extract_windows.py`

---

## 4.1 Branch A — the frozen SSL encoder

| | |
|---|---|
| model | **XLS-R-300M** (`facebook/wav2vec2-xls-r-300m`) |
| parameters | **315,438,720 — frozen** |
| input | **raw waveform.** No mel spectrogram. |
| frame rate | **~20 ms** → hidden states `(T, 1024)` |
| pooling | **mean ⊕ std** over time → **2048** |
| precision | float16 on disk |

```
2 s vowel (32,000 samples) → XLS-R → (99, 1024) → mean⊕std → (2048,)
```

**What the two halves are:** the mean of the 1024 is steady timbre — what the voice *is*.
The std across frames is how much it *moves* — a tremor proxy. Both are needed; neither
alone was competitive.

### Why frozen

1. **~900 patients.** Fine-tuning 315M parameters on that would memorise them.
2. **Identical treatment of all four corpora.** Because the encoder never changes, a
   Spanish/English difference is the *data* differing — not a model that re-tuned itself
   on one corpus. This is what makes the transfer comparison mean anything.
3. It is the field's dominant recipe (`docs/LITERATURE.md` §5).

The encoder **never sees a Parkinson's label.**

### Caching

Keyed by cohort **and file path**, with an identity guard. A cache paired with the wrong
rows does not raise — it silently returns chance performance, so paths are compared, never
just lengths.

---

## 4.2 Branch B — classical acoustic features

**42 descriptors** (`FEATURE_NAMES`), Praat-based — the measures clinicians actually use:

| group | n | features |
|---|---|---|
| phonation | 5 | `jitter_local`, `jitter_rap`, `shimmer_local`, `shimmer_apq3`, `hnr` |
| pitch | 3 | `f0_mean`, **`f0_sd`** (monopitch), `f0_cv` |
| articulation | 8 | `f1/f2/f3_mean`, `f1/f2_sd`, `f1/f2_bandwidth`, `f2_f1_ratio` |
| spectral | 26 | `mfcc_0..12` **mean and SD** |

Expanded from 17 → 42 (**F-24**): added MFCC **standard deviations**, formant bandwidths,
and `f0_sd`, and fixed the MFCC window from librosa's 128 ms music default to **25 ms /
10 ms**. That expansion **halved the gap to SSL**, which is itself a finding — part of the
apparent SSL advantage was an under-specified baseline.

---

## 4.3 Which encoder — and the honest answer about how it was chosen

⚠️ **This selection has the same validity problem the layer choice had (F-29).**

### How it was originally done — and it was a real, significant win

The Layer-1 comparison ran on the **confounded cap-3 cohort** (n = 4,809):

| encoder | config | AUROC | 95% CI | configs tried |
|---|---|---|---|---|
| **xls-r-300m** | last layer | **0.827** | [0.811, 0.842] | **5** — LNone, L9, L11, L12, L15 |
| hubert-large | L12 | 0.745 | [0.728, 0.761] | 2 |
| wav2vec2-base | last layer | 0.720 | [0.701, 0.737] | **1** |
| wavlm-large | L12 | 0.659 | [0.640, 0.677] | 2 |

**The intervals do not overlap.** On that cohort XLS-R beats the others significantly, and
this should not be dismissed — it is a real result under the protocol used. The problem is
the protocol, and it has **two independent defects**:

**Defect 1 — unequal search budget.** XLS-R was evaluated at **five** configurations,
wav2vec2-base at **one**, and that one was its **last layer** — the single configuration
we now have direct evidence is worst for every model family (F-19: gains of +0.049 to
+0.074 over the last layer, across all five encoders). So "XLS-R wins" was in part a
statement about **search effort**, not architecture.

**Defect 2 — confounded cohort.** Deep representations encode age (F-32: the last layer
scores 0.827 confounded and **0.531** age-matched, rank 25 of 25). XLS-R's winning entry
*is* its last layer — the most age-contaminated configuration in the set. So the
comparison was partly measuring *which encoder best encodes age*.

### The corrected comparison — both defects fixed at once

`extract_all_layers.py` + `encoder_layer_sweep.py`, on the **age-matched cap-1 cohort**,
**every layer of every encoder**, equal budget:

| encoder | layers | best layer | AUROC | 95% CI | last layer | gain over last |
|---|---|---|---|---|---|---|
| **wav2vec2-base** | 13 | **0** | **0.6254** | [0.594, 0.658] | 0.5550 | +0.070 |
| hubert-large | 25 | 4 | 0.6198 | [0.587, 0.652] | 0.5515 | +0.068 |
| wavlm-large | 25 | 13 | 0.6178 | [0.584, 0.651] | 0.5684 | +0.049 |
| **xls-r-300m** *(used)* | 25 | 8 | 0.6110 | [0.575, 0.645] | 0.5366 | +0.074 |
| whisper-small | 13 | 8 | 0.5903 | [0.557, 0.622] | 0.5338 | +0.057 |

**Three things fall out, and all three go in the thesis:**

1. **No encoder wins.** Every confidence interval overlaps every other. The spread is
   0.590–0.625 across five architectures and two orders of magnitude of parameter count
   (**F-18**).
2. **The nominal leader is the *smallest* model** — `wav2vec2-base`, at its **embedding
   layer (0)**, before any transformer block runs. That is close to saying the transformer
   stack contributes nothing detectable on the honest signal.
3. **Every encoder peaks early and the last layer is worst for all five** (**F-19**),
   gains of +0.049 to +0.074 over the final layer. Consistent with F-32: depth encodes
   speaker identity, not pathology.

### Why XLS-R is still used

**Not because it won** — it ranks 4th of 5 and is statistically tied with the rest. It is
retained because (a) it is the only **massively multilingual** encoder in the set, which
matches a cross-language thesis, and (b) every downstream result is computed on it, so
switching to a statistically indistinguishable alternative would cost a full re-run and
buy nothing.

> **Write both comparisons, in sequence** — together they are a stronger result than
> either alone, because the finding is about *how encoder comparisons are conducted*, not
> just about these five models:
>
> *"An initial comparison on the uncorrected cohort found XLS-R-300M significantly ahead
> of the alternatives (0.827 against 0.745, 0.720 and 0.659, with non-overlapping
> confidence intervals). That comparison was not equal-budget — XLS-R was evaluated at
> five layers and wav2vec2-base at one, its last — and it was conducted on a cohort with a
> 27.5-year age gap between groups. Repeating it on the age-matched cohort, with every
> layer of every encoder searched, the five encoders become statistically indistinguishable
> (0.590–0.625, all intervals overlapping). XLS-R-300M was retained for its multilingual
> pre-training rather than for benchmark superiority."*

**Never write** that XLS-R was selected because it performed best. On the honest cohort it
did not.

---

## 4.4 Which layer — settled, and it did not help

| question | answer | finding |
|---|---|---|
| how was L15 chosen? | by its results **on the Spanish and Italian test sets** — selection on test data | F-29 |
| what does honest CV prefer? | **L11**; L15 ranks **13 of 25** | F-29 |
| does L11 fix transfer? | **No.** Re-extracted and re-run end to end: 1 win, 2 losses | **F-33** |
| is depth itself a confound? | **Yes.** Last layer 0.827 confounded → **0.531** age-matched, rank 25/25 | **F-32** |

**The honest conclusion is not "L11 is better."** It is that **vowel transfer fails at both
layers, and L15's apparent advantage was manufactured by selecting it on the test sets.**
Removing the selection bias removed the result rather than improving it.

---

## 4.5 Why both branches — the frame-rate argument

This is the strongest single paragraph in the methods chapter, because it is physics
rather than benchmarking.

Jitter is cycle-to-cycle variation of a glottal period of about **10 ms**. XLS-R's frame
is **20 ms**. The encoder is therefore **physically incapable** of representing jitter as
a period measurement — only its spectral consequence. Vocal tremor at **4–6 Hz** sits
comfortably above the frame rate and *is* representable.

So neither branch subsumes the other. That prediction is confirmed: on the honest
age-matched cohort, classical (**0.604**), SSL (**0.632**) and their fusion are
**statistically indistinguishable** — DeLong p = 0.718 / 0.876 / 0.620 (**F-27**).

---

## 4.6 Limitations to state

1. **The encoder was originally selected on a confounded benchmark.** Corrected, and the
   corrected comparison shows no encoder wins (4.3).
2. **The transfer layer was selected on test data** (F-29). Corrected at L11; transfer did
   not improve (F-33).
3. **The best-performing configuration is the smallest model's embedding layer**, which
   undercuts any claim that scale or depth helps here.
4. **XLS-R is retained for multilinguality and continuity, not performance.** Say so.
5. **Embeddings encode nuisance attributes more strongly than the disease** — task 1.000,
   sex 0.971, dataset 0.914, language 0.913, age 0.761, **PD last at 0.703** (F-20).

---

# STEP 5 — The head and the splits  ✅ FINAL

**The only part of the system that learns.** `src/pdvoice/models/head.py` ·
`src/pdvoice/data/splits.py`

---

## 5.1 Architecture

```
Linear(2048 → 256) → BatchNorm1d(256) → ReLU → Dropout(0.3) → Linear(256 → 1)
```

Outputs a **logit**, not a probability. The sigmoid is applied at calibration time
(step 6).

| layer | shape | params |
|---|---|---|
| `Linear` weight | (256, 2048) | 524,288 |
| `Linear` bias | (256,) | 256 |
| `BatchNorm` weight + bias | (256,) ×2 | 512 |
| `Linear` weight | (1, 256) | 256 |
| `Linear` bias | (1,) | 1 |
| **total** | | **525,313** |

**525,313 against the encoder's 315,438,720 — 0.167%.** That ratio is the headline
architectural fact: **99.83% of the system never learns anything about Parkinson's.**

### Why deliberately shallow

1. mPower labels are **self-reported** and noisy; a deep head would fit the noise.
2. The head is **identical across all five encoders**, so the encoder comparison in 4.3
   measures the *encoder*, not a per-encoder tuning budget.
3. ~900 patients cannot support more capacity.

---

## 5.2 Training regime

| setting | value |
|---|---|
| loss | `BCEWithLogitsLoss` with **`pos_weight = n_neg / n_pos`** |
| optimiser | Adam, `lr = 1e-3` |
| max epochs | 200 |
| early stopping | on **validation loss**, `patience = 20`, min-delta 1e-5 |
| on stop | **restore the best epoch's weights** (`load_state_dict(best_state)`) |
| dropout | 0.3 |
| seed | 42 — weight init *and* dropout |

### Class weighting is load-bearing

mPower prevalence is **0.187** (897 PD / 3,912 HC). `pos_weight = neg/pos ≈ 4.36`
compensates. Without it the head can reach 81% accuracy by predicting "healthy" for
everyone — which is also why **accuracy is never reported** in this project.

### Seeding is load-bearing

An unseeded head **swung AUROC by up to ~0.04** run-to-run on XLS-R embeddings. That is
larger than several differences the thesis discusses, so the seed is fixed and stated.
This is the same lesson as F-12 (single-seed pilots hide instability).

### ⚠️ Full-batch training — declare this

The loop takes **one optimiser step per epoch on the entire training set**. There are no
mini-batches. So `max_epochs = 200` means **at most 200 gradient steps in total**.

This is a deliberate consequence of training a 0.5M-parameter head on **cached**
embeddings — the whole training set fits in memory and an epoch costs one matrix multiply.
But it is unusual enough that a reviewer will ask, so state it rather than let it be
discovered. Note also that `BatchNorm` over a full batch is normalisation over the entire
training set, not the usual stochastic estimate.

`fit_mlp_head` records `_stopped_early`, `_epochs_run` and `_best_val_loss` on the
returned model. **`_stopped_early = False` means training hit the epoch ceiling with
validation loss still improving — the head is underfitted and every downstream number is
pessimistic.** These diagnostics are computed but **not persisted to any artifact**, so
the condition has never been audited across runs.

---

## 5.3 The splits — subject-disjoint everywhere

**The cardinal rule: no participant appears in more than one partition.**

| purpose | splitter | properties |
|---|---|---|
| Layer-1 cross-validation | `StratifiedGroupKFold`, **5 folds**, shuffled, seed 42 | subject-disjoint **and** label-balanced |
| validation slice | `GroupShuffleSplit`, **`VAL_FRACTION = 0.15`** | carved from the non-test folds, also subject-disjoint |

```
all folds except test  ──GroupShuffleSplit──►  train (85%)  +  val (15%)
test fold                                      touched once
```

Unlabelled rows get `fold = -1` and are excluded from CV entirely.

### Three jobs for the validation slice

1. **Early stopping** — the epoch whose weights get restored.
2. **Fitting the calibrator** (step 6).
3. **Choosing the operating threshold** (step 6).

Not on train — the head has already fitted that data and is overconfident on it. Not on
test — that would be using the answers to set your own dial.

### `assert_disjoint` is a real guard, not decoration

It raises on any subject appearing in two index groups. Leakage of this kind does **not**
announce itself: it inflates scores quietly, which is exactly the defect behind the
published mPower **0.94** (F-26 — that split is at *recording* level, so the same speaker
appears in train and test).

---

## 5.4 Evaluation is always at SUBJECT level

Predictions are averaged per `subject_id` before any metric is computed.

**Why it is not optional:** one mPower speaker contributed **355 recordings** and another
**1**. Recording-level scoring would let a single person count 355 times, and the heavy
contributors are not a random sample of the cohort.

This single choice is the difference between your numbers and the published mPower
0.94–0.96 figures (F-26).

---

## 5.5 Limitations to state

1. **Full-batch training, ≤ 200 gradient steps.** Defensible for a 0.5M head on cached
   features, but unusual — declare it (5.2).
2. **Underfitting has never been audited.** `_stopped_early` is computed and discarded; if
   it is `False` anywhere, those numbers are pessimistic.
3. **The head is not tuned.** Hidden size 256, dropout 0.3 and lr 1e-3 are defaults, not
   searched. Deliberate — a per-encoder search would have broken the fairness of 4.3 —
   but it means no claim of optimality can be made.
4. **Results are single-seed except where stated.** F-7 and F-22 are 5-seed; most other
   numbers are one seed, and F-12 shows single-seed pilots can be badly overconfident.
5. **Accuracy is never reported**, by design — at prevalence 0.187 a constant "healthy"
   predictor scores 81%.

---

# STEP 6 — Calibrate, then LOCK  ✅ FINAL

`src/pdvoice/calibration.py` · `src/pdvoice/eval/metrics.py`

**This is the step that makes every transfer number mean something.** Without the lock,
external results would be reports of how well the model *can be tuned* to a new corpus,
not of how it *transfers*.

---

## 6.1 Why calibration is needed at all

The head emits **one raw number** — a logit, unbounded. Fine for ranking, useless for
telling a clinician "this person is 70% likely to have Parkinson's".

Worse, a model can be **right about the order and wrong about the numbers**: it can place
every patient above every control and still claim everyone is 95% likely to be ill. The
ranking is perfect; the probabilities are nonsense. AUROC cannot detect this. ECE, MCE and
Brier can.

---

## 6.2 The three candidate maps

| name | form | free parameters |
|---|---|---|
| **Temperature** | `sigmoid(logit / T)` | **1** — scale only |
| **Platt** | logistic regression on the logit | **2** — scale *and* shift |
| **Isotonic** | free monotone step function on `sigmoid(logit)` | non-parametric |

`select_best_by_brier` fits **all three on the validation fold** and keeps the one with the
lowest validation **Brier score** — the selection rule specified in the proposal.

### All three are monotone — this matters

A monotone map cannot reorder anything. Therefore **calibration can never change AUROC or
AUPRC.** It changes ECE, MCE, Brier, and every threshold-dependent metric.

> **Say this when asked whether calibration flatters the results:** *"Calibration is a
> monotone transformation, so it cannot change AUROC by construction. It changes only the
> probability values and the operating point."*

**Observed:** isotonic won all five folds at L11 — unanimous, so not a coin flip. Isotonic
is the most flexible of the three, which says the raw logits need genuine reshaping rather
than rescaling.

---

## 6.3 The operating threshold

`youden_threshold` maximises **Youden's J = sensitivity + specificity − 1**, computed on
the **validation fold**, on calibrated probabilities.

**Fitted on validation, never on train, never on test:**

| fold | why not |
|---|---|
| train | the head has already fitted it and is overconfident there |
| test | choosing a threshold on test data is using the answers to set your own dial |

---

## 6.4 THE LOCK

> **Weights, calibrator and threshold are all frozen BEFORE any external dataset is
> opened. Nothing is re-fitted afterwards.**

```
mPower train  ──► head weights
mPower val    ──► early-stopping epoch + calibrator + threshold
              ══════════ LOCK ══════════
mPower test / NeuroVoz / IPVS / MDVR-KCL  ──► scored, never fitted
```

**What this buys, and why it is worth the cost:** if the threshold were re-picked on each
external set, sensitivity and F1 would rise sharply — and the numbers would mean nothing,
because they would describe a model tuned on its own test data. The locked numbers are
lower and are the ones that answer the research question.

> **Write:** *"The classifier weights, the calibration map and the operating threshold
> were all fixed on a held-out mPower validation fold and applied unchanged to every
> external corpus. No component was re-fitted after the lock."*

---

## 6.5 What the lock reveals — calibration breaks before discrimination does

| corpus | ECE |
|---|---|
| mPower (home) | **0.03** |
| external | **0.22 – 0.40** |

**F-10.** The ranking survives the journey; the probabilities do not. This is a specific,
quotable failure mode that AUROC alone would hide entirely, and it is one of the five
evaluation requirements the thesis argues for.

**Consequence for deployment:** a model can transfer usefully as a *ranker* and be
unusable as a *probability estimate*. Any system that acts on a fixed probability
threshold — which is every screening system — inherits that failure.

---

## 6.6 Limitations to state

1. **The chosen calibrator is not persisted.** `select_best_by_brier` returns the name and
   only **one** artifact in `outputs/logs/` records it. Which map was locked is therefore
   not recoverable for most runs — a reproducibility gap, not a validity one.
2. **NPV is absent from `threshold_metrics`** (it returns accuracy, sensitivity,
   specificity, PPV, F1, balanced accuracy and the confusion matrix). This is one of the
   ethics-form commitments. ✅ **It is recoverable, not lost** — `tn` and `fn` are both
   stored, so `NPV = tn / (tn + fn)` can be computed from any saved artifact.
3. **Calibration and decision curves are not reported for the external datasets** —
   another ethics-form commitment.
4. **Youden's J weights sensitivity and specificity equally**, which is not the clinical
   objective for screening. The DCA work (step 7, `p_t = 0.10`) states the real exchange
   rate — 9 false positives per case caught — but the locked threshold itself is still
   Youden's.

---

# STEP 7 — Evaluation: four questions, not one number  ✅ FINAL

`src/pdvoice/eval/` — `discrimination.py`, `metrics.py`, `dca.py`, `report.py`

**This is the RQ3 contribution.** Each of the four questions exists because reporting only
the one above it would have hidden a real failure found in this project.

**Everything is computed at SUBJECT level**, never per recording (5.4).

---

## 7.1 The four questions

| # | question | measures | what it caught |
|---|---|---|---|
| 1 | Can it separate the groups? | **AUROC**, AUPRC, 95% bootstrap CI | — |
| 2 | Is the difference real? | **DeLong**, paired bootstrap | F-27 — every model difference vanishes age-matched |
| 3 | Do the probabilities mean anything? | **ECE**, MCE, Brier | F-10 — ECE 0.03 home, 0.22–0.40 abroad |
| 4 | Would using it help anyone? | **Decision Curve Analysis** | F-23 — +1.3 per 1,000 |

---

## 7.2 Discrimination

| | |
|---|---|
| **AUROC** | primary metric. Probability a random patient outranks a random control. |
| **AUPRC** | the same idea weighted toward the positive class |
| **95% CI** | `bootstrap_ci`, **N_BOOTSTRAP = 1,000**, seed 42 |

### ⚠️ The AUPRC trap — state it explicitly

**AUROC's chance level is always 0.5. AUPRC's chance level is the prevalence.**

| cohort | prevalence = AUPRC chance |
|---|---|
| mPower cap-3 | **0.187** |
| NeuroVoz | 0.491 |
| IPVS | 0.431 |
| MDVR-KCL | 0.432 |

So an AUPRC of 0.525 on IPVS, where prevalence is 0.431, is **barely above chance** — and
the same number at home would be excellent. **AUPRC is not comparable across these corpora
without its prevalence beside it.** Report them as a pair, always.

---

## 7.3 Significance

| test | use | status |
|---|---|---|
| **DeLong** (`delong_roc_test`) | paired AUROC, same subjects | ✅ implemented **and verified** |
| **paired bootstrap** (`paired_bootstrap_auroc`) | paired AUROC, distribution-free | ✅ implemented |
| **McNemar** (`mcnemar_test`) | paired binary **decisions** | ✅ implemented **and verified** |

### DeLong was verified, and that verification is worth reporting

The implementation carried a source comment saying it had never been checked. It was
validated against three cases with known answers:

| case | expected | got |
|---|---|---|
| identical predictors | p = 1.0 | **1.0** |
| strong vs weak | p ≈ 0 | **3.7e-59** |
| two equal-quality predictors | p ≈ large | **0.553** |

The paired bootstrap agreed each time. **Two independent implementations agreeing to
three decimals is the evidence that both are correct** — and it is why the headline
significance results can be trusted.

### ⚠️ Permutation testing does not work here — F-11

Permutation tests are **blind to class-correlated confounds**. Shuffling labels destroys
the age–label association along with everything else, so the null distribution it builds
is not the null you need. This is why the **age-matched arm** exists: it is the control
that a permutation test cannot substitute for.

---

## 7.4 Calibration

| metric | definition |
|---|---|
| **ECE** | mean \|observed − predicted\| over **10 equal-FREQUENCY bins** (`ECE_BINS = 10`) |
| **MCE** | the **worst** single bin, not the average |
| **Brier** | mean squared error of the probability; **lower is better** |

⚠️ **Equal-frequency, not equal-width.** `calibration_error` sorts by predicted probability
and splits into 10 equal-count bins. This matters: with equal-width bins on a skewed
score distribution most bins are nearly empty and ECE becomes unstable. **Say which you
used** — papers routinely do not, and the two are not comparable.

**Brier mixes discrimination and calibration**, so it is a summary rather than a
diagnosis. Report it with ECE, not instead of it.

---

## 7.5 Clinical utility — Decision Curve Analysis

```
net benefit  =  TP/n  −  (FP/n) × ( p_t / (1 − p_t) )
```

| | |
|---|---|
| threshold band | **0.05 – 0.30** (`DCA_THRESHOLDS`), the screening range |
| headline `p_t` | **0.10** |
| comparators | **treat-all** and **treat-none**, both computed |

**Stating `p_t = 0.10` is a clinical commitment, not a hyperparameter.** It says: *we
would accept **9 false positives to catch 1 true case**.* That exchange rate has to be
declared in advance, and it is the thing DCA adds that no other metric on this page
contains.

**The benchmark that matters is not another model — it is "test everybody."** A model
earns its place only by beating treat-all.

**Result: +1.3 additional patients found per 1,000 people screened** (F-23).

> **Write:** *"AUROC 0.63 sounds like a result. +1.3 per 1,000 is what it is worth."*

---

## 7.6 McNemar — the decision-level test

**Implemented 2026-09-27** in `eval/discrimination.py`, after an audit found it asserted in
five documents and present in none. It now exists, is verified, and is wired into the
comparison path.

### What it adds that DeLong does not

**DeLong asks whether one model RANKS better. McNemar asks whether the decisions it
actually makes are better.** Those come apart — a model can rank better and decide no
better — which is exactly the situation this project keeps finding, so neither test
substitutes for the other.

Only the **discordant** pairs carry information. Cases both models get right, and cases
both get wrong, say nothing about which is better:

```
n01 = model 1 wrong, model 2 right
n10 = model 1 right, model 2 wrong
```

Under H₀ a discordant pair is a coin flip, so `n01 ~ Binomial(n01 + n10, 0.5)`.

| n_discordant | method used |
|---|---|
| **< 25** | **exact binomial** |
| ≥ 25 | chi-square with Edwards' continuity correction, `(|n01 − n10| − 1)² / (n01 + n10)` |
| 0 | returns p = 1.0 — identical decisions, and both formulae are 0/0 here |

The small-sample routing matters: with cohorts of 37–50 speakers (F-34) the exact test is
usually the one that fires, and the chi-square approximation would be unreliable there.

### Verified against known answers

Same discipline as the DeLong validation:

| case | expected | got |
|---|---|---|
| identical decisions | p = 1.0 | **1.0** |
| model 2 fixes 60 of model 1's errors | p ≈ 0 | **2.6e-14** |
| equal quality, 40 errors each | p ≈ 1 | **0.911** |
| n_discordant = 5 | routes to exact | **exact binomial** |
| **vs `statsmodels.mcnemar`** | identical | **match to 1e-12** |

### Where it runs

`fusion.py → pairwise_tests`, alongside DeLong and the paired bootstrap. Each model is
binarised at **its own Youden threshold**, computed on its own scores — comparing
decisions across a shared arbitrary cut such as 0.5 would largely measure which model
happens to be better centred, not which one decides better.

### It independently reproduces the central result

| comparison | | DeLong | bootstrap | **McNemar** |
|---|---|---|---|---|
| SSL vs classical | confounded | 0.0006 | 0.000 | **0.0001** |
| SSL vs fusion | confounded | 0.0000 | 0.000 | **0.0000** |
| classical vs fusion | confounded | 0.0000 | 0.000 | **0.0000** |
| SSL vs classical | **age-matched** | 0.7182 | 0.718 | **0.7891** |
| SSL vs fusion | **age-matched** | 0.8762 | 0.880 | **0.9634** |
| classical vs fusion | **age-matched** | 0.6197 | 0.622 | **0.8443** |

**Three independent tests, two of them asking different questions, all agreeing.** With
the confound in, every model difference is real at both the ranking and the decision
level. With it removed, none of them are, by any test.

That is materially stronger than the two-test version, because a sceptic can no longer
argue the result is an artefact of how AUROC is compared.

⚠️ These McNemar values were computed from the **saved per-subject predictions** and added
to `fusion__xls-r-300m__L15.json` without re-running the experiment. The AUROCs and the
DeLong/bootstrap values are unchanged from the original run.

## 7.7 The operating point

| reported | why |
|---|---|
| Youden's J threshold, **locked on validation** | the decision the system would actually make |
| full confusion matrix (tp/fp/fn/tn) | so any derived metric is recoverable |
| sensitivity, specificity, PPV, F1, balanced accuracy | standard |
| **sensitivity @ 90% specificity** | the realistic screening constraint |
| ❌ **accuracy** — computed but never reported | at prevalence 0.187 a constant "healthy" predictor scores **81%** |

**Sensitivity at 90% specificity is the least flattering and most useful number you have:
0.12–0.15.** The model finds 12 to 15 patients per 100 at a usable operating point.

> **Write:** *"AUROC 0.618 sounds mediocre but alive. At a usable operating point the same
> model finds 15% of patients."*

**NPV** is absent from `threshold_metrics` but **recoverable**: `tn` and `fn` are stored,
so `NPV = tn / (tn + fn)` (see 6.6).

---

## 7.8 Limitations to state

1. **External calibration and decision curves are not reported** — an ethics-form
   commitment still open.
2. **NPV is not emitted**, though it is recoverable from stored confusion matrices.
3. **AUPRC is reported without prevalence in some tables** — always pair them (7.2).
4. **Most numbers are single-seed.** F-7 and F-22 are 5-seed; F-12 shows single-seed
   pilots can be badly overconfident.
5. **The locked threshold is Youden's J**, which weights sensitivity and specificity
   equally — not the screening objective that `p_t = 0.10` declares (6.6).
---

# STEP 8 — Transfer  ✅ FINAL

`scripts/layer2_transfer.py` (design A) · `scripts/speech_agematch.py` (design B)

**There are TWO transfer designs in this project, and they give opposite answers.**
Conflating them is the single easiest way to misreport the thesis, so they are separated
here and must be separated in the write-up.

---

## 8.1 The two designs

| | **A — mPower-trained** | **B — speech-trained** |
|---|---|---|
| trained on | mPower **sustained vowel**, self-reported labels | an **external corpus's connected speech**, clinician labels |
| tested on | every external corpus | the other external corpora |
| model | **locked** — no refitting after step 6 | retrained per source, 5 seeds |
| best result | **0.643** | **0.883** |
| verdict | **fails, and often inverts** | **works** |

Design A answers *"does a phone-app vowel model generalise?"* — **no**.
Design B answers *"does connected speech carry transferable PD signal?"* — **yes**.

---

## 8.2 Three shifts, deliberately separable

| shift | comparison | why it is isolable |
|---|---|---|
| **dataset** | mPower → MDVR-KCL | **both English** — so any drop is not language |
| **language** | mPower → NeuroVoz (es), IPVS (it) | same task, different language |
| **task** | vowel-trained → speech, ddk | same language available on both sides |

Having two English corpora is what makes the dataset/language separation possible at all.
**Result: task shift degrades as much as language shift**, so the fragility is *not*
specifically linguistic (RQ1).

---

## 8.3 Design A results — the locked mPower model

mPower internal ceiling: **0.779** (L11) / 0.763 (L15).

| corpus / task | L11 | L15 | best of all runs |
|---|---|---|---|
| neurovoz / vowel | **0.643** | 0.618 | 0.643 |
| neurovoz / ddk | 0.578 | 0.487 | 0.578 |
| mdvr_kcl / speech | 0.469 | 0.601 | 0.643 *(with adaptation)* |
| ipvs / vowel | 0.287 | 0.356 | 0.638 *(L15 full-window)* |
| ipvs / speech | 0.265 | — | 0.327 |
| neurovoz / speech | 0.255 | — | 0.255 |

**Nothing in the entire project exceeds 0.643 under design A.** Against a 0.779 ceiling
that is a transfer drop of 0.14 at best and 0.71 at worst.

### ⚠️ 12 of 30 runs are significantly INVERTED — F-35 🔴

Twelve of the thirty design-A evaluations have their **entire 95% CI below 0.5**. The
model does not merely fail; it ranks patients **beneath** controls, so flipping its sign
would beat it.

The pattern is **not random across tasks**:

| task | behaviour |
|---|---|
| **connected speech** | inverted almost everywhere — ipvs 0.225–0.327, neurovoz 0.189–0.255 |
| **vowel** | at or above chance on NeuroVoz (0.561–0.643); inverted on IPVS |
| **ddk** | at or just below chance (0.487–0.578) |

The head is trained on a sustained /a/; applied to connected speech, its decision axis
aligns with something *anti*-correlated with PD.

**Not explained.** Candidates to separate: cross-task domain shift; the age axis pointing
the other way in clinical cohorts (mPower patients are *older* than controls, clinical
cohorts are closer to balanced); or a recording-condition axis. **A sign flip is not a
fix** — an inverted detector is still reading the wrong variable.

---

## 8.4 Design B results — speech-trained transfer works

Mean over **5 seeds**, layer 15:

| source | within | → ipvs | → mdvr_kcl | → neurovoz |
|---|---|---|---|---|
| **mdvr_kcl** (n=37) | 0.841 | **0.850** | — | **0.813** |
| **neurovoz** (n=111) | 0.943 | 0.665 | **0.819** | — |
| **ipvs** (n=65) | 0.972 | — | 0.698 | 0.695 |

**Every direction transfers at 0.67–0.85**, against within-corpus 0.84–0.97. That is a
drop of roughly 0.15–0.30 — large, but the result stays clearly useful, which is not true
of anything in design A.

**Why speech and not vowel (F-8):** holding /a/ tests only whether the larynx can sustain
a steady tone. Talking tests whether the articulators can move far enough and fast enough.
Parkinson's is a **movement** disorder, so the moving task detects it. Corroborated
independently by VoiceFM (`docs/LITERATURE.md` §4).

### The age control — and it cuts the other way (F-30)

Speech transfer was checked for the same confound that destroyed design A, using a
**three-arm design**: age matching also halves the cohort, so a drop could be the matching
*or* the lost data. The third arm removes the same number of speakers **at random**,
leaving the age gap intact. Only the difference between arms two and three is attributable
to age.

| direction | full | age-matched | size-matched random | **attributable to age** |
|---|---|---|---|---|
| neurovoz → ipvs | 0.665 | 0.557 | 0.703 | **−0.146** |
| neurovoz → mdvr_kcl | 0.819 | 0.777 | 0.883 | **−0.105** |
| ipvs → neurovoz | 0.695 | 0.564 | 0.746 | **−0.182** |
| ipvs → mdvr_kcl | 0.698 | 0.684 | 0.711 | −0.026 |

**Age matching HURTS speech transfer.** That is the opposite of design A, where matching
*helped* transfer (+0.124) — and it is the key evidence that the speech result is **not**
age-inflated. A confounded result gets *better* when you remove the confound from the test
side; this one gets worse, which is what you expect when age-matching removes genuine
disease-correlated variance along with the confound.

⚠️ **The three-arm design is what makes this claim possible.** Without the random-removal
arm the drop would look like a confound being exposed. Report all three arms.

---

## 8.5 Domain adaptation — the field's proposed fix does not work (F-5)

The centroid correction `x̃ = x − μ_source + μ_target`, using **healthy-only** centroids,
is the fix the cross-lingual literature proposes; it reportedly recovers sensitivity
0.35 → 0.93 and drives language-ID to chance (`docs/LITERATURE.md` §5).

Implemented as `--adapt hc-centroid` (mean only) and `--adapt hc-standardize`
(mean + scale):

| corpus | none | hc-centroid | hc-standardize |
|---|---|---|---|
| mdvr_kcl / speech | 0.595 | **0.643** | 0.629 |
| neurovoz / vowel | 0.606 | 0.609 | **0.614** |
| ipvs / vowel | 0.133 | 0.159 | 0.146 |

**Best case +0.048. It never rescues anything, and it leaves IPVS deeply inverted.**
Disagreeing with a published fix is a contribution and should be reported as one.

⚠️ **The decisive test was never run.** These arms are all on the **confounded** cohort.
Whether the centroid trick works on an **age-matched** cohort — which would show the
published fix silently assumes demographically comparable groups — is the one experiment
the original plan called *"highest novelty per hour"*, and it remains undone.

---

## 8.6 Limitations to state

1. **12 of 30 design-A transfers are significantly inverted, and it is unexplained**
   (8.3, F-35 🔴).
2. **Nothing in design A exceeds 0.643** against a 0.779 ceiling.
3. **Domain adaptation was never tested age-matched** (8.5) — the decisive arm is missing.
4. **MDVR-KCL cannot be age-corrected** (0% age coverage) and has n = 37, so both design-B
   rows involving it are weakly supported.
5. **Design B retrains per source**, so it does not inherit design A's lock. Say which
   design each number comes from, every time.
6. **The transfer drop was computed against the wrong ceiling** in earlier runs — fixed
   2026-09-27; drops reported at L15 before that date are overstated by 0.064.

---

# STEP 9 — Controls and probes  ✅ FINAL

**Every control here exists because a specific alternative explanation had to be killed.**
That is the argument of the thesis: not that the model works, but that the failures have
been diagnosed rather than assumed.

Write this as a table of *rival hypotheses and how each was ruled out*, not as a list of
extra experiments.

---

## 9.1 The control table — what each one kills

| control | rival hypothesis it kills | result | finding |
|---|---|---|---|
| **age-only baseline** | *"the model beats trivial demographics"* | **0.922 vs 0.796 — it does not** | F-1 |
| **age matching** | *"the gap is language"* | age → **0.531**, model survives at 0.632 | F-7, F-22 |
| **label-noise arm** | *"self-report is the problem"* | **+0.012 within, +0.002…+0.021 abroad** | F-2 |
| **raw-acoustic check** | *"the confound is a model artefact"* | visible in LTAS, **no model involved** | F-3 |
| **5 classifiers × 5 encoders** | *"a better architecture would fix it"* | all indistinguishable | F-18, F-27 |
| **embedding probes** | *"what does the vector actually encode?"* | PD ranks **last** | F-20 |
| **sex control** | *"age matching handles demographics"* | it does not — see 9.4 | F-21 |
| **vowel-phoneme control** | *"the wrong vowel was used"* | /a/ vs all vowels: **0.554 vs 0.541** | — |
| **IPVS batch audit** | *"the ceiling is a sample-rate artefact"* | rate alone 0.679, but removing it leaves **1.000** | F-17 |
| **NeuroVoz task-code audit** | *"the corpus is clean"* | filenames alone give **0.895** | F-28 |
| **multi-seed (5)** | *"one run is enough"* | SD up to **±0.106** | F-12 |

---

## 9.2 The label-vs-age discriminator — the cleanest argument in the thesis

Two hypotheses, and they **predict opposite things**. That is what makes this decisive
rather than suggestive.

| | within mPower | IPVS | NeuroVoz | MDVR-KCL |
|---|---|---|---|---|
| baseline | 0.801 ± 0.050 | 0.404 ± 0.106 | 0.607 ± 0.023 | 0.431 ± 0.087 |
| **confident labels only** | 0.813 ± 0.014 | 0.406 ± 0.074 | 0.616 ± 0.032 | 0.452 ± 0.102 |
| **age-matched** | **0.617 ± 0.028** | **0.624 ± 0.086** | 0.601 ± 0.054 | — |

**If labels were the problem**, cleaning them should improve everything. It moves nothing:
**+0.012** at home, **+0.002 to +0.021** abroad — all inside one SD.

**If age were the problem**, removing it should make the home score *worse* and the foreign
scores *better*. It does exactly that: **−0.184 at home, +0.220 on IPVS.**

> **Hurting your own headline number while helping the real test is what a hidden bias
> looks like. Label noise cannot produce that pattern.**

⚠️ Note the SDs. The single-seed versions of these numbers are not reportable (F-12).

---

## 9.3 Embedding probes — what the 2048 numbers actually encode

Linear probes on the frozen embeddings, subject-disjoint:

| attribute | metric | value | chance |
|---|---|---|---|
| **task** (vowel vs speech) | AUROC | **1.0000** | 0.50 |
| **sex** | AUROC | **0.9712** | 0.50 |
| corpus identity (3-way) | accuracy | 0.9138 | 0.33 |
| language (3-way) | accuracy | 0.9125 | 0.33 |
| age, young vs old | AUROC | 0.7607 | 0.50 |
| **PD diagnosis** | AUROC | **0.7028** | 0.50 |
| age (continuous) | R² | 0.1768 | 0.00 |
| recording device (4-way) | accuracy | 0.3819 | 0.25 |

**The disease is LAST among the things the representation encodes** — below task, sex,
corpus, language and age. And the 0.703 is measured on the *confounded* cohort, so part of
even that is age.

**One honest counter-example worth keeping:** recording device probes at **0.382 against a
0.25 chance level** — barely above chance. Not everything is confounded, and reporting the
control that came back negative is what makes the positive ones credible.

---

## 9.4 Sex control — age matching does not handle demographics

| cohort | n | female HC | female PD | age alone | sex alone | age+sex | **SSL** |
|---|---|---|---|---|---|---|---|
| raw | 4,772 | 18.5% | 34.5% | **0.9233** | 0.5800 | 0.9243 | 0.7285 |
| **age-matched** *(used)* | 1,120 | 22.3% | 33.4% | 0.5053 | **0.5554** | 0.5361 | 0.5805 |
| age+sex-matched | 1,120 | 25.2% | 25.2% | 0.5045 | 0.5000 | **0.4735** | 0.5944 |

**Age matching drives the age baseline to chance (0.9233 → 0.5053) and leaves sex
standing** at 0.5554, with the female share still 33.4% vs 22.3%.

⚠️ **The age+sex row already exists** — it was computed by `sex_control.py`. Under joint
matching the demographic baseline falls to **0.4735** (below chance) and the SSL model is
**unharmed**: 0.5805 → **0.5944**.

**Decision (3.6, Option A): age-only matching is retained**, and the residual sex confound
is declared. This row is the evidence for the claim that joint matching was *available and
understood*, not overlooked — and that it would not have cost performance.

---

## 9.5 Data-integrity audits

### IPVS sample-rate batch (F-17)

The 44.1 kHz batch is **class-pure — 160 recordings, every one a patient**. Sample rate
alone predicts the label at **0.679**.

**But it is not load-bearing.** Removing every 44.1 kHz patient leaves within-corpus AUROC
at **1.000**, and removing the young healthy controls as well still leaves **1.000**.

> **Disclose it; do not blame the ceiling on it.** The ceiling is real and the batch effect
> is real, and they are separate facts.

### NeuroVoz vowel task codes (F-28)

| what leaks | AUROC |
|---|---|
| task-code **profile** (which codes a speaker has) | **0.8946** [0.825, 0.951] |
| **recording count alone** | 0.7902 |
| recording-level code | 0.6443 |

**93.1% of third repetitions are patients.** HC average 8.64 recordings, PD 10.66 — so how
*many* files a speaker has almost gives away the diagnosis, with no audio at all.

⚠️ This affects **both sides** of the NeuroVoz vowel comparison in `LITERATURE.md` §3 —
neither our 0.610 nor the published 0.646 is confound-free.

### Vowel-phoneme control

*"Did you just pick the wrong vowel?"* — no:

| subset | NeuroVoz AUROC |
|---|---|
| all vowels | 0.5408 |
| /a/ only | 0.5543 |
| all vowels, 1 per speaker | 0.5644 |
| /a/ only, 1 per speaker | 0.5392 |

Phoneme choice moves nothing. The vowel task is weak because it is a **vowel**, not
because it is the wrong one.

---

## 9.6 The objective function (F-23)

`p_t = 0.10` — **9 false positives accepted per true case**, stated in advance.

At that operating point the best age-matched model gives net benefit **0.09678** against
**0.09614** for screening everybody: **+1.3 additional cases per 1,000 screened.**

> *"AUROC 0.63 sounds like a result. +1.3 per 1,000 is what it is worth."*

---

## 9.7 Limitations to state

1. **Probes are linear.** A non-linear probe would likely recover more; these are lower
   bounds on what is encoded, not upper bounds.
2. **Probes ran on mPower controls at layer 15** — not repeated at L11 or across corpora.
3. **The age R² is low (0.177, MAE 9.1 yr)** while binary young-vs-old is 0.761. The
   embedding encodes an age *band*, not an age. Do not overstate it.
4. **Sex remains uncontrolled by choice** (3.6), with the cost quantified in 9.4.
5. **The IPVS ceiling of 1.000 is unexplained** beyond ruling out the batch effect.
6. **Domain adaptation was never tested age-matched** (8.5).

---

# STEP 10 — Results: the six that matter

| # | result | number | finding |
|---|---|---|---|
| 1 | **Age alone beats a 315M-parameter model** | **0.922** vs 0.796 | F-1, F-22 |
| 2 | **Age-matching removes it** — baseline → chance, model survives | 0.531 vs **0.632** | F-7, F-22 |
| 3 | **It is age, not labels** — the two hypotheses predict opposite things | labels +0.011/+0.010; age **−0.185/+0.124** | F-2 |
| 4 | **No model beats any other** once the confound is gone | p 0.001/0.000/0.000 → **0.718/0.876/0.620** | F-27 |
| 5 | **Contamination is a depth gradient** inside one encoder | 0.827 → **0.531**, rank 25/25 | F-32 |
| 6 | **Small cohorts cannot resolve what they claim** | null change swings **±0.29** at n=37–50 | F-34 |

Supporting: F-8 (speech ≫ vowel, transfers 0.70–0.85), F-9 (high within → worse transfer),
F-23 (+1.3 per 1,000), F-26 (published mPower 0.94 comes from identity leakage).

---

# What is NOT done

**I am not starting any of these without asking.** Listed so the gaps are visible.

| # | gap | cost | why it matters |
|---|---|---|---|
| A | **Domain adaptation was never run age-matched** — only on the confounded cohort | ~30–60 min, all caches exist | Your own plan called this the hinge: *"if it works age-matched and fails confounded, you have shown the published fix silently assumes comparable cohorts."* |
| B | **Sex is controlled for nowhere.** `demographic_match_ids` exists, unused | ~1 h | F-21 🔴. Sex probes at **0.971** |
| C | **Notebook is 7 findings behind** (last run 10 Sep; F-29…F-35 have no cells) | ~1 h | It is the viva evidence artifact |
| D | **F-32 magnitudes come from two protocols** | ~30 min | Turns 🟡 → ✅, yields one strong figure |
| E | Three ethics-form commitments: NPV, external calibration/DCA, age-band breakdown | ~1 h | Already signed up to on the form |
| F | **F-35 inversion undiagnosed** | unknown | Declare as open question |
| G | Bridge2AI access | weeks, needs institutional signature | Declare |
| H | `chunk-drift` never evaluated (caches exist) | ~1 h | Future work |

**Nothing here requires starting over.** A–E are all small, and every one reuses caches
that already exist.
