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

# STEP 4 — Feature extraction

`src/pdvoice/features/ssl.py` · `scripts/extract_windows.py`

| | |
|---|---|
| encoder | **XLS-R-300M**, `facebook/wav2vec2-xls-r-300m` |
| parameters | **315,438,720 — frozen, never updated** |
| input | raw waveform. **No mel spectrogram.** |
| frame rate | ~20 ms → (T, 1024) hidden states |
| pooling | **mean ⊕ std** → **2048** |
| cache | float16, keyed by cohort + path (identity-guarded) |

**Frame-rate consequence for the thesis:** jitter is cycle-to-cycle variation of a ~10 ms
glottal period — *below* the 20 ms frame. XLS-R **cannot** represent jitter as a period
measurement, only its spectral consequence. Vocal tremor at 4–6 Hz sits above it. This is
why classical Praat features and SSL embeddings measure partly **different physics**.

**Encoder and layer choice — both tested:**

| question | answer | finding |
|---|---|---|
| which encoder? | 5 tested (XLS-R, HuBERT, WavLM, wav2vec2, Whisper), every layer | **F-18** — indistinguishable on the honest signal |
| which layer? | all 25 scored by CV on training data alone | **F-29** — L15 ranks 13/25; L11 best |
| does L11 fix transfer? | re-extracted and re-run end to end | **F-33** — no. 1 win, 2 losses |
| do deep layers encode age? | last layer: 0.827 confounded → **0.531** age-matched, rank 25/25 | **F-32** 🟡 |

**Classical features**, in parallel: `features/classical.py`, **42** descriptors (was 17)
— MFCC mean+std at 25 ms/10 ms, formants + bandwidths, F0 SD, jitter, shimmer, HNR
(**F-24**).

---

# STEP 5 — The head

`src/pdvoice/models/head.py` → `fit_mlp_head()`

```
Linear(2048 → 256) → BatchNorm → ReLU → Dropout → Linear(256 → 1)
```

**525,313 parameters = 0.17%** of the system. This is the only part that learns.

**Splits — subject-disjoint everywhere:**

| purpose | splitter |
|---|---|
| Layer-1 CV (5 folds) | `StratifiedGroupKFold` — subject-disjoint *and* label-balanced |
| train/val for locking | `GroupShuffleSplit`, 15% validation |

Early stopping on validation loss, `patience=20`, then restore the best epoch's weights.

---

# STEP 6 — Calibrate, then LOCK

`src/pdvoice/calibration.py` → `select_best_by_brier()`

The head emits a raw logit, not a probability. Three calibrators are fitted **on the
validation fold** and the best by Brier score is kept: **Temperature · Platt · Isotonic**.
(Isotonic won all 5 folds at L11.)

All three are **monotone**, so calibration can never change AUROC — only ECE, MCE, Brier
and anything threshold-dependent.

Then the operating threshold (**Youden's J**) is chosen on the same validation fold.

> **Weights + calibrator + threshold are frozen before any external dataset is opened.
> Nothing is re-fitted afterwards.** This is the property that makes the transfer numbers
> mean anything.

**F-10:** calibration breaks off-domain far worse than discrimination — ECE 0.03 at home,
**0.22–0.40** abroad. The ranking survives the journey; the probabilities do not.

---

# STEP 7 — Evaluation: four questions, not one number

`src/pdvoice/eval/`

| question | measures | finding |
|---|---|---|
| Can it separate them? | **AUROC** (primary), AUPRC, 95% bootstrap CI | AUPRC must be read against **prevalence**, not 0.5 |
| Is the difference real? | **DeLong**, paired bootstrap, McNemar | F-27 — implementations agree to 3 dp |
| Do probabilities mean anything? | Brier, **ECE**, MCE, reliability curve | F-10 |
| Would using it help? | **Decision Curve Analysis**, `p_t = 0.10` | F-23 — **+1.3 per 1,000** |

**Scored per SUBJECT, never per recording** — one mPower speaker has 355 recordings.
Operating point reported as Youden's J *plus* sensitivity at fixed 90% specificity
(**0.12–0.15** — the least flattering, most useful number).

**F-11:** permutation testing is blind to class-correlated confounds — it will not catch
this, which is why the age-matched arm exists.

---

# STEP 8 — Transfer

`scripts/layer2_transfer.py` — locked model, no retraining, no re-fitting.

Three independent shifts, deliberately separable:

| shift | test |
|---|---|
| **dataset** | mPower → MDVR-KCL (both English) |
| **language** | mPower → NeuroVoz (es), IPVS (it) |
| **task** | vowel-trained → speech, ddk |

**Results:** F-1, F-8, F-22, F-30, F-33, **F-35** 🔴.

⚠️ **F-35 — 9 of 21 external evaluations are significantly INVERTED** (whole 95% CI below
0.5). Every connected-speech cell on IPVS and NeuroVoz, while the same head on vowels
stays at or above chance. **Reproducible, unexplained.**

---

# STEP 9 — Controls and probes

Each one exists because a specific failure mode had to be ruled out.

| control | what it rules out | finding |
|---|---|---|
| **age-only baseline** | that the model beats trivial demographics | **F-1** — it does not: 0.922 vs 0.796 |
| **age matching** (5-yr bins) | the age confound | **F-7, F-22** |
| **label-noise arm** | self-reported labels as the cause | **F-2** — +0.011 / +0.010, nothing |
| **embedding probes** | "what does the vector encode?" | **F-20** — task 1.000, sex 0.971, dataset 0.914, language 0.913, age 0.761, **PD last at 0.703** |
| **5 classifiers × 5 encoders** | that the architecture matters | **F-18, F-27** — indistinguishable |
| **raw-acoustic check** | that the confound is a model artifact | **F-3** — visible in LTAS, no model |
| **multi-seed (5)** | single-seed overconfidence | **F-12, F-22** |
| **task-code leak audit** | NeuroVoz metadata leakage | **F-28** — 0.895 from filenames |
| **domain adaptation** (centroid, standardise) | the field's proposed fix | **F-5** — fails ⚠️ *confounded cohort only* |

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
