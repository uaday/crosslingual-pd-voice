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

# STEP 2 — Audio cleaning

`src/pdvoice/preprocess.py` → `preprocess_clip()` / `window_clip()`

| # | stage | vowel | speech |
|---|---|---|---|
| 2.1 | decode + resample | → 16 kHz mono float32 | same |
| 2.2 | remove non-speech | `trim_silence(top_db=30)` | **Silero VAD** |
| 2.3 | window | central **2.0 s** core | **5 s** chunks, **50%** overlap |
| 2.4 | normalise | **−23 dBFS** (EBU R128) | same, before chunking |

**Why 16 kHz:** speech energy is below 8 kHz; by Nyquist, 16 kHz captures all of it.

**What each stage discards — measured, not assumed:**

| stage | discards | cost |
|---|---|---|
| VAD | pauses, hesitation, speech rate | **F-31** 🔴 — timing alone is worth IPVS **0.878**, still **0.769** age-matched |
| 5 s chunks | long-range drift across an utterance | **F-34** — the between-chunk variance is 10–13% of within-chunk |
| 2 s vowel core | 80% of every 10 s mPower vowel | **F-34** — costs nothing that transfers |
| normalisation | absolute loudness (hypophonia) | **F-14** — 0.00% descriptor change over 60 dB; unusable on uncalibrated phones anyway |

✅ **The windowing was tested, not assumed** (F-34). Full-audio and chunk-drift variants
were extracted for every corpus. Full audio improves mPower (0.763 → **0.841**) and is
worth **−0.011** across 9 external comparisons. **Decision: keep 2 s / 5 s.**

---

# STEP 3 — Cohort construction

`src/pdvoice/cohort.py` → `load_mpower_cohort()`

```
load manifest → drop unlabelled → QUALITY CONTROL → cap N per subject (seeded) → attach age
```

⚠️ **QC runs before the cap.** Capping first spends a speaker's N slots on files QC then
deletes, and that loss is **not class-neutral** — patients lost ~15% of slots against ~5%
for controls, because unreadable rows were PD-enriched.

| cohort | recordings | speakers | used for |
|---|---|---|---|
| cap-3, confounded | **12,053** | **4,809** | the main training cohort |
| cap-1 | 4,809 | 4,809 | classical baseline, probes |
| **age-matched** (5-yr bins, 1:1) | 1,124 | 1,124 | **the honest cohort** |

**QC rule** (revised after supervisor review): never discard on amplitude. Normalise first,
then judge voicing — drop only if voiced fraction < ~0.3 or HNR ≈ 0. Justification: F-14.

**The confound this exists to handle:** mPower PD ≈ 60 yrs, HC ≈ 33 — a **27.4-year gap**
(F-1). Age matching is `age_match_ids`; `demographic_match_ids` also matches sex but is
**not currently used anywhere** — see F-21 🔴.

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
