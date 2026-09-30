"""Draft the thesis diagrams as Excalidraw files, for redrawing by hand.

These are the four figures that cannot be generated from data: they describe structure,
not numbers. They are written as native Excalidraw clipboard JSON so each can be opened,
inspected and redrawn rather than accepted as-is.

  fig4_1_pipeline      the proposed framework, end to end            (Chapter 4, Fig 4.1)
  fig4_2_split_lock    what is fitted on which split, and the lock   (Chapter 4)
  fig3_1_preprocessing the four cleaning stages and what each drops  (Chapter 3)
  fig4_3_two_designs   design A against design B                     (Chapter 4)

Open at excalidraw.com: File -> Open, or paste the file contents onto the canvas.

Run:  .venv\\Scripts\\python.exe scripts\\thesis_diagrams.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "thesis" / "diagrams"

BLUE, GREEN, YELLOW, PINK, PURPLE, GREY, RED = (
    "#a5d8ff", "#b2f2bb", "#ffec99", "#ffc9c9", "#d0bfff", "#e9ecef", "#ffc9c9")


class Canvas:
    """Minimal Excalidraw document builder: boxes with bound text, and bound arrows."""

    def __init__(self, seed: int = 7) -> None:
        self.rnd = random.Random(seed)
        self.els: list[dict] = []
        self._id = 0
        self._ord = 0
        self.common = dict(angle=0, strokeColor="#1e1e1e", fillStyle="solid",
                           strokeWidth=2, roughness=1, opacity=100, groupIds=[],
                           frameId=None, isDeleted=False, updated=1, link=None,
                           locked=False)

    def _n(self) -> int:
        return self.rnd.randint(1, 2 ** 31 - 1)

    def _nid(self, t: str) -> str:
        self._id += 1
        return f"{t}{self._id}"

    def _idx(self) -> str:
        self._ord += 1
        return f"a{self._ord:04d}"

    def box(self, x, y, w, h, text, fill=GREY, font=14, dashed=False) -> str:
        bid, tid = self._nid("b"), self._nid("t")
        self.els.append(dict(id=bid, type="rectangle", x=x, y=y, width=w, height=h,
                             backgroundColor=fill, roundness={"type": 3}, seed=self._n(),
                             version=1, versionNonce=self._n(), index=self._idx(),
                             strokeStyle="dashed" if dashed else "solid",
                             boundElements=[{"type": "text", "id": tid}], **self.common))
        lines = text.split("\n")
        th = len(lines) * font * 1.25
        self.els.append(dict(id=tid, type="text", x=x + 8, y=y + (h - th) / 2,
                             width=w - 16, height=th, backgroundColor="transparent",
                             roundness=None, seed=self._n(), version=1,
                             versionNonce=self._n(), index=self._idx(),
                             strokeStyle="solid", boundElements=[], text=text,
                             fontSize=font, fontFamily=2, textAlign="center",
                             verticalAlign="middle", containerId=bid,
                             originalText=text, autoResize=False, lineHeight=1.25,
                             **self.common))
        return bid

    def label(self, x, y, text, font=13, w=340) -> None:
        self.els.append(dict(id=self._nid("l"), type="text", x=x, y=y, width=w,
                             height=font * 1.3 * len(text.split("\n")),
                             backgroundColor="transparent", roundness=None,
                             seed=self._n(), version=1, versionNonce=self._n(),
                             index=self._idx(), strokeStyle="solid", boundElements=[],
                             text=text, fontSize=font, fontFamily=2, textAlign="left",
                             verticalAlign="top", containerId=None, originalText=text,
                             autoResize=True, lineHeight=1.25, **self.common))

    def _geom(self, bid):
        e = next(e for e in self.els if e["id"] == bid)
        return e["x"], e["y"], e["width"], e["height"]

    def arrow(self, a, b, side="v", dashed=False) -> None:
        ax, ay, aw, ah = self._geom(a)
        bx, by, bw, bh = self._geom(b)
        if side == "v":
            x0, y0, x1, y1 = ax + aw / 2, ay + ah, bx + bw / 2, by
        else:
            x0, y0, x1, y1 = ax + aw, ay + ah / 2, bx, by + bh / 2
        self.els.append(dict(id=self._nid("ar"), type="arrow", x=x0, y=y0,
                             width=abs(x1 - x0), height=abs(y1 - y0),
                             backgroundColor="transparent", roundness={"type": 2},
                             seed=self._n(), version=1, versionNonce=self._n(),
                             index=self._idx(),
                             strokeStyle="dashed" if dashed else "solid",
                             boundElements=[], points=[[0, 0], [x1 - x0, y1 - y0]],
                             lastCommittedPoint=None,
                             startBinding={"elementId": a, "focus": 0, "gap": 4},
                             endBinding={"elementId": b, "focus": 0, "gap": 4},
                             startArrowhead=None, endArrowhead="arrow", elbowed=False,
                             **self.common))

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        ids = {e["id"] for e in self.els}
        for e in self.els:
            for k in ("startBinding", "endBinding"):
                if e.get(k) and e[k]["elementId"] not in ids:
                    raise AssertionError(f"{name}: dangling {k}")
            if e.get("containerId") and e["containerId"] not in ids:
                raise AssertionError(f"{name}: dangling containerId")
        (OUT / f"{name}.excalidraw").write_text(
            json.dumps({"type": "excalidraw", "version": 2,
                        "source": "scripts/thesis_diagrams.py",
                        "elements": self.els, "appState": {"viewBackgroundColor": "#fff"},
                        "files": {}}, indent=1), encoding="utf-8")
        nb = sum(1 for e in self.els if e["type"] == "rectangle")
        na = sum(1 for e in self.els if e["type"] == "arrow")
        print(f"  {name}.excalidraw   {nb} boxes, {na} arrows")


# ------------------------------------------------------------------ Figure 4.1
def fig_pipeline() -> None:
    c = Canvas()
    c.label(0, -46, "Figure 4.1  The proposed framework. Only the head is trained; "
                    "the encoder is frozen throughout.", 15, 900)
    a = c.box(0, 0, 250, 74, "Voice recording\nsustained vowel / speech", BLUE)
    b = c.box(0, 120, 250, 96, "Pre-processing\n16 kHz mono  -  VAD\nfixed window  -  -23 dBFS",
              GREEN)
    d = c.box(0, 262, 250, 104,
              "Frozen SSL encoder\nXLS-R-300M\n315,438,720 weights\nNEVER UPDATED", YELLOW)
    e = c.box(0, 412, 250, 78, "Mean + std pooling\nover time\n-> 2048 numbers", GREY)
    f = c.box(0, 536, 250, 92, "MLP head\n2048 -> 256 -> 1\n525,313 weights  (0.17%)", PINK)
    g = c.box(0, 674, 250, 78, "Calibrator\nfitted on validation\n+ operating threshold", PINK)
    h = c.box(0, 798, 250, 60, "LOCK\nnothing re-fitted after this", PURPLE, 15)
    i = c.box(0, 904, 250, 78, "Evaluation\nper SUBJECT, not per recording", PURPLE)
    for u, v in ((a, b), (b, d), (d, e), (e, f), (f, g), (g, h), (h, i)):
        c.arrow(u, v)

    c.box(330, 262, 260, 104,
          "Classical branch\n42 Praat descriptors\njitter, shimmer, HNR,\nF0, formants, MFCC",
          GREEN, 13)
    c.label(330, 420,
            "Run in parallel because they\nmeasure different physics:\n"
            "jitter is a ~10 ms period\nperturbation and the encoder\n"
            "frame is 20 ms, so the encoder\ncannot represent it directly.", 12, 280)
    c.label(330, 560, "Trained on mPower only.\nExternal corpora are scored\n"
                      "by the locked model and\nnever used for fitting.", 12, 280)
    c.save("fig4_1_pipeline")


# ------------------------------------------------------------------ Figure 4.2
def fig_split_lock() -> None:
    c = Canvas(11)
    c.label(0, -46, "Figure 4.2  What is fitted on which split. No speaker appears in "
                    "more than one partition.", 15, 900)
    src = c.box(240, 0, 300, 58, "mPower cohort\n12,053 recordings / 4,809 speakers", BLUE)
    tr = c.box(0, 140, 230, 70, "TRAIN\nsubject-disjoint", GREEN)
    va = c.box(270, 140, 230, 70, "VALIDATION\n15%, subject-disjoint", YELLOW)
    te = c.box(540, 140, 230, 70, "TEST\ntouched once", GREY)
    for t in (tr, va, te):
        c.arrow(src, t)

    w = c.box(0, 262, 230, 58, "head weights", PINK)
    v1 = c.box(270, 262, 230, 58, "early-stopping epoch", PINK)
    v2 = c.box(270, 340, 230, 58, "calibrator (best Brier)", PINK)
    v3 = c.box(270, 418, 230, 58, "operating threshold", PINK)
    c.arrow(tr, w); c.arrow(va, v1)

    lock = c.box(0, 520, 770, 56,
                 "LOCK   -   weights + calibrator + threshold frozen before any external "
                 "dataset is opened", PURPLE, 15)
    ext = c.box(120, 640, 530, 64,
                "NeuroVoz  -  IPVS  -  MDVR-KCL\nscored by the locked model, never fitted",
                GREY)
    c.arrow(lock, ext)
    c.label(0, 596, "If the threshold were re-picked on each external corpus, sensitivity "
                    "and F1 would rise and mean nothing.", 12, 780)
    c.save("fig4_2_split_lock")


# ------------------------------------------------------------------ Figure 3.1
def fig_preprocessing() -> None:
    c = Canvas(17)
    c.label(0, -46, "Figure 3.1  The four pre-processing stages, and what each one "
                    "discards.", 15, 900)
    stages = [
        ("1  Resample\n16 kHz mono", "nothing\n(all files already mono)"),
        ("2  Remove non-speech\nvowel: trim  -  speech: VAD",
         "pauses, hesitation,\nspeech rate\n(worth 0.878 alone on IPVS)"),
        ("3  Fixed window\nvowel: 2 s core\nspeech: 5 s, 50% overlap",
         "long-range variation\nwithin an utterance"),
        ("4  Normalise\n-23 dBFS (EBU R128)",
         "absolute loudness\n(unusable on an\nuncalibrated phone)"),
    ]
    prev = None
    for i, (s, loss) in enumerate(stages):
        x = i * 245
        b = c.box(x, 0, 225, 96, s, GREEN, 13)
        c.box(x, 160, 225, 96, loss, RED, 12)
        if prev:
            c.arrow(prev, b, side="h")
        prev = b
    c.label(0, 290, "Stage 2 is not class-neutral: it retains 59.2% of a patient "
                    "recording against 76.1% of a control on IPVS,\nand the direction "
                    "reverses on NeuroVoz, whose speech task is single words.", 12, 940)
    c.label(0, 350, "Stage 4 is provably inert: attenuating by 60 dB and renormalising "
                    "changes F0, HNR, jitter and shimmer by 0.00%.", 12, 940)
    c.save("fig3_1_preprocessing")


# ------------------------------------------------------------------ Figure 4.3
def fig_two_designs() -> None:
    c = Canvas(23)
    c.label(0, -46, "Figure 4.3  The two transfer designs. Design B is a separate "
                    "observation, not a correction of design A.", 15, 940)
    c.label(0, 0, "DESIGN A  -  the proposed system", 15, 400)
    a1 = c.box(0, 34, 250, 72, "mPower\nsustained vowel\nself-reported labels", BLUE)
    a2 = c.box(0, 150, 250, 62, "train head\nthen LOCK", PINK)
    a3 = c.box(0, 256, 250, 72, "NeuroVoz  -  IPVS\nMDVR-KCL", GREY)
    a4 = c.box(0, 372, 250, 62, "best 0.643\n12 of 30 runs INVERTED", RED)
    for u, v in ((a1, a2), (a2, a3), (a3, a4)):
        c.arrow(u, v)

    c.label(430, 0, "DESIGN B  -  an additional observation", 15, 420)
    b1 = c.box(430, 34, 250, 72, "one external corpus\nCONNECTED SPEECH\nclinician labels",
               BLUE)
    b2 = c.box(430, 150, 250, 62, "FRESH head\nsame frozen encoder", PINK)
    b3 = c.box(430, 256, 250, 72, "the other\nexternal corpora", GREY)
    b4 = c.box(430, 372, 250, 62, "0.665 - 0.850\nevery direction", GREEN)
    for u, v in ((b1, b2), (b2, b3), (b3, b4)):
        c.arrow(u, v)

    c.label(0, 470,
            "The encoder is the same frozen model in both. Design B trains a NEW head "
            "from random initialisation;\nnothing carries over from design A. The two "
            "differ in the elicitation task, the label source and the\ntraining corpus, "
            "so design B does not repair design A -- it shows that what the model is "
            "asked to\nlisten to matters more than which model is used.", 12, 940)
    c.save("fig4_3_two_designs")


def main() -> None:
    fig_pipeline(); fig_split_lock(); fig_preprocessing(); fig_two_designs()
    print(f"\n4 diagrams written to {OUT}")
    print("Open at excalidraw.com (File -> Open) to redraw.")


if __name__ == "__main__":
    main()
