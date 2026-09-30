"""Emit the thesis diagrams as draw.io XML, which Lucidchart imports directly.

Lucidchart: File -> Import -> Diagram, and select the .drawio file. It also accepts
Visio and Gliffy, but draw.io is the only one of the three we can generate faithfully.
The imported shapes are fully editable; treat the output as a starting layout, not a
finished figure.

Four diagrams, matching thesis/diagrams/*.excalidraw:

  fig4_1_pipeline       the proposed framework end to end        (Chapter 4, Figure 4.1)
  fig4_2_split_lock     what is fitted on which split            (Chapter 4)
  fig3_1_preprocessing  the four cleaning stages and their cost  (Chapter 3)
  fig4_3_two_designs    design A against design B                (Chapter 4)

Run:  .venv\\Scripts\\python.exe scripts\\thesis_drawio.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "thesis" / "diagrams"

# draw.io default palette, chosen so the fills stay legible when printed greyscale
FILL = {
    "data":    ("#dae8fc", "#6c8ebf"),   # blue   -- corpora
    "process": ("#d5e8d4", "#82b366"),   # green  -- pre-processing
    "frozen":  ("#fff2cc", "#d6b656"),   # yellow -- frozen encoder
    "learn":   ("#f8cecc", "#b85450"),   # red    -- the parts that learn
    "lock":    ("#e1d5e7", "#9673a6"),   # purple -- lock and evaluation
    "plain":   ("#f5f5f5", "#666666"),   # grey   -- intermediate values
    "warn":    ("#ffe6cc", "#d79b00"),   # orange -- losses and caveats
}


class DrawIO:
    def __init__(self, name: str, w: int = 850, h: int = 1400) -> None:
        self.name, self.w, self.h = name, w, h
        self.cells: list[str] = []
        self.n = 0

    def _id(self) -> str:
        self.n += 1
        return f"n{self.n}"

    def box(self, x, y, w, h, text, kind="plain", dashed=False, font=12, bold=False):
        fill, stroke = FILL[kind]
        cid = self._id()
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontSize={font};align=center;verticalAlign=middle;"
                 f"{'dashed=1;' if dashed else ''}{'fontStyle=1;' if bold else ''}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(text)}" style="{style}" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def note(self, x, y, w, h, text, font=11):
        cid = self._id()
        style = (f"text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;"
                 f"fontSize={font};fontColor=#333333;")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(text)}" style="{style}" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def arrow(self, src, dst, label="", dashed=False):
        cid = self._id()
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;exitX=0.5;exitY=1;"
                 "entryX=0.5;entryY=0;strokeColor=#333333;"
                 f"{'dashed=1;' if dashed else ''}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" edge="1" '
            f'parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    def arrow_h(self, src, dst, label="", dashed=False):
        cid = self._id()
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;exitX=1;exitY=0.5;"
                 "entryX=0;entryY=0.5;strokeColor=#333333;"
                 f"{'dashed=1;' if dashed else ''}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" edge="1" '
            f'parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        xml = (
            '<mxfile host="app.diagrams.net" type="device">\n'
            f'  <diagram name="{escape(self.name)}" id="{escape(self.name)}">\n'
            f'    <mxGraphModel dx="1100" dy="800" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">\n'
            '      <root>\n'
            '        <mxCell id="0"/>\n'
            '        <mxCell id="1" parent="0"/>\n'
            + "\n".join("        " + c for c in self.cells) + "\n"
            '      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')
        (OUT / f"{self.name}.drawio").write_text(xml, encoding="utf-8")
        print(f"  {self.name}.drawio   {self.n} shapes")


# ------------------------------------------------------------------ Figure 4.1
def fig_pipeline() -> None:
    d = DrawIO("fig4_1_pipeline", 900, 1180)
    X, W = 320, 260

    a = d.box(X, 40, W, 60, "Voice recording\nsustained vowel / connected speech", "data")
    b = d.box(X, 150, W, 80,
              "Pre-processing\n16 kHz mono · VAD / trim\nfixed window · −23 dBFS", "process")
    c = d.box(X, 280, W, 90,
              "FROZEN ENCODER\nXLS-R-300M\n315,438,720 weights\nnever updated",
              "frozen", bold=True)
    e = d.box(X, 420, W, 70,
              "Mean + std pooling\nover time frames\n→ 2048 numbers", "plain")
    f = d.box(X, 540, W, 80,
              "MLP head\n2048 → 256 → 1\n525,313 weights  (0.17%)", "learn")
    g = d.box(X, 670, W, 70,
              "Calibrator + threshold\nfitted on VALIDATION\nnever on train or test", "learn")
    h = d.box(X, 790, W, 55, "LOCK\nnothing re-fitted after this line", "lock", bold=True)
    i = d.box(X, 895, W, 70,
              "Evaluation\nscored per SUBJECT\nnot per recording", "lock")

    for u, v in ((a, b), (b, c), (c, e), (e, f), (f, g), (g, h), (h, i)):
        d.arrow(u, v)

    # parallel classical branch
    cl = d.box(20, 280, 250, 90,
               "CLASSICAL BRANCH\n42 Praat descriptors\njitter · shimmer · HNR\nF0 · formants · MFCC",
               "process")
    d.arrow_h(cl, e)

    d.note(20, 400, 250, 150,
           "WHY BOTH BRANCHES\n\n"
           "Jitter is a ~10 ms period\nperturbation. The encoder's\nframe is 20 ms, so it "
           "cannot\nrepresent jitter as a period\nmeasurement — only its\nspectral consequence.\n\n"
           "They measure different physics.")

    d.note(640, 300, 240, 130,
           "WHY FROZEN\n\n"
           "~900 patients — fine-tuning\n315M weights would memorise\nthem.\n\n"
           "Unchanged across corpora, so\na Spanish/English difference is\nthe DATA differing, "
           "not a model\nthat re-tuned itself.")

    d.note(640, 560, 240, 110,
           "TRAINED ON mPOWER ONLY\n\n"
           "12,053 recordings\n4,809 speakers\nsubject-disjoint splits\n\n"
           "External corpora are scored\nby the locked model and never\nused for fitting.")

    d.note(640, 800, 240, 90,
           "THE LOCK IS THE POINT\n\n"
           "Re-picking the threshold per\ncorpus would raise sensitivity\nand mean nothing — "
           "it would\ndescribe a model tuned on its\nown test data.")
    d.save()


# ------------------------------------------------------------------ Figure 4.2
def fig_split_lock() -> None:
    d = DrawIO("fig4_2_split_lock", 880, 760)
    src = d.box(300, 30, 280, 55,
                "mPower cohort\n12,053 recordings / 4,809 speakers", "data")
    tr = d.box(40, 150, 230, 60, "TRAIN\nsubject-disjoint", "process")
    va = d.box(310, 150, 230, 60, "VALIDATION  15%\nsubject-disjoint", "frozen")
    te = d.box(580, 150, 230, 60, "TEST\ntouched once", "plain")
    for t in (tr, va, te):
        d.arrow(src, t)

    w = d.box(40, 260, 230, 50, "head weights", "learn")
    v1 = d.box(310, 260, 230, 45, "early-stopping epoch", "learn")
    v2 = d.box(310, 320, 230, 45, "calibrator (best Brier)", "learn")
    v3 = d.box(310, 380, 230, 45, "operating threshold (Youden)", "learn")
    d.arrow(tr, w)
    d.arrow(va, v1)

    lock = d.box(40, 470, 770, 50,
                 "L O C K   —   weights + calibrator + threshold frozen before any external corpus is opened",
                 "lock", bold=True)
    ext = d.box(180, 590, 480, 55,
                "NeuroVoz · IPVS · MDVR-KCL\nscored by the locked model, never fitted", "plain")
    d.arrow(lock, ext)

    d.note(40, 540, 770, 40,
           "Not fitted on TRAIN: the head has already seen it and is overconfident there.   "
           "Not fitted on TEST: that would be using the answers to set the instrument.")
    d.save()


# ------------------------------------------------------------------ Figure 3.1
def fig_preprocessing() -> None:
    d = DrawIO("fig3_1_preprocessing", 1080, 520)
    stages = [
        ("1  Resample\n16 kHz mono", "loses nothing\nall files already mono"),
        ("2  Remove non-speech\nvowel: trim 30 dB\nspeech: Silero VAD",
         "loses PAUSES,\nhesitation, speech rate\n(0.878 on IPVS alone)"),
        ("3  Fixed window\nvowel: 2 s core\nspeech: 5 s, 50% overlap",
         "loses long-range\nvariation within\nan utterance"),
        ("4  Normalise\n−23 dBFS (EBU R128)",
         "loses absolute loudness\n(unusable on an\nuncalibrated phone)"),
    ]
    prev = None
    for k, (s, loss) in enumerate(stages):
        x = 30 + k * 260
        b = d.box(x, 40, 230, 85, s, "process")
        d.box(x, 175, 230, 85, loss, "warn")
        if prev:
            d.arrow_h(prev, b)
        prev = b

    d.note(30, 300, 1010, 60,
           "STAGE 2 IS NOT CLASS-NEUTRAL.  It retains 59.2% of a patient recording against "
           "76.1% of a control on IPVS,\nand the direction REVERSES on NeuroVoz (88.7% vs "
           "80.9%), whose speech task is isolated words rather than discourse.")
    d.note(30, 380, 1010, 45,
           "STAGE 4 IS PROVABLY INERT.  Attenuating by 60 dB and renormalising changes F0, "
           "HNR, jitter and shimmer by 0.00%.")
    d.note(30, 440, 1010, 45,
           "WINDOWS PER RECORDING DIFFER 25×:  NeuroVoz 1.0 · IPVS 11.2 · MDVR-KCL 31.1. "
           "These are not comparable quantities of material.")
    d.save()


# ------------------------------------------------------------------ Figure 4.3
def fig_two_designs() -> None:
    d = DrawIO("fig4_3_two_designs", 860, 640)
    d.note(30, 20, 360, 30, "DESIGN A — the proposed system", 13)
    a1 = d.box(30, 60, 250, 65,
               "mPower\nsustained vowel\nself-reported labels", "data")
    a2 = d.box(30, 165, 250, 55, "train head → LOCK", "learn")
    a3 = d.box(30, 260, 250, 60, "NeuroVoz · IPVS\nMDVR-KCL", "plain")
    a4 = d.box(30, 360, 250, 60,
               "best 0.643\n12 of 30 runs INVERTED", "warn", bold=True)
    for u, v in ((a1, a2), (a2, a3), (a3, a4)):
        d.arrow(u, v)

    d.note(460, 20, 380, 30, "DESIGN B — an additional observation", 13)
    b1 = d.box(460, 60, 250, 65,
               "one external corpus\nCONNECTED SPEECH\nclinician labels", "data")
    b2 = d.box(460, 165, 250, 55,
               "FRESH head\nsame frozen encoder", "learn")
    b3 = d.box(460, 260, 250, 60, "the other\nexternal corpora", "plain")
    b4 = d.box(460, 360, 250, 60, "0.665 – 0.850\nevery direction", "process", bold=True)
    for u, v in ((b1, b2), (b2, b3), (b3, b4)):
        d.arrow(u, v)

    d.note(30, 460, 810, 120,
           "The encoder is the SAME FROZEN MODEL in both. Design B trains a NEW head from "
           "random initialisation;\nnothing carries over from design A.\n\n"
           "The two differ in elicitation task, label provenance and training corpus, so "
           "design B does not repair design A.\nIt shows that what the model is asked to "
           "LISTEN TO matters more than which model is used.")
    d.save()


def main() -> None:
    fig_pipeline(); fig_split_lock(); fig_preprocessing(); fig_two_designs()
    print(f"\n4 draw.io files written to {OUT}")
    print("Lucidchart: File -> Import -> Diagram -> select the .drawio file")


if __name__ == "__main__":
    main()
