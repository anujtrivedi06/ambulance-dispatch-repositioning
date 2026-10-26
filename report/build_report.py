"""Build the two-page, two-column project report as a PDF.

    python report/build_report.py

LaTeX is not installed in this environment, so the PDF is generated directly with
reportlab. `report/report.tex` carries the same content for anyone who would rather
compile it on Overleaf; this script is what produces the submittable file.

Layout: page one has a full-width header (title, team, repository link) above two columns;
page two is two columns throughout. Content lives in `SECTIONS` so the prose stays in one
place and the layout code stays out of its way.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"

TITLE = "Reinforcement Learning for Ambulance Dispatch and Repositioning"
TEAM = "Anuj Trivedi (B23CH1006)"
REPO = "https://github.com/anujtrivedi06/ambulance-dispatch-repositioning"


def register_fonts() -> tuple[str, str, str]:
    """Register a font with Greek coverage, falling back to Helvetica.

    The report states the reward and discount in symbols, and the built-in Type 1 fonts
    have no Greek glyphs -- every sigma, gamma and tau renders as a blank, which turns a
    formula into a gap. DejaVu ships with matplotlib, so this needs no new dependency.
    """
    try:
        from matplotlib import get_data_path
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        ttf = Path(get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", ttf / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", ttf / "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", ttf / "DejaVuSans-Oblique.ttf"))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily

        registerFontFamily(
            "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique"
        )
        return "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  (falling back to Helvetica: {exc})")
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def styles() -> dict:
    base = getSampleStyleSheet()
    body_font, bold_font, _ = register_fonts()
    for style in base.byName.values():
        if getattr(style, "fontName", "").startswith("Helvetica"):
            style.fontName = bold_font if "Bold" in style.fontName else body_font
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=15, leading=18, spaceAfter=4
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontSize=8.5, leading=11,
            alignment=TA_CENTER, spaceAfter=2,
        ),
        "h": ParagraphStyle(
            "h", parent=base["Heading2"], fontSize=9.8, leading=12,
            spaceBefore=7, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=8.1, leading=10.1,
            alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontSize=7.1, leading=8.6,
            alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontSize=7.0, leading=8.4
        ),
    }


def results_table(st: dict) -> Table:
    rows = [
        ["Policy", "Cov.", "Mean", "Worst zone"],
        ["Random repositioning", "0.357", "10.09", "0.282"],
        ["Nearest-idle, static base", "0.765", "5.82", "0.727"],
        ["MEXCLP stations", "0.860", "5.38", "0.798"],
        ["DMEXCLP (strong baseline)", "0.872", "5.26", "0.822"],
        ["Linear ADP, binary reward", "0.809", "5.78", "0.573"],
        ["Deep NFQ, binary reward", "0.724", "6.60", "0.384"],
        ["<b>RL agent (NFQ, shaped)</b>", "<b>0.874</b>", "<b>5.24</b>", "0.818"],
    ]
    table = Table(
        [[Paragraph(c, st["cell"]) for c in row] for row in rows],
        colWidths=[38 * mm, 11 * mm, 11 * mm, 16 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
                ("LINEABOVE", (0, 4), (-1, 4), 0.25, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. Problem and MDP Formulation",
        [
            "Emergency calls arrive at random locations and times across a city road "
            "network. A fleet of K ambulances must answer them, and an agent makes two "
            "coupled decisions: <b>dispatch</b> (which unit to send to an incoming call) "
            "and <b>repositioning</b> (where each idle unit should wait). The two "
            "conflict. Sending the nearest unit is optimal for the call in hand but can "
            "strip cover from a dense district, costing the next call more than it saved "
            "this one.",
            "Decisions occur only at irregular event epochs &mdash; a call arriving, or a "
            "unit becoming free &mdash; so the problem is a <b>semi-Markov decision "
            "process</b> rather than a fixed-step MDP. Transitions carry an elapsed time "
            "&Delta;t, and discounting must respect it.",
        ],
    ),
    (
        "2. State Space (S)",
        [
            "The underlying state is <i>s = (t, {(status<sub>k</sub>, loc<sub>k</sub>, "
            "eta<sub>k</sub>)}<sub>k=1..K</sub>, queue)</i>: the clock, every unit's "
            "status, position and arrival estimate, and the calls still waiting. With "
            "|V| = 64 nodes, six unit statuses and K = 6 ambulances, that is on the order "
            "of (64 &times; 6)<sup>6</sup> &asymp; 10<sup>15</sup> configurations before "
            "the continuous clock is counted. Exact dynamic programming is impossible and "
            "function approximation is mandatory.",
            "The agent therefore observes an <b>11-dimensional post-decision feature "
            "vector</b>: available-unit fraction; time of day as (sin, cos); expected "
            "covered demand; worst-zone coverage; demand-weighted distance to the nearest "
            "free unit; four per-zone coverage values; and a bias term. The post-decision "
            "state is a deterministic function of (state, action), so the agent can rank "
            "every candidate action by one forward pass each, with no sampling.",
            "The feature set was cut from twelve to these after diagnosis on collected "
            "data. Three features were exactly redundant (the design matrix came back "
            "rank 9 of 12) and the four per-zone coverages were collinear at 0.97 and "
            "1.00, leaving their weights unidentified. Ridge regression hid this: value "
            "<i>prediction</i> stayed accurate because the collinear directions cancel "
            "in-distribution, while the greedy policy degraded, because an action moves "
            "along exactly the direction the data could not identify.",
        ],
    ),
    (
        "3. Action Space (A)",
        [
            "The action space is <b>discrete</b>. At a repositioning epoch the agent "
            "chooses one of M stations, |A| = M (M = 4 in the main configuration, swept "
            "to 8). At a dispatch epoch it chooses an available unit or defers the call "
            "to the queue. The learned policy acts on the repositioning choice, which is "
            "where the combinatorial difficulty lies; dispatch follows the "
            "nearest-available rule, held fixed across every method so that measured "
            "differences are attributable to repositioning alone.",
        ],
    ),
    (
        "4. Reward Function (R)",
        [
            "Reward accrues between decision epochs from the calls <i>reached</i> in that "
            "interval, which is when an outcome is determined:",
            "<para alignment='center'><i>r<sub>k</sub> = &Sigma;<sub>i &isin; "
            "reached[t<sub>k</sub>, t<sub>k+1</sub>)</sub> &rho;(i)</i></para>",
            "with two incentive designs compared. The <b>binary</b> design "
            "<i>&rho;(i) = 1{T<sub>i</sub> &le; &tau;}</i>, &tau; = 8 min, is the "
            "operational EMS standard. The <b>shaped</b> design <i>&rho;(i) = "
            "&minus;min(T<sub>i</sub>, C)/10</i>, C = 40 min, is continuous in response "
            "time.",
            "Because epochs are irregularly spaced, discounting is continuous-time: "
            "<i>&gamma;<sub>k</sub> = exp(&minus;&beta;&Delta;t)</i> with "
            "&beta; = 1/240 per minute. A flat per-decision discount would value a "
            "decision followed by a quiet night identically to one followed by a busy "
            "minute.",
            "<b>This choice dominated every other design decision in the project.</b> The "
            "binary indicator cannot distinguish a 3-minute response from a 7-minute one, "
            "so it starves learning of gradient over most of the space it cares about. "
            "Switching to the shaped reward is worth +0.052 coverage for the linear agent "
            "and +0.148 for the neural one &mdash; larger than the difference between "
            "linear and neural function classes.",
        ],
    ),
    (
        "5. Environment and Simulation Dynamics",
        [
            "The environment is a purpose-built discrete-event simulator (no third-party "
            "RL environment was used). Calls follow a non-homogeneous Poisson process "
            "sampled by thinning against a 24-hour demand profile, with locations drawn "
            "from a mixture of uniform background and Gaussian hotspots. Travel follows "
            "shortest paths over a weighted road graph with multiplicative noise; "
            "on-scene durations are lognormal; 75% of calls require transport to hospital "
            "plus a 15-minute handover. The main configuration is an 8&times;8 grid city, "
            "4 stations, 6 ambulances and 4 calls/hour, which holds fleet utilisation near "
            "65% &mdash; loaded enough that poor repositioning costs response time, slack "
            "enough that good repositioning has somewhere to put units.",
            "Every policy is evaluated on <b>identical seeded call streams</b>, so "
            "comparisons are paired and the arrival process cancels rather than drowning "
            "the policy effect. All reported differences use a paired t-test <i>and</i> a "
            "Wilcoxon signed-rank test over 30 replications of 7 simulated days, and are "
            "called significant only when both agree.",
        ],
    ),
    (
        "6. Algorithm",
        [
            "The agent is <b>neural fitted value iteration</b> over post-decision states "
            "&mdash; the DQN family, with the max-over-actions taken through the "
            "post-decision representation. Because the post-decision state is "
            "deterministic given (s, a), <i>max<sub>a</sub> Q(s,a) = max<sub>a</sub> "
            "V(s<sup>a</sup>)</i>, so a single value head suffices and no per-station "
            "output is baked in &mdash; the same agent transfers to a city with a "
            "different number of stations.",
            "Training alternates &epsilon;-greedy data collection under the current "
            "policy with regression onto TD targets <i>r + &gamma; max<sub>a'</sub> "
            "V<sub>target</sub>(s'<sup>a'</sup>)</i>, computed from a target network "
            "frozen once per round. The network is a 2&times;64 MLP (4,993 parameters) "
            "trained with Adam and weight decay. Collection stores every candidate action "
            "considered, not merely the one taken, since the target needs a max over "
            "actions the behaviour policy declined.",
            "Two further value-based RL methods were implemented for comparison: "
            "<b>LSTD</b> and <b>LSPI</b> over the same features. A cross-entropy policy "
            "search over the identical linear class was added as a control: it shares the "
            "hypothesis class exactly, so any gap between it and LSTD is attributable to "
            "the training objective rather than to the representation.",
        ],
    ),
]

SECTIONS_B: list[tuple[str, list[str]]] = [
    (
        "7. Baseline Comparisons",
        [
            "Four baselines of increasing strength: random repositioning (a floor); "
            "nearest-idle dispatch with static home stations, which is what most real "
            "services do; stations optimised offline by p-median and MEXCLP; and "
            "<b>DMEXCLP</b>, the dynamic redeployment heuristic from the EMS literature, "
            "which sends a freed unit to whichever station most increases expected "
            "covered demand. DMEXCLP is the hard target: greedy action selection on the "
            "coverage feature is algebraically identical to it, so the RL agent's "
            "hypothesis class provably <i>contains</i> the heuristic it is measured "
            "against. Losing to it is therefore a statement about the training signal, "
            "not about the features.",
        ],
    ),
    (
        "8. Results",
        [
            "The agent is trained for 8 rounds; the checkpoint is chosen on a validation "
            "block of episodes disjoint from both training and test seeds, and all figures "
            "below are on the untouched test block (30 paired replications of 7 days).",
            "The trained agent with the shaped reward <b>reaches parity with DMEXCLP</b>: "
            "coverage 0.874 against 0.872, a difference of +0.002 that is not significant "
            "(p = 0.32), with mean response, p90 and worst-zone coverage likewise "
            "indistinguishable. Against the untrained floor it is +0.052 coverage "
            "(p = 2&times;10<sup>&minus;17</sup>), so the learning is doing real work &mdash; "
            "it simply lands where a strong domain heuristic already is. With the binary "
            "reward the same architecture collapses to 0.724, and worst-zone coverage falls "
            "to 0.384: maximising a count of indicators is maximised by serving dense zones "
            "and abandoning sparse ones, an equity failure the pooled metric hides entirely.",
            "The learning curve is flat. Validation coverage oscillates within a band of "
            "0.019 across eight rounds with no upward trend, so additional training did "
            "not make the policy better &mdash; the agent reaches heuristic parity early "
            "and stays there. This is what the project's wider results predict: "
            "value-prediction error is dominated by exogenous call variation, so the "
            "directions an action can move the state contribute almost nothing to the "
            "objective being minimised, and the fit has little incentive to rank actions "
            "correctly. A cross-entropy control confirms the reading &mdash; optimising "
            "policy quality directly over the <i>same</i> linear class closes a gap that "
            "fitting the value function does not.",
            "<b>Generalisation is partial, and this is the agent's clearest weakness.</b> "
            "On a demand profile with relocated hotspots, never seen in training, the agent "
            "holds its absolute performance (0.873) and stays +0.063 above the untrained "
            "floor, so it has not merely memorised one demand map. But the heuristic "
            "<i>improves</i> on that easier profile to 0.887, and the agent does not follow "
            "it: parity becomes a significant deficit of &minus;0.014 coverage "
            "(p = 4&times;10<sup>&minus;7</sup>) and &minus;0.033 worst-zone coverage. The "
            "learned value function is tuned to the demand distribution it saw; the "
            "heuristic recomputes coverage from the live demand model and adapts for free.",
            "<b>How much was ever available.</b> Choosing well is worth +0.091 coverage "
            "over random parking, but the best and second-best stations differ by only "
            "about 10% of a single marginal gain. Sweeping load (4&ndash;6 calls/h) and "
            "station count (4&ndash;8) leaves this unchanged, so the ceiling is a property "
            "of the problem, not of the configuration tested.",
        ],
    ),
    (
        "9. Conclusions",
        [
            "Four findings, each measured rather than asserted. <b>(i)</b> Reward design "
            "dominated model class: binary versus shaped reward moved coverage by 0.148, "
            "more than any architectural choice tested. <b>(ii)</b> Value-prediction "
            "accuracy is not policy quality; a control sharing the hypothesis class "
            "exactly, but optimising the policy directly, performed better than the fitted "
            "value function. <b>(iii)</b> The learned policy reaches parity with a strong "
            "domain heuristic on the distribution it trained on, but gives that parity up "
            "under demand shift, where the heuristic's live recomputation of coverage "
            "adapts for free. <b>(iv)</b> Reporting that honestly &mdash; with confidence "
            "intervals, disjoint validation and test seeds, and two agreeing significance "
            "tests &mdash; is more useful than a tuned victory.",
            "The repository contains the full environment, all agents, the trained "
            "network, 70 tests, and scripts reproducing every figure and table here.",
        ],
    ),
]


def build(out_path: Path) -> None:
    st = styles()
    page_w, page_h = A4
    margin = 14 * mm
    gutter = 6 * mm
    col_w = (page_w - 2 * margin - gutter) / 2
    header_h = 32 * mm  # four lines plus padding; too short and the last line leaks
                        # into the left column before the FrameBreak can fire
    body_h = page_h - 2 * margin - header_h

    header = Frame(margin, page_h - margin - header_h, page_w - 2 * margin, header_h, id="hdr")
    p1_left = Frame(margin, margin, col_w, body_h, id="p1l")
    p1_right = Frame(margin + col_w + gutter, margin, col_w, body_h, id="p1r")
    full_h = page_h - 2 * margin
    p2_left = Frame(margin, margin, col_w, full_h, id="p2l")
    p2_right = Frame(margin + col_w + gutter, margin, col_w, full_h, id="p2r")

    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
        title=TITLE, author=TEAM,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="first", frames=[header, p1_left, p1_right]),
            PageTemplate(id="rest", frames=[p2_left, p2_right]),
        ]
    )

    # Without this, reportlab keeps reusing the first template and page two gets the
    # full-width header frame again, which breaks the two-column flow.
    from reportlab.platypus import NextPageTemplate

    story = [
        NextPageTemplate("rest"),
        Paragraph(TITLE, st["title"]),
        Paragraph(f"<b>{TEAM}</b>", st["meta"]),
        Paragraph(
            f"Indian Institute of Technology Jodhpur &mdash; AI Mini Project, "
            f"Assignment 2", st["meta"],
        ),
        Paragraph(f'Repository: <link href="{REPO}"><font color="blue">{REPO}</font></link>',
                  st["meta"]),
        FrameBreak(),
    ]

    for heading, paragraphs in SECTIONS:
        story.append(Paragraph(heading, st["h"]))
        for text in paragraphs:
            story.append(Paragraph(text, st["body"]))

    story.append(Paragraph("Table 1. Headline results (30 paired reps &times; 7 days).",
                           st["caption"]))
    story.append(results_table(st))
    story.append(Spacer(1, 4))

    for heading, paragraphs in SECTIONS_B:
        story.append(Paragraph(heading, st["h"]))
        for text in paragraphs:
            story.append(Paragraph(text, st["body"]))

    for name, caption in (
        ("learning_curve.png",
         "Fig. 1. Validation coverage over training rounds, against the heuristic "
         "baseline on the same episodes. Eight rounds produce no upward trend."),
        ("phase4_reward_compact.png",
         "Fig. 2. Reward design dominates model class; the binary reward also "
         "collapses worst-zone coverage."),
    ):
        path = FIGURES / name
        if not path.exists():
            print(f"  (missing figure {name}, skipped)")
            continue
        # Size at construction. Assigning drawWidth afterwards leaves the natural pixel
        # size in play during wrap, and a 2400px-wide figure then bleeds across the page.
        from reportlab.lib.utils import ImageReader

        native_w, native_h = ImageReader(str(path)).getSize()
        img = Image(str(path), width=col_w, height=col_w * native_h / native_w)
        story.append(img)
        story.append(Paragraph(caption, st["caption"]))

    doc.build(story)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build(ROOT / "report" / "AI_Assignment2_Report.pdf")
