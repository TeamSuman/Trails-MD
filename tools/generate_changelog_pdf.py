"""Generate a structured PDF changelog for AutoSampler."""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AutoSampler_Changelog.pdf")

NAVY = colors.HexColor("#1A237E")
INDIGO = colors.HexColor("#3949AB")
GREEN = colors.HexColor("#2E7D32")
LGREY = colors.HexColor("#ECEFF1")
DGREY = colors.HexColor("#37474F")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=16,
                    spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=INDIGO, fontSize=12.5,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.7, leading=14,
                      alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.6, textColor=DGREY)
TITLE = ParagraphStyle("Title", parent=ss["Title"], textColor=NAVY, fontSize=26,
                       leading=30, spaceAfter=6)
SUB = ParagraphStyle("Sub", parent=ss["Title"], textColor=INDIGO, fontSize=13,
                     leading=17, spaceAfter=4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.8, leading=11.5, spaceAfter=0)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")

story = []


def bullets(items, style=BODY):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=10, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=12, spaceBefore=1, spaceAfter=6,
    )


def rule():
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=0.6, color=INDIGO))
    story.append(Spacer(1, 5))


# ---------------- Title page ----------------
story.append(Spacer(1, 40))
story.append(Paragraph("AutoSampler", TITLE))
story.append(Paragraph("Development Changelog &amp; Feature Summary", SUB))
rule()
story.append(Paragraph(
    "From an MD coverage sampler to an autonomous, "
    "<b>MSM&#8209;convergence&#8209;driven</b> adaptive&#8209;sampling framework.", BODY))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "This document summarises the new development cycle on the "
    "<b>devel</b> branch and highlights every new feature and improvement over "
    "the original v2.0.0 baseline. All new behaviour is <b>opt&#8209;in</b>; existing "
    "input files keep working unchanged.", BODY))
story.append(Spacer(1, 10))

hi = Table([
    [Paragraph("Baseline", CELLH), Paragraph("New (devel)", CELLH)],
    [Paragraph("v2.0.0 — coverage-driven adaptive sampling", CELL),
     Paragraph("MSM-convergence-driven, HPC-scalable, VAMP-2-optimised", CELL)],
    [Paragraph("Stops on bin-occupancy saturation", CELL),
     Paragraph("Stops on real MSM convergence (timescales / VAMP-2 / error)", CELL)],
    [Paragraph("Fixed CV / TICA / TVAE / PCA / deep-TICA", CELL),
     Paragraph("+ VAMPNet, SPIB, VAMP-2 feature selection &amp; optimisation", CELL)],
    [Paragraph("Local multiprocessing only", CELL),
     Paragraph("Local + SLURM + PBS array jobs, fault-tolerant", CELL)],
    [Paragraph("No tests / CI / docs", CELL),
     Paragraph("96 tests, CI, full docs site, notebook, PDF", CELL)],
], colWidths=[78*mm, 92*mm])
hi.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(hi)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "Status: 96 automated tests passing; continuous integration green on "
    "Python 3.10 &amp; 3.11; ruff-clean. Pull request #1 (devel &#8594; main) open.", SMALL))
story.append(PageBreak())

# ---------------- Headline new features ----------------
story.append(Paragraph("1. Headline new features", H1))
rule()
story.append(Paragraph("Markov State Model (MSM) convergence engine", H2))
story.append(Paragraph(
    "The central gap in v2.0.0 — there was no MSM at all, and \"convergence\" "
    "meant bin-occupancy saturation. The new <b>autosampler/msm/</b> subsystem "
    "(built on deeptime) builds an MSM every iteration and stops sampling when it "
    "is genuinely converged.", BODY))
story.append(bullets([
    "<b>MSMEstimator</b>: clustering (k-means / regular-space) &#8594; transition "
    "counts &#8594; MLE or Bayesian MSM &#8594; implied timescales, VAMP-2 score, "
    "PCCA+ metastable states, stationary distribution.",
    "<b>ConvergenceMonitor</b>: composable, pluggable criteria — implied-timescale "
    "stability, VAMP-2 plateau, stationary-distribution drift, Bayesian "
    "statistical-error thresholds, and a <b>flux-weighted transition-matrix</b> "
    "criterion (analytic Dirichlet <font face='Courier'>T<sub>ij</sub></font> "
    "uncertainty) — combined with all / any + patience to require both kinetic "
    "resolution and statistical convergence of the transition matrix.",
    "<b>MSMSpawner</b> (<font face='Courier'>spawn_scheme: msm</font>): "
    "<b>uncertainty &#215; leverage &#215; flux</b> microstate seeding "
    "(<font face='Courier'>&#960;<sub>i</sub>&#183;|&#968;<sub>i</sub>|&#183;"
    "&#963;<sub>out,i</sub> + &#945;/&#8730;c<sub>i</sub></font>) that throws runs "
    "at the transitions whose in/out rates are uncertain and important; "
    "least-counts fallback before the first MSM, with stable clustering for "
    "comparable microstate IDs across iterations.",
]))

story.append(Paragraph("Cutting-edge &amp; optimised collective variables", H2))
story.append(bullets([
    "New deep CV methods <b>VAMPNet</b> and <b>SPIB</b> (State Predictive "
    "Information Bottleneck) added to a single CV registry beside TICA, TVAE, "
    "PCA, deep-TICA, and deep-LDA.",
    "<b>VAMP-2 feature selection &amp; optimisation</b>: automatically select and "
    "adaptively update the input features (and even the feature <i>type</i>) that "
    "best resolve the slow dynamics, via a greedy VAMP-2 optimisation protocol.",
    "<b>VAMP-2-driven adaptive retraining</b>: retrain the CV only when its score "
    "on fresh data degrades, instead of on a blind fixed schedule.",
]))

story.append(Paragraph("Landscape-adaptive binning", H2))
story.append(Paragraph(
    "The density / weighted-ensemble spawners stratify the CV space into bins. A "
    "uniform grid wastes replicas in flat basins and, worse, lets walkers slide "
    "back across a wide bin at a barrier before the lag time elapses — stalling "
    "the flux. The new <b>autosampler/binning/adaptive.py</b> makes bins "
    "landscape-adaptive, recomputed every iteration (selected by "
    "<font face='Courier'>binning.scheme</font>; opt-in, default "
    "<font face='Courier'>uniform</font>).", BODY))
story.append(bullets([
    "<b>gradient</b>: equi-resistance edges (<font face='Courier'>&#8747; exp(&#946;F) "
    "&#8733; &#8747; 1/P</font>) place boundaries where the sampled density is low "
    "— fine across barriers, coarse in basins.",
    "<b>mab</b>: Minimal-Adaptive-Binning-style uniform bins between the occupied "
    "extremes plus narrow foothold bins at the moving fronts.",
    "<b>eigenvector</b>: bin uniformly along the leading (slowest) CV coordinate "
    "only — a committor proxy that is automatically fine at the barrier and folds "
    "many CVs into one coordinate.",
]))

story.append(Paragraph("Scalability: workstation and HPC", H2))
story.append(bullets([
    "Pluggable execution backends: <b>local</b> (multi-GPU workstation), "
    "<b>SLURM</b>, and <b>PBS/Torque</b> — walkers dispatched as scheduler array "
    "jobs, one per iteration.",
    "Fault tolerant: completion is driven by filesystem result markers and failed "
    "walkers are automatically resubmitted.",
    "Switching from a laptop to a CPU-only HPC cluster is a one-line config change.",
]))

story.append(Paragraph("Analysis, tooling, and end-user experience", H2))
story.append(bullets([
    "<b>Weighted-ensemble</b> resampling: a correct, weight-conserving split/merge "
    "implementation (<font face='Courier'>spawn_scheme: we</font>), replacing a "
    "non-functional placeholder.",
    "<b>MSM analysis &amp; plotting</b>: implied timescales, VAMP-2 / timescale "
    "convergence, free-energy surfaces, metastable free energies, and MSM network "
    "diagrams, via the <font face='Courier'>autosampler-analyze</font> CLI.",
    "<b>One input file for everything</b>: <font face='Courier'>autosampler-init</font> "
    "writes a fully-annotated YAML exposing every method, feature, and "
    "hyperparameter; documented for end users.",
    "Full documentation site, an executed Jupyter notebook tutorial with rendered "
    "plots, and example run scripts for local / SLURM / PBS.",
]))
story.append(PageBreak())

# ---------------- Detailed sections by area ----------------
story.append(Paragraph("2. Detailed changes by area", H1))
rule()

sections = [
    ("Sampling &amp; convergence", [
        "New MSM subsystem (estimator, diagnostics, convergence monitor).",
        "Flux-weighted transition-matrix convergence criterion (analytic Dirichlet "
        "error on T_ij) that augments the spectral criteria under mode: all.",
        "Uncertainty &#215; leverage &#215; flux MSM spawner (least-counts "
        "fallback) and weighted-ensemble spawner.",
        "Landscape-adaptive binning (gradient / mab / eigenvector) for the density "
        "and weighted-ensemble spawners; opt-in, recomputed each iteration.",
        "Convergence now based on MSM kinetics, not just spatial coverage "
        "(legacy occupancy criterion retained as one selectable option).",
    ]),
    ("Collective variables &amp; features", [
        "VAMPNet and SPIB deep CVs; unified CV method registry with availability "
        "checks and actionable install hints.",
        "VAMP-2 feature scoring, candidate ranking, and greedy column/feature-type "
        "optimisation, with adaptive updates during the run.",
        "Adaptive CV retraining policy driven by the VAMP-2 score.",
    ]),
    ("Execution &amp; scalability", [
        "ExecutionBackend abstraction with local / SLURM / PBS implementations.",
        "Per-iteration array-job submission, polling, result collection, and "
        "automatic resubmission of failed walkers.",
        "Configurable scheduler resources (partition/queue, walltime, CPUs/GPUs "
        "per task, memory, module loads).",
    ]),
    ("Robustness, reproducibility &amp; engineering", [
        "Pydantic v2 configuration with strict validation of every option.",
        "Checkpoint format versioning; backward-compatible loading of old "
        "checkpoints; resume restores MSM, feature-selection, and retraining state.",
        "MD subprocess timeouts; trajectory-file validation; portable temp paths; "
        "narrower exception handling; removed dead code.",
        "Deterministic seeding across NumPy, PyTorch, and Lightning.",
        "Test suite (96 tests), GitHub Actions CI, ruff/black/isort, pre-commit, "
        "and contribution guidelines.",
    ]),
    ("Analysis &amp; usability", [
        "autosampler-analyze: one-command multi-panel convergence report.",
        "Matplotlib-free analysis data utilities plus plotting helpers.",
        "autosampler-init starter input file; full MkDocs documentation; "
        "rendered Jupyter notebook tutorial; local/SLURM/PBS example scripts.",
    ]),
]
for title, items in sections:
    story.append(Paragraph(title, H2))
    story.append(bullets(items))

story.append(PageBreak())

# ---------------- Capability comparison table ----------------
story.append(Paragraph("3. Capability comparison vs the original", H1))
rule()
rows = [
    ["Capability", "Original v2.0.0", "New (devel)"],
    ["Convergence", "Bin-occupancy saturation",
     "MSM: timescales, VAMP-2, T_ij flux-weighted error"],
    ["MSM building", "None", "Full pipeline + Bayesian errors + PCCA+"],
    ["CV methods", "fixed, PCA, TICA, TVAE, deep-TICA",
     "+ VAMPNet, SPIB, deep-LDA (unified registry)"],
    ["Feature choice", "Manual, fixed for the run",
     "VAMP-2 selection &amp; optimisation, adaptive"],
    ["CV retraining", "Fixed schedule", "Fixed or VAMP-2-adaptive"],
    ["Spawning", "density, voronoi, lof, fps",
     "+ msm (uncertainty&#215;leverage&#215;flux), we (weighted ensemble)"],
    ["Binning", "Uniform grid only",
     "+ gradient / mab / eigenvector (landscape-adaptive)"],
    ["Execution", "Local multiprocessing",
     "Local + SLURM + PBS array jobs, resubmission"],
    ["Weighted ensemble", "Placeholder stub (no-op)",
     "Correct, weight-conserving split/merge"],
    ["Analysis/plots", "None bundled",
     "autosampler-analyze: ITS, VAMP-2, FES, network"],
    ["Input file", "YAML (sparse examples)",
     "autosampler-init annotated template + docs"],
    ["Tests / CI", "None", "96 tests, CI (3.10 &amp; 3.11), ruff"],
    ["Docs / tutorials", "README only",
     "MkDocs site, notebook tutorial, changelog PDF"],
]
data = [[Paragraph(c, CELLH if i == 0 else CELL) for c in r]
        for i, r in enumerate(rows)]
tbl = Table(data, colWidths=[30*mm, 62*mm, 78*mm], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("TEXTCOLOR", (0, 1), (0, -1), INDIGO),
]))
story.append(tbl)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "<b>Compatibility:</b> every new capability is opt-in. A v2.0.0 input file "
    "runs unchanged; advanced blocks (msm, feature_selection, execution, retrain "
    "policy) are simply omitted by default.", SMALL))
story.append(PageBreak())

# ---------------- New configuration options ----------------
story.append(Paragraph("4. New configuration options", H1))
rule()
story.append(Paragraph(
    "Every option below is optional; omitting the block keeps the previous "
    "behaviour. See the configuration reference and per-topic docs pages.", BODY))
opt_rows = [
    ["Block", "Key", "Default", "What it does"],
    ["msm", "convergence_criteria: transition_matrix", "—",
     "Flux-weighted Dirichlet uncertainty gate on T_ij (params: tol, min_flux)."],
    ["msm", "stable_clustering", "false",
     "Seed clustering from previous centres so microstate IDs / T_ij stay comparable."],
    ["msm", "spawn_uncertainty", "true",
     "Weight msm spawning by outflow uncertainty (× leverage × flux)."],
    ["msm", "spawn_leverage", "1",
     "Number of slow eigenvectors used for the leverage factor."],
    ["msm", "spawn_alpha", "1.0",
     "Weight of the least-counts exploration term in msm spawning."],
    ["binning", "scheme", "uniform",
     "uniform | gradient | mab | eigenvector — landscape-adaptive bin placement."],
    ["binning", "n_fine", "100",
     "Density-histogram resolution for the gradient scheme."],
    ["binning", "smoothing", "3",
     "Density smoothing window for the gradient scheme."],
]
opt_data = [[Paragraph(c, CELLH if i == 0 else CELL) for c in r]
            for i, r in enumerate(opt_rows)]
opt = Table(opt_data, colWidths=[20*mm, 50*mm, 16*mm, 84*mm], repeatRows=1)
opt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("FONTNAME", (1, 1), (1, -1), "Courier"),
    ("FONTSIZE", (1, 1), (1, -1), 7.8),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("TEXTCOLOR", (0, 1), (0, -1), INDIGO),
]))
story.append(opt)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Example — drive convergence with the transition-matrix gate and adaptive bins:",
    SMALL))
code = ParagraphStyle("Code", parent=SMALL, fontName="Courier", fontSize=8,
                      backColor=LGREY, leading=11, leftIndent=6, spaceBefore=2)
for line in [
    "msm:",
    "&nbsp;&nbsp;enabled: true",
    "&nbsp;&nbsp;stable_clustering: true",
    "&nbsp;&nbsp;convergence_mode: all",
    "&nbsp;&nbsp;convergence_criteria:",
    "&nbsp;&nbsp;&nbsp;&nbsp;- {name: vamp2, params: {tol: 0.05}}",
    "&nbsp;&nbsp;&nbsp;&nbsp;- {name: transition_matrix, params: {tol: 0.2, min_flux: 1.0e-4}}",
    "binning:",
    "&nbsp;&nbsp;scheme: gradient",
]:
    story.append(Paragraph(line, code))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Generated for the AutoSampler devel branch. See CHANGELOG.md and the "
    "documentation for full details.", SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(DGREY)
    canvas.drawString(20*mm, 12*mm, "AutoSampler — Development Changelog")
    canvas.drawRightString(190*mm, 12*mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
    topMargin=18*mm, bottomMargin=18*mm,
    title="AutoSampler Development Changelog",
    author="AutoSampler",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
