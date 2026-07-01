"""Generate a structured PDF of the Trails-MD production-readiness review."""

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

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Trails-MD_Review.pdf",
)

NAVY = colors.HexColor("#1A237E")
INDIGO = colors.HexColor("#3949AB")
RED = colors.HexColor("#B71C1C")
AMBER = colors.HexColor("#E65100")
GREEN = colors.HexColor("#2E7D32")
LGREY = colors.HexColor("#ECEFF1")
DGREY = colors.HexColor("#37474F")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=15,
                    spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=INDIGO, fontSize=12,
                    spaceBefore=9, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13.5,
                      alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.4, textColor=DGREY)
TITLE = ParagraphStyle("Title", parent=ss["Title"], textColor=NAVY, fontSize=25,
                       leading=29, spaceAfter=6)
SUB = ParagraphStyle("Sub", parent=ss["Title"], textColor=INDIGO, fontSize=13,
                     leading=17, spaceAfter=4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.3, leading=10.8, spaceAfter=0)
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


def sev_table(rows, col_widths, header_bg=NAVY):
    data = [[Paragraph(c, CELLH if i == 0 else CELL) for c in r]
            for i, r in enumerate(rows)]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)


# ---------------- Title page ----------------
story.append(Spacer(1, 36))
story.append(Paragraph("Trails-MD", TITLE))
story.append(Paragraph("Production &amp; Publication Readiness Review", SUB))
rule()
story.append(Paragraph(
    "An independent code review of the <b>main</b> branch, conducted from the dual "
    "perspective of a code reviewer and a prospective user, ahead of a research "
    "publication announcing the package.", BODY))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Review date: 2026-06-30 &nbsp;|&nbsp; Method: five parallel domain reviewers "
    "(code correctness, documentation, tests/CI/packaging, examples/UX, test-coverage "
    "map) plus direct lead verification of all headline findings.", SMALL))
story.append(Spacer(1, 10))

story.append(Paragraph("Verdict", H2))
story.append(Paragraph(
    "<b>Strong scientific core; not yet release-ready as found.</b> The algorithmic "
    "layer (MSM estimation &amp; convergence, adaptive binning, weighted ensemble, "
    "VAMP-2 feature selection, retraining, scheduler logic) is well-engineered and "
    "genuinely well-tested, and the docs site, annotated template, and rendered "
    "notebook are excellent for a research package. However, the review found "
    "<b>~6 release-gating blockers</b> and a band of correctness, reproducibility, "
    "packaging, and example/test gaps. A subset of blockers has already been fixed "
    "during this review (Section 2); the remainder is itemised with a prioritised "
    "plan.", BODY))
story.append(Spacer(1, 6))

sev_table([
    ["Area", "As-found state", "Severity band"],
    ["Release gates", "Test suite RED; documented install broken; Quick Start example "
     "could not run", "Blocker"],
    ["Correctness / robustness", "Walker-failure &amp; hang handling, engine/scheduler "
     "edge cases, target-mode crash", "Major"],
    ["Reproducibility", "Seeding not actually deterministic across the training loop "
     "(matters for a paper)", "Major"],
    ["Packaging / release", "Incomplete PyPI metadata, heavy deps, partial CI lint, no "
     "release/DOI", "Major"],
    ["Docs / examples / tests", "Gaps: API ref, citations, several features undemoed, "
     "orchestration untested", "Major / Minor"],
], [34 * mm, 104 * mm, 32 * mm])
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Scope note: findings are rated Blocker / Major / Minor / Nit. CONFIRMED items "
    "were traced in source by the lead reviewer; others are reviewer-reported with a "
    "code citation. One widely-suspected bug was investigated and dismissed as a "
    "false positive (see Section 6).", SMALL))
story.append(PageBreak())

# ---------------- Section 1: strengths ----------------
story.append(Paragraph("1. What is already strong", H1))
rule()
story.append(bullets([
    "<b>MSM subsystem</b> (estimator + convergence monitor): connected-set "
    "restriction, MLE/Bayesian estimation, implied timescales, VAMP-2, PCCA+, and a "
    "flux-weighted transition-matrix convergence criterion — with real, "
    "behaviour-asserting tests (recovers 3-state systems, serialization round-trips).",
    "<b>Adaptive binning</b> (gradient / mab / eigenvector), <b>weighted ensemble</b> "
    "(weight-conserving split/merge), <b>VAMP-2 feature selection</b>, and "
    "<b>retraining policies</b> are cleanly factored and tested.",
    "<b>Execution backends</b> (local / SLURM / PBS) behind a clean strategy "
    "interface; scheduler logic is unit-tested with a fake command runner.",
    "<b>Developer experience</b>: annotated input-file template + trails-md-init, a "
    "full MkDocs site, and a rendered Jupyter notebook tutorial that runs on synthetic "
    "data without a GPU.",
    "Test quality where present is high — assertions check real numerical/behavioural "
    "outcomes, not merely the absence of exceptions.",
]))

# ---------------- Section 2: fixed during review ----------------
story.append(Paragraph("2. Fixed during this review", H1))
rule()
story.append(Paragraph(
    "The following were implemented and pushed to branch "
    "<font face='Courier'>claude/trails-md-analysis-plan-jj0n1q</font> (suite green "
    "at 101 tests). They are recorded here so the report doubles as a change log.", BODY))
sev_table([
    ["#", "Item", "Resolution"],
    ["1", "Test suite RED — examples/template.yaml drifted from the module template",
     "Regenerated from the single source; suite green."],
    ["2", "Documented conda install broken (env.yml pinned pydantic 1.x; missing "
     "shapely)", "Bumped to pydantic&gt;=2.0, added shapely + deep-CV backends."],
    ["3", "Delta-checkpoint resume truncated history (broke trails-md-path); "
     "non-atomic writes", "Reconstruct full history across deltas; atomic writes; "
     "tolerate a corrupt delta; added regression tests."],
    ["4", "Local backend aborted the whole iteration if one walker failed",
     "Catch per-walker failure, log, mark unsuccessful (matches scheduler path)."],
    ["5", "Incomplete PyPI metadata; version duplicated",
     "Added readme/license/urls/keywords/authors; single-sourced version; Beta status."],
    ["6", "No citation infrastructure", "Added CITATION.cff + README 'How to cite' "
     "(author list / ORCIDs / DOI flagged as TODO)."],
    ["7", "AlaD Quick Start unrunnable (hardcoded dev path; silent platform typo)",
     "Removed the path, fixed platform_name; documented the GROMACS FF requirement."],
    ["8", "Doc drift (blob/devel link; phi_psi advertised as generic)",
     "Fixed link to main; documented phi_psi as AIB9-specific."],
], [8 * mm, 88 * mm, 74 * mm])
story.append(PageBreak())

# ---------------- Section 3: blockers ----------------
story.append(Paragraph("3. Release-gating blockers", H1))
rule()
story.append(Paragraph(
    "Must be resolved before a general release. Items 1–4, 6 below are addressed in "
    "Section 2; the rest remain open.", BODY))
sev_table([
    ["Blocker", "Detail &amp; status"],
    ["Red test suite",
     "tests/test_input_template.py failed on main. <b>FIXED.</b>"],
    ["Broken documented install",
     "env.yml pydantic 1.x vs Pydantic-v2 code; missing shapely. <b>FIXED.</b>"],
    ["Delta-checkpoint regression",
     "trails-md-path read a truncated (delta-only) history; non-atomic writes. "
     "<b>FIXED.</b>"],
    ["Quick Start cannot run",
     "AlaD needs an external GROMACS force field; no CPU-only hello-world exists. "
     "<b>PARTIAL</b> — de-risked; a self-contained alanine-dipeptide example is the "
     "agreed next step."],
    ["phi_psi crashes off-AIB9",
     "adaptive_feature_type: phi_psi hard-requires 9 AIB residues but is advertised "
     "generically. <b>Documented;</b> code still AIB9-only — rename or generalise."],
    ["No citation / DOI",
     "Gates citability for the paper. <b>Scaffolded</b> (CITATION.cff); needs the "
     "confirmed author list, ORCIDs, affiliations, and a Zenodo DOI."],
], [34 * mm, 136 * mm])

# ---------------- Section 4: major correctness ----------------
story.append(Paragraph("4. Major — correctness &amp; robustness", H1))
rule()
sev_table([
    ["Finding", "Risk", "Location"],
    ["No MD timeout / watchdog on the default (OpenMM, in-process) engine or the local "
     "backend; scheduler poll loop has no overall deadline", "A hung GPU/driver stalls "
     "the campaign forever", "execution/local.py; engines/openmm.py; scheduler.py"],
    ["OpenMM engine only returns True or raises; NaN-recovery re-steps outside any "
     "guard", "An unstable spawn crashes the worker instead of reporting failure",
     "engines/openmm.py"],
    ["Subprocess engines move output after exit-0 without checking a trajectory was "
     "produced", "mdrun exiting 0 with no/empty output → uncaught error",
     "engines/gromacs.py; amber.py"],
    ["Scheduler treats a failed poll command as 'job done'; SLURM job-id match is a "
     "\\b-bounded substring (123 vs 1234)", "Transient squeue/qstat hiccup abandons a "
     "healthy job; id collision", "execution/scheduler.py; slurm.py"],
    ["expected_frames fallback + bare except in frame mapping; only the aggregate "
     "count is checked", "Offsetting over/under-production mis-assigns CV rows → "
     "corrupt lineage", "paths.py"],
    ["Density/Voronoi _weighted_choice uses replace=False with many zero weights "
     "(target mode)", "np.random.choice raises 'fewer non-zero entries than size'",
     "spawners/density.py"],
    ["Triclinic/truncated-octahedron boxes reduced to box diagonal + 90°", "Wrong PBC "
     "for common Amber solvated systems, silently", "engines/amber.py; gromacs.py"],
    ["deep-tica projection runs on CPU tensors while params may be on CUDA; no NaN/"
     "convergence guard in CV training", "Device-mismatch RuntimeError; NaN CVs flow "
     "silently into binning/spawning", "spaces/model.py; spib.py"],
], [92 * mm, 44 * mm, 34 * mm])
story.append(PageBreak())

# ---------------- Section 5: reproducibility / packaging / docs ----------------
story.append(Paragraph("5. Major — reproducibility, packaging, docs/examples/tests", H1))
rule()

story.append(Paragraph("Reproducibility (matters for the publication's claims)", H2))
story.append(bullets([
    "Seed is set once at startup, not before each CV retrain — 2nd+ retrains depend "
    "on all intervening RNG draws.",
    "torch.use_deterministic_algorithms is never enabled; tvae/vampnet DataLoaders get "
    "no seeded generator (vampnet shuffles on the global RNG).",
    "SPIB hardcodes seed=42 and resets the global torch RNG mid-loop, ignoring "
    "random_seed and perturbing later draws.",
    "Density/Voronoi/LOF/FPS spawners and the Voronoi binner draw from the unseeded "
    "global np.random. Net: runs are not bit-reproducible despite the SeedManager.",
]))

story.append(Paragraph("Packaging / release (target: PyPI + conda)", H2))
story.append(bullets([
    "Heavy deps are mandatory; openmm is effectively conda-only, so a plain "
    "pip install often fails to resolve — move openmm (and likely MDAnalysis) to "
    "extras with a lazy import so the base install works.",
    "CI lints only ~18 hand-picked files; the most-edited modules (core.py, config.py, "
    "engines/*, checkpoints/manager.py) are unlinted (~250 ruff findings incl. "
    "<font face='Courier'>from openmm import *</font> star imports). Lint the whole tree.",
    "No coverage gate, no <font face='Courier'>mkdocs build --strict</font> check, no "
    "tag-triggered release/Zenodo workflow; Python 3.12 advertised but untested.",
    "Version single-sourcing &amp; full metadata: <b>DONE</b> in Section 2.",
]))

story.append(Paragraph("Docs, examples &amp; tests", H2))
story.append(bullets([
    "Config schema uses the Pydantic default (extra=ignore): a typo'd key passes "
    "<font face='Courier'>--check</font> silently — consider extra=forbid (and fix the "
    "example configs that rely on it).",
    "configuration.md omits several real keys; CLI reference omits trails-md-init / "
    "-analyze; no API reference (mkdocstrings); method citations are thin "
    "(TICA/VAMPNet/PCCA+/MAB uncited).",
    "No worked example for spib, deep-tica, lof, fps, we, target mode, pbs, or "
    "mab/eigenvector binning — several are headline features.",
    "core.py orchestration, MD engines, half the spawners (voronoi/lof/fps), paths.py, "
    "and 3/5 CLIs are effectively untested; no end-to-end iteration test. (Delta "
    "checkpointing is now covered.)",
]))

# ---------------- Section 6: minor + false positive ----------------
story.append(Paragraph("6. Minor / nits &amp; a dismissed false positive", H1))
rule()
story.append(bullets([
    "Leftover debug <font face='Courier'>print(\"EXACT SPAWN INDICES\", …)</font> "
    "(core.py); hardcoded personal gmx path (paths.py); dead <font face='Courier'>hint"
    "</font> variable (openmm.py).",
    "FPS can return duplicate indices; TVAE BatchNorm + a final batch of size 1 raises; "
    "binning.find_bins raises AttributeError if called before fit.",
    "Broad <font face='Courier'>except Exception</font> swallowing without logging in "
    "several spawner/binner fallbacks; unbounded scheduler _jobs/ artifact growth.",
    "No NaN/inf validation of input CV points across spawners.",
]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Dismissed false positive.</b> A reviewer flagged the MSM least-counts weight "
    "as <font face='Courier'>1/count²</font>. On tracing it, the per-frame value "
    "correctly distributes the microstate-level <font face='Courier'>1/count</font> "
    "least-counts weight across that microstate's frames — exactly mirroring the "
    "MSM-guided path's <font face='Courier'>base / size_per_frame</font>. Left "
    "unchanged.", SMALL))

# ---------------- Section 7: plan ----------------
story.append(Paragraph("7. Recommended path to release", H1))
rule()
story.append(Paragraph("Priority 1 — finish the blockers", H2))
story.append(bullets([
    "Ship the self-contained CPU-only alanine-dipeptide hello-world; generalise or "
    "rename phi_psi; confirm the author list + mint a Zenodo DOI in CITATION.cff.",
]))
story.append(Paragraph("Priority 2 — correctness &amp; reproducibility", H2))
story.append(bullets([
    "Add an MD-walker timeout; make the OpenMM engine report failure instead of "
    "raising; verify trajectories were produced; harden scheduler polling.",
    "Plumb the configured seed through every fit / spawner / DataLoader and enable "
    "torch determinism — required to claim reproducibility in the paper.",
]))
story.append(Paragraph("Priority 3 — packaging, docs, examples, tests", H2))
story.append(bullets([
    "Move openmm/MDAnalysis to extras with lazy imports; lint the whole tree in CI; "
    "add coverage + a strict docs build + a release/Zenodo workflow; test on 3.12.",
    "Add an API reference + a references page; complete the config &amp; CLI docs; add "
    "examples for the undemoed features; broaden tests to orchestration, engines, the "
    "remaining spawners, paths, and CLIs, plus one end-to-end iteration test.",
]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Bottom line: the science and architecture are publication-grade; the gap to "
    "release is engineering hardening, reproducibility, packaging, and coverage — all "
    "tractable, and partly already underway on the review branch.", BODY))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(DGREY)
    canvas.drawString(20 * mm, 12 * mm, "Trails-MD — Production & Publication Readiness Review")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
    topMargin=16 * mm, bottomMargin=18 * mm,
    title="Trails-MD Production & Publication Readiness Review",
    author="Trails-MD review",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
