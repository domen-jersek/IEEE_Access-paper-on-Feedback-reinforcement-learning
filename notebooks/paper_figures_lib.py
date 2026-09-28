"""Figure production library for the IEEE Access manuscript.

Figures are saved as vector PDF + 300-dpi PNG into `paper_ieee_access/paper_figures/`.
Design follows FIGURE_SPECS.md: no embedded titles (captions carry the message),
message annotations inside the panels, fixed palette, direct labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text as _MplText  # noqa: E402

NOTEBOOKS = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOKS))
import paper_lib as L  # noqa: E402

sys.path.insert(0, str(L.ROOT))
from src.config import LiftConfig  # noqa: E402
from src.feedback.lift import compute_lift  # noqa: E402
from src.feedback.loader import load_feedback_bundle  # noqa: E402

OUT = L.ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

COND = "#2b6cb0"      # muted blue (conditioned)
BLIND = "#c05621"     # muted orange (blind)
NEUTRAL = "#4a5568"
ACCENT = "#4a7fb5"    # single-series blue
WARN = "#b7791f"      # muted amber
GREEN = "#2f855a"     # muted green: calibration-repaired
ORACLE = "#b83280"    # magenta (oracle)
SCOPE_COLORS = {"M1_global": "#777777", "M2_team": "#2a6fbb", "M3_class": "#d17a22",
                "M4_intersection": "#16856b", "M5_backoff": "#8b4fb3", "backoff_eb": "#8b4fb3"}
SCOPE_LABELS = {"M1_global": "Global", "M2_team": "Team", "M3_class": "Class",
                "M4_intersection": "Team∩Class", "M5_backoff": "Backoff"}

# Final print sizes (IEEE Access). Fonts below are set for these sizes: a figure
# rendered at 7.16 in wide is printed 1:1 at text width, so 8 pt reads as 8 pt.
DOUBLE = 7.16
SINGLE = 3.46

# Crisp, light style: thin grey frame, short ticks, no chartjunk.
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#333333",
    "text.color": "#222222",
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "grid.alpha": 0.0,
    "grid.linewidth": 0.6,
    "grid.color": "#b0b0b0",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{name}.pdf/.png")


def tag(ax, text: str, dx: float = 0.0, dy: float = 1.02, ha: str = "left", **kw) -> None:
    """Panel tag placed outside the data area (above the axes)."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha=ha, va="bottom", fontsize=9,
            fontweight="bold", color="#111111", **kw)


def annot(ax, text: str, x: float = 0.98, y: float = 0.02, ha: str = "right", va: str = "bottom",
          color: str = NEUTRAL, fontsize: float = 7.5, **kw) -> None:
    """Sparse in-panel annotation. Policy: at most one per panel, in empty space."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=color, **kw)


def check_overlaps(fig, verbose: bool = True) -> list[str]:
    """QA helper: report pairwise intersections of in-axes text objects.

    Includes annotations, panel tags, legends, and axis labels; tick labels are
    excluded (they are managed by matplotlib and cannot be moved independently).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items: list[tuple[str, object]] = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                items.append((t.get_text().replace("\n", " ")[:40], t))
        legend = ax.get_legend()
        if legend is not None:
            for t in legend.get_texts():
                items.append((f"legend:{t.get_text()}", t))
        for label in (ax.xaxis.label, ax.yaxis.label):
            if label.get_text().strip():
                items.append((f"axis:{label.get_text()}", label))
    problems = []
    boxes = []
    for text, artist in items:
        try:
            boxes.append((text, _MplText.get_window_extent(artist, renderer=renderer)))
        except Exception:
            boxes.append((text, artist.get_window_extent(renderer=renderer)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, ba = boxes[i]
            b, bb = boxes[j]
            if ba.overlaps(bb):
                problems.append(f"overlap: '{a}' <-> '{b}'")
    if verbose:
        for p in problems:
            print("  QA", p)
        print(f"  QA overlaps: {len(problems)}")
    return problems



def check_artists(fig, verbose: bool = True, pad: float = 0.5) -> list[str]:
    """QA helper: in-axes text versus drawn data (bars, markers, polylines)."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []
    for ax in fig.axes:
        texts = []
        for t in ax.texts:
            if not t.get_text().strip():
                continue
            bb = _MplText.get_window_extent(t, renderer=renderer)
            texts.append((t.get_text().replace("\n", " ")[:40],
                          Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)))
        if not texts:
            continue

        def point_hit(px, py):
            for name, tb in texts:
                if tb.x0 <= px <= tb.x1 and tb.y0 <= py <= tb.y1:
                    return name
            return None

        from matplotlib.patches import Polygon as _Polygon, Rectangle as _Rectangle
        for patch in ax.patches:
            if isinstance(patch, FancyArrowPatch) or patch.get_gid() == "guide":
                continue
            if isinstance(patch, _Polygon) and not isinstance(patch, _Rectangle):
                try:
                    verts = patch.get_path().vertices
                except Exception:
                    continue
                pts = ax.transData.transform(patch.get_patch_transform().transform(verts))
                for px, py in pts:
                    name = point_hit(px, py)
                    if name:
                        problems.append("curve <-> '%s'" % name)
                        break
            else:
                try:
                    bb = patch.get_window_extent(renderer=renderer)
                except Exception:
                    continue
                for name, tb in texts:
                    if tb.overlaps(bb):
                        problems.append("bar <-> '%s'" % name)
                        break
        for line in ax.lines:
            if line.get_gid() == "guide":
                continue
            xyd = line.get_xydata()
            if len(xyd) == 0:
                continue
            pts = ax.transData.transform(xyd)
            for px, py in pts[:: max(1, len(pts) // 200)]:
                name = point_hit(px, py)
                if name:
                    problems.append("line <-> '%s'" % name)
                    break
        for coll in ax.collections:
            if isinstance(coll, LineCollection):
                stop = False
                for seg in coll.get_segments():
                    for px, py in ax.transData.transform(seg):
                        name = point_hit(px, py)
                        if name:
                            problems.append("segment <-> '%s'" % name)
                            stop = True
                            break
                    if stop:
                        break
    problems = sorted(set(problems))
    if verbose:
        for problem in problems:
            print("  QAart", problem)
        print("  QA artist hits:", len(problems))
    return problems


def hgrid(ax) -> None:
    """API compatibility; the editorial style uses no grid."""
    return


def zero(ax, **kw) -> None:
    kw.setdefault("color", "#3d4451")
    kw.setdefault("lw", 0.7)
    line = ax.axhline(0.0, **kw)
    line.set_gid("guide")


# ---------------------------------------------------------------------------
# shared data loaders
# ---------------------------------------------------------------------------

def resc() -> pd.DataFrame:
    return L.load_rescore_comparison()


def resc_row(folder: str) -> pd.Series:
    return resc()[resc()["run"] == folder].iloc[0]


def dev_details(folder: str) -> dict:
    return {r["ticket_id"]: r for r in L.load_details(folder)}


def m4_intersection_lift_values(protocol: str = "conditioned") -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate lifts for the M4 (team∩class) routing on dev, exactly via src.lift.

    Returns (laplace_lifts, eb_lifts) over all candidates of the dev top-100 pools.
    """
    dataset = L.load_dataset().set_index("seq_id")
    dev = L.load_splits(42)["dev"]
    pools = pd.read_parquet(L.RESULTS / "retriever_ladder" / "pools" / "dense_minilm_dev_seed42.parquet")
    bundle = load_feedback_bundle(L.DATA / f"feedback_{protocol}.db", exclude_query_ids=set(dev), mode="continuous")
    lap_cfg, eb_cfg = LiftConfig.laplace(), LiftConfig.laplace_eb(2.0)
    laplace, eb = [], []
    for query_id, group in pools.groupby("query_id"):
        row = dataset.loc[query_id]
        key = f"intersection:{row['intent_class']}:{row['Team->Name']}"
        prior = bundle.prior_mean(key)
        for candidate in group["seq_id"]:
            evidence = bundle.scores.get(str(candidate), {}).get(key, {"pos": 0.0, "neg": 0.0})
            pos, neg = float(evidence.get("pos", 0.0)), float(evidence.get("neg", 0.0))
            laplace.append(compute_lift(pos, neg, lap_cfg))
            eb.append(compute_lift(pos, neg, eb_cfg, prior))
    return np.asarray(laplace), np.asarray(eb)
