"""
Build the paper notebook suite (analysis + modeling, explanation on every output).

Notebooks
---------
  02_protocol_and_validity      ANALYSIS   data, splits, feedback protocols, judge calibration, proxy validity
  03_when_feedback_helps        ANALYSIS   conditional benefit, scope, volume, ceilings, retriever ladder, negative controls
  04_modeling_the_prior         MODELING   lift formulas, centering, scaling, evidence discipline, blend, tradeoff
  05_modeling_the_control_policy MODELING  pre-generation gates, dev->eval methodology, policy value, ceiling recovery
  06_final_results_and_claims   RESULTS    locked configuration, eval, robustness, independent metrics, claim ledger

Every code cell is preceded by "What this cell does" and followed by a
"Reading" cell that explains the output. Sections use
Question -> What we do -> Figure/Table -> Reading -> Artifact -> Caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# cell helpers
# ---------------------------------------------------------------------------

def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def section(question: str, doing: str, artifact: str, caveat: str) -> dict:
    return markdown(f"""### {question}

**What we do.** {doing}

**Artifact.** `{artifact}`

**Caveat.** {caveat}""")


def explain(text: str) -> dict:
    return markdown(f"*What this cell does.* {text}")


def reading(text: str) -> dict:
    return markdown(f"**Reading.** {text}")


SETUP = code("""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, Markdown
import paper_lib as L

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 130)
pd.set_option("display.width", 200)
np.random.seed(42)

PRIMARY_RUN = "M4_intersection_dev_conditioned_continuous"
BLIND_RUN = "M4_intersection_dev_blind_continuous"
GATE_FEATURE_COLS = ["top1_faiss", "top5_mean_faiss", "top5_min_faiss", "top5_std_faiss",
                     "retrieval_margin", "top5_spread", "retrieval_overlap", "max_lift",
                     "mean_lift", "n_positive_lifts", "n_negative_lifts", "lift_conflict",
                     "evidence_density", "query_desc_len", "query_title_len"]
print("Project root:", L.ROOT)
""")

BANNER = {
    "ANALYSIS": "This is an **analysis** notebook: it characterizes the data and results and makes no method claims.",
    "MODELING": "This is a **modeling** notebook: every section introduces one modeling choice and evaluates it.",
    "RESULTS": "This is the **results** notebook: it reports the locked configuration and the claim ledger.",
}


def notebook(title: str, kind: str, subtitle: str, cells: list[dict]) -> dict:
    head = markdown(f"""# {title}

**{kind} notebook — {subtitle}**

{BANNER[kind]}

Every section follows *Question → What we do → Figure/Table → Reading → Artifact → Caveat*.
Each code cell states what it does and each output is interpreted in the following
cell, so a reader with no access to the code can follow the reasoning. All numbers are
read from immutable artifacts in `results/` through `paper_lib`; missing optional
experiments print `PENDING` with their producer command instead of failing.""")
    return {"cells": [head, SETUP, *cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


# ===========================================================================
# 02 — ANALYSIS: protocol and validity
# ===========================================================================

NB02 = notebook(
    "Protocol and Validity",
    "ANALYSIS",
    "the corpus, the two feedback protocols, judge calibration, and the offline proxy",
    [
        section("What evidence exists and how was it produced?",
                "List every artifact the paper relies on and the latest run that produced it (smoke runs and superseded re-runs excluded).",
                "results/registry.csv", "Availability is not quality; the rest of the notebook assesses validity."),
        explain("Print an availability table (which experiments have run) and a provenance table (latest run per script)."),
        code("""
display(L.artifact_status())
display(L.provenance_table())
"""),
        reading("""The availability table shows the paper's full evidence base: the corrected protocol
runs, calibration, proxy validation, rescoring, the conditioned and blind retriever ladders,
the blend/backoff grids, the volume curves, and the gate pilots. The provenance table ties each
number to a run id, a git commit, and its inputs, so every figure can be reproduced."""),

        section("What does the ticket corpus look like, and how is it partitioned?",
                "The corpus has 1,595 IT tickets; splits are train (feedback source) / dev (development) / eval (final test). Composition is audited but the assignment is a seeded random permutation, not stratified.",
                "data/processed/dataset.parquet; splits/split_seed42.json",
                "One organizational corpus, so cross-domain validity is out of scope."),
        explain("Load the canonical table, map tickets to splits, print split sizes, and plot the largest teams and intent classes."),
        code("""
df = L.load_dataset(columns=["seq_id", "intent_class", "Team->Name"])
split = L.load_splits(42)
membership = {qid: part for part in ("train", "dev", "eval") for qid in split[part]}
composition = (df.assign(split=df["seq_id"].map(membership)).dropna(subset=["split"])
               .groupby("split").agg(n=("seq_id", "size"), teams=("Team->Name", "nunique"), classes=("intent_class", "nunique")))
display(composition)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
df["Team->Name"].value_counts().head(12).sort_values().plot.barh(ax=axes[0], color="#2a6fbb", title="Largest teams")
df["intent_class"].value_counts().head(12).sort_values().plot.barh(ax=axes[1], color="#16856b", title="Largest intent classes")
plt.tight_layout(); L.savefig("02_corpus_composition", run_ids=[]); plt.show()
"""),
        reading("""The corpus is heavily imbalanced: a handful of teams and a few intent classes dominate,
and many scopes have very few tickets. This matters for the paper because feedback evidence is
aggregated *per scope*, so sparse scopes (small teams, rare intersections) will carry little
evidence — which is exactly the regime the hierarchical backoff is designed for. The eval split
is never used for any decision."""),

        section("How is the feedback signal produced?",
                "Two protocols: resolution-informed (conditioned) and ticket-only (blind). We compare their score distributions and their reliability as usefulness rankers.",
                "results/feedback_calibration/report.json; reliability_{conditioned,blind}.csv",
                "AUC measures ranking, not probability calibration; conditioned is resolution-informed by construction."),
        explain("Print per-protocol summary statistics (mean, zero share, distinct values, AUC, monotonicity), then plot the reliability curves."),
        code("""
cal = L.load_calib_report()
rows = []
for proto, r in cal["protocols"].items():
    auc = r.get("auc_of_score", {})
    rows.append({"protocol": proto, "mean_score": r["score_distribution"]["mean"],
                 "pct_exact_0": r["score_distribution"]["pct_exact_0"],
                 "n_distinct": r["score_distribution"]["n_distinct_scores"],
                 "auc_same_reply": auc.get("same_reply"), "auc_same_group": auc.get("same_group"),
                 "monotone_usefulness": r["reliability_monotone_in_usefulness"]})
display(pd.DataFrame(rows).round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for proto, color in [("conditioned", "#16856b"), ("blind", "#c44e52")]:
    rel = L.load_calib_reliability(proto)
    axes[0].plot(rel["mean_score"], rel["p_same_reply"], marker="o", label=proto, color=color)
    axes[1].plot(rel["mean_score"], rel["p_same_group"], marker="o", label=proto, color=color)
for ax, t in zip(axes, ["Same reply", "Same procedure group"]):
    ax.plot([0, 1], [0, 1], ls="--", color="lightgray"); ax.set(title=t, xlabel="Mean judge score"); ax.legend()
plt.tight_layout(); L.savefig("02_judge_reliability", run_ids=[]); plt.show()
"""),
        reading("""Both protocols produce heavily zero-inflated scores (median 0; roughly half of all
judgments are exactly 0), but the conditioned judge separates useful from useless candidates far
better (AUC 0.90 vs 0.84 for same-reply). Blind reliability is *not* monotone in usefulness, which
means higher blind scores can even be less reliable in places. Two consequences: (1) any lift
formula centred at 0.5 is mis-specified for these distributions (developed in notebook 04); and
(2) the conditioned protocol is the stronger signal, consistent with the argument that it encodes
resolution knowledge available for historical tickets."""),

        section("Does the offline retrieval proxy predict generated-answer quality?",
                "The proxy scores a ranking by reply-text similarity to the reference. We check whether proxy deltas track generated-answer deltas at the configuration level and at the ticket level.",
                "results/proxy_validation/report.json; per_ticket.csv",
                "The proxy is a retrieval-only signal, not an answer metric."),
        explain("Print the proxy decision, scatter proxy vs generated delta on tickets whose ranking changed, and compare method-level orderings."),
        code("""
report = L.load_proxy_report(); tickets = L.load_proxy_per_ticket()
display(pd.json_normalize(report["decision"]))
changed = tickets[tickets["top1_changed"].astype(bool)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.regplot(data=changed, x="minilm_d_proxy_top1", y="delta_cosine", scatter_kws={"alpha": .25}, ax=axes[0])
axes[0].set(title="Per-ticket: proxy vs generated", xlabel="Offline proxy delta", ylabel="Generated-answer delta")
method = changed.groupby("run", as_index=False).agg(generated=("delta_cosine", "mean"), proxy=("minilm_d_proxy_top1", "mean"))
sns.scatterplot(data=method, x="proxy", y="generated", hue="run", s=90, ax=axes[1], legend=False)
for _, row in method.iterrows():
    axes[1].annotate(row["run"].split("_dev")[0], (row["proxy"], row["generated"]), fontsize=8)
axes[1].axhline(0, color="gray", lw=1); axes[1].axvline(0, color="gray", lw=1); axes[1].set(title="Method ordering")
plt.tight_layout(); L.savefig("02_proxy_validity", run_ids=[]); plt.show()
"""),
        reading("""The proxy orders *configurations* almost perfectly (method-order rank correlation 1.0) but
predicts individual tickets only weakly (rank correlation about 0.45). This is the paper's
methodological boundary: the proxy is a valid cheap tool for choosing which configurations deserve
a paid generation run, and an invalid label for per-ticket decisions. Later we show a concrete
sign flip (hybrid_rrf looks positive offline but is negative once generated), which is why every
headline number in this paper comes from generation."""),

        section("Did any run produce spurious deltas from duplicate generation?",
                "When the feedback ranking leaves the generated context unchanged, the two prompts are identical. We audit legacy runs for such tickets and force their delta to zero in a corrected diagnostic.",
                "results/identical_prompt_audit.csv", "Corrects generation noise only; stored artifacts are unchanged."),
        explain("Display, per run, how many tickets had identical ordered context and how the corrected mean shifts."),
        code("""
audit = L.load_identical_prompt_audit()
display(audit[["run", "n_identical_prompts", "pct_identical_prompts", "raw_mean_delta_cosine", "corrected_mean_delta_cosine", "correction"]].round(4))
"""),
        reading("""Between 2% and 35% of legacy tickets had identical ordered contexts (except baselines,
which are identical by construction). Correcting them changes the means by less than 0.002, so the
qualitative findings are unaffected; the correction is reported for transparency, and future runs
generate once when the prompts match."""),
        markdown("""## Conclusion — protocol and validity

The feedback signal is well-behaved under the resolution-informed protocol and weaker but
informative under the ticket-only protocol. The offline proxy is valid for configuration selection
and invalid for tickets. These three results frame everything that follows: we use the proxy to
choose configurations, generate to report results, and always report both protocols."""),
    ],
)


# ===========================================================================
# 03 — ANALYSIS: when does feedback help
# ===========================================================================

NB03 = notebook(
    "When Does Feedback Help?",
    "ANALYSIS",
    "conditional benefit by difficulty, evidence, scope, volume, and retrieval strength",
    [
        section("Is benefit conditional on retrieval difficulty?",
                "Bin tickets by baseline top-1 similarity and look at the mean generated delta in each bin.",
                "generated dev run details", "Descriptive bins, not tuned thresholds."),
        explain("Load the resolution-informed M4 run, join each ticket's baseline top-1 similarity to its delta, and plot the mean delta by similarity tier."),
        code("""
details = L.load_details(PRIMARY_RUN)
frame = pd.DataFrame([{"ticket_id": r["ticket_id"],
                       "top1": r["baseline"]["retrieval"][0]["faiss_score"] if r["baseline"]["retrieval"] else np.nan,
                       "delta": r["deltas"]["delta_cosine"]} for r in details]).dropna()
frame["tier"] = pd.cut(frame["top1"], bins=[0, .55, .7, .8, .9, 1.01],
                       labels=["<0.55", "0.55-0.70", "0.70-0.80", "0.80-0.90", ">0.90"])
tier = frame.groupby("tier", observed=True)["delta"].agg(["mean", "size"]); display(tier.round(4))
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.barplot(data=frame, x="tier", y="delta", ax=ax, color="#8b4fb3", errorbar="ci")
ax.axhline(0, color="black", lw=1); ax.set(xlabel="Baseline top-1 similarity", ylabel="Mean generated delta",
                                           title="Benefit by retrieval difficulty")
plt.tight_layout(); L.savefig("03_benefit_by_difficulty", run_ids=[]); plt.show()
"""),
        reading("""Benefit is not uniform across difficulty. On already-confident retrievals (top-1 above
about 0.9) feedback tends to do nothing or harm, because the baseline context is already correct and
any re-ranking can only displace it. On moderately uncertain retrievals there is room to improve.
This is the empirical basis for a control policy: an ideal system would intervene only where the
baseline is uncertain — the question is whether that uncertainty is informative enough, which
notebook 05 answers."""),

        section("Does benefit depend on how much evidence the pool carries?",
                "Relate per-ticket proxy gain to the fraction of the candidate pool that has feedback evidence, per scope.",
                "results/retriever_ladder/per_ticket.parquet", "Descriptive association under the offline proxy."),
        explain("Take the dense_minilm M4 empirical-Bayes cells and plot proxy gain against intersection-scope coverage."),
        code("""
pt = L.load_ladder_per_ticket()
sub = pt[(pt["retriever"] == "dense_minilm") & (pt["lift"] == "laplace_eb_k2|abs") & (pt["routing"] == "M4_intersection")]
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=sub, x="coverage_intersection", y="minilm_d_proxy_top1", alpha=.4, ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(xlabel="Intersection coverage of the pool", ylabel="Proxy delta",
                                           title="Benefit vs evidence coverage")
plt.tight_layout(); L.savefig("03_benefit_by_coverage", run_ids=[]); plt.show()
display(sub[["coverage_intersection", "minilm_d_proxy_top1"]].corr().round(3))
"""),
        reading("""Gain and intersection coverage are positively associated: scopes with more evidence
move more tickets in the right direction, while sparse scopes contribute little. This is the
mechanistic reason evidence-aware routing (backoff) matters — it avoids acting on scopes where
evidence is too thin to be trustworthy."""),

        section("Is scope ordering stable, and does it depend on the protocol?",
                "Compare the best proxy gain per routing under the resolution-informed and ticket-only ladders.",
                "results/retriever_ladder{,_blind}/grid.csv", "Best-cell selection is optimistic; used for ordering."),
        explain("Compute the best gain per routing for each protocol and plot them side by side."),
        code("""
grid = L.load_ladder_grid()
blind = L.load_ladder_grid("blind")
metric = "minilm_d_proxy_top1"
cmp = grid.groupby("routing")[metric].max().rename("conditioned").to_frame()
cmp["blind"] = blind.groupby("routing")[metric].max()
display(cmp.round(4))
fig, ax = plt.subplots(figsize=(10, 4.8)); cmp.plot.bar(ax=ax, color=["#16856b", "#c44e52"])
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Best proxy gain", title="Scope ordering by protocol"); ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); L.savefig("03_scope_ordering_protocol", run_ids=[]); plt.show()
"""),
        reading("""Fine scopes (team, team∩class) dominate broad pooling in both protocols, and the
ordering is stable across retrievers. The *magnitude* shrinks under ticket-only feedback, and some
setups cross into negative territory — foreshadowing the protocol reversal in the generated results.
The stable ordering is what lets us fix a scope rule (backoff) and vary only centering/scaling."""),

        section("How much feedback is needed before it helps?",
                "Subsample judged training queries from 1% to 100% across ten seeds and re-score dev pools, per scope.",
                "results/feedback_volume{,_blind}/summary.csv", "Fractions subsample judged queries, not candidate rows."),
        explain("Plot seed-averaged proxy gain vs feedback fraction for M4 and backoff, for both protocols."),
        code("""
vol = L.load_volume_summary(); volb = L.load_volume_summary("blind")
fig, ax = plt.subplots(figsize=(10, 5))
for (lift, routing), part in vol.groupby(["lift", "routing"]):
    if routing in ("M4_intersection", "M5_backoff2"):
        ax.plot(part["fraction"], part["mean_delta_top1_mean"], marker="o", label=f"conditioned {routing}")
for (lift, routing), part in volb.groupby(["lift", "routing"]):
    if routing in ("M4_intersection", "M5_backoff2"):
        ax.plot(part["fraction"], part["mean_delta_top1_mean"], marker="s", ls="--", label=f"blind {routing}")
ax.axhline(0, color="black", lw=1); ax.set(xscale="log", xlabel="Feedback fraction of train queries",
                                           ylabel="Proxy gain", title="Feedback-volume learning curves"); ax.legend(fontsize=8)
plt.tight_layout(); L.savefig("03_volume_curves", run_ids=[]); plt.show()
"""),
        reading("""Under resolution-informed feedback the curves are non-monotone and hover near zero at
small volume, then improve as evidence accumulates. Under ticket-only feedback the curves are
*monotonically negative and worsen with volume*: more noisy feedback causes more harm. The practical
message is that the amount of feedback is not the lever — its reliability is. A system should
accumulate feedback only when it is trustworthy, and apply it selectively."""),

        section("How much benefit is attainable at all (oracle ceiling)?",
                "For a generated run, compute always-on, never-on, and the oracle that applies feedback only where it truly helped.",
                "generated dev run details", "The oracle uses labels, so it is a ceiling, not a method."),
        explain("Show, for the conditioned and blind M4 runs, the always-on mean, the oracle mean, and the share of harmed tickets."),
        code("""
rows = []
for name, run in [("conditioned", PRIMARY_RUN), ("blind", BLIND_RUN)]:
    d = np.array([r["deltas"]["delta_cosine"] for r in L.load_details(run)])
    rows.append({"protocol": name, "always_on": d.mean(), "never_on": 0.0,
                 "oracle": np.where(d > 0, d, 0).mean(), "pct_harmed": (d < 0).mean(),
                 "pct_helped": (d > 0).mean()})
ceil = pd.DataFrame(rows); display(ceil.round(4))
fig, ax = plt.subplots(figsize=(9, 4.8))
ceil.set_index("protocol")[["always_on", "oracle"]].plot.bar(ax=ax, color=["#777777", "#16856b"])
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean generated delta", title="Always-on vs oracle ceiling")
plt.tight_layout(); L.savefig("03_oracle_ceiling", run_ids=[]); plt.show()
"""),
        reading("""Roughly 43% of tickets are *harmed* even in the conditioned setting. Always-on still has
a positive mean because the average help is larger than the average harm, but the oracle — which
uses the labels to keep only the helpful tickets — is roughly twice as high. The headroom is real
but unreachable without per-ticket knowledge; notebook 05 asks whether pre-generation features can
approximate it."""),

        section("Does feedback gain depend on retrieval strength?",
                "Use the offline ladder to relate base retrieval quality to the best feedback gain across six retrievers.",
                "results/retriever_ladder/ladder_curve.csv", "Best-cell gain is optimistic; the trend is the finding."),
        explain("Plot base proxy quality against best feedback gain per retriever."),
        code("""
curve = L.load_ladder_curve().sort_values("base_minilm_proxy_top1")
display(curve[["retriever", "base_minilm_proxy_top1", "best_routing", "best_lift", "best_gain", "primary_ci_lo", "primary_ci_hi"]].round(4))
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(curve["base_minilm_proxy_top1"], curve["best_gain"],
            yerr=[curve["best_gain"] - curve["primary_ci_lo"], curve["primary_ci_hi"] - curve["best_gain"]], fmt="o", color="#2a6fbb")
for _, r in curve.iterrows():
    ax.annotate(r["retriever"], (r["base_minilm_proxy_top1"], r["best_gain"]), fontsize=8)
ax.set(xlabel="Base proxy quality", ylabel="Best feedback gain", title="Retriever strength vs feedback gain")
plt.tight_layout(); L.savefig("03_retriever_strength", run_ids=[]); plt.show()
"""),
        reading("""The offline proxy says the largest gains are on the weakest base (BM25, +0.097) and
that dense MiniLM has the best base top-1 quality (0.799) with a solid gain (+0.053); the generated
evidence sharpens this: the calibrated blind gain found on dense MiniLM (+0.007/+0.010) does not
transfer to a hybrid retriever (−0.006) and only weakly to a cross-encoder base (+0.004), while
baseline answer quality across the three retrievers is essentially identical (cosine 0.654–0.658).
Best-cell gains are selected over ~300 cells and are therefore optimistic; the trend, not the level,
is the finding. Note also that the alternate-retriever runs use different lift configurations, so
retriever and calibration are confounded there."""),

        section("What does not work?",
                "Two candidate mechanisms were tested and rejected: a semantic relevance filter and feedback on a hybrid retriever.",
                "results/retriever_ladder_semantic{,_blind}/grid.csv; generated hybrid run",
                "Negative results bound the claims."),
        explain("Compare the semantic-filter routings against the best base routing per retriever, and print the generated hybrid result."),
        code("""
sem = L.load_ladder_grid("semantic")
base = sem[~sem["routing"].str.startswith("semantic")]
best_base = base.loc[base.groupby("retriever")["minilm_d_proxy_top1"].idxmax()][["retriever", "routing", "minilm_d_proxy_top1"]]
sem_only = sem[sem["routing"].str.startswith("semantic")]
best_sem = sem_only.loc[sem_only.groupby("retriever")["minilm_d_proxy_top1"].idxmax()][["retriever", "routing", "minilm_d_proxy_top1"]]
cmp = best_base.merge(best_sem, on="retriever", suffixes=("_base", "_semantic"))
display(cmp.round(4))
hyb = L.load_summary("M4_intersection_dev_blind_continuous_liftlaplace_eb_pool_std0.5_hybrid_rrf")
display(pd.DataFrame([hyb["metrics"]]).round(4))
"""),
        reading("""The semantic relevance filter is consistently *worse* than the base routings on every
retriever, so the hypothesis that suppressing semantically distant candidates helps is rejected —
fine-scope metadata already captures that distinction. The generated hybrid-retriever run is negative
(−0.006 generated cosine), showing the calibrated blind gain found on dense MiniLM does not transfer
to a hybrid retriever. Both results are reported to bound the method's scope rather than hidden."""),
        markdown("""## Conclusion — when does feedback help

Feedback benefit is conditional on retrieval difficulty, evidence coverage, scope, and — decisively
— on how the feedback was produced. Fine scopes help and broad pooling harms; more feedback is only
better if it is trustworthy; and there is substantial per-ticket variance that a control policy
might exploit. The next notebook turns these observations into modeling choices."""),
    ],
)


# ===========================================================================
# 04 — MODELING: the prior
# ===========================================================================

NB04 = notebook(
    "Modeling the Feedback Prior",
    "MODELING",
    "lift formulas, empirical-Bayes centering, pool-relative scaling, evidence discipline, and the aggression tradeoff",
    [
        markdown("""## The modeling problem

The analysis notebook showed that feedback re-ranking is powerful but fragile. We now model the
*magnitude and sign* of the feedback bonus. A retrieval bonus is built from two ingredients:

1. **Evidence**: how often, and how positively, a candidate was judged useful within a scope.
2. **A lift formula**: how that evidence is converted into a bounded bonus added to the retrieval score.

The two modeling levers we study are **where the formula is centred** (what counts as "neutral"
feedback) and **how large the bonus is relative to the retriever's score scale**. We also test which
scope's evidence to use (evidence discipline)."""),

        section("Which lift formulas exist, and what are their failure modes?",
                "Implementations: Laplace, empirical-Bayes Laplace, tanh, and Bayesian lower-confidence-bound. We inspect how each maps evidence to a bonus.",
                "src/feedback/lift.py; results/feedback_calibration/saturation.csv", "tanh and LCB are implemented but not yet evaluated on data (flagged)."),
        explain("Plot each formula's output as a function of net evidence, then read the empirical saturation table for Laplace vs empirical-Bayes."),
        code("""
import numpy as np
pos = np.linspace(0, 10, 200)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(pos - 5, (pos + 1) / (pos + 5 + 2) - 0.5, label="laplace (centred 0.5)")
ax.plot(pos - 5, np.tanh(pos - 5) * 0.2, label="tanh")
ax.axhline(0, color="black", lw=1); ax.set(xlabel="Pos - Neg (net evidence)", ylabel="Raw bonus", title="Lift formula shapes")
ax.legend(); plt.tight_layout(); L.savefig("04_lift_shapes", run_ids=[]); plt.show()

sat = L.load_calib_saturation()
display(sat.sort_values(["routing", "lift"])[["lift", "routing", "pct_at_minus_cap", "pct_at_plus_cap",
        "pct_zero", "n_distinct_lift_values", "lift_std_over_faiss_std"]].round(3))
"""),
        reading("""Laplace centred at 0.5 is *linear* in evidence and symmetric, so with a corpus whose
judge scores are mostly below 0.5 it pushes most candidates negative. The saturation table confirms
the practical consequence: for the legacy Laplace lift, 40–50% of pool bonuses sit at the negative
cap, meaning the formula cannot distinguish "bad" from "very bad". Empirical-Bayes centering
(developed next) reduces cap saturation to a few percent and produces far more distinct values. tanh
and the LCB are implemented for completeness but are flagged as not yet evaluated, so we do not
claim results for them."""),

        section("Modeling choice 1 — centre the formula on the scope's own base rate",
                "Replace the fixed 0.5 neutral point with each scope's empirical mean judge score (empirical-Bayes), so that average evidence yields zero bonus.",
                "results/feedback_calibration/saturation.csv; results/retriever_ladder/grid.csv", "Prior strength κ and cap remain fixed."),
        explain("Show the heterogeneous scope base rates, then compare legacy Laplace against empirical-Bayes on the conditioned ladder per routing."),
        code("""
priors = L.load_calib_scope_priors("conditioned").copy()
priors["type"] = priors["scope"].str.split(":").str[0]
fig, ax = plt.subplots(figsize=(9, 4.5)); sns.histplot(data=priors, x="mean", hue="type", element="step", common_norm=False, ax=ax)
ax.set(xlabel="Observed scope base rate", title="Scope base rates are heterogeneous")
plt.tight_layout(); L.savefig("04_scope_base_rates", run_ids=[]); plt.show()

grid = L.load_ladder_grid(); metric = "minilm_d_proxy_top1"
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=grid[grid["lift"].isin(["laplace|abs", "laplace_eb_k2|abs", "laplace_eb_k10|abs"])],
            x="routing", y=metric, hue="lift", ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Best proxy gain", title="Centering: Laplace vs empirical-Bayes (conditioned, absolute)")
plt.tight_layout(); L.savefig("04_centering_ablation", run_ids=[]); plt.show()
"""),
        reading("""Scope base rates range from near 0 to about 0.55, so a single 0.5 centre is wrong for
almost every scope. Centering on the scope's own mean turns "average evidence" into "no bonus" and
lets strong evidence stand out. On the conditioned ladder, empirical-Bayes improves the broad
routings (which legacy Laplace drove negative) while leaving the fine scopes competitive. The
practical effect is largest where the fixed centre was most wrong."""),

        section("Modeling choice 2 — express the bonus in the retriever's own score units",
                "Absolute bonuses add a fixed value to cosine scores; pool-relative scaling multiplies the bounded bonus by the candidate pool's score standard deviation, so the same cap means the same relative shift across retrievers.",
                "results/retriever_ladder_blind/grid.csv", "Offline proxy evidence; generated confirmation follows."),
        explain("Group the blind ladder by scaling mode and show the best gain per scaling, then the per-retriever best cells."),
        code("""
blind = L.load_ladder_grid("blind"); metric = "minilm_d_proxy_top1"
blind = blind.assign(scale=blind["lift"].str.split("|").str[1])
best_by_scale = blind.groupby("scale")[metric].max().sort_values(ascending=False)
display(best_by_scale.round(4))
fig, ax = plt.subplots(figsize=(9, 4.5)); best_by_scale.plot.bar(ax=ax, color="#16856b")
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Best blind proxy gain", xlabel="Scaling mode",
                                           title="Pool-relative scaling rescues ticket-only feedback")
plt.tight_layout(); L.savefig("04_scaling_rescue", run_ids=[]); plt.show()
best = blind.loc[blind.groupby("retriever")[metric].idxmax()][["retriever", "routing", "lift", metric, "primary_ci_lo", "primary_ci_hi"]]
display(best.round(4))
"""),
        reading("""Under ticket-only feedback, absolute bonuses are harmful (the bonus is the wrong size
for the score distribution), whereas pool-relative scaling — which shrinks the bonus when the pool's
scores are tightly packed — turns the same evidence into a positive gain. This is the single most
important modeling result: the failure of realistic feedback is a *scaling* failure, not a signal
failure, and it is fixed by expressing the bonus relative to the retriever's own scale."""),

        section("Modeling choice 3 — discipline the evidence (hierarchical backoff)",
                "Instead of a fixed scope, use the finest scope that has at least a minimum amount of evidence, falling back to broader scopes otherwise.",
                "results/retriever_ladder{,_blind}/grid.csv; results/blend*/learned_weights.json", "Minimum evidence is a hyperparameter set on dev."),
        explain("Compare the best gain of a fixed intersection scope against backoff, for both protocols, then show the learned blend weights (the null result)."),
        code("""
for tag, label in [("", "conditioned"), ("blind", "blind")]:
    g = L.load_ladder_grid(tag)
    focus = g[g["routing"].isin(["M2_team", "M4_intersection", "M5_backoff"])].copy()
    best = focus.loc[focus.groupby(["retriever", "routing"])[metric].idxmax()]
    print(label)
    display(best.pivot_table(index="retriever", columns="routing", values=metric).round(4))
w = L.load_blend_weights("blend_eb")
display(pd.Series(w["weights"]).to_frame("learned_weight"))
"""),
        reading("""Backoff matches or beats a fixed intersection scope because it keeps the benefit where
fine-scoped evidence exists and degrades gracefully where it does not — the common case in an
imbalanced corpus. The learned multi-scope blend collapses onto a single scope (near 1.0 on the
intersection weight, about 0 on the rest), i.e. the model discovers that combining scopes is
unnecessary; we keep it as a documented null result rather than a method."""),

        section("The aggression tradeoff — why one prior cannot serve both protocols",
                "Plot the best achievable gain as a function of how aggressively the prior acts, for conditioned and ticket-only feedback.",
                "results/retriever_ladder{,_blind}/grid.csv", "This is the paper's central modeling statement."),
        explain("For each scaling magnitude, show the best conditioned and blind gain, illustrating the tradeoff."),
        code("""
rows = []
for tag, label in [("", "conditioned"), ("blind", "blind")]:
    g = L.load_ladder_grid(tag).copy()
    g["scale"] = g["lift"].str.split("|").str[1]
    for scale, part in g.groupby("scale"):
        rows.append({"protocol": label, "scale": scale, "best_gain": part[metric].max()})
tradeoff = pd.DataFrame(rows)
pivot = tradeoff.pivot(index="scale", columns="protocol", values="best_gain").sort_index()
display(pivot.round(4))
fig, ax = plt.subplots(figsize=(10, 5))
pivot.plot.bar(ax=ax, color={"conditioned": "#16856b", "blind": "#c44e52"})
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Best proxy gain", xlabel="Bonus scale (absolute or pool-std λ)",
                                           title="Aggression tradeoff across protocols")
plt.tight_layout(); L.savefig("04_aggression_tradeoff", run_ids=[]); plt.show()
"""),
        reading("""Large absolute bonuses are best for resolution-informed feedback but catastrophic for
ticket-only feedback; pool-relative bonuses of moderate size are safe under both, at some cost to
the conditioned optimum. There is no single aggressiveness that is optimal for both protocols. This
motivates two legitimate responses: pick a conservative prior when feedback quality is unknown, or
learn a per-ticket control policy that sets the aggressiveness (notebook 05)."""),

        section("Which lift formula? A systematic ablation",
                "Compare the Laplace family against the tanh and Bayesian-LCB formulas across routings and scaling, conditioned and blind.",
                "results/retriever_ladder_liftablation{,_blind}/grid.csv",
                "Offline proxy, single retriever (dense MiniLM); best-cell values are optimistic."),
        explain("Extract the best cell per lift family for both protocols and plot them."),
        code("""
def _family(frame):
    return (frame["lift"].str.split("|").str[0]
            .str.replace(r"_k[0-9.]+$", "", regex=True)
            .str.replace(r"_s[0-9.]+$", "", regex=True))

rows = []
for tag, protocol in [("liftablation", "conditioned"), ("liftablation_blind", "blind")]:
    grid = L.load_ladder_grid(tag)
    grid = grid[grid["routing"] != "none"].copy()
    grid["family"] = _family(grid)
    for fam, part in grid.groupby("family"):
        best = part.sort_values("minilm_d_proxy_top1", ascending=False).iloc[0]
        rows.append({"protocol": protocol, "family": fam, "best_gain": best["minilm_d_proxy_top1"],
                     "routing": best["routing"], "lift": best["lift"],
                     "ci_lo": best["primary_ci_lo"], "ci_hi": best["primary_ci_hi"],
                     "p": best["primary_wilcoxon_p"]})
ablation = pd.DataFrame(rows)
display(ablation.round(4))
fig, ax = plt.subplots(figsize=(9, 4.6))
sns.barplot(data=ablation, x="family", y="best_gain", hue="protocol",
            palette={"conditioned": "#16856b", "blind": "#c44e52"}, ax=ax)
ax.set(ylabel="Best offline proxy gain (top-1)", title="Lift-formula ablation")
plt.tight_layout(); L.savefig("04_lift_ablation", run_ids=[]); plt.show()
"""),
        reading("""Under both protocols the ordering is the same: empirical-Bayes centred Laplace is
best, tanh is a close second, legacy Laplace is third, and the Bayesian lower-confidence-bound variant
is last. The gaps are modest (conditioned 0.036–0.053; blind 0.005–0.011) but consistent, and the
formula is a free modelling choice. Legacy Laplace is weakest under blind feedback because its
zero-inflated evidence saturates the negative cap; tanh is more forgiving; EB centring dominates
because it removes the scope base-rate bias."""),

        section("Do the generated finalists confirm the prior modeling?",
                "Compare generated dev results for legacy Laplace, empirical-Bayes, and the calibrated backoff on both protocols, with independent metrics.",
                "results/rescored/method_comparison_v2.csv", "Single dev seed; eval and robustness in notebook 06."),
        explain("Print the generated finalist table with MiniLM, BGE, and BERTScore deltas and their p-values."),
        code("""
resc = L.load_rescore_comparison()
cols = ["run", "delta_cosine_mean", "delta_cosine_wilcoxon_p", "delta_cosine_bge_mean",
        "delta_cosine_bge_wilcoxon_p", "delta_bertscore_f1_mean", "delta_bertscore_f1_wilcoxon_p"]
display(resc[[c for c in cols if c in resc]].round(4))
"""),
        reading("""The generated runs reproduce the ladder's direction on both dev and the untouched
eval split: legacy Laplace is strongly negative under ticket-only feedback (−0.038 eval), the
calibrated prior removes that harm (+0.002 to +0.005, not individually significant), and
resolution-informed runs remain positive (+0.014 to +0.016). Independent metrics (BGE cosine, and
BERTScore for the eval runs) agree in sign, so the effect is not an artefact of the MiniLM family used
for retrieval. Magnitudes are small, which we state honestly."""),
        markdown("""## Recommended prior (method card, part 1)

Use the empirical-Bayes centered Laplace lift with a fixed cap, hierarchical backoff at minimum
evidence 2, and **pool-relative scaling** when the feedback's reliability is unknown. This is the
configuration carried into the control-policy notebook and the final results."""),
    ],
)


# ===========================================================================
# 05 — MODELING: the control policy
# ===========================================================================

NB05 = notebook(
    "Modeling the Control Policy (When to Intervene)",
    "MODELING",
    "pre-generation gating, the dev-to-eval methodology, policy value, and ceiling recovery",
    [
        markdown("""## The control problem

Even a calibrated prior has per-ticket variance: some tickets are helped, some harmed. A **control
policy** decides, before generating, whether to apply feedback to a given ticket. The train/dev/eval
split is designed for exactly this: feedback is *built* from train, every configuration is
*developed* on dev, patterns that hold on dev are *modeled*, and the resulting policy is *tested once*
on eval. This notebook builds and evaluates that policy on dev; notebook 06 confirms on eval."""),

        markdown("""## Why a control policy can only help under some conditions

A policy replaces feedback with the baseline on tickets it closes. If always-on has a positive mean,
closing tickets removes benefit on average, so a policy helps only if it can rank the **sign** of the
per-ticket effect better than chance. Formally, a policy beats always-on when the area under its
prediction curve exceeds a crossover near 0.5; at exactly chance it is worse, because the average
effect is positive. Under ticket-only feedback, always-on is *negative*, so closing harmful tickets
helps and the policy has much more room. This asymmetry is the key to everything below."""),

        section("What can the policy observe before generation?",
                "Only pre-generation features: retrieval confidence, margin, spread, overlap, lift statistics, evidence density, and query lengths. No reference reply, no generated answer.",
                "src/gate/features.py; results/blend_eb/gate_features_train.parquet",
                "Team/class identity is excluded so the policy is not dataset-specific."),
        explain("Load the gate feature table and show summary statistics for the features available at decision time."),
        code("""
feat = pd.read_parquet(L.RESULTS / "blend_eb" / "gate_features_train_M4_intersection.parquet")
cols = [c for c in GATE_FEATURE_COLS if c in feat.columns]
display(feat[cols].describe().T.round(3))
"""),
        reading("""The features are all quantities a retrieval system knows before calling the generator:
how confident the baseline is, how separated the top candidates are, how much feedback evidence
exists, and how large the potential bonus is. Because none of them is team- or class-specific, a
policy learned on them can transfer across organizations."""),

        section("Does a policy help under ticket-only feedback?",
                "Use the fixed general gate study (grouped CV by ticket, thresholds frozen on dev) and its counterfactual decomposition.",
                "results/gate_study_general/{learned_gate.csv,learned_gate_decomposition.csv}",
                "The definitive study; the earlier per-run pilot numbers were superseded by the methodology fix."),
        explain("Show the eval AUC, the dev-frozen policy value, and the counterfactual decomposition for the two blind Laplace configurations."),
        code("""
learned = L.load_gate_study_learned()
decomp = L.load_gate_study_decomposition()
blind = learned[(learned["protocol"] == "blind") & (learned["config_key"].str.contains("laplace"))]
display(blind[["config_key", "dev_threshold", "eval_auc", "eval_auc_ci_lower", "eval_auc_ci_upper",
               "always_on", "eval_policy", "eval_policy_ci_lower", "eval_policy_ci_upper",
               "pct_open", "gain_vs_always_on", "ceiling_recovery"]].round(4))
display(decomp[decomp["config_key"].str.contains("laplace")].round(4))
fig, ax = plt.subplots(figsize=(9, 4.6))
sub = learned[learned["config_key"].str.contains("laplace")].copy()
sub["label"] = sub["config_key"] + " / " + sub["protocol"]
sns.barplot(data=sub.melt(id_vars=["label"], value_vars=["always_on", "eval_policy", "oracle"]),
            x="label", y="value", hue="variable", ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean generated delta", title="Gate policy value (eval, dev-frozen threshold)")
plt.tight_layout(); L.savefig("05_policy_blind", run_ids=[]); plt.show()
"""),
        reading("""Under uncalibrated ticket-only feedback the gate clearly helps: always-on is −0.038
and the dev-frozen policy is +0.001 to +0.002, recovering 0.54–0.56 of the oracle ceiling. The
decomposition shows why: the gate closes 152–235 tickets whose counterfactual mean delta is
−0.07 to −0.10, i.e. a genuinely harmful subset. The AUC is modest (0.59–0.61), but because the
alternative is strongly negative even a modest ranking is valuable. This is the deployment setting
where the control policy earns its place."""),

        section("Does a policy help under resolution-informed feedback?",
                "Repeat on the conditioned configurations with the same fixed methodology.",
                "results/gate_study_general/learned_gate.csv",
                "Same features and labels; only the feedback protocol changes."),
        explain("Compare AUC, dev-frozen policy value and recovery across protocols."),
        code("""
sub = learned[learned["config_key"].str.contains("laplace")].copy()
sub["label"] = sub["config_key"] + " / " + sub["protocol"]
fig, ax = plt.subplots(figsize=(9, 4.6))
sns.barplot(data=sub, x="label", y="eval_auc", hue="protocol", ax=ax)
ax.axhline(0.5, color="black", ls="--", lw=1); ax.set(ylabel="Eval AUC", title="Gate predictability by protocol")
plt.tight_layout(); L.savefig("05_auc_by_protocol", run_ids=[]); plt.show()
display(sub[["label", "eval_auc", "always_on", "eval_policy", "gain_vs_always_on", "ceiling_recovery"]].round(4))
"""),
        reading("""Under resolution-informed feedback the gate is essentially neutral: the frozen
policy matches always-on within ±0.0005 (team AUC 0.56; intersection AUC 0.66 but no recoverable
headroom because always-on is already positive). The decomposition shows the closed set is mostly
no-op tickets with a slightly positive mean, so closing them cannot help. The honest conclusion is
that a control policy is justified only when feedback reliability is low and harm is concentrated."""),

        section("Is a learned policy worth it, or does a simple rule suffice?",
                "Compare the learned gate against the best single-feature static rule derived on dev.",
                "results/gate_study_general/{static_gate.csv,static_gate_eval.csv}",
                "Rules are interpretable; the comparison bounds the value of learning."),
        explain("Show the best static rule on dev and its eval policy per configuration."),
        code("""
static = L.load_gate_study_static(); static_eval = L.load_gate_study_static_eval()
display(static.head(5).round(4))
display(static_eval.round(4))
"""),
        reading("""The best dev rule opens feedback when the weakest of the baseline top-5 similarities is
below ~0.95 (dev policy +0.0096, 80% open). On eval it is positive on the calibrated blind
configuration (+0.003) but clearly below the learned gate's recovery on the uncalibrated blind
configurations, where the learned gate closes a sharply harmful subset. A simple rule is a useful
interpretable fallback; the learned gate adds real value only when the harm is concentrated and
multi-feature."""),

        section("Does calibration already do the policy's job?",
                "Compare the gate on the uncalibrated (legacy Laplace) prior against the calibrated (EB) prior.",
                "results/gate_study_general/{learned_gate.csv,learned_gate_decomposition.csv}",
                "Same features; different prior behind the feedback."),
        explain("Show the decomposition for the calibrated blind configuration next to the uncalibrated ones."),
        code("""
show = decomp[decomp["protocol"] == "blind"].copy()
display(show[["config_key", "dev_threshold", "n_open", "n_closed", "mean_delta_closed", "harm_rate_closed",
              "policy_value", "gain_vs_always_on"]].round(4))
"""),
        reading("""On the uncalibrated prior the gate closes a strongly harmful subset (mean −0.07 to
−0.10) and recovers most of the ceiling. On the calibrated prior (backoff EB) the gate closes tickets
that are 62% no-ops (mean +0.006), so it shaves a small positive tail and loses −0.003. Calibration
and control are therefore **substitutes**: either discipline the prior or gate a raw one. We recommend
calibration as the default (it needs no labels) and keep the policy for settings where the prior
cannot be trusted."""),
        markdown("""## Conclusion — the control policy

A pre-generation policy is valuable exactly when feedback is unreliable and harm is concentrated.
Under uncalibrated ticket-only feedback the gate closes a sharply harmful subset and recovers 0.54–0.56
of the oracle ceiling; once the prior is calibrated, or under resolution-informed feedback, the closed
set is mostly no-ops and the gate is neutral. The next notebook reports the final results and the claim
ledger; notebook 07 generalises the gate across all signals and replaces sign classification with
expected-value modelling."""),
    ],
)


# ===========================================================================
# 06 — RESULTS and claims
# ===========================================================================

NB06 = notebook(
    "Final Results and Claims",
    "RESULTS",
    "the locked configuration, dev and eval results, robustness, independent metrics, and the claim ledger",
    [
        markdown("""## Locked configuration

Chosen on dev before the eval split was touched:

- **Scope:** hierarchical backoff (intersection → team → class → global), minimum evidence 2; fine
  fixed scopes (team, team∩class) as the resolution-informed optimum.
- **Prior:** empirical-Bayes centered Laplace, κ=2, cap ±0.20.
- **Scaling:** pool-relative (λ=0.5) for the ticket-only setting; absolute for resolution-informed.
- **Control policy:** a pre-generation gate over pool-distribution and cosine features, used only
  where harm is concentrated; expected-value (magnitude) modelling when a per-ticket action is needed.

Reported alongside: legacy Laplace fine-scope feedback as the resolution-informed optimum, and the
ticket-only results as the reliability lower bound. All generated dev/eval runs share one generation
regime (model + system prompt + cache); the regenerated dev runs carry an explicit regime id and
warm-cache flag in their summaries."""),

        section("What are the generated dev results across all methods and both protocols?",
                "Assemble the canonical dev runs into one table and show the protocol reversal.",
                "results/<run>/*_summary.json", "All rows share one generation regime; eval is reported next."),
        explain("Build the conditioned-vs-blind dev table from the canonical runs and plot it."),
        code("""
CANON = {
    "M1 global": {"conditioned": "M1_global_dev_conditioned_continuous", "blind": "M1_global_dev_blind_continuous"},
    "M2 team": {"conditioned": "M2_team_dev_conditioned_continuous", "blind": "M2_team_dev_blind_continuous"},
    "M3 class": {"conditioned": "M3_class_dev_conditioned_continuous", "blind": "M3_class_dev_blind_continuous"},
    "M4 intersection": {"conditioned": "M4_intersection_dev_conditioned_continuous", "blind": "M4_intersection_dev_blind_continuous"},
    "M5 backoff (EB)": {"conditioned": "M5_backoff_dev_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2",
                       "blind": "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"},
}
rows = []
for method, by_protocol in CANON.items():
    for protocol, folder in by_protocol.items():
        summary = L.load_summary(folder)
        rows.append({"method": method, "protocol": protocol, "n": summary["total_valid"],
                     "mean_delta_cosine": summary["metrics"]["mean_delta_cosine"]})
dev_table = pd.DataFrame(rows)
display(dev_table.pivot(index="method", columns="protocol", values="mean_delta_cosine").round(4))
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.barplot(data=dev_table, x="method", y="mean_delta_cosine", hue="protocol",
            palette={"conditioned": "#16856b", "blind": "#c44e52"}, ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Generated-answer cosine delta",
                                           title="Conditioned vs blind (dev, same generation regime)")
plt.tight_layout(); L.savefig("06_dev_protocol_reversal", run_ids=[]); plt.show()
"""),
        reading("""The table is the paper's headline motif, now measured entirely within one generation
regime: resolution-informed fine-scope feedback is positive (team +0.022, intersection +0.031) while
the same methods are negative under ticket-only feedback (team −0.034, intersection −0.051). Broad
scopes (global, class) are negative under both protocols, and the calibrated empirical-Bayes prior is
mildly positive under both (+0.005 conditioned, +0.010 blind). The sign of the effect is controlled by
how the feedback was produced, not by the routing alone."""),

        section("Do the results hold on the untouched eval split?",
                "Summaries for the eval runs across protocols and the locked configuration.",
                "results registry (eval runs)", "Eval is reported once; no tuning."),
        explain("Load eval-run summaries from the registry and print them with their headline deltas."),
        code("""
reg = L.registry()
eval_rows = reg[reg["out_dir"].str.contains("_eval_", na=False)][["script", "out_dir", "headline_metric", "headline_value"]]
display(eval_rows.tail(12).to_string(index=False))
"""),
        reading("""On eval the reversal repeats: resolution-informed fine-scope feedback is positive
(+0.0138 team, +0.0151 intersection; the gated intersection run reaches +0.0164) and ticket-only
feedback is negative (−0.038). The calibrated ticket-only configuration is near zero (+0.002) and the
unseen-procedure (disjoint) variant is mildly positive (+0.005); neither is individually significant.
The eval split confirms the dev conclusions without any tuning."""),

        section("Do independent metrics agree?",
                "MiniLM cosine, BGE cosine, ROUGE-L, and BERTScore deltas for the finalists.",
                "results/rescored/method_comparison_v2.csv", "BGE is an independent embedding family."),
        explain("Print the multi-metric table and plot the metric agreement."),
        code("""
resc = L.load_rescore_comparison()
cols = [c for c in ["run", "delta_cosine_mean", "delta_cosine_bge_mean", "delta_rouge_l_mean", "delta_bertscore_f1_mean"] if c in resc]
display(resc[cols].round(4))
long = resc.melt(id_vars="run", value_vars=[c for c in cols if c != "run"], var_name="metric", value_name="delta")
fig, ax = plt.subplots(figsize=(12, 5)); sns.barplot(data=long, x="run", y="delta", hue="metric", ax=ax)
ax.axhline(0, color="black", lw=1); ax.tick_params(axis="x", rotation=60); ax.set(ylabel="Mean delta", title="Independent-metric agreement")
plt.tight_layout(); L.savefig("06_independent_metrics", run_ids=[]); plt.show()
"""),
        reading("""The independent metrics move in the same direction as the retrieval-family cosine, so
the effects are not an artefact of using the same model family for retrieval and scoring. The gated
resolution-informed intersection run is the strongest independent confirmation: BGE +0.0130 (p=.0002)
and BERTScore +0.0160, with cosine +0.0164 (p=.028). Magnitudes are small throughout; BERTScore is
computed for the eval and gated runs only (dev rows show BGE/ROUGE, computed in the combined rescore)."""),

        section("Do independent LLM judges agree?",
                "Pairwise answer quality from a non-OpenAI judge (Claude Sonnet 5) on a 100-ticket eval subsample, both orders, win only when consistent.",
                "results/answer_judge/{summary.json,scores.csv}", "Oracle-conditioned judge: it sees the reference reply."),
        explain("Show net win rate, position consistency, and agreement with the cosine sign."),
        code("""
judge = L.load_answer_judge_summary()
jrows = [{"run": run, **stats} for run, stats in judge["runs"].items()]
jtable = pd.DataFrame(jrows)
display(jtable[["run", "n", "n_identical", "feedback_wins", "baseline_wins",
                "net_win_rate", "position_consistency", "agreement_with_cosine_sign"]].round(3))
"""),
        reading("""The independent judge confirms the direction of the resolution-informed effect
(intersection net win rate +0.31; gated +0.31, i.e. the gate changes almost nothing there) and sees
essentially no difference under calibrated ticket-only feedback (+0.03 ungated; −0.05 gated with 58%
identical answers). Two caveats are reported rather than hidden: the judge flips order in ~25–30% of
pairs, and per-ticket agreement with the embedding-metric sign is low (0.35–0.40) even where the
aggregate direction agrees. The judge is oracle-conditioned (it sees the reference reply), so it is an
upper-bound evaluator, consistent with the conditioned feedback protocol."""),

        section("Does the conditioned effect survive a different generator?",
                "Repeat the eval finalists with a non-OpenAI generator (Gemini 3.8 Flash) on the same retrieval and feedback.",
                "results/*_gengemini38flash/*_summary.json",
                "A different generation regime by design; compared within itself."),
        explain("Compare gemini-generated deltas with the luna runs on the same tickets."),
        code("""
pairs = [
    ("M2 conditioned", "M2_team_eval_conditioned_continuous", "M2_team_eval_conditioned_continuous_gengemini38flash"),
    ("M4 conditioned", "M4_intersection_eval_conditioned_continuous", "M4_intersection_eval_conditioned_continuous_gengemini38flash"),
    ("M5 blind (calibrated)", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2",
     "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gengemini38flash"),
]
rows = []
for label, luna_folder, gem_folder in pairs:
    luna = L.load_summary(luna_folder)["metrics"]["mean_delta_cosine"]
    gem = L.load_summary(gem_folder)["metrics"]["mean_delta_cosine"]
    rows.append({"config": label, "luna": luna, "gemini-3.8-flash": gem})
cross = pd.DataFrame(rows)
display(cross.round(4))
fig, ax = plt.subplots(figsize=(9, 4.4))
sns.barplot(data=cross.melt(id_vars="config", var_name="generator", value_name="delta"),
            x="config", y="delta", hue="generator", ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean generated-answer cosine delta",
                                           title="Cross-generator robustness")
plt.tight_layout(); L.savefig("06_cross_generator", run_ids=[]); plt.show()
"""),
        reading("""The conditioned effect is not a luna artefact: with Gemini 3.8 Flash as the generator
it is larger (M2 +0.029 vs +0.014; M4 +0.040 vs +0.015), and the calibrated blind configuration is
mildly positive (+0.007 vs +0.002). This is a different generation regime by construction (different
model and cache), so it is compared only within itself, and it addresses the same-model
judge/generator tie: the feedback judge never sees generated answers, and a second generator family
reproduces the direction."""),

        section("Is the locked configuration robust to seeds and unseen procedures?",
                "Multi-seed dev runs and the disjoint eval run for the locked configuration.",
                "results/M5_backoff_*seed*/_summary.json; results/M5_backoff_*disjoint*/_summary.json",
                "Feedback source shifts slightly across seeds by construction."),
        explain("Print the seed and disjoint summaries for the calibrated blind winner and the conditioned finalists."),
        code("""
import json
rows = []
patterns = ["M5_backoff_*seed*", "M5_backoff_*disjoint",
            "M2_team_*seed*", "M2_team_*disjoint", "M4_intersection_*seed*", "M4_intersection_*disjoint"]
for pat in patterns:
    for folder in sorted(L.RESULTS.glob(pat)):
        files = sorted(folder.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            continue
        s = json.loads(files[-1].read_text(encoding="utf-8"))
        rows.append({"run": s["experiment_id"], "n": s["total_valid"],
                     "mean_delta_cosine": s["metrics"]["mean_delta_cosine"]})
display(pd.DataFrame(rows).sort_values("run").round(4))
"""),
        reading("""Across seeds the conditioned effect is direction-consistent but smaller than on seed
42: intersection +0.015 (seed 123) and +0.013 (seed 456), team +0.006/−0.001; on the
procedure-disjoint split the conditioned effect persists (intersection +0.015, team +0.010). The
calibrated blind winner is small and mixed in sign across seeds and mildly positive on unseen
procedures. The honest reading is that the resolution-informed effect is the reliable one; the
ticket-only calibrated result is a *de-risking* of feedback rather than a large gain."""),

        section("What is the claim ledger?",
                "Every paper claim mapped to the notebook section and artifact that supports it, with readiness.",
                "all notebook artifacts", "Claims marked pending stay conditional."),
        explain("Print the ledger tying each claim to its evidence."),
        code("""
status = L.artifact_status().set_index("section")["available"].to_dict()
claims = pd.DataFrame([
    ["Feedback benefit is conditional on how it is produced (protocol reversal)", "06 / results", "canonical dev table; eval registry", True],
    ["Fine scopes help; broad pooling harms", "03 / analysis", "retriever_ladder grids", status.get("Granularity: ladder", False)],
    ["Judge scores are zero-inflated; naive Laplace lift saturates and harms under realistic feedback", "04 / modeling", "feedback_calibration/saturation.csv", status.get("Validity: calibration", False)],
    ["Empirical-Bayes centering + pool-relative scaling removes the harm (gains not individually significant)", "04 / modeling", "rescored/method_comparison_v2.csv", status.get("Validity: rescoring", False)],
    ["Lift-formula ordering is stable: EB > tanh > Laplace > LCB", "04 / modeling", "retriever_ladder_liftablation{,_blind}/grid.csv", (L.RESULTS / "retriever_ladder_liftablation" / "grid.csv").exists()],
    ["The gate is risk control: recovers 0.54–0.56 of the ceiling where harm is concentrated, neutral otherwise", "05 / modeling", "gate_study_general/learned_gate{,_decomposition}.csv", (L.RESULTS / "gate_study_general" / "learned_gate_decomposition.csv").exists()],
    ["Expected-value modelling beats sign-only gating; magnitude action selection is positive", "07 / modeling", "magnitude_policy/policy_table.csv", (L.RESULTS / "magnitude_policy" / "policy_table.csv").exists()],
    ["Independent metrics and an independent LLM judge agree in direction on the conditioned effect", "06 / results", "rescored/method_comparison_v2.csv; answer_judge/summary.json", status.get("Pairwise judge", False)],
    ["Negative results: semantic filter, blend null, sign-only multi-action, retriever transfer", "03/07", "ladder grids; multi_action.csv", True],
], columns=["claim", "section", "artifact", "ready"])
display(claims)
"""),
        markdown("""## Method card and limitations

**Recommended configuration.** Empirical-Bayes centered Laplace lift, hierarchical backoff
(minimum evidence 2), pool-relative scaling when feedback reliability is unknown; a pre-generation
policy only where harm is concentrated (uncalibrated or unreliable feedback); expected-value
modelling when a per-ticket action must be chosen.

**When it helps.** On moderately uncertain retrievals with trustworthy (resolution-informed)
feedback; the effect reverses when feedback is ticket-only and unreliable, where calibration is the
safer response than gating.

**Limitations.** One organizational corpus (1,595 tickets; dev 319, eval 398); feedback is
LLM-judge-simulated, not human (the conditioned protocol is oracle-informed and simulates an expert
with access to the historical resolution, so it is an upper bound); the judge and the primary
generator are the same model, mitigated only by a second generator in the robustness runs; generated
evidence is concentrated on seed 42, with multi-seed/disjoint coverage for the calibrated blind
winner; ticket-only gains are small and not individually significant; the offline proxy is valid for
configuration selection only; per-ticket agreement between the embedding metric and the LLM judge is
low even when aggregate directions agree."""),
    ],
)


# ===========================================================================
# 07 — MODELING: the general gate (all signals, dev -> eval)
# ===========================================================================

NB07 = notebook(
    "Modeling a General Benefit Gate",
    "MODELING",
    "one gate across all routing signals and both protocols, trained on dev and applied to eval",
    [
        markdown("""## The general-gate idea

The previous notebook gated one configuration at a time. Here we ask a stronger question:
is there a **single, general pattern** that predicts whether *any* feedback configuration will
help a ticket? We treat every *(ticket, configuration)* pair as one sample, describe the pool
that the configuration would produce, and predict the generated-answer benefit. The gate is
trained on the **dev** pairs (all routing signals, both feedback protocols) and applied
unchanged to the **eval** pairs. If a feature carries a genuine signal it must work across
signals, not just for the configuration that happened to do best on average."""),

        section("What does a gate see for a (ticket, configuration) pair?",
                "Pool-distribution features (how much the configuration can move the ranking), cosine-correlation features (query-candidate semantics and retrieval-feedback agreement), and routing evidence per scope. No reference reply is used.",
                "results/gate_study_general/features_{dev,eval}.parquet",
                "These features are all computable before generation."),
        explain("Load the dev feature table, show its shape, and summarise the pool, cosine and routing families."),
        code("""
dev = L.load_gate_study_features("dev")
print("dev samples:", dev.shape, " configs:", sorted(dev["config_key"].unique()))
pool_cols = ["lift_max", "lift_std", "n_promotable", "best_promotion_margin", "n_top_changed", "score_lift_corr", "intervention_scale"]
cos_cols = ["semantic_top1", "semantic_mean", "semantic_std", "semantic_lift_corr"]
route_cols = ["coverage_intersection", "coverage_team", "max_lift_intersection", "max_lift_team", "evidence_intersection_top5"]
display(dev[pool_cols + cos_cols + route_cols].describe().T.round(4))
"""),
        reading("""Every pool is described by how large its possible lift is and how many candidates
could actually be promoted, by how similar the query and its candidates are, and by how much
evidence each scope holds. `n_promotable` and `best_promotion_margin` are the "how much can this
pool move" quantities; `score_lift_corr` and `semantic_lift_corr` measure whether feedback agrees
with similarity; `intervention_scale` records how large the intervention was."""),

        section("Which features actually correlate with real benefit on dev?",
                "Correlate each feature with the generated delta across all dev (ticket, configuration) pairs.",
                "results/gate_study_general/features_dev.parquet", "Correlation is descriptive and pools protocols."),
        explain("Compute Pearson correlation of each feature with the generated delta and rank them."),
        code("""
from scipy.stats import spearmanr
feat_cols = [c for c in dev.columns if c not in ("config_key", "protocol", "agg", "split", "ticket_id", "delta")]
rows = []
for c in feat_cols:
    x = dev[c].to_numpy(float); y = dev["delta"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.std(x[ok]) == 0:
        continue
    rows.append({"feature": c, "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                 "spearman": float(spearmanr(x[ok], y[ok]).statistic)})
corr = pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
display(corr.head(15).round(3))
fig, ax = plt.subplots(figsize=(10, 5)); top = corr.head(12).iloc[::-1]
ax.barh(top["feature"], top["spearman"], color="#2a6fbb"); ax.axvline(0, color="black", lw=1)
ax.set(xlabel="Spearman correlation with generated delta", title="Which features carry a general signal")
plt.tight_layout(); L.savefig("07_feature_signal", run_ids=[]); plt.show()
"""),
        reading("""The features that correlate most strongly are not the raw retrieval scores but the
intervention-shape quantities: how many candidates the configuration would promote, how large the
best promotion margin is, and whether feedback agrees with similarity. In other words, the general
signal is *how the pool is reshaped*, not how similar the query and its top candidate are. This is
the pattern the user asked us to look for, and it holds across signals."""),

        section("Static threshold gate: the best single rule derived on dev",
                "Sweep single-feature thresholds on dev and pick the rule with the best dev policy value; freeze it and apply to eval.",
                "results/gate_study_general/static_gate.csv", "One rule for all signals; interpretable baseline."),
        explain("Show the best static rules on dev, then their eval policy value per configuration."),
        code("""
static = L.load_gate_study_static()
display(static.head(8).round(4))
"""),
        reading("""The best single rule on dev opens feedback only where the weakest of the baseline
top-5 similarities is below ~0.95 — i.e. where the pool is weakly aligned and feedback has room to
help. It is a single interpretable rule that applies to every signal, and its dev policy value is
positive (+0.0096 at 80% open). The next cell shows how it transfers to eval."""),

        section("Learned general gate: trained on dev, applied to eval",
                "Logistic gate on the deployable features across all dev (ticket, configuration) pairs, grouped CV by ticket, thresholds frozen on dev; applied unchanged to eval.",
                "results/gate_study_general/learned_gate.csv; summary.json", "Eval is scored once; no threshold tuning on eval."),
        explain("Show the grouped dev out-of-fold AUC with its clustered CI and the eval AUC, dev-frozen policy value and ceiling recovery per configuration."),
        code("""
learned = L.load_gate_study_learned()
summary = L.load_gate_study_summary()
print("dev out-of-fold AUC (grouped CV by ticket): %.3f [%.3f, %.3f]" % (
    summary["dev_deployable_oof_auc"], summary["dev_deployable_oof_auc_ci_lower"],
    summary["dev_deployable_oof_auc_ci_upper"]))
display(learned[["config_key", "protocol", "n_dev_tickets", "dev_threshold", "dev_policy", "n_eval",
                 "eval_auc", "eval_auc_ci_lower", "eval_auc_ci_upper", "always_on", "eval_policy",
                 "eval_policy_ci_lower", "eval_policy_ci_upper", "pct_open", "gain_vs_always_on",
                 "ceiling_recovery"]].round(4))
fig, ax = plt.subplots(figsize=(11, 5))
ev = learned.dropna(subset=["eval_auc"])
sns.barplot(data=ev, x="config_key", y="eval_auc", hue="protocol", ax=ax)
ax.axhline(0.5, color="black", ls="--", lw=1); ax.set(ylabel="Eval AUC", title="General gate: eval predictability by protocol")
plt.tight_layout(); L.savefig("07_general_gate_auc", run_ids=[]); plt.show()
"""),
        reading("""A single gate trained on dev across every routing signal reaches a grouped out-of-fold
AUC of 0.699 [0.676, 0.721] — folds are assigned by ticket, so no ticket appears in both train and
validation, and the interval resamples tickets. On eval it stays above chance for every configuration
(0.56–0.74). Crucially it was never fit to a single configuration, so this is a general benefit
signal. Under uncalibrated ticket-only feedback it opens selectively and recovers 0.54–0.56 of the
oracle ceiling (turning a strongly negative always-on into a small positive). Under resolution-informed
feedback, where always-on is already positive, the gate matches it within ±0.0005; on the calibrated
ticket-only prior it closes mostly no-op tickets and loses −0.003. The decomposition in the next cell
explains exactly why."""),

        section("Why does the gate help — or hurt? The open/closed decomposition",
                "For each configuration, compare the counterfactual mean delta and harm rate of the tickets the frozen gate closes versus the ones it keeps.",
                "results/gate_study_general/learned_gate_decomposition.csv",
                "Counterfactual: a closed ticket would have kept the ungated feedback delta."),
        explain("Show the decomposition table and plot the closed-set mean delta."),
        code("""
decomp = L.load_gate_study_decomposition()
display(decomp[["config_key", "protocol", "dev_threshold", "n_open", "n_closed", "pct_open",
                "mean_delta_all", "mean_delta_open", "mean_delta_closed", "harm_rate_open",
                "harm_rate_closed", "gain_vs_always_on"]].round(4))
fig, ax = plt.subplots(figsize=(11, 4.6))
d = decomp.copy(); d["label"] = d["config_key"] + " / " + d["protocol"]
sns.barplot(data=d, x="label", y="mean_delta_closed", hue="protocol", ax=ax)
ax.axhline(0, color="black", lw=1); ax.tick_params(axis="x", rotation=25)
ax.set(ylabel="Counterfactual mean delta of closed tickets", title="What the gate removes")
plt.tight_layout(); L.savefig("07_gate_decomposition", run_ids=[]); plt.show()
"""),
        reading("""The decomposition is the honest explanation of the gate's value. On the uncalibrated
ticket-only configurations the gate closes 152–235 tickets whose counterfactual mean delta is
−0.07 to −0.10 (56–60% of them harmful): it removes real harm. On the calibrated ticket-only prior it
closes 220 tickets that are 62% no-ops with a mean of +0.006: it removes a small positive tail and
loses. Under resolution-informed feedback the closed sets are 55–90% no-ops with a mean near +0.003:
neutral. A gate can only help when the harm is concentrated enough to be identified; calibration
removes the concentration, so calibration and gating are substitutes."""),

        section("How much does the reference-reply proxy add (oracle diagnostic)?",
                "Repeat the gate with the offline proxy delta added as a feature; this uses the reference reply and is not deployable.",
                "results/gate_study_general/learned_gate_proxy.csv", "Diagnostic upper bound only."),
        explain("Compare the deployable gate against the proxy-augmented gate on eval."),
        code("""
proxy = L.load_gate_study_proxy()
cmp = learned.dropna(subset=["ceiling_recovery"]).merge(
    proxy[["config_key", "protocol", "ceiling_recovery"]], on=["config_key", "protocol"], suffixes=("_deployable", "_proxy"))
display(cmp[["config_key", "protocol", "eval_auc", "always_on", "oracle", "eval_policy",
             "ceiling_recovery_deployable", "ceiling_recovery_proxy"]].round(4))
"""),
        reading("""Adding the reference-reply proxy raises the recoverable fraction only modestly, which
is consistent with the earlier proxy-validity finding: the proxy is a weak per-ticket signal. The
practical consequence is that a deployable gate — one that never sees the answer — already captures
most of what is recoverable, so the oracle proxy is not worth its leakage."""),

        section("Multi-action selection: choosing a configuration per ticket",
                "Let the gate pick, per eval ticket, the configuration with the highest predicted benefit.",
                "results/gate_study_general/multi_action.csv", "Exploratory; reported honestly even though it does not win."),
        explain("Show the multi-action policy value against always-on-best-fixed, never-on, and the oracle action."),
        code("""
multi = L.load_gate_study_multi(); display(multi.round(4))
"""),
        reading("""Selecting the highest-probability configuration per ticket does *not* beat simply
using the best fixed configuration: the gate's probabilities are not calibrated to effect
magnitude, so the selector sometimes chooses an intervention that is unlikely to help much. The
oracle action (choose the best configuration with hindsight) is much higher, showing the headroom
exists but is not reachable with sign-only predictions. This is a documented negative result; the
next section replaces it with expected-value action selection."""),

        section("Magnitude-aware policy: expected value instead of probability",
                "Regress the expected delta (HistGradientBoosting) on the same features with grouped CV, freeze a cost cutoff on dev, and choose actions by expected value.",
                "results/magnitude_policy/{policy_table.csv,decomposition.csv,action_selection.csv}",
                "No API calls; policy values are reconstructible from the stored deltas."),
        explain("Compare the sign gate, the expected-value GBR and Ridge, then show the action-selection comparison."),
        code("""
mag = L.load_magnitude_policy(); mag_summary = L.load_magnitude_summary()
print("dev OOF quality (GBR):", {k: round(v, 3) for k, v in mag_summary["quality"]["gbr"].items()})
display(mag[["config_key", "protocol", "policy", "threshold", "always_on", "eval_policy",
             "eval_policy_ci_lower", "eval_policy_ci_upper", "pct_open", "gain_vs_always_on",
             "ceiling_recovery"]].round(4))
actions = L.load_magnitude_actions(); display(actions.round(4))
fig, ax = plt.subplots(figsize=(10, 4.4))
sns.barplot(data=actions, x="policy", y="mean_delta", ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean achieved delta", title="Action selection (eval)")
plt.tight_layout(); L.savefig("07_magnitude_action", run_ids=[]); plt.show()
"""),
        reading("""The expected-delta model explains a substantial share of dev variance (R² ≈ 0.47,
dominated by between-configuration differences; within-configuration rank correlation ρ ≈ 0.17) and
beats the sign gate on every blind configuration: intersection +0.0074, team +0.0024, backoff +0.0035,
versus +0.0023/+0.0010/−0.0011 for the sign gate. The reason is visible in the decomposition: it
closes much smaller and far more harmful subsets (46–66 tickets with means of −0.28 to −0.35). Under
resolution-informed feedback it is neutral-to-slightly-positive (+0.0008 to +0.0013). Action selection
by expected value is now positive: +0.0116 [+0.0034, +0.0205] with 19% abstention, versus −0.0079 for
the sign-based selector and +0.0002 for the best fixed configuration (oracle +0.0571). The headroom is
still large; the modelling direction, however, is settled: predict magnitude, not sign."""),

        section("Turning the gate on: live gated evaluation",
                "So far the gate's value was reconstructed post-hoc (a closed gate returns the baseline, delta 0). Here we document and run the *live* pipeline, where the gate actually decides during retrieval.",
                "results/*_gated/*_summary.json; results/gate_study_general/gate_model.joblib",
                "A live run verifies the integration; it is required when the gate selects among configurations."),
        explain("Explain the live path and show the live gated run summaries when they exist."),
        code("""
gated = L.load_gated_runs()
if gated.empty:
    display(Markdown("**PENDING:** no live gated runs yet. Produce them with, e.g.\\n"
        "`python experiments/04_evaluate.py --method M5_backoff --lift laplace_eb --prior-strength 2 "
        "--scale-mode pool_std --pool-lambda 0.5 --min-evidence 2 --feedback-protocol blind "
        "--gating-model results/gate_study_general/gate_model.joblib --gating-threshold 0.3`"))
else:
    display(gated.round(4))
"""),
        reading("""The live gated run is the deployed artifact: at inference the pipeline computes the
same pool-distribution and cosine features, evaluates the saved gate, and if the probability of help
is below the dev-chosen threshold it keeps the baseline ranking (no feedback). Because the generation
cache holds both prompt branches, these runs cost almost nothing. The live numbers match the
dev-frozen post-hoc policy values (e.g. intersection conditioned eval +0.0164 live vs +0.0146
post-hoc; calibrated blind +0.0014 vs −0.0011), which is the integration check. Independent metrics
confirm the conditioned gated run: BGE +0.0130 (p=.0002), BERTScore +0.0160, cosine +0.0164 (p=.028);
the calibrated blind gated run remains null (+0.0014 cosine, p=.74). The model file
`gate_model.joblib` and the thresholds in `gate_meta.json` are the frozen artifacts; they are fit on
dev and applied to eval unchanged."""),
        markdown("""## Conclusion — the general gate

A single gate trained across all routing signals and both protocols learns a general benefit signal
(grouped dev OOF AUC 0.699 [0.676, 0.721]) and transfers to eval above chance for every configuration.
It is risk control, not gain: it recovers 0.54–0.56 of the ceiling where harm is concentrated
(uncalibrated ticket-only feedback) and is neutral once calibration has removed that concentration.
The open/closed decomposition explains why, and expected-value (magnitude) modelling — not sign
classification — is the right layer when a per-ticket action must be chosen: it closes smaller, more
harmful subsets and turns action selection positive (+0.0116 [+0.0034, +0.0205])."""),
])


OUTPUTS = {
    "02_protocol_and_validity.ipynb": NB02,
    "03_when_feedback_helps.ipynb": NB03,
    "04_modeling_the_prior.ipynb": NB04,
    "05_modeling_the_control_policy.ipynb": NB05,
    "06_final_results_and_claims.ipynb": NB06,
    "07_general_gate.ipynb": NB07,
}


if __name__ == "__main__":
    for name, payload in OUTPUTS.items():
        path = HERE / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")