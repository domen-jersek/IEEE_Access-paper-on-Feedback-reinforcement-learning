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
        reading("""Weaker retrievers (BM25) benefit most from feedback — feedback partly compensates for weak
lexical retrieval — while stronger retrievers gain less but still positively. This is a useful
deployment property: the method is most valuable exactly where retrieval alone is not enough."""),

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
fine-scope metadata already captures that distinction. The hybrid-retriever run is significantly
negative, showing the calibrated blind gain found on dense MiniLM does not transfer to a hybrid
retriever. Both results are reported to bound the method's scope rather than hidden."""),
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
        reading("""The generated runs reproduce the ladder's direction: legacy Laplace is strongly
negative under ticket-only feedback, the calibrated prior removes that harm and is mildly positive,
and resolution-informed runs remain positive. Independent metrics (BGE cosine, BERTScore) agree in
sign, so the effect is not an artefact of the MiniLM family used for retrieval. Magnitudes are
small and not individually significant on dev, which we state honestly."""),
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
                "Train a logistic policy on generated dev labels with out-of-fold probabilities and evaluate its policy value.",
                "results/gate_pilot_blind/{gate_pilot.csv,policy.csv}", "Pilot on dev; eval confirmation follows."),
        explain("Show the out-of-fold AUC and, for each threshold, the mean generated delta under the policy, alongside always-on, never-on, and oracle."),
        code("""
pilot = L.load_gate_pilot("blind"); policy = L.load_gate_pilot_policy("blind")
display(pilot[["run", "n", "oof_auc", "mean_delta_always_on", "mean_delta_oracle", "ceiling_recovery"]].round(4))
display(policy.round(4))
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=policy[policy["policy"].isin(["learned", "always_on", "oracle"])],
             x="threshold", y="mean_delta", hue="run", style="policy", marker="o", ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Mean generated delta under policy", title="Policy value (ticket-only feedback)")
plt.tight_layout(); L.savefig("05_policy_blind", run_ids=[]); plt.show()
"""),
        reading("""Under ticket-only feedback the policy clearly helps: always-on is strongly negative,
while opening only the upper portion of predicted tickets recovers a substantial part of the oracle
ceiling (roughly 0.6–0.7 in the earlier analysis). The AUC is modest (about 0.63–0.71), but because
the alternative is strongly negative, even a modest ranking is valuable. This is the deployment
setting where the control policy earns its place."""),

        section("Does a policy help under resolution-informed feedback?",
                "Repeat the same OOF policy evaluation on the resolution-informed runs.",
                "results/gate_pilot_conditioned/gate_pilot.csv", "Same features and label; only the feedback protocol changes."),
        explain("Print the conditioned policy summary and compare its AUC and ceiling recovery to the blind case."),
        code("""
cond = L.load_gate_pilot("conditioned")
display(cond[["run", "n", "oof_auc", "mean_delta_always_on", "mean_delta_oracle", "ceiling_recovery"]].round(4))
focus = pd.concat([pilot.assign(protocol="blind"), cond.assign(protocol="conditioned")])
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.barplot(data=focus, x="run", y="oof_auc", hue="protocol", ax=ax)
ax.axhline(0.5, color="black", ls="--", lw=1); ax.set(ylabel="Out-of-fold AUC", title="Policy predictability by protocol")
plt.tight_layout(); L.savefig("05_auc_by_protocol", run_ids=[]); plt.show()
"""),
        reading("""Under resolution-informed feedback the AUC is near chance (about 0.53–0.56) and the
policy cannot beat always-on. There is nothing to fix: the prior is already well-behaved and the
remaining per-ticket variance is not predictable from pre-generation features. The honest conclusion
is that a control policy is justified only when feedback reliability is low."""),

        section("Is a learned policy worth it, or does a simple rule suffice?",
                "Compare the best learned threshold policy against the best single-feature rule on the same tickets.",
                "results/gate_pilot_*/rules.csv", "Rules are interpretable; the comparison bounds the value of learning."),
        explain("For each run, show the best learned policy value and the best simple rule value side by side."),
        code("""
rows = []
for tag in ["blind", "conditioned"]:
    pol = L.load_gate_pilot_policy(tag); rl = L.load_gate_pilot_rules(tag)
    learned = pol[pol["policy"] == "learned"].groupby("run")["mean_delta"].max()
    rules = rl.groupby("run")["mean_delta"].max()
    for run in learned.index:
        rows.append({"run": run, "protocol": tag, "best_learned": learned[run],
                     "best_rule": rules.get(run, np.nan)})
cmp = pd.DataFrame(rows); display(cmp.round(4))
"""),
        reading("""Under ticket-only feedback the learned policy beats the best single-feature rule,
justifying the classifier. Under resolution-informed feedback both are dominated by always-on, again
showing that no policy is needed there. The simple rule (open when baseline confidence is low) is a
useful interpretable fallback but is not sufficient on its own."""),

        section("Does calibration already do the policy's job?",
                "Compare the policy value on the uncalibrated (legacy Laplace) prior against the calibrated prior.",
                "results/gate_pilot_blind{,_eb}/gate_pilot.csv", "Same features; different prior behind the feedback."),
        explain("Print both blind policy summaries side by side."),
        code("""
pl = L.load_gate_pilot("blind"); peb = L.load_gate_pilot("blind_eb")
both = pd.concat([pl.assign(prior="legacy Laplace"), peb.assign(prior="calibrated EB")])
display(both[["prior", "run", "oof_auc", "mean_delta_always_on", "mean_delta_oracle", "ceiling_recovery"]].round(4))
"""),
        reading("""On the uncalibrated prior the policy recovers a large fraction of the ceiling, turning a
clearly negative always-on into a positive policy. On the calibrated prior, always-on is already
near-safe, so the policy's *additional* value is small even though its AUC is higher. Calibration
and control are therefore **substitutes**: either discipline the prior or gate a raw one. We
recommend calibration as the default (it needs no labels) and keep the policy for settings where the
prior cannot be trusted."""),
        markdown("""## Conclusion — the control policy

A pre-generation policy is valuable exactly when feedback is unreliable and always-on is harmful.
Under ticket-only feedback it turns a negative always-on into a positive policy and recovers most of
the oracle ceiling; under resolution-informed feedback it is unnecessary. The next notebook confirms
the resulting recommendation on the untouched eval split."""),
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

- **Scope:** hierarchical backoff (intersection → team → class → global), minimum evidence 2.
- **Prior:** empirical-Bayes centered Laplace, κ=2, cap ±0.20.
- **Scaling:** pool-relative (λ=0.5) for the ticket-only setting; absolute for resolution-informed.
- **Control policy:** learned gate over pre-generation features, used only when feedback reliability is low.

Reported alongside: legacy Laplace fine-scope feedback as the resolution-informed optimum, and the
ticket-only results as the reliability lower bound."""),

        section("What are the generated dev results across all methods and both protocols?",
                "Assemble every generated dev run's summary into one table and show the protocol reversal.",
                "results/*_dev_*/*_summary.json", "Dev is used for development; eval is reported next."),
        explain("Build the conditioned-vs-blind dev table and plot it."),
        code("""
runs = L.load_run_summaries()
pivot = runs.pivot_table(index=["method", "agg"], columns="protocol", values="mean_delta_cosine")
display(pivot.round(4))
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=runs[runs["protocol"].isin(["conditioned", "blind"])], x="method", y="mean_delta_cosine",
            hue="protocol", palette={"conditioned": "#16856b", "blind": "#c44e52"}, ax=ax)
ax.axhline(0, color="black", lw=1); ax.set(ylabel="Generated-answer cosine delta", title="Conditioned vs blind (dev)")
plt.tight_layout(); L.savefig("06_dev_protocol_reversal", run_ids=[]); plt.show()
"""),
        reading("""The table is the paper's headline motif: the methods with positive deltas under
resolution-informed feedback have negative deltas under ticket-only feedback, and vice versa for the
broad scopes. The sign of the effect is controlled by how the feedback was produced, not by the
routing alone."""),

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
(+0.013 to +0.015) and ticket-only feedback is negative (−0.038). The calibrated ticket-only
configuration is near zero to slightly positive, and the unseen-procedure (disjoint) variant is
mildly positive. The eval split confirms the dev conclusions without any tuning."""),

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
the effects are not an artefact of using the same model family for retrieval and scoring. Magnitudes
are small; the resolution-informed conditioned run with empirical-Bayes even reaches significance on
BERTScore (about +0.013), which is the strongest independent confirmation in the study."""),

        section("Is the locked configuration robust to seeds and unseen procedures?",
                "Multi-seed dev runs and the disjoint eval run for the locked configuration.",
                "results/M5_backoff_*seed*/_summary.json; results/M5_backoff_*disjoint*/_summary.json",
                "Feedback source shifts slightly across seeds by construction."),
        explain("Print the locked configuration's seed and disjoint summaries."),
        code("""
import glob, json
rows = []
patterns = ["M5_backoff_*seed*", "M5_backoff_*disjoint"]
for pat in patterns:
    for folder in sorted(L.RESULTS.glob(pat)):
        for f in folder.glob("*_summary.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"run": s["experiment_id"], "n": s["total_valid"], "mean_delta_cosine": s["metrics"]["mean_delta_cosine"]})
display(pd.DataFrame(rows).round(4))
"""),
        reading("""Across seeds the locked configuration is small and mixed in sign, and it is mildly
positive on unseen procedures. The honest reading is that the calibrated ticket-only result is a
*de-risking* of feedback rather than a large gain; the large, reliable effect in this study is the
resolution-informed one."""),

        section("What is the claim ledger?",
                "Every paper claim mapped to the notebook section and artifact that supports it, with readiness.",
                "all notebook artifacts", "Claims marked pending stay conditional."),
        explain("Print the ledger tying each claim to its evidence."),
        code("""
status = L.artifact_status().set_index("section")["available"].to_dict()
claims = pd.DataFrame([
    ["Feedback benefit is conditional and oracle-inflated", "03 / analysis", "dev + eval summaries", True],
    ["Fine scopes help; broad pooling harms", "03 / analysis", "retriever_ladder grids", status.get("Granularity: ladder", False)],
    ["The failure is a calibration/scaling defect", "04 / modeling", "saturation.csv; blind grid", status.get("Granularity: blind ladder", False)],
    ["Pool-relative scaling rescues ticket-only feedback", "04 / modeling", "retriever_ladder_blind/grid.csv", status.get("Granularity: blind ladder", False)],
    ["The learned blend collapses (null)", "04 / modeling", "blend_eb/learned_weights.json", status.get("Granularity: blend", False)],
    ["A pre-generation policy recovers the ceiling when feedback is unreliable", "05 / modeling", "gate_pilot_blind", status.get("Gate pilot: blind", False)],
    ["A policy is unnecessary under resolution-informed feedback", "05 / modeling", "gate_pilot_conditioned", status.get("Gate pilot: blind", False)],
    ["Independent metrics agree with the retrieval metric", "06 / results", "rescored/method_comparison_v2.csv", status.get("Validity: rescoring", False)],
], columns=["claim", "section", "artifact", "ready"])
display(claims)
"""),
        markdown("""## Method card and limitations

**Recommended configuration.** Empirical-Bayes centered Laplace lift, hierarchical backoff
(minimum evidence 2), pool-relative scaling when feedback reliability is unknown; a learned
pre-generation policy only for unreliable feedback.

**When it helps.** On moderately uncertain retrievals with trustworthy (resolution-informed)
feedback; the effect reverses when feedback is ticket-only and unreliable.

**Limitations.** One organizational corpus; LLM-generated feedback; generated evidence concentrated
on seed 42 with limited multi-seed coverage; ticket-only gains are small and not individually
significant; resolution-informed judging uses the historical resolution and overstates what a
real-time user without resolution knowledge could provide. The offline proxy is valid for
configuration selection only."""),
    ],
)


OUTPUTS = {
    "02_protocol_and_validity.ipynb": NB02,
    "03_when_feedback_helps.ipynb": NB03,
    "04_modeling_the_prior.ipynb": NB04,
    "05_modeling_the_control_policy.ipynb": NB05,
    "06_final_results_and_claims.ipynb": NB06,
}


if __name__ == "__main__":
    for name, payload in OUTPUTS.items():
        path = HERE / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")