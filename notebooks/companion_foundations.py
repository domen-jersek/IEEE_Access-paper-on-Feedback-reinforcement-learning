"""S1-S3: problem and running example, data and splits, protocols and judge."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, idea, meaning, md, sec, take, why


def cells() -> list[dict]:
    return [
        md("""
# Part I — Foundations

This part establishes the problem, the data, the two feedback protocols, the generation-regime
discipline, and the metric/statistical protocol. Everything later in the notebook rests on these
definitions, so each choice is justified rather than assumed.
"""),

        # ------------------------------------------------------------------ S1
        sec("S1. Why would a retrieval system need feedback?",
            "RAG retrieves the textually most similar historical tickets, but textual similarity is not "
            "procedural correctness. If this gap exists, some baseline answers must follow the wrong "
            "procedure even while looking relevant.",
            "A real ticket where the baseline retrieval finds a similar but procedurally wrong "
            "historical reply, and feedback finds the right one."),
        idea("""Two tickets can share almost all of their vocabulary and still require different
resolutions. \"I cannot connect to client software\" retrieves historical tickets about *network and
VPN access* because they share the words \"connect\" and \"client\"; the correct resolution, however,
may be a request form for *locally installed software*. A retrieval system optimising cosine
similarity has no way to prefer the procedurally correct neighbour. Feedback is the natural fix:
promote candidates whose historical replies were judged useful, even if they are less similar. Before
building anything, we show that this failure mode is real, not hypothetical."""),
        code("""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_records(folder):
    \"\"\"Newest details JSON of a run, keyed by ticket id.\"\"\"
    return {r["ticket_id"]: r for r in L.load_details(folder)}


def trunc(text, n=170):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def show_case(folder, ticket_id, top=3, chars=170):
    \"\"\"Print a compact baseline-vs-feedback comparison for one ticket.\"\"\"
    record = load_records(folder)[ticket_id]
    rows = []
    for side, label in (("baseline", "baseline (no feedback)"), ("feedback", "feedback")):
        for c in record[side]["retrieval"][:top]:
            rows.append({
                "side": label,
                "rank": int(c.get("feedback_rank", c.get("faiss_rank", 0))),
                "candidate": c["seq_id"],
                "similarity": round(float(c.get("faiss_score", 0.0)), 3),
                "lift": round(float(c.get("feedback_lift", 0.0)), 3),
                "reply (truncated)": trunc(c.get("reply", ""), chars),
            })
    display(pd.DataFrame(rows))
    print("QUERY TITLE   :", trunc(record["query_title"], 200))
    print("REFERENCE     :", trunc(record["reference_reply"], 260))
    print("BASELINE ANS  :", trunc(record["baseline"]["answer"], 260))
    print("FEEDBACK ANS  :", trunc(record["feedback"]["answer"], 260))
    print(f"DELTA COSINE  : {record['deltas']['delta_cosine']:+.3f}")
    return record


RUN_M4_COND_EVAL = "M4_intersection_eval_conditioned_continuous"
show_case(RUN_M4_COND_EVAL, "R-377");
"""),
        howto("""Three retrieved candidates per side, with the retrieval similarity and (for feedback) the
applied lift. The last lines are the reference (what the support team actually wrote), the baseline
answer, and the feedback answer, each truncated. `DELTA COSINE` is how much closer the feedback
answer is to the reference."""),
        meaning("""The query asks for a connection to client software. The baseline top-1 is a
*network-access* candidate (`R-1372`, \"Cannot access client intranet destination\") that shares the
words \"client\" and \"access\", and the baseline answer accordingly routes the user to a VPN/access
form. The reference, however, is a *local software support* form. Feedback promotes `R-5`, whose
reply is exactly that form, and the feedback answer follows it. The delta is +0.540 for this ticket —
one of the largest on the split — and the mechanism is exactly the one we hypothesised."""),
        why("""Dense retrieval optimises semantic proximity, which conflates two different notions:
\"about the same words\" and \"resolved the same way\". The conditioned judge has access to the
historical resolution; it can therefore rate `R-5` as procedurally equivalent to the reference even
though its text is less similar to the query. The baseline has no such signal."""),
        artefact("""One example is an anecdote, not evidence. Section S6 measures the effect over all
398 eval tickets with confidence intervals, and Section S13 shows that the failure mode is systematic
(the semantic filter, which tries to fix it heuristically, fails on every retriever). The example's
role is to make the later numbers concrete."""),
        take("""Textual similarity and procedural correctness genuinely diverge; feedback can supply the
missing procedural signal in the easiest case — when the judge knows the resolution."""),

        # ------------------------------------------------------------------ S2
        sec("S2. What does the corpus look like?",
            "Every later result — scope starvation, class imbalance, near-duplicate procedures — is a "
            "property of the data. We quantify those properties first so the modelling choices are "
            "interpretable.",
            "Corpus composition by team and intent class, split sizes, and repetition statistics."),
        idea("""An IT service desk corpus is not a balanced benchmark. A few teams handle most tickets,
the long tail is large, and many tickets are near-duplicates of each other because procedures are
templated. Both facts matter: imbalance creates the scope-level calibration problem (S11), and
near-duplication determines what the procedure-disjoint split (S22) has to control for."""),
        code("""
ds = L.load_dataset()
split = L.load_splits(42)
split_disjoint = L.load_splits(42, regime="disjoint")
sizes = {k: len(v) for k, v in split.items() if isinstance(v, list)}
sizes_disjoint = {k: len(v) for k, v in split_disjoint.items() if isinstance(v, list)}

teams = ds["Team->Name"].value_counts()
classes = ds["intent_class"].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1.2, 0.7]})
top = teams.head(12)[::-1]
axes[0].barh(top.index, top.values, color=L.ACCENT_COLOR)
axes[0].set_xlabel("tickets")
L.takeaway(axes[0], f"{len(teams)} teams, long-tailed")
L.note(axes[0], f"top 3 teams = {top.values[-3:].sum() / len(ds):.0%} of corpus", loc="lower right")

axes[1].bar(classes.index, classes.values, color=L.COND_COLOR)
axes[1].tick_params(axis="x", rotation=75)
for tick in axes[1].get_xticklabels():
    tick.set_ha("right")
axes[1].set_ylabel("tickets")
L.takeaway(axes[1], f"{len(classes)} intent classes, imbalanced")
L.note(axes[1], f"largest class = {classes.iloc[0] / len(ds):.0%}", loc="upper right")

labels = ["train", "dev", "eval"]
x = np.arange(len(labels))
axes[2].bar(x - 0.2, [sizes[k] for k in labels], width=0.4, color=L.ACCENT_COLOR, label="random")
axes[2].bar(x + 0.2, [sizes_disjoint.get(k, 0) for k in labels], width=0.4,
            color=L.WARN_COLOR, label="disjoint")
axes[2].set_xticks(x, labels)
axes[2].set_ylabel("tickets")
axes[2].legend(fontsize=8, frameon=False)
L.takeaway(axes[2], "Splits: 878 / 319 / 398")
L.note(axes[2], "5 seeds; disjoint variants too", loc="upper right")
plt.tight_layout()
L.savefig("companion_02_corpus", run_ids=[])
plt.show()

print(f"Tickets: {len(ds)}   Teams: {ds['Team->Name'].nunique()}   Classes: {ds['intent_class'].nunique()}")
print(f"Random splits:    {sizes}")
print(f"Disjoint splits:  {sizes_disjoint}")
"""),
        code("""
groups = json.loads((L.DATA / "procedure_groups.json").read_text(encoding="utf-8"))
pd.set_option("display.max_colwidth", 120)
display(pd.Series(groups, name="value").to_string())

ladder = L.load_ladder_curve()
base_hit = float(ladder.loc[ladder["retriever"] == "dense_minilm", "base_minilm_same_reply_hit1"].iloc[0])
print(f"\\nBaseline top-1 is the EXACT historical reply for only {base_hit:.1%} of dev tickets "
      f"(the missing procedural signal that feedback has to supply).")
"""),
        howto("""Left: tickets per resolving team (top 12). Middle: tickets per intent class. Right:
split sizes for the random and disjoint regimes. The second cell lists the near-duplicate statistics
computed from reply/title/subcategory hashes."""),
        meaning("""The corpus has 1,595 tickets across 37 teams and 17 intent classes; the top three
teams hold about a third of the tickets and the class distribution is similarly skewed. Many tickets
share replies and procedures: identical replies appear for multiple tickets, and 264 tickets share a
(title, subcategory) group. Most importantly, the baseline top-1 candidate has the *exact* historical
reply for only about 31 % of dev tickets — the remaining 69 % require some procedural inference or,
when the retrieval is wrong, expose exactly the failure mode from S1."""),
        why("""Templated resolutions (form redirects, onboarding/offboarding boilerplate) make replies
repetitive; ticket volume is concentrated in a few services. Both are typical of enterprise service
desks, which is why we report them rather than smoothing them away."""),
        artefact("""Near-duplicates could make the evaluation optimistic: if the same procedure appears
in train and eval, feedback could simply memorise it. This is why the procedure-disjoint regime
exists (S22): identical procedures never cross the split boundary, and the conditioned effect still
holds there."""),
        take("""The corpus is small, imbalanced, and repetitive; the evaluation must control for
near-duplication, and scope-level base rates will matter."""),

        # ------------------------------------------------------------------ S3
        sec("S3. What are the two feedback protocols, and how reliable is the judge?",
            "The paper's central contrast is that the same reranking helps or harms depending on what "
            "the judge knows. We must show that the two protocols really differ in information and "
            "that the judge's scores carry usable signal in both.",
            "A schematic of the two protocols, the judge-score distributions (zero inflation), and "
            "reliability curves with AUC."),
        idea("""In the *conditioned* protocol the judge sees the ticket, the historical ground-truth
reply, and the candidate; it answers \"would following this candidate lead to the same resolution?\"
This simulates an expert with the answer key — a strong but non-deployable upper bound. In the
*blind* protocol the judge sees only the ticket and the candidate, exactly like a user browsing past
replies. If both protocols produce the same rerankings, feedback would be free; our hypothesis is
that they differ sharply, and that difference is the paper."""),
        code("""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def protocol_diagram(ax, protocol):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    conditioned = protocol == "conditioned"

    def box(x, y, w, h, text, color, alpha=0.12, fontsize=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                    linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    magenta = "#8b4fb3"
    box(0.03, 0.55, 0.42, 0.30, "QUERY TICKET\\ntitle + description\\n(self-excluded from the pool)", L.ACCENT_COLOR)
    box(0.55, 0.55, 0.42, 0.30,
        "HISTORICAL GROUND-TRUTH REPLY\\n(seen only in conditioned)" if conditioned else
        "NO GROUND TRUTH\\n(blind = deployable setting)",
        magenta if conditioned else L.BLIND_COLOR)
    box(0.03, 0.08, 0.42, 0.28, "RETRIEVED CANDIDATE\\nhistorical ticket + reply", L.ACCENT_COLOR)
    box(0.55, 0.08, 0.42, 0.28, "JUDGE SCORE 0–1\\nprocedural match / usefulness", L.COND_COLOR)
    ax.add_patch(FancyArrowPatch((0.45, 0.22), (0.55, 0.22), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=L.NEUTRAL_COLOR))
    ax.add_patch(FancyArrowPatch((0.45, 0.70), (0.55, 0.70), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=L.NEUTRAL_COLOR))
    title = "conditioned: the judge sees the resolution" if conditioned else "blind: the judge sees only the ticket"
    ax.set_title(title, fontsize=11, fontweight="bold", color=L.COND_COLOR if conditioned else L.BLIND_COLOR)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
protocol_diagram(axes[0], "conditioned")
protocol_diagram(axes[1], "blind")
plt.tight_layout()
L.savefig("companion_03_protocols", run_ids=[])
plt.show()
"""),
        code("""
rel_c = L.load_calib_reliability("conditioned")
rel_b = L.load_calib_reliability("blind")
report = L.load_calib_report()
auc_c = report["protocols"]["conditioned"]["auc_of_score"]["same_reply"]
auc_b = report["protocols"]["blind"]["auc_of_score"]["same_reply"]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for rel, protocol, color in ((rel_c, "conditioned", L.COND_COLOR), (rel_b, "blind", L.BLIND_COLOR)):
    axes[0].plot(rel["mean_score"], rel["n"] / rel["n"].sum(), marker="o", ms=4,
                 color=color, label=protocol)
axes[0].set_yscale("log")
axes[0].set_xlabel("judge score")
axes[0].set_ylabel("share of all judged pairs (log)")
axes[0].legend(frameon=False)
L.takeaway(axes[0], "Judge scores are zero-inflated (both protocols)")
L.note(axes[0], f"conditioned: {rel_c['n'].iloc[0] / rel_c['n'].sum():.0%} exact zeros · "
                f"blind: {rel_b['n'].iloc[0] / rel_b['n'].sum():.0%}")

for rel, protocol, color in ((rel_c, "conditioned", L.COND_COLOR), (rel_b, "blind", L.BLIND_COLOR)):
    axes[1].plot(rel["mean_score"], rel["mean_usefulness_sim"], marker="o", ms=4,
                 color=color, label=protocol)
axes[1].set_xlabel("judge score")
axes[1].set_ylabel("mean reply-similarity of the candidate to the reference")
axes[1].legend(frameon=False)
L.takeaway(axes[1], "Conditioned scores are far more reliable")
L.note(axes[1], f"AUC(same reply): conditioned {auc_c:.2f} vs blind {auc_b:.2f}")
plt.tight_layout()
L.savefig("companion_03_judge_reliability", run_ids=[])
plt.show()

for protocol in ("conditioned", "blind"):
    dist = report["protocols"][protocol]["score_distribution"]
    print(f"{protocol:12s} n={dist['n_rows']:,}  median={dist['quantiles']['0.5']:.2f}  "
          f"exact zeros={dist['pct_exact_0']:.1%}  distinct scores={dist['n_distinct_scores']}")
"""),
        howto("""Left: the share of judged pairs at each score (log scale) — the spike at 0 is the
zero-inflation. Right: for each score bin, the average similarity between the candidate's reply and
the reference reply; a monotone curve means the judge ranks by procedural usefulness. `AUC(same
reply)` is the probability that a truly matching candidate gets a higher score than a non-matching
one."""),
        meaning("""Both protocols are dominated by exact zeros (about 59 % conditioned and 48 %
blind), and the judge uses only ~22–24 distinct score values; it is a coarse, zero-inflated signal.
Nevertheless it carries information: the conditioned score has AUC 0.90 for identifying the exact
historical reply versus 0.84 blind. The blind judge is weaker but far from random — it sees
plausibility, just not the resolution."""),
        why("""The judge is an LLM at temperature 0 asked for a single 0–1 score; its prompt anchors
0.0 — \"unrelated procedure\" — which is the modal answer for a retrieved candidate pool that mostly
contains non-matching tickets. Because the scores are used as *counts* in a popularity lift, this
zero-inflation will have mechanical consequences (S11) rather than being a cosmetic detail."""),
        artefact("""The reliability curve uses reply-similarity to the reference as a proxy for
usefulness, because the reference is the only ground truth available for every pair. That proxy is
imperfect; Section S21 replaces it with an independent LLM judge on generated answers, which agrees in
direction. The absolute AUC values should be read as evidence of ordering, not as calibrated
probabilities."""),
        take("""The two protocols differ in real information (AUC 0.90 vs 0.84) and both are
zero-inflated; the conditioned protocol is an upper bound, the blind protocol the deployable case.
"""),
    ]
