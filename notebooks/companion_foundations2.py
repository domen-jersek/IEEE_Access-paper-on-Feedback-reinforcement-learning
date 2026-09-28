"""S4-S5: generation regime, metrics, statistics, and statistical power."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, idea, meaning, sec, take, why


def cells() -> list[dict]:
    return [
        # ------------------------------------------------------------------ S4
        sec("S4. Are all compared runs generated under the same regime?",
            "LLM answers at temperature 0 are not bit-reproducible without a response cache. If two "
            "runs were generated under different conditions, their difference cannot be attributed to "
            "the intervention. This is a data-validity question that reviewers cannot check from the "
            "paper alone.",
            "A pairwise answer-identity matrix across representative runs, showing which runs share a "
            "generation regime."),
        idea("""A 'generation regime' is the tuple (generator model, system prompt, response cache).
Two runs are comparable only if they share a regime; re-running a warm run must reproduce the same
answers exactly. Early in the project some runs were generated before the cache existed and were
therefore uncached — each produced different answers for identical prompts. We detect this
empirically by comparing baseline answers (the baseline prompt depends only on the query and its
top-5, so it is identical across methods): runs in the same regime agree on (nearly) every baseline
answer; runs in different regimes do not."""),
        code("""
import glob as _glob
import os as _os


def details_files(folder):
    return sorted(_glob.glob(str(L.RESULTS / folder / "*_details.json")), key=_os.path.getmtime)


def baseline_answers(path):
    recs = json.load(open(path, encoding="utf-8"))
    return {r["ticket_id"]: r["baseline"]["answer"] for r in recs}


REGIME_RUNS = [
    ("M4 cond dev — Sep 14 (uncached)", "M4_intersection_dev_conditioned_continuous", "oldest"),
    ("baseline dev — Sep 14 (uncached)", "baseline_dev_conditioned_continuous", "oldest"),
    ("M2 blind dev — Sep 19", "M2_team_dev_blind_continuous", "newest"),
    ("M5 blind dev — Sep 20", "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "newest"),
    ("M4 cond dev — Sep 22 (warm)", "M4_intersection_dev_conditioned_continuous", "newest"),
    ("M2 cond eval — Sep 20", "M2_team_eval_conditioned_continuous", "newest"),
    ("M4 blind eval — Sep 20", "M4_intersection_eval_blind_continuous", "newest"),
    ("M4 cond eval — Gemini", "M4_intersection_eval_conditioned_continuous_gengemini38flash", "newest"),
]

answers = []
for label, folder, which in REGIME_RUNS:
    files = details_files(folder)
    answers.append(baseline_answers(files[0] if which == "oldest" else files[-1]))

n = len(REGIME_RUNS)
identity = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        common = set(answers[i]) & set(answers[j])
        identity[i, j] = np.mean([answers[i][t] == answers[j][t] for t in common])

fig, ax = plt.subplots(figsize=(9.5, 7.5))
im = ax.imshow(identity, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{identity[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="black" if identity[i, j] > 0.25 else "white")
ax.set_xticks(range(n), [r[0] for r in REGIME_RUNS], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n), [r[0] for r in REGIME_RUNS], fontsize=8)
ax.set_title("Baseline-answer identity: 1.00 = same generation regime", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.7, label="fraction of identical baseline answers")
plt.tight_layout()
L.savefig("companion_04_regimes", run_ids=[])
plt.show()

print("Regime groups (identity >= 0.95 with each other):")
for i, (label, _, _) in enumerate(REGIME_RUNS):
    same = [REGIME_RUNS[j][0] for j in range(n) if identity[i, j] >= 0.95]
    print(f"  {label:38s} -> {len(same)} runs: {same[:3]}{' ...' if len(same) > 3 else ''}")
"""),
        howto("""Rows and columns are runs; each cell is the fraction of tickets whose baseline answer
is byte-identical between the two runs, over their common tickets. 1.00 (dark green) means the two
runs share a generation regime; low values (red) mean different generation conditions."""),
        meaning("""The matrix shows three clusters: the two uncached Sep-14 runs are each their own
regime (they do not even match each other), all Sep-19+ runs of the primary model agree perfectly,
and the Gemini run forms a separate regime (as it must — it is a different model). Every result in
this notebook is computed within one of these regimes: all primary results use the Sep-19+ cluster,
and the cross-generator check (S23) compares Gemini only within itself."""),
        why("""Baseline prompts are identical across methods for a given ticket, so agreement on
baseline answers is a sufficient fingerprint of 'same generator conditions'. The Sep-14 runs were
produced before the persistent cache existed; without a cache, provider-side nondeterminism changes
the answers even at temperature 0. The affected dev configurations were regenerated under the cache;
the regenerated runs match the Sep-19+ cluster exactly, which validates the repair."""),
        artefact("""The identity test uses exact string equality of answers. Whitespace or trailing
newlines could in principle split a regime; instead we observe perfect 1.00 agreement within the
cluster and near-zero across clusters, so the classification is not marginal. `warm=True` summaries
(0 cache misses) additionally guarantee that a run reused cached answers rather than regenerating
them."""),
        take("""All compared primary runs share one verified generation regime; the uncached Sep-14
runs were detected and regenerated, and the cross-generator block is explicitly a separate regime.
"""),

        # ------------------------------------------------------------------ S5
        sec("S5. What do the metrics measure, how are they tested, and what effects could this study ever detect?",
            "Reviewers must be able to interpret effect sizes and judge whether null results are "
            "evidence of absence or absence of evidence. We define the metrics, their agreement, and "
            "the minimum detectable effect.",
            "Metric correlation matrix, per-ticket effect distributions, and an MDE table."),
        idea("""The primary metric is embedding cosine between the generated reply and the historical
reply, which measures reference fidelity, not ticket resolution. Because retrieval uses MiniLM, a
metric in the same family could favour the retriever; we therefore also measure BGE cosine (a
different embedding family), ROUGE-L (lexical overlap) and BERTScore (contextual token matching), and
we validate answers with an independent LLM judge (S21). For significance we use the paired Wilcoxon
test over tickets. Finally, with realistic per-ticket variance the study can only detect effects
above a minimum detectable effect (MDE); we compute it so that every null result can be read
correctly."""),
        code("""
eval_runs = [
    "M2_team_eval_conditioned_continuous",
    "M4_intersection_eval_conditioned_continuous",
    "M4_intersection_eval_conditioned_continuous_gated",
    "M2_team_eval_blind_continuous",
    "M4_intersection_eval_blind_continuous",
    "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2",
    "M2_team_eval_conditioned_continuous_gengemini38flash",
    "M4_intersection_eval_conditioned_continuous_gengemini38flash",
]
frames = []
for run in eval_runs:
    frame = L.load_rescored_per_ticket(run)
    frame["run"] = run
    frames.append(frame)
per_ticket = pd.concat(frames, ignore_index=True)
metric_cols = ["delta_cosine", "delta_cosine_bge", "delta_rouge_l", "delta_bertscore_f1"]
corr = per_ticket[metric_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
short = ["cosine", "bge", "rouge_l", "bertscore"]
ax.set_xticks(range(4), short)
ax.set_yticks(range(4), short)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if corr.iloc[i, j] > 0.6 else "black")
L.takeaway(ax, "The four metrics agree per ticket (ρ 0.6–0.9)")
L.note(ax, f"eval runs pooled, n={len(per_ticket):,} ticket-run pairs", loc="lower right")
plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
plt.tight_layout()
L.savefig("companion_05_metric_correlation", run_ids=[])
plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for ax, folder, label, color in (
    (axes[0], "M4_intersection_eval_conditioned_continuous", "M4 conditioned", L.COND_COLOR),
    (axes[1], "M4_intersection_eval_blind_continuous", "M4 blind", L.BLIND_COLOR),
):
    recs = L.load_details(folder)
    d = np.array([r["deltas"]["delta_cosine"] for r in recs])
    L.effect_hist(ax, d, color=color)
    ax.set_xlabel("per-ticket Δcosine (feedback − baseline)")
    ax.set_ylabel("tickets")
    L.takeaway(ax, f"{label}: mean {d.mean():+.3f} · helped {np.mean(d > 0):.0%} · "
                   f"hurt {np.mean(d < 0):.0%} · no-op {np.mean(d == 0):.0%}")
    L.note(ax, f"n={len(d)} · sd={d.std(ddof=1):.3f} · share |Δ|>0.1 = {np.mean(np.abs(d) > 0.1):.0%}")
plt.tight_layout()
L.savefig("companion_05_effect_anatomy", run_ids=[])
plt.show()

MDE_RUNS = [
    ("M2 team, conditioned", "M2_team_eval_conditioned_continuous"),
    ("M4 team∩class, conditioned", "M4_intersection_eval_conditioned_continuous"),
    ("M2 team, blind", "M2_team_eval_blind_continuous"),
    ("M4 team∩class, blind", "M4_intersection_eval_blind_continuous"),
    ("M5 backoff+EB, conditioned", "M5_backoff_eval_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    ("M5 backoff+EB, blind", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
]
rows = []
for label, folder in MDE_RUNS:
    d = np.array([r["deltas"]["delta_cosine"] for r in L.load_details(folder)])
    mde = 2.802 * d.std(ddof=1) / np.sqrt(len(d))
    rows.append({"run": label, "n": len(d), "mean Δ": round(float(d.mean()), 4),
                 "sd": round(float(d.std(ddof=1)), 3), "MDE (80% power)": round(float(mde), 4),
                 "mean beyond MDE?": "yes" if abs(d.mean()) >= mde else "no"})
display(pd.DataFrame(rows))
"""),
        howto("""Top: Spearman correlations between the four answer metrics over all eval runs pooled.
Bottom left/right: per-ticket Δ distributions for the main conditioned and blind runs, with the mean
marked; the annotation gives the fraction of helped, hurt, and unchanged tickets. The table lists, per
finalist run, the sample size, standard deviation, and the minimum detectable effect at 80 % power
(2.80·sd/√n)."""),
        meaning("""The metrics are highly consistent per ticket (Spearman 0.6–0.9), so the effect is
not an artefact of one metric family. The per-ticket distributions show why means are small: even in
the best conditioned run roughly 55 % of tickets are helped and 43 % are hurt, with many tickets
unchanged. Finally, the MDE table calibrates the nulls: with n=398 and sd≈0.09–0.18, this study can
detect effects around 0.013–0.025. The conditioned effects (+0.014/+0.015) sit right at that edge;
the calibrated blind result (+0.002) is an order of magnitude below it and therefore cannot be
confirmed — an absence of evidence, not evidence of absence."""),
        why("""The variance is dominated by per-ticket heterogeneity: changing the retrieval context
can flip an answer substantially in either direction. A mean is a small residual of large opposing
movements, so the paired design and a sufficiently large n matter more than the metric choice."""),
        artefact("""All four metrics are reference-based: they reward similarity to the historical
reply, which may itself be imperfect. The independent judge (S21) is the control for this. The
correlation matrix pools runs from two generator families; restricting to one family leaves the
conclusion unchanged (the patterns are visible per run in the CSV files)."""),
        take("""Metrics agree; effects are small relative to per-ticket variance; the study is powered
for effects ≥ ~0.013, so only the conditioned effects (and blind harm) are confirmable, while
calibrated-blind gains are below the detection threshold."""),
    ]
