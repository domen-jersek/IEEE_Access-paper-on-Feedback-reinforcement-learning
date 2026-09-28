"""S25-S28: practical recipe, limitations, reviewer FAQ, reproducibility."""
from __future__ import annotations

from companion_helpers import code, md, qa, sec


def cells() -> list[dict]:
    return [
        md("""
# Part VI — What does it mean?
"""),

        # ------------------------------------------------------------------ S25
        sec("S25. What should a practitioner do?",
            "The paper's value is in the decision rules that follow from the evidence.",
            "A four-step recipe with the conditions under which each step applies."),
        md("""
**1. Decide what your feedback is.** If your signal is produced with access to resolutions or
outcomes (an oracle-side source), treat its measured gain as an upper bound. If it is produced
ticket-only (users, a blind judge), it is deployable — and it will harm retrieval unless calibrated.

**2. Calibrate before you re-rank.** Centre the prior on each scope's base rate (empirical Bayes) and
express the bonus in units of the pool's score spread. This is free, needs no labels, and is the
difference between −0.05 and neutral (S12). Do not tune the formula shape; centre the prior (S9).

**3. Prefer fine scopes, never global pooling (S7).** Use hierarchical backoff: the finest scope with
at least two observations; fall back when a ticket is starved (S10). Expect no further gains beyond a
modest judging budget: volume saturates early.

**4. Add control only as insurance (S16–S18).** If feedback is untrustworthy and always-on is
negative, a pre-generation gate recovers most of the attainable ceiling — and it is unnecessary once
the prior is calibrated. For per-ticket decisions (which configuration, whether to intervene at all),
model the *expected delta*, not the sign; sign-only selection is negative, expected-value selection is
positive.

**5. Validate against a stronger retriever (S8).** Feedback gains shrink as retrieval improves, and
transfer is not guaranteed; if a strong retriever or reranker is already deployed, test feedback on
top of it before trusting it.
"""),

        # ------------------------------------------------------------------ S26
        sec("S26. What do we NOT claim?",
            "Explicit non-claims prevent reviewers from reading more into the results than the design "
            "supports, and they define the next experiments.",
            "A list of boundaries, each tied to the evidence."),
        md("""
- **No deployable improvement.** The ticket-only (blind) mean effect is null after calibration
  (+0.002 on eval, below the study's detection threshold). What the deployable pipeline buys is
  *harm avoidance* and a decision layer, not a confirmed gain.
- **No human feedback.** The feedback is produced by an LLM judge at temperature 0. The conditioned
  protocol is resolution-informed and simulates an expert with the answer; it is an upper bound.
  Human validation remains future work.
- **No generalisation beyond one corpus.** 1,595 tickets from one organization in one language. The
  mechanisms should transfer; the magnitudes need re-measurement.
- **No resolution metric.** Quality is measured as similarity to the historical reply plus an
  LLM-judge procedure comparison. Whether a generated reply actually resolved the ticket is not
  measured.
- **No per-ticket metric/judge agreement.** The embedding metric and the independent judge agree in
  aggregate (S21) but disagree on many individual tickets (sign agreement 0.35–0.40 conditioned).
- **No same-model independence for feedback.** The judge that rates candidates and the primary
  generator belong to the same family; the judge never sees generated answers, and the Gemini
  replication (S23) bounds the concern without eliminating it.
"""),

        # ------------------------------------------------------------------ S27
        sec("S27. Reviewer FAQ",
            "Collected questions a reviewer is likely to ask, each answered with the section and "
            "numbers that settle it.",
            "Twenty-five short questions and answers."),
        qa("Why not use human feedback ratings?",
           "Human ratings were out of scope for this study; the feedback is LLM-judge simulated, and the "
           "conditioned protocol is explicitly an oracle-side upper bound. The paper states this as a "
           "limitation and designs around it: the blind protocol is the deployable comparator, and a "
           "second generator family (S23) removes the obvious single-model concern. A human annotation "
           "study is the natural next step."),
        qa("What exactly is the conditioned protocol, and why call it an upper bound?",
           "The conditioned judge sees the ticket, the candidate reply, *and* the historical ground-truth "
           "reply, and scores procedural equivalence. A real system does not have the resolution when a "
           "ticket arrives, so the resulting gains (+0.014/+0.015) cannot be deployed; they measure how "
           "much the signal could be worth with perfect resolution knowledge (S3, S6)."),
        qa("Why are the effects so small?",
           "Per-ticket variance is large (sd 0.09–0.18) and effects are means of opposing movements "
           "(43 % of tickets are hurt even in the best conditioned run). A mean of +0.015 corresponds to "
           "a systematic shift of the whole distribution toward the reference, not to a few tickets "
           "(S5, S6)."),
        qa("Is the study powered to detect the effects it reports?",
           "Yes for the conditioned effects and blind harm (MDE ≈ 0.013–0.025; observed +0.014/+0.015 and "
           "−0.038), and no for the calibrated blind gain (+0.002 is far below MDE). We report the latter "
           "as an absence of evidence and never claim it as a positive result (S5)."),
        qa("Why freeze thresholds on dev instead of optimising them on eval?",
           "Tuning on eval would leak and inflate policy values. Every gate and magnitude model is "
           "trained on dev with grouped CV, the threshold is frozen, and eval is scored once. The eval "
           "threshold sweep exists only as a diagnostic artifact (S16, S19)."),
        qa("Why grouped cross-validation?",
           "Every ticket appears once per configuration in the training table. Row-level folds would put "
           "the same ticket's other configurations in both train and test, leaking ticket identity. "
           "GroupKFold by ticket removes that leakage; clustered bootstrap intervals resample tickets "
           "rather than rows (S15)."),
        qa("Why can a gate be worse than always-on?",
           "A gate replaces feedback with the baseline on closed tickets, so its gain is exactly "
           "−(n_closed/N)·mean_delta_closed. If the closed set has a non-negative mean — no-ops after "
           "calibration — the gate can only lose. That is why it helps on the raw prior (recovery "
           "0.54–0.56) and not after calibration (S16, S17)."),
        qa("So are calibration and gating alternatives?",
           "Yes, they are substitutes: both address the prior's treatment of low-evidence candidates. "
           "Calibrate first because it needs no labels; add a gate only when the prior cannot be trusted "
           "(S17, S25)."),
        qa("Why does a sign classifier fail for per-ticket decisions?",
           "Policy value is linear in the effect, so the optimal decision uses E[delta], not "
           "P(delta>0). The sign gate closes 152–235 tickets with mean −0.07…−0.10; the magnitude model "
           "closes 46–66 tickets with mean −0.28…−0.35, and action selection flips from −0.0079 to "
           "+0.0116 (S18)."),
        qa("What does the cosine metric actually measure?",
           "Reference fidelity: the embedding cosine between the generated reply and the historical "
           "reply that the support team wrote. It is not resolution success. We control for the metric "
           "family with BGE, ROUGE-L, BERTScore and an independent judge (S5, S20, S21)."),
        qa("Why is BERTScore missing for some runs?",
           "It is compute-heavy on CPU. It is computed for the eval and gated finalists, which carry the "
           "headline results; dev runs have BGE and ROUGE, which already provide an independent "
           "embedding family. The omission is stated in S20."),
        qa("Why do the dev and eval AUCs differ, and why is the team-conditioned AUC only 0.56?",
           "AUC depends on the base rates and on how much harmful mass exists in that configuration. "
           "Conditioned feedback is already near-optimal (always-on positive), so there is little to "
           "rank; the gate is neutral by design (S17). A high AUC is not required when there is no "
           "recoverable headroom."),
        qa("What is novel relative to the earlier conference paper?",
           "The conference paper was exploratory on 250 queries: cosine gating failed, pool filtering "
           "was inconclusive, and it ended by proposing a learned gate. This work is a new environment "
           "with 1,595 tickets, train/dev/eval discipline, two protocols, generated evaluations with "
           "independent metrics and an independent judge, a calibration mechanism and fix, a "
           "leak-free gate with a decomposition law, a magnitude decision layer, and robustness "
           "(seeds, unseen procedures, second generator). The two share one idea and no evidence."),
        qa("Why is the cross-generator effect larger?",
           "Measured facts: Gemini's answers start closer to the reference and change *less* with "
           "context (0.13 vs 0.18), yet the feedback benefit is larger (+0.040 vs +0.015). The likely "
           "explanation is greater form/template fidelity, so promoting the correct procedural candidate "
           "translates more directly into reference match; we present this as a hypothesis (S23)."),
        qa("Could the results be the same-model judge/generator artefact?",
           "The feedback judge never sees generated answers. The Gemini replication shows the sign and "
           "significance of the effect do not depend on the primary model family. Shared priors could "
           "still inflate absolute magnitudes, which is why we report magnitudes as generator-dependent "
           "(S23, S26)."),
        qa("Why not a second corpus?",
           "Not available within the project constraints. The procedure-disjoint split and the "
           "second-generator replication are the internal substitutes; external validity is listed as a "
           "limitation (S22, S26)."),
        qa("Why do the alternate-retriever runs use different lift settings?",
           "Each retriever was tuned on dev in the earlier development phase; the comparison is "
           "therefore confounded between retriever and calibration. The offline ladder (constant grid "
           "per retriever) supports the same direction, and the confound is stated explicitly (S8, S13)."),
        qa("Why does the semantic filter fail?",
           "Fine-scope metadata already encodes procedural relevance better than query-candidate text "
           "cosine. The filter removes candidates that are textually distant but procedurally right, so "
           "it is worse on every retriever (S13)."),
        qa("Why does the learned blend collapse?",
           "Scope lifts are highly correlated and add variance; the train-selected weights put almost "
           "all mass on one scope, so the blend adds parameters without signal (S13)."),
        qa("What is the oracle and what does 'recovery' mean?",
           "The oracle applies feedback only to tickets that actually improved, and removes all harm; it "
           "uses the labels and is unattainable. Recovery is (policy − always-on)/(oracle − always-on), "
           "so 0 means 'no better than applying feedback always' and 1 means 'as good as the oracle' "
           "(S5, S16)."),
        qa("What does 'warm' mean, and why does it matter?",
           "A warm run had zero cache misses: every prompt was served from the shared response cache, "
           "so the run is fully reproducible. Runs from different generation regimes are not comparable; "
           "we detected and regenerated the uncached early runs (S4)."),
        qa("How is the data anonymized and can it be shared?",
           "The corpus is anonymized enterprise IT data (names, URLs and identifiers replaced). The "
           "organization's data cannot be shared publicly; the code, manifests and frozen artifacts are "
           "released with the paper, and all generated evidence can be reproduced from the source data "
           "within the organization."),
        qa("Why two protocols instead of one?",
           "Because the sign of the result is controlled by the protocol. Reporting only the "
           "resolution-informed protocol would overstate deployable performance; reporting only the "
           "deployable one would miss the mechanism. The contrast is the paper."),
        qa("What is the single most important practical result?",
           "Calibrate the prior. It is free, it removes a systematic harm of roughly −0.05, and it "
           "determines whether a gate can help at all."),

        # ------------------------------------------------------------------ S28
        sec("S28. How can everything be reproduced?",
            "IEEE Access expects reproducibility of the analyses; the repository contains the pipeline, "
            "the frozen artifacts, and the tests used for every number in this notebook.",
            "The command list, artifact checks, and the verification guarantees."),
        code("""
commands = pd.DataFrame([
    ("canonicalize + split + index", "python experiments/00_canonicalize.py; 01_split.py; 02_build_index.py"),
    ("feedback databases", "python experiments/03_build_feedback.py  (LLM judge; cached)"),
    ("generated evaluations", "python experiments/04_evaluate.py --method M2_team --split eval  (etc.)"),
    ("independent metrics", "python experiments/08_rescore.py --runs <globs>"),
    ("independent answer judge", "python experiments/09_llm_judge_pairwise.py --judge-model <non-OpenAI>"),
    ("calibration + proxy + ladder", "python experiments/10_feedback_calibration.py; 07_validate_proxy.py; 11_retriever_ladder.py"),
    ("gate study", "python experiments/17_gate_study.py --tag general"),
    ("magnitude policy", "python experiments/18_magnitude_policy.py"),
    ("notebooks", "python notebooks/build_paper_companion.py; python notebooks/validate_ieee_notebooks.py --execute"),
    ("tests", "pytest -q  (74 unit tests)"),
], columns=["step", "command"])
display(commands)
status = L.artifact_status()
print(f"Artifact checks available: {int(status['available'].sum())} / {len(status)}")
reg = L.registry()
print(f"Registered runs: {len(reg)}")
print("Every summary records its generation regime (model, system prompt, cache hits/misses, warm flag);")
print("every run folder carries an immutable manifest, and results/registry.csv indexes all runs.")
"""),
        md("""
**Verification guarantees.** All numbers in this notebook are produced by executing the cells above
against the frozen artifacts in `results/`. Thresholds and configurations were selected on dev and
frozen before eval; the gate and magnitude models use grouped folds by ticket; confidence intervals
resample tickets. The notebook itself is executed end-to-end before submission, so the outputs shown
are the outputs of the committed code and data.
"""),
    ]
