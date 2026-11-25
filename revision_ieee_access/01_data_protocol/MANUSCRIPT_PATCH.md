# Manuscript patch — data regime, generated results, R1-M7

Copy-pasteable text for the revision. Numbers: `tables/r1m7_split_overlap.csv`,
`tables/r1m7_retrieval_panels.csv`, `tables/r1m7_seed_robustness.csv`,
`tables/generated_deltas.csv`, `tables/generated_protocol_contrast.csv`.
Model, temperature and system prompt are unchanged from the submitted run
(regime `51fd9f4b37e900bb`).

## A. New methods subsection (after the corpus description)

```latex
\subsection{History-only evaluation and structural disjointness}
\label{sec:data-regime}

All revision experiments separate three things that the submitted evaluation
conflated: the candidate archive, the scored feedback rows, and the reference
reply of the evaluated ticket. For a query $q$ in partition $P$, the retriever may
return only tickets in the historical partition $H$ (the training partition), and
$q \notin H$. The dense index is rebuilt from the frozen embeddings of $H$; BM25
document frequencies are estimated on $H$ only; hybrid fusion and cross-encoder
re-ranking operate on the same restricted pool. A feedback row $(q',c',s)$ is
admissible only if $q' \in H$ and $c' \in H$, with $q' \neq c'$. Scope base rates
$\pi_s$ are estimated from exactly the admissible rows, so the prior and the counts
it centres cannot draw on a different evidence set. The reference reply of the
evaluated ticket is never read by retrieval, feedback construction, or generation.

Structural disjointness is defined by connected components over ticket signatures,
not by a conjunction of hashes. We report two versioned rules. \emph{Duplicate}
links tickets with an identical normalized full reply or an identical normalized
title and subcategory. \emph{Prefix} additionally links tickets whose replies share
the first 300 normalized characters, using both the submitted raw-cut definition
and a normalized-cut definition. Groups are assigned atomically to train, dev, or
eval. Neither rule is a semantic proof that two tickets follow the same procedure;
we therefore call the resulting panels \emph{structurally} disjoint and treat
procedure identity as a proxy, to be validated by domain experts.

Routing labels are the recorded resolving team and the reply-derived intent class,
held fixed across panels so that the manipulated factor is the data boundary.
Results are consequently conditional on those labels and are not a claim that they
are observable when a ticket arrives. Model, temperature and system prompt are
identical to the submitted runs; only the data boundary and the evidence source
change.
```

## B. Results

```latex
\subsection{What survives procedure-disjoint history}
\label{sec:disjoint}

Table~\ref{tab:overlap} shows what the submitted splits guaranteed. The submitted
random split lets 58.3\% of evaluation tickets share a reply opening with
training, which reproduces the reviewer's estimate, but only 14.8\% share an
identical full reply. The submitted ``procedure-disjoint'' split is not
procedure-disjoint under any component-wise definition: 12.8\% of its evaluation
tickets share an identical reply and 53.3\% share a reply opening with training.
The submitted combined hash required reply, title, and subcategory to match
simultaneously, which does not keep related tickets together.

\begin{table}[t]
\centering
\caption{Evaluation tickets whose train partition contains a ticket with the same
structural signature. ``Duplicate union'' is the connected-component rule over
full reply or title+subcategory.}
\label{tab:overlap}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Split & Full reply & Reply opening & Title+subcat & Duplicate union \\
\midrule
Submitted random (398)      & 59 (14.8\%) & 232 (58.3\%) & 61 (15.3\%) & 116 (29.1\%) \\
Submitted disjoint (390)    & 50 (12.8\%) & 208 (53.3\%) & 38 (9.7\%)  & 88 (22.6\%) \\
Duplicate panel (398)       & 0           & 193 (48.5\%) & 0           & 0 \\
Prefix panel (427)          & 0           & 0            & 0           & 0 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:generated-disjoint} reports generated-answer effects under the
corrected regimes with the frozen configuration (empirical Bayes, $\kappa=2$,
intersection scope, absolute scaling). Blind feedback costs $-0.0444$ when
evaluation tickets share procedures with history and becomes indistinguishable
from zero under strict disjointness ($+0.0008$, p = .86). The direct protocol
contrast falls from $+0.0549$ to $+0.0085$, but remains positive
(p = .032). The conditioned effect is small in every regime
($+0.0093$ to $+0.0149$); on the strict panel its cosine interval excludes zero
while the rank test is marginal (p = .066) and ROUGE-L is not significant. We
therefore describe it as a small, metric-dependent effect, not as a confirmed
gain. A reply-similarity baseline that uses no LLM judge matches the judge on
overlapping history ($+0.0143$ versus $+0.0105$) but not on strict history
($+0.0026$ versus $+0.0093$).

\begin{table}[t]
\centering
\caption{Generated-answer cosine difference against baseline (paired, per ticket)
under three data regimes and three evidence sources. Brackets are 95\% paired
bootstrap intervals; p-values are two-sided Wilcoxon.}
\label{tab:generated-disjoint}
\begin{tabular}{@{}lccc@{}}
\toprule
Panel & Conditioned & Blind & Reply similarity \\
\midrule
Random    & $+0.0105$ $[+0.0005, +0.0212]$ & $-0.0444$ $[-0.0606, -0.0285]$ & $+0.0143$ $[+0.0047, +0.0242]$ \\
Duplicate & $+0.0149$ $[+0.0059, +0.0246]$ & $-0.0214$ $[-0.0367, -0.0073]$ & $+0.0081$ $[+0.0016, +0.0152]$ \\
Prefix    & $+0.0093$ $[+0.0026, +0.0169]$ & $+0.0008$ $[-0.0063, +0.0084]$ & $+0.0026$ $[-0.0039, +0.0097]$ \\
\bottomrule
\end{tabular}
\end{table}

Median paired differences are zero in every cell because 55\% of strict-panel
tickets and 23\% of random-panel tickets keep an unchanged answer. The per-ticket
distribution is zero-inflated; the pre-specified statistic is the mean paired
difference with a ticket-level bootstrap interval, reported together with the share
of changed tickets. This also explains why a mean whose interval excludes zero can
coexist with a non-significant rank test (for example blind feedback on the
duplicate panel: $-0.0214$ $[-0.0367, -0.0073]$ with p = .66).

Three families were fixed before adjustment (conditioned and blind across panels;
reply similarity; protocol contrast) and Holm-corrected within family. Blind harm
on the random panel survives (.0009). The conditioned effect on random data does
not (.1165), the duplicate effect does (.0106), and the strict effect does not
(.1965). The protocol contrast survives on all panels (strict .0322), but the
strict estimate ($+0.0085$) lies below its 80\% minimum detectable effect
(0.0101): the design is not powered to estimate an effect of that size precisely.
We report it as small and imprecisely estimated, not as evidence of absence.

For retrievers, we repeated the comparison with the same evidence and lift under
pool-relative scaling. The feedback gain is then similar for dense, lexical and
hybrid retrieval (random: $+0.064$, $+0.060$, $+0.033$; strict: $+0.009$,
$+0.006$, $+0.008$). The apparent dependence on retriever strength in the
submitted paper is a scaling artefact: absolute lift units are cosine units and
are not meaningful on BM25 scores.

At retrieval level the legacy Laplace lift shows the same pattern more strongly:
its conditioned top-1 proxy falls from $+0.034$ to $+0.002$ and is significant in
0 of 5 group-aware splits on the strict panel, while the calibrated lift keeps a
small positive point estimate ($+0.009$, 2 of 5). The retrieval proxy ranks
configurations correctly but does not certify significance; the generated
intervals and the independent metric do.

For the decision policy, always-on blind feedback is already neutral on the strict
panel ($+0.0008$); the oracle bound (apply feedback only when it helps) is
$+0.0190$, so any gate can add at most about $0.018$ and only with outcome
knowledge. On the random panel, where always-on is harmful ($-0.0444$), the oracle
bound is $+0.0258$: the gate matters mainly where the prior is uncalibrated, which
supports the narrower claim in the revision. End-to-end generation cost is
0.48–0.87 seconds per request on CPU, with two requests per ticket for baseline and
feedback; feedback-database preparation is a separate one-off cost.

These are still similarity-based outcomes. Section~\ref{sec:human} reports the
blinded expert assessment of procedural correctness on the frozen outputs.
```

## B2. Human evaluation

```latex
\subsection{Expert assessment of procedural correctness}
\label{sec:human}

Two GFT support engineers were invited to rate the same blinded sample. At the time
of writing one expert (E1) has completed all 105 items; the second is pending. We
report E1 descriptively and will add inter-rater agreement when E2 completes.

The sample contains three arms of 35 tickets, changed pairs only (the feedback
answer differs from the baseline), stratified by resolving team, with A/B order
randomised and balanced within each arm. For every reply the expert marked correct,
partially correct, incorrect or cannot judge, flagged a wrong or unsafe step and a
missing step, and then chose the better reply, with equal, both incorrect and cannot
judge as explicit options. Strict correctness counts only \emph{correct}; lenient
correctness also counts \emph{partially correct}. Intervals are paired bootstrap
intervals over tickets; p-values are exact McNemar tests on discordant pairs.

\begin{table}[t]
\centering
\caption{Expert assessment (E1, 105 items, 35 per arm). Strict = correct only.}
\label{tab:human}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Arm & Baseline & Feedback & $\Delta$ [95\,\% CI] & McNemar $p$ \\
\midrule
Random, blind        & 0.647 & 0.794 & $+0.147$ $[0.000, +0.294]$ & .25 \\
Strict, blind        & 0.545 & 0.515 & $-0.030$ $[-0.121, +0.061]$ & 1.00 \\
Strict, conditioned  & 0.559 & 0.559 & $0.000$ $[-0.088, +0.088]$ & 1.00 \\
\bottomrule
\end{tabular}
\end{table}

Three results matter. First, on the strict panels the expert sees no benefit and no
harm: both intervals include zero and the point estimates are within $\pm0.03$,
matching the small automatic effects there. Second, the random arm shows a raw
$+0.147$, but the feedback replies are 173 characters longer and the expert
preference tracks that difference (Spearman $+0.63$); among pairs whose lengths
differ by at most 100 characters the effect is exactly zero. The apparent benefit on
familiar procedures is a preference for longer, more detailed replies, not a
demonstrated procedural improvement. Third, the expert and the cosine metric point
in opposite directions on that arm (expert $+0.147$, automatic $-0.088$; Spearman
$-0.48$, AUC $0.22$). Neither measure is a clean criterion: cosine rewards
similarity to one recorded reply, the expert is sensitive to length.

Preferences were equally informative: 42 of 105 items were rated equal and 32 both
incorrect. A raw A/B preference alone would have hidden a third of the sample as
“both wrong”. With 2–7 discordant pairs per arm the study is a validity check, not
a confirmatory test; we report intervals and do not treat a non-significant result
as evidence of absence.
```

When E2 completes, add the agreement table (percent agreement, Cohen's kappa with
bootstrap interval, weighted kappa, PABAK/AC1) and a combined per-item estimate.
The same pipeline is `experiments/33_human_eval.py`; no reprocessing of items is
needed.

## C2. Response — human evaluation (R1-M8, R2-1, R2-16)

```text
We added an independent expert assessment of procedural correctness on a blinded
sample of 105 tickets, split into three arms: familiar procedures with blind
feedback, new procedures with blind feedback, and new procedures with
conditioned feedback. Each reply was rated for correctness, flagged for wrong or
unsafe steps and missing steps, and compared pairwise, with ties, both-incorrect and
cannot-judge kept explicit. One expert has completed the sample; the second is
pending, so we report the first descriptively and will add inter-rater agreement
when it is available.

The expert does not confirm the automatic harm: on the random arm the expert
prefers the feedback replies, while the cosine metric penalises them. The
preference is explained by answer length: the feedback replies are 173 characters
longer, the length-preference correlation is +0.63, and among length-matched pairs
the effect is zero. On the strict panels the expert sees neither benefit nor harm,
consistent with the small automatic effects. We therefore report that similarity to
one recorded reply and expert procedural correctness diverge, and that the expert
judgement is itself sensitive to answer length; neither is treated as a clean
criterion.
```

## C. Response-to-reviewer paragraph (R1-M7)

```text
We agree, and the revision changes the data regime rather than only moving a row.
First, we verified what the submitted split guaranteed: the combined hash required
simultaneous equality of reply prefix, title, and subcategory, so the
"procedure-disjoint" split still let 12.8% of eval tickets share an identical
reply and 53.3% share a reply opening with training. Second, we rebuilt the
control with versioned structural groups (identical full reply OR identical
title+subcategory; and a conservative variant that also merges shared reply
openings), assigned groups atomically, and restricted the candidate index, BM25
IDF, feedback rows, and priors to the training partition. Third, we regenerated
every answer with the identical model, temperature and prompt. On the strictest
panel, blind harm disappears (-0.0444 to +0.0008, p = .86), the protocol contrast
shrinks six-fold but stays positive (+0.0085, Holm p = .032), and the conditioned
effect is small and metric-dependent (+0.0093 on cosine with p = .066 by rank test,
non-significant ROUGE-L, and below the study's 80% minimum detectable effect).
With the same lift under pool-relative scaling, the feedback gain is similar
across dense, BM25 and hybrid retrieval, so the earlier retriever-strength
conclusion was a scaling artefact. The disjoint panels therefore belong in the
main results, and the "922 identical-reply tickets" claim is corrected: 922
tickets share a reply opening, while 238 share an identical full reply.
```

## D. Claim corrections required elsewhere

| Location | Change |
|---|---|
| Title | Drop the causal "opposite sign" promise; frame as procedure overlap and calibration. |
| Abstract | Harm removed under strict disjointness; conditioned effect small and metric-dependent; no confirmed calibrated gain. |
| RQ1 | Ask whether any benefit survives procedure-disjoint history; answer with generated Table~\ref{tab:generated-disjoint}. |
| Section IV-B | Correct the two-vote symmetry statement (R1-M4); cite the empirical vote distribution for the asymmetry. |
| Table 2 | Add the zero-inflation note and the share of changed tickets; report median CI as [0, 0] where applicable. |
| Section V-C / IX | Replace the retriever-strength conclusion with the matched-configuration evidence or soften it. |
| Gate section | Present the oracle bound and the blind-only table; say the gate matters where always-on is harmful. |
| Cost section | Two requests per ticket, 0.5–0.9 s per request end-to-end; database preparation separate. |
| Section IX | Replace the five-step recipe with hypotheses conditioned on structural history overlap. |
| Contribution | Diagnostic of saturation + evidence that part of the measured gain is procedure overlap + calibrated null, not a new reranking method. |
