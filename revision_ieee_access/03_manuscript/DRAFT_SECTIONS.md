# Draft sections for the revision

Working draft, not the manuscript. LaTeX blocks are ready to be adapted into
`access.tex`. Numbers come from `tables/generated_deltas.csv`,
`tables/generated_protocol_contrast.csv`, `tables/r1m7_split_overlap.csv`,
`tables/r1m7_seed_robustness.csv`, `tables/power_multiplicity.csv`,
`tables/matched_retrievers.csv`, `tables/decision_headroom.csv`,
`tables/generation_cost.csv`, `LIFT_DIAGNOSTICS.md`, and
`../02_human_study/HUMAN_EVAL_RESULTS.md`.

## Abstract

```latex
Feedback from earlier ticket--candidate pairs is widely used to rerank retrieval
candidates in retrieval-augmented generation for IT support. We show that the
measured benefit depends less on the judge than on whether evaluation tickets share
procedures with the historical base. Using 1,595 enterprise tickets, we rebuild the
evaluation so that retrieval, feedback and priors use only the training partition,
and compare familiar-procedure, exact-duplicate-removed and strict
procedure-disjoint regimes. Under strict disjointness, blind feedback no longer
harms ($-0.044$ to $+0.001$) and the conditioned effect is small ($+0.009$),
metric-dependent and not robust to multiplicity correction. A free reply-similarity
baseline matches the LLM judge only when procedures overlap. A blinded expert
assessment of 105 tickets finds no difference on new procedures; on familiar
procedures the expert's apparent preference for feedback is explained by answer
length, and expert and cosine metric disagree. Gains reported on overlapping splits
are therefore partly borrowed from procedure overlap; calibration removes harm
without a confirmed gain; and similarity to one recorded reply is not procedural
correctness.
```

## 1 Introduction

```latex
\section{Introduction}

Support organizations answer thousands of tickets, and many restate problems that
have been solved before. Retrieval-augmented generation (RAG) makes reuse natural:
retrieve the most similar historical tickets and let a language model write the first
reply from their resolutions \cite{b1,b30}. The implicit assumption is that textual
similarity tracks procedural similarity. It often does not.

Enterprise ticket streams make this failure common and its measurement hard.
Procedures are long-tailed and organized by team and intent, so many tickets have few
close historical neighbours, while the closest neighbours may belong to a different
procedure that shares vocabulary. Unlike search or recommendation, the domain has no
click stream: the only observable signal of usefulness is whether a procedure matches
the one that resolved the ticket. We therefore simulate the missing feedback with
large language model (LLM) judges and treat the judge's information as an
experimental variable.

The submitted version of this study reported that the same fine-scope reranking
helps under a judge that sees the historical resolution (conditioned) and harms under
a judge that sees only the ticket (blind). The present revision asks a prior
question: \emph{does any of that gain survive when the evaluation ticket does not
share a procedure or a reply template with the historical base?} The answer changes
the reading of the earlier result. A large part of the measured benefit comes from
retrieval targets that are near-duplicates of training tickets, and it disappears
when those are removed.

We rebuild the evaluation around three data regimes. In the \emph{familiar} regime
the historical base may contain a ticket with the same procedure or reply template.
In the \emph{duplicate} regime tickets with an identical normalized full reply or an
identical title and subcategory are kept out of the evaluation side. In the
\emph{strict} regime tickets that even share the first 300 reply characters are kept
apart. Every regime uses a training-only candidate index, training-only BM25
statistics, feedback rows with both endpoints in training, and priors estimated from
exactly those rows.

The results are consistent across retrieval-level and generated-answer analyses.
Under the strict regime the blind harm disappears ($-0.044$ on the familiar regime,
$+0.001$ on the strict one), the conditioned effect is small ($+0.009$), and its
interval includes values that the design cannot distinguish from zero after
multiplicity correction. A reply-similarity baseline that uses no LLM judge matches
the judge on the familiar regime ($+0.014$ versus $+0.011$) but not on the strict one
($+0.003$ versus $+0.009$). A blinded expert assessment of 105 tickets finds no
difference on new procedures; on familiar procedures the expert prefers the feedback
replies, but those replies are 173 characters longer and the preference disappears
when lengths are matched.

The paper makes four contributions. First, it treats the data regime as the
experimental variable and shows that the submitted ``procedure-disjoint'' split was
not procedure-disjoint: 12.8\,\% of its evaluation tickets shared an identical reply
and 53.3\,\% a reply opening with training. Second, it isolates the components of the
calibration repair under a matched budget: tuned Laplace is still at or below zero on
strict history, pool scaling reduces the lift magnitude, and scope accounts for about
half of the apparent improvement. Third, it reports a calibrated outcome: harm is
removed, but no gain is confirmed; the retriever-strength conclusion of the submitted
paper is a scaling artefact. Fourth, it adds an expert validity check showing that
similarity to one recorded reply and procedural correctness diverge, and that expert
judgement is itself sensitive to answer length.

Section~\ref{sec:related} positions the work, Section~\ref{sec:data} defines the data
regimes, Section~\ref{sec:method} the method, Sections~\ref{sec:legacy}--\ref{sec:human}
the results, and Sections~\ref{sec:discussion} and~\ref{sec:conclusion} the discussion
and conclusion.
```

## 2 Related work — additions

Keep the existing subsections on RAG/reranking, learning to rank, LLM-as-judge and
calibration. Add the following paragraph and replace the “Relationship to prior
work” subsection with the text in `PRIOR_WORK_TABLE.md`.

```latex
\subsection{Related threads added in this revision}

Three literatures place the result. Work on LLM relevance judgments shows that
models can approximate human relevance assessments but carry position, verbosity and
self-preference biases \cite{b35,b36,b37}; our expert study adds a concrete case in
which the automatic judge and the human disagree in a predictable direction. Work on
counterfactual and unbiased learning to rank makes the logging policy explicit and
corrects for exposure and position bias \cite{b38}; our setting has no exposure bias,
because every candidate in the pool is judged, but it has a different pathology: the
evidence is sparse, zero-heavy, and its conversion into a lift amplifies the base
rate. Work on ranking items with few ratings---Bayesian averages and Wilson
intervals---is the natural home for the empirical-Bayes repair \cite{b39,b40}; our
contribution is not the smoother itself but the demonstration that it removes harm
without creating gain once procedure overlap is controlled.

RAG evaluation separates retrieval quality from answer quality \cite{b1} and has
recently produced automated suites for faithfulness and answer relevance
\cite{b41,b42}. Those suites score an answer against retrieved context; they do not
tell whether the answer describes the procedure that resolves the ticket. Our expert
study is a reminder that the two questions are different.
```

New references to add and verify (tentative):
`b35` Faggioli et al., perspectives on LLMs for relevance judgment (ICTIR 2023);
`b36` Zheng et al., MT-Bench/Chatbot Arena (NeurIPS 2023);
`b37` Wang et al., LLMs are not fair evaluators (ACL 2024);
`b38` Joachims, Swaminathan, Schnabel, unbiased learning-to-rank with biased feedback
(WSDM 2017);
`b39` Wilson, probable inference (JASA 1927);
`b40` a Bayesian-average or Wilson-interval ranking reference (verify before use);
`b41` Es et al., RAGAS (EACL 2024 demo);
`b42` Saad-Falcon et al., ARES (NAACL 2024). Also add the MiniLM source
(Reimers and Gurevych, EMNLP 2019) and check the venue versions of the arXiv entries.

## 3 Data regimes

```latex
\section{Data regimes and structural disjointness}
\label{sec:data}

\subsection{Structural groups}
We define two versioned rules over normalized signatures. The \emph{duplicate} rule
links tickets with an identical normalized full reply or an identical normalized
title and subcategory. The \emph{prefix} rule additionally links tickets whose
replies share the first 300 normalized characters, using both the submitted raw-cut
definition and a normalized-cut definition. Groups are the connected components of
the resulting graph; an empty signature never links tickets. The duplicate rule yields
1,218 components (largest 83); the prefix rule yields 676 (largest 299). These are
structural proxies, not semantic proof that two tickets follow the same procedure; we
therefore call the panels structurally disjoint and treat procedure identity as a
proxy.

\subsection{Group-aware splits}
Whole groups are assigned atomically to train, dev or eval. The duplicate split has
878/319/398 tickets; the prefix split has 856/312/427. The prefix split cannot hit
878/319/398 without cutting the 299-ticket component, which we refused to do. Five
seeds were generated; the primary analyses use seed 42 and the others are reported as
robustness. No outcome enters the assignment.

\subsection{Allowed history, candidates and evidence}
For a query $q$ in partition $P$, the retriever may return only tickets in the
training partition $H$, and $q \notin H$. The dense index is rebuilt from the frozen
embeddings of $H$; BM25 document frequencies are estimated on $H$ only; hybrid fusion
and cross-encoder re-ranking operate on the same restricted pool. A feedback row
$(q',c',s)$ is admissible only if $q' \in H$ and $c' \in H$, with $q' \neq c'$. The
scope base rates $\pi_s$ are estimated from exactly the admissible rows, so the prior
and the counts it centres cannot draw on a different evidence set.

\subsection{Routing metadata}
Routing uses the recorded resolving team and the reply-derived intent class, held
fixed across panels so that the manipulated factor is the data boundary. The intent
class is a mapping of a label extracted from the historical first reply, not a
prediction from the incoming ticket. Results are therefore conditional on those
labels; we return to this in the limitations.
```

Table 1 (overlap) from `MANUSCRIPT_PATCH.md` can be reused verbatim.

## 4 Method

```latex
\section{Method}
\label{sec:method}

\subsection{Retrieval}
For each query the retriever returns a pool of $K=100$ tickets from the allowed
history $H$, and the generator receives the top $k=5$. The primary retriever is the
frozen dense MiniLM index; BM25 and hybrid reciprocal-rank fusion are used in the
matched comparison of Section~\ref{sec:repair}. We do not compare retriever
strength under different lift configurations, because absolute lift units are not
comparable across score geometries.

\subsection{Lift}
Let $\mathrm{pos}$ and $\mathrm{neg}$ be the number of positive and negative judge
scores for candidate $c$ in scope $s$, and $n=\mathrm{pos}+\mathrm{neg}$. The
legacy lift is
\begin{equation}
p=\frac{\mathrm{pos}+\alpha}{n+\alpha+\beta},\qquad
\ell_{\mathrm{L}}=\operatorname{clamp}_{[-c,c]}\!\left(
(p-\tfrac12)\,\min\!\left(1,\tfrac{n}{2}\right)m\right),
\label{eq:legacy}
\end{equation}
with $\alpha=\beta=1$, $m=0.80$ and $c=0.20$. The empirical-Bayes lift is
\begin{equation}
\bar p_s=\pi_s,\quad a=\kappa\pi_s,\quad
p_{\mathrm{post}}=\frac{\mathrm{pos}+a}{n+\kappa},\qquad
\ell_{\mathrm{EB}}=\operatorname{clamp}_{[-c,c]}\!\left(
(p_{\mathrm{post}}-\pi_s)\,m\right),
\label{eq:eb}
\end{equation}
where $\pi_s$ is the mean judge score of scope $s$ over the admissible training
rows, with a global fallback when a scope has no rows. The finalist uses
$\kappa=2$. Two differences from Eq.~\ref{eq:legacy} matter: the neutral point is
the scope base rate rather than $0.5$, and the $n/2$ ramp is absent because the
centring term already attenuates sparse evidence. The legacy ramp is one of the
factors varied in the ablation, not an incidental implementation detail.

Pool-relative scaling expresses the bounded lift in retrieval-score units:
\begin{equation}
\ell'=\ell\cdot\frac{\lambda\,\sigma_{\mathrm{pool}}}{c},
\label{eq:pool}
\end{equation}
where $\sigma_{\mathrm{pool}}$ is the standard deviation of the retrieval scores in
the pool and $\lambda=1$ in the finalist. A candidate with no evidence receives
$\ell=0$; the clamp is applied before scaling. The parameters $\kappa$ and
$\lambda$ were selected on dev over $\{2,5,10,20,50\}$ and $\{0.25,0.5,1,2,4\}$,
respectively; the per-cell values are in the provenance table.

\subsection{Scopes}
The candidate lift is routed by the recorded resolving team and the reply-derived
intent class. We evaluate Global, Team, Class, Team$\cap$Class and Backoff, with
Team$\cap$Class as the finalist. The labels are held fixed across panels so that
the manipulated factor is the data boundary. They are recorded outcomes, not
predictions from the incoming ticket; results are therefore conditional on them,
and Section~\ref{sec:discussion} states this limitation.

\subsection{Control}
The gate is a logistic model over pool-shape and evidence features, trained with
grouped cross-validation by ticket and frozen on dev. The revision reports it as
risk control, not as a source of gain. The oracle bound applies feedback only when
it helps on the observed outcome and therefore uses information no deployable
policy has; it bounds the headroom, not the performance of a feasible gate. The
expected-value analysis is exploratory and is confined to the appendix because its
rows are two per ticket and are not independent.

\subsection{Evaluation}
The primary outcome is the per-ticket change in cosine between the generated first
reply and the recorded reference reply, averaged over the evaluation tickets.
ROUGE-L is reported as a secondary lexical metric. The expert assessment of
Section~\ref{sec:human} is the independent check of procedural correctness. Because
the study sample contains changed pairs only, we report the conditional effect and
the population effect, the latter as the changed share times the conditional
effect; unchanged pairs have a difference of zero by construction. BGE and
BERTScore are not recomputed for every revision panel: the offline encoder is slow
on CPU, and the independent check in this revision is the expert assessment. The
submitted study's BGE and BERTScore results are retained for the legacy
comparison, and the appendix states the difference in coverage.
```

## 5 The submitted result and its scope

```latex
\section{The submitted result and its scope}
\label{sec:legacy}

In the submitted evaluation, conditioned feedback improved generated first replies
($+0.014$ on eval, $p=.012$; BGE $p=.001$) and blind feedback harmed them
($-0.038$, $p=.017$). The reversal was confined to the legacy Laplace lift at Team
and Team$\cap$Class scope. With the prior repaired, the blind effect was $+0.002$
($p=.86$) and the conditioned effect was also null: Backoff+EB was $+0.005$ on dev
($p=.90$) and $-0.002$ on eval ($p=.29$). No cell in that study produced a confirmed
calibrated gain.

The revision keeps these numbers as a documented legacy finding and asks what the
evaluation allowed. The submitted ``procedure-disjoint'' split combined reply, title
and subcategory into a single hash; simultaneous equality does not keep related
tickets apart. Measured on the canonical corpus, 12.8\,\% of its evaluation tickets
shared an identical reply and 53.3\,\% a reply opening with training. A random split
shared 14.8\,\% and 58.3\,\%. The sign flip is therefore observed in a regime where
the evaluation ticket often has a near-duplicate in the historical base.
```

## 6 Where the gain comes from

```latex
\section{Where the gain comes from}
\label{sec:regime}

Table~\ref{tab:generated} reports generated-answer effects under the three regimes
with the frozen configuration (empirical Bayes, $\kappa=2$, intersection, absolute
scaling). Blind feedback costs $-0.0444$ when procedures overlap and becomes
indistinguishable from zero under strict disjointness ($+0.0008$, $p=.86$). The
direct protocol contrast falls from $+0.0549$ to $+0.0085$ but remains positive
($p=.032$). The conditioned effect is small in every regime ($+0.0093$ to $+0.0149$);
on the strict panel its cosine interval excludes zero while the rank test is marginal
($p=.066$) and ROUGE-L is not significant. A reply-similarity baseline matches the
judge on overlapping history ($+0.0143$ versus $+0.0105$) but not on strict history
($+0.0026$ versus $+0.0093$).

At retrieval level the legacy Laplace lift shows the same pattern more strongly: its
conditioned top-1 proxy falls from $+0.034$ to $+0.002$ and is significant in 0 of 5
group-aware splits on the strict panel, while the calibrated lift keeps a small
positive point estimate ($+0.009$, 2 of 5). The retrieval proxy ranks configurations
correctly but does not certify significance; the generated intervals do.

Three families were fixed before multiplicity adjustment (conditioned and blind
across panels; reply similarity; protocol contrast) and Holm-corrected within family.
Blind harm on the familiar panel survives ($.0009$); the conditioned effect on the
familiar panel does not ($.1165$), the duplicate effect does ($.0106$), and the
strict effect does not ($.1965$). The strict protocol contrast survives ($.0322$) but
lies below its 80\,\% minimum detectable effect ($0.0101$), so we describe it as small
and imprecisely estimated, not as evidence of absence.
```

Use the table from `MANUSCRIPT_PATCH.md` (Table 4 there) or the generated deltas
table.

## 7 Controlled repair

```latex
\section{Controlled repair}
\label{sec:repair}

The submitted comparison changed scope, prior, scaling and the backoff fallback at
once. Table~\ref{tab:ablation} varies one factor at a time on dev. The published
Laplace cell (m=0.80, cap=0.20) ranks 23rd of 25 Laplace configurations on the strict
panel; tuning moves it from $-0.0083$ to $-0.0023$, and pool-relative scaling to
$-0.0042$. All remain at or below zero. Empirical Bayes at $\kappa=2$ improves the
duplicate panel ($+0.050$ absolute) but is indistinguishable from zero on the strict
panel ($-0.001$ absolute, $+0.000$ pooled). The scope change costs about half of the
apparent effect, and backoff does not recover it.

With one feedback configuration and pool-relative scaling, the feedback gain is
similar across dense, lexical and hybrid retrieval: on the familiar panel $+0.033$,
$+0.064$ and $+0.060$; on the strict panel $+0.008$, $+0.009$ and $+0.006$. The
submitted conclusion that stronger retrieval reduces the benefit is therefore a
scaling artefact: absolute lift units are cosine units and are not meaningful on BM25
scores.
```

## 8 Expert evaluation

Use the full text in `MANUSCRIPT_PATCH.md` (B2), with the one-expert caveat and the
second-rater placeholder. Key numbers: strict correctness 0.545/0.515 (new/blind),
0.559/0.559 (new/conditioned), 0.647/0.794 (familiar/blind); length +173 characters
and Spearman +0.63; expert +0.147 versus automatic $-0.088$ on the familiar panel;
42 equal and 32 both incorrect.

## 9 Validation

```latex
\section{Validation}
\label{sec:validation}

\subsection{Secondary metric and seeds}
ROUGE-L agrees in sign with cosine on the familiar and duplicate panels but not on
the strict conditioned panel, where its interval includes zero; we report it as a
secondary metric, not as confirmation. Five group-aware splits were generated for
each structural rule. On the strict panel the legacy Laplace conditioned proxy is
significant in 0 of 5 splits, while the calibrated lift has a small positive point
estimate in 2 of 5; the seed range is reported in the appendix. The generated
panels use seed 42; the other seeds test the retrieval-level conclusion and are not
a second set of generated answers.

\subsection{Power and multiplicity}
The minimum detectable effect at 80\,\% power is 0.0103 for the strict conditioned
cosine effect and 0.0101 for the strict protocol contrast. The observed values are
0.0093 and 0.0085, so the design is not powered to estimate effects of that size
precisely. We describe them as small and imprecise, not as evidence of absence. An
equivalence claim would require a pre-specified margin and a corresponding test;
we do not make one. Three families were fixed before adjustment (conditioned and
blind across panels; reply similarity; protocol contrast) and Holm-corrected within
family. The strict conditioned effect does not survive (p=.1965); the strict
protocol contrast does (p=.0322) but lies below its MDE.

\subsection{Cost}
End-to-end generation takes 0.48--0.87 seconds per request on CPU, with two
requests per ticket for baseline and feedback; a panel of 398--427 tickets takes
356--617 seconds. Feedback-database preparation is a separate one-off cost and is
not included. The calibrated lift adds no model call at inference; its cost is the
offline feedback construction.

\subsection{What is not validated}
The revision does not have an external corpus, a second organisation, real user
feedback, or a completed second expert. BGE and BERTScore are not recomputed for
every revision panel; the independent check is the expert assessment, and it is
one rater at the time of writing. These are limits on generalisation, not
implementation details, and Section~\ref{sec:discussion} states them as such.
```

## 10 Discussion and limitations

```latex
\section{Discussion and limitations}
\label{sec:discussion}

Three practical consequences follow. First, a feedback gain measured on a random
split should be re-measured after removing near-duplicate procedures; otherwise the
number describes the data regime as much as the method. Second, tuning the legacy
baseline matters: an untuned Laplace lift is not a fair comparator, and a tuned one
still does not produce a confirmed gain on strict history. Third, similarity to a
single recorded reply is not procedural correctness: the expert and the cosine metric
diverge, and the expert's preference is itself sensitive to answer length. Evaluation
protocols should therefore state the evidence set, keep the prior aligned with it,
and add an expert check with length control.

The claims are limited to one enterprise corpus, one organisation and one language.
The expert study uses 105 changed pairs, 35 per arm; with 2--7 discordant pairs per arm
it can detect only large differences, and one expert is not a population sample. The
second expert is pending; until then no inter-rater agreement is reported. Procedural
correctness is not ticket resolution, and the automatic metric compares to one
recorded reply. Feedback is simulated by an LLM judge, not collected from users.
Routing labels are the recorded resolving team and the reply-derived intent class,
so the results are label-conditional. Hosted-model drift is bounded by a frozen
generation regime and hashed artifacts, not eliminated.
```

## 11 Conclusion

```latex
\section{Conclusion}

The same reranking can look helpful or harmful depending on what the judge knows, but
the larger determinant in our data is whether evaluation tickets share procedures with
the historical base. When they do not, the blind harm disappears, the calibrated gain
is small and metric-dependent, and a free reply-similarity prior no longer matches the
LLM judge. An expert assessment confirms neither harm nor benefit on new procedures
and shows that the familiar-procedure preference is explained by reply length. We
conclude that feedback reranking gains reported on overlapping splits are partly
borrowed, that calibration removes harm without a confirmed gain, and that similarity
to one recorded reply is not procedural correctness. The practical recipe is a set of
hypotheses to test in other corpora, not a general law.
```

## Editor checks

- Replace the submitted abstract entirely; it contains retracted claims.
- Remove “for free”, “exactly”, “standard” and “oracle” where the revision no longer
  supports them.
- Name the generator and judge models at the start of the methods, with API
  identifiers and dates.
- Add the license and the public-artifact statement.
- Re-check every number against the tables; the submitted values for the legacy
  cells stay, the new values replace the headline.
