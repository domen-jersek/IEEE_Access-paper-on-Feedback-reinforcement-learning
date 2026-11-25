# Human evaluation — what we did and what we found

*Written to be read without prior knowledge of the paper or the dataset.*

## The short version

A support system suggests a first reply to an IT ticket. We asked a GFT support
expert to compare two versions of that suggestion, blind, on 105 real tickets. On
tickets whose procedure the system had already seen, the expert preferred the
version that uses feedback from earlier tickets — but that version simply wrote
longer replies, and when replies are the same length the preference disappears. On
tickets with a procedure the system had **not** seen, the expert found no difference
at all. The automatic similarity score used in the paper disagrees with the expert
on the first group: it penalises the longer replies that the expert prefers.

## What was compared

An IT ticket arrives. The system retrieves five similar historical tickets and a
language model writes a first reply from them. Two retrieval versions were compared:

- **Baseline:** rank historical tickets by semantic similarity to the new ticket.
- **Feedback:** additionally boost tickets that received good scores in earlier
  ticket–candidate comparisons.

The feedback scores come in two kinds:

- **Blind:** the score is based on the ticket text only, as a real user could
  provide it.
- **Conditioned:** the score is based on the ticket text *and* the recorded
  resolution of the historical ticket. This is an upper bound: a user cannot supply
  the resolution of the ticket they are currently asking about.

Two data situations were compared:

- **Familiar procedures:** the historical base contains tickets very close to the
  evaluated one, so the system can reuse a procedure it has already seen.
- **New procedures:** such near-duplicates are removed from the historical base, so
  the system has to generalise.

The three groups below combine these two dimensions.

## What the expert did

One GFT support engineer rated 105 tickets, 35 per group. For each ticket the
application showed two candidate replies, A and B, in random order, without saying
which system produced which. For each reply the expert chose **correct**,
**partially correct**, **incorrect** or **cannot judge**, marked whether it
contained a wrong or unsafe step or missed a required step, and then chose the better
reply, with *equal*, *both incorrect* and *cannot judge* available.

**Strict correctness** means the share of replies rated exactly *correct*. It is the
main number below; the lenient version also counts *partially correct*.

## The results

| Group | Tickets | Baseline correct | Feedback correct | Difference |
|---|---:|---:|---:|---|
| Familiar procedures / blind feedback | 34 | 64.7 % | 79.4 % | +14.7 points (interval 0.0 to +29.4) |
| New procedures / blind feedback | 33 | 54.5 % | 51.5 % | −3.0 points (interval −12.1 to +6.1) |
| New procedures / conditioned feedback | 34 | 55.9 % | 55.9 % | 0.0 points (interval −8.8 to +8.8) |

1. **On new procedures there is no difference.** Both intervals include zero. The
   expert does not confirm that feedback helps, and does not confirm that it harms.
2. **The familiar-procedure gain is a length effect.** The feedback replies in that
   group were 173 characters longer on average, and the expert's preference tracked
   length (correlation +0.63). Among pairs whose lengths differ by at most 100
   characters, the difference is **exactly zero**. The expert was not instructed to
   reward length; this is a property of the task, and it matters.
3. **The automatic metric and the expert disagree.** On familiar procedures the
   automatic cosine score says the feedback replies are worse (−0.088), while the
   expert says they are better (+0.147). The metric rewards closeness to one recorded
   reply; the expert rewards a reply that looks more complete. Neither is a clean
   measure of procedural correctness.
4. **“Both wrong” was common.** Of 105 tickets, 42 were rated equal and 32 were rated
   both incorrect. A simple A/B preference would have hidden a third of the sample as
   “both wrong”. That is why each reply was rated separately.

## What this means

- The paper's claim that blind feedback **harms** replies is **not supported** by the
  expert.
- The paper cannot claim that feedback **helps** either: on new procedures the
  difference is zero, and on familiar procedures it is explained by reply length.
- Similarity to one recorded reply and expert judgement measure different things.
  The paper must say this explicitly and must not treat either as ground truth.

## Limits

- **One expert.** A second GFT expert is being arranged. Until then we cannot report
  how well two experts agree, and the result is one person's assessment.
- **105 tickets, 35 per group.** The study can detect only large differences; the
  intervals above are the honest statement, not the p-values.
- **Changed pairs only.** The sample contains tickets where feedback changed the
  reply; unchanged pairs would show the same reply twice.
- **Procedural correctness, not resolution.** The expert judges whether the reply
  describes the right procedure, not whether the user's problem was ultimately
  solved.

## Status

First expert complete. Second expert pending; when the ratings arrive, the same
analysis adds inter-rater agreement and a combined estimate. The manuscript section
will state the one-expert limitation if the second does not arrive in time.

## Where to read more

- `HUMAN_EVAL_RESULTS.md` — full tables, sensitivity analyses, reviewer mapping.
- `../../notebooks/09_human_evaluation.ipynb` — executed notebook with all figures.
- `results/alberto/` — aggregate tables and figures (no ticket text).
- `STUDY_DESIGN.md` — the design, fixed before the first rating.
