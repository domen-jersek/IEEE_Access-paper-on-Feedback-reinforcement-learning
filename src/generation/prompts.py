"""
Prompt templates for first-reply generation.
Single clean format, no legacy variants.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an experienced IT support agent. You write concise, professional, "
    "and procedurally correct first replies to IT support tickets. "
    "Use the retrieved historical tickets as guidance for the correct procedure. "
    "Do not invent steps that are not supported by the retrieved context."
)

GENERATION_TEMPLATE = """\
You are drafting the first reply to an IT support ticket.

TICKET
Title: {query_title}
Description: {query_description}

HISTORICAL TICKETS RETRIEVED FROM THE KNOWLEDGE BASE
{retrieved_context}

Write the first reply to the ticket. Match the procedural content of the most
appropriate retrieved historical reply. If the correct resolution is a form or
a redirect to another service, say so explicitly. Keep the reply professional
and self-contained.

First reply:"""


def build_retrieved_context(candidates) -> str:
    lines = []
    for i, row in enumerate(candidates, 1):
        lines.append(
            f"--- Historical ticket {i} (similarity={row.get('faiss_score', 0.0):.3f}) ---\n"
            f"Title: {row.get('Title_anon', '')}\n"
            f"Reply: {row.get('first_reply', '')}\n"
        )
    return "\n".join(lines)


def build_generation_prompt(query_title: str, query_description: str, candidates) -> str:
    return GENERATION_TEMPLATE.format(
        query_title=query_title,
        query_description=query_description,
        retrieved_context=build_retrieved_context(candidates),
    )


JUDGE_CONDITIONED_TEMPLATE = """\
You are an IT support expert evaluating retrieval quality for a ticketing system.

QUERY TICKET
Title: {query_title}
Description: {query_description}

EXPECTED (GROUND-TRUTH) FIRST REPLY
{reference_reply}

RETRIEVED CANDIDATE
Title: {candidate_title}
First reply: {candidate_reply}

Rate how well the retrieved candidate's reply matches the expected reply in
procedural content (i.e., would following the candidate lead to resolving the
ticket the same way as the expected reply?).
Score from 0.0 to 1.0:
  1.0 = same procedure / directly usable
  0.5 = partially relevant
  0.0 = unrelated procedure
Respond with ONLY the number, no explanation."""


JUDGE_BLIND_TEMPLATE = """\
You are an experienced IT support agent who just received the following ticket.

TICKET
Title: {query_title}
Description: {query_description}

A colleague suggests using the following reply from a similar historical ticket.

HISTORICAL REPLY
{candidate_reply}

Rate how useful this historical reply would be as a starting point for answering
the ticket, based only on the ticket text above (no ground truth is available).
Score from 0.0 to 1.0:
  1.0 = directly usable / resolves the intent
  0.5 = partially relevant
  0.0 = useless / wrong procedure
Respond with ONLY the number, no explanation."""


def build_judge_prompt(
    protocol: str,
    query_title: str,
    query_description: str,
    candidate_title: str,
    candidate_reply: str,
    reference_reply: str = "",
) -> str:
    if protocol == "conditioned":
        return JUDGE_CONDITIONED_TEMPLATE.format(
            query_title=query_title,
            query_description=query_description,
            reference_reply=reference_reply,
            candidate_title=candidate_title,
            candidate_reply=candidate_reply,
        )
    elif protocol == "blind":
        return JUDGE_BLIND_TEMPLATE.format(
            query_title=query_title,
            query_description=query_description,
            candidate_reply=candidate_reply,
        )
    raise ValueError(f"Unknown judge protocol: {protocol}")