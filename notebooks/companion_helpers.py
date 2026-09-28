"""Cell constructors and the fixed explanation grammar for the paper companion.

Every section follows the same grammar so that no claim is left unexplained:

    sec(...)     section header: question, why it matters, what you will see
    idea(...)    the mechanism we expect, in prose, before looking at data
    code(...)    what we do
    howto(...)   how to read the figure/table (axes, units, n, reference lines)
    meaning(...) what it shows, with the numbers
    why(...)     the causal explanation
    artefact(...) could it be an artefact? alternative explanations, controls
    take(...)    one-line takeaway
"""
from __future__ import annotations


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def sec(question: str, why: str, see: str) -> dict:
    return md(f"---\n\n## {question}\n\n**Why it matters.** {why}\n\n**What you will see.** {see}")


def idea(text: str) -> dict:
    return md(f"**The idea.** {text}")


def howto(text: str) -> dict:
    return md(f"**How to read it.** {text}")


def meaning(text: str) -> dict:
    return md(f"**What it shows.** {text}")


def why(text: str) -> dict:
    return md(f"**Why this happens.** {text}")


def artefact(text: str) -> dict:
    return md(f"**Could this be an artefact?** {text}")


def take(text: str) -> dict:
    return md(f"**Takeaway.** {text}")


def qa(question: str, answer: str) -> dict:
    return md(f"**Q: {question}**\n\nA: {answer}")
