import asyncio

from src.evaluation.protocol import _generate_pair


class FakeClient:
    def __init__(self):
        self.prompts = []

    async def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return f"answer-{len(self.prompts)}"


def test_identical_prompts_generate_once():
    client = FakeClient()
    baseline, feedback = asyncio.run(_generate_pair(client, "same", "same"))
    assert baseline == feedback
    assert client.prompts == ["same"]


def test_different_prompts_generate_twice():
    client = FakeClient()
    baseline, feedback = asyncio.run(_generate_pair(client, "baseline", "feedback"))
    assert baseline != feedback
    assert sorted(client.prompts) == ["baseline", "feedback"]
