"""Cost tracking example: wrap a model with track_costs and print a session report.

Uses an in-file fake model so it runs offline; swap FakeModel for any
BaseLLM-compatible model (OpenAIModel, ClaudeModel, ...) in real code.
"""

import asyncio

from multimind.observability import (
    Budget,
    BudgetExceededError,
    CostTracker,
    cost_summary,
    track_costs,
)


class FakeModel:
    """Offline BaseLLM-compatible stand-in with blended per-token pricing."""

    PROVIDER_NAME = "FakeProvider"

    def __init__(self, model_name="fake-4o-mini", cost_per_token=0.000000375):
        self.model_name = model_name
        self.cost_per_token = cost_per_token

    async def generate(self, prompt, **kwargs):
        return f"Canned answer to: {prompt[:40]}"

    async def chat(self, messages, **kwargs):
        return "Canned chat reply covering all points raised."

    async def generate_stream(self, prompt, **kwargs):
        for word in "streamed canned answer token by token".split():
            yield word + " "


async def main():
    tracker = CostTracker()  # add jsonl_path="costs.jsonl" to persist records

    cheap = track_costs(FakeModel("fake-4o-mini"), tracker=tracker, tag="drafting")
    fancy = track_costs(
        FakeModel("fake-4o", cost_per_token=0.00000625), tracker=tracker, tag="review"
    )

    await cheap.generate("Summarize the quarterly report in three bullet points.")
    await cheap.chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Draft a short status update for the team."},
        ]
    )
    async for _ in cheap.generate_stream("Stream a haiku about observability."):
        pass
    await fancy.generate("Review the draft above for tone and accuracy.")

    print(tracker.report())
    print()
    print("By tag:", tracker.by_tag())

    # Budget: the ceiling is enforced before the next call is dispatched.
    budget = Budget(max_cost=0.00001, period="session")
    guarded = track_costs(FakeModel("fake-4o", cost_per_token=0.00000625), budget=budget)
    await guarded.generate("First call fits under the budget.")
    try:
        await guarded.generate("Second call is blocked before dispatch.")
    except BudgetExceededError as exc:
        print(f"\nBudget enforced: {exc}")

    # The budget-guarded model used the global default tracker (no tracker arg).
    print("\nDefault (session) tracker:")
    cost_summary()


if __name__ == "__main__":
    asyncio.run(main())
