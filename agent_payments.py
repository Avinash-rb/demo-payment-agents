"""Agent 1: tool calling + conversation memory (checkpointer).
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common import TRANSACTIONS, chat, get_model


@tool
def get_transaction(txn_id: str) -> dict:
    """Look up a payment transaction by ID (e.g. T1001): amount, status, settled amount, date and shortfall."""
    txn = TRANSACTIONS.get(txn_id.upper())
    if txn is None:
        return {"error": f"No transaction {txn_id}"}
    # Computed here so the LLM never does arithmetic; it only reports this value.
    return {**txn, "shortfall": round(txn["amount"] - (txn["settled_amount"] or 0), 2)}


@tool
def list_settlement_issues() -> list[str]:
    """List transactions that are not settled, or settled for a different amount than captured."""
    return [
        txn_id for txn_id, t in TRANSACTIONS.items()
        if t["settled_amount"] is None or t["settled_amount"] != t["amount"]
    ]


def build_agent(checkpointer=None):
    return create_agent(
        model=get_model(),
        tools=[get_transaction, list_settlement_issues],
        system_prompt=(
            "You are a payments support assistant. Use the tools for every fact and never guess amounts. "
            "Report amounts and differences exactly as the tools return them; never calculate them yourself. Amounts are in INR."
        ),
        checkpointer=checkpointer,
        name="payments_agent",
    )


if __name__ == "__main__":
    chat(build_agent(InMemorySaver()))
