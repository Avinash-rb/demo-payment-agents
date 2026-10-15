"""Agent 3: middleware. Human approval before refunds, PII masking, call limits, and a custom logger.

Try: "Customer on T1007 says the headphones never arrived, my card was 4111 1111 1111 1111. Please refund."
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    wrap_tool_call,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_payments import get_transaction
from common import chat, get_model


@tool
def issue_refund(txn_id: str, amount: float, reason: str) -> str:
    """Refund the customer for a transaction. A human must approve this before it runs."""
    return f"Refund of INR {amount:,.2f} issued for {txn_id} ({reason})."  # simulated


@wrap_tool_call
def log_tool_calls(request, handler):
    """Custom middleware: runs around every tool call."""
    print(f"   [log] running {request.tool_call['name']}")
    return handler(request)


def build_agent(checkpointer=None):
    return create_agent(
        model=get_model(),
        tools=[get_transaction, issue_refund],
        system_prompt="You handle refund requests. Check the transaction first and only refund when it is justified.",
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "reject"]}}),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),  # model never sees full card numbers
            ModelCallLimitMiddleware(run_limit=6),  # stop runaway loops
            log_tool_calls,
        ],
        checkpointer=checkpointer, 
        name="refunds_agent",
    )


if __name__ == "__main__":
    chat(build_agent(InMemorySaver()))
