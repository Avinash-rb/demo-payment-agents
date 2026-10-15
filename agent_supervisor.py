"""Supervisor: the three agents wrapped as tools of one coordinating agent (multi-agent in LangChain).

Try: "Which transactions have settlement issues, and what does policy say about escalating them?"
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import agent_payments
import agent_policy_rag
import agent_refunds_hitl
from common import chat, get_model

# Sub-agents have no checkpointer of their own; they share the supervisor's.
payments = agent_payments.build_agent()
policy = agent_policy_rag.build_agent()
refunds = agent_refunds_hitl.build_agent()


def _ask(agent, request: str) -> str:
    return agent.invoke({"messages": [{"role": "user", "content": request}]})["messages"][-1].text


@tool
def ask_payments_agent(request: str) -> str:
    """Questions about specific transactions and settlement status."""
    return _ask(payments, request)


@tool
def ask_policy_agent(question: str) -> str:
    """Questions about settlement policy and dispute procedures."""
    return _ask(policy, question)


@tool
def ask_refunds_agent(request: str) -> str:
    """Refund requests. The refunds agent asks a human for approval before refunding."""
    return _ask(refunds, request)


supervisor = create_agent(
    model=get_model(),
    tools=[ask_payments_agent, ask_policy_agent, ask_refunds_agent],
    system_prompt="You are a merchant operations assistant. Delegate to the right agent and combine their answers.",
    checkpointer=InMemorySaver(),
    name="supervisor",
)

if __name__ == "__main__":
    chat(supervisor)
