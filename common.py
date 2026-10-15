"""Shared setup: Gemini model, sample data, and a small chat loop that handles approvals."""

import os

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.types import Command

load_dotenv(override=True)


def get_model():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "gemini-embedding-001"),
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


# Sample payments data (fictional)
TRANSACTIONS = {
    "T1001": {"merchant": "Kiran Electronics", "amount": 2499.00, "status": "SETTLED", "settled_amount": 2499.00, "settled_on": "2026-10-12"},
    "T1006": {"merchant": "Kiran Electronics", "amount": 7999.00, "status": "CAPTURED", "settled_amount": None, "settled_on": None},
    "T1007": {"merchant": "Kiran Electronics", "amount": 2199.00, "status": "SETTLED", "settled_amount": 2199.00, "settled_on": "2026-10-12",
              "dispute": "Customer says headphones never arrived; courier shows parcel returned to origin"},
    "T1009": {"merchant": "Kiran Electronics", "amount": 1499.00, "status": "SETTLED", "settled_amount": 1449.00, "settled_on": "2026-10-12"},
}


def _ask_approval(action: dict) -> dict:
    print(f"\n⏸  Approval needed: {action['name']}({action['args']})")
    if input("   approve? [y/n] > ").strip().lower().startswith("y"):
        return {"type": "approve"}
    return {"type": "reject", "message": input("   reason > ").strip() or "Rejected by reviewer."}


def _print_tool_calls(messages: list) -> None:
    """Show which tools the agent called during the last turn."""
    turn = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        turn.append(message)
    for message in reversed(turn):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"   [tool] {call['name']}({call['args']})")


def chat(agent, thread_id: str = "demo") -> None:
    """Multi-turn chat. The checkpointer remembers the conversation for this thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    print("Type 'exit' to quit.")
    while True:
        text = input("\nyou > ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        # Human-in-the-loop: the agent paused before a sensitive tool. Ask, then resume.
        while "__interrupt__" in result:
            request = result["__interrupt__"][0].value
            decisions = [_ask_approval(action) for action in request["action_requests"]]
            result = agent.invoke(Command(resume={"decisions": decisions}), config)
        _print_tool_calls(result["messages"])
        print(f"agent > {result['messages'][-1].text}")
