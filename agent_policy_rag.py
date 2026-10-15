"""Agent 2: RAG as a tool. Splitter -> embeddings -> vector store -> retriever tool.
"""

from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from common import chat, get_embeddings, get_model

# Demo policy text
POLICIES = {
    "settlement_policy": """Captured card transactions settle T+1. The acquirer deducts 2% MDR plus 18% GST on the MDR.
Transactions are matched by RRN and amount; differences up to INR 1 are rounding.
A transaction missing on T+1 should be checked again on T+2 before raising a ticket with the acquirer.
A settlement gap of INR 1,000 or more for one day must be escalated to finance the same day.""",
    "dispute_sop": """For 'goods not received' disputes, courier proof of delivery is required.
If the courier shows the parcel was returned to origin, the merchant should accept the dispute and refund the customer.
For fraud disputes, a successful 3-D Secure authentication is the strongest evidence.
Every refund needs approval from an ops analyst; refunds above INR 5,000 also need finance approval.""",
}


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    docs = [Document(page_content=text, metadata={"source": name}) for name, text in POLICIES.items()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(chunks)
    return store


@tool
def search_policy(query: str) -> str:
    """Search the internal settlement policy and dispute SOP. Use for any rule or procedure question."""
    docs = get_vector_store().similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


def build_agent(checkpointer=None):
    return create_agent(
        model=get_model(),
        tools=[search_policy],
        system_prompt="Answer policy questions only from search_policy results and cite the [source]. If the policy doesn't cover it, say so.",
        checkpointer=checkpointer,
        name="policy_agent",
    )


if __name__ == "__main__":
    chat(build_agent(InMemorySaver()))
