# LangChain Agents: Learning Demo

Small examples I built while learning LangChain 1.0 agents (`create_agent`), using a payments theme and sample data. The detailed implementation will follow the requirement specification.

| File | What it shows |
|---|---|
| `agent_payments.py` | Tool calling, and conversation memory with a checkpointer |
| `agent_policy_rag.py` | RAG as a tool: text splitter → Gemini embeddings → vector store → search tool |
| `agent_refunds_hitl.py` | Middleware: human approval before refunds, card-number masking, call limit, custom logger |
| `agent_supervisor.py` | Multi-agent: the three agents above used as tools by one supervisor agent |
| `common.py` | Gemini model, sample transactions, chat loop that handles approvals |

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY
.\.venv\Scripts\python.exe agent_payments.py
.\.venv\Scripts\python.exe agent_policy_rag.py
.\.venv\Scripts\python.exe agent_refunds_hitl.py
.\.venv\Scripts\python.exe agent_supervisor.py
```

Each file opens a chat prompt (`you >`). Type `exit` to quit. The agent remembers the conversation until you exit.

## Demo questions

### Payments agent (`agent_payments.py`)

| Ask | What it shows |
|---|---|
| Which transactions have settlement issues? | Calls `list_settlement_issues`; the comparison is done in Python |
| What's the status of T1009? | Calls `get_transaction`; the ₹50 shortfall comes from the tool, not the LLM |
| How much is it short by? *(right after the previous question)* | Memory: the checkpointer lets it know "it" means T1009 |
| What about T1006? | Memory again; T1006 was captured but never settled |
| Show me transaction T9999 | The tool returns an error and the agent says it doesn't exist instead of crashing |

### Policy agent (`agent_policy_rag.py`)

| Ask | What it shows |
|---|---|
| When should a settlement gap be escalated to finance? | RAG: retrieves the ₹1,000 rule and cites `[settlement_policy]` |
| What should we do if a transaction is missing from the settlement file? | Retrieves the "check again on T+2" rule |
| What evidence do we need to fight a fraud dispute? | Retrieves from `[dispute_sop]` (3-D Secure) |
| Who needs to approve a refund of ₹6,000? | Combines two rules: ops analyst, plus finance above ₹5,000 |
| What is our policy on UPI refunds? | Not in the documents, so it should say so instead of making something up |

### Refunds agent (`agent_refunds_hitl.py`)

| Ask | What it shows |
|---|---|
| Customer on T1007 says the headphones never arrived, my card was 4111 1111 1111 1111. Please refund. | Looks up the transaction first, then pauses with `approve? [y/n]`. Answer `y` to see the refund run. The card number is masked before Gemini sees it. |
| *(same question again)* and answer `n` with a reason | The refund never runs; the agent reports the rejection |
| Refund T1001, the customer changed their mind. | T1001 has no dispute; the agent should question the refund, and the approval gate still applies if it tries |
| Refund T1007 and T1009. | Two refund requests; you approve or reject each one |

### Supervisor (`agent_supervisor.py`)

| Ask | What it shows |
|---|---|
| Which transactions have settlement issues, and what does policy say about escalating them? | Delegates to the payments and policy agents in the same turn, then combines the answers |
| Is T1009 settled correctly? If not, what should we do according to policy? | Uses the payments agent's answer to decide what to ask the policy agent |
| Refund T1007, the customer never received the headphones. | Routes to the refunds agent; its approval prompt pauses the whole supervisor, then resumes |
| What evidence do we need to fight a fraud chargeback? | Routes to the policy agent only |

On the first supervisor question, the answer applies the ₹1,000 escalation rule to each transaction separately. The policy means the day's total gap (₹8,049). This shows why totals and threshold checks should be done in code rather than by the LLM.

## What I learned

- **`create_agent` runs on LangGraph.** That means LangChain agents get LangGraph features such as checkpointers, interrupts and streaming without writing a graph.
- **A checkpointer is passed straight to `create_agent(checkpointer=...)`.**
  - `InMemorySaver` is for development.
  - `SqliteSaver` survives restarts.
  - `PostgresSaver` is for production.
  - Memory works per `thread_id`. Human-in-the-loop requires a checkpointer, because the paused run is saved there.
- **Middleware adds behaviour without changing the agent.**
  - Used in the demo: `HumanInTheLoopMiddleware`, `PIIMiddleware` and `ModelCallLimitMiddleware`.
  - Also available: `SummarizationMiddleware`, `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `ToolRetryMiddleware`, `ToolCallLimitMiddleware`, `LLMToolSelectorMiddleware`, `ContextEditingMiddleware` and `TodoListMiddleware`.
  - Custom middleware uses decorators such as `@before_model` and `@wrap_tool_call`.
- **Multi-agent works by wrapping agents as tools.** When a sub-agent asks for approval, the supervisor pauses too.
- **Where LangGraph comes in.** These agents choose their own steps. A fixed business process with guaranteed order, parallel steps and approval stages needs a LangGraph `StateGraph`. These agents can be added to it as nodes.
