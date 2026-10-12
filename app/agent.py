"""Kestrel AI assistant: a LangGraph agent that answers questions and acts on the churn database.

Flow (the graph):   START -> agent (LLM decides) -> tools (run database functions) -> agent -> ... -> END
The LLM never touches the database itself. It can only call the small, safe tools defined in make_tools().
"""
import os

from fastapi import HTTPException

from .config import settings

SYSTEM_PROMPT = """You are Kestrel Assistant, an AI helper for bank staff who manage customer churn.
Rules:
- Use the tools for every fact, number or customer. Never guess or invent customers, scores or reasons.
- Risk is shown as a percentage. "High" tier means at or above the threshold from get_overview; Medium is above half of it.
- Explain reasons in plain language (for example "few transactions recently pushes risk up").
- Only call update_customer_status when the user clearly asks to change a status. Say what you changed afterwards.
- If a tool returns an error, tell the user simply. If a question is not about customers or churn, say you can only help with Kestrel.
- Keep answers short: a few sentences or a small list."""


class AgentUnavailable(Exception):
    """Raised when no LLM key is configured."""


DEFAULT_MODELS = {"ANTHROPIC_API_KEY": "anthropic:claude-sonnet-5-5", "OPENAI_API_KEY": "openai:gpt-4o-mini",
                  "GOOGLE_API_KEY": "google_genai:gemini-2.5-flash-lite"}


def get_llm():
    model = settings.llm_model or next((m for k, m in DEFAULT_MODELS.items() if os.getenv(k)), "")
    if not model:
        raise AgentUnavailable("The AI assistant is not set up yet. Add GOOGLE_API_KEY (free), ANTHROPIC_API_KEY or "
                               "OPENAI_API_KEY to your .env file or host settings and restart the app.")
    from langchain.chat_models import init_chat_model
    return init_chat_model(model, temperature=0)


def make_tools(db, user):
    """Tools the agent may call. They reuse the existing API functions, so the rules stay the same everywhere."""
    from langchain_core.tools import tool
    from . import main as api  # imported here (not at the top) to avoid a circular import

    def safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException as e:
            return {"error": e.detail}

    @tool
    def get_overview() -> dict:
        """Dashboard numbers: total customers, how many are High, Medium and Low churn risk, average risk,
        the High-risk threshold and the expected number of churners among High-risk customers."""
        o = api.overview(db=db, user=user)
        return {k: o[k] for k in ("total", "high", "medium", "low", "avg_risk", "threshold", "expected_churners_high")}

    @tool
    def list_customers(tier: str = "", status: str = "", search: str = "", limit: int = 10) -> dict:
        """List customers, highest churn risk first.
        tier: High, Medium or Low (empty for all). status: Not contacted, Contacted, Saved or Lost (empty for all).
        search: part of a customer id. limit: how many to return (max 25)."""
        r = api.customers(q=search, tier=tier, status=status, limit=max(1, min(int(limit), 25)), offset=0,
                          db=db, user=user)
        return {"total_matching": r["total"],
                "customers": [{"id": c["id"], "segment": c["segment"], "risk_pct": round(c["probability"] * 100),
                               "tier": c["tier"], "main_driver": c["top_factor"], "status": c["status"]}
                              for c in r["items"]]}

    @tool
    def get_customer(customer_id: str) -> dict:
        """Details for one customer: churn risk, the top reasons behind the score, a recommended action
        and the latest status changes."""
        d = safe(api.customer, customer_id, db=db, user=user)
        if "error" in d:
            return d
        return {"id": d["id"], "segment": d["segment"], "status": d["status"],
                "risk_pct": round(d["probability"] * 100), "tier": d["tier"],
                "top_reasons": [{"signal": f["label"], "value": f["value"],
                                 "effect": "raises risk" if f["contribution"] > 0 else "lowers risk"}
                                for f in d["features"][:4]],
                "recommended_action": d["action"], "recent_history": d["history"][:3]}

    @tool
    def segment_summary() -> dict:
        """Churn risk by customer segment (card type): number of customers, average risk, High-risk count, top driver."""
        return api.segments(db=db, user=user)

    @tool
    def update_customer_status(customer_id: str, status: str, note: str = "") -> dict:
        """Set a customer's status to Not contacted, Contacted, Saved or Lost, and log it with an optional note.
        Only call this when the user clearly asked for the change."""
        if user.role == "viewer":
            return {"error": "This account has view-only access and cannot change statuses."}
        return safe(api.set_action, customer_id, api.ActionIn(status=status, note=note[:500]), db=db, user=user)

    return [get_overview, list_customers, get_customer, segment_summary, update_customer_status]


def build_graph(llm, tools):
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.messages import SystemMessage

    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def run_agent(db, user, history, message, llm=None) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    graph = build_graph(llm or get_llm(), make_tools(db, user))
    msgs = []
    for h in history[-10:]:
        text = str(h.get("content", ""))[:2000]
        if text:
            msgs.append(HumanMessage(text) if h.get("role") == "user" else AIMessage(text))
    while msgs and isinstance(msgs[0], AIMessage):  # a conversation must start with the user
        msgs.pop(0)
    msgs.append(HumanMessage(message))

    out = graph.invoke({"messages": msgs}, config={"recursion_limit": 12})["messages"]
    new = out[len(msgs):]
    steps = [{"tool": c["name"], "args": c["args"]}
             for m in new if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    reply = next((_text(m.content) for m in reversed(new) if isinstance(m, AIMessage) and not m.tool_calls), "")
    return {"reply": reply or "Sorry, I could not finish that. Try asking in a simpler way.", "steps": steps}
