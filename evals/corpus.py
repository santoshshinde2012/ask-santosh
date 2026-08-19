"""A small, fixed corpus with labelled queries — the ground truth for the gate.

Fixed rather than live: pulling the real Medium feed in CI would make the gate
flaky and network-bound, and the point here is to detect *our* regressions, not
Medium's availability. The topics mirror the real corpus so the queries are
representative.
"""
from __future__ import annotations

from ask_santosh.models import Document

DOCUMENTS: list[Document] = [
    Document(
        title="Building a Local AI Video Agent",
        url="https://example.invalid/framesleuth",
        text=(
            "FrameSleuth is a local-first system that turns a screen recording into a "
            "structured Context Bundle for a coding agent. It extracts keyframes from the "
            "video, transcribes the narration, and captures console and network activity, "
            "then emits repro steps and error evidence over MCP. Frames never leave the "
            "machine, which is what makes it usable on a customer bug report."
        ),
    ),
    Document(
        title="The Medallion Architecture Reconsidered",
        url="https://example.invalid/medallion",
        text=(
            "The medallion architecture organises a lakehouse into bronze, silver and gold "
            "layers. Bronze holds raw ingested records, silver applies cleaning and "
            "conformance, gold serves aggregated business tables. Where it cracks is "
            "ownership: gold tables accumulate business logic nobody maintains, and the "
            "layer boundary stops describing a real contract between teams."
        ),
    ),
    Document(
        title="Designing a Production MCP Server",
        url="https://example.invalid/mcp",
        text=(
            "A production Model Context Protocol server is mostly about tool design, not "
            "transport. Keep each tool narrow, return structured errors the model can act "
            "on, and budget the token cost of every response. Pagination, idempotency and "
            "rate limiting matter more than clever prompting once real API complexity "
            "arrives."
        ),
    ),
    Document(
        title="Evaluating RAG Systems in Production",
        url="https://example.invalid/evals",
        text=(
            "Evaluating retrieval augmented generation needs reference-free metrics because "
            "you rarely have gold answers. Faithfulness catches hallucination against the "
            "retrieved context, answer relevancy catches drift from the question, and "
            "contextual relevancy catches a retriever returning passages that do not help. "
            "Gate them in continuous integration or they rot."
        ),
    ),
    Document(
        title="Why Your Churn Model Is Not a Risk Model",
        url="https://example.invalid/churn",
        text=(
            "A churn model predicts who leaves. A risk model prices what that departure "
            "costs. Optimising a single probability threshold conflates the two and quietly "
            "destroys expected value, because the customers most likely to churn are not "
            "the customers whose churn hurts most."
        ),
    ),
    Document(
        title="Databricks Genie: Agent Mode vs Chat Mode",
        url="https://example.invalid/genie",
        text=(
            "Genie agent mode plans multi step work and calls tools, where chat mode answers "
            "a single natural language question against a semantic model. Agent mode costs "
            "more tokens per request and is harder to attribute in a budget, so the switch "
            "is a governance decision as much as a capability one."
        ),
    ),
    # --- deliberate near-misses -------------------------------------------
    # These share vocabulary with the documents above. Without them every query
    # has exactly one lexical match and the gate passes no matter what breaks.
    Document(
        title="The AI Consumption Plane on Databricks",
        url="https://example.invalid/consumption-plane",
        text=(
            "The consumption plane is where governed data meets agents and applications. "
            "It sits above the lakehouse and unity catalog, exposing tools, semantic models "
            "and vector indexes with the same governance as tables. Token cost and audit "
            "attribution become first class concerns once agents read from it."
        ),
    ),
    Document(
        title="Orchestrating Production Multi-Agent Systems",
        url="https://example.invalid/multi-agent",
        text=(
            "A supervisor routes an incoming request to specialist agents and a synthesizer "
            "merges their findings with citations. Each specialist is interchangeable behind "
            "one protocol, so adding a capability never edits the supervisor. Most production "
            "failures are routing failures, not model failures."
        ),
    ),
    Document(
        title="Feature Store vs Gold Data Products",
        url="https://example.invalid/feature-store",
        text=(
            "Gold tables serve analytics; a feature store serves models with point in time "
            "correctness and online lookup. Putting machine learning features in gold tables "
            "invites training serving skew, because the batch table has no notion of what was "
            "known at prediction time."
        ),
    ),
    Document(
        title="Every Design Pattern Casts a Shadow in ML",
        url="https://example.invalid/patterns",
        text=(
            "A pattern that tidies code can hide a modelling assumption. A factory that builds "
            "estimators hides which hyperparameters were searched, a decorator that caches "
            "predictions hides staleness, and an abstract base class hides that two models "
            "disagree about what a probability means."
        ),
    ),
    Document(
        title="Data Governance and the Challenges Agents Bring",
        url="https://example.invalid/governance",
        text=(
            "Governance assumed a human read the data. An agent reads everything it is "
            "permitted to and composes it in ways no reviewer anticipated, so row level "
            "controls and audit trails have to describe intent, not just access. Lineage "
            "must survive a tool call."
        ),
    ),
    Document(
        title="Turning Recordings into Tests",
        url="https://example.invalid/recordings-to-tests",
        text=(
            "Once a session is a structured bundle it can become a regression test. The "
            "assertions come from what the narration claimed should happen, and the fixtures "
            "come from the captured network traffic, so the replay does not need the original "
            "environment."
        ),
    ),
]

# question -> the url that should be retrieved
LABELLED: dict[str, str] = {
    "What is FrameSleuth and what problem does it solve?": "https://example.invalid/framesleuth",
    "How do I keep frames from leaving my machine when reporting a bug?": "https://example.invalid/framesleuth",
    "What are the bronze silver and gold layers?": "https://example.invalid/medallion",
    "Where does the medallion architecture fall short?": "https://example.invalid/medallion",
    "How should I design tools for an MCP server?": "https://example.invalid/mcp",
    "Which metrics catch hallucination in a RAG pipeline?": "https://example.invalid/evals",
    "Why are reference-free metrics used to evaluate retrieval?": "https://example.invalid/evals",
    "Why is predicting churn different from pricing risk?": "https://example.invalid/churn",
    "Should I switch Genie from chat mode to agent mode?": "https://example.invalid/genie",

    # Paraphrases with little lexical overlap - the cases that actually regress.
    "How do I stop my supervisor from editing when I add a capability?": "https://example.invalid/multi-agent",
    "What goes wrong if I serve model inputs from an aggregated table?": "https://example.invalid/feature-store",
    "Why does an autonomous reader break assumptions built for people?": "https://example.invalid/governance",
    "Can a captured session become a regression test?": "https://example.invalid/recordings-to-tests",
    "Where do tools and vector indexes get governed alongside tables?": "https://example.invalid/consumption-plane",
    "How can tidy abstractions conceal a modelling assumption?": "https://example.invalid/patterns",
    "What hides which hyperparameters were actually searched?": "https://example.invalid/patterns",
    "Which layer holds raw ingested records before cleaning?": "https://example.invalid/medallion",
    "How do I budget the token cost of a tool response?": "https://example.invalid/mcp",
}
