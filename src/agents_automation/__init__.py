"""Guard rails for a multi-role LLM office assistant: grounded document Q&A, team-chat
summaries and workflow interviews, on one guarded core.

Everything in this package runs offline against deterministic fakes. Optional live adapters
live in ``agents_automation.llm``: the chat adapters are used by ``agents-eval --live``, and
``OpenAICompatibleEmbedder`` is there for wiring real embeddings into the knowledge role.
"""

__version__ = "0.1.0"
