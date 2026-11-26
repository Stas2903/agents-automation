"""The only seam between the system and language models.

Everything else depends on the :class:`ChatModel` and :class:`Embedder` protocols, never on a
provider SDK, so the whole system can run against deterministic fakes ("fake exactly at the
seam").
"""

from agents_automation.llm.seam import ChatModel, Embedder, Message, TransportError, Vector

__all__ = ["ChatModel", "Embedder", "Message", "TransportError", "Vector"]
