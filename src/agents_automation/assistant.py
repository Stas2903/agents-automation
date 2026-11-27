"""The office assistant: one chat surface, three roles.

Every message on the ``office_chat`` surface goes through the intent router, which may only
choose among the roles that surface offers. The chosen role then runs with its own sealed
plan; the trusted envelope (user id, session) never comes from the message text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents_automation.budget import Skip, TurnBudget
from agents_automation.intent import CLARIFY_REPLY, How, pick_role
from agents_automation.interview.turn import Interviewer
from agents_automation.knowledge.role import KnowledgeRole
from agents_automation.llm import Message
from agents_automation.llm.boundary import PlacedModel
from agents_automation.pipeline.planner import Request
from agents_automation.summarizer.role import SummarizerRole

SURFACE = "office_chat"


@dataclass(frozen=True)
class AssistantReply:
    role: str | None
    how: How
    reply: str
    detail: str  # a one-line note for logs and the demo (status, sources, skips)
    skips: tuple[Skip, ...] = field(default=())


class OfficeAssistant:
    def __init__(
        self,
        *,
        knowledge: KnowledgeRole,
        summarizer: SummarizerRole,
        interviewer: Interviewer,
        classifier: PlacedModel | None,
    ) -> None:
        self.knowledge = knowledge
        self.summarizer = summarizer
        self.interviewer = interviewer
        self.classifier = classifier
        # interview history per (session, user): another user reusing a session id never
        # sees someone else's turns (and the store refuses them as an author anyway)
        self._history: dict[tuple[str, str], list[Message]] = {}

    def handle(
        self, user_id: str, text: str, *, session_id: str, budget: TurnBudget | None = None
    ) -> AssistantReply:
        open_interview = bool(self._history.get((session_id, user_id)))
        routed = pick_role(
            SURFACE, text, self.classifier, continuing="interviewer" if open_interview else None
        )
        if routed.role is None:
            return AssistantReply(None, routed.how, CLARIFY_REPLY, routed.reason)
        if routed.role == "knowledge":
            answer = self.knowledge.ask(
                Request(SURFACE, "ask", session_id, user_id), routed.text, budget
            )
            cited = ", ".join(answer.citations) or "-"
            return AssistantReply(
                "knowledge",
                routed.how,
                answer.reply,
                f"{answer.status}; cites {cited}",
                answer.skips,
            )
        if routed.role == "summarizer":
            summary = self.summarizer.summarize(
                Request(SURFACE, "summarize", session_id, user_id), routed.text, budget
            )
            return AssistantReply(
                "summarizer", routed.how, summary.reply, summary.status, summary.skips
            )
        store = self.interviewer.store
        if session_id not in store.sessions:  # never re-create (and re-own) an existing one
            store.create_session(session_id, frozenset({user_id}))
        history = self._history.setdefault((session_id, user_id), [])
        result = self.interviewer.run_turn(
            Request(SURFACE, "interview_turn", session_id, user_id), list(history), routed.text
        )
        history += [Message("user", routed.text), Message("assistant", result.reply)]
        return AssistantReply(
            "interviewer",
            routed.how,
            result.reply,
            f"corrective: {result.corrective or '-'}",
            result.skips,
        )
