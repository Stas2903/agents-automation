"""The office-chat demo: one surface, three roles, each guard visible in the output. Also the
interactive chat (``agents-demo --chat``) on the same offline setup."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable, Iterable, Sequence

from agents_automation.assistant import AssistantReply, OfficeAssistant
from agents_automation.intent import classifier_answer
from agents_automation.interview.turn import Interviewer
from agents_automation.knowledge.ingest import load_documents
from agents_automation.llm import Message
from agents_automation.llm.boundary import PlacedModel
from agents_automation.llm.fakes import RuleModel, ScriptedModel
from agents_automation.office import (
    EXTERNAL,
    SELF_HOSTED,
    TODAY,
    OfficeData,
    build_office_data,
    knowledge_role,
    summarizer_role,
)
from agents_automation.offline_roles import extractive_answerer
from agents_automation.pipeline.planner import Request
from agents_automation.pipeline.tools import InMemoryStore

# (user, message) pairs; users are role ids from the synthetic directory
MESSAGES = (
    ("u-staff", "When do I need a receipt?"),
    ("u-staff", "/ask What are the salary bands?"),
    ("u-hr", "/ask When do salary reviews happen?"),
    ("u-staff", "Can you summarize #general from 2026-01-05 to 2026-03-27?"),
    ("u-staff", "Summarize #leads for March"),
    ("u-lead", "Catch me up on #leads for the last two weeks"),
    ("u-staff", "What's new this week?"),
    ("u-staff", "Anything new about approvals?"),
    ("u-staff", "I want to describe how our expense process works"),
)


def _wrap(prefix: str, text: str) -> list[str]:
    pad = " " * 13
    out = []
    for i, line in enumerate(text.splitlines() or [""]):
        first = f"{prefix:<11}> " if i == 0 else pad
        out.append(textwrap.fill(line, width=100, initial_indent=first, subsequent_indent=pad))
    return out


def _outcome(got: AssistantReply) -> list[str]:
    route = f"{got.role} (by {got.how})" if got.role else f"clarify (by {got.how})"
    lines = [f"{'':13}[routed to {route}; {got.detail}]"]
    lines.extend(f"{'':13}[{skip}]" for skip in got.skips)
    lines.extend(_wrap("assistant", got.reply))
    return lines


def _render(user: str, text: str, got: AssistantReply) -> list[str]:
    return _wrap(user, text) + _outcome(got)


def render_office_demo() -> str:
    data = build_office_data()
    classifier = ScriptedModel(
        "offline-classifier", [classifier_answer("summarizer"), classifier_answer("reviewer")]
    )
    interviewer_model = ScriptedModel(
        "offline-interviewer",
        [
            json.dumps(
                {
                    "reply": "Sure. I'll email you a summary when we are done. "
                    "What is the first step?",
                    "actions": [],
                }
            ),
            json.dumps({"reply": "Sure. What is the first step, and who does it?", "actions": []}),
        ],
    )
    assistant = OfficeAssistant(
        knowledge=knowledge_role(data),
        summarizer=summarizer_role(data),
        interviewer=Interviewer(interviewer_model, InMemoryStore()),
        classifier=PlacedModel(classifier, SELF_HOSTED),
    )
    report = data.ingest_report
    lines = [
        "=== The office chat: one surface, three roles  [office-chat, synthetic]",
        f"ingest   : {report.embedded} chunks embedded, quarantined {', '.join(report.quarantined)}"
        f" (planted instructions); held out of the chat: {', '.join(data.chat.held_out)}",
    ]
    for i, (user, text) in enumerate(MESSAGES):
        lines.extend(_render(user, text, assistant.handle(user, text, session_id=f"demo-{i}")))

    # The same HR question when only an external model is available: nothing is sent.
    external_only = knowledge_role(data, [PlacedModel(extractive_answerer("external"), EXTERNAL)])
    answer = external_only.ask(
        Request("office_chat", "ask", "demo-x", "u-hr"), "When do salary reviews happen?"
    )
    lines.append("--- only an external model is available (cleared for INTERNAL data):")
    lines.extend(_wrap("u-hr", "/ask When do salary reviews happen?"))
    note = "the confidential passages were not sent" if answer.status == "boundary" else "-"
    lines.append(f"{'':13}[{answer.status}: {note}]")
    lines.extend(_wrap("assistant", answer.reply))
    return "\n".join(lines)


CHAT_QUIT = frozenset({"/quit", "/exit"})
CHAT_QUESTIONS = (
    "Sure. What is the first step, and who does it?",
    "Noted. What happens next, and who does it?",
    "Does anything else happen before that step is done?",
    "Is anyone else involved, or is that the end of it?",
)


def _chat_question(messages: Sequence[Message]) -> str:
    asked = sum(m.role == "assistant" for m in messages)  # consecutive replies never repeat
    question = CHAT_QUESTIONS[asked % len(CHAT_QUESTIONS)]
    return json.dumps({"reply": question, "actions": []})


def chat_banner(data: OfficeData, user: str) -> list[str]:
    groups = data.directory.groups(user)
    docs = [d.title for d in load_documents() if not groups.isdisjoint(d.groups)]
    channels = [
        f"#{c.name}" for c in data.chat.channels.values() if not groups.isdisjoint(c.member_groups)
    ]
    return [
        f"=== The office chat as {user}  [offline, synthetic data, today is {TODAY.isoformat()}]",
        *_wrap("documents", ", ".join(docs) or "none (this user is in no group)"),
        *_wrap("channels", ", ".join(channels) or "none"),
        *_wrap(
            "how",
            "Ask a question, ask for a summary of a channel, or describe a workflow. "
            "/ask, /summarize and /describe pick a role directly; /quit leaves. Offline there "
            "is no intent classifier, so a message no rule recognises gets a question back.",
        ),
    ]


def run_chat(
    user: str, messages: Iterable[str], write: Callable[[str], None], *, echo: bool = False
) -> None:
    """The office chat for typed messages, on the same offline stand-ins as the demo. ``echo``
    repeats each message (for piped input, where nobody saw it typed)."""
    data = build_office_data()
    assistant = OfficeAssistant(
        knowledge=knowledge_role(data),
        summarizer=summarizer_role(data),
        interviewer=Interviewer(RuleModel("offline-interviewer", _chat_question), InMemoryStore()),
        classifier=None,
    )
    for line in chat_banner(data, user):
        write(line)
    for raw in messages:
        text = raw.strip()
        if text.lower() in CHAT_QUIT:
            return
        if text:
            got = assistant.handle(user, text, session_id="chat")
            for line in _render(user, text, got) if echo else _outcome(got):
                write(line)
