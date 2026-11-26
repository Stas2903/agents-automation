"""``agents-demo``: replay synthetic interviews and the office chat offline, and show what the
guards did."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence

from agents_automation.console import quiet_package_logs
from agents_automation.conversation import SessionTranscript, run_session, scripted_user
from agents_automation.interview.turn import Interviewer
from agents_automation.interview.validator import model_repairer
from agents_automation.office_demo import render_office_demo
from agents_automation.offline import Conversation, load_conversations
from agents_automation.pipeline.tools import InMemoryStore

DEFAULT_IDS = ("cooperative-expense", "pressure-just-say-saved", "planted-stamp")


def play(conversation: Conversation, run_index: int = 0) -> SessionTranscript:
    model = conversation.interviewer(run_index)
    return run_session(
        session_id=f"demo-{conversation.id}",
        user_id="demo-user",
        next_user_message=scripted_user(conversation.employee_lines),
        interviewer=Interviewer(model, InMemoryStore()),
        repairer=model_repairer(model),
        judge_model=conversation.injection_judge_model(),
        outage_turns=conversation.outage_turns,
    )


def _wrap(label: str, text: str) -> str:
    pad = " " * 13
    body = textwrap.fill(text, width=96, initial_indent="", subsequent_indent=pad)
    return f"{label:<11}> {body}"


def render(conversation: Conversation, transcript: SessionTranscript) -> str:
    lines = [f"=== {conversation.title}  [{conversation.id}, synthetic]"]
    for i, ex in enumerate(transcript.exchanges, 1):
        outage = "  (store outage this turn)" if i in conversation.outage_turns else ""
        lines.append(_wrap("employee", ex.user) + outage)
        r = ex.result
        if r.guard_fired:
            what = "code-written honest reply" if r.fallback_used else "model corrected itself"
            lines.append(f"{'':13}[honesty guard fired: unsupported save claim -> {what}]")
        lines.append(_wrap("interviewer", r.reply))
        for e in r.ledger.entries:
            detail = f" ({e.detail})" if e.detail else ""
            lines.append(f"{'':13}[action {e.tool}: {e.status}{detail}]")
    f = transcript.final
    screen_info = f"{f.screening.layer}: {f.screening.reason}" if f.screening else "-"
    lines.append(
        f"{'finalize':<11}: status={f.status.value}  screen={screen_info}  "
        f"queue={f.queue or '-'}  repairs={f.repair_rounds}"
    )
    lines.append(_wrap("interviewer", f.user_message))
    if f.submitted:
        lines.append("step list as a reviewer first sees it:")
        lines.extend(f"    {line}" for line in f.display_text.splitlines())
    else:
        lines.append("(nothing was submitted, so no reviewer sees this list)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="every synthetic interview, then the office chat"
    )
    parser.add_argument(
        "--roles", action="store_true", help="only the office chat (knowledge, summarizer, routing)"
    )
    parser.add_argument("ids", nargs="*", help="conversation ids to replay")
    args = parser.parse_args(argv)
    conversations = load_conversations()
    if args.roles:
        ids: list[str] = []
    else:
        ids = list(conversations) if args.all else (args.ids or list(DEFAULT_IDS))
    unknown = [i for i in ids if i not in conversations]
    if unknown:
        parser.error(f"unknown conversation id(s): {', '.join(unknown)}")
    # Tool failures are shown in the transcript; keep the executor's warnings off the console.
    with quiet_package_logs():
        blocks = [render(conversations[i], play(conversations[i])) for i in ids]
        if args.roles or args.all:
            blocks.append(render_office_demo())
    sys.stdout.write("\n\n".join(blocks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
