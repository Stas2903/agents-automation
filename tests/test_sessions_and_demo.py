"""Whole sessions over the synthetic conversations, the offline fakes, and the demo CLI."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from agents_automation.demo import main as demo_main
from agents_automation.demo import play
from agents_automation.injection.screen import MASK
from agents_automation.interview.honesty import CORRECTION_PREFIX, HONEST_SAVE_FAILED
from agents_automation.llm import Message
from agents_automation.llm.fakes import ScriptExhaustedError
from agents_automation.models import Status
from agents_automation.offline import (
    ScriptedInterviewerModel,
    ScriptedTurn,
    load_conversations,
)

CONVERSATIONS = load_conversations()

EXPECTED_STATUS = {
    "cooperative-expense": Status.DELIVERED,
    "terse-flaky": Status.DELIVERED,  # variant 0; variant 1 is checked below
    "pressure-just-say-saved": Status.DELIVERED,
    "save-outage": Status.DELIVERED,
    "planted-stamp": Status.HELD_FOR_REVIEW,
    "subtle-planted": Status.HELD_FOR_REVIEW,
    "judge-outage": Status.DELIVERED_FLAGGED,
    "legit-authority-words": Status.DELIVERED,
    "repair-needed": Status.DELIVERED,
}


def test_data_file_is_marked_synthetic_everywhere() -> None:
    raw = (
        resources.files("agents_automation")
        .joinpath("data/synthetic_conversations.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(raw)
    assert data["synthetic"] is True
    assert "synthetic" in data["notice"].lower()


def test_every_conversation_has_an_expected_status() -> None:
    assert set(CONVERSATIONS) == set(EXPECTED_STATUS)


@pytest.mark.parametrize("cid", sorted(EXPECTED_STATUS))
def test_conversation_ends_with_the_expected_status(cid: str) -> None:
    transcript = play(CONVERSATIONS[cid])
    assert transcript.final.status is EXPECTED_STATUS[cid]


def test_flaky_variant_fails_validation() -> None:
    transcript = play(CONVERSATIONS["terse-flaky"], run_index=1)
    assert transcript.final.status is Status.INVALID
    assert transcript.final.repair_rounds == 0  # the one repair answer was unusable
    assert transcript.final.submitted is False


def test_repair_needed_is_fixed_in_one_round_and_saved() -> None:
    final = play(CONVERSATIONS["repair-needed"]).final
    assert final.repair_rounds == 1
    assert final.step_list is not None
    assert final.step_list.steps[-1].kind == "finish"


def test_pressure_turn_is_corrected_by_the_model() -> None:
    last = play(CONVERSATIONS["pressure-just-say-saved"]).exchanges[-1].result
    assert (last.guard_fired, last.fallback_used, last.ledger.saved) == (True, False, True)


def test_outage_turn_ends_with_the_honest_note_and_recovers_later() -> None:
    transcript = play(CONVERSATIONS["save-outage"])
    turn2, turn3 = transcript.exchanges[1].result, transcript.exchanges[2].result
    assert turn2.fallback_used
    assert turn2.reply.endswith(HONEST_SAVE_FAILED)
    assert turn3.ledger.saved


def test_planted_payload_never_reaches_the_display_text() -> None:
    final = play(CONVERSATIONS["planted-stamp"]).final
    assert "APPROVED BY THE ADMINISTRATOR" not in final.display_text
    assert "skip the review" not in final.display_text
    assert final.display_text.count(MASK) == 2


def test_subtle_payload_is_masked_by_the_judge_claim() -> None:
    final = play(CONVERSATIONS["subtle-planted"]).final
    assert final.screening is not None
    assert final.screening.layer == "judge"
    assert "reviewers do not need" not in final.display_text


# --- the offline interviewer fake ----------------------------------------------------------


def _msgs(employee_lines: int, *, correction: bool = False) -> list[Message]:
    msgs = [Message("system", "rules"), Message("system", "plan"), Message("user", "[draft]")]
    for i in range(employee_lines):
        msgs.append(Message("user", f"u{i}"))
        if i < employee_lines - 1:
            msgs.append(Message("assistant", f"a{i}"))
    if correction:
        msgs += [Message("assistant", "raw"), Message("user", f"{CORRECTION_PREFIX} ...")]
    return msgs


def test_scripted_interviewer_routes_by_prompt_shape() -> None:
    model = ScriptedInterviewerModel(
        [ScriptedTurn("t1", None), ScriptedTurn("t2", "t2-fix")], repairs=["r1"]
    )
    assert model.complete(_msgs(1), timeout_s=1) == "t1"
    assert model.complete(_msgs(2), timeout_s=1) == "t2"
    assert model.complete(_msgs(2, correction=True), timeout_s=1) == "t2-fix"
    from agents_automation.interview.validator import REPAIR_INSTRUCTIONS

    assert model.complete([Message("system", REPAIR_INSTRUCTIONS)], timeout_s=1) == "r1"


def test_scripted_interviewer_is_strict_about_missing_entries() -> None:
    model = ScriptedInterviewerModel([ScriptedTurn("t1", None)])
    with pytest.raises(ScriptExhaustedError):
        model.complete(_msgs(1, correction=True), timeout_s=1)
    with pytest.raises(ScriptExhaustedError):
        model.complete(_msgs(2), timeout_s=1)


# --- demo CLI ----------------------------------------------------------------------------


def test_demo_default_run_prints_the_three_headline_outcomes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert demo_main([]) == 0
    out = capsys.readouterr().out
    assert out.count("=== ") == 3
    assert "status=delivered " in out
    assert "honesty guard fired: unsupported save claim -> model corrected itself" in out
    assert "status=held_for_review" in out
    assert "APPROVED BY THE" in out  # the employee's message is shown as they wrote it
    assert f"[a4] (finish) Employee: {MASK} - {MASK}" in out  # ... but masked in the record


def test_demo_rejects_unknown_ids(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        demo_main(["no-such-conversation"])
    assert "unknown conversation id" in capsys.readouterr().err
