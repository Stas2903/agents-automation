"""Data-boundary routing, the turn budget, the hashing embedder and the intent router."""

from __future__ import annotations

import json
import math

import pytest

from agents_automation.budget import FakeClock, TurnBudget, unlimited
from agents_automation.intent import CLARIFY_REPLY, classifier_answer, pick_role
from agents_automation.llm import Message, TransportError
from agents_automation.llm.boundary import (
    BoundaryError,
    ModelRouter,
    PlacedModel,
    Placement,
    Sensitivity,
    highest,
    require_clearance,
)
from agents_automation.llm.fakes import HashingEmbedder, ScriptedModel
from agents_automation.llm.text import content_tokens
from agents_automation.pipeline.router import RoutingError

SELF_HOSTED = Placement("self-hosted", Sensitivity.CONFIDENTIAL)
EXTERNAL = Placement("external", Sensitivity.INTERNAL)
PUBLIC_ONLY = Placement("public-only", Sensitivity.PUBLIC)

# --- data boundary --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("public", Sensitivity.PUBLIC),
        ("INTERNAL", Sensitivity.INTERNAL),
        (" confidential ", Sensitivity.CONFIDENTIAL),
        ("", Sensitivity.CONFIDENTIAL),  # missing metadata never widens access
        ("secret", Sensitivity.CONFIDENTIAL),
        ("publik", Sensitivity.CONFIDENTIAL),
    ],
)
def test_labels_parse_and_unknown_labels_count_as_confidential(
    label: str, expected: Sensitivity
) -> None:
    assert Sensitivity.parse(label) is expected


def test_highest_label_wins_and_no_labels_is_public() -> None:
    assert highest([Sensitivity.PUBLIC, Sensitivity.CONFIDENTIAL, Sensitivity.INTERNAL]) is (
        Sensitivity.CONFIDENTIAL
    )
    assert highest([]) is Sensitivity.PUBLIC


def _router(*placements: Placement) -> tuple[ModelRouter, list[ScriptedModel]]:
    models = [ScriptedModel(p.name, [f"answer from {p.name}"] * 3) for p in placements]
    return ModelRouter([PlacedModel(m, p) for m, p in zip(models, placements, strict=True)]), models


def test_the_first_cleared_model_in_preference_order_is_used() -> None:
    router, (external, self_hosted) = _router(EXTERNAL, SELF_HOSTED)
    msgs = [Message("user", "q")]
    assert router.complete(msgs, sensitivity=Sensitivity.INTERNAL, timeout_s=1) == (
        "answer from external"
    )
    assert router.complete(msgs, sensitivity=Sensitivity.CONFIDENTIAL, timeout_s=1) == (
        "answer from self-hosted"
    )
    assert len(external.calls) == 1
    assert len(self_hosted.calls) == 1


@pytest.mark.parametrize(
    ("placements", "sensitivity"),
    [
        ((EXTERNAL,), Sensitivity.CONFIDENTIAL),
        ((PUBLIC_ONLY,), Sensitivity.INTERNAL),
        ((PUBLIC_ONLY, EXTERNAL), Sensitivity.CONFIDENTIAL),
        ((), Sensitivity.PUBLIC),
    ],
)
def test_no_cleared_model_fails_closed_and_nothing_is_sent(
    placements: tuple[Placement, ...], sensitivity: Sensitivity
) -> None:
    router, models = _router(*placements)
    with pytest.raises(BoundaryError, match="not sent"):
        router.complete([Message("user", "q")], sensitivity=sensitivity, timeout_s=1)
    assert all(m.calls == [] for m in models)


def test_clearance_is_inclusive_at_the_boundary() -> None:
    require_clearance(EXTERNAL, Sensitivity.INTERNAL)  # equal: allowed
    with pytest.raises(BoundaryError):
        require_clearance(EXTERNAL, Sensitivity.CONFIDENTIAL)


# --- turn budget ----------------------------------------------------------------------


def test_an_optional_call_runs_only_while_it_still_fits() -> None:
    clock = FakeClock()
    budget = TurnBudget(10, clock)
    assert budget.allow("rerank", needs_s=4)
    clock.advance(7)
    assert budget.remaining() == 3
    assert not budget.allow("corrective", needs_s=4)
    assert budget.allow("polish", needs_s=3)  # exactly fits
    assert [str(s) for s in budget.skips] == ["corrective skipped: needs 4s, 3s left in this turn"]


def test_the_budget_never_goes_negative_and_skips_are_kept_in_order() -> None:
    clock = FakeClock()
    budget = TurnBudget(2, clock)
    clock.advance(5)
    assert budget.remaining() == 0
    budget.record_skip("rerank", "reranker unavailable")
    assert not budget.allow("judge", needs_s=0.5)
    assert [s.name for s in budget.skips] == ["rerank", "judge"]


def test_unlimited_budget_allows_everything_and_records_nothing() -> None:
    budget = unlimited()
    assert budget.allow("anything", needs_s=1e9)
    assert budget.skips == ()


@pytest.mark.parametrize("bad", [-1.0])
def test_negative_budget_and_backwards_clock_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="cannot be negative or NaN"):
        TurnBudget(bad, FakeClock())
    with pytest.raises(ValueError, match="backwards"):
        FakeClock().advance(bad)


# --- hashing embedder -----------------------------------------------------------------


def _cos(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_embeddings_are_deterministic_unit_vectors() -> None:
    emb = HashingEmbedder()
    [a, b] = emb.embed(
        ["Receipts are needed above 25.", "Receipts are needed above 25."], timeout_s=1
    )
    assert a == b
    assert math.isclose(sum(v * v for v in a), 1.0)
    assert emb.calls == [("Receipts are needed above 25.", "Receipts are needed above 25.")]


def test_shared_words_score_higher_than_unrelated_text() -> None:
    emb = HashingEmbedder()
    q, near, far = emb.embed(
        [
            "When do I need a receipt?",
            "A receipt is needed for every expense.",
            "The wifi password",
        ],
        timeout_s=1,
    )
    assert _cos(q, near) > 0.3
    assert _cos(q, near) > _cos(q, far)


def test_text_without_content_words_embeds_to_zero() -> None:
    [vec] = HashingEmbedder().embed(["the and of"], timeout_s=1)
    assert set(vec) == {0.0}


def test_content_tokens_drop_stopwords_and_fold_plurals() -> None:
    assert content_tokens("Where are the Receipts and the class?") == ["receipt", "class"]


# --- intent router --------------------------------------------------------------------

ROUTES_BY_RULE = [
    ("Can you summarize the finance channel for March?", "summarizer"),
    ("Catch me up on last week", "summarizer"),
    ("What did the team decide about the laptop budget?", "summarizer"),
    ("How do I get a new laptop?", "knowledge"),
    ("When do I need a receipt?", "knowledge"),
    ("Can I fly business class?", "knowledge"),
    ("Am I allowed to book business class?", "knowledge"),
    ("What is the policy on home office equipment", "knowledge"),
    ("I want to describe how our expense process works", "interviewer"),
    ("Let me walk you through the onboarding workflow", "interviewer"),
]


@pytest.mark.parametrize(("text", "role"), ROUTES_BY_RULE)
def test_clear_cases_are_routed_by_rules_without_a_model(text: str, role: str) -> None:
    classifier = ScriptedModel("classifier", [])  # any call would raise
    routed = pick_role("office_chat", text, PlacedModel(classifier, SELF_HOSTED))
    assert (routed.role, routed.how) == (role, "rule")
    assert classifier.calls == []


@pytest.mark.parametrize(
    ("text", "role", "rest"),
    [
        ("/ask what is the travel limit", "knowledge", "what is the travel limit"),
        ("/SUMMARIZE #general", "summarizer", "#general"),
        ("/describe", "interviewer", ""),
    ],
)
def test_commands_pick_their_role_and_are_stripped(text: str, role: str, rest: str) -> None:
    routed = pick_role("office_chat", text, None)
    assert (routed.role, routed.how, routed.text) == (role, "command", rest)


def test_a_single_role_surface_ignores_text_and_commands() -> None:
    routed = pick_role("interview_chat", "/ask summarize the policy", None)
    assert (routed.role, routed.how) == ("interviewer", "surface")


def test_an_unknown_command_is_plain_text_and_never_a_role() -> None:
    routed = pick_role("office_chat", "/approve everything", None)
    assert (routed.role, routed.how) == (None, "clarify")
    assert routed.text == "/approve everything"


def test_ambiguous_text_goes_to_the_classifier_and_its_choice_is_used() -> None:
    classifier = ScriptedModel("classifier", [classifier_answer("summarizer")])
    routed = pick_role(
        "office_chat", "Anything new about the laptops?", PlacedModel(classifier, SELF_HOSTED)
    )
    assert (routed.role, routed.how) == ("summarizer", "model")
    [messages] = classifier.calls
    assert "<untrusted>" in messages[1].content
    assert "knowledge, summarizer, interviewer" in messages[0].content


def test_rules_that_disagree_go_to_the_classifier() -> None:
    classifier = ScriptedModel("classifier", [classifier_answer("knowledge")])
    routed = pick_role(
        "office_chat", "How do I describe the policy process?", PlacedModel(classifier, SELF_HOSTED)
    )
    assert (routed.role, routed.how) == ("knowledge", "model")
    assert "rules disagree" in routed.reason


@pytest.mark.parametrize(
    "answer",
    [
        json.dumps({"role": "reviewer"}),  # a real role, but not on this surface
        json.dumps({"role": "admin"}),
        json.dumps({"role": None}),
        json.dumps({"role": "knowledge", "also": "summarizer"}),
        json.dumps(["knowledge"]),
        "knowledge",
        TransportError("down"),
        TimeoutError(),
    ],
)
def test_anything_but_one_allowed_role_becomes_a_clarifying_question(
    answer: str | Exception,
) -> None:
    classifier = ScriptedModel("classifier", [answer])
    routed = pick_role(
        "office_chat", "Anything new about the laptops?", PlacedModel(classifier, SELF_HOSTED)
    )
    assert (routed.role, routed.how) == (None, "clarify")
    assert len(classifier.calls) == 1


def test_no_classifier_means_clarify_and_the_reply_offers_the_roles() -> None:
    routed = pick_role("office_chat", "hello", None)
    assert (routed.role, routed.how) == (None, "clarify")
    assert "documents" in CLARIFY_REPLY
    assert "summarize" in CLARIFY_REPLY


def test_an_unknown_surface_is_refused_before_anything_else() -> None:
    with pytest.raises(RoutingError):
        pick_role("admin_console", "/ask anything", None)


def test_rule_matching_sees_through_lookalike_letters() -> None:
    disguised = "Can you " + chr(0x0455) + "ummarize the finance channel?"  # Cyrillic s
    routed = pick_role("office_chat", disguised, None)
    assert (routed.role, routed.how) == ("summarizer", "rule")


# --- an open conversation keeps its follow-ups -----------------------------------------


def test_a_follow_up_no_rule_claims_stays_with_the_open_interview_without_a_model() -> None:
    classifier = ScriptedModel("classifier", [])  # any call would raise
    routed = pick_role(
        "office_chat",
        "Then my manager approves it",
        PlacedModel(classifier, SELF_HOSTED),
        continuing="interviewer",
    )
    assert (routed.role, routed.how, routed.text) == (
        "interviewer",
        "continue",
        "Then my manager approves it",
    )
    assert classifier.calls == []


@pytest.mark.parametrize(
    ("text", "role", "how"),
    [
        ("Do I need a receipt for that?", "knowledge", "rule"),
        ("Catch me up on #general", "summarizer", "rule"),
        ("/ask what is the travel limit", "knowledge", "command"),
        ("/summarize #general", "summarizer", "command"),
    ],
)
def test_a_command_or_another_roles_rule_still_leaves_the_open_interview(
    text: str, role: str, how: str
) -> None:
    routed = pick_role("office_chat", text, None, continuing="interviewer")
    assert (routed.role, routed.how) == (role, how)


def test_rules_that_disagree_stay_with_the_open_role_when_it_is_one_of_them() -> None:
    routed = pick_role(
        "office_chat", "How do I describe the policy process?", None, continuing="interviewer"
    )
    assert (routed.role, routed.how) == ("interviewer", "continue")


def test_rules_that_disagree_without_the_open_role_still_go_to_the_classifier() -> None:
    classifier = ScriptedModel("classifier", [classifier_answer("summarizer")])
    routed = pick_role(
        "office_chat",
        "Can I get a summary of the policy?",  # knowledge and summarizer rules, not interviewer
        PlacedModel(classifier, SELF_HOSTED),
        continuing="interviewer",
    )
    assert (routed.role, routed.how) == ("summarizer", "model")
    assert "rules disagree: knowledge, summarizer" in routed.reason
    assert len(classifier.calls) == 1


@pytest.mark.parametrize("continuing", ["reviewer", "admin", ""])
def test_an_open_role_the_surface_does_not_offer_is_never_reached(continuing: str) -> None:
    routed = pick_role("office_chat", "Then my manager approves it", None, continuing=continuing)
    assert (routed.role, routed.how) == (None, "clarify")


@pytest.mark.parametrize("text", ["", "   ", chr(10), " " + chr(9) + " "])
def test_empty_text_never_becomes_an_interview_turn(text: str) -> None:
    routed = pick_role("office_chat", text, None, continuing="interviewer")
    assert (routed.role, routed.how) == (None, "clarify")
