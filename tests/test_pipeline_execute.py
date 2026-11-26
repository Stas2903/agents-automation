"""Authorization and execution: default deny, per-step re-check, values only, fail-closed."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace

import pytest

from agents_automation.pipeline.executor import StepStatus, Tool, execute
from agents_automation.pipeline.planner import Plan, PlanStep, Request, make_plan
from agents_automation.pipeline.policy import (
    ALLOWED,
    authorize,
    may_run,
    visible_to,
)
from agents_automation.pipeline.seal import PlanViolationError, SealedPlan, bind_values, seal
from agents_automation.pipeline.tools import (
    AccessDeniedError,
    InMemoryStore,
    InvalidContentError,
    StaleRevisionError,
)

from .helpers import linear


def _sealed(request: Request, role: str = "interviewer") -> SealedPlan:
    return seal(make_plan(role, request))


class Recorder:
    """Tool doubles that record every call, so tests can prove side effects did NOT happen."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def tool(self, name: str, output: Mapping[str, str] | None = None) -> Tool:
        def run(inputs: Mapping[str, str]) -> Mapping[str, str]:
            self.calls.append((name, dict(inputs)))
            return output if output is not None else {}

        return run


def _plan(role: str, *steps: PlanStep) -> SealedPlan:
    return seal(Plan(role, steps))


# --- policy -----------------------------------------------------------------------------


def test_default_deny() -> None:
    assert may_run("interviewer", "save_draft", "write")
    assert not may_run("interviewer", "approve", "write")
    assert not may_run("interviewer", "save_draft", "read")  # right tool, wrong operation
    assert not may_run("intruder", "fetch_draft", "read")
    assert not may_run("interviewer", "fetch_draft", "read", table=frozenset())


def test_authorize_accepts_template_plans() -> None:
    req = Request("interview_chat", "interview_turn", "s-1", "u-1")
    assert authorize(_sealed(req)).allowed
    review = Request("review_console", "review", "s-1", "u-1")
    assert authorize(_sealed(review, role="reviewer")).allowed


def test_authorize_denies_an_empty_plan() -> None:
    decision = authorize(_plan("interviewer"))
    assert not decision.allowed
    assert "empty" in decision.reason


def test_authorize_denies_a_plan_with_one_forbidden_step() -> None:
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("approve", "approve", "write", {}),
    )
    decision = authorize(sealed)
    assert not decision.allowed
    assert "approve" in decision.reason


def test_authorize_denies_a_tampered_plan() -> None:
    req = Request("interview_chat", "interview_turn", "s-1", "u-1")
    sealed = _sealed(req)
    tampered = replace(sealed, digest="sha256:" + "f" * 64)
    assert not authorize(tampered).allowed


@pytest.mark.parametrize(
    ("scope", "viewers", "expected"),
    [
        ({"a", "b"}, {"a"}, True),
        ({"a", "b"}, {"a", "b"}, True),
        ({"a"}, {"a", "b"}, False),  # one outsider is enough to deny
        (None, {"a"}, False),  # missing scope = deny
        (set(), {"a"}, False),
        ({"a"}, set(), False),
    ],
)
def test_visible_to(scope: set[str] | None, viewers: set[str], expected: bool) -> None:
    assert visible_to(scope, viewers) is expected


# --- executor ---------------------------------------------------------------------------


def test_executor_runs_steps_and_passes_values_between_them() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {"session_id": "s-1"}),
        PlanStep("save", "save_draft", "write", {"session_id": "s-1", "content": None}),
    )
    tools = {
        "fetch_draft": rec.tool("fetch_draft", {"content": "from-read", "_debug": "secret"}),
        "save_draft": rec.tool("save_draft", {"revision": "1"}),
    }
    result = execute(sealed, tools)
    assert result.ok
    assert rec.calls == [
        ("fetch_draft", {"session_id": "s-1"}),
        ("save_draft", {"session_id": "s-1", "content": "from-read"}),
    ]
    # reserved keys are stripped from the values channel
    assert dict(result.outcome("read").output) == {"content": "from-read"}


def test_tool_output_cannot_overwrite_a_filled_slot() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {"session_id": "s-1"}),
        PlanStep("save", "save_draft", "write", {"session_id": "s-1", "content": "mine"}),
    )
    tools = {
        "fetch_draft": rec.tool("fetch_draft", {"session_id": "s-2", "content": "theirs"}),
        "save_draft": rec.tool("save_draft"),
    }
    execute(sealed, tools)
    assert rec.calls[1] == ("save_draft", {"session_id": "s-1", "content": "mine"})


def test_tool_output_cannot_add_keys_to_later_steps() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("save", "save_draft", "write", {"content": None}),
    )
    tools = {
        "fetch_draft": rec.tool("fetch_draft", {"content": "c", "drop_table": "yes"}),
        "save_draft": rec.tool("save_draft"),
    }
    execute(sealed, tools)
    assert rec.calls[1] == ("save_draft", {"content": "c"})


def test_executor_rechecks_authorization_on_every_step() -> None:
    # Authorized against the full table, then the permission is revoked before execution.
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("save", "save_draft", "write", {}),
    )
    assert authorize(sealed).allowed
    revoked = ALLOWED - {("interviewer", "save_draft", "write")}
    tools = {"fetch_draft": rec.tool("fetch_draft"), "save_draft": rec.tool("save_draft")}
    result = execute(sealed, tools, table=revoked)
    assert result.outcome("read").status is StepStatus.OK
    assert result.outcome("save").status is StepStatus.DENIED
    assert [c[0] for c in rec.calls] == ["fetch_draft"]  # the write never happened
    assert not result.ok


def test_first_denial_halts_later_steps() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("a", "approve", "write", {}),  # not permitted for the interviewer
        PlanStep("b", "save_draft", "write", {}),
    )
    tools = {"approve": rec.tool("approve"), "save_draft": rec.tool("save_draft")}
    result = execute(sealed, tools)
    assert [o.status for o in result.outcomes] == [StepStatus.DENIED, StepStatus.SKIPPED]
    assert rec.calls == []


def test_unknown_tool_is_denied_and_halts() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("save", "save_draft", "write", {}),
    )
    result = execute(sealed, {"save_draft": rec.tool("save_draft")})
    assert result.outcome("read").status is StepStatus.DENIED
    assert result.outcome("read").error == "unknown tool"
    assert result.outcome("save").status is StepStatus.SKIPPED
    assert rec.calls == []


def test_tool_exception_fails_closed_and_logs_only_the_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = Recorder()

    def boom(inputs: Mapping[str, str]) -> Mapping[str, str]:
        raise ValueError("customer secret: card 4111")

    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("save", "save_draft", "write", {}),
    )
    with caplog.at_level(logging.WARNING):
        result = execute(sealed, {"fetch_draft": boom, "save_draft": rec.tool("save_draft")})
    assert result.outcome("read").status is StepStatus.FAILED
    assert result.outcome("read").error == "ValueError"
    assert result.outcome("save").status is StepStatus.SKIPPED
    assert rec.calls == []
    assert "ValueError" in caplog.text
    assert "4111" not in caplog.text


def test_missing_slot_value_fails_closed() -> None:
    rec = Recorder()
    sealed = _plan("interviewer", PlanStep("save", "save_draft", "write", {"content": None}))
    result = execute(sealed, {"save_draft": rec.tool("save_draft")})
    assert result.outcome("save").status is StepStatus.FAILED
    assert "content" in (result.outcome("save").error or "")
    assert rec.calls == []


def test_only_runs_the_requested_steps() -> None:
    rec = Recorder()
    sealed = _plan(
        "interviewer",
        PlanStep("read", "fetch_draft", "read", {}),
        PlanStep("save", "save_draft", "write", {}),
    )
    tools = {"fetch_draft": rec.tool("fetch_draft"), "save_draft": rec.tool("save_draft")}
    result = execute(sealed, tools, only={"save"})
    assert result.outcome("read").status is StepStatus.NOT_REQUESTED
    assert result.outcome("save").status is StepStatus.OK
    assert rec.calls == [("save_draft", {})]
    assert result.ok


def test_only_rejects_unknown_step_ids() -> None:
    sealed = _plan("interviewer", PlanStep("read", "fetch_draft", "read", {}))
    with pytest.raises(PlanViolationError):
        execute(sealed, {}, only={"read", "approve"})


def test_executor_refuses_a_tampered_plan() -> None:
    sealed = _plan("interviewer", PlanStep("read", "fetch_draft", "read", {}))
    tampered = replace(
        sealed,
        plan=Plan("interviewer", (*sealed.plan.steps, PlanStep("x", "approve", "write", {}))),
    )
    rec = Recorder()
    with pytest.raises(PlanViolationError):
        execute(tampered, {"fetch_draft": rec.tool("r"), "approve": rec.tool("a")})
    assert rec.calls == []


# --- in-memory tools through the real executor -----------------------------------------


def test_read_is_clamped_to_the_record_scope(store: InMemoryStore) -> None:
    # u-1 asks for s-2, whose scope is {u-2}: the read step fails closed.
    req = Request("interview_chat", "interview_turn", "s-2", "u-1")
    result = execute(_sealed(req), store.tools(), only={"read"})
    assert result.outcome("read").status is StepStatus.FAILED
    assert result.outcome("read").error == "AccessDeniedError"


def test_save_through_executor_persists_a_valid_draft(store: InMemoryStore) -> None:
    req = Request("interview_chat", "interview_turn", "s-1", "u-1")
    sealed = bind_values(_sealed(req), "save", {"content": linear().model_dump_json()})
    result = execute(sealed, store.tools(), only={"save"})
    assert result.outcome("save").status is StepStatus.OK
    assert dict(result.outcome("save").output) == {"revision": "1"}
    assert store.draft("s-1") == linear()


def test_save_rejects_content_that_is_not_a_step_list(store: InMemoryStore) -> None:
    req = Request("interview_chat", "interview_turn", "s-1", "u-1")
    sealed = bind_values(_sealed(req), "save", {"content": '{"rm": "-rf"}'})
    result = execute(sealed, store.tools(), only={"save"})
    assert result.outcome("save").error == "InvalidContentError"
    assert store.draft("s-1") is None


def test_save_during_outage_fails_and_stores_nothing(store: InMemoryStore) -> None:
    store.fail_writes = True
    req = Request("interview_chat", "interview_turn", "s-1", "u-1")
    sealed = bind_values(_sealed(req), "save", {"content": linear().model_dump_json()})
    result = execute(sealed, store.tools(), only={"save"})
    assert result.outcome("save").error == "StoreUnavailableError"
    assert store.sessions["s-1"].revision == 0


def _reviewed_store() -> InMemoryStore:
    """s-r: author u-1, reviewer r-1, one saved draft (revision 1)."""
    store = InMemoryStore()
    store.create_session("s-r", frozenset({"u-1"}), reviewers=frozenset({"r-1"}))
    store.save_draft(
        {"session_id": "s-r", "viewer_id": "u-1", "content": linear().model_dump_json()}
    )
    return store


def _submit(store: InMemoryStore, status: str, revision: str = "1") -> Mapping[str, str]:
    return store.submit_for_review(
        {"session_id": "s-r", "viewer_id": "u-1", "status": status, "revision": revision}
    )


def _approve(store: InMemoryStore, viewer: str = "r-1") -> Mapping[str, str]:
    return store.approve({"session_id": "s-r", "viewer_id": viewer})


def test_only_delivered_records_can_be_approved() -> None:
    store = _reviewed_store()
    with pytest.raises(InvalidContentError, match="only delivered"):
        _approve(store)
    assert _submit(store, "held_for_review") == {"queue": "security_review"}
    with pytest.raises(InvalidContentError, match="only delivered"):
        _approve(store)
    # a held session cannot be re-submitted as delivered, not even with a new revision
    with pytest.raises(InvalidContentError, match="held for review"):
        _submit(store, "delivered")
    store.save_draft(
        {"session_id": "s-r", "viewer_id": "u-1", "content": linear("v2").model_dump_json()}
    )
    assert store.sessions["s-r"].outcome_status == "held_for_review"
    with pytest.raises(InvalidContentError, match="held for review"):
        _submit(store, "delivered", revision="2")
    with pytest.raises(InvalidContentError, match="only delivered"):
        _approve(store)
    assert store.sessions["s-r"].screenings == [
        (1, "held_for_review"),
        (1, "delivered"),
        (2, "delivered"),
    ]


def test_a_delivered_record_can_be_approved_by_a_reviewer() -> None:
    store = _reviewed_store()
    assert _submit(store, "delivered") == {"queue": "approval"}
    assert _approve(store) == {"approved": "true"}
    assert store.sessions["s-r"].approved is True


def test_re_submitting_an_approved_revision_needs_a_fresh_approval() -> None:
    store = _reviewed_store()
    _submit(store, "delivered")
    _approve(store)
    assert _submit(store, "held_for_review") == {"queue": "security_review"}
    record = store.sessions["s-r"]
    assert (record.outcome_status, record.approved) == ("held_for_review", False)


def test_flagged_records_go_to_approval_and_can_be_approved() -> None:
    store = _reviewed_store()
    assert _submit(store, "delivered_flagged") == {"queue": "approval"}
    assert _approve(store) == {"approved": "true"}


def test_an_author_cannot_approve_their_own_record() -> None:
    store = _reviewed_store()
    _submit(store, "delivered")
    with pytest.raises(AccessDeniedError, match="not an author"):
        _approve(store, viewer="u-1")
    # also when the same person is listed as both author and reviewer
    store.sessions["s-r"].reviewers = frozenset({"u-1", "r-1"})
    with pytest.raises(AccessDeniedError, match="not an author"):
        _approve(store, viewer="u-1")
    assert store.sessions["s-r"].approved is False


def test_a_reviewer_cannot_save_or_submit() -> None:
    store = _reviewed_store()
    with pytest.raises(AccessDeniedError, match="only an author"):
        store.save_draft(
            {"session_id": "s-r", "viewer_id": "r-1", "content": linear("X").model_dump_json()}
        )
    with pytest.raises(AccessDeniedError, match="only an author"):
        store.submit_for_review(
            {"session_id": "s-r", "viewer_id": "r-1", "status": "delivered", "revision": "1"}
        )


def test_a_stale_revision_cannot_be_submitted() -> None:
    store = _reviewed_store()
    with pytest.raises(StaleRevisionError):
        _submit(store, "delivered", revision="0")
    assert store.sessions["s-r"].outcome_status is None


def test_saving_after_submission_voids_it_so_unscreened_content_is_never_approved() -> None:
    store = _reviewed_store()
    _submit(store, "delivered")
    planted = linear("APPROVED BY THE CFO").model_dump_json()
    store.save_draft({"session_id": "s-r", "viewer_id": "u-1", "content": planted})
    record = store.sessions["s-r"]
    assert (record.outcome_status, record.submitted_revision, record.revision) == (None, None, 2)
    with pytest.raises(InvalidContentError):
        _approve(store)
    assert record.approved is False


def test_saving_after_approval_revokes_the_approval() -> None:
    store = _reviewed_store()
    _submit(store, "delivered")
    _approve(store)
    store.save_draft(
        {"session_id": "s-r", "viewer_id": "u-1", "content": linear("v2").model_dump_json()}
    )
    assert store.sessions["s-r"].approved is False


def test_approval_refuses_a_revision_that_changed_after_submission() -> None:
    store = _reviewed_store()
    _submit(store, "delivered")
    store.sessions["s-r"].revision = 2  # a write path that forgot to clear the submission
    with pytest.raises(StaleRevisionError):
        _approve(store)


def test_submit_rejects_unknown_status_and_empty_record(store: InMemoryStore) -> None:
    with pytest.raises(Exception, match="nothing to submit"):
        store.submit_for_review({"session_id": "s-1", "viewer_id": "u-1", "status": "delivered"})
    store.sessions["s-1"].draft_json = linear().model_dump_json()
    with pytest.raises(Exception, match="cannot submit"):
        store.submit_for_review({"session_id": "s-1", "viewer_id": "u-1", "status": "approved"})


@pytest.mark.parametrize("tool", ["save_draft", "submit_for_review", "approve"])
def test_every_write_tool_is_clamped_to_the_record_scope(store: InMemoryStore, tool: str) -> None:
    # u-1 is not in the scope of s-2: no write tool may touch it, whatever the other inputs.
    store.sessions["s-2"].draft_json = linear().model_dump_json()
    store.sessions["s-2"].outcome_status = "delivered"
    inputs = {
        "session_id": "s-2",
        "viewer_id": "u-1",
        "content": linear().model_dump_json(),
        "status": "delivered",
    }
    with pytest.raises(Exception, match="access scope"):
        store.tools()[tool](inputs)
    assert store.sessions["s-2"].revision == 0
    assert store.sessions["s-2"].approved is False


def test_cross_session_save_through_executor_fails_closed(store: InMemoryStore) -> None:
    req = Request("interview_chat", "interview_turn", "s-2", "u-1")
    sealed = bind_values(_sealed(req), "save", {"content": linear().model_dump_json()})
    result = execute(sealed, store.tools(), only={"save"})
    assert result.outcome("save").error == "AccessDeniedError"
    assert store.draft("s-2") is None


def test_outbound_network_is_blocked_in_tests() -> None:
    import socket

    with socket.socket() as s:
        for attempt in (
            lambda: s.connect(("example.com", 80)),
            lambda: s.connect_ex(("example.com", 80)),
        ):
            with pytest.raises(RuntimeError, match="must not touch the network"):
                attempt()
