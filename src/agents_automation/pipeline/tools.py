"""In-memory tools and store. A real deployment would put a database behind the same tools.

Who may do what to a record:
* authors (the interviewed employees) read, save drafts and submit;
* reviewers read and approve, and never their own work (a reviewer who is also an author
  cannot approve);
* an approval binds to the exact revision that was screened and submitted: saving a new draft
  clears the submission and any approval, so screened and approved content cannot diverge;
* a hold is sticky for the session: saving again or re-finalizing cannot re-roll it, and
  every screening outcome is appended to a log the security reviewer can read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import ValidationError

from agents_automation.models import StepList
from agents_automation.pipeline.executor import Tool
from agents_automation.pipeline.policy import visible_to


class StoreUnavailableError(Exception):
    pass


class AccessDeniedError(Exception):
    pass


class InvalidContentError(Exception):
    pass


class StaleRevisionError(Exception):
    """The record changed after the revision this action refers to."""


FINAL_STATUSES = ("delivered", "delivered_flagged", "held_for_review")
APPROVABLE_STATUSES = ("delivered", "delivered_flagged")


@dataclass
class DraftRecord:
    session_id: str
    authors: frozenset[str]
    reviewers: frozenset[str] = frozenset()
    draft_json: str | None = None
    revision: int = 0
    outcome_status: str | None = None
    submitted_revision: int | None = None
    approved: bool = False
    held: bool = False  # sticky: once held, only a security reviewer can release the session
    screenings: list[tuple[int, str]] = field(default_factory=list)  # append-only log

    @property
    def scope(self) -> frozenset[str]:
        return self.authors | self.reviewers


class InMemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, DraftRecord] = {}
        self.fail_writes = False  # flip to simulate an outage

    def create_session(
        self,
        session_id: str,
        participants: frozenset[str],
        reviewers: frozenset[str] = frozenset(),
    ) -> DraftRecord:
        if session_id in self.sessions:  # re-creating would hand the record to new authors
            raise ValueError(f"session {session_id!r} already exists")
        record = DraftRecord(session_id=session_id, authors=participants, reviewers=reviewers)
        self.sessions[session_id] = record
        return record

    def _get(self, session_id: str) -> DraftRecord:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise AccessDeniedError("no such session") from None

    def draft(self, session_id: str) -> StepList | None:
        record = self._get(session_id)
        if record.draft_json is None:
            return None
        return StepList.model_validate_json(record.draft_json)

    def _get_scoped(self, inputs: Mapping[str, str]) -> DraftRecord:
        """Every tool, read or write, is clamped to the record's access scope."""
        record = self._get(inputs["session_id"])
        if not visible_to(record.scope, {inputs["viewer_id"]}):
            raise AccessDeniedError("caller outside the record's access scope")
        return record

    def _get_as_author(self, inputs: Mapping[str, str]) -> DraftRecord:
        record = self._get_scoped(inputs)
        if inputs["viewer_id"] not in record.authors:
            raise AccessDeniedError("only an author of the record may change it")
        return record

    # --- tools -------------------------------------------------------------------------

    def fetch_draft(self, inputs: Mapping[str, str]) -> Mapping[str, str]:
        record = self._get_scoped(inputs)
        return {
            "draft": record.draft_json or "",
            "revision": str(record.revision),
            "held": "true" if record.held else "false",
        }

    def save_draft(self, inputs: Mapping[str, str]) -> Mapping[str, str]:
        if self.fail_writes:
            raise StoreUnavailableError("store is unavailable")
        record = self._get_as_author(inputs)
        try:
            StepList.model_validate_json(inputs["content"])
        except ValidationError:
            raise InvalidContentError("content is not a valid step list") from None
        record.draft_json = inputs["content"]
        record.revision += 1
        # New content has not been screened: any earlier submission or approval is void.
        # A hold is NOT cleared: re-saving cannot be used to re-roll the screening.
        record.outcome_status = "held_for_review" if record.held else None
        record.submitted_revision = None
        record.approved = False
        return {"revision": str(record.revision)}

    def submit_for_review(self, inputs: Mapping[str, str]) -> Mapping[str, str]:
        if self.fail_writes:
            raise StoreUnavailableError("store is unavailable")
        record = self._get_as_author(inputs)
        if inputs["status"] not in FINAL_STATUSES:
            raise InvalidContentError(f"cannot submit with status {inputs['status']!r}")
        if record.draft_json is None:
            raise InvalidContentError("nothing to submit")
        if inputs["revision"] != str(record.revision):
            raise StaleRevisionError("the draft changed after it was screened")
        record.screenings.append((record.revision, inputs["status"]))
        if record.held and inputs["status"] != "held_for_review":
            raise InvalidContentError("held for review: only a reviewer can release it")
        record.outcome_status = inputs["status"]
        record.submitted_revision = record.revision
        record.approved = False  # a (re-)submission always needs a fresh approval
        if inputs["status"] == "held_for_review":
            record.held = True
        queue = "security_review" if inputs["status"] == "held_for_review" else "approval"
        return {"queue": queue}

    def approve(self, inputs: Mapping[str, str]) -> Mapping[str, str]:
        record = self._get_scoped(inputs)
        viewer = inputs["viewer_id"]
        if viewer not in record.reviewers or viewer in record.authors:
            raise AccessDeniedError("only a reviewer who is not an author may approve")
        if record.outcome_status not in APPROVABLE_STATUSES:
            raise InvalidContentError("only delivered records can be approved")
        if record.submitted_revision != record.revision:
            raise StaleRevisionError("the draft changed after it was submitted")
        record.approved = True
        return {"approved": "true"}

    def tools(self) -> dict[str, Tool]:
        return {
            "fetch_draft": self.fetch_draft,
            "save_draft": self.save_draft,
            "submit_for_review": self.submit_for_review,
            "approve": self.approve,
        }
