"""Mutation check for the critical guards.

For each mutant: copy the project to a temp directory, break exactly one guard, run the whole
test suite, and require that it FAILS, and that the named "killer" test is among the
failures. A guard whose removal leaves the suite green is not really tested.

Usage:  uv run python scripts/mutate_guards.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = "src/agents_automation"


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    killer: str  # a test (node id substring) that must fail


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "plan hash ignores slot keys",
        f"{PKG}/pipeline/seal.py",
        '"slot_keys": sorted(s.slots),',
        "",
        "test_hash_changes_when_structure_changes",
    ),
    Mutant(
        "bind_values accepts new slot keys",
        f"{PKG}/pipeline/seal.py",
        "    if undeclared:",
        "    if False:",
        "test_bind_values_rejects_new_slot_key",
    ),
    Mutant(
        "bind_values overwrites trusted slots",
        f"{PKG}/pipeline/seal.py",
        "        if step.slots[key] is not None:",
        "        if False:",
        "test_bind_values_refuses_to_overwrite_a_trusted_slot",
    ),
    Mutant(
        "untrusted text can close its fence",
        f"{PKG}/llm/text.py",
        'replace_spans(text, spans, "[tag removed]")',
        "text",
        "test_untrusted_text_cannot_close_its_fence",
    ),
    Mutant(
        "default allow instead of default deny",
        f"{PKG}/pipeline/policy.py",
        "    return (role, tool, operation) in table",
        "    return True",
        "test_default_deny",
    ),
    Mutant(
        "executor trusts the earlier authorization",
        f"{PKG}/pipeline/executor.py",
        "        if not may_run(plan.role, step.tool, step.operation, table):",
        "        if False:",
        "test_executor_rechecks_authorization_on_every_step",
    ),
    Mutant(
        "executor keeps going after a tool failure",
        f"{PKG}/pipeline/executor.py",
        "error=type(exc).__name__)\n            )\n            halted = True",
        "error=type(exc).__name__)\n            )\n            halted = False",
        "test_tool_exception_fails_closed_and_logs_only_the_class",
    ),
    Mutant(
        "write tools ignore the access scope",
        f"{PKG}/pipeline/tools.py",
        '        if not visible_to(record.scope, {inputs["viewer_id"]}):',
        "        if False:",
        "test_every_write_tool_is_clamped_to_the_record_scope",
    ),
    Mutant(
        "detector ignores negation",
        f"{PKG}/interview/honesty.py",
        "    if _negated(clause, match.group(0)):",
        "    if False:",
        "test_lets_it_pass",
    ),
    Mutant(
        "honesty guard never fires",
        f"{PKG}/interview/turn.py",
        "unsupported_claims=[c.sentence for c in claims] if not ledger.saved else [],",
        "unsupported_claims=[],",
        "test_claim_without_save_triggers_one_correction_that_saves",
    ),
    Mutant(
        "correction sent as a trailing system message (assistant prefill)",
        f"{PKG}/interview/turn.py",
        'Message("user", note),',
        'Message("system", note),',
        "test_claim_without_save_triggers_one_correction_that_saves",
    ),
    Mutant(
        "validator accepts loops that never finish",
        f"{PKG}/interview/validator.py",
        'if s.id not in can_finish and s.kind != "finish" and s.next:',
        "if False:",
        "test_loop_with_no_way_to_a_finish_is_caught",
    ),
    Mutant(
        "layer 1 hits do not hold",
        f"{PKG}/injection/screen.py",
        "    if hits:",
        "    if False:",
        "test_pattern_layer_holds_without_calling_the_judge",
    ),
    Mutant(
        "judge failure fails open WITHOUT a flag",
        f"{PKG}/injection/screen.py",
        'Status.DELIVERED_FLAGGED, "judge", (), verdict, text, f"fail open',
        'Status.DELIVERED, "judge", (), verdict, text, f"fail open',
        "test_judge_failure_fails_open_but_flagged",
    ),
    Mutant(
        "a single run is enough for a verdict",
        f"{PKG}/evals/runmatrix.py",
        "    if len(runs) < MIN_RUNS:",
        "    if False:",
        "test_fewer_than_two_runs_is_rejected",
    ),
    Mutant(
        "VOID runs are counted",
        f"{PKG}/evals/runmatrix.py",
        "    counted = [r for r in runs if r is not RunOutcome.VOID]",
        "    counted = list(runs)",
        "test_grade_runs",
    ),
    Mutant(
        "a model outage counts as a verdict",
        f"{PKG}/evals/runner.py",
        "    if infrastructure is not None:",
        "    if False:",
        "test_repair_outage_makes_the_run_void_not_fail",
    ),
    Mutant(
        "delivered counts even when nothing was submitted",
        f"{PKG}/evals/scenarios.py",
        "if expected is not Status.INVALID and not t.final.submitted:",
        "if False:",
        "test_status_check_requires_a_real_submission",
    ),
    Mutant(
        "executor keeps going after a missing slot",
        f"{PKG}/pipeline/executor.py",
        '{sorted(missing)}",\n                )\n            )\n            halted = True',
        '{sorted(missing)}",\n                )\n            )\n            halted = False',
        "test_missing_slot_halts_the_rest_of_the_plan",
    ),
    Mutant(
        "a stale revision can be submitted",
        f"{PKG}/pipeline/tools.py",
        'if inputs["revision"] != str(record.revision):',
        "if False:",
        "test_a_stale_revision_cannot_be_submitted",
    ),
    Mutant(
        "blank judge claims are kept (mask everything)",
        f"{PKG}/injection/judge.py",
        "if c.strip() and c in text",
        "if c in text",
        "test_blank_judge_claims_are_dropped_and_never_mask_everything",
    ),
    Mutant(
        "an author may approve their own record",
        f"{PKG}/pipeline/tools.py",
        "if viewer not in record.reviewers or viewer in record.authors:",
        "if viewer not in record.reviewers:",
        "test_an_author_cannot_approve_their_own_record",
    ),
    Mutant(
        "saving does not void an earlier submission",
        f"{PKG}/pipeline/tools.py",
        "        record.submitted_revision = None\n        record.approved = False\n",
        "",
        "test_saving_after_submission_voids_it_so_unscreened_content_is_never_approved",
    ),
    Mutant(
        "a held session can be re-submitted as delivered",
        f"{PKG}/pipeline/tools.py",
        'if record.held and inputs["status"] != "held_for_review":',
        "if False:",
        "test_only_delivered_records_can_be_approved",
    ),
    Mutant(
        "re-saving clears a hold",
        f"{PKG}/pipeline/tools.py",
        'record.outcome_status = "held_for_review" if record.held else None',
        "record.outcome_status = None",
        "test_re_saving_the_same_content_cannot_reroll_a_hold",
    ),
    Mutant(
        "re-finalizing a held session can mint a new verdict",
        f"{PKG}/finalize.py",
        "status = Status.HELD_FOR_REVIEW if already_held else screening.status",
        "status = screening.status",
        "test_re_finalizing_a_held_record_cannot_release_it",
    ),
    Mutant(
        "a re-submission keeps an old approval",
        f"{PKG}/pipeline/tools.py",
        "        record.approved = False  # a (re-)submission always needs a fresh approval\n",
        "",
        "test_re_submitting_an_approved_revision_needs_a_fresh_approval",
    ),
    Mutant(
        "screening matches the raw text instead of the normalised view",
        f"{PKG}/injection/screen.py",
        "for hit in find_markers(view.text):",
        "for hit in find_markers(text):",
        "test_spelling_tricks_do_not_hide_a_stamp_from_the_pattern_layer",
    ),
    Mutant(
        "combining marks are not stripped",
        f"{PKG}/llm/text.py",
        '_INVISIBLE = {"Cf", "Mn", "Me"}',
        '_INVISIBLE = {"Cf"}',
        "test_invisible_marks_and_more_look_alikes_do_not_hide_a_stamp",
    ),
    Mutant(
        "earlier-save references are trusted without an earlier draft",
        f"{PKG}/interview/honesty.py",
        "if earlier_saves_possible and _EARLIER.match(tail):",
        "if _EARLIER.match(tail):",
        "test_earlier_save_references_count_as_claims_when_no_draft_exists",
    ),
    Mutant(
        "'before' counts as a reference to an earlier save",
        f"{PKG}/interview/honesty.py",
        "(?:earlier|previously|last",
        "(?:earlier|before|previously|last",
        "test_flags_the_claim",
    ),
    Mutant(
        "a fence tag must close on the same short line to be caught",
        f"{PKG}/llm/text.py",
        r"untrusted\b(?:[^<>\n]{0,200}>)?",
        r"untrusted\b[^>\n]{0,40}>",
        "test_long_split_or_lookalike_fence_tags_are_neutralised",
    ),
    Mutant(
        "claims about the system's own storage are excused",
        f"{PKG}/interview/honesty.py",
        "if not (_DRAFT_TARGET_RE.match(target) or _OWN_STORAGE.match(target)):",
        "if not _DRAFT_TARGET_RE.match(target):",
        "test_flags_the_claim",
    ),
    Mutant(
        "claims inside a question sentence are ignored",
        f"{PKG}/interview/honesty.py",
        '    if not sentence.endswith("?"):\n        return sentence\n',
        '    if not sentence.endswith("?"):\n        return sentence\n    return None\n',
        "test_flags_the_claim",
    ),
    Mutant(
        "residual stitched claims survive the fallback",
        f"{PKG}/interview/turn.py",
        "            if claims:  # removing sentences can stitch a new claim together",
        "            if False:",
        "test_stitched_residual_claim_is_replaced_by_the_honest_note",
    ),
    Mutant(
        "CLI exits 0 on a failing gate",
        f"{PKG}/evals/runner.py",
        "GateStatus.PASS: 0, GateStatus.FAIL: 1, GateStatus.INCOMPLETE: 2}[gate.status]",
        "GateStatus.PASS: 0, GateStatus.FAIL: 0, GateStatus.INCOMPLETE: 2}[gate.status]",
        "test_cli_exit_codes",
    ),
    Mutant(
        "FLAKY passes a hard gate",
        f"{PKG}/evals/runmatrix.py",
        "verdicts[s] in (Verdict.FAIL, Verdict.FLAKY)",
        "verdicts[s] is Verdict.FAIL",
        "test_fail_or_flaky_on_a_hard_gate_fails_it",
    ),
    Mutant(
        "'earlier than' excuses a claim as an earlier save",
        f"{PKG}/interview/honesty.py",
        r'r"(?!\s+than\b)",',
        'r"",',
        "test_flags_the_claim",
    ),
    Mutant(
        "plural storage nouns are the employee's own systems",
        f"{PKG}/interview/honesty.py",
        "(?:system|platform|database|server|record)s?|cloud)",
        "(?:system|platform|database|server|record)|cloud)",
        "test_flags_the_claim",
    ),
    Mutant(
        "'and then' excuses the conversation's own artifact",
        f"{PKG}/interview/honesty.py",
        "_PROCESS_CONTINUES.match(tail) and not about_artifact",
        "_PROCESS_CONTINUES.match(tail)",
        "test_flags_the_claim",
    ),
    Mutant(
        "lower-case Cherokee is not folded",
        f"{PKG}/llm/text.py",
        "        ch = ch.upper()  # lower-case Cherokee renders like Latin small capitals",
        "        pass",
        "test_every_confusable_table_entry_folds_to_its_latin_letter",
    ),
    Mutant(
        "confusables are looked up only after NFKC",
        f"{PKG}/llm/text.py",
        "    if ch in _CONFUSABLES:",
        "    if False:",
        "test_every_confusable_table_entry_folds_to_its_latin_letter",
    ),
    # --- knowledge role ----------------------------------------------------------------
    Mutant(
        "the access scope matches every group",
        f"{PKG}/knowledge/store.py",
        'WHERE grp IN ({marks})", sorted(groups)',
        'WHERE grp IN ({marks}) OR 1 = 1", sorted(groups)',
        "test_a_document_outside_the_readers_groups_is_never_retrieved",
    ),
    Mutant(
        "turning chunk ids into text skips the access check",
        f"{PKG}/knowledge/store.py",
        'f"WHERE c.chunk_id IN ({marks}) AND c.doc_id IN ({allowed})",',
        'f"WHERE c.chunk_id IN ({marks}) AND (c.doc_id IN ({allowed}) OR 1 = 1)",',
        "test_a_document_outside_the_readers_groups_is_never_retrieved",
    ),
    Mutant(
        "the vector scan skips the access check",
        f"{PKG}/knowledge/store.py",
        'WHERE doc_id IN ({allowed}) ORDER BY chunk_id",',
        'WHERE (doc_id IN ({allowed}) OR 1 = 1) ORDER BY chunk_id",',
        "test_a_document_outside_the_readers_groups_is_never_retrieved",
    ),
    Mutant(
        "user text reaches the full-text query unquoted",
        f"{PKG}/knowledge/store.py",
        'match = " OR ".join(f\'"{t}"\' for t in cleaned)',
        'match = " OR ".join(cleaned)',
        "test_full_text_query_syntax_in_user_text_is_treated_as_plain_words",
    ),
    Mutant(
        "an answer may cite passages it was not given",
        f"{PKG}/knowledge/role.py",
        "if not set(citations) <= set(allowed_ids):",
        "if False:",
        "test_the_gate_rejects_answers_that_are_not_grounded_in_the_given_passages",
    ),
    Mutant(
        "an answer without citations is accepted",
        f"{PKG}/knowledge/role.py",
        "if not citations or not all(isinstance(c, str) for c in citations):",
        "if not all(isinstance(c, str) for c in citations):",
        "test_the_gate_rejects_answers_that_are_not_grounded_in_the_given_passages",
    ),
    Mutant(
        "a question without a relevant source still goes to the model",
        f"{PKG}/knowledge/role.py",
        "        if not relevant:",
        "        if False:",
        "test_a_question_no_source_covers_gets_no_model_call",
    ),
    Mutant(
        "the reranker may name passages that were not candidates",
        f"{PKG}/knowledge/retrieve.py",
        "or not set(order) <= set(by_id)",
        "or False",
        "test_a_failing_reranker_is_skipped_and_the_fused_order_is_kept",
    ),
    Mutant(
        "a reranker not cleared for the passages is still called",
        f"{PKG}/knowledge/retrieve.py",
        "if reranker.placement.clearance < label:",
        "if False:",
        "test_a_reranker_not_cleared_for_the_passages_is_never_called",
    ),
    Mutant(
        "chunks with planted instructions are indexed",
        f"{PKG}/knowledge/ingest.py",
        "        planted = {i for i, c in enumerate(chunks) "
        "if _planted(_plain(c.heading, c.text))}",
        "        planted: set[int] = set()",
        "test_a_chunk_with_planted_instructions_is_quarantined_and_never_retrieved",
    ),
    Mutant(
        "unchanged chunks are embedded again",
        f"{PKG}/knowledge/ingest.py",
        "if known.get(ord_) == version:",
        "if False:",
        "test_reingesting_an_unchanged_corpus_embeds_nothing",
    ),
    # --- data boundary and budget ----------------------------------------------------
    Mutant(
        "the router ignores clearance when picking a model",
        f"{PKG}/llm/boundary.py",
        "if placed.placement.clearance >= sensitivity:",
        "if True:",
        "test_the_first_cleared_model_in_preference_order_is_used",
    ),
    Mutant(
        "clearance is never enforced",
        f"{PKG}/llm/boundary.py",
        "if placement.clearance < sensitivity:",
        "if False:",
        "test_an_embedder_not_cleared_for_a_document_never_sees_it",
    ),
    Mutant(
        "an unknown sensitivity label counts as public",
        f"{PKG}/llm/boundary.py",
        "            return cls.CONFIDENTIAL",
        "            return cls.PUBLIC",
        "test_labels_parse_and_unknown_labels_count_as_confidential",
    ),
    Mutant(
        "an optional call always fits the budget",
        f"{PKG}/budget.py",
        "if left >= needs_s:",
        "if True:",
        "test_an_optional_call_runs_only_while_it_still_fits",
    ),
    # --- summarizer --------------------------------------------------------------------
    Mutant(
        "every channel is readable by everyone",
        f"{PKG}/summarizer/chat.py",
        "return bool(self.member_groups & set(groups))",
        "return True",
        "test_channels_are_members_only",
    ),
    Mutant(
        "chat messages with planted instructions are kept",
        f"{PKG}/summarizer/chat.py",
        "(held if find_markers(normalize_text(msg.text)) else kept).append(msg)",
        "kept.append(msg)",
        "test_a_message_with_planted_instructions_is_held_out_of_everything",
    ),
    Mutant(
        "a message's day is its UTC day, not the team's",
        f"{PKG}/summarizer/chat.py",
        "return message.sent_at.astimezone(self.tz).date()",
        "return message.sent_at.date()",
        "test_a_message_belongs_to_its_day_in_the_team_timezone",
    ),
    Mutant(
        "a written summary may cite messages outside its period",
        f"{PKG}/summarizer/summaries.py",
        "if parsed is None or not set(parsed[1]) <= period_ids:",
        "if parsed is None:",
        "test_a_summary_citing_messages_outside_its_period_is_not_stored",
    ),
    Mutant(
        "a composed summary may cite anything that exists",
        f"{PKG}/summarizer/role.py",
        "if cid not in material or m is None or m.channel != period.channel:",
        "if m is None:",
        "test_the_judge_checks_channel_and_period_even_for_ids_in_the_material",
    ),
    Mutant(
        "a composed summary may cite messages outside the period",
        f"{PKG}/summarizer/role.py",
        "if not period.start <= chat.local_date(m) <= period.end:",
        "if False:",
        "test_the_judge_checks_channel_and_period_even_for_ids_in_the_material",
    ),
    Mutant(
        "the composer runs whatever the budget says",
        f"{PKG}/summarizer/role.py",
        'if budget.allow("compose", needs_s=COMPOSE_NEEDS_S):',
        "if True:",
        "test_a_spent_budget_skips_the_composer_and_says_so",
    ),
    Mutant(
        "a period in the future is accepted",
        f"{PKG}/summarizer/periods.py",
        "if start > today:",
        "if False:",
        "test_impossible_periods_are_refused",
    ),
    # --- routing and the interviewer's checks ----------------------------------------
    Mutant(
        "a chosen role outside the surface is accepted",
        f"{PKG}/pipeline/router.py",
        "if chosen not in roles:",
        "if False:",
        "test_a_chosen_role_outside_the_surface_is_refused",
    ),
    Mutant(
        "the classifier may name any role",
        f"{PKG}/intent.py",
        "    if isinstance(role, str) and role in roles:",
        "    if isinstance(role, str):",
        "test_anything_but_one_allowed_role_becomes_a_clarifying_question",
    ),
    Mutant(
        "a negated promise counts as a promise",
        f"{PKG}/interview/checks.py",
        "if _NEGATED_INTENT.search(view[: m.start()]):",
        "if False:",
        "test_everything_else_is_not_a_promise",
    ),
    Mutant(
        "handing the draft over for review counts as a promise",
        f"{PKG}/interview/checks.py",
        "            if _HAND_OVER.match(view[m.end() :]) or informing:",
        "            if informing:",
        "test_everything_else_is_not_a_promise",
    ),
    Mutant(
        "the corrective runs whatever the budget says",
        f"{PKG}/interview/turn.py",
        'if budget.allow("corrective", needs_s=CORRECTIVE_NEEDS_S):',
        "if True:",
        "test_a_spent_turn_budget_skips_the_corrective_and_the_fallbacks_still_apply",
    ),
    Mutant(
        "a promise that survives the correction is not removed",
        f"{PKG}/interview/turn.py",
        "        if promises:\n            promise_removed = True",
        "        if False:\n            promise_removed = True",
        "test_a_promise_that_survives_the_correction_is_removed_by_code",
    ),
    # --- review fix wave ---------------------------------------------------------------
    Mutant(
        "a refused relabel keeps the old copy",
        f"{PKG}/knowledge/ingest.py",
        'report.refused.append(f"{doc.doc_id}: {exc}")\n'
        "            report.removed += store.forget_document(doc.doc_id)",
        'report.refused.append(f"{doc.doc_id}: {exc}")',
        "test_a_refused_stricter_relabel_removes_the_old_looser_copy",
    ),
    Mutant(
        "ingest screens the raw text, not the normalised view",
        f"{PKG}/knowledge/ingest.py",
        "return bool(find_markers(normalize_text(text)))",
        "return bool(find_markers(text))",
        "test_planted_chunks_are_quarantined_whatever_their_spelling",
    ),
    Mutant(
        "the chat screens the raw text, not the normalised view",
        f"{PKG}/summarizer/chat.py",
        "(held if find_markers(normalize_text(msg.text)) else kept).append(msg)",
        "(held if find_markers(msg.text) else kept).append(msg)",
        "test_planted_chat_messages_are_held_out_whatever_their_spelling",
    ),
    Mutant(
        "stale chunks of a document are kept",
        f"{PKG}/knowledge/store.py",
        "removed = sum(self._delete_chunk(cid) for cid in stale)",
        "removed = 0",
        "test_chunks_that_no_longer_exist_are_removed",
    ),
    Mutant(
        "the question does not count towards a prompt's label",
        f"{PKG}/knowledge/retrieve.py",
        "return highest([QUESTION_LABEL, *(h.chunk.sensitivity for h in hits)])",
        "return highest(h.chunk.sensitivity for h in hits)",
        "test_the_question_itself_is_never_sent_to_a_public_only_model",
    ),
    Mutant(
        "a prompt's label comes from its first passage only",
        f"{PKG}/knowledge/retrieve.py",
        "return highest([QUESTION_LABEL, *(h.chunk.sensitivity for h in hits)])",
        "return highest([QUESTION_LABEL, *(h.chunk.sensitivity for h in hits[:1])])",
        "test_one_confidential_passage_among_internal_ones_keeps_the_answer_inside",
    ),
    Mutant(
        "vector scores assume unit-length vectors",
        f"{PKG}/knowledge/retrieve.py",
        "return dot / norm if norm else 0.0",
        "return dot",
        "test_relevance_does_not_depend_on_the_vectors_length",
    ),
    Mutant(
        "a classifier not cleared for user text is asked anyway",
        f"{PKG}/intent.py",
        "if classifier.placement.clearance < USER_TEXT:",
        "if False:",
        "test_a_classifier_not_cleared_for_user_text_is_never_asked",
    ),
    Mutant(
        "the office chat interviewer forgets the conversation",
        f"{PKG}/assistant.py",
        'Request(SURFACE, "interview_turn", session_id, user_id), list(history), routed.text',
        'Request(SURFACE, "interview_turn", session_id, user_id), [], routed.text',
        "test_the_office_chat_interviewer_remembers_the_conversation",
    ),
    Mutant(
        "an existing session can be created again",
        f"{PKG}/pipeline/tools.py",
        "if session_id in self.sessions:  # re-creating",
        "if False:  # re-creating",
        "test_the_store_refuses_to_recreate_an_existing_session",
    ),
    Mutant(
        "every sentence ending in a question mark is skipped",
        f"{PKG}/interview/checks.py",
        'if view.rstrip().endswith("?") and _INVERTED_QUESTION.match(view):',
        'if view.rstrip().endswith("?"):',
        "test_more_promise_phrasings_are_found",
    ),
    Mutant(
        "a composed summary need not come from the material",
        f"{PKG}/summarizer/role.py",
        "if cid not in material or m is None or m.channel != period.channel:",
        "if m is None or m.channel != period.channel:",
        "test_the_judge_needs_both_the_material_and_the_period",
    ),
    Mutant(
        "a composed summary may cite messages after the period",
        f"{PKG}/summarizer/role.py",
        "if not period.start <= chat.local_date(m) <= period.end:",
        "if not period.start <= chat.local_date(m):",
        "test_the_judge_needs_both_the_material_and_the_period",
    ),
    # --- fix wave 2 --------------------------------------------------------------------
    Mutant(
        "a narrowing relabel is not applied before the embed",
        f"{PKG}/knowledge/ingest.py",
        "                store.set_metadata(doc.doc_id, doc.title, doc.sensitivity, doc.groups)",
        "                pass",
        "test_a_failed_embed_on_a_narrowing_reingest_still_narrows",
    ),
    Mutant(
        "a widening re-ingest keeps the old copy until the embed",
        f"{PKG}/knowledge/ingest.py",
        "                # widening: the old text must never be readable under the new access\n"
        "                report.removed += store.forget_document(doc.doc_id)",
        "                # widening: the old text must never be readable under the new access\n"
        "                pass",
        "test_a_failed_embed_on_a_widening_reingest_leaves_nothing_readable",
    ),
    Mutant(
        "a marker split across chunks keeps the old copy",
        f"{PKG}/knowledge/ingest.py",
        '            report.quarantined.append(f"{doc.doc_id} (planted text spans chunks)")\n'
        "            report.removed += store.forget_document(doc.doc_id)",
        '            report.quarantined.append(f"{doc.doc_id} (planted text spans chunks)")',
        "test_a_marker_split_across_chunks_quarantines_the_whole_document",
    ),
    Mutant(
        "only chunks past the end are treated as stale",
        f"{PKG}/knowledge/store.py",
        "                if ord_ not in wanted",
        "                if ord_ > max(wanted, default=-1)",
        "test_a_middle_chunk_that_turns_planted_loses_its_old_row",
    ),
    Mutant(
        "forgetting a document leaves its groups",
        f"{PKG}/knowledge/store.py",
        '            self.db.execute("DELETE FROM doc_groups WHERE doc_id = ?", (doc_id,))\n'
        '            self.db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))',
        '            self.db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))',
        "test_forgetting_a_document_leaves_no_rows_behind",
    ),
    Mutant(
        "a negation anywhere before the promise cancels it",
        f"{PKG}/interview/checks.py",
        '    r"(?:\\s+(?:you|them|him|her|anyone))?(?:\\s+for\\s+sure)?'
        '(?:\\s+(?:that|if|whether))?\\s*$",',
        '    r"(?:\\s+(?:you|them|him|her|anyone))?(?:\\s+for\\s+sure)?'
        '(?:\\s+(?:that|if|whether))?",',
        "test_these_are_still_promises",
    ),
    Mutant(
        "'not sure' does not govern a promise",
        f"{PKG}/interview/checks.py",
        '    rf"|(?:am|{_APOS}m)\\s+not\\s+sure)"',
        '    rf"|(?:am|{_APOS}m)\\s+not\\s+sure_never)"',
        "test_governed_negations_and_real_questions_are_not_promises",
    ),
    # --- fix wave 3 --------------------------------------------------------------------
    Mutant(
        "a looser label with the same groups counts as narrowing",
        f"{PKG}/knowledge/ingest.py",
        "if doc.sensitivity >= old_label and set(doc.groups) <= old_groups:",
        "if set(doc.groups) <= old_groups:",
        "test_a_looser_label_with_the_same_groups_counts_as_widening",
    ),
    Mutant(
        "any overlap of groups counts as narrowing",
        f"{PKG}/knowledge/ingest.py",
        "if doc.sensitivity >= old_label and set(doc.groups) <= old_groups:",
        "if doc.sensitivity >= old_label and set(doc.groups) & old_groups:",
        "test_more_groups_with_the_same_label_counts_as_widening",
    ),
    Mutant(
        "a document listed twice is accepted",
        f"{PKG}/knowledge/ingest.py",
        "    if duplicates:",
        "    if False:",
        "test_a_document_listed_twice_in_one_batch_is_refused",
    ),
    Mutant(
        "a marker cut across chunks is not seen",
        f"{PKG}/knowledge/ingest.py",
        '        if _planted(" ".join(c.text for c in kept)) or _planted(_as_read(kept)):',
        "        if _planted(_as_read(kept)) and False:",
        "test_a_marker_split_across_chunks_quarantines_the_whole_document",
    ),
    Mutant(
        "a marker cut across headings is not seen",
        f"{PKG}/knowledge/ingest.py",
        '        if _planted(" ".join(c.text for c in kept)) or _planted(_as_read(kept)):',
        '        if _planted(" ".join(c.text for c in kept)):',
        "test_a_marker_cut_across_heading_levels_is_caught",
    ),
    Mutant(
        "a heading path is screened with its separators",
        f"{PKG}/knowledge/ingest.py",
        "    return f\"{heading.replace(' > ', ' ')} {text}\"",
        '    return f"{heading} {text}"',
        "test_a_heading_path_marker_quarantines_only_its_chunk",
    ),
    Mutant(
        "two kinds of period are guessed instead of asked",
        f"{PKG}/summarizer/periods.py",
        "        if len(found) > 1:",
        "        if False:",
        "test_two_kinds_of_period_are_asked_about_not_guessed",
    ),
    Mutant(
        "'up to today' counts as a second period",
        f"{PKG}/summarizer/periods.py",
        '        plain = _CLOSING_TODAY.sub(" ", text)',
        "        plain = text",
        "test_today_that_only_closes_the_period_is_not_a_second_period",
    ),
    Mutant(
        "present-tense informing counts as a promise",
        f"{PKG}/interview/checks.py",
        "            informing = _INFORMING.match(view[m.start() :])",
        "            informing = None",
        "test_governed_negations_and_real_questions_are_not_promises",
    ),
    Mutant(
        "a discourse word hides a real question",
        f"{PKG}/interview/checks.py",
        '    r"^\\s*(?:(?:so|okay|ok|and|then|well|now)\\s*,?\\s+){0,2}"',
        '    r"^\\s*"',
        "test_governed_negations_and_real_questions_are_not_promises",
    ),
    # --- fix wave 4 --------------------------------------------------------------------
    Mutant(
        "'share that' excuses a promise whatever follows",
        f"{PKG}/interview/checks.py",
        '    r"(?:the|this|it|we|i|you|our|your|there|everything)\\b"',
        '    r"\\w+\\b"',
        "test_these_are_still_promises",
    ),
    Mutant(
        "'call this' excuses a promise whatever follows",
        f"{PKG}/interview/checks.py",
        '    r"|call\\s+(?:this|that|it)\\s+(?:the|a|an|step|stage|part|phase)\\b)",',
        '    r"|call\\s+(?:this|that|it)\\b)",',
        "test_these_are_still_promises",
    ),
    Mutant(
        "only the first period of each kind is seen",
        f"{PKG}/summarizer/periods.py",
        "                for m in pattern.finditer(plain)",
        "                for m in filter(None, [pattern.search(plain)])",
        "test_two_kinds_of_period_are_asked_about_not_guessed",
    ),
    Mutant(
        "'until today' does not end the period today",
        f"{PKG}/summarizer/periods.py",
        "            end = today\n",
        "            pass\n",
        "test_until_today_ends_the_period_today",
    ),
    Mutant(
        "the modal 'may' after a subject is read as May",
        f"{PKG}/summarizer/periods.py",
        "(?<!\\bi\\s)",
        "",
        "test_the_modal_may_after_a_subject_is_not_the_month",
    ),
    # --- fix wave 5 --------------------------------------------------------------------
    Mutant(
        "'from <date> until today' loses its start date",
        f"{PKG}/summarizer/periods.py",
        "                end = today if last_day.lower() == "
        '"today" else date.fromisoformat(last_day)',
        "                end = date.fromisoformat(last_day)",
        "test_review_round_period_phrasings",
    ),
    Mutant(
        "'since <month>' covers only that month",
        f"{PKG}/summarizer/periods.py",
        "        closed_by_today = plain != text or _SINCE.search(text) is not None",
        "        closed_by_today = plain != text",
        "test_review_round_period_phrasings",
    ),
    Mutant(
        "a deadline 'today' counts as a period",
        f"{PKG}/summarizer/periods.py",
        '        text = _DEADLINE_TODAY.sub(" ", _CHANNEL.sub(" ", text))',
        '        text = _CHANNEL.sub(" ", text)',
        "test_review_round_period_phrasings",
    ),
    Mutant(
        "a channel name is read as a month",
        f"{PKG}/summarizer/periods.py",
        '        text = _DEADLINE_TODAY.sub(" ", _CHANNEL.sub(" ", text))',
        '        text = _DEADLINE_TODAY.sub(" ", text)',
        "test_review_round_period_phrasings",
    ),
    Mutant(
        "an informing sentence with a future tail is excused",
        f"{PKG}/interview/checks.py",
        "            if informing and _FUTURE_TAIL.search(view[m.start() :]):",
        "            if False:",
        "test_these_are_still_promises",
    ),
    Mutant(
        "an open role the surface does not offer is reachable",
        f"{PKG}/intent.py",
        "    if text.strip() and continuing in roles and (not matched or continuing in matched):",
        "    if text.strip() and continuing and (not matched or continuing in matched):",
        "test_an_open_role_the_surface_does_not_offer_is_never_reached",
    ),
    Mutant(
        "the open role wins over rules that do not name it",
        f"{PKG}/intent.py",
        "    if text.strip() and continuing in roles and (not matched or continuing in matched):",
        "    if text.strip() and continuing in roles:",
        "test_rules_that_disagree_without_the_open_role_still_go_to_the_classifier",
    ),
    Mutant(
        "a follow-up never continues the open interview",
        f"{PKG}/assistant.py",
        'continuing="interviewer" if open_interview else None',
        "continuing=None",
        "test_a_follow_up_continues_the_interview_with_its_history",
    ),
    Mutant(
        "any user's interview in any session counts as open",
        f"{PKG}/assistant.py",
        "        open_interview = bool(self._history.get((session_id, user_id)))",
        "        open_interview = bool(self._history)",
        "test_another_user_or_another_session_does_not_continue_the_interview",
    ),
    Mutant(
        "the offline answerer ignores headings",
        f"{PKG}/offline_roles.py",
        '                coverage(question, f"{heading} {sentence}"),',
        "                coverage(question, sentence),",
        "test_hr_gets_the_salary_bands_from_the_bands_section_not_the_reviews_one",
    ),
    Mutant(
        "the offline answerer ignores which section's own heading matches",
        f"{PKG}/offline_roles.py",
        "                coverage(question, section),",
        "                0.0,",
        "test_hr_gets_the_salary_bands_from_the_bands_section_not_the_reviews_one",
    ),
    Mutant(
        "the offline answerer breaks heading ties by order, not by the sentence",
        f"{PKG}/offline_roles.py",
        "                coverage(question, sentence),\n            )",
        "                0.0,\n            )",
        "test_a_heading_match_alone_goes_to_the_sentence_that_matches_by_itself",
    ),
    Mutant(
        "empty text becomes an interview turn",
        f"{PKG}/intent.py",
        "    if text.strip() and continuing in roles and (not matched or continuing in matched):",
        "    if continuing in roles and (not matched or continuing in matched):",
        "test_empty_text_never_becomes_an_interview_turn",
    ),
    Mutant(
        "/done does not close the interview",
        f"{PKG}/assistant.py",
        "            self._history.pop((session_id, user_id), None)",
        "            pass",
        "test_done_closes_the_open_interview_and_follow_ups_are_routed_afresh",
    ),
    Mutant(
        "a failed first turn opens the interview",
        f"{PKG}/assistant.py",
        "        open_interview = bool(self._history.get((session_id, user_id)))",
        "        open_interview = (session_id, user_id) in self._history",
        "test_a_failed_first_turn_does_not_open_the_interview",
    ),
)

_FAILED = re.compile(r"^FAILED (\S+)", re.MULTILINE)


def _copy_project(dest: Path) -> None:
    for name in ("src", "tests"):
        shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")


def _pytest(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "project"
        _copy_project(work)
        code, out = _pytest(work)
        if code != 0:
            print("baseline suite is not green in the copy; aborting\n" + out[-2000:])
            return 2
        print(f"baseline green; checking {len(MUTANTS)} mutants")

        survivors = 0
        for m in MUTANTS:
            target = work / m.path
            original = target.read_text(encoding="utf-8")
            if original.count(m.old) != 1:
                print(f"STALE   {m.name}: pattern found {original.count(m.old)} times")
                survivors += 1
                continue
            target.write_text(original.replace(m.old, m.new), encoding="utf-8")
            try:
                code, out = _pytest(work)
            finally:
                target.write_text(original, encoding="utf-8")
            failed = _FAILED.findall(out)
            # exact test name match ("file::name" or "file::name[param]")
            names = {f.split("::")[-1].split("[")[0] for f in failed}
            killed_by_killer = m.killer in names
            if code != 0 and killed_by_killer:
                print(f"killed  {m.name}  ({len(failed)} failing tests)")
            else:
                survivors += 1
                print(
                    f"SURVIVED {m.name}: exit={code}, killer {m.killer!r} failed={killed_by_killer}"
                )
        print(f"{len(MUTANTS) - survivors}/{len(MUTANTS)} mutants killed")
        return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
