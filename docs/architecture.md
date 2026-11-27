# Architecture

## Zones

| Zone | Modules | May read untrusted text? |
|---|---|---|
| Trusted | `pipeline/router.py`, `pipeline/planner.py`, `pipeline/seal.py` (hashing), `pipeline/policy.py`, `directory.py` | No. `make_plan(role, request)` has no parameter for it; user ids and groups come from the envelope and the directory. |
| Routing | `intent.py` | Yes, but its answer is limited to the roles the trusted surface offers, or a clarifying question. |
| Untrusted | `pipeline/executor.py`, `pipeline/tools.py`, the three roles (`interview/`, `knowledge/`, `summarizer/`) after sealing, `finalize.py` | Yes, but it can only fill slot values of the sealed plan. |
| Model seam | `llm/seam.py` (`ChatModel`, `Embedder`), `llm/boundary.py`, `llm/fakes.py`, `llm/*_live.py` | It is the only way to reach a model. Knowledge and summarizer calls go through `ModelRouter.complete`, which takes the label of the data they carry. |

Six import-linter contracts in `pyproject.toml` keep these boundaries:
- `llm` imports nothing from the project;
- `pipeline` never imports a role, the intent router (`intent.py`) or the conversation layer;
- `injection` depends on no role;
- nothing in the product imports `evals`;
- the three roles never import each other;
- no module imports a provider SDK statically.

## Routing a message (`intent.py`, `pipeline/router.py`)

1. The surface (from the trusted envelope) fixes the candidate roles: `interview_chat` has
   only the interviewer, `review_console` only the reviewer, and `office_chat` has knowledge,
   summarizer and interviewer.
2. A single-role surface needs no choice. Otherwise an explicit command (`/ask`,
   `/summarize`, `/describe`) picks its role, and keyword rules (run on the normalised view)
   decide the clear cases.
3. A follow-up in an open interview stays with the interviewer. When the caller already has
   interview turns in this session (known on the trusted side, never from the text), a
   message that no rule claims for another role, such as "Then my manager approves it", is
   the next interview turn without a classifier call. A command or another role's rule still
   leaves the interview ("Do I need a receipt for that?" goes to knowledge), empty text never
   continues it, and an open role the surface does not offer is never used. `/done` closes the
   interview, so the next message is routed afresh.
4. Still undecided (no rule, or rules that disagree): a classifier model is asked. It must be
   placed with a clearance for internal data (the message is user text), and its JSON answer
   must name exactly one role the surface offers. The clarify reason says why it was not
   used, for example the role it named.
5. Anything else (no classifier, an outage, an unknown role, malformed output) becomes a
   clarifying question. `route(surface, chosen)` then refuses any role the surface does not
   offer, so even a buggy caller cannot reach one.

## A knowledge question (`knowledge/`)

1. The plan `(knowledge, ask)` is one read step, `search_docs`. `viewer_id` comes from the
   envelope and the question is bound as the open `query` value.
2. The user's groups come from the directory, never from the text. Both searches run under
   `WHERE doc_id IN (SELECT doc_id FROM doc_groups WHERE grp IN (...))`: FTS5 BM25, with every
   term quoted so query syntax in user text is never interpreted, and a scan of stored vectors
   scored by cosine. Turning chunk ids back into text checks the same scope again. A document
   without groups is visible to nobody, and an unknown user has no groups.
3. The two rankings are fused with reciprocal rank fusion (`k = 60`, ties broken by id).
4. The reranker is optional and never fatal. Without one, nothing happens. It is skipped, and
   the skip recorded, when it is not cleared for the prompt's label (the passages and the
   question), out of time budget, unreachable, or when it answers with anything other than an
   ordering of the given candidates.
5. Grounded-answer gate:
   - no relevant passage (a keyword match, or a vector cosine of 0.3 or more) means "I couldn't
     find this..." without any model call;
   - otherwise one answer call goes through the data-boundary router with only the relevant
     passages and the question, fenced. Its label is the highest of the passages' labels and
     INTERNAL, because the question is user text;
   - the answer must cite at least one id, and only ids of passages it was given; anything
     else is rejected. The model may decline.
6. Ingest is idempotent and fails closed:
   - a batch that lists a document twice is refused before anything is changed;
   - each chunk stores its text hash, embedding-model id and chunker version, and an unchanged
     chunk is not embedded again;
   - a document whose label the embedder is not cleared for is refused: nothing of it is sent,
     and anything indexed for it before is removed, so a stricter relabel never leaves the
     old, looser copy searchable;
   - before any embed call, a first pass narrows stored access: a narrowing relabel (stricter
     label, a subset of the groups) is applied to the stored copy, and a document whose access
     widens or moves to other groups loses its stored copy (groups are not hierarchical);
   - then each document is embedded and written in one transaction (label, groups, chunks), so
     a failure anywhere in a batch never leaves a copy readable by more people than allowed;
   - a chunk whose heading and text match the Layer 1 patterns on the normalised view is
     quarantined, and if a match only appears when the chunks are joined (a marker cut across
     chunks or across a heading), the whole document is. Layers 2 and 3 do not run on
     documents.

## A summary request (`summarizer/`)

1. The plan `(summarizer, summarize)` is one read step, `read_chat`. The request text is
   parsed into three plain values (channel, start, end) and bound as slot values. It
   understands explicit ranges (which always win), "last N days/weeks", month names (the most
   recent one when no year is given; the verb in "May I ...?" or "I may need it" is not read
   as a month), and "this/last week", "this/last month", "yesterday" and "today". When a
   request names more than one period ("yesterday's thread about the May release", "January
   and February"), it asks instead of guessing. "... up to / until today" and "since ..." are
   not a second period: they end the named period today ("from 2026-03-01 until today",
   "since February"). A deadline ("I need it today") and channel names ("#may-launch") are
   not read as periods.
   Without a channel it reads `#general`; without a period it recognises, the last 7 days.
   The end is clipped to today, and a period in the future, a reversed one, one of zero days
   or one longer than 366 days is refused.
2. The read step checks membership with the user's groups from the directory. An unknown
   channel and a forbidden one get the same answer, so channels cannot be probed.
3. Every message belongs to its day in the team's timezone. Messages that match the Layer 1
   patterns on the normalised view are held out of everything.
4. The write side wrote daily, weekly and monthly summaries ahead of time, only for complete
   periods up to the "written through" date. Each summary cites the ids it is based on, and
   a summary citing anything outside its period was not stored.
5. The read side places whole months first (for spans of 56 days or more), then whole
   Monday-to-Sunday weeks in the gaps (14 days or more), then daily summaries, then raw
   messages for the days that no summary covers.
6. One compose call goes through the data-boundary router. A citation judge accepts the
   result only if every cited id is in the material and is a real message of that channel
   inside the period. When the composer is out of budget, not cleared, unreachable or
   rejected, the reply is a digest written by code from the same coverage, and the reason
   is reported.

## One interview turn (`interview/turn.py`)

1. `route(surface, "interviewer")`, then `make_plan(role, request)` picks a fixed template.
   For an interview turn that is `fetch_draft` (read) and `save_draft` (write). The slots
   `session_id` and `viewer_id` come from the request; `content` stays open.
2. `seal(plan)` hashes role, step ids, tools, operations and slot **keys**, not values.
3. `authorize(sealed)` checks every step against the `(role, tool, operation)` allow-set.
   An empty plan is denied. The turn is refused before any model call if this fails.
4. The executor runs only the `read` step. Tools are clamped to the record's access scope.
   The stored draft is untrusted from here on.
5. `compose_prompt` builds trusted instructions, then the sealed plan's open actions, then
   fenced untrusted content. Every prompt that carries untrusted text (the turn, the repair
   request, the injection judge, and the knowledge and summarizer calls) uses one helper,
   `llm.text.fence_untrusted`. It neutralises every fence-tag opener in the original text,
   whatever its case, spacing, attributes, look-alike letters or punctuation, or invisible or
   blank characters. The replacement also takes the rest of the tag up to a `>` on the same
   line within 200 characters; text on other lines is never touched.
6. **One** model call returns `{"reply": ..., "actions": [...]}`. A parse failure produces a
   safe reply and runs nothing. An unreachable model produces a safe reply marked
   `transport_failed`, and the harness counts that run as VOID.
7. Each action must match a write step of the sealed plan. `bind_values` refuses unknown
   steps, undeclared keys, non-string values and overwriting filled (trusted) slots. The
   executor re-checks authorization for that one step. Every outcome goes into the ledger.
8. The reply checks (`interview/checks.py`) run in priority order:
   1. honesty: a save claim the ledger does not support;
   2. promise: "I'll email you ...", "I'm going to remind you ...", "I'll be sending ...",
      also with a tag ("..., OK?"); a real question ("Shall I ...?") and handing the draft
      over for review are allowed, and a negation only counts when it governs the promise
      ("I can't promise I'll ...");
   3. repetition: the same question as in the previous message.

   Only the first one that fires asks for a correction, as one user-role note with a fixed
   prefix, so a turn costs at most one extra call. That call is optional: when the turn
   budget does not leave enough time it is skipped, and the skip is reported. The code
   fallbacks then run in every case. A remaining promise is removed and replaced by "I can't
   send, remind or schedule anything myself". A remaining unsupported claim is removed and
   replaced by "I have not saved ..." or "... the save did not go through". A repeated
   question is only reported.

On the office chat the assistant keeps each user's interview history per session (another
user reusing the session id gets neither the history nor the draft, and the store refuses to
re-create an existing session). That history is what keeps follow-ups with the interviewer
(see routing, step 3). Finalize runs on the interview surface.

`agents-demo --chat --user u-hr` runs this surface on typed messages, on the same offline
stand-ins as the demo. It has no classifier, so a message that no command, rule or open
interview decides gets the clarifying question, and its interviewer stand-in only asks for
the next step.

## Finalize (`finalize.py`)

1. Same trusted-zone steps with the `finalize` template (`read`, `save`, `submit`).
2. Read the saved draft. No draft means `invalid`.
3. Validate the step list:
   - one start and at least one finish;
   - no duplicate or dangling `next` entries, no self loops, no dead ends;
   - every step reachable from the start, and every step able to reach a finish;
   - decisions with 2+ branches.

   On issues there are at most 2 repair rounds through the model, each result validated
   again. Still invalid means `invalid`, and nothing is submitted. If the repair model is
   unreachable, the result is `invalid` with `transport_failed` set.
4. Screen the list **that will ship** (after repair). Layer 1 and Layer 2 match on a
   normalised view: NFKC, invisible characters, combining marks and blank fillers removed, and
   look-alikes from Cyrillic, Greek, Armenian, Cherokee and Latin small capitals folded. Hits
   map back to the original text, which the judge reads and the reviewer sees:
   - Layer 1 hit: `held_for_review`, no judge call, hit spans masked.
   - No Layer 2 vocabulary: `delivered`, no judge call.
   - Judge says planted: `held_for_review`, and the judge's claims (only those that really
     occur in the text) are masked.
   - Judge says clean: `delivered`.
   - Judge unreachable, times out, or answers malformed or inconsistent JSON:
     `delivered_flagged`.
   - No judge configured: `delivered_flagged` (layer `no_judge`).
5. Save a repaired list, then submit with the status **and the revision that was screened**.
   - The store refuses a stale revision, and refuses to re-submit a revision that is already
     held (only a reviewer can release it).
   - Held records go to a `security_review` queue, delivered ones to `approval`.
   - Code chooses the employee's message from the status; a failed save or submit is
     reported as "nothing has been submitted".
   - `status` is the content verdict and `submitted` says whether it actually reached a
     queue; the harness's status checks require both.

## The store (`pipeline/tools.py`)

- Authors (the interviewed employees) read, save and submit; reviewers read and approve.
- A reviewer who is also an author cannot approve: no one approves their own work.
- Saving a new draft voids any earlier submission and approval, and an approval is refused
  unless the submitted revision is the current one, so screened and approved content cannot
  diverge. Every (re-)submission needs a fresh approval.
- A hold is sticky for the session:
  - saving again keeps it;
  - finalize keeps the status at `held_for_review` whatever a new screening says, and the
    store refuses any other status;
  - every screening outcome is appended to a log for the security reviewer.

  Releasing a hold is the security reviewer's job and is out of scope here.

## Eval harness (`evals/`)

```text
for each interview scenario, for run in 1..N (N >= 2):
    fresh store, fresh models (offline: scripts; live: real models)
    run_session(sim user <-> real interview loop) -> finalize
    any model outage (interviewer, repair, simulated user)      -> VOID
    deterministic checks (independent of the product guards)  -> FAIL on the first failure
    rubric judge (different model family; 1 retry; then VOID)  -> PASS / FAIL / VOID
for each role scenario, for run in 1..N:
    build the synthetic office with the configured system, run one situation
    -> PASS / FAIL (VOID if a model is unreachable)
grade_runs -> PASS / FLAKY / FAIL / VOID
gate: every hard-gate scenario must be PASS (FLAKY fails it; VOID makes it INCOMPLETE;
      a selection with no hard-gate scenario is INCOMPLETE, never PASS)
```

The cross-family check runs once, before any run; a same-family judge is a configuration
error (exit code 3). `--mutant NAME` swaps in one disabled guard and succeeds only if the
gate fails. There are eight mutants: claim detection, the injection screen, validation, access
control, the data boundary, grounded answers, summary citations and promises. A test pins the
exact scenarios each mutant must fail, across interview and role scenarios.
