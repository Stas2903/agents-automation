"""The save-claim detector's contract. Every row is synthetic, written for this repository.

FIRES: the interviewer asserts that the draft (or "it", "everything" ...) was persisted.
STAYS_SILENT: negations, intent, conditionals, questions, quoted text, reported speech, the
employee describing their own systems, and other senses of "save".
"""

from __future__ import annotations

import pytest

from agents_automation.interview.honesty import (
    HONEST_NOT_SAVED,
    HONEST_SAVE_FAILED,
    detect_save_claims,
    remove_claims,
)

FIRES = [
    "I've saved the draft.",
    "I have saved your process.",
    "I saved it.",
    "Great, I've just recorded the new approval step.",
    "We've stored everything so far.",
    "Your draft is saved.",
    "The draft has been saved.",
    "It's saved.",
    "It’s saved.",
    "Everything is recorded now.",
    "All of this has been saved.",
    "That's been saved too.",
    "Saved!",
    "Done, saved.",
    "All recorded.",
    "Got it — saved.",
    "I've added the manager approval to your draft.",
    "I have updated the step list.",
    "I've noted that down in my notes.",
    "Your changes have been saved.",
    "The steps are now stored safely.",
    "Thanks! I saved your answers, so we can continue later.",
    "I checked the receipt step and saved it.",
    "Okay, the workflow has been updated.",
    "When you mentioned the second approval, I added it to the draft.",
    "I've logged the finance step.",
    "This is saved in your draft.",
    "I've recorded it in the ERP.",
    "That is a good point. I have already saved it.",
    # a claim followed by the turn's one question
    "I've saved the draft, who approves it next?",
    "Thanks! Your draft has been saved; anything else?",
    # short and bullet forms
    "Draft saved.",
    "Saved the draft with all six steps.",
    "- Saved the draft with all steps.",
    "Saved it to your draft.",
    # adverbs and modified subjects
    "I have successfully saved the draft.",
    "I've safely stored your draft.",
    "Your workflow draft is saved.",
    "The updated draft has been saved.",
    "All six steps are saved.",
    "Everything you told me is saved.",
    "I've got everything saved.",
    # a negation or "as you said" that does not scope over the claim
    "No problem I've saved it.",
    "As you said I've saved it.",
    # an opening phrase is not intent ("can confirm" asserts)
    "I can confirm I've saved the draft.",
    "Good news: I can confirm your draft is saved.",
    "You'll be glad to hear I've saved it.",
    "I'd like you to know I've saved the draft.",
    "You can relax I've saved everything.",
    # the system's own storage is still a claim
    "Everything is saved for you.",
    "It's all saved for you now.",
    "Your draft is saved in the system.",
    "Your draft is saved to your account.",
    "It's saved under your name.",
    "It's stored safely in our database.",
    "Your draft is saved on the server.",
    "Your draft is stored in the cloud.",
    "Your draft is saved and then it goes to review.",
    "I've saved no fewer than six steps.",
    "I've saved no less than six steps.",
    "Your draft is saved on the platform.",
    "Everything is stored in our records.",
    # plurals, adjectives and "this" in front of the system's own storage
    "Your draft is saved on our servers.",
    "It's stored on our servers.",
    "Your draft is saved in our databases.",
    "Your draft is saved in our systems.",
    "Your draft is saved in the internal system.",
    "Your draft is saved in our own database.",
    "It is saved in the secure database.",
    "Your draft is saved in this system.",
    # purposes and places on the interviewer's side
    "Your draft is saved for later.",
    "It's saved for later.",
    "Your answers are saved for next time.",
    "Your draft is saved for review.",
    "It's saved on our side.",
    "Your draft is saved on our end.",
    "Your draft is saved with us.",
    "It is saved in this chat.",
    # "and then" after the conversation's own artifact is not a process step
    "Your answers are saved and then I will summarise them.",
    "Your changes are saved and then we go on.",
    "Everything you told me is saved and then I'll check it.",
    "All six steps are saved and then we can finish.",
    # "earlier than" compares; it does not point at an earlier save
    "I saved it earlier than planned.",
    "I saved it earlier than you asked.",
    # "before" is a conjunction here, not a reference to an earlier save (rows hold even
    # though the default assumes an earlier draft may exist)
    "I've saved the new steps before you go.",
    "I've saved everything before we wrap up.",
    "I've saved the approval step before you asked.",
    "Your changes were saved before the timeout.",
]

STAYS_SILENT = [
    # negation
    "I haven't saved the draft yet.",
    "I have not saved anything.",
    "Nothing has been saved so far.",
    "It isn't saved yet.",
    "The draft was not saved.",
    "Not everything is recorded yet.",
    "None of it is saved.",
    "I don't think it's saved yet.",
    "Not saved.",
    # intent / future / modal
    "I'll save the draft once you confirm.",
    "I will save it when we finish.",
    "I'm going to record this step.",
    "Let me save that for you.",
    "I can record it as a decision step.",
    "It would be saved automatically in a real system.",
    "Saving it now.",
    "We need to record who approves the expense.",
    # conditional / temporal subordinate clause
    "If you confirm, it's saved.",
    "Once you confirm, I'll save it.",
    "Before it's saved, please check the amounts.",
    "Until it is saved, nothing is final.",
    # questions
    "Shall I save the draft?",
    "Has it been saved?",
    "Do you want me to save it now?",
    "What happens after the expense is recorded?",
    "You've saved the receipt in the folder, right?",
    # reported speech and quotes
    "You said the receipt is saved in the shared folder.",
    "You mentioned it's recorded by finance.",
    "The manager said everything is stored in the archive.",
    'The button in your tool says "Saved".',
    "Your message said 'I saved the receipt'.",
    # the employee's own systems and third parties
    "So the invoice is recorded in the accounting system.",
    "Then it's recorded in the ERP by the clerk.",
    "And after that it is stored in the archive for seven years.",
    "The steps are recorded by the finance team.",
    "The finance clerk has stored the receipts in the archive.",
    "You saved the receipt in the shared folder.",
    "I asked whether it was saved by your team.",
    # other senses of "save"
    "This saves you a lot of time.",
    "That step saved you two days.",
    "I've saved you some time by grouping these steps.",
    # each guard has at least one row that only it suppresses
    "The clerk mentioned we stored the receipts.",  # reported speech
    "After it is saved, finance pays the expense.",  # subordinate clause
    "When everything is recorded, I'll send it on.",
    "So it's saved?",  # question
    "And the draft is saved, right?",  # tag question
    'The screen shows "Everything is saved".',  # quoted text
    "I haven't checked it and saved it yet.",  # negation inside the match
    "I will make sure it is saved.",  # intent
    "I need to confirm everything is saved.",
    # earlier saves cannot be checked against this turn's ledger
    "The draft we saved earlier has three steps. Who pays the expense?",
    # the employee's own systems, paraphrased back
    "So it's stored for seven years.",
    "So that's logged automatically by the system.",
    "Then it's recorded automatically.",
    "Sounds like everything is stored centrally.",
    "That's saved me a lot of time.",
    "Right, so it is logged and then checked by finance.",
    "So we stored nothing digitally back then.",
    "I'll make sure it's saved.",
    "So it's recorded in the accounting system.",
    "So it is stored in the finance database.",
    "Then it is stored until the end of the month.",
    # a reference to an earlier turn, followed by a new clause
    "The draft we saved earlier you said has three steps.",
    "The draft we saved earlier we can now extend.",
    "The draft I saved earlier I think is fine.",
    # a reference to an earlier turn (only when a draft exists, see the test below)
    "The draft I saved for you last turn has three steps.",
    # the code-written honest notes must never trigger the guard themselves
    HONEST_NOT_SAVED,
    HONEST_SAVE_FAILED,
]


# Accepted false positives. The guard errs towards firing: a false positive costs one
# corrective model call, while a false negative lets an untrue "I saved it" reach the
# employee. Pinned here so a change in either direction is a deliberate decision.
KNOWN_FALSE_POSITIVES = [
    # "before" is left out of the earlier-save words because it is usually a conjunction
    "The draft I saved before has three steps.",
    # "the system" / "the database" / "review" are the interviewer's storage words, even
    # when the employee's own process is being paraphrased
    "So it's recorded in the system by finance.",
    "Then it is stored in the database of the accounting team.",
    "So it is logged for review by finance.",
    # "draft" in the subject keeps "...and then ..." a claim
    "So the purchase draft is saved and then approved by your manager.",
]


@pytest.mark.parametrize("text", KNOWN_FALSE_POSITIVES)
def test_known_false_positives_still_fire(text: str) -> None:
    assert len(detect_save_claims(text)) == 1


@pytest.mark.parametrize("text", FIRES)
def test_flags_the_claim(text: str) -> None:
    claims = detect_save_claims(text)
    assert len(claims) == 1, f"expected exactly one claim in {text!r}, got {claims}"


@pytest.mark.parametrize("text", STAYS_SILENT)
def test_lets_it_pass(text: str) -> None:
    assert detect_save_claims(text) == [], f"unexpected claim in {text!r}"


def test_tables_do_not_overlap() -> None:
    assert not set(FIRES) & set(STAYS_SILENT)
    assert not set(KNOWN_FALSE_POSITIVES) & (set(FIRES) | set(STAYS_SILENT))


def test_claim_carries_sentence_and_phrase() -> None:
    [claim] = detect_save_claims("Thanks for that. I've saved the draft. What happens next?")
    assert claim.sentence == "I've saved the draft."
    assert claim.phrase == "I've saved"


def test_one_claim_per_sentence_across_a_reply() -> None:
    reply = "Saved! The draft has been saved. Who signs off the payment?"
    assert [c.sentence for c in detect_save_claims(reply)] == [
        "Saved!",
        "The draft has been saved.",
    ]


def test_negation_only_suppresses_its_own_clause() -> None:
    # "not" in an earlier clause does not cancel a later, separate claim.
    assert len(detect_save_claims("That is not a problem, I've saved it.")) == 1


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "?", "..."])
def test_empty_and_degenerate_input(text: str) -> None:
    assert detect_save_claims(text) == []


def test_remove_claims_keeps_the_other_sentences() -> None:
    reply = "Thanks, that helps. I've saved the draft. Who pays the expense?"
    claims = detect_save_claims(reply)
    assert remove_claims(reply, claims) == "Thanks, that helps. Who pays the expense?"


@pytest.mark.parametrize(
    "text",
    [
        "Your draft was saved previously, so you can go.",
        "The draft we saved earlier has three steps.",
        "The draft I saved for you last turn has three steps.",
    ],
)
def test_earlier_save_references_count_as_claims_when_no_draft_exists(text: str) -> None:
    assert detect_save_claims(text) == []  # a draft from an earlier turn may exist
    assert len(detect_save_claims(text, earlier_saves_possible=False)) == 1


def test_claim_sentences_keep_the_original_text() -> None:
    name = chr(0x0421) + "ergei"  # starts with a Cyrillic letter that looks like a Latin C
    reply = f"Thanks, {name}. I've saved the draft. Who pays?"
    [claim] = detect_save_claims(reply)
    assert claim.sentence == "I've saved the draft."
    assert remove_claims(reply, [claim]) == f"Thanks, {name}. Who pays?"
