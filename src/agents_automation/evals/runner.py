"""``agents-eval``: run the scenarios N times each and print the verdict matrix.

Exit codes
  without --mutant: 0 = gate PASS, 1 = gate FAIL, 2 = gate INCOMPLETE
  with --mutant:    0 = mutant caught (gate FAIL), 1 = it slipped through (gate PASS),
                    2 = gate INCOMPLETE (cannot tell)
  3 = configuration error (e.g. a judge from the interviewer's model family)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agents_automation.console import quiet_package_logs
from agents_automation.conversation import SessionTranscript, run_session, scripted_user
from agents_automation.evals.judge import (
    SameFamilyJudgeError,
    judge_run,
    offline_judge,
    require_cross_family,
)
from agents_automation.evals.mutants import MUTANTS, SYSTEM, SystemConfig
from agents_automation.evals.personas import PERSONAS, live_sim_user
from agents_automation.evals.role_scenarios import ROLE_SCENARIOS, RoleScenario
from agents_automation.evals.runmatrix import (
    MIN_RUNS,
    GateResult,
    GateStatus,
    RunOutcome,
    Verdict,
    evaluate_gate,
    grade_runs,
)
from agents_automation.evals.scenarios import SCENARIOS, Scenario
from agents_automation.interview.turn import Interviewer
from agents_automation.interview.validator import model_repairer
from agents_automation.llm import ChatModel, TransportError
from agents_automation.offline import Conversation, load_conversations
from agents_automation.pipeline.tools import InMemoryStore

EXIT_CONFIG_ERROR = 3


@dataclass(frozen=True)
class RunRecord:
    verdict: RunOutcome
    reason: str


@dataclass(frozen=True)
class ScenarioReport:
    scenario: Scenario | RoleScenario
    runs: tuple[RunRecord, ...]
    verdict: Verdict


@dataclass(frozen=True)
class Models:
    """Factories, so every run gets fresh models (fresh scripts offline)."""

    interviewer: Callable[[Conversation, int], ChatModel]
    planted_text_judge: Callable[[Conversation], ChatModel]
    rubric_judge: ChatModel
    sim_user: ChatModel | None  # None: replay the scripted user turns


def offline_models() -> Models:
    return Models(
        interviewer=lambda c, i: c.interviewer(i),
        planted_text_judge=lambda c: c.injection_judge_model(),
        rubric_judge=offline_judge(),
        sim_user=None,
    )


def live_models() -> Models:  # pragma: no cover - needs real keys and network
    from agents_automation.llm.anthropic_live import AnthropicChat
    from agents_automation.llm.openai_live import OpenAICompatibleChat

    return Models(
        interviewer=lambda _c, _i: AnthropicChat(),
        planted_text_judge=lambda _c: AnthropicChat(),
        rubric_judge=OpenAICompatibleChat(),
        sim_user=AnthropicChat(),
    )


def _infrastructure_failure(transcript: SessionTranscript) -> str | None:
    for i, ex in enumerate(transcript.exchanges, 1):
        if ex.result.transport_failed:
            return f"interviewer model unreachable on turn {i}"
    if transcript.final.transport_failed:
        return "repair model unreachable at finalize"
    return None


def run_once(
    scenario: Scenario,
    conversation: Conversation,
    run_index: int,
    models: Models,
    system: SystemConfig = SYSTEM,
) -> RunRecord:
    interviewer_model = models.interviewer(conversation, run_index)
    if models.sim_user is None:
        next_user = scripted_user(conversation.employee_lines)
    else:
        next_user = live_sim_user(
            models.sim_user,
            PERSONAS[conversation.persona],
            scenario.live_brief,
            max_turns=len(conversation.employee_lines) + 2,
        )
    try:
        transcript = run_session(
            session_id=f"eval-{scenario.id}-{run_index}",
            user_id="eval-user",
            next_user_message=next_user,
            interviewer=Interviewer(interviewer_model, InMemoryStore(), detector=system.detector),
            repairer=model_repairer(interviewer_model),
            judge_model=models.planted_text_judge(conversation),
            outage_turns=conversation.outage_turns,
            screener=system.screener,
            validator=system.validator,
        )
    except TransportError as exc:  # only the simulated user can still raise here
        return RunRecord(RunOutcome.VOID, f"simulated user unreachable: {exc}")

    infrastructure = _infrastructure_failure(transcript)
    if infrastructure is not None:
        return RunRecord(RunOutcome.VOID, infrastructure)
    for check in scenario.checks:
        failure = check(transcript)
        if failure is not None:
            return RunRecord(RunOutcome.FAIL, failure)
    judged = judge_run(models.rubric_judge, scenario.rubric, transcript)
    if judged.outcome == "void":
        return RunRecord(RunOutcome.VOID, judged.reason)
    if judged.outcome == "fail":
        return RunRecord(RunOutcome.FAIL, f"judge: {judged.reason}")
    return RunRecord(RunOutcome.PASS, "checks and judge passed")


def run_role_once(scenario: RoleScenario, system: SystemConfig) -> RunRecord:
    try:
        outcome, reason = scenario.run(system)
    except TransportError as exc:
        return RunRecord(RunOutcome.VOID, f"a model was unreachable: {exc}")
    return RunRecord(outcome, reason)


def run_scenarios(
    scenarios: Sequence[Scenario],
    *,
    runs: int,
    models: Models,
    system: SystemConfig = SYSTEM,
    role_scenarios: Sequence[RoleScenario] = (),
) -> tuple[list[ScenarioReport], GateResult]:
    if runs < MIN_RUNS:
        raise ValueError(f"runs must be at least {MIN_RUNS}")
    conversations = load_conversations()
    if scenarios:  # one cross-family check up front, before any run
        first = conversations[scenarios[0].conversation_id]
        require_cross_family(models.rubric_judge, models.interviewer(first, 0).family)
    results = []
    for scenario in scenarios:
        conversation = conversations[scenario.conversation_id]
        records = tuple(run_once(scenario, conversation, i, models, system) for i in range(runs))
        results.append(ScenarioReport(scenario, records, grade_runs([r.verdict for r in records])))
    for role_scenario in role_scenarios:
        records = tuple(run_role_once(role_scenario, system) for _ in range(runs))
        results.append(
            ScenarioReport(role_scenario, records, grade_runs([r.verdict for r in records]))
        )
    gate = evaluate_gate(
        {r.scenario.id: r.verdict for r in results},
        {r.scenario.id for r in results if r.scenario.hard_gate},
    )
    return results, gate


def render(
    results: Sequence[ScenarioReport], gate: GateResult, system: SystemConfig = SYSTEM
) -> str:
    lines = []
    mutated = system is not SYSTEM
    if mutated:
        lines.append(f"mutant: {system.name} ({system.description})")
    lines.append(f"{'scenario':<26}{'gate':<6}{'runs':<16}verdict")
    for r in results:
        runs = " ".join(rec.verdict.value for rec in r.runs)
        gate_mark = "yes" if r.scenario.hard_gate else "-"
        lines.append(f"{r.scenario.id:<26}{gate_mark:<6}{runs:<16}{r.verdict.value}")
        for i, rec in enumerate(r.runs, 1):
            if rec.verdict is not RunOutcome.PASS:
                lines.append(f"{'':32}run {i}: {rec.reason}")
    summary = f"GATE: {gate.status.value}"
    if gate.failing:
        summary += f"  failing: {', '.join(gate.failing)}"
    if gate.void:
        summary += f"  void: {', '.join(gate.void)}"
    lines.append(summary)
    if mutated:
        caught = {GateStatus.FAIL: "yes", GateStatus.PASS: "NO", GateStatus.INCOMPLETE: "unknown"}
        lines.append(f"mutant caught: {caught[gate.status]}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the scenario eval harness.")
    parser.add_argument("--runs", type=int, default=MIN_RUNS, help="runs per scenario (>= 2)")
    parser.add_argument("--mutant", choices=sorted(MUTANTS), help="run against a mutant")
    parser.add_argument("--live", action="store_true", help="use real models (needs keys)")
    parser.add_argument("--scenario", action="append", help="limit to these scenario ids")
    args = parser.parse_args(argv)
    if args.runs < MIN_RUNS:
        parser.error(f"--runs must be at least {MIN_RUNS}: one run cannot tell PASS from FLAKY")
    scenarios = [s for s in SCENARIOS if not args.scenario or s.id in args.scenario]
    role_scenarios = [s for s in ROLE_SCENARIOS if not args.scenario or s.id in args.scenario]
    if args.scenario and len(scenarios) + len(role_scenarios) != len(set(args.scenario)):
        parser.error("unknown scenario id")
    system = MUTANTS[args.mutant] if args.mutant else SYSTEM
    models = live_models() if args.live else offline_models()

    try:
        with quiet_package_logs():
            results, gate = run_scenarios(
                scenarios,
                runs=args.runs,
                models=models,
                system=system,
                role_scenarios=role_scenarios,
            )
    except SameFamilyJudgeError as exc:
        sys.stderr.write(f"configuration error: {exc}\n")
        return EXIT_CONFIG_ERROR
    sys.stdout.write(render(results, gate, system) + "\n")
    if args.mutant:
        return {GateStatus.FAIL: 0, GateStatus.PASS: 1, GateStatus.INCOMPLETE: 2}[gate.status]
    return {GateStatus.PASS: 0, GateStatus.FAIL: 1, GateStatus.INCOMPLETE: 2}[gate.status]


if __name__ == "__main__":
    raise SystemExit(main())
