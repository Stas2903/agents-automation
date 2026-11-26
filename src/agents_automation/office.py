"""Builds the synthetic office: the handbook ingested into the knowledge store, the team chat
with its summaries written, and the roles on top, all with the offline models. Used by the
demo, the eval scenarios and the tests, so they all exercise the same setup."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from agents_automation.directory import Directory, load_directory
from agents_automation.knowledge.ingest import IngestReport, ingest, load_documents
from agents_automation.knowledge.role import AnswerParser, KnowledgeRole, parse_answer
from agents_automation.knowledge.store import KnowledgeStore
from agents_automation.llm.boundary import (
    ModelRouter,
    PlacedEmbedder,
    PlacedModel,
    Placement,
    Sensitivity,
)
from agents_automation.llm.fakes import HashingEmbedder
from agents_automation.offline_roles import (
    coverage_reranker,
    extractive_answerer,
    summary_composer,
    summary_writer,
)
from agents_automation.summarizer.chat import TeamChat, load_team_chat
from agents_automation.summarizer.periods import PeriodRequest
from agents_automation.summarizer.role import CitationJudge, SummarizerRole, make_citation_judge
from agents_automation.summarizer.summaries import SummaryShelf, write_summaries

SELF_HOSTED = Placement("self-hosted", Sensitivity.CONFIDENTIAL)
EXTERNAL = Placement("external", Sensitivity.INTERNAL)
TODAY = date(2026, 3, 27)  # the demo's "today": the platform's clock, a trusted input


@dataclass(frozen=True)
class OfficeData:
    directory: Directory
    store: KnowledgeStore
    embedder: HashingEmbedder
    ingest_report: IngestReport
    chat: TeamChat
    shelf: SummaryShelf


def build_office_data(directory: Directory | None = None) -> OfficeData:
    store, embedder = KnowledgeStore(), HashingEmbedder()
    report = ingest(store, load_documents(), PlacedEmbedder(embedder, SELF_HOSTED))
    chat = load_team_chat()
    shelf = write_summaries(chat, ModelRouter([PlacedModel(summary_writer(), SELF_HOSTED)]))
    return OfficeData(directory or load_directory(), store, embedder, report, chat, shelf)


def knowledge_role(
    data: OfficeData,
    models: Sequence[PlacedModel] | None = None,
    *,
    answer_parser: AnswerParser = parse_answer,
) -> KnowledgeRole:
    router = ModelRouter(models or [PlacedModel(extractive_answerer(), SELF_HOSTED)])
    reranker = PlacedModel(coverage_reranker(), SELF_HOSTED)
    return KnowledgeRole(
        data.store,
        data.directory,
        PlacedEmbedder(data.embedder, SELF_HOSTED),
        router,
        reranker=reranker,
        answer_parser=answer_parser,
    )


def summarizer_role(
    data: OfficeData,
    models: Sequence[PlacedModel] | None = None,
    *,
    judge_factory: Callable[[TeamChat, PeriodRequest], CitationJudge] = make_citation_judge,
) -> SummarizerRole:
    router = ModelRouter(models or [PlacedModel(summary_composer(), SELF_HOSTED)])
    return SummarizerRole(
        data.chat, data.shelf, data.directory, router, today=TODAY, judge_factory=judge_factory
    )
