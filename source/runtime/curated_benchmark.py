"""Build and validate a small curated benchmark for report discovery.

The benchmark is intentionally compact:
- 20 representative report-search queries
- 5 retrieved candidate documents per query
- separate retrieval and quality annotations

All rows are synthetic but realistic enough for ranking/evaluation tests.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "curated_benchmark"


QUERY_FIELDS = [
    "query_id",
    "query",
    "topic",
    "industry",
    "geography",
    "target_year_min",
    "target_year_max",
    "intent",
    "notes",
]

DOCUMENT_FIELDS = [
    "doc_id",
    "query_id",
    "retrieved_rank",
    "title",
    "source",
    "source_type",
    "url",
    "year",
    "document_type",
    "is_pdf",
    "snippet",
]

RETRIEVAL_FIELDS = [
    "query_id",
    "doc_id",
    "relevance_label",
    "result_class",
    "ranking_preference",
    "rationale",
]

QUALITY_FIELDS = [
    "doc_id",
    "report_validity_label",
    "deer_method_label",
    "deer_evidence_label",
    "deer_transparency_label",
    "deer_recency_label",
    "authority_label",
    "verification_support_label",
    "overall_quality_label",
    "rationale",
]

SOURCE_TYPE_AUTHORITY: dict[str, float] = {
    "government": 0.95,
    "intergovernmental": 0.93,
    "academic": 0.90,
    "research_institute": 0.85,
    "research_provider": 0.82,
    "professional_association": 0.80,
    "industry_association": 0.78,
    "consulting": 0.75,
    "industry_research": 0.72,
    "trade_media": 0.58,
    "news_media": 0.55,
    "vendor": 0.38,
    "unknown": 0.35,
}

DOCUMENT_TYPE_VALIDITY: dict[str, float] = {
    "research_report": 0.95,
    "whitepaper": 0.68,
    "archived_pdf": 0.55,
    "news_article": 0.20,
    "landing_page": 0.10,
}


@dataclass(frozen=True)
class QuerySeed:
    query_id: str
    query: str
    topic: str
    industry: str
    geography: str
    year_min: int
    year_max: int
    strong_source: str
    strong_source_type: str
    partial_source: str
    partial_source_type: str


QUERY_SEEDS: list[QuerySeed] = [
    QuerySeed("q001", "enterprise AI adoption benchmark report 2025", "enterprise AI adoption", "technology", "global", 2024, 2025, "McKinsey Global Institute", "consulting", "TechVendor Insights", "vendor"),
    QuerySeed("q002", "renewable energy investment outlook 2024 report", "renewable energy investment", "energy", "global", 2023, 2025, "International Energy Agency", "intergovernmental", "SolarWorks Research", "vendor"),
    QuerySeed("q003", "electric vehicle battery market forecast 2025", "EV battery market", "automotive", "global", 2024, 2026, "BloombergNEF", "research_provider", "ChargePoint Strategy Lab", "vendor"),
    QuerySeed("q004", "semiconductor supply chain resilience report 2024", "semiconductor supply chain", "semiconductors", "global", 2023, 2025, "OECD", "intergovernmental", "Silicon Logistics Weekly", "trade_media"),
    QuerySeed("q005", "cybersecurity spending trends benchmark 2024", "cybersecurity spending", "cybersecurity", "north america", 2023, 2025, "Deloitte Center for Cyber", "consulting", "SecureStack Blog", "vendor"),
    QuerySeed("q006", "cloud computing industry outlook 2025 pdf", "cloud computing outlook", "technology", "global", 2024, 2025, "IDC", "research_provider", "CloudOps Digest", "trade_media"),
    QuerySeed("q007", "digital health market report 2024", "digital health market", "healthcare", "global", 2023, 2025, "World Health Organization", "government", "HealthApp Analytics", "vendor"),
    QuerySeed("q008", "sustainable aviation fuel market analysis 2025", "sustainable aviation fuel", "aviation", "global", 2024, 2026, "International Air Transport Association", "industry_association", "JetFuelNow", "vendor"),
    QuerySeed("q009", "commercial real estate outlook report 2024", "commercial real estate outlook", "real estate", "united states", 2023, 2025, "Urban Land Institute", "research_institute", "MetroProperty News", "news_media"),
    QuerySeed("q010", "fintech payments industry benchmark 2025", "fintech payments", "financial services", "global", 2024, 2025, "World Bank", "intergovernmental", "PayFlow Blog", "vendor"),
    QuerySeed("q011", "global e-commerce logistics report 2024", "e-commerce logistics", "logistics", "global", 2023, 2025, "DHL Research", "industry_research", "ShipFast Marketing", "vendor"),
    QuerySeed("q012", "agricultural technology market forecast 2025", "agricultural technology", "agriculture", "global", 2024, 2026, "FAO", "intergovernmental", "AgriCloud Insights", "vendor"),
    QuerySeed("q013", "generative AI in banking research report 2025", "generative AI in banking", "banking", "global", 2024, 2025, "Accenture Research", "consulting", "BankTech Wire", "trade_media"),
    QuerySeed("q014", "hydrogen economy outlook 2024 report", "hydrogen economy", "energy", "global", 2023, 2025, "Hydrogen Council", "industry_association", "GreenMolecule Blog", "vendor"),
    QuerySeed("q015", "insurance claims automation benchmark 2024", "insurance claims automation", "insurance", "north america", 2023, 2025, "Capgemini Research Institute", "consulting", "ClaimsBot Labs", "vendor"),
    QuerySeed("q016", "space economy market size report 2025", "space economy market size", "aerospace", "global", 2024, 2026, "OECD Space Forum", "intergovernmental", "Orbital Markets Daily", "news_media"),
    QuerySeed("q017", "water infrastructure investment needs report 2024", "water infrastructure investment", "infrastructure", "united states", 2023, 2025, "American Society of Civil Engineers", "professional_association", "PipeWorks Solutions", "vendor"),
    QuerySeed("q018", "retail media network benchmark report 2025", "retail media networks", "retail", "north america", 2024, 2025, "IAB Research", "industry_association", "AdRetail Blog", "vendor"),
    QuerySeed("q019", "workforce skills gap survey report 2024", "workforce skills gap", "labor market", "global", 2023, 2025, "World Economic Forum", "research_institute", "TalentSuite Marketing", "vendor"),
    QuerySeed("q020", "data center energy demand outlook 2025", "data center energy demand", "technology infrastructure", "global", 2024, 2026, "International Energy Agency", "intergovernmental", "RackScale Blog", "vendor"),
]


def _slug(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def build_queries() -> list[dict[str, Any]]:
    """Return query-layer benchmark rows."""
    return [
        {
            "query_id": seed.query_id,
            "query": seed.query,
            "topic": seed.topic,
            "industry": seed.industry,
            "geography": seed.geography,
            "target_year_min": seed.year_min,
            "target_year_max": seed.year_max,
            "intent": "find recent credible industry or research reports",
            "notes": "Synthetic benchmark query for capstone evaluation.",
        }
        for seed in QUERY_SEEDS
    ]


def _candidate_templates(seed: QuerySeed) -> list[dict[str, Any]]:
    topic_title = seed.topic.title()
    year = seed.year_max
    old_year = max(2018, seed.year_min - 5)
    slug = _slug(seed.topic)

    return [
        {
            "title": f"{topic_title} Outlook {year}: Benchmark Report",
            "source": seed.strong_source,
            "source_type": seed.strong_source_type,
            "year": year,
            "document_type": "research_report",
            "is_pdf": "true",
            "path": f"{slug}/benchmark-report-{year}.pdf",
            "snippet": f"Full report with methodology, survey sample, references, charts, and forecasts for {seed.topic}.",
            "relevance_label": 3,
            "result_class": "good_report",
            "ranking_preference": 1,
            "report_validity_label": 3,
            "deer_method_label": 3,
            "deer_evidence_label": 3,
            "deer_transparency_label": 3,
            "deer_recency_label": 3,
            "authority_label": 3,
            "verification_support_label": 3,
            "overall_quality_label": 3,
            "retrieval_rationale": "Directly matches the query topic and looks like a full, current report.",
            "quality_rationale": "Strong compressed DEER signals: method, evidence, transparency, and recency are all present.",
        },
        {
            "title": f"{topic_title} Market Update {seed.year_min}",
            "source": seed.partial_source,
            "source_type": seed.partial_source_type,
            "year": seed.year_min,
            "document_type": "whitepaper",
            "is_pdf": "true",
            "path": f"{slug}/market-update-{seed.year_min}.pdf",
            "snippet": f"Short whitepaper summarizing {seed.topic} trends with selected statistics and limited source notes.",
            "relevance_label": 2,
            "result_class": "mediocre_report",
            "ranking_preference": 2,
            "report_validity_label": 2,
            "deer_method_label": 1,
            "deer_evidence_label": 2,
            "deer_transparency_label": 1,
            "deer_recency_label": 2,
            "authority_label": 1,
            "verification_support_label": 2,
            "overall_quality_label": 2,
            "retrieval_rationale": "Relevant and report-like, but weaker source authority and thinner methodology.",
            "quality_rationale": "Some evidence is present, but method and transparency are limited.",
        },
        {
            "title": f"{topic_title}: What Buyers Need to Know",
            "source": seed.partial_source,
            "source_type": seed.partial_source_type,
            "year": year,
            "document_type": "landing_page",
            "is_pdf": "false",
            "path": f"{slug}/buyers-guide",
            "snippet": f"Promotional overview using {seed.topic} keywords, a lead form, and product claims without references.",
            "relevance_label": 1,
            "result_class": "weak_page",
            "ranking_preference": 4,
            "report_validity_label": 0,
            "deer_method_label": 0,
            "deer_evidence_label": 0,
            "deer_transparency_label": 0,
            "deer_recency_label": 3,
            "authority_label": 1,
            "verification_support_label": 0,
            "overall_quality_label": 0,
            "retrieval_rationale": "Contains query terms but is a marketing page rather than a report.",
            "quality_rationale": "No visible method, source support, or verification-friendly evidence.",
        },
        {
            "title": f"Breaking News: Investment Surges Across Adjacent Markets",
            "source": "Market Daily News",
            "source_type": "news_media",
            "year": year,
            "document_type": "news_article",
            "is_pdf": "false",
            "path": f"{slug}/adjacent-market-news",
            "snippet": f"News article with one mention of {seed.topic}, focused mostly on quarterly earnings and executive quotes.",
            "relevance_label": 1,
            "result_class": "false_positive",
            "ranking_preference": 5,
            "report_validity_label": 0,
            "deer_method_label": 0,
            "deer_evidence_label": 1,
            "deer_transparency_label": 0,
            "deer_recency_label": 3,
            "authority_label": 1,
            "verification_support_label": 1,
            "overall_quality_label": 1,
            "retrieval_rationale": "Topical overlap is shallow and the page is not a research report.",
            "quality_rationale": "Recent but not methodical; evidence is mostly quotations rather than report support.",
        },
        {
            "title": f"{topic_title} Historical Overview {old_year}",
            "source": "Archive Research Library",
            "source_type": "unknown",
            "year": old_year,
            "document_type": "archived_pdf",
            "is_pdf": "true",
            "path": f"{slug}/historical-overview-{old_year}.pdf",
            "snippet": f"Older PDF that mentions {seed.topic} but predates the target period and lacks current forecasts.",
            "relevance_label": 1,
            "result_class": "stale_report",
            "ranking_preference": 3,
            "report_validity_label": 2,
            "deer_method_label": 1,
            "deer_evidence_label": 1,
            "deer_transparency_label": 1,
            "deer_recency_label": 0,
            "authority_label": 1,
            "verification_support_label": 1,
            "overall_quality_label": 1,
            "retrieval_rationale": "Report-like but stale for a recent benchmark query.",
            "quality_rationale": "Some structure exists, but weak recency makes it poor for current verification.",
        },
    ]


def build_documents_and_annotations() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return document rows plus retrieval and quality label rows."""
    documents: list[dict[str, Any]] = []
    retrieval_annotations: list[dict[str, Any]] = []
    quality_annotations: list[dict[str, Any]] = []

    for seed in QUERY_SEEDS:
        for index, candidate in enumerate(_candidate_templates(seed), start=1):
            doc_id = f"d{seed.query_id[1:]}_{index:02d}"
            documents.append(
                {
                    "doc_id": doc_id,
                    "query_id": seed.query_id,
                    "retrieved_rank": index,
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "source_type": candidate["source_type"],
                    "url": f"https://benchmark.local/{candidate['path']}",
                    "year": candidate["year"],
                    "document_type": candidate["document_type"],
                    "is_pdf": candidate["is_pdf"],
                    "snippet": candidate["snippet"],
                }
            )
            retrieval_annotations.append(
                {
                    "query_id": seed.query_id,
                    "doc_id": doc_id,
                    "relevance_label": candidate["relevance_label"],
                    "result_class": candidate["result_class"],
                    "ranking_preference": candidate["ranking_preference"],
                    "rationale": candidate["retrieval_rationale"],
                }
            )
            quality_annotations.append(
                {
                    "doc_id": doc_id,
                    "report_validity_label": candidate["report_validity_label"],
                    "deer_method_label": candidate["deer_method_label"],
                    "deer_evidence_label": candidate["deer_evidence_label"],
                    "deer_transparency_label": candidate["deer_transparency_label"],
                    "deer_recency_label": candidate["deer_recency_label"],
                    "authority_label": candidate["authority_label"],
                    "verification_support_label": candidate["verification_support_label"],
                    "overall_quality_label": candidate["overall_quality_label"],
                    "rationale": candidate["quality_rationale"],
                }
            )

    return documents, retrieval_annotations, quality_annotations


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_dataset_tables(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, list[dict[str, str]]]:
    """Load all benchmark CSV files into plain row dictionaries."""
    return {
        "queries": _read_csv(dataset_dir / "queries.csv"),
        "documents": _read_csv(dataset_dir / "documents.csv"),
        "retrieval_annotations": _read_csv(dataset_dir / "retrieval_annotations.csv"),
        "quality_annotations": _read_csv(dataset_dir / "quality_annotations.csv"),
    }


def _build_proxy_report_text(document: dict[str, str]) -> str:
    """Create report-like text from document metadata without using label files."""
    title = document.get("title", "")
    source = document.get("source", "")
    snippet = document.get("snippet", "")
    year = document.get("year", "")
    doc_type = document.get("document_type", "")

    if doc_type == "research_report":
        extra = (
            "Executive summary. Methodology. Research design. We conducted a survey sample "
            "and analyzed market data across regions. Findings include 35% growth and 18% "
            "adoption change. Results. Forecast. Conclusion. References. "
            f"1 Source: {source} analysis {year}. 2 Source: public dataset {year}."
        )
    elif doc_type == "whitepaper":
        extra = (
            "Overview. Limited methodology. The analysis summarizes selected statistics and "
            "market projections. Findings include 12% growth. Source notes are provided, "
            "but assumptions and sampling details are brief."
        )
    elif doc_type == "archived_pdf":
        extra = (
            "Introduction. Background. Methods. Older market overview with historical tables. "
            "References are limited and forecasts do not cover the target year."
        )
    elif doc_type == "news_article":
        extra = (
            "News article. Executive quotes and recent announcements. No methodology, no "
            "sample design, and limited source notes."
        )
    elif doc_type == "landing_page":
        extra = (
            "Product page. Sign up, request demo, contact sales, learn more. Promotional "
            "claims without methodology, references, or transparent data sources."
        )
    else:
        extra = "Short web result with limited context."

    return " ".join(part for part in [title, source, snippet, extra] if part).strip()


def document_to_rank_report_candidate(document: dict[str, str], query: str) -> dict[str, Any]:
    """Adapt one documents.csv row to the candidate shape consumed by rank_reports()."""
    from source.extractor import extract_signals, source_score
    from source.scoring import compute_relevance_score

    text = _build_proxy_report_text(document)
    source = str(document.get("source", "")).strip()
    source_type = str(document.get("source_type", "unknown")).strip()
    document_type = str(document.get("document_type", "unknown")).strip()
    year = _to_int(document.get("year"))
    is_pdf = _to_bool(document.get("is_pdf"))
    url = str(document.get("url", "")).strip()

    metadata = {
        "year": year,
        "source": source,
        "is_pdf": is_pdf,
        "format": "pdf" if is_pdf else "html",
    }
    signals = extract_signals(text, metadata)
    signals["source_name"] = source
    signals["source"] = source_score(source, text)
    signals["_text"] = text
    signals["source_class"] = source_type
    signals["authority_prior"] = SOURCE_TYPE_AUTHORITY.get(source_type, SOURCE_TYPE_AUTHORITY["unknown"])
    signals["report_type"] = document_type
    signals["report_validity_score_classifier"] = DOCUMENT_TYPE_VALIDITY.get(document_type, 0.35)

    return {
        "doc_id": document.get("doc_id", ""),
        "query_id": document.get("query_id", ""),
        "title": document.get("title", ""),
        "url": url,
        "source": source,
        "year": year,
        "snippet": document.get("snippet", ""),
        "text": text,
        "is_pdf": is_pdf,
        "relevance": compute_relevance_score(query, text),
        "signals": signals,
    }


def rank_documents_from_csv(dataset_dir: Path = DEFAULT_OUTPUT_DIR, top_k: int | None = None) -> list[dict[str, Any]]:
    """Feed documents.csv through the existing rank_reports() path."""
    from source.scoring import rank_reports

    tables = load_dataset_tables(dataset_dir)
    query_by_id = {row["query_id"]: row["query"] for row in tables["queries"]}
    docs_by_query: dict[str, list[dict[str, str]]] = {}
    for document in tables["documents"]:
        docs_by_query.setdefault(document["query_id"], []).append(document)

    ranked_rows: list[dict[str, Any]] = []
    for query_id, documents in sorted(docs_by_query.items()):
        query = query_by_id.get(query_id, "")
        candidates = [document_to_rank_report_candidate(document, query) for document in documents]
        doc_by_url = {candidate["url"]: candidate for candidate in candidates}
        ranked = rank_reports(candidates, top_k=top_k, query=query)
        for rank, row in enumerate(ranked, start=1):
            candidate = doc_by_url.get(str(row.get("url", "")), {})
            ranked_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "rank": rank,
                    "doc_id": candidate.get("doc_id", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "score": row.get("score", 0.0),
                    "score_breakdown": row.get("score_breakdown", {}),
                    "reason": row.get("reason", ""),
                }
            )
    return ranked_rows


def _candidate_score(score_breakdown: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(weights[key] * _to_float(score_breakdown.get(key)) for key in weights)


def _dcg(gains: list[float]) -> float:
    return sum((2.0 ** gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(gains))


def _ndcg(labels: list[float], k: int) -> float:
    actual = labels[:k]
    ideal = sorted(labels, reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(actual) / ideal_dcg


def _pairwise_accuracy(ranked_doc_ids: list[str], preference_by_doc: dict[str, int]) -> float:
    checked = 0
    correct = 0
    position_by_doc = {doc_id: index for index, doc_id in enumerate(ranked_doc_ids)}
    doc_ids = list(preference_by_doc)
    for left_index, left_doc_id in enumerate(doc_ids):
        for right_doc_id in doc_ids[left_index + 1:]:
            left_pref = preference_by_doc[left_doc_id]
            right_pref = preference_by_doc[right_doc_id]
            if left_pref == right_pref:
                continue
            checked += 1
            expected_first = left_doc_id if left_pref < right_pref else right_doc_id
            expected_second = right_doc_id if expected_first == left_doc_id else left_doc_id
            if position_by_doc[expected_first] < position_by_doc[expected_second]:
                correct += 1
    return correct / checked if checked else 0.0


def _quality_target(quality_row: dict[str, str]) -> float:
    fields = [
        "report_validity_label",
        "deer_method_label",
        "deer_evidence_label",
        "deer_transparency_label",
        "authority_label",
        "verification_support_label",
        "overall_quality_label",
    ]
    values = [_to_float(quality_row.get(field)) for field in fields]
    return sum(values) / len(values) if values else 0.0


def _combined_label(
    doc_id: str,
    retrieval_by_doc: dict[str, dict[str, str]],
    quality_by_doc: dict[str, dict[str, str]],
) -> float:
    relevance = _to_float(retrieval_by_doc[doc_id].get("relevance_label"))
    quality = _quality_target(quality_by_doc[doc_id])
    return 0.60 * relevance + 0.40 * quality


def evaluate_weight_set(
    ranked_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    weights: dict[str, float],
    k: int = 5,
) -> dict[str, float]:
    """Evaluate one final-score weight set against benchmark labels."""
    retrieval_by_doc = {row["doc_id"]: row for row in retrieval_rows}
    quality_by_doc = {row["doc_id"]: row for row in quality_rows}

    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in ranked_rows:
        rows_by_query.setdefault(row["query_id"], []).append(row)

    ndcg_values: list[float] = []
    precision_values: list[float] = []
    pairwise_values: list[float] = []
    top1_values: list[float] = []

    for rows in rows_by_query.values():
        sorted_rows = sorted(
            rows,
            key=lambda item: _candidate_score(item["score_breakdown"], weights),
            reverse=True,
        )
        doc_ids = [str(row["doc_id"]) for row in sorted_rows]
        labels = [_combined_label(doc_id, retrieval_by_doc, quality_by_doc) for doc_id in doc_ids]
        relevant_flags = [1.0 if _to_float(retrieval_by_doc[doc_id]["relevance_label"]) >= 2 else 0.0 for doc_id in doc_ids[:k]]
        preference_by_doc = {
            doc_id: _to_int(retrieval_by_doc[doc_id]["ranking_preference"])
            for doc_id in doc_ids
        }

        ndcg_values.append(_ndcg(labels, k=k))
        precision_values.append(sum(relevant_flags) / k)
        pairwise_values.append(_pairwise_accuracy(doc_ids, preference_by_doc))
        top1_values.append(1.0 if preference_by_doc[doc_ids[0]] == 1 else 0.0)

    query_count = max(len(rows_by_query), 1)
    mean_ndcg = sum(ndcg_values) / query_count
    mean_precision = sum(precision_values) / query_count
    mean_pairwise = sum(pairwise_values) / query_count
    top1_accuracy = sum(top1_values) / query_count
    objective = 0.60 * mean_ndcg + 0.25 * mean_pairwise + 0.15 * top1_accuracy

    return {
        "objective": round(objective, 6),
        "mean_ndcg_at_k": round(mean_ndcg, 6),
        "mean_precision_at_k": round(mean_precision, 6),
        "mean_pairwise_accuracy": round(mean_pairwise, 6),
        "top1_accuracy": round(top1_accuracy, 6),
    }


def iter_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    """Generate final-score weight combinations that sum to 1."""
    scale = round(1.0 / step)
    weights: list[dict[str, float]] = []
    for relevance in range(1, scale):
        for validity in range(1, scale - relevance):
            for quality in range(1, scale - relevance - validity):
                authority = scale - relevance - validity - quality
                if authority <= 0:
                    continue
                weights.append(
                    {
                        "relevance_score": round(relevance * step, 4),
                        "report_validity_score": round(validity * step, 4),
                        "quality_score": round(quality * step, 4),
                        "authority_score": round(authority * step, 4),
                    }
                )
    return weights


def tune_final_weights(
    dataset_dir: Path = DEFAULT_OUTPUT_DIR,
    step: float = 0.05,
    k: int = 5,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Search FINAL_WEIGHTS values using the curated benchmark labels."""
    tables = load_dataset_tables(dataset_dir)
    ranked_rows = rank_documents_from_csv(dataset_dir, top_k=None)
    results: list[dict[str, Any]] = []

    for weights in iter_weight_grid(step=step):
        metrics = evaluate_weight_set(
            ranked_rows=ranked_rows,
            retrieval_rows=tables["retrieval_annotations"],
            quality_rows=tables["quality_annotations"],
            weights=weights,
            k=k,
        )
        results.append({"weights": weights, **metrics})

    results.sort(
        key=lambda row: (
            row["objective"],
            row["mean_ndcg_at_k"],
            row["mean_pairwise_accuracy"],
            row["top1_accuracy"],
        ),
        reverse=True,
    )
    return results[:top_n]


def write_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Write the starter benchmark CSV files and schema notes."""
    queries = build_queries()
    documents, retrieval_annotations, quality_annotations = build_documents_and_annotations()

    _write_csv(output_dir / "queries.csv", QUERY_FIELDS, queries)
    _write_csv(output_dir / "documents.csv", DOCUMENT_FIELDS, documents)
    _write_csv(output_dir / "retrieval_annotations.csv", RETRIEVAL_FIELDS, retrieval_annotations)
    _write_csv(output_dir / "quality_annotations.csv", QUALITY_FIELDS, quality_annotations)
    (output_dir / "README.md").write_text(build_schema_readme(), encoding="utf-8")


def _expect_fields(path: Path, rows: list[dict[str, str]], expected: list[str], errors: list[str]) -> None:
    if not rows:
        errors.append(f"{path.name} has no rows")
        return
    actual = set(rows[0].keys())
    missing = [field for field in expected if field not in actual]
    if missing:
        errors.append(f"{path.name} missing columns: {', '.join(missing)}")


def _int_in_range(value: str, low: int, high: int) -> bool:
    try:
        numeric = int(value)
    except ValueError:
        return False
    return low <= numeric <= high


def validate_dataset(dataset_dir: Path = DEFAULT_OUTPUT_DIR) -> list[str]:
    """Validate benchmark shape and label ranges.

    Returns a list of errors. An empty list means the dataset is valid.
    """
    paths = {
        "queries": dataset_dir / "queries.csv",
        "documents": dataset_dir / "documents.csv",
        "retrieval": dataset_dir / "retrieval_annotations.csv",
        "quality": dataset_dir / "quality_annotations.csv",
    }
    errors: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"Missing {name} file: {path}")
    if errors:
        return errors

    queries = _read_csv(paths["queries"])
    documents = _read_csv(paths["documents"])
    retrieval = _read_csv(paths["retrieval"])
    quality = _read_csv(paths["quality"])

    _expect_fields(paths["queries"], queries, QUERY_FIELDS, errors)
    _expect_fields(paths["documents"], documents, DOCUMENT_FIELDS, errors)
    _expect_fields(paths["retrieval"], retrieval, RETRIEVAL_FIELDS, errors)
    _expect_fields(paths["quality"], quality, QUALITY_FIELDS, errors)

    query_ids = [row["query_id"] for row in queries]
    doc_ids = [row["doc_id"] for row in documents]
    if len(query_ids) != len(set(query_ids)):
        errors.append("queries.csv contains duplicate query_id values")
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("documents.csv contains duplicate doc_id values")
    if len(queries) != 20:
        errors.append(f"Expected 20 queries, found {len(queries)}")
    if len(documents) != 100:
        errors.append(f"Expected 100 documents, found {len(documents)}")

    query_id_set = set(query_ids)
    doc_id_set = set(doc_ids)
    counts_by_query = {query_id: 0 for query_id in query_id_set}
    for row in documents:
        query_id = row["query_id"]
        if query_id not in query_id_set:
            errors.append(f"Document {row['doc_id']} references unknown query_id {query_id}")
        else:
            counts_by_query[query_id] += 1
        if row["is_pdf"] not in {"true", "false"}:
            errors.append(f"Document {row['doc_id']} has invalid is_pdf value {row['is_pdf']}")

    for query_id, count in sorted(counts_by_query.items()):
        if count != 5:
            errors.append(f"Query {query_id} has {count} documents; expected 5")

    retrieval_doc_ids = {row["doc_id"] for row in retrieval}
    quality_doc_ids = {row["doc_id"] for row in quality}
    if retrieval_doc_ids != doc_id_set:
        errors.append("retrieval_annotations.csv doc_id set does not match documents.csv")
    if quality_doc_ids != doc_id_set:
        errors.append("quality_annotations.csv doc_id set does not match documents.csv")

    for row in retrieval:
        if row["query_id"] not in query_id_set:
            errors.append(f"Retrieval annotation references unknown query_id {row['query_id']}")
        if not _int_in_range(row["relevance_label"], 0, 3):
            errors.append(f"Retrieval label for {row['doc_id']} must be in 0..3")
        if not _int_in_range(row["ranking_preference"], 1, 5):
            errors.append(f"Ranking preference for {row['doc_id']} must be in 1..5")

    quality_label_fields = [
        "report_validity_label",
        "deer_method_label",
        "deer_evidence_label",
        "deer_transparency_label",
        "deer_recency_label",
        "authority_label",
        "verification_support_label",
        "overall_quality_label",
    ]
    for row in quality:
        for field in quality_label_fields:
            if not _int_in_range(row[field], 0, 3):
                errors.append(f"{field} for {row['doc_id']} must be in 0..3")

    return errors


def build_schema_readme() -> str:
    """Return human-readable schema notes for the benchmark folder."""
    return """# Curated Report Verification Benchmark

This folder contains a small synthetic benchmark for an AI-assisted report search and evaluation agent.
It is designed for a capstone project: small enough to inspect manually, structured enough to evaluate retrieval, ranking, report quality, and lightweight verification behavior.

## Files

- `queries.csv`: representative user queries for report discovery.
- `documents.csv`: five retrieved candidate documents per query. This file contains retrieval metadata only, not quality labels.
- `retrieval_annotations.csv`: query-document relevance labels and result classes.
- `quality_annotations.csv`: document-level report quality labels using compressed DEER-inspired dimensions.

## Label Scale

All quality and relevance labels use the same interpretable 0-3 scale:

- `0`: absent, poor, or not useful
- `1`: weak
- `2`: acceptable
- `3`: strong

`ranking_preference` uses `1` for the best candidate within a query and `5` for the weakest.

## Compressed DEER-Inspired Dimensions

DEER is used only as an expert-informed annotation framework. The benchmark compresses it into four practical dimensions:

- `deer_method_label`: whether the document explains method, sample, scope, or analytical approach.
- `deer_evidence_label`: whether claims are supported by data, references, charts, or source notes.
- `deer_transparency_label`: whether assumptions, definitions, limitations, or sources are visible.
- `deer_recency_label`: whether the document is current enough for the query.

The dataset intentionally does not copy the full DEER rubric.

## Intended Uses

- Retrieval evaluation: did the agent find report-like candidates?
- Ranking evaluation: did strong reports rank above weak pages and false positives?
- Quality scoring calibration: do scoring coefficients align with human-readable labels?
- Verification support evaluation: does the selected report expose enough evidence for lightweight claim checks?

## Synthetic Data Note

The rows are synthetic starter examples. URLs use `https://benchmark.local/...` so they cannot be confused with live documents.
Replace or extend rows with real search results when you are ready for a stronger benchmark.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate the curated report benchmark.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write", action="store_true", help="Write starter CSV files before validation.")
    parser.add_argument("--rank", action="store_true", help="Rank documents.csv through source.scoring.rank_reports().")
    parser.add_argument("--tune", action="store_true", help="Tune FINAL_WEIGHTS against benchmark labels.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff used by ranking metrics.")
    parser.add_argument("--step", type=float, default=0.05, help="Grid-search step size for weight tuning.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of tuned weight sets to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        write_dataset(args.output_dir)
        print(f"Wrote benchmark files to {args.output_dir}")

    errors = validate_dataset(args.output_dir)
    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Dataset validation passed: {args.output_dir}")

    if args.rank:
        ranked_rows = rank_documents_from_csv(args.output_dir, top_k=args.k)
        print(f"\nTop ranked rows from rank_reports() path, k={args.k}:")
        for row in ranked_rows[: min(20, len(ranked_rows))]:
            breakdown = row.get("score_breakdown", {})
            print(
                f"{row['query_id']} #{row['rank']} {row['doc_id']} "
                f"score={_to_float(row['score']):.3f} "
                f"rel={_to_float(breakdown.get('relevance_score')):.3f} "
                f"valid={_to_float(breakdown.get('report_validity_score')):.3f} "
                f"quality={_to_float(breakdown.get('quality_score')):.3f} "
                f"auth={_to_float(breakdown.get('authority_score')):.3f}"
            )

    if args.tune:
        tuned = tune_final_weights(args.output_dir, step=args.step, k=args.k, top_n=args.top_n)
        print(f"\nTop {len(tuned)} FINAL_WEIGHTS candidates:")
        for index, row in enumerate(tuned, start=1):
            weights = row["weights"]
            print(
                f"{index}. objective={row['objective']:.6f} "
                f"ndcg@{args.k}={row['mean_ndcg_at_k']:.6f} "
                f"pairwise={row['mean_pairwise_accuracy']:.6f} "
                f"top1={row['top1_accuracy']:.6f} "
                f"weights={weights}"
            )


if __name__ == "__main__":
    main()
