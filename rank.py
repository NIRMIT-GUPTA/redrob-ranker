#!/usr/bin/env python3
"""
Offline Redrob candidate ranker.

Produces the required top-100 CSV:
candidate_id,rank,score,reasoning

The ranker is intentionally deterministic, CPU-only, dependency-free, and
streaming for JSONL/GZIP inputs. It combines JD fit, career evidence, skills,
behavioral availability signals, and anomaly penalties for likely honeypots.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


TOP_K = 100
REFERENCE_DATE = date(2026, 6, 29)

CONSULTING_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "hcl",
    "tech mahindra",
    "lti mindtree",
}

PRODUCT_INDUSTRIES = {
    "software",
    "saas",
    "ai/ml",
    "ai services",
    "conversational ai",
    "healthtech ai",
    "fintech",
    "edtech",
    "e-commerce",
    "adtech",
    "gaming",
    "internet",
    "marketplace",
    "healthtech",
}

TIER1_CITIES = {
    "pune",
    "noida",
    "gurgaon",
    "gurugram",
    "delhi",
    "new delhi",
    "hyderabad",
    "bangalore",
    "bengaluru",
    "mumbai",
    "navi mumbai",
    "chennai",
}

PROFICIENCY_VALUE = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}

CORE_SKILL_WEIGHTS = {
    "information retrieval": 0.105,
    "embeddings": 0.095,
    "vector search": 0.100,
    "semantic search": 0.080,
    "recommendation systems": 0.105,
    "sentence transformers": 0.075,
    "rag": 0.055,
    "llms": 0.040,
    "fine-tuning llms": 0.050,
    "lora": 0.030,
    "qlora": 0.030,
    "pinecone": 0.055,
    "faiss": 0.055,
    "milvus": 0.050,
    "qdrant": 0.055,
    "weaviate": 0.050,
    "elasticsearch": 0.045,
    "opensearch": 0.045,
    "hugging face transformers": 0.045,
    "mlops": 0.040,
    "mlflow": 0.035,
    "kubeflow": 0.030,
    "bentoml": 0.030,
    "weights & biases": 0.025,
    "feature engineering": 0.030,
    "data science": 0.020,
    "python": 0.045,
    "fastapi": 0.020,
}

HYPE_ONLY_SKILLS = {"langchain", "prompt engineering"}

TEXT_GROUPS: Sequence[Tuple[str, Sequence[str], float, int]] = (
    (
        "retrieval",
        (
            "retrieval",
            "information retrieval",
            "semantic search",
            "vector search",
            "dense retrieval",
            "hybrid retrieval",
            "bm25",
            "embeddings",
            "embedding",
            "sentence transformer",
            "bge",
            "e5",
            "rag",
            "candidate-jd matching",
            "matching pipeline",
        ),
        0.26,
        5,
    ),
    (
        "ranking_eval",
        (
            "ranking",
            "ranker",
            "recommendation",
            "recommender",
            "search relevance",
            "ndcg",
            "mrr",
            "map",
            "offline benchmark",
            "evaluation harness",
            "a/b test",
            "ab test",
            "online experiment",
        ),
        0.24,
        5,
    ),
    (
        "infra",
        (
            "pinecone",
            "qdrant",
            "milvus",
            "faiss",
            "weaviate",
            "elasticsearch",
            "opensearch",
            "vector database",
            "index refresh",
            "retrieval-quality regression",
            "embedding drift",
        ),
        0.18,
        4,
    ),
    (
        "production",
        (
            "production",
            "deployed",
            "shipped",
            "real users",
            "scale",
            "latency",
            "monitoring",
            "regression",
            "online",
            "platform",
            "owned",
        ),
        0.17,
        5,
    ),
    (
        "product_domain",
        (
            "recruiter",
            "candidate",
            "talent",
            "marketplace",
            "matching",
            "search",
            "hr-tech",
            "hr tech",
        ),
        0.08,
        4,
    ),
    (
        "implementation",
        (
            "python",
            "fastapi",
            "airflow",
            "spark",
            "kafka",
            "mlflow",
            "kubeflow",
            "feature pipeline",
        ),
        0.07,
        4,
    ),
)

WORD_RE = re.compile(r"[a-z0-9+#.]+")


@dataclass
class FeatureBreakdown:
    score: float
    title_score: float
    text_score: float
    skill_score: float
    experience_score: float
    location_score: float
    behavior_score: float
    product_score: float
    education_score: float
    anomaly_penalty: float
    matched_terms: List[str] = field(default_factory=list)
    matched_skills: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def unique_hits(text: str, terms: Sequence[str]) -> List[str]:
    text_l = text.lower()
    hits = []
    for term in terms:
        if term in text_l:
            hits.append(term)
    return dedupe_keep_order([h[:-1] if h.endswith("s") else h for h in hits])


def candidate_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []) or []:
        parts.extend(
            [
                job.get("title", ""),
                job.get("industry", ""),
                job.get("description", ""),
                job.get("company", ""),
            ]
        )
    for edu in candidate.get("education", []) or []:
        parts.extend([edu.get("degree", ""), edu.get("field_of_study", ""), edu.get("institution", "")])
    return " ".join(str(p) for p in parts if p)


def title_relevance(title: str) -> float:
    t = norm(title)
    if not t:
        return 0.0
    high = (
        "senior ai engineer",
        "lead ai engineer",
        "staff machine learning engineer",
        "senior machine learning engineer",
        "senior ml engineer",
    )
    if any(x in t for x in high):
        return 1.0
    if "recommendation systems engineer" in t or "search engineer" in t:
        return 0.95
    if "senior nlp engineer" in t or "senior applied scientist" in t:
        return 0.90
    if "applied ml engineer" in t or "machine learning engineer" in t:
        return 0.86
    if t in {"ml engineer", "ai engineer", "nlp engineer"} or " ml engineer" in t:
        return 0.82
    if "senior software engineer (ml)" in t or ("software engineer" in t and "ml" in t):
        return 0.74
    if "data scientist" in t:
        return 0.66
    if "data engineer" in t or "analytics engineer" in t or "backend engineer" in t:
        return 0.42
    if "software engineer" in t or "full stack" in t or "cloud engineer" in t:
        return 0.34
    if "computer vision" in t or "speech" in t or "robotics" in t:
        return 0.24
    return 0.0


def score_titles(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    current = title_relevance(profile.get("current_title", ""))
    history = [title_relevance(job.get("title", "")) for job in candidate.get("career_history", []) or []]
    best_history = max(history or [0.0])
    return clamp(0.70 * current + 0.30 * best_history)


def score_text(text: str) -> Tuple[float, List[str]]:
    score = 0.0
    matched: List[str] = []
    for _name, terms, weight, cap in TEXT_GROUPS:
        hits = unique_hits(text, terms)
        matched.extend(hits[:3])
        score += weight * min(len(hits), cap) / cap
    return clamp(score), dedupe_keep_order(matched)


def skill_quality(skill: Dict[str, Any]) -> float:
    prof = PROFICIENCY_VALUE.get(norm(skill.get("proficiency")), 0.35)
    duration = float(skill.get("duration_months") or 0)
    endorsements = float(skill.get("endorsements") or 0)
    duration_factor = clamp(duration / 36.0)
    endorsement_factor = clamp(math.log1p(endorsements) / math.log(61))
    return prof * (0.55 + 0.35 * duration_factor + 0.10 * endorsement_factor)


def score_skills(candidate: Dict[str, Any], text: str) -> Tuple[float, List[str]]:
    total = 0.0
    matched: List[str] = []
    skills = candidate.get("skills", []) or []
    for skill in skills:
        name = norm(skill.get("name"))
        if name in CORE_SKILL_WEIGHTS:
            total += CORE_SKILL_WEIGHTS[name] * skill_quality(skill)
            matched.append(skill.get("name", ""))
    if "python" in text.lower():
        total += 0.035
        matched.append("Python")
    return clamp(total), dedupe_keep_order([m for m in matched if m])[:8]


def score_experience(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 10.0:
        return 0.78
    if 3.0 <= years < 4.0 or 10.0 < years <= 12.0:
        return 0.48
    if 12.0 < years <= 15.0:
        return 0.26
    return 0.18


def score_location(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = norm(profile.get("country"))
    location = norm(profile.get("location"))
    willing = bool(signals.get("willing_to_relocate"))
    mode = norm(signals.get("preferred_work_mode"))

    if country != "india":
        return 0.28 if willing else 0.18

    score = 0.56
    if any(city in location for city in ("pune", "noida")):
        score = 1.0
    elif any(city in location for city in TIER1_CITIES):
        score = 0.82
    if willing:
        score = max(score, 0.86)
    if mode in {"hybrid", "flexible", "onsite"}:
        score += 0.08
    elif mode == "remote" and not willing:
        score -= 0.10
    return clamp(score)


def score_signup_tenure(signals: Dict[str, Any]) -> float:
    signup = parse_date(signals.get("signup_date"))
    if not signup:
        return 0.45
    days = max(0, (REFERENCE_DATE - signup).days)
    if days < 14:
        return 0.45
    if days < 45:
        return 0.72
    if days <= 900:
        return 1.0
    return 0.82


def score_activity(signals: Dict[str, Any]) -> float:
    last_active = parse_date(signals.get("last_active_date"))
    days_inactive = 365
    if last_active:
        days_inactive = max(0, (REFERENCE_DATE - last_active).days)
    if days_inactive <= 14:
        return 1.0
    if days_inactive <= 30:
        return 0.90
    if days_inactive <= 60:
        return 0.75
    if days_inactive <= 90:
        return 0.58
    if days_inactive <= 180:
        return 0.34
    return 0.12


def score_notice(days: int) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.78
    if days <= 90:
        return 0.48
    if days <= 120:
        return 0.25
    return 0.12


def score_applications(count: int) -> float:
    if 1 <= count <= 5:
        return 1.0
    if count == 0:
        return 0.58
    if count <= 10:
        return 0.78
    if count <= 20:
        return 0.45
    return 0.22


def score_salary_expectation(signals: Dict[str, Any]) -> float:
    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    low = float(salary.get("min") or 0)
    high = float(salary.get("max") or 0)
    if not low or not high:
        return 0.45
    if low > high:
        return 0.05
    midpoint = (low + high) / 2.0
    spread = high - low
    if midpoint < 12:
        level = 0.45
    elif midpoint <= 85:
        level = 1.0
    elif midpoint <= 120:
        level = 0.70
    else:
        level = 0.35
    if spread > 55:
        level -= 0.12
    return clamp(level)


def score_assessments(signals: Dict[str, Any]) -> float:
    assessments = signals.get("skill_assessment_scores", {}) or {}
    if not assessments:
        return 0.35
    values = []
    for name, value in assessments.items():
        weight = 1.25 if norm(name) in CORE_SKILL_WEIGHTS else 0.75
        values.append(weight * clamp(float(value or 0) / 100.0))
    return clamp(sum(values) / max(1, len(values)))


def score_work_mode(signals: Dict[str, Any]) -> float:
    mode = norm(signals.get("preferred_work_mode"))
    willing = bool(signals.get("willing_to_relocate"))
    if mode in {"hybrid", "flexible"}:
        return 1.0
    if mode == "onsite":
        return 0.88
    if mode == "remote":
        return 0.70 if willing else 0.42
    return 0.50


def score_behavior(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {}) or {}
    profile_complete = clamp(float(signals.get("profile_completeness_score") or 0) / 100.0)
    signup_tenure = score_signup_tenure(signals)
    active = score_activity(signals)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.35
    views = clamp(float(signals.get("profile_views_received_30d") or 0) / 80.0)
    applications = score_applications(int(signals.get("applications_submitted_30d") or 0))
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    response = clamp((response_rate - 0.05) / 0.80)
    response_hours = float(signals.get("avg_response_time_hours") or 280.0)
    speed = clamp(1.0 - response_hours / 240.0)
    assessments = score_assessments(signals)
    connections = clamp(math.log1p(float(signals.get("connection_count") or 0)) / math.log(1001))
    endorsements = clamp(math.log1p(float(signals.get("endorsements_received") or 0)) / math.log(251))
    notice_score = score_notice(int(signals.get("notice_period_days") or 180))
    salary = score_salary_expectation(signals)
    work_mode = score_work_mode(signals)
    relocate = 1.0 if signals.get("willing_to_relocate") else 0.45
    github_raw = float(signals.get("github_activity_score", -1))
    github = 0.35 if github_raw < 0 else clamp(github_raw / 100.0)
    search_appearance = clamp(float(signals.get("search_appearance_30d") or 0) / 250.0)
    saved = clamp(float(signals.get("saved_by_recruiters_30d") or 0) / 15.0)
    interview = clamp((float(signals.get("interview_completion_rate") or 0.0) - 0.30) / 0.70)
    offer_raw = float(signals.get("offer_acceptance_rate", -1))
    offer = 0.55 if offer_raw < 0 else clamp(offer_raw)
    verified = (
        0.34 * bool(signals.get("verified_email"))
        + 0.33 * bool(signals.get("verified_phone"))
        + 0.33 * bool(signals.get("linkedin_connected"))
    )

    return clamp(
        0.040 * profile_complete
        + 0.020 * signup_tenure
        + 0.095 * active
        + 0.040 * open_to_work
        + 0.035 * views
        + 0.030 * applications
        + 0.105 * response
        + 0.060 * speed
        + 0.055 * assessments
        + 0.025 * connections
        + 0.030 * endorsements
        + 0.085 * notice_score
        + 0.030 * salary
        + 0.035 * work_mode
        + 0.025 * relocate
        + 0.045 * github
        + 0.035 * search_appearance
        + 0.050 * saved
        + 0.075 * interview
        + 0.035 * offer
        + 0.050 * verified
    )


def score_product_context(candidate: Dict[str, Any], text: str) -> float:
    profile = candidate.get("profile", {})
    jobs = candidate.get("career_history", []) or []
    industries = [norm(profile.get("current_industry"))] + [norm(j.get("industry")) for j in jobs]
    product_hits = sum(1 for industry in industries if industry in PRODUCT_INDUSTRIES)
    product_score = clamp(product_hits / max(1, min(3, len(industries))))
    product_text = unique_hits(
        text,
        (
            "product",
            "users",
            "recruiter",
            "marketplace",
            "search platform",
            "candidate",
            "saas",
            "scale",
        ),
    )
    return clamp(0.65 * product_score + 0.35 * min(len(product_text), 4) / 4)


def score_education(candidate: Dict[str, Any]) -> float:
    best = 0.25
    for edu in candidate.get("education", []) or []:
        field = norm(edu.get("field_of_study"))
        degree = norm(edu.get("degree"))
        tier = norm(edu.get("tier"))
        field_score = 0.0
        if any(x in field for x in ("computer science", "machine learning", "data science", "artificial intelligence")):
            field_score = 0.75
        elif any(x in field for x in ("mathematics", "statistics", "information technology")):
            field_score = 0.55
        elif "electronics" in field:
            field_score = 0.35
        degree_bonus = 0.10 if any(x in degree for x in ("m.tech", "m.e", "m.sc", "ph.d", "phd")) else 0.0
        tier_bonus = {"tier_1": 0.15, "tier_2": 0.10, "tier_3": 0.04}.get(tier, 0.0)
        best = max(best, clamp(field_score + degree_bonus + tier_bonus))
    return best


def anomaly_penalty(
    candidate: Dict[str, Any],
    title_score: float,
    text_score: float,
    skill_score: float,
    matched_skills: Sequence[str],
) -> Tuple[float, List[str]]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    jobs = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    years = float(profile.get("years_of_experience") or 0)
    text = candidate_text(candidate).lower()

    penalty = 0.0
    concerns: List[str] = []

    expert_zero = sum(
        1
        for skill in skills
        if norm(skill.get("proficiency")) == "expert" and int(skill.get("duration_months") or 0) == 0
    )
    if expert_zero >= 3:
        penalty += 0.35
        concerns.append(f"{expert_zero} expert skills list zero months")

    expert_count = sum(1 for skill in skills if norm(skill.get("proficiency")) == "expert")
    advanced_count = sum(1 for skill in skills if norm(skill.get("proficiency")) in {"advanced", "expert"})
    if years < 4 and advanced_count >= 8:
        penalty += 0.18
        concerns.append("seniority claimed ahead of experience")

    core_skill_count = len([s for s in skills if norm(s.get("name")) in CORE_SKILL_WEIGHTS])
    if core_skill_count >= 7 and title_score < 0.35 and text_score < 0.35:
        penalty += 0.32
        concerns.append("AI keyword-heavy skills without matching career evidence")

    hype_count = len([s for s in skills if norm(s.get("name")) in HYPE_ONLY_SKILLS])
    if hype_count and text_score < 0.30 and core_skill_count < 4:
        penalty += 0.10
        concerns.append("framework/prompt signal is not backed by IR experience")

    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    salary_min = float(salary.get("min") or 0)
    salary_max = float(salary.get("max") or 0)
    if salary_min and salary_max and salary_min > salary_max:
        penalty += 0.15
        concerns.append("salary range is internally inconsistent")

    durations = sum(int(j.get("duration_months") or 0) for j in jobs)
    if durations and abs(durations / 12.0 - years) > 3.0:
        penalty += 0.08
        concerns.append("career duration and profile experience diverge")

    short_jobs = [int(j.get("duration_months") or 0) for j in jobs if int(j.get("duration_months") or 0) > 0]
    if len(short_jobs) >= 4 and sum(short_jobs) / len(short_jobs) < 20 and years >= 7:
        penalty += 0.06
        concerns.append("many short tenures")

    all_consulting = bool(jobs) and all(
        norm(j.get("company")) in CONSULTING_COMPANIES or norm(j.get("industry")) in {"it services", "consulting"}
        for j in jobs
    )
    if all_consulting and text_score < 0.45:
        penalty += 0.16
        concerns.append("mostly services/consulting background")

    current_title = norm(profile.get("current_title"))
    if any(x in current_title for x in ("research", "computer vision", "speech", "robotics")):
        if not any(x in text for x in ("retrieval", "ranking", "recommendation", "search")):
            penalty += 0.14
            concerns.append("primary domain is away from NLP/IR")

    if "langchain" in {norm(s.get("name")) for s in skills} and years < 3:
        penalty += 0.07
        concerns.append("recent framework exposure is not senior AI evidence")

    if expert_count >= 10 and years < 6:
        penalty += 0.20
        concerns.append("too many expert claims for experience level")

    return clamp(penalty, 0.0, 0.75), concerns[:3]


def score_candidate(candidate: Dict[str, Any]) -> FeatureBreakdown:
    profile = candidate.get("profile", {})
    text = candidate_text(candidate)
    title_score = score_titles(candidate)
    text_score, matched_terms = score_text(text)
    skill_score, matched_skills = score_skills(candidate, text)
    experience_score = score_experience(float(profile.get("years_of_experience") or 0))
    location_score = score_location(candidate)
    behavior_score = score_behavior(candidate)
    product_score = score_product_context(candidate, text)
    education_score = score_education(candidate)
    penalty, concerns = anomaly_penalty(candidate, title_score, text_score, skill_score, matched_skills)

    base = (
        0.22 * title_score
        + 0.25 * text_score
        + 0.17 * skill_score
        + 0.10 * experience_score
        + 0.09 * product_score
        + 0.08 * location_score
        + 0.04 * education_score
        + 0.05 * behavior_score
    )
    availability_multiplier = 0.72 + 0.38 * behavior_score
    final = clamp(base * availability_multiplier - penalty)

    return FeatureBreakdown(
        score=final,
        title_score=title_score,
        text_score=text_score,
        skill_score=skill_score,
        experience_score=experience_score,
        location_score=location_score,
        behavior_score=behavior_score,
        product_score=product_score,
        education_score=education_score,
        anomaly_penalty=penalty,
        matched_terms=matched_terms,
        matched_skills=matched_skills,
        concerns=concerns,
    )


def best_evidence_job(candidate: Dict[str, Any], features: FeatureBreakdown) -> Dict[str, Any]:
    jobs = candidate.get("career_history", []) or []
    if not jobs:
        return {}
    terms = [norm(t) for t in features.matched_terms[:8]]
    scored = []
    for idx, job in enumerate(jobs):
        text = norm(" ".join([job.get("title", ""), job.get("industry", ""), job.get("description", "")]))
        hits = sum(1 for term in terms if term and term in text)
        title_boost = title_relevance(job.get("title", ""))
        product_boost = 0.4 if norm(job.get("industry")) in PRODUCT_INDUSTRIES else 0.0
        current_boost = 0.3 if job.get("is_current") else 0.0
        scored.append((hits + title_boost + product_boost + current_boost, -idx, job))
    return max(scored, key=lambda x: (x[0], x[1]))[2]


def skill_phrase(candidate: Dict[str, Any], features: FeatureBreakdown) -> str:
    wanted = {norm(name) for name in features.matched_skills}
    details = []
    for skill in candidate.get("skills", []) or []:
        if norm(skill.get("name")) in wanted:
            prof = norm(skill.get("proficiency")) or "listed"
            months = int(skill.get("duration_months") or 0)
            if months:
                details.append(f"{prof} {skill.get('name')} ({months}m)")
            else:
                details.append(f"{prof} {skill.get('name')}")
        if len(details) >= 3:
            break
    if details:
        return ", ".join(details)
    if features.matched_skills:
        return ", ".join(features.matched_skills[:3])
    return ""


def logistics_phrase(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    location = profile.get("location") or profile.get("country") or "unknown location"
    country = norm(profile.get("country"))
    willing = bool(signals.get("willing_to_relocate"))
    mode = norm(signals.get("preferred_work_mode"))

    if country == "india":
        loc_l = norm(location)
        if any(city in loc_l for city in ("pune", "noida")):
            return f"already in a preferred office market ({location})"
        if any(city in loc_l for city in TIER1_CITIES):
            return f"based in a target Indian metro ({location})"
        if willing:
            return f"India-based and willing to relocate from {location}"
        return f"India-based in {location}"

    if willing:
        return f"outside India ({location}) but marked willing to relocate"
    if mode == "remote":
        return f"outside India ({location}) and remote-only, which is a location caveat"
    return f"outside India ({location}), so location is a caveat"


def behavior_phrase(signals: Dict[str, Any]) -> str:
    response = float(signals.get("recruiter_response_rate") or 0)
    notice = int(signals.get("notice_period_days") or 0)
    last_active = signals.get("last_active_date", "unknown")
    interview = float(signals.get("interview_completion_rate") or 0)
    saved = int(signals.get("saved_by_recruiters_30d") or 0)

    positives = []
    if response >= 0.75:
        positives.append(f"very responsive to recruiters ({response:.2f})")
    elif response >= 0.50:
        positives.append(f"solid recruiter response ({response:.2f})")
    else:
        positives.append(f"recruiter response is only {response:.2f}")

    if notice <= 30:
        positives.append(f"short {notice}d notice")
    elif notice <= 60:
        positives.append(f"manageable {notice}d notice")
    else:
        positives.append(f"{notice}d notice")

    if interview >= 0.80:
        positives.append(f"{interview:.2f} interview completion")
    elif saved >= 8:
        positives.append(f"{saved} recruiter saves in 30d")
    else:
        positives.append(f"last active {last_active}")

    return ", ".join(positives)


def rank_band(rank: int) -> str:
    if rank <= 10:
        return "top"
    if rank <= 35:
        return "strong"
    if rank <= 70:
        return "middle"
    return "borderline"


def rank_opening(rank: int) -> str:
    band = rank_band(rank)
    if band == "top":
        return "standout top-10"
    if band == "strong":
        return "strong top-50"
    if band == "middle":
        return "solid but less dominant"
    return "top-100 inclusion"


def concern_phrase(candidate: Dict[str, Any], features: FeatureBreakdown, rank: int) -> str:
    signals = candidate.get("redrob_signals", {}) or {}
    response = float(signals.get("recruiter_response_rate") or 0)
    notice = int(signals.get("notice_period_days") or 0)
    applications = int(signals.get("applications_submitted_30d") or 0)
    active = score_activity(signals)
    if features.concerns:
        return features.concerns[0]
    if notice > 90:
        return f"{notice}d notice period raises closing risk"
    if response < 0.25:
        return f"low recruiter response rate ({response:.2f})"
    if active < 0.40:
        return f"recent activity is weak (last active {signals.get('last_active_date', 'unknown')})"
    if features.location_score < 0.35:
        return "location/logistics are weaker than the core technical fit"
    if features.title_score < 0.70 and features.text_score >= 0.65:
        return "title is adjacent, but the career text carries the fit"
    if rank > 70 and applications == 0:
        return "passive candidate signal is weaker than higher-ranked profiles"
    if rank > 70:
        return "good enough for the top 100, but the evidence is less distinctive than higher ranks"
    if rank > 35 and features.behavior_score < 0.58:
        return "availability signals are only moderate"
    return ""


def reason_for(candidate: Dict[str, Any], features: FeatureBreakdown, rank: int) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    title = profile.get("current_title", "Candidate")
    years = float(profile.get("years_of_experience") or 0)
    job = best_evidence_job(candidate, features)
    company = job.get("company") or profile.get("current_company", "their current company")
    job_title = job.get("title") or title
    industry = job.get("industry") or profile.get("current_industry", "")
    duration = int(job.get("duration_months") or 0)
    terms = ", ".join(features.matched_terms[:3]) if features.matched_terms else "production AI/search work"
    skills = skill_phrase(candidate, features)
    logistics = logistics_phrase(candidate)
    behavior = behavior_phrase(signals)
    concern = concern_phrase(candidate, features, rank)
    opening = rank_opening(rank)
    band = rank_band(rank)
    style = (sum(ord(ch) for ch in candidate.get("candidate_id", "")) + rank) % 6

    if style == 0:
        first = (
            f"Rank {rank} is a {opening} match: {title} with {years:.1f} yrs total, and the best evidence in a "
            f"{job_title} stint at {company}"
        )
        if duration:
            first += f" ({duration} months)"
        first += f" covering {terms}."
        second = f"The supporting signals are {skills or 'more career-led than skill-list-led'}, {logistics}, and {behavior}."
    elif style == 1:
        first = (
            f"I would keep this as a {opening} profile because the experience is not just AI keywords: "
            f"{job_title} at {company} in {industry or 'a product context'} maps to {terms}."
        )
        second = (
            f"They have {years:.1f} yrs overall, {skills or 'credible career-history evidence'}, "
            f"plus {behavior}; {logistics}."
        )
    elif style == 2:
        first = (
            f"The fit at rank {rank} comes from search/ranking-style work rather than a generic ML label: "
            f"{terms} appears in the {job_title} work at {company}."
        )
        second = (
            f"With {years:.1f} yrs, {skills or 'limited but relevant listed skills'}, {behavior}, and {logistics}, "
            f"the profile lines up with Redrob's founding AI-engineer brief."
        )
    elif style == 3:
        first = (
            f"{title}, {years:.1f} yrs, is a {opening} fit for the role's retrieval/ranking mandate; "
            f"the strongest evidence is {job_title} work at {company} around {terms}."
        )
        second = f"Skills/signals check out via {skills or 'career text over keyword count'}, with {behavior}; {logistics}."
    elif style == 4:
        first = (
            f"This profile earns rank {rank} mainly on JD-critical evidence: {terms} from "
            f"{job_title} work at {company}, not generic AI interest."
        )
        second = (
            f"The concrete facts are {title}, {years:.1f} yrs, {skills or 'career-backed AI/search experience'}, "
            f"and Redrob availability signals of {behavior}; {logistics}."
        )
    else:
        if band == "borderline":
            first = (
                f"This is a top-100 rather than top-tier pick: {title} with {years:.1f} yrs has enough "
                f"retrieval/ranking relevance through {job_title} work at {company} ({terms})."
            )
        else:
            first = (
                f"The profile clears the JD bar because {job_title} work at {company} gives direct evidence of {terms}."
            )
        second = (
            f"Named skills/signals supporting the rank: {skills or 'career evidence instead of a long skill list'}; "
            f"{behavior}; {logistics}."
        )

    if concern:
        label = "Main caveat" if band in {"top", "strong"} else "Reason it is not ranked higher"
        second += f" {label}: {concern}."
    return first + " " + second


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = norm(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def iter_candidates(path: str | Path) -> Iterator[Dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with open(source, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return
    with open(source, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rank_candidates(path: str | Path, limit: int = TOP_K) -> List[Dict[str, Any]]:
    heap: List[Tuple[float, str, Dict[str, Any], FeatureBreakdown]] = []
    for candidate in iter_candidates(path):
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        features = score_candidate(candidate)
        item = (features.score, cid, candidate, features)
        if len(heap) < max(limit * 5, 500):
            heapq.heappush(heap, item)
        else:
            if (features.score, cid) > (heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, item)

    ordered = sorted(heap, key=lambda x: (-round(x[0], 4), x[1]))[:limit]
    rows = []
    for rank, (score, _cid, candidate, features) in enumerate(ordered, start=1):
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rank": rank,
                "score": f"{score * 100:.2f}",
                "reasoning": reason_for(candidate, features, rank),
                "_features": features,
            }
        )
    return rows


def write_submission(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "reasoning": row["reasoning"],
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates and emit a ranked CSV.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl, candidates.jsonl.gz, or a JSON list.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=TOP_K, help="Number of rows to output. Defaults to 100.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    limit = min(max(1, args.limit), TOP_K)
    rows = rank_candidates(args.candidates, limit=limit)
    write_submission(rows, args.out)
    print(f"Wrote {len(rows)} ranked candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
