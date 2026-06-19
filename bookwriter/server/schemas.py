"""Pydantic request/response models for the HTTP API.

Kept deliberately permissive on the response side: the bible / graph / cost
payloads are produced by the core package's own ``to_dict`` / snapshot helpers
and forwarded verbatim, so they are typed as open dicts rather than re-modelled
(which would risk drifting from the core).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class CreateBookRequest(BaseModel):
    # Bounds keep a single request from triggering runaway token spend / huge
    # outputs (and a real cost-DoS if the app is ever exposed beyond localhost).
    premise: str = Field(..., min_length=1, max_length=20000)
    chapters: Optional[int] = Field(None, ge=1, le=200)
    words_per_chapter: int = Field(2000, ge=100, le=20000)
    title: Optional[str] = Field(None, max_length=300)
    genre: Optional[str] = Field(None, max_length=200)
    guidance: Optional[str] = Field(None, max_length=8000)
    profile: str = "balanced"
    mock: bool = False
    use_cache: bool = True
    run_continuity_check: bool = True
    # Per-book LLM backend selection (None -> server default from env). `model` is
    # an optional "Text model" applied to the prose stages for this book.
    provider: Optional[str] = None
    model: Optional[str] = None
    # Generate one inline illustration per chapter during the full-book write,
    # using the configured image provider (default Pixio). Best-effort.
    chapter_images: bool = False


class WriteRequest(BaseModel):
    only: Optional[List[int]] = Field(None, max_length=500)
    restart: bool = False


class SettingsUpdate(BaseModel):
    # Map of managed env-var name -> value ("" / null clears the override).
    values: Dict[str, Optional[str]] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    kind: str = "llm"            # "llm" | "image"
    provider: Optional[str] = None


class KdpRequest(BaseModel):
    # Optional so "Auto-fill with AI" works before an author is entered (the book
    # summary carries no author); generate_kdp_metadata tolerates blanks.
    author_first: str = ""
    author_last: str = ""
    language: str = "English"
    subtitle: Optional[str] = None
    series: str = ""
    series_part: str = ""
    edition: str = ""
    contributors: List[Dict[str, str]] = Field(default_factory=list)
    publishing_rights: str = "owned"     # 'owned' | 'public_domain'
    sexually_explicit: bool = False
    reading_age_min: str = ""
    reading_age_max: str = ""
    cover_svg: Optional[str] = None
    mock: Optional[bool] = None


class PricingRequest(BaseModel):
    list_price: float = Field(..., ge=0, le=100000)
    marketplace: str = "US"
    paper: str = "white"


class MarketingRequest(BaseModel):
    mock: Optional[bool] = None


class CoverRequest(BaseModel):
    # Optional typography overrides so the generated cover matches the live form;
    # all fall back to the saved/planned metadata when omitted.
    title: Optional[str] = Field(None, max_length=300)
    subtitle: Optional[str] = Field(None, max_length=300)
    author_first: Optional[str] = Field(None, max_length=120)
    author_last: Optional[str] = Field(None, max_length=120)
    mock: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class BookSummary(BaseModel):
    id: str
    title: str
    logline: str = ""
    genre: str = ""
    chapters_total: int = 0
    chapters_written: int = 0
    words: int = 0
    created_at: str = ""
    profile: str = "balanced"
    mock: bool = False


class BooksResponse(BaseModel):
    books: List[BookSummary]


class CreateBookResponse(BaseModel):
    book: BookSummary
    bible: Dict[str, Any]


class ChapterOutlineItem(BaseModel):
    number: int
    title: str
    act: int = 1
    written: bool = False
    word_count: int = 0


class BookDetailResponse(BaseModel):
    book: BookSummary
    bible: Dict[str, Any]
    chapters: List[ChapterOutlineItem]
    cost: Optional[Dict[str, Any]] = None


class ChapterResponse(BaseModel):
    number: int
    title: str
    text: str = ""
    word_count: int = 0
    synopsis_line: str = ""
    fingerprint: str = ""
    written: bool = False
    plan: Dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    characters: List[Dict[str, Any]] = Field(default_factory=list)
    locations: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    threads: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    synopsis: List[str] = Field(default_factory=list)


class CostResponse(BaseModel):
    snapshot: Optional[Dict[str, Any]] = None
    report: str = ""


class ManuscriptResponse(BaseModel):
    markdown: str = ""
    words: int = 0


class KdpResponse(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    listing: str = ""
    paths: Dict[str, str] = Field(default_factory=dict)


class KdpMetadataResponse(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    has_api_key: bool = False
    provider: str = "anthropic"


class WriteStartedResponse(BaseModel):
    status: str = "started"


class DeletedResponse(BaseModel):
    status: str = "deleted"


class ProfileInfo(BaseModel):
    # ``stages`` is a heterogeneous map ({plan: str, ..., check: {model, effort}})
    # emitted verbatim by the service, so it stays Dict[str, Any] rather than a
    # fixed per-stage model.
    name: str
    stages: Dict[str, Any]
    prices: Dict[str, Dict[str, float]]


class ProfilesResponse(BaseModel):
    default: str
    profiles: List[ProfileInfo]
