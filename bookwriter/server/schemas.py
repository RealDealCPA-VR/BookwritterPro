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
    premise: str = Field(..., min_length=1)
    chapters: Optional[int] = None
    words_per_chapter: int = 2000
    title: Optional[str] = None
    genre: Optional[str] = None
    guidance: Optional[str] = None
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
    only: Optional[List[int]] = None
    restart: bool = False


class KdpRequest(BaseModel):
    author_first: str
    author_last: str
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
    list_price: float
    marketplace: str = "US"
    paper: str = "white"


class MarketingRequest(BaseModel):
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


class ProfileStage(BaseModel):
    model: str
    effort: str
    thinking: bool = True


class ProfileInfo(BaseModel):
    name: str
    stages: Dict[str, Any]
    prices: Dict[str, Dict[str, float]]


class ProfilesResponse(BaseModel):
    default: str
    profiles: List[ProfileInfo]
