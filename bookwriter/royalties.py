"""Deterministic KDP royalty estimator — pure Python, no deps.

Two products, one estimate each:

  EBOOK — Kindle's 35% vs 70% royalty plans. The 70% plan is only *eligible* when the
  list price is between $2.99 and $9.99 (US); outside that band only 35% is available.
  The 70% plan also subtracts a per-MB delivery fee (~$0.06/MB in the US). We report the
  applicable royalty AND the alternate plan so the author can compare.

  PAPERBACK — KDP prints on demand and pays 60% of list price minus a printing cost. The
  printing cost follows KDP's documented tiered formula (US, B&W, white paper); we encode
  it as clearly-labelled APPROXIMATE constants (a fixed per-book charge plus a per-page
  charge above a page threshold). Paperback royalty = 0.60 * list_price - printing_cost,
  floored at 0.

All money is rounded to cents. These are ESTIMATES — confirm in the KDP UI.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Ebook constants
# ---------------------------------------------------------------------------
EBOOK_70_MIN = 2.99             # inclusive lower bound for 70% eligibility (US)
EBOOK_70_MAX = 9.99             # inclusive upper bound for 70% eligibility (US)
DELIVERY_FEE_PER_MB_US = 0.06   # APPROXIMATE US delivery fee on the 70% plan ($/MB)
MIN_DELIVERY_FEE = 0.0          # KDP has no minimum, but the file is billed in MB

# ---------------------------------------------------------------------------
# Paperback printing cost — US, B&W ink, white paper. APPROXIMATE; KDP's exact
# figures change over time, so these are documented constants you can tune.
#   fixed per-book charge        ~ $1.00
#   per-page charge ABOVE the    ~ $0.012/page  (pages 1..threshold are "free"
#     free threshold                under the fixed charge)
#   free page threshold          110 pages
# Below ~110 pages KDP charges only the flat fee; above it, the per-page charge
# applies to every page (this matches KDP's published B&W formula closely enough
# for a planning estimate).
# ---------------------------------------------------------------------------
PB_FIXED_COST_US = 1.00
PB_PER_PAGE_US = 0.012
PB_FREE_PAGE_THRESHOLD = 110

PAPERBACK_ROYALTY_RATE = 0.60   # KDP pays 60% of list price for paperbacks


def _money(x: float) -> float:
    """Round to cents (half-up enough for planning; banker's rounding is fine here)."""
    return round(float(x) + 1e-9, 2)


def estimate_page_count(graph, words_per_page: int = 300) -> int:
    """Estimate paperback page count from total chapter words (min 24)."""
    total = 0
    for n in getattr(graph, "chapters", {}):
        rec = graph.chapters[n]
        wc = getattr(rec, "word_count", 0) or 0
        if not wc and getattr(rec, "text", ""):
            wc = len(rec.text.split())
        total += wc
    wpp = words_per_page or 300
    return max(24, int(round(total / wpp)) or 24)


def _paperback_printing_cost(page_count: int, paper: str = "white") -> float:
    """APPROXIMATE US B&W print cost. Cream costs marginally more per page."""
    per_page = PB_PER_PAGE_US
    if (paper or "white").lower() == "cream":
        per_page = PB_PER_PAGE_US + 0.001  # cream is slightly pricier per page
    pages_charged = max(0, page_count - PB_FREE_PAGE_THRESHOLD)
    if page_count <= PB_FREE_PAGE_THRESHOLD:
        cost = PB_FIXED_COST_US
    else:
        cost = PB_FIXED_COST_US + pages_charged * per_page
    return _money(cost)


def estimate_royalties(
    *,
    list_price: float,
    marketplace: str = "US",
    page_count: int,
    trim: Tuple[float, float] = (6.0, 9.0),
    paper: str = "white",
    ebook_file_mb: float = 1.0,
) -> Dict[str, Any]:
    """Estimate ebook + paperback per-sale royalties for a list price.

    Returns ``{ebook:{...}, paperback:{...}, assumptions:[...], note:...}``.
    Deterministic and dependency-free. Money rounded to cents.
    """
    price = max(0.0, float(list_price))
    mb = max(0.0, float(ebook_file_mb))
    pages = max(1, int(page_count))

    # ---- EBOOK ------------------------------------------------------------
    eligible_70 = EBOOK_70_MIN <= price <= EBOOK_70_MAX
    delivery_fee = _money(max(MIN_DELIVERY_FEE, mb * DELIVERY_FEE_PER_MB_US))

    royalty_70 = _money(max(0.0, 0.70 * (price - delivery_fee))) if eligible_70 else None
    royalty_35 = _money(max(0.0, 0.35 * price))

    if eligible_70:
        plan = "70%"
        ebook_royalty = royalty_70
        alternate = {"plan": "35%", "royalty_per_sale": royalty_35,
                     "delivery_fee": 0.0}
    else:
        plan = "35%"
        ebook_royalty = royalty_35
        # show what 70% *would* pay if it were eligible, for comparison context
        alternate = {
            "plan": "70%",
            "eligible": False,
            "reason": f"price must be ${EBOOK_70_MIN}-${EBOOK_70_MAX} for the 70% plan",
            "royalty_per_sale": _money(max(0.0, 0.70 * (price - delivery_fee))),
            "delivery_fee": delivery_fee,
        }

    ebook = {
        "list_price": _money(price),
        "plan": plan,
        "royalty_per_sale": ebook_royalty,
        "delivery_fee": delivery_fee if plan == "70%" else 0.0,
        "eligible_for_70": eligible_70,
        "alternate_plan": alternate,
    }

    # ---- PAPERBACK --------------------------------------------------------
    printing_cost = _paperback_printing_cost(pages, paper=paper)
    pb_royalty = _money(max(0.0, PAPERBACK_ROYALTY_RATE * price - printing_cost))
    paperback = {
        "list_price": _money(price),
        "trim": [float(trim[0]), float(trim[1])],
        "paper": (paper or "white").lower(),
        "page_count": pages,
        "royalty_rate": PAPERBACK_ROYALTY_RATE,
        "printing_cost": printing_cost,
        "royalty_per_sale": pb_royalty,
        "below_cost": (PAPERBACK_ROYALTY_RATE * price) < printing_cost,
    }

    assumptions = [
        f"Marketplace: {marketplace} (figures use US constants).",
        f"Ebook 70% plan eligible only for list price ${EBOOK_70_MIN}-${EBOOK_70_MAX}; "
        f"otherwise 35%.",
        f"Ebook 70% plan subtracts an APPROXIMATE delivery fee of "
        f"${DELIVERY_FEE_PER_MB_US}/MB (file ~{mb} MB).",
        f"Paperback royalty = {PAPERBACK_ROYALTY_RATE:.0%} x list price - printing cost.",
        f"Printing cost (US, B&W, {(paper or 'white').lower()} paper) APPROXIMATE: "
        f"${PB_FIXED_COST_US:.2f} fixed + ${PB_PER_PAGE_US}/page above "
        f"{PB_FREE_PAGE_THRESHOLD} pages.",
    ]

    return {
        "ebook": ebook,
        "paperback": paperback,
        "assumptions": assumptions,
        "note": "estimates — confirm in KDP",
    }
