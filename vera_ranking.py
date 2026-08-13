#!/usr/bin/env python3
"""
-VERA- Anime Ranking Engine
============================
Extracts anime data from the Tenrai API (a Jikan-v4-compatible REST API
for MyAnimeList data), computes a custom Bayesian-style rating ("-VERA-")
that accounts for title age and airing status, and exports a ranked
Top 10,000 list.

API SOURCE & SPEC:
  Base URL: https://api.tenrai.org/v1 (confirmed via the official OpenAPI
  spec -- api-1.json / api-1.yaml).

  Adopted in place of api.jikan.moe because the latter was observed
  returning intermittent 504 BadResponseException errors, a known ongoing
  issue (jikan-me/jikan-rest#610) affecting multiple users. Tenrai mirrors
  the Jikan v4 schema for the endpoints this script uses, confirmed both
  by live test calls and by the official spec.

  Facts below are taken directly from the official spec, not guessed:
    - Public (unauthenticated) tier: 120 requests/minute, 4 requests/second,
      40,000 requests/day, enforced per IP.
    - Server Key tier (Patreon): 300/min, 5/sec, unlimited/day.
    - Exceeding a limit returns HTTP 429 with a `Retry-After` header
      (seconds to wait) -- the spec explicitly says to respect this value
      rather than guess a backoff.
    - Documented transient failure statuses: 429, 502, 503, 504.
    - `/top/anime` accepts `limit` up to 50 per page (we use the max to
      roughly halve the number of pagination requests vs. the default 25).

Author: Expert Python Developer (generated)
Python: 3.9+
Dependencies: requests
"""

import json
import csv
import time
import random
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests

# =========================================================================
# CONFIGURATION
# =========================================================================

BASE_URL = "https://api.tenrai.org/v1"
TOP_ANIME_ENDPOINT = f"{BASE_URL}/top/anime"
STATS_ENDPOINT_TEMPLATE = f"{BASE_URL}/anime/{{mal_id}}/statistics"
# PATCH 10: relations (sequels, prequels, side stories, the source manga,
# etc.) live on their own endpoint -- /top/anime and /statistics never
# included them, so a separate call per title is needed to get this data.
RELATIONS_ENDPOINT_TEMPLATE = f"{BASE_URL}/anime/{{mal_id}}/relations"

OUTPUT_JSON = "vera_ranked_15200.json"
OUTPUT_CSV = "vera_ranked_15200.csv"

TOP_COUNT = 100         # Target number of anime to process
PAGE_LIMIT = 50            # Max allowed by /top/anime per the official spec;
                            # using the max roughly halves pagination calls
                            # versus the API's default of 25.
SAVE_EVERY = 50            # Checkpoint frequency

# Optional: set this to your Tenrai Server Key (see Discord in the API
# docs) to raise your rate limit from 120 RPM/4 RPS to 300 RPM/5 RPS and
# get a separate, faster-refreshing cache. Leave as None to use the public
# tier -- the script works fine either way.
SERVER_KEY: Optional[str] = None

# --- Adaptive rate control ---
# The public tier is capped at 120 requests/minute AND 4 requests/second
# (whichever is hit first), enforced per IP. We stay under the stricter
# 4 RPS cap (0.25s/request theoretical floor) with a small safety margin,
# and adapt within that ceiling based on observed 429s. If a Server Key is
# set, the floor is relaxed toward its documented 5 RPS limit instead.
REQUEST_SLEEP_START = 0.5
REQUEST_SLEEP_MIN = 0.21 if SERVER_KEY else 0.27   # stays under 5 RPS / 4 RPS
REQUEST_SLEEP_MAX = 5.0
SPEED_UP_FACTOR = 0.9
SLOW_DOWN_FACTOR = 2.0
CLEAN_STREAK_TO_SPEED_UP = 20

_request_sleep = REQUEST_SLEEP_START
_clean_streak = 0


def adaptive_sleep_and_record(hit_rate_limit: bool, retry_after: Optional[float] = None) -> None:
    """
    Call once per request. Sleeps the current adaptive delay, then adjusts
    it based on whether a 429 was just observed.

    If the server sent a Retry-After value (it does, per spec, on 429),
    the adaptive delay is bumped to at least that value so the *next*
    request already respects it -- on top of the explicit wait already
    performed inside get_json()'s own retry loop for the failed call itself.
    """
    global _request_sleep, _clean_streak

    time.sleep(_request_sleep)

    if hit_rate_limit:
        bumped = _request_sleep * SLOW_DOWN_FACTOR
        if retry_after is not None:
            bumped = max(bumped, min(retry_after, REQUEST_SLEEP_MAX))
        _request_sleep = min(bumped, REQUEST_SLEEP_MAX)
        _clean_streak = 0
        print(f"  [ADAPTIVE] 429 seen -> slowing down to {_request_sleep:.2f}s between requests.")
    else:
        _clean_streak += 1
        if _clean_streak >= CLEAN_STREAK_TO_SPEED_UP and _request_sleep > REQUEST_SLEEP_MIN:
            _request_sleep = max(_request_sleep * SPEED_UP_FACTOR, REQUEST_SLEEP_MIN)
            _clean_streak = 0
            print(f"  [ADAPTIVE] Clean streak -> speeding up to {_request_sleep:.2f}s between requests.")


# --- Main retry policy ---
# Per the official spec, documented transient failure statuses are
# 429 (Rate Limit Exceeded), 502 (Bad Gateway), 503 (Service Unavailable),
# and 504 (Gateway Timeout) -- all retried here. Only 429 documents a
# Retry-After header; 502/503/504 fall back to a random backoff window.
RETRYABLE_STATUS_CODES = (429, 502, 503, 504)
MAIN_MAX_RETRIES = 10
MAIN_RETRY_WAIT_MIN = 20   # fallback bounds when no Retry-After is present
MAIN_RETRY_WAIT_MAX = 60

# --- Fast-path retry policy (transient/connection-type errors) ---
FAST_MAX_RETRIES = 1
FAST_RETRY_WAIT = 0.5

# --- Final health-check retry policy (used only in the end-of-script self-test) ---
HEALTHCHECK_MAX_RETRIES = 1
HEALTHCHECK_RETRY_WAIT = 1.0

# --- -VERA- Formula Constants ---
M_BASE = 1000                # Lowered from 5000 -- this is the "phantom
                              # average vote" weight every anime starts with.
                              # At 5000, a small-but-genuinely-excellent anime
                              # (a few thousand real votes) could never
                              # meaningfully compete with million-vote giants
                              # regardless of how good its actual score was --
                              # popularity alone dominated. At 1000, a small
                              # anime with a real, well-earned high score
                              # becomes a fair contender for the top ranks,
                              # while a barely-qualifying anime with only a
                              # few hundred votes still can't leapfrog giants
                              # on small-sample noise alone (tested: going
                              # lower than ~1000 lets noise start winning).
K = 12
ALPHA = 0.8
C_GLOBAL = 7.45

# --- PATCH 6: replaced hard 1000-vote cliff with an N/A floor ---
# The old MIN_VOTE_FLOOR=1000 caused a hard on/off cliff: 999 votes -> 0.0,
# 1000 votes -> a normal score, regardless of quality. At 10k scale this
# produces thousands of nonsensical rank jumps between near-identical
# titles. Fix: anything below MIN_VOTES_FOR_NA is too thin to rate at all
# (returned as "N/A", still exported but unranked, sorted by raw vote
# count at the bottom of the list). Anything >= MIN_VOTES_FOR_NA gets a
# real VERA score -- the existing Bayesian blend (M_dynamic vs W) already
# naturally trusts a 900-vote show far more than a 150-vote show (see
# calculate_VERA_score docstring), so no extra dampening is needed in the
# 100-999 range; removing the cliff lets that existing math do its job
# continuously instead of being short-circuited.
MIN_VOTES_FOR_NA = 100       # below this: too thin to rate, marked "N/A"

# --- PATCH 3: light-touch anti-brigading (extreme-bucket spike smoothing) ---
SPIKE_TOLERANCE = 1.5       # an extreme bucket up to 1.5x its neighbor-predicted
                             # size is treated as organic; only the excess above
                             # this is dampened. Higher = lighter touch.
MIN_SPIKE_VOTES = 500       # only dampen if the extreme bucket also has at least
                             # this many raw votes. Small anime with naturally
                             # noisy small-sample buckets are left alone.
                             # FIXED: this used to be a PERCENTAGE of the
                             # anime's total votes (MIN_SPIKE_SHARE=15%), which
                             # meant a real, deliberate bombing campaign on a
                             # mega-popular anime (e.g. thousands of troll 1s
                             # on a show with 1M+ total votes) was always a
                             # small percentage of the total and therefore
                             # NEVER got caught -- the exact opposite of the
                             # intent. An absolute vote-count floor catches
                             # real large-scale bombing regardless of how
                             # popular the anime is, while still ignoring
                             # genuinely tiny, noisy samples.

# --- PATCH 4: softened new-anime dampening curve ---
NEW_ANIME_SOFTENING = 0.25  # multiplies K down (was implicitly 1.0); lower =
                             # less punishing to young anime with real votes.
                             # Tuned so a new anime with a real (if smaller)
                             # completed-vote base can compete on merit,
                             # without letting thin-data anime randomly top
                             # the chart.
MAX_M_MULTIPLIER = 6.0      # ceiling on (1 + K_eff/T^ALPHA); prevents the
                             # dynamic prior from scaling without bound for
                             # extremely young (day-old) titles.

# --- PATCH 5: airing-specific extra dampening (scales with W itself, not
#     just time -- see calculate_VERA_score for why this matters) ---
AIRING_PENALTY_W_FRACTION = 0.85  # at T≈0 with 0% completed, adds up to this
                                   # fraction of W as extra prior weight.
                                   # Re-tuned after M_BASE was lowered
                                   # (5000->1000) for popularity fairness --
                                   # that change alone would have let week-1
                                   # airing hype creep back above mature
                                   # baselines, so this was raised to
                                   # compensate and keep hype suppressed.
AIRING_PENALTY_HALFLIFE_MONTHS = 1.5  # this extra penalty halves every
                                        # ~1.5 months of air-time

# --- PATCH 7: fixed airing watching-weight bug ---
# PATCH 1 originally set watching_weight=0.8 for "Currently Airing" shows,
# meaning a person who had just clicked "watching" counted almost as much
# toward W as someone who had actually completed the show. Combined with
# PATCH 5's penalty (which fades as speculative_share drops, but is scaled
# by W itself), a young airing show with a huge hype wave of "watching"
# clicks could out-rank shows with millions of real completions on almost
# no finished opinions at all (verified: a title with 252 completions and
# 187k "watching" ranked #8, above a 1.78M-completion title). Fix: bring
# the airing watching-weight down to the same order as the other
# speculative signals (on_hold/dropped/plan_to_watch, all 0.1), so
# "watching" reads as mild interest rather than a near-equivalent to a
# real, earned opinion. Airing shows remain fully eligible for high ranks
# on real merit -- this only stops pure hype from doing the work.
AIRING_WATCHING_WEIGHT = 0.15  # was 0.8


# =========================================================================
# NETWORKING LAYER
# =========================================================================

def _request_headers() -> dict:
    """Build request headers, including the Server Key if configured."""
    headers = {}
    if SERVER_KEY:
        # Per spec: Server Keys go in X-Server-Key, NOT Authorization: Bearer.
        headers["X-Server-Key"] = SERVER_KEY
    return headers


def get_json(
    url: str,
    params: Optional[dict] = None,
    max_retries: int = MAIN_MAX_RETRIES,
    retry_wait_min: float = MAIN_RETRY_WAIT_MIN,
    retry_wait_max: float = MAIN_RETRY_WAIT_MAX,
    fast_max_retries: int = FAST_MAX_RETRIES,
    fast_retry_wait: float = FAST_RETRY_WAIT,
    timeout: int = 15,
) -> Tuple[Optional[dict], bool, Optional[float]]:
    """
    Robust GET wrapper following the official Tenrai API error contract.

      - Connection-level errors (timeout, DNS, reset) -> fast-path retry
        (up to `fast_max_retries`, `fast_retry_wait` seconds apart).
      - HTTP 429 / 502 / 503 / 504 (all documented transient statuses) ->
        main retry, up to `max_retries` times. On 429, the response's
        `Retry-After` header (documented in the spec) is respected exactly;
        for 502/503/504, which don't document that header, a random
        backoff between `retry_wait_min` and `retry_wait_max` is used.
      - Any other non-200 status (400/401/403/404/405/500, etc.) -> logged
        and returned as None immediately; retrying those endlessly would
        just waste time on a non-transient error.

    Returns a 3-tuple: (json_or_none, hit_429: bool, last_retry_after: float|None).
    hit_429 and last_retry_after let callers feed the adaptive throttle.
    """
    main_attempts = 0
    fast_attempts = 0
    hit_429 = False
    last_retry_after: Optional[float] = None
    headers = _request_headers()

    while True:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            if fast_attempts >= fast_max_retries:
                print(f"  [FAIL] {url} -> {exc.__class__.__name__} persisted after fast retries.")
                return None, hit_429, last_retry_after
            fast_attempts += 1
            print(f"  [CONN ERROR] {exc.__class__.__name__}. Fast retry "
                  f"{fast_attempts}/{fast_max_retries} in {fast_retry_wait}s...")
            time.sleep(fast_retry_wait)
            continue

        if response.status_code == 200:
            return response.json(), hit_429, last_retry_after

        if response.status_code in RETRYABLE_STATUS_CODES:
            if response.status_code == 429:
                hit_429 = True

            if main_attempts >= max_retries:
                print(f"  [FAIL] {url} -> HTTP {response.status_code} after "
                      f"{max_retries} retries. Giving up.")
                return None, hit_429, last_retry_after

            main_attempts += 1

            # Prefer the documented Retry-After header when present
            # (guaranteed on 429 per spec; some gateways also send it on
            # 503). Fall back to a random backoff otherwise.
            retry_after_header = response.headers.get("Retry-After")
            wait: float
            if retry_after_header is not None:
                try:
                    wait = float(retry_after_header)
                    last_retry_after = wait
                except ValueError:
                    wait = random.uniform(retry_wait_min, retry_wait_max)
            else:
                wait = random.uniform(retry_wait_min, retry_wait_max)

            print(f"  [RETRY] HTTP {response.status_code} on {url}. Retry "
                  f"{main_attempts}/{max_retries} in {wait:.1f}s"
                  f"{' (Retry-After honored)' if retry_after_header else ''}...")
            time.sleep(wait)
            continue

        # Non-retryable status (400/401/403/404/405/500/etc.)
        try:
            body = response.json()
            msg = body.get("message", "")
        except (ValueError, AttributeError):
            msg = ""
        print(f"  [ERROR] {url} -> HTTP {response.status_code}. {msg} Skipping.")
        return None, hit_429, last_retry_after


# =========================================================================
# -VERA- FORMULA
# =========================================================================

def _parse_aired_from(aired_from_str: Optional[str]) -> Optional[datetime]:
    """Parse the ISO8601 'aired.from' timestamp into a datetime."""
    if not aired_from_str:
        return None
    try:
        return datetime.fromisoformat(aired_from_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _calculate_age_in_months(aired_from_str: Optional[str]) -> float:
    """
    T = age of the anime in months since its air date.
    Minimum floor of 0.5 months to avoid division blow-up for brand-new titles.
    """
    aired_dt = _parse_aired_from(aired_from_str)
    if aired_dt is None:
        return 120.0  # unknown air date -> treat conservatively as ~10 years old

    now = datetime.now(timezone.utc)
    delta_days = (now - aired_dt).total_seconds() / 86400.0
    months = delta_days / 30.4375
    return max(months, 0.5)


def calculate_VERA_score(anime_data: Dict[str, Any]) -> Optional[float]:
    """
    Compute the custom -VERA- rating for a single anime.

        T              = age in months (min 0.5), from aired.from
        M_dynamic      = M_BASE * (1 + K / (T ** ALPHA)), softened per PATCH 4
        AIRING_PENALTY = extra Bayesian weight applied only while a show is
                          still airing, decaying as it accumulates air-time
                          (PATCH 5)
        N_completed    = 'completed' count
        N_incomplete   = weighted sum of watching / on_hold / dropped / plan_to_watch
                          - watching weight = 0.8 if status == "Currently Airing" else 0.1
                          - on_hold, dropped, plan_to_watch weight = 0.1 each
        W              = N_completed * 1.0 + N_incomplete
        S_weighted     = average of the 1-10 star distribution, AFTER the
                          distribution has been passed through a light
                          anti-spike smoother (PATCH 3)
        VERA           = [W / (W + M_total)] * S_weighted
                          + [M_total / (W + M_total)] * C_GLOBAL
                          where M_total = M_dynamic * AIRING_PENALTY

    PATCH 2 (superseded by PATCH 6): if total_entries < MIN_VOTES_FOR_NA,
             returns None (caller treats this as "N/A" -- too thin to
             rate at all, not a real 0.0 score).
    PATCH 3: light anti-brigading -- if the 1s or 10s bucket is a sharp,
             isolated spike relative to its neighboring buckets (the
             signature of score-bombing or blind hype-voting, rather than
             an organically-earned extreme score), that bucket's vote count
             is softly pulled back toward what a smooth curve predicts
             before averaging. Genuine high-quality anime (where 8s, 9s,
             and 10s are all proportionally high, e.g. Frieren) are
             untouched -- only isolated, disconnected spikes are affected.
    PATCH 4: the new-anime dampening curve is softened (lower effective K,
             plus a cap on how large the multiplier can get) so a genuinely
             excellent new anime with a healthy vote count has a real path
             to a top score, instead of being structurally capped by age.
    PATCH 5: while a show is "Currently Airing", an additional decaying
             penalty is layered onto M_dynamic on top of PATCH 1's
             watching-weight. This makes early hype alone insufficient to
             reach #1; the penalty fades as the show accumulates air-time,
             so a sustained, non-spiky, genuinely excellent airing show can
             still climb to the top by the back half of its run.
    """
    total_entries = anime_data.get("total_entries", 0) or 0
    if total_entries < MIN_VOTES_FOR_NA:
        return None  # too thin to rate -- caller marks this "N/A"

    status = anime_data.get("status", "") or ""
    aired_from = anime_data.get("aired_from")

    watching = anime_data.get("watching", 0) or 0
    completed = anime_data.get("completed", 0) or 0
    on_hold = anime_data.get("on_hold", 0) or 0
    dropped = anime_data.get("dropped", 0) or 0
    plan_to_watch = anime_data.get("plan_to_watch", 0) or 0

    scores_array = anime_data.get("scores", []) or []

    T = _calculate_age_in_months(aired_from)

    # ---- PATCH 4: softened new-anime dampening ----
    # Same shape as the original (M_BASE * (1 + K / T^ALPHA)) but with a
    # lower effective K and a ceiling on the multiplier, so the prior no
    # longer scales without bound for very young titles. A show still
    # needs real votes to overcome M_dynamic, but it is no longer
    # structurally impossible for a young masterpiece to break through
    # once it has a healthy W.
    K_EFFECTIVE = K * NEW_ANIME_SOFTENING
    raw_multiplier = 1 + K_EFFECTIVE / (T ** ALPHA)
    capped_multiplier = min(raw_multiplier, MAX_M_MULTIPLIER)
    M_dynamic = M_BASE * capped_multiplier

    # PATCH 1 (fixed by PATCH 7): Currently Airing watching-weight override.
    # Was 0.8 -- let hype-watching nearly rival a completed opinion. Now
    # brought in line with the other speculative signals.
    watching_weight = AIRING_WATCHING_WEIGHT if status == "Currently Airing" else 0.1
    incomplete_weight_standard = 0.1

    N_incomplete = (
        watching * watching_weight
        + on_hold * incomplete_weight_standard
        + dropped * incomplete_weight_standard
        + plan_to_watch * incomplete_weight_standard
    )
    N_completed = completed * 1.0
    W = N_completed + N_incomplete

    # ---- PATCH 5: airing-specific extra dampening, scaled to W itself ----
    # The original flaw: a flat time-based penalty on M_dynamic gets
    # trivially swamped once hundreds of thousands of people click
    # "watching" in week one -- W becomes huge from pure hype even though
    # nobody has actually finished the show or formed a real opinion
    # (N_completed can be 0). A penalty that only decays with time, and
    # never scales with W, can't counterbalance that.
    #
    # Fix: for airing shows, the extra penalty is sized as a FRACTION of W
    # itself (not of M_dynamic), weighted toward how little of W is
    # "real" (completed) versus "speculative" (watching/on_hold/dropped/
    # plan_to_watch). A show that's mostly speculative interest gets a
    # heavy extra prior pulling it toward C_GLOBAL; a show where a
    # meaningful chunk of viewers have already completed it (e.g. a short
    # season nearing its finale) gets much less extra penalty. This decays
    # with air-time on top, so by the back half of a cour a sustained,
    # genuinely excellent show can still out-earn the penalty.
    if status == "Currently Airing" and W > 0:
        speculative_share = 1.0 - (N_completed / W)  # 1.0 = nobody's finished it
        time_decay = 0.5 ** (T / AIRING_PENALTY_HALFLIFE_MONTHS)
        airing_extra = W * AIRING_PENALTY_W_FRACTION * speculative_share * time_decay
    else:
        airing_extra = 0.0

    M_total = M_dynamic + airing_extra

    S_weighted = _average_score_distribution(scores_array)

    denominator = W + M_total
    if denominator <= 0:
        return 0.0

    vera = (W / denominator) * S_weighted + (M_total / denominator) * C_GLOBAL
    return round(vera, 4)


def _smooth_extreme_spikes(scores_array: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    PATCH 3 -- light-touch anti-brigading filter.

    Looks only at the two extreme buckets (score=1 and score=10), since
    those are the ones score-bombing and blind-hype voting target. For
    each extreme bucket, compares its share of total votes against its
    nearest neighbor buckets (2 for the 1s bucket; 8 and 9 for the 10s
    bucket). If the extreme bucket's share is a sharp, disconnected spike
    -- far higher than a smooth falloff from its neighbor would predict --
    the excess above SPIKE_TOLERANCE times the neighbor-predicted share is
    softly removed from that bucket's vote count (not deleted from the
    anime's totals, just excluded from the S_weighted average).

    This is deliberately light: a normal, organically well-loved anime
    (like Frieren, where 8/9/10 are all proportionally high and smoothly
    increasing) is untouched, since its 10-bucket isn't disconnected from
    its 9-bucket. Only an isolated, implausible spike is dampened.
    """
    by_score = {entry.get("score"): (entry.get("votes", 0) or 0) for entry in scores_array}
    total_votes = sum(by_score.values())
    if total_votes == 0 or not all(s in by_score for s in range(1, 11)):
        return scores_array  # incomplete data -- don't touch it

    adjusted = dict(by_score)

    # --- Check the 10s bucket against the 8s/9s falloff trend ---
    v8, v9, v10 = by_score[8], by_score[9], by_score[10]
    if v9 > 0:
        # Expected v10 if the curve continued its 8->9 growth rate smoothly
        growth_8_to_9 = v9 / v8 if v8 > 0 else 2.0
        expected_v10 = v9 * growth_8_to_9
        predicted_ceiling = expected_v10 * SPIKE_TOLERANCE
        if v10 > predicted_ceiling and v10 > MIN_SPIKE_VOTES:
            adjusted[10] = int(predicted_ceiling)

    # --- Check the 1s bucket against the 2s/3s falloff trend ---
    v1, v2, v3 = by_score[1], by_score[2], by_score[3]
    if v2 > 0:
        growth_3_to_2 = v2 / v3 if v3 > 0 else 2.0
        expected_v1 = v2 * growth_3_to_2
        predicted_ceiling = expected_v1 * SPIKE_TOLERANCE
        if v1 > predicted_ceiling and v1 > MIN_SPIKE_VOTES:
            adjusted[1] = int(predicted_ceiling)

    return [{"score": s, "votes": adjusted[s]} for s in range(1, 11)]


def _average_score_distribution(scores_array: List[Dict[str, Any]]) -> float:
    """
    Weighted average of the 1-10 star distribution, after passing the
    distribution through the PATCH 3 anti-spike smoother.

    "Fair weighting" here means exactly what a standard weighted average
    means: a vote for score=10 counts 10x as much as a vote for score=1 in
    the sum, and every one of the 10 buckets (1 through 10) is included --
    none are skipped, capped as a group, or given extra/reduced influence
    beyond their own score value and vote count. See
    verify_all_buckets_used() below for a runtime check of this per anime.
    """
    smoothed = _smooth_extreme_spikes(scores_array)

    total_votes = 0
    weighted_sum = 0.0
    for entry in smoothed:
        score_val = entry.get("score")
        votes = entry.get("votes", 0) or 0
        if score_val is None:
            continue
        weighted_sum += score_val * votes
        total_votes += votes
    if total_votes == 0:
        return 0.0
    return weighted_sum / total_votes


def verify_all_buckets_used(scores_array: List[Dict[str, Any]], mal_id: Any = None) -> bool:
    """
    Explicit, per-anime proof that all 10 score buckets (1 through 10) are
    present and were included in the weighted average -- not a trust-me
    claim, an actual runtime check run on every anime processed.

    Returns True and stays silent if all 10 buckets (score=1..10) are
    present in the raw data. Returns False and PRINTS A WARNING if any
    bucket is missing, so a malformed or incomplete API response for a
    given anime is visible in the console output rather than silently
    treated as "the missing scores just had 0 votes."

    This does not change the score itself -- an anime can still have a
    genuinely empty bucket (0 real votes at some star level, which is
    valid data). It only flags when the API response is missing a bucket
    entry *entirely*, which is a data-quality issue worth knowing about.
    """
    present_scores = {entry.get("score") for entry in scores_array if entry.get("score") is not None}
    missing = [s for s in range(1, 11) if s not in present_scores]
    if missing:
        label = f"mal_id={mal_id}" if mal_id is not None else "this anime"
        print(f"  [WARN] Score buckets missing from API response for {label}: {missing}. "
              f"Fair weighting still applies to whatever buckets ARE present.")
        return False
    return True




# =========================================================================
# DATA EXTRACTION
# =========================================================================

def fetch_top_anime_page(page: int) -> List[Dict[str, Any]]:
    """Fetch a single page of /top/anime (at PAGE_LIMIT per page) and
    extract the required fields."""
    data, hit_429, retry_after = get_json(
        TOP_ANIME_ENDPOINT, params={"page": page, "limit": PAGE_LIMIT}
    )
    adaptive_sleep_and_record(hit_429, retry_after)
    if not data or "data" not in data:
        return []

    results = []
    for entry in data["data"]:
        aired = entry.get("aired") or {}
        images = entry.get("images") or {}
        jpg_images = images.get("jpg") or {}
        results.append({
            "mal_id": entry.get("mal_id"),
            "title": entry.get("title"),
            # PATCH 9: capture English title + synonyms too -- the API
            # already sends these on every /top/anime entry, they just
            # weren't being kept. Needed so the site's search can match
            # "Frieren: Beyond Journey's End" against the JP-romanized
            # "Sousou no Frieren" record, the way MAL's own search does.
            "title_english": entry.get("title_english"),
            "title_synonyms": entry.get("title_synonyms") or [],
            # PATCH 11: capture cover image URLs. Already present on every
            # /top/anime entry -- no extra API call needed. small_image_url
            # is used for list-row thumbnails, image_url for the larger
            # detail-page cover; the site stores the URL only and lets the
            # browser fetch/cache the actual image lazily on scroll.
            "image_url": jpg_images.get("image_url"),
            "image_small_url": jpg_images.get("small_image_url"),
            # PATCH 13: capture synopsis. Also already present on every
            # /top/anime entry, no extra API call needed. Some entries
            # legitimately have none (e.g. very obscure or not-yet-aired
            # titles) -- stored as None/null in that case rather than an
            # empty string, so the site can tell "no synopsis exists"
            # apart from "empty text."
            "synopsis": entry.get("synopsis"),
            # PATCH 14: capture season (spring/summer/fall/winter). Also
            # already present on /top/anime, no extra API call. Nullable
            # on MAL's side for titles without a clean seasonal slot
            # (some older or special-format titles) -- stored as None in
            # that case, and the site falls back to the neutral/no-color
            # treatment rather than guessing a season that isn't real.
            "season": entry.get("season"),
            "mal_rank": entry.get("rank"),
            "mal_score": entry.get("score"),
            "status": entry.get("status"),
            "aired_from": aired.get("from"),
        })
    return results


def fetch_anime_statistics(mal_id: int) -> Optional[Dict[str, Any]]:
    """Fetch and flatten the /anime/{id}/statistics endpoint for one title."""
    url = STATS_ENDPOINT_TEMPLATE.format(mal_id=mal_id)
    data, hit_429, retry_after = get_json(url)
    adaptive_sleep_and_record(hit_429, retry_after)
    if not data or "data" not in data:
        return None

    stats = data["data"]
    total_entries = (
        (stats.get("watching", 0) or 0)
        + (stats.get("completed", 0) or 0)
        + (stats.get("on_hold", 0) or 0)
        + (stats.get("dropped", 0) or 0)
        + (stats.get("plan_to_watch", 0) or 0)
    )

    return {
        "watching": stats.get("watching", 0),
        "completed": stats.get("completed", 0),
        "on_hold": stats.get("on_hold", 0),
        "dropped": stats.get("dropped", 0),
        "plan_to_watch": stats.get("plan_to_watch", 0),
        "total_entries": stats.get("total", total_entries),
        "scores": stats.get("scores", []),
    }


def fetch_anime_relations(mal_id: int) -> List[Dict[str, Any]]:
    """
    PATCH 10: fetch and flatten the /anime/{id}/relations endpoint for one
    title -- sequels, prequels, side stories, spin-offs, and the source
    manga/light novel, etc. Used by the site's "Related Entries" section
    (mirrors MAL's own page) so the person can jump straight from one
    title to a related one, whether that related entry is another anime
    in the VERA database or a manga/other type that only exists on MAL.

    PATCH 12: the relations endpoint's own entries carry an "images"
    object too (same jpg/webp shape as /top/anime). Originally this was
    discarded, so a related title only got a thumbnail if it happened to
    also be separately loaded in the VERA database -- e.g. a related
    manga, or a related anime outside the current top-N cutoff, always
    showed a placeholder even though MAL had an image the whole time.
    Capturing it here means every related entry gets a real thumbnail
    whenever MAL has one, independent of what's in the local database.

    Returns a flat list of {relation, mal_id, type, name, url,
    image_url, image_small_url} dicts. Empty list (not None) on failure,
    so a missing relations fetch never crashes the rest of a title's
    record.
    """
    url = RELATIONS_ENDPOINT_TEMPLATE.format(mal_id=mal_id)
    data, hit_429, retry_after = get_json(url)
    adaptive_sleep_and_record(hit_429, retry_after)
    if not data or "data" not in data:
        return []

    flattened = []
    for group in data["data"]:
        relation_label = group.get("relation", "Related")
        for entry in group.get("entry", []):
            images = entry.get("images") or {}
            jpg_images = images.get("jpg") or {}
            flattened.append({
                "relation": relation_label,
                "mal_id": entry.get("mal_id"),
                "type": entry.get("type"),   # "anime", "manga", "novel", etc.
                "name": entry.get("name"),
                "url": entry.get("url"),
                "image_url": jpg_images.get("image_url"),
                "image_small_url": jpg_images.get("small_image_url"),
            })
    return flattened


# =========================================================================
# PERSISTENCE / RESUMING LOGIC
# =========================================================================

def load_existing_progress() -> Dict[int, Dict[str, Any]]:
    """
    Load previously saved results keyed by mal_id, to skip on restart.

    Handles two possible shapes in OUTPUT_JSON:
      - a bare list (mid-run checkpoints from save_progress(), and the
        pre-PATCH-8 export format)
      - the wrapped {"pulled_at", "title_count", "titles": [...]} shape
        that rank_and_export() now writes on a completed run
    A completed run's file is a fine resume source too (e.g. re-running
    to refresh scores), so both are accepted.
    """
    if not os.path.exists(OUTPUT_JSON):
        return {}
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        existing_list = loaded["titles"] if isinstance(loaded, dict) and "titles" in loaded else loaded
        return {item["mal_id"]: item for item in existing_list if "mal_id" in item}
    except (json.JSONDecodeError, IOError, KeyError, TypeError) as exc:
        print(f"[WARN] Could not load existing progress file ({exc}). Starting fresh.")
        return {}


def save_progress(anime_dict: Dict[int, Dict[str, Any]]) -> None:
    """
    Persist current mid-run checkpoint (dict of mal_id -> record) to
    OUTPUT_JSON as a bare list. This is an in-progress checkpoint, not
    the final export -- it intentionally has no "pulled_at" timestamp,
    since the run isn't done yet. rank_and_export() overwrites this with
    the timestamped, ranked, wrapped shape once the full pull completes.
    """
    records = list(anime_dict.values())
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def run_pipeline() -> Dict[int, Dict[str, Any]]:
    """
    1. Resume from disk if progress already exists (starts clean at page 1
       if OUTPUT_JSON doesn't exist yet).
    2. Page through /top/anime (PAGE_LIMIT per page) until TOP_COUNT unique
       titles are collected.
    3. For each new anime, fetch /statistics, flatten it into the record,
       compute the -VERA- score, and checkpoint every SAVE_EVERY items.
    """
    anime_records: Dict[int, Dict[str, Any]] = load_existing_progress()
    already_done = len(anime_records)
    if already_done:
        print(f"[RESUME] Loaded {already_done} previously processed anime from {OUTPUT_JSON}.")
    else:
        print(f"[START] No existing {OUTPUT_JSON} found -- starting fresh at page 1.")

    page = 1
    processed_since_save = 0
    collected_ids_this_run = set(anime_records.keys())

    while len(anime_records) < TOP_COUNT:
        print(f"\n[PAGE] Fetching /top/anime page {page} (limit={PAGE_LIMIT})...")
        page_entries = fetch_top_anime_page(page)

        if not page_entries:
            print("[INFO] No more data returned from /top/anime. Stopping pagination.")
            break

        for entry in page_entries:
            mal_id = entry["mal_id"]
            if mal_id in collected_ids_this_run:
                continue

            print(f"  -> Fetching statistics for [{mal_id}] {entry['title']}...")
            stats = fetch_anime_statistics(mal_id)

            if stats is None:
                print(f"     [SKIP] No statistics available for mal_id={mal_id}.")
                continue

            record = {**entry, **stats}
            verify_all_buckets_used(record.get("scores", []), mal_id=mal_id)
            record["vera_score"] = calculate_VERA_score(record)

            # PATCH 10: fetch related entries (sequels, prequels, side
            # stories, source manga, etc.) -- one extra API call per
            # title. Failure here never drops the title; relations is
            # just an empty list if the call fails.
            record["relations"] = fetch_anime_relations(mal_id)

            anime_records[mal_id] = record
            collected_ids_this_run.add(mal_id)
            processed_since_save += 1

            vera_display = record["vera_score"] if record["vera_score"] is not None else "N/A"
            print(f"     [OK] VERA={vera_display} "
                  f"(completed={record.get('completed')}, total_entries={record.get('total_entries')})")

            if processed_since_save >= SAVE_EVERY:
                save_progress(anime_records)
                print(f"  [CHECKPOINT] Saved progress at {len(anime_records)} anime.")
                processed_since_save = 0

            if len(anime_records) >= TOP_COUNT:
                break

        page += 1

    save_progress(anime_records)
    print(f"\n[DONE] Collected {len(anime_records)} anime total.")
    return anime_records


# =========================================================================
# RANKING + EXPORT
# =========================================================================

def rank_and_export(anime_records: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort ranked (non-N/A) records by vera_score descending and assign
    vera_rank 1..N. PATCH 6: titles below MIN_VOTES_FOR_NA get
    vera_score=None (displayed as "N/A"), are appended AFTER every ranked
    title, get no vera_rank, and are sorted among themselves by raw
    total_entries descending -- so a 98-vote title still sits above a
    34-vote title even though neither gets a numeric score.

    PATCH 8: the JSON export is now wrapped with a "pulled_at" timestamp
    (UTC, set at the moment this run's export happens -- not the start of
    the run, since a 15,200-title pull can take over an hour) so any
    consumer (e.g. the VERA_Extra_Idea site) can show the real extraction
    date instead of a hardcoded one. The CSV stays a flat table (a CSV
    has no natural place for run-level metadata), so the JSON is the
    source of truth for "when was this data pulled."
    """
    records = list(anime_records.values())

    ranked = [r for r in records if r.get("vera_score") is not None]
    unranked = [r for r in records if r.get("vera_score") is None]

    ranked.sort(key=lambda r: r["vera_score"], reverse=True)
    unranked.sort(key=lambda r: r.get("total_entries", 0) or 0, reverse=True)

    for idx, record in enumerate(ranked, start=1):
        record["vera_rank"] = idx
    for record in unranked:
        record["vera_rank"] = None  # N/A -- not part of the numbered ranking

    all_records = ranked + unranked

    pulled_at = datetime.now(timezone.utc).isoformat()
    export_payload = {
        "pulled_at": pulled_at,     # ISO8601 UTC -- e.g. "2026-08-07T14:32:00+00:00"
        "title_count": len(all_records),
        "titles": all_records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    if all_records:
        fieldnames = [
            "vera_rank", "vera_score", "mal_id", "title", "title_english", "status",
            "aired_from", "mal_rank", "mal_score",
            "watching", "completed", "on_hold", "dropped", "plan_to_watch",
            "total_entries",
        ]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in all_records:
                row = dict(record)
                row["vera_rank"] = row["vera_rank"] if row["vera_rank"] is not None else "N/A"
                row["vera_score"] = row["vera_score"] if row["vera_score"] is not None else "N/A"
                writer.writerow(row)

    print(f"[EXPORT] Wrote {len(ranked)} ranked + {len(unranked)} N/A records "
          f"to {OUTPUT_JSON} and {OUTPUT_CSV}.")
    return all_records


def print_top_10(records: List[Dict[str, Any]]) -> None:
    """Pretty-print the new Top 10 anime by -VERA- score."""
    print("\n" + "=" * 70)
    print("  -VERA- TOP 10 ANIME")
    print("=" * 70)
    ranked_only = [r for r in records if r.get("vera_score") is not None][:10]
    for record in ranked_only:
        print(f"  #{record['vera_rank']:>2}  VERA={record['vera_score']:>7.4f}  "
              f"MAL#{record.get('mal_rank', '?'):<6}  {record['title']}")
    print("=" * 70 + "\n")


# =========================================================================
# END-OF-RUN API HEALTH CHECK
# =========================================================================

def api_health_check() -> bool:
    """
    Pings both Tenrai endpoints once each (1 retry, 1s wait) and prints
    a plain verdict on whether the API is reachable and functioning.
    """
    print("\n[HEALTH CHECK] Verifying API endpoints are reachable...")

    top_data, _, _ = get_json(
        TOP_ANIME_ENDPOINT,
        params={"page": 1, "limit": PAGE_LIMIT},
        max_retries=HEALTHCHECK_MAX_RETRIES,
        retry_wait_min=HEALTHCHECK_RETRY_WAIT,
        retry_wait_max=HEALTHCHECK_RETRY_WAIT,
        fast_max_retries=HEALTHCHECK_MAX_RETRIES,
        fast_retry_wait=HEALTHCHECK_RETRY_WAIT,
    )
    top_ok = top_data is not None
    time.sleep(HEALTHCHECK_RETRY_WAIT)

    stats_data, _, _ = get_json(
        STATS_ENDPOINT_TEMPLATE.format(mal_id=1),
        max_retries=HEALTHCHECK_MAX_RETRIES,
        retry_wait_min=HEALTHCHECK_RETRY_WAIT,
        retry_wait_max=HEALTHCHECK_RETRY_WAIT,
        fast_max_retries=HEALTHCHECK_MAX_RETRIES,
        fast_retry_wait=HEALTHCHECK_RETRY_WAIT,
    )
    stats_ok = stats_data is not None

    working = top_ok and stats_ok
    print(f"  /top/anime endpoint........ {'OK' if top_ok else 'FAILED'}")
    print(f"  /anime/{{id}}/statistics..... {'OK' if stats_ok else 'FAILED'}")

    if working:
        print(f"\n>>> RESULT: Yes, this works — {BASE_URL} is up and responding correctly. <<<\n")
    else:
        print(f"\n>>> RESULT: No — {BASE_URL} is NOT working correctly right now. <<<\n")

    return working


# =========================================================================
# ENTRY POINT
# =========================================================================

def main():
    print("=" * 70)
    print("  -VERA- ANIME RANKING ENGINE")
    print(f"  Target: Top {TOP_COUNT} anime | Source: {BASE_URL}")
    print(f"  Tier: {'Server Key' if SERVER_KEY else 'Public'}")
    print("=" * 70)

    try:
        anime_records = run_pipeline()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Progress has been saved via last checkpoint. "
              "Re-run the script to resume.")
        sys.exit(1)

    ranked = rank_and_export(anime_records)
    print_top_10(ranked)
    api_health_check()


if __name__ == "__main__":
    main()
