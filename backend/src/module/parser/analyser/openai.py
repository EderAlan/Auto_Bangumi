import json
import logging
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from module.models import Bangumi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Episode schema (used as prompt reference, no longer via beta.parse)
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    title_en: Optional[str]
    title_zh: Optional[str]
    title_jp: Optional[str]
    season: int
    season_raw: str
    episode: int
    sub: str
    group: str
    resolution: str
    source: str


# ---------------------------------------------------------------------------
# Anime-specific system prompt (replaces the generic DEFAULT_PROMPT)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an anime title parser. Given a raw anime torrent filename, extract structured information and return ONLY valid JSON.

Fields:
- title_en: English anime title (string or null)
- title_zh: Chinese anime title (string or null)
- title_jp: Japanese anime title (string or null)
- season: season number (integer, default 1)
- season_raw: raw season text as it appears in the title (string, default "")
- episode: episode number (integer, default 1)
- sub: subtitle language info (string, default "")
- group: fansub group name, usually in [brackets] (string, default "")
- resolution: video resolution, e.g. "1080p", "720p", "2160p" (string, default "")
- source: video source, e.g. "Baha", "Bilibili", "AT-X", "Web", "WebRip" (string, default "")

Rules:
- At least one of title_en, title_zh, title_jp must be non-null
- season and episode must be integers
- Extract group from the first [bracketed] text in the title
- "第X集" or "第X话" means episode X; "第X季" or "第X期" or "Season X" means season X
- If context provides the anime name, use it for the title fields rather than re-extracting from the raw title
- Distinguish: episode number is the individual episode within a season, NOT the absolute episode number
- Do not fabricate data — if you cannot extract a field, leave it as default value
"""

# Legacy prompt kept for reference / compatibility
DEFAULT_PROMPT = SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# LRU cache (module-level, shared across all OpenAIParser instances)
# ---------------------------------------------------------------------------

_CACHE_SIZE = 500
_cache: OrderedDict[str, Optional[dict]] = OrderedDict()
_SENTINEL = object()  # cached failure marker


def _cache_get(raw: str) -> Optional[dict] | object:
    """Return cached dict, _SENTINEL for cached failure, or None for miss."""
    if raw in _cache:
        _cache.move_to_end(raw)
        val = _cache[raw]
        return _SENTINEL if val is None else val
    return None


def _cache_put(raw: str, result: Optional[dict]):
    if raw in _cache:
        del _cache[raw]
    elif len(_cache) >= _CACHE_SIZE:
        _cache.popitem(last=False)
    _cache[raw] = result


# ---------------------------------------------------------------------------
# Context helpers for name extraction / DB / TMDB lookup
# ---------------------------------------------------------------------------

_STRIP_PATTERNS = [
    re.compile(p)
    for p in [
        r"\[\d+[vV]?\d*\]",
        r"【\d+[vV]?\d*】",
        r"第?\d{1,3}[话話集]",
        r"\[第?\d{1,3}[话話集]\]",
        r"E[Pp]?\d{1,3}",
        r"\[E[Pp]?\d{1,3}\]",
        r"S\d{1,2}(eason)?\s?\d{1,2}",
        r"\[S\d{1,2}(eason)?\s?\d{1,2}\]",
        r"第\d{1,2}[季期]",
        r"\[第\d{1,2}[季期]\]",
        r"10\d{2,3}p",
        r"2160p",
        r"4K",
        r"HEVC|AVC|x264|x265|h264|h265",
        r"B-Global|Baha|Bilibili|AT-X|WebRip|Web",
        r"CHS|CHT|GB|BIG5|简[体中]?|繁[体中]?",
        r"MP4|MKV|FLV|AVI",
        r"\bEND\b",
        r"\[END\]",
        r"OVA|OAD|SP",
    ]
]


def extract_anime_name(raw: str) -> str:
    """Strip metadata markers to extract the anime name for DB/TMDB lookup."""
    s = raw
    s = re.sub(r"\[.*?\]", " ", s)
    s = re.sub(r"【.*?】", " ", s)
    for pat in _STRIP_PATTERNS:
        s = pat.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = s.strip("- /_~")
    return s


def gather_context(raw: str, language: str, db_session=None) -> dict:
    """Gather context for LLM prompt from DB and TMDB.

    Args:
        raw: the raw torrent title
        language: user's preferred language
        db_session: optional SQLModel Session for DB lookups

    Returns:
        dict with keys ``similar_entries`` and ``tmdb_info``.
    """
    ctx: dict[str, Any] = {"similar_entries": [], "tmdb_info": None}

    # Step 1: DB lookup (requires session)
    if db_session is not None:
        try:
            from module.database.bangumi import BangumiDatabase

            db = BangumiDatabase(db_session)
            all_entries = db.search_all()
            extracted_name = extract_anime_name(raw)
            for entry in all_entries:
                if not entry.title_raw:
                    continue
                if entry.title_raw in raw or (
                    extracted_name and extracted_name in entry.title_raw
                ):
                    ctx["similar_entries"].append(
                        {
                            "official_title": entry.official_title,
                            "title_raw": entry.title_raw,
                            "season": entry.season,
                        }
                    )
                if len(ctx["similar_entries"]) >= 3:
                    break
        except Exception as e:
            logger.debug("DB context lookup failed: %s", e)

    # Step 2: TMDB lookup if no DB match
    if not ctx["similar_entries"]:
        name = extract_anime_name(raw)
        if name and len(name) >= 2:
            try:
                from module.parser.analyser.tmdb_parser import (
                    tmdb_parser as _tmdb_parser_sync,
                )

                import asyncio

                info = asyncio.run(_tmdb_parser_sync(name, language))
                if info:
                    ctx["tmdb_info"] = {
                        "official_title": info.title,
                        "original_title": getattr(info, "original_title", info.title),
                        "year": getattr(info, "year", ""),
                        "last_season": getattr(info, "last_season", 0),
                    }
            except Exception as e:
                logger.debug("TMDB context lookup failed: %s", e)

    return ctx


def _build_user_message(raw: str, ctx: dict, language: str) -> str:
    """Build user message with embedded context."""
    parts = [f'Raw title: "{raw}"']
    parts.append(f"User language: {language}")

    if ctx.get("similar_entries"):
        parts.append("\nSimilar entries found in database:")
        for i, entry in enumerate(ctx["similar_entries"], 1):
            parts.append(
                f'  {i}. official_title="{entry["official_title"]}", '
                f'title_raw="{entry["title_raw"]}", season={entry["season"]}'
            )

    if ctx.get("tmdb_info"):
        t = ctx["tmdb_info"]
        parts.append("\nTMDB match:")
        parts.append(
            f'  official_title="{t["official_title"]}", '
            f'original_title="{t["original_title"]}", '
            f"year={t['year']}, last_season={t['last_season']}"
        )

    parts.append("\nParse this anime title into the JSON structure described above.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAIParser — compatible with both OpenAI and DeepSeek
# ---------------------------------------------------------------------------


class OpenAIParser:
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_type: str = "openai",
        **kwargs,
    ) -> None:
        """Parser that uses an OpenAI-compatible chat completions API.

        Supports OpenAI, Azure OpenAI, DeepSeek, and any other
        OpenAI-compatible provider.

        Args:
            api_key: API key for the provider.
            api_base: Base URL for the API endpoint.
            model: Model name to use.
            api_type: ``"openai"`` or ``"azure"``.
            **kwargs: Extra parameters (deployment_id, api_version, etc.).
        """
        if not api_key:
            raise ValueError("API key is required.")
        if api_type == "azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                base_url=api_base,
                azure_deployment=kwargs.get("deployment_id", ""),
                api_version=kwargs.get("api_version", "2023-05-15"),
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=30)

        self.model = model
        self.openai_kwargs = kwargs

    def parse(
        self,
        text: str,
        prompt: str | None = None,
        asdict: bool = True,
        context: dict | None = None,
        language: str = "zh",
    ) -> dict | str | None:
        """Parse raw anime title text with LLM.

        Args:
            text: The raw title to parse.
            prompt: Custom system prompt (defaults to anime-specific prompt).
            asdict: Return result as dict if True.
            context: Optional context dict from ``gather_context()``.
            language: User language for prompt construction.

        Returns:
            Parsed dict/string, or None on failure.
        """
        if not prompt:
            prompt = SYSTEM_PROMPT

        # Check cache
        cached = _cache_get(text)
        if cached is not None:
            if cached is _SENTINEL:
                logger.debug("LLM cache hit (failure): %s", text)
                return None
            logger.debug("LLM cache hit (success): %s", text)
            return cached if asdict else json.dumps(cached)

        logger.info("LLM parsing: %s", text[:80])

        # Build messages with optional context
        ctx = context or {}
        user_message = _build_user_message(text, ctx, language)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        result = self._call_with_retry(messages)
        _cache_put(text, result)
        return result

    def _call_with_retry(self, messages: list) -> Optional[dict]:
        """Call the API with one retry. Returns parsed dict or None."""
        params = self._prepare_params(messages)

        for attempt in range(2):
            try:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    future = worker.submit(
                        self.client.chat.completions.create, **params
                    )
                    resp = future.result()
                    content = resp.choices[0].message.content
                    return _parse_json_response(content)
            except json.JSONDecodeError as e:
                logger.warning(
                    "LLM JSON parse failed (attempt %d): %s", attempt + 1, e
                )
            except Exception as e:
                logger.warning(
                    "LLM API call failed (attempt %d): %s", attempt + 1, e
                )
            if attempt == 0:
                time.sleep(1)
        return None

    def _prepare_params(self, messages: list) -> dict[str, Any]:
        params = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        api_type = self.openai_kwargs.get("api_type", "openai")
        if api_type == "azure":
            params["deployment_id"] = self.openai_kwargs.get("deployment_id", "")
            params["api_version"] = self.openai_kwargs.get(
                "api_version", "2023-05-15"
            )
        return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_response(content: str) -> Optional[dict]:
    """Parse JSON from LLM response, with fallback extraction."""
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Cannot parse LLM response as JSON: %s", content[:200])
            return None
